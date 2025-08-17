# -*- coding: utf-8 -*-
"""
rag/generator.py (improved v2, 2025-08-18)
- 규칙 기반 우선, LLM은 선택적(키 없으면 자동 비활성)
- 단답형: 정의/조문특정 패턴 확대 + 상위 문맥 4~5개까지 탐색 + 로컬 폴백 도입
- 단답형 전체 try/except 가드로 generation_failed 방지
- 사지선다: 컨텍스트 점수화 유지(안정성)
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Optional, Tuple

# ---------------------- 설정 ----------------------
try:
    from config import config
    OPENAI_API_KEY = getattr(config, "openai_api_key", "")
    DEBUG_MODE = bool(getattr(config, "debug_mode", False))
except Exception:
    import os as _os
    OPENAI_API_KEY = _os.getenv("OPENAI_API_KEY", "")
    DEBUG_MODE = _os.getenv("DEBUG_MODE", "false").lower() in ("1","true","yes")

logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------- OpenAI (선택) ----------------------
try:
    from openai import OpenAI  # type: ignore
    _CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as _e:
    logger.warning(f"OpenAI 사용 불가: {type(_e).__name__}: {_e}")
    _CLIENT = None

# ---------------------- utils (안전 폴백) ----------------------
try:
    from .utils import extract_question_type, validate_answer_quality, enhanced_postprocess_answer
except Exception:
    def extract_question_type(q: str) -> str:
        q = (q or "")
        if re.search(r'(몇|얼마|기간|언제)', q): return 'period'
        if re.search(r'(누구|기관|어디|담당)', q): return 'organization'
        if re.search(r'제\d+조', q): return 'article_specific'
        if re.search(r'(무엇|뜻|의미|정의)', q): return 'definition'
        return 'general'
    def validate_answer_quality(a: str, q: str, ctxs: List[Dict]) -> Tuple[bool,float,str]:
        if not a: return (False, 0.0, "empty")
        ln = len(a)
        if ln < 1 or ln > 30: return (False, 0.0, "len")
        present = any(a in (c.get("text","") or "") for c in (ctxs or []))
        return (present, 0.6 if present else 0.4, "")
    def enhanced_postprocess_answer(a: str, ctxs: List[Dict], q: Optional[str]=None, question_type: Optional[str]=None) -> str:
        return a.strip()

# ---------------------- 경량 통계 ----------------------
from copy import deepcopy as _dc
_generation_stats = {"short_calls": 0, "mcq_calls": 0, "llm_calls": 0}
def _bump(k:str): _generation_stats[k] = int(_generation_stats.get(k,0)) + 1
def get_generation_stats(reset: bool = False):
    snap = _dc(_generation_stats)
    if reset:
        for k in list(_generation_stats.keys()): _generation_stats[k] = 0
    return snap

# ---------------------- 공통 유틸/정규식 ----------------------
_HANGUL_NOUNish = re.compile(r'[가-힣]{2,}')
_NUM    = re.compile(r'\d{1,4}(?:\.\d+)?')
_PCT    = re.compile(r'\d{1,3}\s*%')
_PERIOD = re.compile(r'(\d{1,4})\s*(영업일|일|주|개월|달|월|년)\s*(?:이내|이상|초과|이하)?')
_RANGE  = re.compile(r'(\d{1,4})\s*[-~]\s*(\d{1,4})\s*(일|주|개월|월|년)')
_ARTICLE= re.compile(r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?')
_ORG_SUFFIX = r'(?:위원회|감독원|장관|부장관|총재|은행|공사|청|부|처|원|법원|검찰청)'
_ORG   = re.compile(rf'[가-힣]{{2,}}{_ORG_SUFFIX}')

# 정의 패턴 확대
_DEF_PATTERNS = [
    r'“?([^”\n]{2,30})”?\s*(?:란|이라\s*함|라\s*함)\s*([^.\n]{2,120}?)(?:이라\s*한다|로\s*한다|을\s*말한다|라고\s*한다|로\s*본다)',
    r'“?([^”\n]{2,30})”?\s*(?:의\s*의미|의\s*정의)\s*는\s*([^.\n]{2,120}?)\s*(?:이다|로\s*한다)',
    r'“?([^”\n]{2,30})”?\s*은\s*([^.\n]{2,120}?)\s*이다',
    r'“?([^”\n]{2,30})”?\s*을\s*말한다\s*:\s*([^.\n]{2,120})',
    r'“?([^”\n]{2,30})”?\s*이라\s*한다\s*:\s*([^.\n]{2,120})',
]
_INCLUDE_RULE = re.compile(r'“?([^”\n]{2,30})”?\s*에는\s*([^.\n]{2,120})\s*(?:등이|등을)?\s*포함된다')

_JOSA_TAIL  = re.compile(r'(?:에|에서|으로|의|를|을|은|는|가|이|와|과)$')
_PAREN_TRIM = re.compile(r'[「」“”"()]')
_STOPWORDS  = {"사항","것","경우","등","자","때","해당","관련","규정"}

def _ctx_text(contexts: List[Dict], k:int=3, limit:int=1500) -> str:
    parts = []
    for c in contexts[:k]:
        t = (c.get("text","") or "").strip()
        if t: parts.append(t[:limit])
    return "\n".join(parts)

def _ctx_text_wide(contexts: List[Dict], qtype:str, limit:int=1800) -> str:
    k = 5 if qtype in ("definition","article_specific") else 3
    return _ctx_text(contexts, k=k, limit=limit)

def _tokenize_words(s: str) -> List[str]:
    if not s: return []
    s = re.sub(r"[^0-9A-Za-z가-힣%]+", " ", s)
    return [w for w in s.split() if len(w) >= 2]

def _count_occurrences(s: str, ctxs: List[Dict]) -> int:
    if not s: return 0
    return sum((c.get("text","") or "").count(s) for c in ctxs)

def _clean_phrase(s: str) -> str:
    s = (s or "").strip()
    s = _PAREN_TRIM.sub('', s)
    s = re.sub(r'\s+', ' ', s)
    for _ in range(2):
        s = _JOSA_TAIL.sub('', s).strip()
    for sw in list(_STOPWORDS):
        s = re.sub(fr'{sw}$', '', s).strip()
    return s[:30]

def _article_tokens(s: str) -> List[str]:
    toks = []
    for a in _ARTICLE.findall(s or ""):
        t = f"제{a[0]}조"
        if a[1]: t += f"제{a[1]}항"
        if a[2]: t += f"제{a[2]}호"
        toks.append(t)
    return toks

def _best_sentence_for_article(ctx: str, anchors: List[str]) -> str:
    sents = re.split(r'(?<=[.다])\s+', ctx)
    best, score = "", -1
    for s in sents:
        hit = sum(1 for a in anchors if a and a in s)
        if hit > score:
            best, score = s, hit
    return best or ctx

# ---------------------- 패턴 추출기 ----------------------
class LocalPatternExtractor:
    def extract(self, question: str, contexts: List[Dict], qtype: str) -> List[Tuple[str,float]]:
        text = _ctx_text_wide(contexts, qtype, limit=2200)
        cands: List[Tuple[str,float]] = []
        if not text: return cands

        if qtype == 'period':
            for m in _RANGE.finditer(text):
                cands.append((f"{m.group(1)}~{m.group(2)}{m.group(3)}", 0.86))
            for m in _PERIOD.finditer(text):
                cands.append((f"{m.group(1)}{m.group(2)}", 0.80))

        elif qtype == 'organization':
            for m in _ORG.finditer(text):
                cands.append((m.group(), 0.80))

        elif qtype == 'article_specific':
            anchors = _article_tokens(question)
            core = _best_sentence_for_article(text, anchors) if anchors else text
            for m in _ARTICLE.finditer(core):
                s = f"제{m.group(1)}조"
                if m.group(2): s += f"제{m.group(2)}항"
                if m.group(3): s += f"제{m.group(3)}호"
                cands.append((s, 0.92))
            for m in re.finditer(r'「([^」]{2,40})」', core):
                cands.append((_clean_phrase(m.group(1)), 0.78))
            for m in _PCT.finditer(core):
                cands.append((m.group().replace(' ', ''), 0.70))

# >>>>>>> generator.py PATCH A (definition 분기) START >>>>>>>
        elif qtype == 'definition':
            ctx = text
            ask_term = bool(re.search(r'(무엇|뜻|의미|정의)', question or ''))
            ask_body = bool(re.search(r'(요건|조건|내용|포함|해당)', question or ''))

            term_hits: List[Tuple[str,float]] = []
            body_hits: List[Tuple[str,float]] = []

            for pat in _DEF_PATTERNS:
                for m in re.finditer(pat, ctx):
                    try:
                        term = _clean_phrase(m.group(1))
                        body = _clean_phrase(m.group(2))
                    except Exception:
                        continue
                    if term: term_hits.append((term, 0.78))
                    if body: body_hits.append((body, 0.89))

            for m in _INCLUDE_RULE.finditer(ctx):
                try:
                    _t = _clean_phrase(m.group(1))
                    _b = _clean_phrase(m.group(2))
                except Exception:
                    continue
                if _b:
                    body_hits.append((_b, 0.80))

            # 질문 의도에 맞게 우선순위 합치기
            picked: List[Tuple[str,float]] = []
            if ask_term and not ask_body:
                picked = term_hits + body_hits[:3]
            elif ask_body and not ask_term:
                picked = body_hits + term_hits[:2]
            else:
                picked = body_hits + term_hits

            # 아래 공통 흐름으로 전달
            cands.extend(picked)
# <<<<<<< generator.py PATCH A (definition 분기) END <<<<<<<

        else:  # general
            for m in _PCT.finditer(text):
                cands.append((m.group().replace(" ",""), 0.70))
            for m in _NUM.finditer(text):
                if len(m.group()) <= 6:
                    cands.append((m.group(), 0.60))
            for m in _HANGUL_NOUNish.finditer(text):
                if 2 <= len(m.group()) <= 10 and m.group() not in _STOPWORDS:
                    cands.append((m.group(), 0.56))

        # 중복 제거 + 빈도/길이 보정
        dedup: Dict[str,float] = {}
        for s, sc in cands:
            s = s.strip()
            if not s: continue
            if s not in dedup or sc > dedup[s]:
                dedup[s] = sc

        rescored: List[Tuple[str,float]] = []
        for s, prior in dedup.items():
            freq = _count_occurrences(s, contexts)
            bonus = min(0.25, 0.05 * freq)
            ln = len(s)
            penalty = -0.05 if ln > 24 else 0.0
            rescored.append((s, max(0.0, prior + bonus + penalty)))

        rescored.sort(key=lambda x: (-x[1], len(x[0])))
        return rescored[:12]

# ---------------------- LLM 백업 ----------------------
def _try_llm_short_answer(question: str, contexts: List[Dict]) -> Optional[str]:
    if _CLIENT is None:
        return None
    try:
        ctx = _ctx_text(contexts, k=2, limit=400)
        prompt = (
            "아래 문맥만 근거로, 질문에 대한 '아주 짧은 정답' 한 개만 한국어로 출력하세요.\n"
            "불확실하면 '정보 없음'이라고만 쓰세요. 15자 이내.\n\n"
            f"문맥:\n{ctx}\n\n질문: {question}\n정답:"
        )
        resp = _CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0, max_tokens=20
        )
        ans = (resp.choices[0].message.content or "").strip()
        if ans and ans != "정보 없음" and len(ans) <= 20:
            return ans
    except Exception as e:
        logger.warning(f"LLM short fallback 실패: {e}")
    return None

def _try_llm_mcq(question: str, choices: List[str], contexts: List[Dict]) -> Optional[str]:
    if _CLIENT is None:
        return None
    try:
        ctx = _ctx_text(contexts, k=2, limit=800)
        opts = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        prompt = (
            "아래 문맥만 근거로 가장 타당한 선택지 번호(1~4)만 출력하세요. 근거가 없으면 0을 출력.\n\n"
            f"문맥:\n{ctx}\n\n질문:{question}\n선택지:\n{opts}\n\n정답 번호:"
        )
        resp = _CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0, max_tokens=5
        )
        out = (resp.choices[0].message.content or "").strip()
        if out and out.isdigit():
            n = int(out)
            if 1 <= n <= len(choices):
                return choices[n-1]
    except Exception as e:
        logger.warning(f"LLM mcq fallback 실패: {e}")
    return None

# ---------------------- 로컬 폴백(LLM 없음용) ----------------------
def _fallback_rule_based_short(question: str, contexts: List[Dict], qtype: str) -> Optional[str]:
    """LLM이 없거나 후보가 없을 때 마지막으로 시도하는 규칙 기반 폴백."""
    text = _ctx_text_wide(contexts, qtype, limit=2200)
    if not text: return None

    if qtype in ("definition", "article_specific"):
        extractor = LocalPatternExtractor()
        cands = extractor.extract(question, contexts, qtype)
        if cands:
            return _clean_phrase(cands[0][0])

    m = _RANGE.search(text) or _PERIOD.search(text) or _PCT.search(text) or _NUM.search(text)
    if m:
        raw = m.group(0)
        return _clean_phrase(raw.replace(" ", ""))

    tokens = _tokenize_words(text)
    if not tokens: return None
    freq = {}
    for w in tokens:
        if w in _STOPWORDS: continue
        freq[w] = freq.get(w, 0) + 1
    best = max(freq.items(), key=lambda x: (x[1], -len(x[0])))[0] if freq else None
    return _clean_phrase(best) if best else None

# ---------------------- 단답형 ----------------------
def generate_answer_short(question: str, contexts: List[Dict]) -> str:
    _bump('short_calls')
    try:
        if not contexts:
            return "정보 불충분"

        qtype = extract_question_type(question) or 'general'
        px = LocalPatternExtractor()
        candidates = px.extract(question, contexts, qtype)

# >>>>>>> generator.py PATCH B (final score) START >>>>>>>
        best, best_s = None, 0.0
        for cand, prior in candidates[:8]:
            ok, qual, _ = validate_answer_quality(cand, question, contexts)
            if not ok:
                continue
            freq = _count_occurrences(cand, contexts)
            type_bonus = 0.08 if qtype in ('period','organization','article_specific') else (0.04 if qtype=='definition' else 0.0)
            final = prior*0.38 + qual*0.62 + min(0.20, 0.03*freq) + type_bonus
            if final > best_s:
                best, best_s = cand, final
# <<<<<<< generator.py PATCH B (final score) END <<<<<<<

        if (best is None or best_s < 0.55):
            llm_ans = _try_llm_short_answer(question, contexts)
            if llm_ans:
                _bump('llm_calls')
                ok, qual, _ = validate_answer_quality(llm_ans, question, contexts)
                if ok:
                    best, best_s = llm_ans, max(best_s, 0.60)
            if best is None:
                rb = _fallback_rule_based_short(question, contexts, qtype)
                if rb:
                    best, best_s = rb, max(best_s, 0.56)

        if best:
            best = _clean_phrase(best)
            return enhanced_postprocess_answer(best, contexts, question, qtype)

        rb2 = _fallback_rule_based_short(question, contexts, qtype)
        return rb2 or "정보 불충분"

    except Exception as e:
        logger.warning(f"[short] generation error: {type(e).__name__}: {e}")
        return "정보 불충분"

# ---------------------- 사지선다 ----------------------
def _tokenize_for_overlap(s: str) -> List[str]:
    return [w for w in _tokenize_words(s) if len(w) >= 2 and w not in _STOPWORDS]

def _score_choice_against_context(choice: str, ctx_text: str) -> int:
    s = 0
    if not choice: return s
    ch = choice.strip()
    if not ch: return s

    if ch in ctx_text: s += 120
    if ch.replace(" ","") in ctx_text.replace(" ",""): s += 80

    for m in _NUM.findall(ch):
        if m in ctx_text: s += 30
    for m in _PCT.findall(ch):
        if m.replace(" ","") in ctx_text.replace(" ",""): s += 40
    for m in _ARTICLE.findall(ch):
        tok = f"제{m[0]}조"
        if tok in ctx_text: s += 35
    if _ORG.search(ch) and _ORG.search(ctx_text): s += 35

    toks = set(_tokenize_for_overlap(ch))
    hit = sum(1 for w in toks if w in ctx_text)
    s += min(60, hit * 6)

    if len(ch) > 40: s -= 10
    return s

def generate_answer_mcq(question: str, choices: List[str], contexts: List[Dict]) -> str:
    _bump('mcq_calls')
    if not choices: return ""
    if not contexts: return choices[0]

    ctx_text = _ctx_text(contexts, k=3, limit=2000)
    scored = [(_score_choice_against_context(ch, ctx_text), ch) for ch in choices]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_choice = scored[0]

    if top_score >= 90 and (len(scored) == 1 or top_score - scored[1][0] >= 15):
        return top_choice

    llm_pick = _try_llm_mcq(question, choices, contexts)
    if llm_pick:
        _bump('llm_calls')
        return llm_pick

    return top_choice
