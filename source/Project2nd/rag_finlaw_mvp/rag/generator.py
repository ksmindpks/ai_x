# -*- coding: utf-8 -*-
# rag/generator.py (상단 일부 교체/추가)

import re
from typing import List, Dict, Optional, Tuple

# --- 안전한 config import (패키지/루트 모두 지원) ---
try:
    # 패키지 내부 ../config.py 형태
    from ..config import OPENAI_API_KEY, DEBUG_HYBRID
except Exception:
    try:
        # 프로젝트 루트 config.py
        from config import OPENAI_API_KEY, DEBUG_HYBRID
    except Exception:
        OPENAI_API_KEY = None
        DEBUG_HYBRID = False

# --- 안전한 utils import (패키지/루트 모두 지원) ---
try:
    # 같은 패키지 내 rag/utils.py
    from .utils import (
        enhanced_postprocess_answer,
        analyze_question_intent,
        validate_answer_by_intent,
    )
except Exception:
    try:
        # 프로젝트 루트 utils.py
        from utils import (
            enhanced_postprocess_answer,
            analyze_question_intent,
            validate_answer_by_intent,
        )
    except Exception as _e:
        raise ImportError(
            "utils 모듈을 찾을 수 없습니다. rag/utils.py 또는 프로젝트 루트의 utils.py 위치를 확인하세요."
        ) from _e

# OpenAI 클라이언트
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    # openai 미설치/키 미설정 시에도 import 에러로 죽지 않도록
    client = None

# -----------------------------
# 공통 유틸
# -----------------------------
def _join_top_contexts(contexts: List[Dict], k: int = 3, each_limit: int = 800) -> str:
    return "\n".join([c.get("text", "")[:each_limit] for c in contexts[:k]])

def _strict_span_prompt(question: str, context_text: str, intent: dict) -> str:
    guide_by_type = {
        "period": "숫자와 단위를 함께 포함한 정확한 구절만 추출하세요. 예: '3개월', '30일', '25%'.",
        "entity": "기관명/직책명의 정확한 표기를 그대로 추출하세요.",
        "action": "구체적 조치/방법 표현만 추출하세요. 예: '지체 없이 시정 요구', '서면 또는 전자문서'.",
        "content": "질문에 해당하는 정확한 내용 구절만 추출하세요.",
        "general": "질문에 해당하는 정확한 내용 구절만 추출하세요.",
    }
    hint = guide_by_type.get(intent.get("type", "general"), guide_by_type["general"])
    return f"""당신은 법령 스팬 추출기입니다. 아래 문맥에서 **질문에 해당하는 정확한 구절(스팬)**만 그대로 복사해 답하세요.
문맥에 **그대로 존재하는 문자열**만 허용됩니다. 문맥에 없으면 '정보 불충분'으로 답하세요.

지침:
1) 생성/의역 금지, 원문 스팬만 허용
2) {hint}
3) 답은 2~60자 이내의 한 줄

[문맥]
{context_text}

[질문]
{question}

[정답]
"""

def _find_by_patterns(text: str, patterns: List[str]) -> Optional[str]:
    for p in patterns or []:
        m = re.findall(p, text)
        if m:
            ans = m[0]
            if isinstance(ans, tuple):
                ans = next((x for x in ans if x), "").strip()
            if ans:
                return ans.strip()
    return None

def _extract_nearby(text: str, keyword: str, radius: int) -> str:
    i = text.find(keyword)
    if i < 0:
        return ""
    s = max(0, i - radius)
    e = min(len(text), i + len(keyword) + radius)
    return text[s:e]

# -----------------------------
# 법령/조문 앵커링
# -----------------------------
def _extract_law_and_article(q: str) -> Tuple[str, List[str]]:
    law = ""
    m = re.search(r'「([^」]{4,80})」', q or "")
    if m:
        law = m.group(1)
    arts = []
    for a, b, c in re.findall(r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?', q or ""):
        s = f"제{a}조" + (f"제{b}항" if b else "") + (f"제{c}호" if c else "")
        arts.append(s)
    return law, arts

def _prioritize_by_law_article(question: str, ctxs: List[Dict]) -> List[Dict]:
    law, arts = _extract_law_and_article(question)
    if not ctxs:
        return ctxs
    def hit(c):
        t = c.get("text", "")
        score = 0
        if law and law in t: score += 2
        for a in arts:
            if a and a in t: score += 3
        return score
    scored = [(hit(c), c) for c in ctxs]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored if s > 0][:3]
    if top:
        rest = [c for s, c in scored if s == 0]
        return top + rest
    return [c for s, c in scored]

# -----------------------------
# 규칙 추출(Period/Entity/Action/Content)
# -----------------------------
def extract_period_enhanced(text: str, intent: dict, question: str) -> Optional[str]:
    patterns = []
    if "유예" in intent.get("context_clues", []):
        patterns = [
            r"유예\s*기간[은는]?\s*(\d+\s*(?:개월|일|년))",
            r"(\d+\s*(?:개월|일|년))\s*(?:동안\s*)?유예",
            r"유예[^.\n]{0,30}(\d+\s*(?:개월|일|년))",
        ]
    elif "담보조달" in question or "비율" in question:
        patterns = [r"(\d+\s*%)", r"(\d+\s*퍼센트)"]
    elif any(k in question for k in ["기간","며칠","며칠간","몇 일","몇 개월"]):
        patterns = [r"(\d+\s*개월)", r"(\d+\s*일)", r"(\d+\s*년)"]
    else:
        patterns = [r"(\d+\s*개월)", r"(\d+\s*일)", r"(\d+\s*년)", r"(\d+\s*%)"]

    # 후보 수집
    cands = []
    for p in patterns:
        for m in re.finditer(p, text):
            span = m.group(1) if m.groups() else m.group(0)
            if span:
                cands.append((m.start(), span.strip()))
    if not cands:
        return None

    # 앵커어(기간/유예/비율/% 등)와의 거리
    anchors = re.findall(r"(기간|유예|비율|퍼센트|%)", question) or ["기간", "비율"]
    def near_score(pos: int) -> int:
        best = 10**9
        for a in anchors:
            for m in re.finditer(re.escape(a), text):
                best = min(best, abs(pos - m.start()))
        return best

    # 단위 선호도: intent.priority + 문맥 빈도
    prio_units = intent.get("priority_patterns", [])
    def unit_of(s: str) -> str:
        if "개월" in s: return "개월"
        if "일" in s: return "일"
        if "년" in s: return "년"
        if "%" in s or "퍼센트" in s: return "%"
        return ""
    unit_freq = {}
    for _, sp in cands:
        u = unit_of(sp)
        if u:
            unit_freq[u] = unit_freq.get(u, 0) + 1

    def rank(c):
        pos, sp = c
        u = unit_of(sp)
        rank_u = (0 if not prio_units else (-1 if any(u == pu for pu in prio_units) else 0)) + (-unit_freq.get(u, 0))
        return (near_score(pos), rank_u, len(sp))

    cands.sort(key=rank)
    return cands[0][1] if cands else None

def extract_entity_enhanced(text: str, intent: dict, question: str) -> Optional[str]:
    # 동사 앵커 주변 우선 탐색
    verbs = ["승인", "지정", "고시", "통지", "정하는", "관장", "관리", "감독"]
    ent_pat = r"(기획재정부장관|금융위원회|한국은행총재|한국은행|새마을금고|임치소|신용협동조합|은행|금고|위원회)"
    hits = []
    for v in verbs:
        for m in re.finditer(re.escape(v), text):
            seg = _extract_nearby(text, v, 60)
            mm = re.findall(ent_pat, seg)
            if mm:
                hits += [(abs(m.start() - text.find(x)), x) for x in mm]
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[0][1]
    return _find_by_patterns(text, [ent_pat])

def extract_action_enhanced(text: str, intent: dict, question: str) -> Optional[str]:
    ctx = intent.get("context_clues", [])
    patterns = []
    if "위법행위" in ctx:
        patterns += [r"(지체\s*없이\s*(?:시정|정정|조치)\s*(?:요구|명령))", r"(즉시\s*중단\s*조치)"]
    if "통지" in question or "방법" in question:
        patterns += [r"(서면\s*(?:또는|및)\s*전자문서)", r"(서면\s*통지)", r"(전자문서\s*통지)"]
    if "절차" in question or "승인" in question:
        patterns += [r"(국무회의[^\n]{0,15}심의[^\n]{0,15}승인)", r"(대통령[^\n]{0,10}승인)", r"(신청서[^\n]{0,20}제출)"]
    # 일반 규칙
    patterns += [r"([가-힣\s]{2,20}(?:하여야|해야)\s*한다)"]
    return _find_by_patterns(text, patterns)

def extract_content_enhanced(text: str, intent: dict, question: str) -> Optional[str]:
    patterns = []
    target = intent.get("target")
    if target == "consideration_object":
        patterns = [
            r"(개인금융채무자의\s*권익과\s*금융질서)",
            r"([가-힣\s]{5,35})[을를]\s*(?:고려|반영)(?:해야|하여야)",
        ]
    elif target == "inclusion_content":
        patterns = [
            r"(업무의\s*분장\s*및\s*조직구조)",
            r"(임원.*?직원.*?교육)",
            r"(산출업무규정.*?내용)",
            r"(정관\s*또는\s*이에\s*준하는\s*규정)",
        ]
    else:
        patterns = [
            r"란\s*([^.]{8,60})(?:이다|을\s*말한다)",
            r"\"([^\"\n]{5,50})\"[이라고]?\s*(?:한다|정의)",
            r"([가-힣\s]{5,40})(?:을|를)\s*(?:말한다|의미한다)",
        ]
    return _find_by_patterns(text, patterns)

def extract_fallback(question: str, contexts: List[Dict]) -> Optional[str]:
    text = "\n".join([c.get("text", "") for c in contexts[:2]])
    patterns = [
        r"(\d+\s*개월)", r"(\d+\s*일)", r"(\d+\s*%)",
        r"(기획재정부장관)", r"(한국은행총재|한국은행)",
        r"([가-힣]{3,10}\s*(?:조치|방법|내용))",
    ]
    return _find_by_patterns(text, patterns)

# -----------------------------
# LLM 추출
# -----------------------------
def extract_by_enhanced_llm(question: str, contexts: List[Dict], intent: dict) -> Optional[str]:
    context_text = _join_top_contexts(contexts, k=3, each_limit=800)
    prompt = _strict_span_prompt(question, context_text, intent)
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 법령 스팬 추출 전문가다. 문맥에 **존재하는 구절**만 그대로 복사해서 답한다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=80,
        )
        ans = (res.choices[0].message.content or "").strip()
        if ans and ans != "정보 불충분" and 2 <= len(ans) <= 60 and validate_answer_by_intent(ans, intent):
            all_text = context_text
            if (ans in all_text) or (ans.replace(" ", "") in all_text.replace(" ", "")):
                return ans
    except Exception as e:
        if DEBUG_HYBRID:
            print(f"[LLM Short Error] {e}")
    return None

# -----------------------------
# Public API
# -----------------------------
def generate_answer_short(question: str, contexts: List[Dict]) -> str:
    """단답형: LLM 1차 → 규칙 2차 → 폴백. 법령/조문 앵커링 적용."""
    if not contexts:
        return "정보 없음"

    # 상위 컨텍스트 정렬 + 법령/조문 앵커링
    good0 = [c for c in contexts if c.get("final_score", c.get("score", 0)) >= 0.35] or contexts[:3]
    good = _prioritize_by_law_article(question, good0)

    # 1) 의도 분석
    intent = analyze_question_intent(question)

    # 2) LLM 스팬 추출 (1차)
    ans = extract_by_enhanced_llm(question, good, intent)
    if ans:
        return enhanced_postprocess_answer(ans, good, intent)

    # 3) 규칙 추출 (2차)
    try:
        # 조문 매칭 우선 텍스트
        text_for_rules = good[0].get("text", "") if good else ""
        if intent["type"] == "period":
            ans = extract_period_enhanced(text_for_rules, intent, question)
        elif intent["type"] == "entity":
            ans = extract_entity_enhanced(text_for_rules, intent, question)
        elif intent["type"] == "action":
            ans = extract_action_enhanced(text_for_rules, intent, question)
        else:
            ans = extract_content_enhanced(text_for_rules, intent, question)
        if ans:
            return enhanced_postprocess_answer(ans, good, intent)
    except Exception as e:
        if DEBUG_HYBRID:
            print(f"[Rule Error] {e}")

    # 4) 폴백
    ans = extract_fallback(question, good)
    if ans:
        return enhanced_postprocess_answer(ans, good, intent)

    return "정보 불충분"

# -----------------------------
# MCQ
# -----------------------------
def _score_choice_against_contexts(choice: str, contexts: List[Dict]) -> float:
    """선택지 점수: 키워드 매칭 + 수치 매칭 + 위치 가중 + 커버리지 + generic penalty."""
    ctx = _join_top_contexts(contexts, k=3, each_limit=600)
    score = 0.0

    # 무의미 선택지 패널티
    generic = {"법령", "규정", "기준", "내용", "정의"}
    if choice.strip() in generic:
        score -= 20.0

    # 1) 정확 포함 보너스
    if choice and choice in ctx:
        score += 60.0

    # 2) 토큰 매칭(긴 키워드 가중)
    words = re.findall(r"[가-힣]{2,}", choice or "")
    for w in words:
        if w in ctx:
            score += min(len(w), 10) * 1.5

    # 3) 숫자/단위 매칭
    nums = re.findall(r"\d+\s*(?:개월|일|년|%|퍼센트|만원|억원)?", choice or "")
    for n in nums:
        if n.strip() and n.strip() in ctx:
            score += 25.0

    # 4) 위치 가중(앞쪽일수록 가점)
    idx = ctx.find(choice) if choice else -1
    if idx >= 0:
        pos_bonus = max(0.0, 20.0 - (idx / max(1, len(ctx)) * 20.0))
        score += pos_bonus

    # 5) 커버리지(선택지 토큰의 60% 이상이 문맥에 존재)
    if words:
        covered = sum(1 for w in words if w in ctx)
        if covered / len(words) >= 0.6:
            score += 20.0

    return score

def generate_answer_mcq(question: str, choices: List[str], contexts: List[Dict]) -> str:
    """MCQ: 스코어 강화 + 근소 차이면 LLM 재판정."""
    if not choices or not contexts:
        return choices[0] if choices else ""

    good = [c for c in contexts if c.get("final_score", c.get("score", 0)) >= 0.8] or contexts[:3]
    scores = []
    for i, choice in enumerate(choices):
        s = _score_choice_against_contexts(choice, good)
        scores.append((i, s, choice))
    scores.sort(key=lambda x: x[1], reverse=True)

    top_idx, top_score, top_choice = scores[0]
    need_llm = False
    if len(scores) >= 2 and (top_score - scores[1][1]) < 10.0:
        need_llm = True
    if top_score < 12.0:
        need_llm = True

    if not need_llm:
        return top_choice

    # LLM 재판정
    ctx = _join_top_contexts(good, k=3, each_limit=800)
    prompt = f"""문맥을 바탕으로 가장 옳은 선택지를 하나 고르세요.
각 선택지를 검토한 뒤 최종 번호만 답하세요. (1~{len(choices)} 중 하나)

[문맥]
{ctx}

[질문]
{question}

[선택지]
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(choices)])}

[정답 번호]
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        ans = (res.choices[0].message.content or "").strip()
        for i in range(1, len(choices) + 1):
            if str(i) in ans:
                return choices[i - 1]
    except Exception as e:
        if DEBUG_HYBRID:
            print(f"[MCQ LLM Error] {e}")

    return top_choice
