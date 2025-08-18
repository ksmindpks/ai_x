# -*- coding: utf-8 -*-
"""
rag/generator.py
- 규칙 기반 우선 + LLM 폴백(컨텍스트 근거 강제) + 품질 검증
- 기존 외부 유틸 의존도를 낮추고, 최소한의 내장 검증으로 안전 작동
"""
from __future__ import annotations
# --- optional .env backup load (lightweight) ---
try:
    import os
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("UPSTAGE_API_KEY") or os.getenv("PINECONE_API_KEY")):
        from dotenv import load_dotenv, find_dotenv
        p = find_dotenv(usecwd=True)
        if p:
            load_dotenv(p, override=False)
except Exception:
    pass
# -----------------------------------------------

import os, re, unicodedata
from typing import List, Dict, Tuple, Optional, Any

# === 설정: 환경변수 또는 기본값 ===
ANS_CONF_TH = float(os.getenv("ANSWER_CONF_THRESHOLD") or 0.42)
LLM_FBK_TH  = float(os.getenv("LLM_FALLBACK_THRESHOLD") or 0.42)
LLM_MAX_CTX = int(os.getenv("LLM_MAX_CTX") or 6)
USE_LLM_FB  = (os.getenv("USE_LLM_FALLBACK") or "true").lower() in ("1","true","yes","y")

# LLM 브릿지
from .llm_bridge import llm_available, ask_json, ctx_join_for_llm

# ---------------------------------------------------------
# 간단 정규화/검증 유틸 (외부 utils 없이 최소 동작 보장)
# ---------------------------------------------------------
_WS = re.compile(r"\s+")

def _clean(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = _WS.sub(" ", s)
    return s

def _contains_span(answer: str, contexts: List[Dict[str, Any]]) -> bool:
    """컨텍스트 내에 answer가 부분 문자열로 존재하는지(완화 매칭)."""
    a = _clean(answer)
    if not a:
        return False
    for c in (contexts or []):
        t = _clean(c.get("text","") or "")
        if a and a in t:
            return True
    return False

def validate_answer_quality(answer: str, question: str, contexts: List[Dict[str, Any]]) -> Tuple[bool, float, Dict]:
    """
    아주 단순한 품질 검증:
    - 컨텍스트 내 스팬 포함 여부
    - 길이/문장부호 이상치 간단 패널티
    """
    ans = _clean(answer)
    if not ans:
        return False, 0.0, {"reason": "empty"}

    span_ok = _contains_span(ans, contexts)
    score = 0.0
    if span_ok:
        score += 0.6
    # 길이 패널티(너무 김/너무 짧음)
    L = len(ans)
    if 1 <= L <= 64:
        score += 0.3
    else:
        score -= 0.1

    # 기호 과다 사용 패널티
    if re.search(r"[{}<>|\\^~]", ans):
        score -= 0.2

    score = max(0.0, min(1.0, score))
    return (score >= LLM_FBK_TH), score, {"span_ok": span_ok, "len": L}

# ---------------------------------------------------------
# 규칙 기반(단답형) 베이스라인: 간단 키워드/조문 추출
# ---------------------------------------------------------
_PAT_ARTICLE = re.compile(r"(제\s*\d+\s*조(?:의\s*\d+)?)(?:\s*\(?\d+\)?\s*항)?")
_PAT_PERIOD  = re.compile(r"(\d{4}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?)")
_PAT_ORG     = re.compile(r"([가-힣A-Za-z]{2,20}(위원회|부|청|원|공사|공단))")

def _rule_candidates_short(question: str, contexts: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    아주 가벼운 규칙 후보들: 조문, 기간, 기관명
    스코어는 경험적 가중치
    """
    q = _clean(question)
    texts = " ".join([_clean(c.get("text","") or "") for c in (contexts or [])])

    cands: List[Tuple[str, float]] = []

    for m in _PAT_ARTICLE.finditer(texts):
        cands.append((m.group(1), 0.55))
    for m in _PAT_PERIOD.finditer(texts):
        cands.append((m.group(1), 0.50))
    for m in _PAT_ORG.finditer(texts):
        cands.append((m.group(1), 0.45))

    # 상위 5개만
    uniq = []
    seen = set()
    for a, s in sorted(cands, key=lambda x: x[1], reverse=True):
        if a not in seen:
            seen.add(a)
            uniq.append((a, s))
        if len(uniq) >= 5:
            break
    return uniq


# ---------------------------------------------------------
# LLM 폴백
# ---------------------------------------------------------
def _try_llm_short_answer(question: str, contexts: List[Dict[str, Any]]) -> Optional[str]:
    if not llm_available():
        return None

    sys_p = (
        "당신은 한국어 법령 QA 보조자입니다. 다음 CONTEXT에 근거하여 질문에 답하세요. "
        "정답이 CONTEXT에 없으면 반드시 '정보 없음'이라고만 답하세요. "
        "정답은 1줄의 짧은 문자열로만 주세요."
    )
    ctx = ctx_join_for_llm(contexts, LLM_MAX_CTX)
    usr_p = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "JSON으로만 응답하세요. 스키마: {\"answer\": string}\n"
        "제한사항:\n"
        "- CONTEXT에 없는 내용은 쓰지 않음\n"
        "- 가능하면 CONTEXT의 표현을 그대로 사용\n"
        "- 한 줄 요약 형태, 어미/조사 최소화\n"
    )
    obj = ask_json(sys_p, usr_p, {"answer": ""})
    if not obj:
        return None
    ans = _clean(obj.get("answer","") or "")
    if ans in ("", "정보없음", "정보 없음", "모름", "해당 없음"):
        return None
    return ans


def _try_llm_mcq(question: str, choices: List[str], contexts: List[Dict[str, Any]]) -> Optional[int]:
    if not llm_available():
        return None

    sys_p = (
        "당신은 한국어 법령 MCQ 보조자입니다. CONTEXT에 근거하여 정답 보기의 인덱스(0~n-1)만 고르세요. "
        "근거가 없으면 '정보 없음'이라고만 답하세요."
    )
    ctx = ctx_join_for_llm(contexts, LLM_MAX_CTX)
    cs = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    usr_p = (
        f"QUESTION:\n{question}\n\nCHOICES:\n{cs}\n\nCONTEXT:\n{ctx}\n\n"
        "JSON 스키마: {\"index\": string}\n"
        "조건: 반드시 choices 중 하나의 인덱스만, 숫자 문자열로."
    )
    obj = ask_json(sys_p, usr_p, {"index": ""})
    if not obj:
        return None
    m = re.search(r"\d+", obj.get("index",""))
    if not m:
        return None
    k = int(m.group(0))
    return k if 0 <= k < len(choices) else None


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def generate_answer_short(question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, float, Dict]:
    """
    규칙 기반 후보를 먼저 시도하고, 자신감이 낮으면 LLM 폴백.
    최종 채택 전에는 항상 validate로 컨텍스트 근거 확인.
    """
    # 규칙 기반 후보
    rule_cands = _rule_candidates_short(question, contexts)
    rule_ans, rule_score = ("", 0.0)
    if rule_cands:
        # 가장 높은 후보 검증
        a0, s0 = rule_cands[0]
        ok, qs, meta = validate_answer_quality(a0, question, contexts)
        if ok:
            rule_ans, rule_score = a0, max(s0, qs)
        else:
            rule_ans, rule_score = a0, s0 * 0.6  # 낮춤

    # 자신감 충분하면 규칙으로 종결
    if rule_score >= ANS_CONF_TH:
        return rule_ans, rule_score, {"route": "rule"}

    # LLM 폴백
    if USE_LLM_FB:
        cand = _try_llm_short_answer(question, contexts)
        if cand:
            ok, qs, meta = validate_answer_quality(cand, question, contexts)
            if ok and qs >= max(rule_score, LLM_FBK_TH):
                return cand, qs, {"route": "llm"}

    # 폴백 실패 → 규칙(저신뢰)
    return (rule_ans or ""), rule_score, {"route": "rule-fallback"}


def generate_answer_mcq(question: str, choices: List[str], contexts: List[Dict[str, Any]]) -> Tuple[int, float, Dict]:
    """
    MCQ: 간단 규칙(컨텍스트 포함률) → 부족하면 LLM 폴백 → 검증
    """
    # 아주 간단한 규칙: 각 보기 텍스트가 컨텍스트 내에 얼마나 노출되는지 카운트
    scores = []
    for i, ch in enumerate(choices):
        ch_clean = _clean(ch)
        cnt = 0
        for c in (contexts or []):
            t = _clean(c.get("text","") or "")
            if ch_clean and ch_clean in t:
                cnt += 1
        scores.append((i, float(cnt)))

    scores.sort(key=lambda x: x[1], reverse=True)
    rule_idx, rule_score = (scores[0] if scores else (0, 0.0))
    # 정규화 감각
    if scores and scores[0][1] > 0:
        rule_score = min(1.0, 0.4 + 0.1 * scores[0][1])

    if rule_score >= ANS_CONF_TH:
        return rule_idx, rule_score, {"route": "rule"}

    if USE_LLM_FB:
        li = _try_llm_mcq(question, choices, contexts)
        if li is not None:
            # 검증: 선택지 텍스트를 단답 검증으로 점수화
            pred = choices[li]
            ok, qs, meta = validate_answer_quality(pred, question, contexts)
            if ok and qs >= max(rule_score, LLM_FBK_TH):
                return li, qs, {"route": "llm"}

    return rule_idx, rule_score, {"route": "rule-fallback"}


def get_generation_stats() -> Dict[str, Any]:
    """
    외부에서 수집용(간단 placeholder)
    """
    return {
        "llm_enabled": llm_available(),
        "ans_conf_th": ANS_CONF_TH,
        "llm_fallback_th": LLM_FBK_TH,
        "llm_max_ctx": LLM_MAX_CTX,
    }
