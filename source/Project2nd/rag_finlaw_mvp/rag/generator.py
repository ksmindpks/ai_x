# rag/generator.py (발췌)

from typing import List, Dict
import re
from openai import OpenAI
from config import OPENAI_API_KEY
from .prompt import (
    SYSTEM_BASE,
    USER_TEMPLATE_SHORT_EXTRACT,
    USER_TEMPLATE_SHORT,
    USER_TEMPLATE_MCQ,
)

_client = None
def _client_once():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def select_best_single_ctx(hits: List[Dict]) -> str:
    """retriever가 준 hits 중 score가 가장 높은 것의 text만 사용"""
    if not hits:
        return ""
    best = max(hits, key=lambda h: h.get("score", 0.0))
    return best.get("text", "")

# --- 컨텍스트 빌더: 스팬 과제에서는 자르지 않는 게 안전 ---
def build_context(hits: List[Dict]) -> str:
    parts = []
    for h in hits:
        fn = h.get("filename", "unknown")
        ck = h.get("chunk_index", "?")
        txt = h.get("text", "")            # retriever가 'text' 메타를 제공해야 함
        parts.append(f"[{fn}#{ck}] {txt}")
    return "\n\n".join(parts)

def answer_short_extract(question: str, hits: List[Dict], model: str = "gpt-4o-mini") -> str:
    client = _client_once()
    ctx = select_best_single_ctx(hits)
    if not ctx:
        return "정답: (근거 없음)"  # 검색 미히트 시 안전 리턴

    user = USER_TEMPLATE_SHORT_EXTRACT.format(question=question, context=ctx)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "지시된 형식만 출력(START/END 또는 NOTFOUND). 생성 금지."},
            {"role": "user", "content": user},
        ],
        temperature=0.0, top_p=1.0, presence_penalty=0.0, frequency_penalty=0.0,
    )
    out = resp.choices[0].message.content.strip()

    if out.upper().startswith("NOTFOUND"):
        return "정답: (근거 없음)"

    m1 = re.search(r"START:\s*(\d+)", out)
    m2 = re.search(r"END:\s*(\d+)", out)
    if not (m1 and m2):
        return "정답: (근거 없음)"

    s, e = int(m1.group(1)), int(m2.group(1))
    if 0 <= s < e <= len(ctx):
        span = ctx[s:e]
        return f"정답: {span}"
    return "정답: (근거 없음)"

def answer_short(question: str, hits: List[Dict], model: str = "gpt-4o-mini") -> str:
    client = _client_once()
    contexts = build_context(hits)
    if not contexts:
        return "정답: (근거 없음)"
    
    user = USER_TEMPLATE_SHORT.format(question=question, contexts=contexts)
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": user}
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"오류 발생: {str(e)}"

def answer_mcq(question: str, choices: List[str], hits: List[Dict], model: str = "gpt-4o-mini"):
    client = _client_once()
    contexts = build_context(hits)
    choice_block = "\n".join([f"- {c}" for c in choices])
    user = USER_TEMPLATE_MCQ.format(question=question, choices=choice_block, contexts=contexts)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":SYSTEM_BASE},
            {"role":"user","content":user}
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content
