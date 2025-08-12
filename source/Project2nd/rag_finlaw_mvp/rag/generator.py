# rag/generator.py
# --- 단답형 태그 기반 프롬프트 세트 (generator.py에 추가/교체) ---
# --- MCQ(사지선다형) 선택지 소거형 프롬프트 버전 ---
import re
from typing import List, Dict
from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL
from rag.utils import postprocess_answer  # context 단위 보정용

client = OpenAI(api_key=OPENAI_API_KEY)

MCQ_SYS = """역할: 법령 QA 다지선다 채점 모델.
규칙:
- 컨텍스트에 근거가 '등장하는지'를 기준으로 선택지를 소거한 뒤 정답을 고른다.
- 절차:
  1) 각 선택지별로 컨텍스트에 해당 문구/조문/동일 의미 표현이 '정확히' 혹은 '동일 의미'로 존재하는지 확인한다.
  2) 근거가 없으면 '근거 없음'으로 소거한다.
  3) '근거 있음'인 선택지만 남긴다.
  4) 남은 것 중 질문과 가장 직접적으로 일치하는 하나를 고른다.
- 출력은 반드시 정답의 번호만(1/2/3/4) 출력한다. 다른 글자/기호/설명 금지.
"""

MCQ_USER_TMPL = """질문:
{question}

컨텍스트(근거 확인용):
{context}

선택지:
{choices}

지시:
- 각 선택지에 대해 컨텍스트 근거 유무를 확인해 소거 후 하나만 고른다.
- 정답 번호만 출력(1/2/3/4).
"""

_num_re = re.compile(r"\b([1-4])\b")

def _extract_choice_index(text: str) -> int:
    m = _num_re.search(text or "")
    return int(m.group(1)) if m else -1

def _best_overlap_choice(choices: List[str], context_text: str) -> int:
    # 간단한 겹침 기반 폴백: 선택지 문자열과 컨텍스트의 공통 토큰 수로 점수화
    ctx = context_text.lower()
    def score(ch):
        toks = set(re.findall(r"[가-힣A-Za-z0-9]+", ch.lower()))
        return sum(tok in ctx for tok in toks)
    scores = [score(c) for c in choices]
    return (scores.index(max(scores)) + 1) if choices else 1  # 1-based

ANSWER_TAG_OPEN = "<answer>"
ANSWER_TAG_CLOSE = "</answer>"

# SHORT_SYS 내 규칙 일부 강화/수정
SHORT_SYS = f"""역할: 법령 QA '추출' 모델.
규칙:
- 제공된 컨텍스트 안에서 '원문 그대로 연속된 문자열'을 1개만 답변.
- 설명/부연/의역/단위변경 금지, 줄임 금지.
- 컨텍스트에 정답이 정확히 없으면 가장 근접한 원문 표현 1개를 그대로 복사.
- 반드시 {ANSWER_TAG_OPEN}정답내용{ANSWER_TAG_CLOSE} 형식으로만 출력.
"""

SHORT_USER_TMPL = f"""질문:
{{question}}

컨텍스트(다음에서 답을 '그대로' 발췌):
{{context}}

출력형식 예시:
{ANSWER_TAG_OPEN}3개월{ANSWER_TAG_CLOSE}
{ANSWER_TAG_OPEN}개인금융채무자의 권익과 금융질서{ANSWER_TAG_CLOSE}

정답을 위 태그 형식으로만 출력하세요.
"""

_tag_re = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

# 태그 누락 fallback
def _extract_answer_tag_or_fallback(text: str) -> str:
    m = _tag_re.search(text or "")
    if m:
        return m.group(1).strip()
    # fallback: 첫 줄 or 첫 문장
    t = (text or "").strip()
    t = t.splitlines()[0] if "\n" in t else t
    t = t.split("。")[0].split(".")[0]
    return t.strip()[:120]

def generate_answer_short(question: str, contexts: List[Dict]) -> str:
    """단답형 - 태그 기반 '원문 발췌' + 단위 보정"""
    if not contexts:
        return "정보 없음"

    # 상위 2~3개 컨텍스트 결합 (너무 길면 잘라서)
    context_text = " ".join([c.get('text', '')[:300] for c in contexts[:4]])

    prompt = SHORT_USER_TMPL.format(
        question=question.strip(),
        context=context_text
    )

    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,   # gpt-4o-mini 권장, 필요 시 교체
            messages=[
                {"role": "system", "content": SHORT_SYS},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=48   # 단답형 여유 조금
        )
        raw = resp.choices[0].message.content.strip()
        extracted = _extract_answer_tag_or_fallback(raw)

        # 단위/형태 보정 (숫자만 나온 경우 등)
        return postprocess_answer(extracted, "short", context_text)

    except Exception:
        return "오류"

# -------- 사지선다형 --------
def generate_answer_mcq(question: str, choices: List[str], contexts: List[Dict]) -> str:
    """선택지 소거 절차를 명시한 MCQ 생성기(번호 파싱 + 안전 폴백)"""
    if not choices:
        return ""

    # 상위 4개 컨텍스트, 각 300자 제한으로 토큰 방어
    context_text = " ".join([c.get("text", "")[:300] for c in (contexts or [])[:4]])

    # 선택지 포맷팅
    choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    prompt = MCQ_USER_TMPL.format(
        question=question.strip(),
        context=context_text,
        choices=choices_text
    )

    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,   # gpt-4o-mini 권장, 환경에 맞게 유지
            messages=[
                {"role": "system", "content": MCQ_SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=8
        )
        raw = resp.choices[0].message.content.strip()
        idx = _extract_choice_index(raw)

        # 파싱 실패 시: 컨텍스트-선택지 겹침 폴백
        if idx not in (1,2,3,4):
            idx = _best_overlap_choice(choices, context_text)

        # 최종 선택지 텍스트 반환
        return choices[idx-1] if 1 <= idx <= len(choices) else choices[0]

    except Exception:
        # 완전 폴백: 겹침 점수 기반
        idx = _best_overlap_choice(choices, context_text)
        return choices[idx-1] if 1 <= idx <= len(choices) else (choices[0] if choices else "")
