import re

DISCLAIMER = "본 답변은 정보 제공 목적이며, 법적·재정적 자문이 아닙니다. 최종 판단은 관련 기관 안내와 원문을 확인하세요."

def detect_mcq(question: str, choices: list[str] | None) -> bool:
    return bool(choices and len(choices) >= 2)

def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()
