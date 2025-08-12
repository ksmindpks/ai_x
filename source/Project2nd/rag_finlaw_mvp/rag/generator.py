# rag/generator.py
from typing import List, Dict
from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer_short(question: str, contexts: List[Dict]) -> str:
    """단답형 - 숫자와 용어 중심"""
    if not contexts:
        return "정보 없음"
    
    context_text = contexts[0].get('text', '')[:500]  # 최고 점수 컨텍스트만
    
    # 질문 유형 파악
    if "얼마" in question or "금액" in question:
        prompt_type = "숫자로만 답하세요 (예: 5천만원, 3개월)"
    elif "무엇" in question or "정의" in question:
        prompt_type = "명사로만 답하세요"
    else:
        prompt_type = "5단어 이내로 답하세요"
    
    prompt = f"""문맥: {context_text}

질문: {question}

{prompt_type}

답:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "정확한 용어나 숫자만 답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )
        
        answer = response.choices[0].message.content.strip()
        
        # "없음", "정해진 금액 없음" 같은 답변 제거
        if "없음" in answer or "정보" in answer:
            # 컨텍스트에서 숫자 추출 시도
            import re
            numbers = re.findall(r'\d+[천만억]?[원월개]?', context_text)
            if numbers:
                return numbers[0]
        
        return answer
        
    except Exception as e:
        return "오류"

def generate_answer_mcq(question: str, choices: List[str], contexts: List[Dict]) -> str:
    """사지선다형 - 선택지만 반환"""
    if not choices:
        return ""
    
    if not contexts:
        return choices[0]  # 기본값
    
    context_text = " ".join([c.get('text', '')[:200] for c in contexts[:2]])
    
    # 선택지를 번호로
    choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    
    # 단순 프롬프트
    prompt = f"""문맥: {context_text}

질문: {question}

선택지:
{choices_text}

답 (번호만):"""
    
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": "번호만 답하세요. 1, 2, 3, 또는 4"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip()
        
        # 번호 추출
        for i, choice in enumerate(choices, 1):
            if str(i) in answer:
                return choice
        
        # 첫 번째 선택지가 답에 있으면
        for choice in choices:
            if choice[:10] in answer or answer in choice:
                return choice
        
        # 기본값
        return choices[0]
        
    except Exception as e:
        print(f"MCQ 생성 오류: {str(e)[:50]}")
        return choices[0] if choices else ""