# rag/generator.py - 최적화된 통합 버전
import re
from typing import List, Dict, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_answer_from_context(question: str, context: str) -> Optional[str]:
    """규칙 기반 답변 추출 (GPT 전 사전처리)"""
    
    # 패턴 1: "~은/는 X이다/입니다" 형식
    if "무엇" in question or "정의" in question:
        patterns = [
            r"(?:은|는)\s+([가-힣\s]+)(?:이다|입니다|이라|를\s말한다)",
            r"\"([^\"]+)\"(?:이라|라고)\s+한다",
        ]
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1).strip()
    
    # 패턴 2: 숫자 + 단위
    if "얼마" in question or "몇" in question or "기간" in question:
        # 더 정확한 패턴
        patterns = [
            r'(\d+개월)',
            r'(\d+일)',
            r'(\d+년)',
            r'(\d+천만원)',
            r'(\d+만원)',
            r'(\d+억원)',
            r'(\d+%)',
            r'(\d+영업일)',
        ]
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1)
    
    # 패턴 3: 방법/방식
    if "방법" in question or "방식" in question or "어떻게" in question:
        if "서면" in context and "전자문서" in context:
            return "서면 또는 전자문서"
        elif "서면" in context:
            return "서면"
    
    return None


def generate_answer_short(question: str, contexts: List[Dict]) -> str:
    """단답형 - 3단계 접근법"""
    if not contexts:
        return "정보 없음"
    
    # Step 1: 높은 점수 컨텍스트 선택
    good_contexts = [c for c in contexts if c.get('score', 0) >= 0.6]
    if not good_contexts:
        good_contexts = contexts[:3]
    
    # Step 2: 규칙 기반 추출 시도
    for ctx in good_contexts[:2]:
        text = ctx.get('text', '')
        rule_answer = extract_answer_from_context(question, text)
        if rule_answer:
            return rule_answer
    
    # Step 3: GPT 기반 추출 (규칙 실패 시)
    context_text = good_contexts[0].get('text', '')[:600]
    
    # 질문 키워드 추출
    keywords = []
    if "법" in question:
        law_match = re.search(r'법\s*제(\d+)조', question)
        if law_match:
            keywords.append(f"제{law_match.group(1)}조")
    
    # 강화된 프롬프트
    system_prompt = """당신은 법령 문서에서 정답을 추출하는 전문가입니다.
규칙:
1. 문맥에 있는 내용만 답변
2. 조문 번호가 아닌 실제 내용을 답변
3. 완전한 구문으로 답변 (단, 20자 이내)
4. 숫자는 반드시 단위 포함"""
    
    # 질문 타입별 프롬프트
    if "무엇" in question or "정의" in question:
        user_prompt = f"""문맥:
{context_text}

질문: {question}

위 문맥에서 정의나 설명을 찾아 답하세요.
답:"""
    
    elif "얼마" in question or "몇" in question:
        user_prompt = f"""문맥:
{context_text}

질문: {question}

숫자와 단위를 정확히 포함하여 답하세요. (예: 3개월, 5천만원)
답:"""
    
    elif "언제" in question or "기한" in question or "기간" in question:
        user_prompt = f"""문맥:
{context_text}

질문: {question}

기간이나 날짜를 찾아 답하세요. (예: 10일, 3개월, 10영업일 전)
답:"""
    
    else:
        user_prompt = f"""문맥:
{context_text}

질문: {question}

문맥에서 정확한 답을 찾아 완전한 구문으로 답하세요.
조문 번호(예: 25조)가 아닌 실제 내용을 답하세요.
답:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 균형잡힌 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=40
        )
        
        answer = response.choices[0].message.content.strip()
        
        # 후처리
        answer = answer.replace("답:", "").strip()
        answer = answer.replace("답 :", "").strip()
        
        # 조문 번호만 있으면 거부
        if re.match(r'^\d+조?$', answer) or re.match(r'^\d+호$', answer):
            # 컨텍스트에서 해당 조문 내용 찾기
            pattern = f"{answer}.*?([가-힣].{{5,30}})"
            match = re.search(pattern, context_text)
            if match:
                answer = match.group(1)
            else:
                answer = "정보 불충분"
        
        # 길이 제한
        if len(answer) > 50:
            # 첫 문장만
            if "." in answer:
                answer = answer.split(".")[0]
            else:
                answer = answer[:50]
        
        return answer
        
    except Exception as e:
        print(f"생성 오류: {str(e)[:100]}")
        return "오류"


def generate_answer_mcq(question: str, choices: List[str], contexts: List[Dict]) -> str:
    """사지선다형 - 개선된 선택 로직"""
    if not choices:
        return ""
    
    if not contexts:
        return choices[0]
    
    # 컨텍스트 준비
    context_text = "\n---\n".join([
        c.get('text', '')[:400] for c in contexts[:3]
    ])
    
    # 선택지 준비
    choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    
    # 프롬프트
    system_prompt = "법령 문제의 정답을 선택하는 전문가입니다. 문맥을 근거로 가장 정확한 답을 선택합니다."
    
    user_prompt = f"""문맥:
{context_text}

질문: {question}

선택지:
{choices_text}

문맥을 근거로 가장 정확한 선택지의 번호만 답하세요.
답: """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip()
        
        # 번호 추출 및 매칭
        for i, choice in enumerate(choices, 1):
            if str(i) in answer or f"{i}." in answer or f"({i})" in answer:
                return choice
        
        # 텍스트 매칭 (번호 없을 때)
        answer_lower = answer.lower()
        for choice in choices:
            if choice.lower() in answer_lower or answer_lower in choice.lower():
                return choice
        
        # 기본값
        return choices[0]
        
    except Exception as e:
        print(f"MCQ 오류: {str(e)[:100]}")
        return choices[0]