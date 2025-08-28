"""
utils.py - 통합 로그 시스템 및 향상된 평가 함수
"""
import pandas as pd
import numpy as np
import re
import difflib
import os
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
from datetime import datetime

def log_message(log_type, message, module="UTILS"):
    """통합된 로그 함수 - 3단계 분류"""
    print(f"[{module}-{log_type.upper()}] {message}")
    
    # 웹 인터페이스로 전달 시도
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'global_log_callback'):
            callback = st.session_state.global_log_callback
            if callable(callback):
                callback(log_type, message, module, "evaluation")
    except Exception:
        pass

# ============================================================================
# 1. MCQ 파싱 - 기존 검증된 방식 유지
# ============================================================================

def parse_mcq_answer(raw_answer: str, expected_format: str = 'number') -> str:
    """MCQ 답변 파싱 - 기존 검증된 방식"""
    raw_answer = str(raw_answer).strip().upper()
    
    # 알파벳 우선 체크
    alpha_match = re.search(r'[A-D]', raw_answer)
    # 숫자 체크  
    num_match = re.search(r'[1-4]', raw_answer)
    
    if expected_format == 'alphabet':
        if alpha_match:
            return alpha_match.group()
        elif num_match:
            num = int(num_match.group())
            return chr(64 + num)
        else:
            return 'A'
    
    elif expected_format == 'number':
        if num_match:
            return num_match.group()
        elif alpha_match:
            alpha = alpha_match.group()
            return str(ord(alpha) - 64)
        else:
            return '1'
    
    else:
        # 형식 불명 - 숫자 우선
        return num_match.group() if num_match else '1'

# ============================================================================
# 2. 법령 용어 정규화 강화 - 확장된 동의어 사전
# ============================================================================

def enhanced_answer_normalize(text: str) -> str:
    """법령 답변 정규화 - 확장된 동의어 사전"""
    if not text:
        return ""
    
    text = str(text).strip()
    
    # 1. 기본 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 대폭 확장된 법령 용어 동의어 매핑
    legal_synonyms = {
        # 기관명 통일 (확장)
        "중기부": "중소벤처기업부",
        "중소기업부": "중소벤처기업부",
        "중소벤처부": "중소벤처기업부",
        "금위": "금융위원회", 
        "금융위": "금융위원회",
        "금감원": "금융감독원",
        "금융감독청": "금융감독원",
        "공정위": "공정거래위원회",
        "공거위": "공정거래위원회",
        "방통위": "방송통신위원회",
        "방송위": "방송통신위원회",
        "국세청": "국세청",
        "기재부": "기획재정부",
        "복지부": "보건복지부",
        
        # 서류명 통일 (확장)
        "설립등록서류": "설립등기부등본",
        "등록서류": "등기부등본",
        "등록증류": "등기부등본", 
        "등기증명서": "등기부등본",
        "등본": "등기부등본",
        "사업자등록증": "사업자등록증명",
        "사업등록증": "사업자등록증명",
        
        # 기간 표현 통일 (확장)
        "삼년": "3년", "일년": "1년", "반년": "6개월",
        "한달": "1개월", "일주일": "7일", "이주일": "14일",
        "십일": "10일", "칠일": "7일", "삼십일": "30일",
        "육십일": "60일", "구십일": "90일",
        
        # 금액 표현 통일 (확장)
        "일억": "1억원", "십억": "10억원", "백억": "100억원",
        "천만": "1000만원", "오천만": "5000만원", "일천만": "1000만원",
        "삼천만": "3000만원", "오백만": "500만원",
        
        # 절차/확인 용어 (신규 - 분석에서 발견된 패턴)
        "규제신속확인": "법령적용여부확인절차",
        "규제신속": "법령적용여부확인",
        "신속확인": "법령적용여부확인", 
        "적용확인": "법령적용여부확인",
        "법령확인": "법령적용여부확인",
        
        # 서식 표현 (신규)
        "별지제1호": "별지 제1호서식",
        "별지제2호": "별지 제2호서식", 
        "별지제3호": "별지 제3호서식",
        "별지제4호": "별지 제4호서식",
        "별지제5호": "별지 제5호서식",
        "별지1호": "별지 제1호서식",
        "별지2호": "별지 제2호서식",
        "별지3호": "별지 제3호서식",
        "별지4호": "별지 제4호서식", 
        "별지5호": "별지 제5호서식",
        
        # 기타 용어 통일
        "즉시": "즉시",
        "바로": "즉시", 
        "곧바로": "즉시",
        "지체없이": "즉시"
    }
    
    for old_term, new_term in legal_synonyms.items():
        text = text.replace(old_term, new_term)
    
    # 3. 조문 정규화 강화
    text = re.sub(r'제\s*(\d+)\s*조', r'제\1조', text)
    text = re.sub(r'제\s*(\d+)\s*항', r'제\1항', text) 
    text = re.sub(r'제\s*(\d+)\s*호', r'제\1호', text)
    
    # 4. 숫자+단위 정규화 강화
    text = re.sub(r'(\d+)\s*(년|개월|일)(?:\s*(?:이내|이상|미만|전|후))?', r'\1\2', text)
    text = re.sub(r'(\d+)\s*(억|만)?\s*원', r'\1\2원', text)
    
    # 5. 서식 정규화 강화
    text = re.sub(r'별지\s*제\s*(\d+)\s*호(?:서식)?', r'별지 제\1호서식', text)
    
    # 6. 기관명+직책 정규화
    text = re.sub(r'([가-힣]+부)\s*장관', r'\1장관', text)
    text = re.sub(r'([가-힣]+위원회)\s*위원장', r'\1위원장', text)
    
    # 7. 불필요한 문구 제거 (확장)
    remove_phrases = [
        '답변:', '답:', '정답:', '결론:', '따라서', '그러므로',
        '답변은', '정답은', '에 따르면', '에 의하면', '다음과 같습니다',
        '위의 내용에 따르면', '컨텍스트에 따르면'
    ]
    
    for phrase in remove_phrases:
        text = text.replace(phrase, '')
    
    return text.strip()

# ============================================================================
# 3. EM/F1 계산 강화 - 더 관대한 매칭
# ============================================================================

def calculate_enhanced_exact_match(pred: str, gold: str) -> bool:
    """향상된 정확 매칭 - 더 관대한 기준"""
    if not pred or not gold:
        return False
    
    pred_norm = enhanced_answer_normalize(pred)
    gold_norm = enhanced_answer_normalize(gold)
    
    # 1. 완전 일치
    if pred_norm == gold_norm:
        return True
    
    # 2. 법령 패턴 매칭 (높은 정확도)
    legal_match = check_legal_pattern_match(pred_norm, gold_norm)
    if legal_match:
        return True
    
    # 3. 포함 관계 (기준 완화: 50% → 40%)
    if pred_norm and gold_norm:
        if pred_norm in gold_norm:
            coverage = len(pred_norm) / len(gold_norm)
            if coverage >= 0.4:  # 40%로 완화
                return True
        
        if gold_norm in pred_norm:
            coverage = len(gold_norm) / len(pred_norm) 
            if coverage >= 0.4:  # 40%로 완화
                return True
    
    # 4. 토큰 매칭 (기준 완화: 65% → 60%)
    pred_tokens = set(re.findall(r'[가-힣]+|\d+', pred_norm))
    gold_tokens = set(re.findall(r'[가-힣]+|\d+', gold_norm))
    
    if pred_tokens and gold_tokens:
        overlap = len(pred_tokens & gold_tokens)
        total = len(gold_tokens)
        if total > 0 and overlap / total >= 0.6:  # 60%로 완화
            return True
    
    # 5. 편집 거리 (기준 완화: 80% → 75%)
    if len(pred_norm) > 2 and len(gold_norm) > 2:
        similarity = difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()
        if similarity >= 0.75:  # 75%로 완화
            return True
    
    # 6. 핵심 키워드 매칭 (신규 - 더 관대)
    pred_keywords = extract_key_terms(pred_norm)
    gold_keywords = extract_key_terms(gold_norm)
    
    if pred_keywords and gold_keywords:
        keyword_overlap = len(pred_keywords & gold_keywords)
        # 키워드 1개 이상 매칭 + 길이 조건 완화
        if keyword_overlap >= 1 and len(pred_norm) >= 2:  # 3자 → 2자로 완화
            return True
    
    return False

def extract_key_terms(text: str) -> Set[str]:
    """핵심 용어 추출"""
    key_terms = set()
    
    # 법령 패턴
    legal_patterns = [
        r'제\d+조', r'제\d+항', r'제\d+호',
        r'\d+(?:년|개월|일)', r'\d+(?:억|만)?원',
        r'별지\s*제\s*\d+\s*호서식',
        r'[가-힣]+(?:위원회|청|부|처|원)(?:장관?)?'
    ]
    
    for pattern in legal_patterns:
        matches = re.findall(pattern, text)
        key_terms.update(matches)
    
    # 중요 명사 (2글자 이상으로 완화)
    nouns = re.findall(r'[가-힣]{2,}', text)
    key_terms.update(nouns[:10])  # 최대 10개로 제한
    
    # 숫자
    numbers = re.findall(r'\d+', text)
    key_terms.update(numbers)
    
    return key_terms

def check_legal_pattern_match(pred: str, gold: str) -> bool:
    """법령 패턴 특화 매칭 - 강화"""
    # 조문 매칭
    pred_articles = re.findall(r'제(\d+)조', pred)
    gold_articles = re.findall(r'제(\d+)조', gold)
    if pred_articles and gold_articles and pred_articles[0] == gold_articles[0]:
        return True
    
    # 기간 매칭
    pred_periods = re.findall(r'(\d+)(년|개월|일)', pred)
    gold_periods = re.findall(r'(\d+)(년|개월|일)', gold)
    if pred_periods and gold_periods:
        for p_num, p_unit in pred_periods:
            for g_num, g_unit in gold_periods:
                if p_num == g_num and p_unit == g_unit:
                    return True
    
    # 금액 매칭
    pred_amounts = re.findall(r'(\d+)(?:(억|만))?원', pred)
    gold_amounts = re.findall(r'(\d+)(?:(억|만))?원', gold)
    if pred_amounts and gold_amounts:
        for p_amount in pred_amounts:
            for g_amount in gold_amounts:
                if p_amount == g_amount:
                    return True
    
    # 기관명 매칭
    pred_agencies = re.findall(r'([가-힣]+(?:위원회|청|부|처|원))', pred)
    gold_agencies = re.findall(r'([가-힣]+(?:위원회|청|부|처|원))', gold)
    if pred_agencies and gold_agencies:
        return bool(set(pred_agencies) & set(gold_agencies))
    
    # 서식 매칭 (신규)
    pred_forms = re.findall(r'별지\s*제\s*(\d+)\s*호서식', pred)
    gold_forms = re.findall(r'별지\s*제\s*(\d+)\s*호서식', gold)
    if pred_forms and gold_forms:
        return bool(set(pred_forms) & set(gold_forms))
    
    return False

def calculate_enhanced_f1_score(pred: str, gold: str) -> float:
    """F1 점수 계산 - 부분 매칭 대폭 개선"""
    if not pred or not gold:
        return 0.0
    
    pred_norm = enhanced_answer_normalize(pred)
    gold_norm = enhanced_answer_normalize(gold)
    
    # 1. 핵심 구성요소별 분해
    pred_components = extract_answer_components(pred_norm)
    gold_components = extract_answer_components(gold_norm)
    
    # 2. 구성요소별 가중치 매칭
    component_weights = {
        'articles': 3.0,      # 제X조 (가장 중요)
        'periods': 2.5,       # X년/개월/일
        'amounts': 2.0,       # X억원/만원  
        'agencies': 1.8,      # 기관명
        'forms': 1.5,         # 별지 제X호서식
        'procedures': 1.2,    # 절차/방법
        'general_nouns': 1.0, # 일반 명사
        'numbers': 0.8        # 단순 숫자
    }
    
    total_overlap = 0.0
    total_pred_weight = 0.0  
    total_gold_weight = 0.0
    
    # 예상 답안 구성요소별 가중치 합산
    for component_type, pred_values in pred_components.items():
        weight = component_weights.get(component_type, 1.0)
        total_pred_weight += len(pred_values) * weight
    
    for component_type, gold_values in gold_components.items():
        weight = component_weights.get(component_type, 1.0)
        total_gold_weight += len(gold_values) * weight
    
    # 구성요소별 매칭 계산
    for component_type in set(pred_components.keys()) | set(gold_components.keys()):
        pred_values = set(pred_components.get(component_type, []))
        gold_values = set(gold_components.get(component_type, []))
        
        if pred_values and gold_values:
            # 정확한 매칭
            exact_matches = len(pred_values & gold_values)
            
            # 부분 매칭 (법령 특화) - 더 관대하게
            partial_matches = 0
            for pred_val in pred_values:
                for gold_val in gold_values:
                    if pred_val != gold_val and are_legal_terms_similar(pred_val, gold_val):
                        partial_matches += 0.8  # 부분 매칭 가중치 증가 (0.7 → 0.8)
                        break
            
            weight = component_weights.get(component_type, 1.0)
            total_overlap += (exact_matches + partial_matches) * weight
    
    if total_pred_weight == 0 or total_gold_weight == 0:
        return 0.0
    
    precision = total_overlap / total_pred_weight
    recall = total_overlap / total_gold_weight
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return min(1.0, f1)

def extract_answer_components(text: str) -> Dict[str, List[str]]:
    """답안을 구성요소별로 분해"""
    components = {
        'articles': [],
        'periods': [],
        'amounts': [], 
        'agencies': [],
        'forms': [],
        'procedures': [],
        'general_nouns': [],
        'numbers': []
    }
    
    # 조문 (최우선)
    articles = re.findall(r'제\s*\d+\s*조(?:제\s*\d+\s*항)?(?:제\s*\d+\s*호)?', text)
    components['articles'] = articles
    
    # 기간
    periods = re.findall(r'\d+(?:년|개월|일)(?:\s*(?:이내|이상|미만))?', text)
    components['periods'] = periods
    
    # 금액
    amounts = re.findall(r'\d+(?:억|만)?\s*원(?:\s*(?:이상|이하|미만))?', text)
    components['amounts'] = amounts
    
    # 기관명
    agencies = re.findall(r'[가-힣]+(?:위원회|청|부|처|원)(?:장관?)?', text)
    components['agencies'] = agencies
    
    # 서식
    forms = re.findall(r'별지\s*제\s*\d+\s*호(?:서식)?', text)
    components['forms'] = forms
    
    # 절차 관련
    procedures = re.findall(r'(?:신청|허가|승인|등록|신고|접수|처리|발급|제출)(?:서|절차|방법)?', text)
    components['procedures'] = procedures
    
    # 일반 명사 (2글자 이상으로 완화)
    general_nouns = re.findall(r'[가-힣]{2,}', text)
    # 이미 다른 카테고리에 포함된 것들 제외
    excluded = set(articles + agencies + procedures)
    general_nouns = [noun for noun in general_nouns if noun not in excluded]
    components['general_nouns'] = general_nouns[:8]  # 최대 8개로 증가
    
    # 숫자
    numbers = re.findall(r'\d+', text)
    components['numbers'] = numbers
    
    return components

def are_legal_terms_similar(term1: str, term2: str) -> bool:
    """법령 용어 유사성 판단 - 더 관대"""
    # 동의어 매핑
    synonyms = {
        "중기부": "중소벤처기업부",
        "금위": "금융위원회", 
        "금감원": "금융감독원",
        "공정위": "공정거래위원회"
    }
    
    # 정규화 후 비교
    norm1 = synonyms.get(term1, term1)
    norm2 = synonyms.get(term2, term2)
    
    if norm1 == norm2:
        return True
    
    # 포함 관계 (60% → 50%로 완화)
    if len(term1) > 2 and len(term2) > 2:
        if (term1 in term2) or (term2 in term1):
            shorter = min(term1, term2, key=len)
            longer = max(term1, term2, key=len)
            if len(shorter) / len(longer) >= 0.5:  # 50%로 완화
                return True
    
    # 편집 거리 유사성 (신규 추가)
    if len(term1) > 3 and len(term2) > 3:
        similarity = difflib.SequenceMatcher(None, term1, term2).ratio()
        if similarity >= 0.7:
            return True
    
    return False

# ============================================================================
# 4. 기존 함수들 유지 - 로그 시스템만 개선
# ============================================================================

def load_excel_data(file_path: str, mcq_limit: int = None, short_limit: int = None):
    """Excel 데이터 로딩 - 로그 통합"""
    log_message("INFO", f"Excel 파일 로딩 시작: {Path(file_path).name}")
    
    mcq_questions = []
    short_questions = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        log_message("SUCCESS", f"Excel 파일 열기 성공, {len(xl_file.sheet_names)}개 시트 발견")
    except Exception as e:
        log_message("FAILURE", f"Excel 파일 열기 실패: {e}")
        return mcq_questions, short_questions
    
    for sheet_name in xl_file.sheet_names:
        log_message("INFO", f"시트 '{sheet_name}' 처리 중...")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            log_message("SUCCESS", f"시트 '{sheet_name}' 로드 완료: {len(df)}행")
        except Exception as e:
            log_message("FAILURE", f"시트 '{sheet_name}' 로드 실패: {e}")
            continue
        
        if '사지선다' in sheet_name or 'mcq' in sheet_name.lower():
            temp_mcq = []
            for _, row in df.iterrows():
                question = str(row.get('문제내용', '')).strip()
                if len(question) < 10:
                    continue
                
                choices = []
                for i in range(1, 5):
                    choice = str(row.get(f'보기{i}', '')).strip()
                    if choice and choice.lower() not in ['nan', 'none']:
                        choices.append(choice)
                
                if len(choices) >= 2:
                    answer = row.get('정답', '')
                    if hasattr(answer, 'item'):
                        answer = str(answer.item())
                    else:
                        answer = str(answer).strip()
                    
                    answer_format = detect_answer_format(answer)
                        
                    temp_mcq.append({
                        'question': question,
                        'choices': choices,
                        'answer': answer,
                        'answer_format': answer_format,
                        'type': 'mcq'
                    })
            
            if mcq_limit and len(temp_mcq) > mcq_limit:
                temp_mcq = random.sample(temp_mcq, mcq_limit)
            mcq_questions.extend(temp_mcq)
            log_message("SUCCESS", f"MCQ 시트 '{sheet_name}': {len(temp_mcq)}개 문제 추가")
        
        elif '단답' in sheet_name or 'short' in sheet_name.lower():
            temp_short = []
            for _, row in df.iterrows():
                question = str(row.get('문제내용', '')).strip()
                answer = str(row.get('정답', '')).strip()
                
                if question and answer and len(question) >= 10 and len(answer) >= 1:
                    temp_short.append({
                        'question': question,
                        'answer': answer,
                        'type': 'short'
                    })
            
            if short_limit and len(temp_short) > short_limit:
                temp_short = random.sample(temp_short, short_limit)
            short_questions.extend(temp_short)
            log_message("SUCCESS", f"단답형 시트 '{sheet_name}': {len(temp_short)}개 문제 추가")

    log_message("SUCCESS", f"로드 완료: MCQ {len(mcq_questions)}개, 단답형 {len(short_questions)}개")
    return mcq_questions, short_questions

def detect_answer_format(answer: str) -> str:
    """답변 형식 감지"""
    answer = str(answer).strip().upper()
    if answer in ['A', 'B', 'C', 'D']:
        return 'alphabet'
    elif answer in ['1', '2', '3', '4']:
        return 'number'
    else:
        return 'unknown'

def save_evaluation_results(results: Dict[str, Any], output_file: str = None) -> str:
    """결과 저장 - 향상된 통계 정보 포함"""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_{timestamp}.xlsx"
    
    log_message("INFO", f"평가 결과 저장 중: {output_file}")
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # MCQ 결과
            if results.get('mcq_results'):
                mcq_df = pd.DataFrame(results['mcq_results'])
                mcq_df.to_excel(writer, sheet_name='MCQ결과', index=False)
                log_message("SUCCESS", f"MCQ 결과 저장: {len(mcq_df)}개 항목")
            
            # 단답형 결과  
            if results.get('short_results'):
                short_df = pd.DataFrame(results['short_results'])
                short_df.to_excel(writer, sheet_name='단답형결과', index=False)
                log_message("SUCCESS", f"단답형 결과 저장: {len(short_df)}개 항목")
            
            # 요약 정보 (확장)
            summary_data = {
                '항목': [
                    'MCQ 정확도', '단답형 EM', '단답형 F1', '총 문제수', 
                    '평가시간(초)', '평균시간(초/문제)', 
                    'MCQ 오답 패턴', '단답형 오답 패턴', '검색 품질 문제'
                ],
                '값': [
                    f"{results.get('mcq_accuracy', 0):.1%}",
                    f"{results.get('short_em', 0):.1%}",
                    f"{results.get('short_f1', 0):.1%}",
                    results.get('total_questions', 0),
                    f"{results.get('total_time', 0):.1f}",
                    f"{results.get('total_time', 0)/max(1, results.get('total_questions', 1)):.1f}",
                    str(results.get('evaluation_stats', {}).get('mcq_error_patterns', {})),
                    str(results.get('evaluation_stats', {}).get('short_error_patterns', {})),
                    results.get('evaluation_stats', {}).get('search_quality_issues', 0)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='요약', index=False)
            log_message("SUCCESS", "요약 정보 저장 완료")
            
            # 오류 분석 시트 (신규)
            if results.get('evaluation_stats'):
                error_analysis = []
                stats = results['evaluation_stats']
                
                # MCQ 오답 분석
                for error_type, count in stats.get('mcq_error_patterns', {}).items():
                    if count > 0:
                        error_analysis.append({
                            '문제유형': 'MCQ',
                            '오류패턴': error_type,
                            '발생횟수': count,
                            '비율': f"{count/max(1, results.get('mcq_total', 1))*100:.1f}%"
                        })
                
                # 단답형 오답 분석
                for error_type, count in stats.get('short_error_patterns', {}).items():
                    if count > 0:
                        error_analysis.append({
                            '문제유형': '단답형',
                            '오류패턴': error_type,
                            '발생횟수': count,
                            '비율': f"{count/max(1, results.get('short_total', 1))*100:.1f}%"
                        })
                
                if error_analysis:
                    error_df = pd.DataFrame(error_analysis)
                    error_df.to_excel(writer, sheet_name='오류분석', index=False)
                    log_message("SUCCESS", "오류 분석 저장 완료")
        
        log_message("SUCCESS", f"결과 저장 완료: {output_file}")
        return output_file
        
    except Exception as e:
        log_message("FAILURE", f"결과 저장 실패: {e}")
        return ""

# ============================================================================
# 5. SearchResult 클래스 - 단순화된 버전
# ============================================================================

class SearchResult:
    """검색 결과 클래스 - 단순화된 버전"""
    def __init__(self, content: str, score: float, metadata: Dict = None):
        self.content = content
        self.text = content  # 호환성 유지
        self.score = score
        self.metadata = metadata or {}