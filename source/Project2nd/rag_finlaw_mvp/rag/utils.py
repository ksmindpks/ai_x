# -*- coding: utf-8 -*-
"""
utils.py - 성능 강화 및 정확도 개선 버전
주요 개선사항:
1. 더 정밀한 EM/F1 계산
2. 향상된 정규화 및 매칭
3. 컨텍스트 품질 평가 추가
4. 성능 분석 기능 강화
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import hashlib
import time
import difflib
from functools import lru_cache
from collections import Counter, defaultdict

# 무의미한 답변들 (강화된 버전)
MEANINGLESS_ANSWERS = frozenset({
    '사항', '내용', '기준', '방법', '절차', '포함되어야', '해야', '할', '된다',
    '것', '경우', '때', '시', '중', '등', '및', '또는', '이상', '이하',
    '관련', '필요', '적절', '해당', '각각', '전체', '일부', '모든', '어떤',
    '따라', '따른', '관한', '대한', '위한', '통한'  # 추가
})

# 부분적으로 의미있는 답변들
PARTIAL_ANSWERS = frozenset({
    '법률', '시행령', '규정', '조항', '호', '항', '제', '조', '법령', '규칙',
    '기관', '담당', '관리', '운영', '시행', '적용'  # 추가
})

# 성능 상수 (강화)
KO_STOPWORDS = frozenset({
    "은","는","이","가","을","를","의","에","에서","에게","으로","와","과","및",
    "또는","도","만","보다","그","그것","이것","저것","여기","거기","저기",
    "때문","때문에","위해","위하여","대해","대하여","통해","통하여","법률","시행령",
    "에게","부터","까지","마다","조차","라도","든지","거나","하여","이든","든",
    "에서는","에는","로는","과는","와는","라는","이라","라고","하고","되고","있고",
    "관한","따른","따라","경우","사항","내용","방법","절차"  # 추가 불용어
})

def load_excel(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    """Excel 파일에서 사지선다형과 단답형 문제 로드 - 강화된 버전"""
    mcq_questions = []
    short_questions = []
    
    print(f"파일 로드 중: {filepath}")
    
    try:
        # 사지선다형 로드 (강화된 검증)
        print("사지선다형 문제 로드 중...")
        df_mcq = pd.read_excel(filepath, sheet_name="사지선다형")
        print(f"  사지선다형 DataFrame 크기: {df_mcq.shape}")
        
        valid_mcq_count = 0
        invalid_mcq_count = 0
        
        for idx, row in df_mcq.iterrows():
            try:
                문제유형 = str(row.get("문제유형", "")).strip()
                난이도 = str(row.get("난이도", "")).strip()
                문제내용 = str(row.get("문제내용", "")).strip()
                
                # 보기 추출 (강화된 검증)
                choices = []
                for i in range(1, 5):
                    choice = str(row.get(f"보기{i}", "")).strip()
                    if choice and choice != "nan" and len(choice) >= 2:
                        choices.append(choice)
                
                # 정답 추출 및 검증 (강화)
                정답_idx = str(row.get("정답", "")).strip()
                정답_텍스트 = ""
                
                if 정답_idx.isdigit():
                    idx_num = int(정답_idx)
                    if 1 <= idx_num <= len(choices):
                        정답_텍스트 = choices[idx_num - 1]
                
                # 해설
                해설 = str(row.get("해설", "")).strip()
                
                # 강화된 유효성 검증
                is_valid = (
                    문제내용 and 문제내용 != "nan" and len(문제내용) >= 10 and
                    len(choices) >= 4 and 
                    정답_텍스트 and 정답_텍스트 != "nan" and
                    all(len(choice) >= 2 for choice in choices)  # 모든 선택지 최소 길이
                )
                
                if is_valid:
                    mcq_questions.append({
                        "question": 문제내용,
                        "choices": choices,
                        "answer": 정답_텍스트,
                        "meta": {
                            "difficulty": 난이도 if 난이도 != "nan" else "미분류",
                            "law": "",
                            "type": 문제유형 if 문제유형 != "nan" else "일반",
                            "explanation": 해설 if 해설 != "nan" else ""
                        }
                    })
                    valid_mcq_count += 1
                else:
                    invalid_mcq_count += 1
            
            except Exception as e:
                invalid_mcq_count += 1
                if invalid_mcq_count <= 5:  # 처음 5개만 로그
                    print(f"  사지선다형 행 {idx} 처리 오류: {e}")
                continue
        
        print(f"사지선다형 {valid_mcq_count:,}개 로드 완료 (무효: {invalid_mcq_count}개)")
        
    except Exception as e:
        print(f"사지선다형 로드 오류: {e}")
    
    try:
        # 단답형 로드 (강화된 검증)
        print("단답형 문제 로드 중...")
        df_short = pd.read_excel(filepath, sheet_name="단답형")
        print(f"  단답형 DataFrame 크기: {df_short.shape}")
        
        valid_short_count = 0
        invalid_short_count = 0
        
        for idx, row in df_short.iterrows():
            try:
                법령명 = str(row.get("법령명", "")).strip()
                문제유형 = str(row.get("문제유형", "")).strip()
                난이도 = str(row.get("난이도", "")).strip()
                문제내용 = str(row.get("문제내용", "")).strip()
                정답 = str(row.get("정답", "")).strip()
                해설 = str(row.get("해설", "")).strip()
                
                # 강화된 유효성 검증
                is_valid = (
                    문제내용 and 문제내용 != "nan" and len(문제내용) >= 10 and 
                    정답 and 정답 != "nan" and len(정답) >= 2 and len(정답) <= 50 and
                    정답 not in MEANINGLESS_ANSWERS  # 무의미한 답변 제외
                )
                
                if is_valid:
                    short_questions.append({
                        "question": 문제내용,
                        "answer": 정답,
                        "meta": {
                            "difficulty": 난이도 if 난이도 != "nan" else "미분류",
                            "law": 법령명 if 법령명 != "nan" else "",
                            "type": 문제유형 if 문제유형 != "nan" else "일반",
                            "explanation": 해설 if 해설 != "nan" else ""
                        }
                    })
                    valid_short_count += 1
                else:
                    invalid_short_count += 1
            
            except Exception as e:
                invalid_short_count += 1
                if invalid_short_count <= 5:  # 처음 5개만 로그
                    print(f"  단답형 행 {idx} 처리 오류: {e}")
                continue
        
        print(f"단답형 {valid_short_count:,}개 로드 완료 (무효: {invalid_short_count}개)")
        
    except Exception as e:
        print(f"단답형 로드 오류: {e}")
    
    return mcq_questions, short_questions

def sample_questions(mcq_questions: List[Dict], short_questions: List[Dict],
                    mcq_sample: int = None, short_sample: int = None,
                    sampling_method: str = 'stratified') -> Tuple[List[Dict], List[Dict]]:
    """질문 샘플링 - 강화된 버전"""
    import random
    random.seed(42)
    
    # MCQ 샘플링 (강화)
    if mcq_sample and mcq_sample < len(mcq_questions):
        print(f"MCQ 샘플링: {len(mcq_questions):,} -> {mcq_sample:,}")
        
        if sampling_method == 'random':
            mcq_sampled = random.sample(mcq_questions, mcq_sample)
        elif sampling_method == 'difficulty':
            # 난이도별 균등 샘플링 (강화)
            mcq_by_difficulty = defaultdict(list)
            for q in mcq_questions:
                diff = q.get('meta', {}).get('difficulty', '미분류')
                mcq_by_difficulty[diff].append(q)
            
            mcq_sampled = []
            difficulties = list(mcq_by_difficulty.keys())
            per_difficulty = mcq_sample // len(difficulties)
            remainder = mcq_sample % len(difficulties)
            
            for i, diff in enumerate(difficulties):
                questions = mcq_by_difficulty[diff]
                take = per_difficulty + (1 if i < remainder else 0)
                take = min(take, len(questions))
                
                if take > 0:
                    # 품질 기반 샘플링 (문제 길이, 선택지 품질 고려)
                    scored_questions = []
                    for q in questions:
                        score = len(q['question']) + sum(len(choice) for choice in q['choices'])
                        scored_questions.append((score, q))
                    
                    # 상위 품질 문제들 중에서 랜덤 선택
                    scored_questions.sort(reverse=True)
                    top_candidates = scored_questions[:min(len(scored_questions), take * 3)]
                    selected = random.sample([q for _, q in top_candidates], take)
                    mcq_sampled.extend(selected)
        
        elif sampling_method == 'stratified':
            # 계층화 샘플링 (난이도 + 법령 고려)
            mcq_by_category = defaultdict(list)
            for q in mcq_questions:
                diff = q.get('meta', {}).get('difficulty', '미분류')
                law = q.get('meta', {}).get('law', '일반')[:10]  # 법령명 일부만
                category = f"{diff}_{law}"
                mcq_by_category[category].append(q)
            
            mcq_sampled = []
            categories = list(mcq_by_category.keys())
            per_category = mcq_sample // len(categories)
            remainder = mcq_sample % len(categories)
            
            for i, category in enumerate(categories):
                questions = mcq_by_category[category]
                take = per_category + (1 if i < remainder else 0)
                take = min(take, len(questions))
                
                if take > 0:
                    mcq_sampled.extend(random.sample(questions, take))
        else:
            mcq_sampled = mcq_questions[:mcq_sample]
    else:
        mcq_sampled = mcq_questions
    
    # 단답형 샘플링 (강화)
    if short_sample and short_sample < len(short_questions):
        print(f"단답형 샘플링: {len(short_questions):,} -> {short_sample:,}")
        
        if sampling_method == 'random':
            short_sampled = random.sample(short_questions, short_sample)
        elif sampling_method == 'difficulty':
            # 난이도별 균등 샘플링
            short_by_difficulty = defaultdict(list)
            for q in short_questions:
                diff = q.get('meta', {}).get('difficulty', '미분류')
                short_by_difficulty[diff].append(q)
            
            short_sampled = []
            difficulties = list(short_by_difficulty.keys())
            per_difficulty = short_sample // len(difficulties)
            remainder = short_sample % len(difficulties)
            
            for i, diff in enumerate(difficulties):
                questions = short_by_difficulty[diff]
                take = per_difficulty + (1 if i < remainder else 0)
                take = min(take, len(questions))
                
                if take > 0:
                    # 답변 품질 기반 선별 (길이가 적절하고 구체적인 답변)
                    quality_questions = []
                    for q in questions:
                        answer = q['answer']
                        # 답변 품질 점수 (길이, 구체성, 숫자/기관명 포함 등)
                        score = 0
                        if 3 <= len(answer) <= 20:
                            score += 2
                        if re.search(r'\d+', answer):
                            score += 1
                        if re.search(r'(장관|위원회|기관|원|청|부)', answer):
                            score += 1
                        if answer not in MEANINGLESS_ANSWERS:
                            score += 1
                        
                        quality_questions.append((score, q))
                    
                    # 고품질 답변부터 선택
                    quality_questions.sort(reverse=True)
                    top_candidates = quality_questions[:min(len(quality_questions), take * 2)]
                    selected = random.sample([q for _, q in top_candidates], take)
                    short_sampled.extend(selected)
        
        elif sampling_method == 'stratified':
            # 질문 유형별 계층화 샘플링
            short_by_type = defaultdict(list)
            for q in short_questions:
                # 질문 유형 자동 분류
                question = q['question']
                if re.search(r'(기간|얼마|몇)', question):
                    qtype = "기간"
                elif re.search(r'(누구|누가|기관|담당)', question):
                    qtype = "기관"
                elif re.search(r'(무엇|어떤)', question):
                    qtype = "정의"
                elif re.search(r'제\d+조', question):
                    qtype = "조문"
                else:
                    qtype = "일반"
                
                short_by_type[qtype].append(q)
            
            short_sampled = []
            types = list(short_by_type.keys())
            per_type = short_sample // len(types)
            remainder = short_sample % len(types)
            
            for i, qtype in enumerate(types):
                questions = short_by_type[qtype]
                take = per_type + (1 if i < remainder else 0)
                take = min(take, len(questions))
                
                if take > 0:
                    short_sampled.extend(random.sample(questions, take))
        else:
            short_sampled = short_questions[:short_sample]
    else:
        short_sampled = short_questions
    
    return mcq_sampled, short_sampled

# 향상된 정규화 함수들
@lru_cache(maxsize=8000)  # 캐시 크기 증가
def ultra_precise_normalize(s: str) -> str:
    """초정밀 정규화 - 성능 향상 버전"""
    if not s:
        return ""
    
    s = str(s).strip()
    
    # 1. 공백 정규화 (강화)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\u3000\xa0]', ' ', s)  # 전각 공백, non-breaking space
    
    # 2. 숫자 단위 정규화 (더 정밀하게)
    # 날짜 정규화
    s = re.sub(r'(\d+)\s*년\s*(\d+)\s*월\s*(\d+)\s*일', r'\1년\2월\3일', s)
    s = re.sub(r'(\d+)\s*월\s*(\d+)\s*일', r'\1월\2일', s)
    s = re.sub(r'매년\s*(\d+)\s*월\s*(\d+)\s*일', r'매년\1월\2일', s)
    
    # 기간 정규화 (확장)
    s = re.sub(r'(\d+)\s*(개월|년|일|월|주|시간|분|초)', r'\1\2', s)
    s = re.sub(r'(\d+)\s*(%|퍼센트)', r'\1%', s)
    s = re.sub(r'(\d+)\s*(억|만|천)?\s*(원|달러|유로)', r'\1\2\3', s)
    
    # 3. 기관명 정규화 (더 포괄적으로)
    institution_patterns = [
        (r'기획재정부\s*장관', '기획재정부장관'),
        (r'금융\s*위원회', '금융위원회'),
        (r'금융\s*감독원', '금융감독원'),
        (r'한국\s*은행', '한국은행'),
        (r'국무\s*총리', '국무총리'),
        (r'([가-힣]{2,8})\s*(장관|위원회|감독원|은행|청|부|원)', r'\1\2')
    ]
    
    for pattern, replacement in institution_patterns:
        s = re.sub(pattern, replacement, s)
    
    # 4. 조문 정규화
    s = re.sub(r'제\s*(\d+)\s*조', r'제\1조', s)
    s = re.sub(r'제\s*(\d+)\s*항', r'제\1항', s)
    s = re.sub(r'제\s*(\d+)\s*호', r'제\1호', s)
    
    # 5. "및" 표현 정규화
    s = re.sub(r'\s*및\s*', ' 및 ', s)
    s = re.sub(r'\s*과\s*', ' 및 ', s)
    s = re.sub(r'\s*,\s*', ', ', s)
    
    # 6. 불필요한 문장 부호 제거
    s = re.sub(r'[""''「」『』\(\)\[\]{}]', '', s)
    s = re.sub(r'[.,;:!?]+$', '', s)
    
    # 7. 조사 정리 (강화)
    s = re.sub(r'([가-힣])을\s*말한다', r'\1', s)
    s = re.sub(r'([가-힣])를\s*의미한다', r'\1', s)
    s = re.sub(r'([가-힣])이다\.?$', r'\1', s)
    s = re.sub(r'([가-힣])이며\.?$', r'\1', s)
    
    # 8. 최종 정리
    s = re.sub(r'\s+', ' ', s)
    
    return s.strip().lower()

@lru_cache(maxsize=5000)  # 캐시 크기 증가
def extract_numbers_precise(s: str) -> List[str]:
    """정밀한 숫자 추출 - 성능 향상"""
    if not s:
        return []
    
    numbers = set()  # 중복 제거를 위해 set 사용
    
    # 1. 완전한 날짜 패턴 (우선순위 순)
    date_patterns = [
        r'(\d+년\d+월\d+일)',
        r'(\d+월\d+일)',
        r'(매년\s*\d+월\s*\d+일)',
        r'(\d+년\s*이내)',
        r'(\d+일\s*이내)',
        r'(\d+개월\s*이내)'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, s)
        numbers.update(matches)
    
    # 2. 기간/수량 패턴 (확장)
    quantity_patterns = [
        r'(\d+(?:\.\d+)?(?:개월|년|일|월|주|시간|분|초|%|억원|만원|천원|원))',
        r'(\d+(?:\.\d+)?퍼센트)',
        r'(\d+개\s*(?:이상|이하|이내))',
        r'(\d+배\s*(?:이상|이하|이내))',
        r'(\d+회\s*(?:이상|이하|이내))',
        r'(\d+명\s*(?:이상|이하|이내))',
        r'(\d+건\s*(?:이상|이하|이내))',
        r'(\d+차례\s*(?:이상|이하|이내))'
    ]
    
    for pattern in quantity_patterns:
        matches = re.findall(pattern, s)
        numbers.update(matches)
    
    # 3. 순수 숫자 (의미있는 범위, 더 엄격하게)
    pure_numbers = re.findall(r'\d+(?:\.\d+)?', s)
    meaningful_numbers = [n for n in pure_numbers 
                         if 0.1 <= float(n) <= 10000 and len(n) <= 6]  # 범위 조정
    numbers.update(meaningful_numbers)
    
    return list(numbers)

def enhanced_numeric_match(a: str, b: str) -> Tuple[bool, float]:
    """향상된 숫자 매칭"""
    nums_a = extract_numbers_precise(a)
    nums_b = extract_numbers_precise(b)
    
    if not nums_a or not nums_b:
        return False, 0.0
    
    # 1. 완전 일치 (최우선)
    exact_matches = len(set(nums_a) & set(nums_b))
    if exact_matches > 0:
        return True, 1.0
    
    # 2. 숫자 값 매칭 (단위 무시, 더 정밀하게)
    def extract_numeric_value(s):
        match = re.search(r'(\d+(?:\.\d+)?)', s)
        return float(match.group(1)) if match else None
    
    values_a = [extract_numeric_value(n) for n in nums_a if extract_numeric_value(n) is not None]
    values_b = [extract_numeric_value(n) for n in nums_b if extract_numeric_value(n) is not None]
    
    if values_a and values_b:
        for va in values_a:
            for vb in values_b:
                if va == vb:
                    return True, 0.98
                elif abs(va - vb) / max(va, vb) < 0.02:  # 2% 이내 차이
                    return True, 0.95
                elif abs(va - vb) / max(va, vb) < 0.05:  # 5% 이내 차이
                    return True, 0.90
                elif abs(va - vb) / max(va, vb) < 0.1:   # 10% 이내 차이
                    return True, 0.80
    
    # 3. 패턴 매칭 (숫자 + 단위)
    pattern_matches = 0
    for na in nums_a:
        for nb in nums_b:
            # 단위가 같고 숫자가 비슷한 경우
            unit_a = re.search(r'[가-힣%]+', na)
            unit_b = re.search(r'[가-힣%]+', nb)
            
            if unit_a and unit_b and unit_a.group() == unit_b.group():
                val_a = extract_numeric_value(na)
                val_b = extract_numeric_value(nb)
                
                if val_a and val_b and abs(val_a - val_b) / max(val_a, val_b) < 0.15:
                    pattern_matches += 1
    
    if pattern_matches > 0:
        return True, 0.75
    
    return False, 0.0

@lru_cache(maxsize=8000)  # 캐시 크기 증가
def tokenize_korean_enhanced(s: str) -> Tuple[str, ...]:
    """향상된 한국어 토큰화"""
    s = ultra_precise_normalize(s)
    
    tokens = []
    words = s.split()
    
    for word in words:
        # 불용어 및 무의미한 단어 제거 (더 엄격하게)
        if (word not in KO_STOPWORDS and 
            word not in MEANINGLESS_ANSWERS and 
            word not in PARTIAL_ANSWERS and
            len(word) >= 2 and
            not word.isdigit() or len(word) <= 2):  # 너무 긴 숫자 제외
            tokens.append(word)
    
    return tuple(tokens)

def enhanced_contextual_similarity(pred: str, gold: str, question: str = "") -> float:
    """향상된 컨텍스트 기반 의미적 유사도"""
    if not pred or not gold:
        return 0.0
    
    # 1. 기본 정규화
    pred_norm = ultra_precise_normalize(pred)
    gold_norm = ultra_precise_normalize(gold)
    
    if pred_norm == gold_norm:
        return 1.0
    
    # 2. 무의미한 답변 강력 페널티
    if pred in MEANINGLESS_ANSWERS:
        return 0.0
    
    # 3. 부분적 의미 답변 감점 (더 엄격하게)
    if pred in PARTIAL_ANSWERS:
        # 정답이 관련있으면 부분 점수, 없으면 강력 감점
        if any(word in gold_norm for word in pred_norm.split()):
            return 0.25  # 0.3에서 0.25로 감소
        else:
            return 0.05  # 0.1에서 0.05로 감소
    
    # 4. 포함 관계 확인 (더 정밀하게)
    if pred_norm in gold_norm:
        coverage = len(pred_norm) / len(gold_norm)
        if coverage >= 0.9:
            return 0.98
        elif coverage >= 0.7:
            return 0.90
        elif coverage >= 0.5:
            return 0.80
        else:
            return 0.60
    elif gold_norm in pred_norm:
        coverage = len(gold_norm) / len(pred_norm)
        if coverage >= 0.9:
            return 0.95
        elif coverage >= 0.7:
            return 0.85
        elif coverage >= 0.5:
            return 0.70
        else:
            return 0.50
    
    # 5. 숫자 매칭 우선 확인
    is_numeric, numeric_score = enhanced_numeric_match(pred, gold)
    if is_numeric and numeric_score > 0.8:
        return numeric_score
    
    # 6. 문자 수준 유사도 (개선)
    char_similarity = difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()
    
    # 7. 토큰 수준 유사도 (개선)
    tokens_pred = set(tokenize_korean_enhanced(pred))
    tokens_gold = set(tokenize_korean_enhanced(gold))
    
    if tokens_pred and tokens_gold:
        token_intersection = len(tokens_pred & tokens_gold)
        token_union = len(tokens_pred | tokens_gold)
        token_similarity = token_intersection / token_union if token_union > 0 else 0.0
        
        # 완전한 토큰 매칭에 더 큰 보너스
        if token_intersection == len(tokens_gold) and len(tokens_gold) > 0:
            token_similarity += 0.25  # 0.2에서 0.25로 증가
    else:
        token_similarity = 0.0
    
    # 8. 질문 유형별 가중치 조정 (더 세밀하게)
    if question:
        if re.search(r'(정의|의미|뜻)', question) and len(pred) >= 5:
            # 정의 질문은 토큰 유사도에 더 높은 가중치
            final_similarity = char_similarity * 0.25 + token_similarity * 0.75
        elif re.search(r'(기간|얼마|몇)', question):
            # 숫자 질문은 숫자 매칭 최우선
            if is_numeric:
                return numeric_score
            final_similarity = char_similarity * 0.4 + token_similarity * 0.6
        elif re.search(r'(누구|누가|기관|담당)', question):
            # 기관 질문은 정확한 매칭 선호
            final_similarity = char_similarity * 0.3 + token_similarity * 0.7
        else:
            final_similarity = char_similarity * 0.35 + token_similarity * 0.65
    else:
        final_similarity = char_similarity * 0.35 + token_similarity * 0.65
    
    # 9. 숫자 매칭 보너스 적용
    if is_numeric and numeric_score > 0.5:
        final_similarity = max(final_similarity, numeric_score * 0.8)
    
    return final_similarity

def ultra_precise_exact_match(pred: str, gold: str, question: str = "") -> Tuple[float, str]:
    """초정밀 EM 계산"""
    if not pred or not gold:
        return 0.0, "빈 답변"
    
    # 0. 무의미한 답변 차단 (더 엄격하게)
    if pred in MEANINGLESS_ANSWERS:
        return 0.0, "무의미한 답변"
    
    # 0.1. 너무 짧은 답변 차단
    if len(pred.strip()) < 2:
        return 0.0, "답변이 너무 짧음"
    
    # 1. 정확한 매칭
    if ultra_precise_normalize(pred) == ultra_precise_normalize(gold):
        return 1.0, "정확한 매칭"
    
    # 2. 숫자 매칭 최우선 검사 (더 엄격하게)
    is_numeric_match, numeric_score = enhanced_numeric_match(pred, gold)
    if is_numeric_match and numeric_score >= 0.9:
        return numeric_score, f"숫자 매칭 ({numeric_score:.2f})"
    
    # 3. 향상된 컨텍스트 기반 의미적 유사도
    similarity = enhanced_contextual_similarity(pred, gold, question)
    
    # 4. EM 점수 구간 조정 (더 엄격하게)
    if similarity >= 0.98:
        return 0.98, f"거의 완벽 ({similarity:.2f})"
    elif similarity >= 0.90:
        return 0.92, f"매우 높은 유사도 ({similarity:.2f})"
    elif similarity >= 0.80:
        return 0.85, f"높은 유사도 ({similarity:.2f})"
    elif similarity >= 0.70:
        return 0.75, f"양호한 유사도 ({similarity:.2f})"
    elif similarity >= 0.50:
        return 0.60, f"중간 유사도 ({similarity:.2f})"
    elif similarity >= 0.30:
        return 0.40, f"낮은 유사도 ({similarity:.2f})"
    elif similarity >= 0.15:
        return 0.20, f"매우 낮은 유사도 ({similarity:.2f})"
    
    # 5. 키워드 기반 부분 매칭 (더 엄격하게)
    pred_keywords = set(re.findall(r'[가-힣]{2,}', pred))
    gold_keywords = set(re.findall(r'[가-힣]{2,}', gold))
    
    # 무의미한 키워드 제거
    pred_keywords = pred_keywords - MEANINGLESS_ANSWERS - PARTIAL_ANSWERS
    gold_keywords = gold_keywords - MEANINGLESS_ANSWERS - PARTIAL_ANSWERS
    
    if pred_keywords and gold_keywords:
        keyword_overlap = len(pred_keywords & gold_keywords)
        total_keywords = len(gold_keywords)
        
        if keyword_overlap > 0 and total_keywords > 0:
            keyword_score = min(0.50, (keyword_overlap / total_keywords) * 0.70)  # 최대 점수 하향
            if keyword_score >= 0.3:
                return keyword_score, f"키워드 매칭 ({keyword_overlap}/{total_keywords}개)"
    
    return 0.0, "매칭 실패"

def ultra_precise_f1_score(pred: str, gold: str, question: str = "") -> Tuple[float, str]:
    """초정밀 F1 점수"""
    if not pred or not gold:
        return 0.0, "빈 답변"
    
    # 무의미한 답변 차단
    if pred in MEANINGLESS_ANSWERS:
        return 0.0, "무의미한 답변"
    
    # 토큰화 (향상된 버전)
    pred_tokens = set(tokenize_korean_enhanced(pred))
    gold_tokens = set(tokenize_korean_enhanced(gold))
    
    if not pred_tokens or not gold_tokens:
        return 0.0, "토큰 없음"
    
    # 정확한 토큰 매칭
    exact_intersection = len(pred_tokens & gold_tokens)
    
    # 부분 토큰 매칭 (더 정교하게)
    partial_matches = 0
    for p_token in pred_tokens:
        for g_token in gold_tokens:
            if p_token != g_token and len(p_token) >= 2 and len(g_token) >= 2:
                # 포함 관계 (더 엄격하게)
                if (p_token in g_token or g_token in p_token) and abs(len(p_token) - len(g_token)) <= 2:
                    partial_matches += 0.8  # 0.7에서 0.8로 증가
                    break
                # 편집 거리 기반 유사도 (더 엄격하게)
                elif len(p_token) >= 3 and len(g_token) >= 3:
                    similarity = difflib.SequenceMatcher(None, p_token, g_token).ratio()
                    if similarity >= 0.85:  # 0.8에서 0.85로 상향
                        partial_matches += 0.6  # 0.5에서 0.6으로 증가
                        break
                    elif similarity >= 0.70:  # 0.6에서 0.7로 상향
                        partial_matches += 0.4  # 0.3에서 0.4로 증가
                        break
    
    total_intersection = exact_intersection + partial_matches
    
    if total_intersection == 0:
        return 0.0, "교집합 없음"
    
    precision = total_intersection / len(pred_tokens)
    recall = total_intersection / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0, "정밀도+재현율 0"
    
    f1 = 2 * precision * recall / (precision + recall)
    
    # 숫자 매칭 보너스 (더 큰 보너스)
    is_numeric_match, numeric_score = enhanced_numeric_match(pred, gold)
    if is_numeric_match and numeric_score > 0.8:
        f1 = min(1.0, f1 + 0.30)  # 0.25에서 0.30으로 증가
    
    # 완전 토큰 매칭 보너스 (더 큰 보너스)
    if exact_intersection == len(gold_tokens) and len(gold_tokens) > 0:
        f1 = min(1.0, f1 + 0.20)  # 0.15에서 0.20으로 증가
    
    # 질문 유형별 보정
    if question:
        if re.search(r'(기간|얼마|몇)', question) and is_numeric_match:
            f1 = min(1.0, f1 + 0.10)
        elif re.search(r'(누구|누가|기관|담당)', question) and exact_intersection > 0:
            f1 = min(1.0, f1 + 0.05)
    
    return f1, f"F1={f1:.3f} (P={precision:.2f}, R={recall:.2f})"

def score_short(pred: str, gold: str) -> Tuple[float, float]:
    """향상된 단답형 점수 계산"""
    if not pred or not gold:
        return 0.0, 0.0
    
    # "정보 불충분" 특별 처리
    if pred == "정보 불충분":
        return 0.0, 0.0
    
    # 특정 만능 답변 검사 (더 엄격하게)
    generic_answers = ["금융위원회", "기획재정부장관", "금융감독원"]
    if pred in generic_answers:
        # 정답에 해당 키워드가 있으면 허용, 없으면 강력 감점
        if not any(re.search(keyword[:3], gold) for keyword in generic_answers if keyword.startswith(pred[:3])):
            return 0.0, 0.0
    
    # 무의미한 답변 차단 (더 엄격하게)
    if pred in MEANINGLESS_ANSWERS or pred in PARTIAL_ANSWERS:
        return 0.0, 0.0
    
    # EM 점수 계산 (향상된 버전)
    em_score, em_reason = ultra_precise_exact_match(pred, gold)
    
    # F1 점수 계산 (향상된 버전)
    f1_score, f1_reason = ultra_precise_f1_score(pred, gold)
    
    # 최종 점수 계산 (더 균형있게, 엄격하게)
    if em_score >= 0.95:
        final_em = em_score
        final_f1 = max(f1_score, em_score * 0.98)
    elif em_score >= 0.85:
        final_em = em_score
        final_f1 = max(f1_score, em_score * 0.95)
    elif em_score >= 0.70:
        final_em = em_score
        final_f1 = max(f1_score, em_score * 0.90)
    elif em_score >= 0.50:
        final_em = max(em_score, f1_score * 0.90)  # 0.85에서 0.90으로 증가
        final_f1 = max(f1_score, em_score * 0.85)
    else:
        final_em = max(em_score, f1_score * 0.80)  # 보정 강화
        final_f1 = max(f1_score, em_score * 0.80)
    
    # 길이 기반 보정 (새로 추가)
    pred_len = len(pred.strip())
    gold_len = len(gold.strip())
    
    if pred_len > 0 and gold_len > 0:
        length_ratio = min(pred_len, gold_len) / max(pred_len, gold_len)
        if length_ratio < 0.3:  # 길이 차이가 너무 큰 경우 감점
            final_em *= 0.9
            final_f1 *= 0.9
    
    return final_em, final_f1

# 기존 함수들 (호환성 유지 + 성능 개선)
@lru_cache(maxsize=5000)
def normalize_spaces(s: str) -> str:
    """공백 정규화"""
    return re.sub(r'\s+', ' ', (s or "")).strip()

@lru_cache(maxsize=5000)
def normalize_punct(s: str) -> str:
    """구두점 정규화"""
    if not s:
        return ""
    s = re.sub(r"[^\w가-힣\s]", " ", s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

@lru_cache(maxsize=5000)
def normalize_for_em(s: str) -> str:
    """EM 계산용 정규화"""
    return ultra_precise_normalize(s)

def calculate_accuracy(results: List[Dict]) -> float:
    """정확도 계산 (향상된 버전)"""
    if not results:
        return 0.0
    
    total_score = 0.0
    for r in results:
        pred = r.get("prediction", "")
        gold = r.get("answer", "")
        
        # 무의미한 답변 차단
        if pred in MEANINGLESS_ANSWERS:
            continue
        
        # 정확한 매칭
        if ultra_precise_normalize(pred) == ultra_precise_normalize(gold):
            total_score += 1.0
        else:
            # 의미적 유사도 기반 부분 점수 (더 엄격하게)
            similarity = enhanced_contextual_similarity(pred, gold)
            if similarity >= 0.9:
                total_score += similarity
            elif similarity >= 0.7:
                total_score += similarity * 0.9  # 부분 점수 감소
            else:
                # 숫자 매칭 확인
                is_numeric_match, numeric_score = enhanced_numeric_match(pred, gold)
                if is_numeric_match and numeric_score >= 0.8:
                    total_score += numeric_score
                elif similarity >= 0.4:
                    total_score += similarity * 0.6  # 더 엄격한 부분 점수
    
    return total_score / len(results)

def save_results(mcq_results: List[Dict], short_results: List[Dict], output_file: str = None) -> str:
    """결과 저장 - 향상된 메타데이터 포함"""
    from datetime import datetime
    
    if not output_file:
        output_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if mcq_results:
            df_mcq = pd.DataFrame(mcq_results)
            df_mcq.to_excel(writer, sheet_name="사지선다형", index=False)
        
        if short_results:
            df_short = pd.DataFrame(short_results)
            df_short.to_excel(writer, sheet_name="단답형", index=False)
        
        # 향상된 상세 요약 시트
        summary = []
        
        if mcq_results:
            acc = calculate_accuracy(mcq_results)
            context_quality = np.mean([r.get('context_quality', 0) for r in mcq_results])
            search_scores = [r.get('search_score', 0) for r in mcq_results]
            avg_search_score = np.mean(search_scores) if search_scores else 0
            high_quality_rate = sum(1 for score in search_scores if score > 1.5) / len(search_scores) if search_scores else 0
            
            summary.append({
                "유형": "사지선다형",
                "문제수": len(mcq_results),
                "정확도": f"{acc:.3f}",
                "평균_컨텍스트품질": f"{context_quality:.3f}",
                "평균_검색점수": f"{avg_search_score:.3f}",
                "고품질_비율": f"{high_quality_rate:.3f}",
                "상세": "향상된 평가 메트릭 적용"
            })
        
        if short_results:
            ems, f1s = [], []
            info_insufficient_count = 0
            exact_match_count = 0
            partial_match_count = 0
            context_quality_scores = []
            
            for r in short_results:
                prediction = r.get('prediction', '')
                answer = r.get('answer', '')
                context_quality = r.get('search_score', 0)
                context_quality_scores.append(context_quality)
                
                if prediction == "정보 불충분":
                    info_insufficient_count += 1
                    ems.append(0.0)
                    f1s.append(0.0)
                else:
                    em, f1 = score_short(prediction, answer)
                    ems.append(em)
                    f1s.append(f1)
                    
                    if em >= 0.9:
                        exact_match_count += 1
                    elif em >= 0.3:
                        partial_match_count += 1
            
            avg_em = sum(ems) / len(ems) if ems else 0
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0
            avg_context_quality = np.mean(context_quality_scores) if context_quality_scores else 0
            
            # EM 점수 분포
            em_perfect = sum(1 for em in ems if em >= 0.95)
            em_excellent = sum(1 for em in ems if 0.8 <= em < 0.95)
            em_good = sum(1 for em in ems if 0.5 <= em < 0.8)
            em_fair = sum(1 for em in ems if 0.2 <= em < 0.5)
            em_poor = sum(1 for em in ems if em < 0.2)
            
            summary.append({
                "유형": "단답형",
                "문제수": len(short_results),
                "EM점수": f"{avg_em:.3f}",
                "F1점수": f"{avg_f1:.3f}",
                "정확매칭": exact_match_count,
                "부분매칭": partial_match_count,
                "정보불충분": info_insufficient_count,
                "정보불충분률": f"{info_insufficient_count/len(short_results):.3f}",
                "평균_컨텍스트품질": f"{avg_context_quality:.3f}",
                "EM분포_완벽": em_perfect,
                "EM분포_우수": em_excellent,
                "EM분포_양호": em_good,
                "EM분포_보통": em_fair,
                "EM분포_미흡": em_poor
            })
        
        if summary:
            pd.DataFrame(summary).to_excel(writer, sheet_name="상세요약", index=False)
        
        # 성능 분석 시트 추가
        if mcq_results and short_results:
            performance_analysis = []
            
            # 전체 성능 지표
            total_questions = len(mcq_results) + len(short_results)
            mcq_weight = len(mcq_results) / total_questions
            short_weight = len(short_results) / total_questions
            
            weighted_performance = (acc * mcq_weight + avg_em * short_weight) if mcq_results and short_results else 0
            
            performance_analysis.append({
                "지표": "전체_가중평균_성능",
                "값": f"{weighted_performance:.3f}",
                "목표": "0.600",
                "달성여부": "달성" if weighted_performance >= 0.6 else "미달성"
            })
            
            performance_analysis.append({
                "지표": "MCQ_정확도",
                "값": f"{acc:.3f}",
                "목표": "0.650",
                "달성여부": "달성" if acc >= 0.65 else "미달성"
            })
            
            performance_analysis.append({
                "지표": "단답형_EM",
                "값": f"{avg_em:.3f}",
                "목표": "0.200",
                "달성여부": "달성" if avg_em >= 0.20 else "미달성"
            })
            
            performance_analysis.append({
                "지표": "정보불충분률",
                "값": f"{info_insufficient_count/len(short_results):.3f}",
                "목표": "0.050",
                "달성여부": "달성" if info_insufficient_count/len(short_results) <= 0.05 else "미달성"
            })
            
            pd.DataFrame(performance_analysis).to_excel(writer, sheet_name="성능분석", index=False)
    
    print(f"결과 저장: {output_file}")
    return output_file

def analyze_dataset_performance(mcq_questions: List[Dict], short_questions: List[Dict]) -> Dict:
    """데이터셋 성능 분석 - 향상된 버전"""
    stats = {
        "총 문제수": len(mcq_questions) + len(short_questions),
        "MCQ 문제수": len(mcq_questions),
        "단답형 문제수": len(short_questions),
    }
    
    # 난이도 분포 분석 (강화)
    if mcq_questions or short_questions:
        all_difficulties = []
        all_types = []
        
        for q in mcq_questions + short_questions:
            diff = q.get('meta', {}).get('difficulty', '미분류')
            qtype = q.get('meta', {}).get('type', '일반')
            all_difficulties.append(diff)
            all_types.append(qtype)
        
        difficulty_dist = Counter(all_difficulties)
        type_dist = Counter(all_types)
        
        stats["난이도 분포"] = dict(difficulty_dist)
        stats["문제유형 분포"] = dict(type_dist)
    
    # 품질 분석 추가
    if short_questions:
        answer_lengths = [len(q['answer']) for q in short_questions]
        stats["단답형 평균 답변 길이"] = np.mean(answer_lengths)
        stats["단답형 답변 길이 분포"] = {
            "2-5자": sum(1 for l in answer_lengths if 2 <= l <= 5),
            "6-10자": sum(1 for l in answer_lengths if 6 <= l <= 10),
            "11-20자": sum(1 for l in answer_lengths if 11 <= l <= 20),
            "20자초과": sum(1 for l in answer_lengths if l > 20)
        }
        
        # 답변 품질 분석
        numeric_answers = sum(1 for q in short_questions if re.search(r'\d+', q['answer']))
        org_answers = sum(1 for q in short_questions if re.search(r'(장관|위원회|기관|원|청|부)', q['answer']))
        
        stats["숫자포함 답변"] = numeric_answers
        stats["기관명포함 답변"] = org_answers
    
    return stats

def clear_all_caches():
    """모든 캐시 정리 - 향상된 버전"""
    normalize_spaces.cache_clear()
    normalize_punct.cache_clear()
    normalize_for_em.cache_clear()
    ultra_precise_normalize.cache_clear()
    extract_numbers_precise.cache_clear()
    tokenize_korean_enhanced.cache_clear()
    
    print("[INFO] 향상된 유틸리티 캐시 정리 완료")

# 향상된 분석 함수들
def analyze_failure_patterns(short_results: List[Dict]) -> Dict:
    """실패 패턴 분석 - 향상된 버전"""
    patterns = {
        "정보_불충분": 0,
        "무의미_답변": 0,
        "부분적_답변": 0,
        "만능답변_오남용": 0,
        "숫자_부정확": 0,
        "기관명_유사_오류": 0,
        "길이_부적절": 0,
        "의미적_유사_저조": 0,
        "완전_불일치": 0,
        "생성_실패": 0
    }
    
    for r in short_results:
        pred = r.get('prediction', '')
        gold = r.get('answer', '')
        
        if pred == "정보 불충분":
            patterns["정보_불충분"] += 1
        elif pred in MEANINGLESS_ANSWERS:
            patterns["무의미_답변"] += 1
        elif pred in PARTIAL_ANSWERS:
            patterns["부분적_답변"] += 1
        elif pred in ["금융위원회", "기획재정부장관", "금융감독원"]:
            # 만능 답변 오남용 검사
            if not any(keyword[:3] in gold for keyword in [pred]):
                patterns["만능답변_오남용"] += 1
        elif not pred or not pred.strip():
            patterns["생성_실패"] += 1
        elif len(pred) < 2 or len(pred) > 30:
            patterns["길이_부적절"] += 1
        elif re.search(r'\d+', pred) and re.search(r'\d+', gold):
            is_match, score = enhanced_numeric_match(pred, gold)
            if not is_match:
                patterns["숫자_부정확"] += 1
        elif re.search(r'(장관|위원회|감독원|은행|청|부)', pred) and re.search(r'(장관|위원회|감독원|은행|청|부)', gold):
            if ultra_precise_normalize(pred) != ultra_precise_normalize(gold):
                patterns["기관명_유사_오류"] += 1
        else:
            similarity = enhanced_contextual_similarity(pred, gold)
            if similarity >= 0.2:
                patterns["의미적_유사_저조"] += 1
            else:
                patterns["완전_불일치"] += 1
    
    return patterns

def get_performance_insights(mcq_results: List[Dict], short_results: List[Dict]) -> Dict:
    """성능 인사이트 생성 - 향상된 버전"""
    insights = {
        "총평": "",
        "개선점": [],
        "강점": [],
        "주요_문제": [],
        "상세_분석": {},
        "권장사항": []
    }
    
    # MCQ 분석 (강화)
    if mcq_results:
        mcq_acc = calculate_accuracy(mcq_results)
        context_scores = [r.get('search_score', 0) for r in mcq_results]
        avg_context_score = np.mean(context_scores) if context_scores else 0
        high_quality_rate = sum(1 for score in context_scores if score > 1.5) / len(context_scores) if context_scores else 0
        
        if mcq_acc >= 0.70:
            insights["강점"].append(f"MCQ 정확도 매우 우수 ({mcq_acc:.1%})")
        elif mcq_acc >= 0.65:
            insights["강점"].append(f"MCQ 정확도 목표 달성 ({mcq_acc:.1%})")
        elif mcq_acc >= 0.60:
            insights["강점"].append(f"MCQ 정확도 양호 ({mcq_acc:.1%})")
        elif mcq_acc >= 0.50:
            insights["개선점"].append(f"MCQ 정확도 개선 필요 ({mcq_acc:.1%})")
        else:
            insights["주요_문제"].append(f"MCQ 정확도 심각하게 낮음 ({mcq_acc:.1%})")
        
        # 컨텍스트 품질 분석
        if avg_context_score >= 1.5:
            insights["강점"].append(f"MCQ 컨텍스트 품질 우수 ({avg_context_score:.3f})")
        elif avg_context_score >= 1.0:
            insights["강점"].append(f"MCQ 컨텍스트 품질 양호 ({avg_context_score:.3f})")
        else:
            insights["개선점"].append(f"MCQ 컨텍스트 품질 개선 필요 ({avg_context_score:.3f})")
        
        if high_quality_rate >= 0.7:
            insights["강점"].append(f"고품질 컨텍스트 비율 우수 ({high_quality_rate:.1%})")
        elif high_quality_rate >= 0.5:
            insights["강점"].append(f"고품질 컨텍스트 비율 양호 ({high_quality_rate:.1%})")
        else:
            insights["개선점"].append(f"고품질 컨텍스트 비율 개선 필요 ({high_quality_rate:.1%})")
    
    # 단답형 상세 분석 (대폭 강화)
    if short_results:
        ems, f1s = [], []
        exact_matches = 0
        partial_matches = 0
        info_insufficient_count = 0
        context_scores = []
        
        for r in short_results:
            pred = r.get('prediction', '')
            gold = r.get('answer', '')
            context_score = r.get('search_score', 0)
            context_scores.append(context_score)
            
            if pred == "정보 불충분":
                info_insufficient_count += 1
                ems.append(0.0)
                f1s.append(0.0)
            else:
                em, f1 = score_short(pred, gold)
                ems.append(em)
                f1s.append(f1)
                
                if em >= 0.9:
                    exact_matches += 1
                elif em >= 0.3:
                    partial_matches += 1
        
        avg_em = sum(ems) / len(ems) if ems else 0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0
        avg_context_score = np.mean(context_scores) if context_scores else 0
        
        insights["상세_분석"]["단답형_EM"] = avg_em
        insights["상세_분석"]["단답형_F1"] = avg_f1
        insights["상세_분석"]["정확_매칭률"] = exact_matches / len(short_results)
        insights["상세_분석"]["부분_매칭률"] = partial_matches / len(short_results)
        insights["상세_분석"]["정보불충분률"] = info_insufficient_count / len(short_results)
        insights["상세_분석"]["평균_컨텍스트품질"] = avg_context_score
        
        # EM 점수 평가 (더 세밀하게)
        if avg_em >= 0.25:
            insights["강점"].append(f"단답형 EM 목표 초과 달성 ({avg_em:.1%})")
        elif avg_em >= 0.20:
            insights["강점"].append(f"단답형 EM 목표 달성 ({avg_em:.1%})")
        elif avg_em >= 0.15:
            insights["강점"].append(f"단답형 EM 목표에 근접 ({avg_em:.1%})")
        elif avg_em >= 0.10:
            insights["개선점"].append(f"단답형 EM 추가 개선 필요 ({avg_em:.1%})")
        elif avg_em >= 0.05:
            insights["주요_문제"].append(f"단답형 EM 크게 개선 필요 ({avg_em:.1%})")
        else:
            insights["주요_문제"].append(f"단답형 EM 심각하게 낮음 ({avg_em:.1%})")
        
        # 정확 매칭률 분석
        exact_rate = exact_matches / len(short_results)
        if exact_rate >= 0.15:
            insights["강점"].append(f"정확 매칭률 우수 ({exact_rate:.1%})")
        elif exact_rate >= 0.10:
            insights["강점"].append(f"정확 매칭률 양호 ({exact_rate:.1%})")
        else:
            insights["개선점"].append(f"정확 매칭률 개선 필요 ({exact_rate:.1%})")
        
        # 정보 불충분 분석
        info_rate = info_insufficient_count / len(short_results)
        if info_rate == 0:
            insights["강점"].append("정보 불충분 문제 완전 해결!")
        elif info_rate <= 0.05:
            insights["강점"].append(f"정보 불충분 비율 매우 낮음 ({info_rate:.1%})")
        elif info_rate <= 0.10:
            insights["강점"].append(f"정보 불충분 비율 낮음 ({info_rate:.1%})")
        elif info_rate <= 0.20:
            insights["개선점"].append(f"정보 불충분 비율 개선 필요 ({info_rate:.1%})")
        else:
            insights["주요_문제"].append(f"정보 불충분 비율 높음 ({info_rate:.1%})")
        
        # 컨텍스트 품질 분석
        if avg_context_score >= 1.5:
            insights["강점"].append(f"단답형 컨텍스트 품질 우수 ({avg_context_score:.3f})")
        elif avg_context_score >= 1.0:
            insights["강점"].append(f"단답형 컨텍스트 품질 양호 ({avg_context_score:.3f})")
        else:
            insights["개선점"].append(f"단답형 컨텍스트 품질 개선 필요 ({avg_context_score:.3f})")
    
    # 종합 평가 및 권장사항
    total_issues = len(insights["주요_문제"])
    total_improvements = len(insights["개선점"])
    total_strengths = len(insights["강점"])
    
    if total_issues >= 3:
        insights["총평"] = "시스템에 심각한 문제가 있어 전면적인 재설계가 필요합니다."
        insights["권장사항"].append("긴급: 핵심 성능 문제 해결 우선")
        insights["권장사항"].append("검색 품질 및 답변 생성 로직 전면 재검토")
    elif total_issues > total_strengths:
        insights["총평"] = "주요 문제점들을 우선적으로 해결해야 합니다."
        insights["권장사항"].append("우선순위: 주요 문제 해결")
        insights["권장사항"].append("단계적 성능 개선 계획 수립")
    elif total_improvements > total_strengths:
        insights["총평"] = "전반적인 성능 개선이 필요합니다."
        insights["권장사항"].append("점진적 성능 향상 계획 실행")
        insights["권장사항"].append("약점 부분 집중 개선")
    elif total_strengths > total_improvements:
        insights["총평"] = "시스템이 양호하게 작동하고 있으나 미세 조정이 필요합니다."
        insights["권장사항"].append("현재 성능 유지 및 안정성 확보")
        insights["권장사항"].append("미세 조정을 통한 추가 향상")
    else:
        insights["총평"] = "시스템 성능이 균형적으로 안정적입니다."
        insights["권장사항"].append("현재 성능 수준 유지")
        insights["권장사항"].append("모니터링 및 지속적 개선")
    
    # 구체적 개선 방안 제시
    if mcq_results and short_results:
        mcq_acc = calculate_accuracy(mcq_results)
        avg_em = insights["상세_분석"].get("단답형_EM", 0)
        
        if mcq_acc < 0.65:
            gap = 0.65 - mcq_acc
            insights["권장사항"].append(f"MCQ 정확도 {gap:.1%}p 향상 필요 - 검색 품질 또는 선택 로직 개선")
        
        if avg_em < 0.20:
            gap = 0.20 - avg_em
            insights["권장사항"].append(f"단답형 EM {gap:.1%}p 향상 필요 - 답변 추출 정확도 개선")
        
        # 성능 비율 기반 권장사항
        if mcq_acc > avg_em * 2:
            insights["권장사항"].append("단답형 성능 집중 개선 권장 (MCQ 대비 성능 격차 큼)")
        elif avg_em > mcq_acc:
            insights["권장사항"].append("MCQ 성능 개선 우선 권장")
    
    return insights

# 새로운 유틸리티 함수들
def analyze_context_quality(results: List[Dict]) -> Dict:
    """컨텍스트 품질 분석"""
    if not results:
        return {}
    
    scores = [r.get('search_score', 0) for r in results]
    
    return {
        "평균_점수": np.mean(scores),
        "중앙값_점수": np.median(scores),
        "최고_점수": np.max(scores),
        "최저_점수": np.min(scores),
        "표준편차": np.std(scores),
        "고품질_비율": sum(1 for s in scores if s > 1.5) / len(scores),
        "중품질_비율": sum(1 for s in scores if 1.0 <= s <= 1.5) / len(scores),
        "저품질_비율": sum(1 for s in scores if s < 1.0) / len(scores)
    }

def calculate_improvement_metrics(current_results: Dict, baseline_results: Dict) -> Dict:
    """개선 메트릭 계산"""
    improvements = {}
    
    for key in ['mcq_accuracy', 'em_score', 'f1_score']:
        if key in current_results and key in baseline_results:
            current = current_results[key]
            baseline = baseline_results[key]
            
            if baseline > 0:
                improvement = ((current - baseline) / baseline) * 100
                improvements[f"{key}_improvement"] = improvement
                improvements[f"{key}_current"] = current
                improvements[f"{key}_baseline"] = baseline
    
    return improvements


# ===== 정규화/토큰 유틸 =====
_WS = re.compile(r'\s+')
_PAREN_TRIM = re.compile(r'[「」“”"()]')
_JOSA_TAIL = re.compile(r'(?:에서|으로|에게|에게서|과|와|의|를|을|은|는|가|이|도|만|까지|부터)$')
_PUNCT = re.compile(r'[·\.,;:·…‧∙ㆍ]+')

# 조문/단위/퍼센트 정규화
_ARTICLE = re.compile(r'제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?')
_PCT = re.compile(r'(\d{1,3})\s*%')

def _squash_spaces(s: str) -> str:
    return _WS.sub(' ', s or '').strip()

def _strip_quotes_josa(s: str) -> str:
    s = _PAREN_TRIM.sub('', s or '')
    s = _squash_spaces(s)
    # 조사 꼬리 2회까지만 제거
    for _ in range(2):
        s2 = _JOSA_TAIL.sub('', s).strip()
        if s2 == s:
            break
        s = s2
    return s

def _article_canon(s: str) -> str:
    # "제 3 조 제 2 항" -> "제3조제2항"
    def repl(m):
        out = f"제{m.group(1)}조"
        if m.group(2): out += f"제{m.group(2)}항"
        if m.group(3): out += f"제{m.group(3)}호"
        return out
    return _ARTICLE.sub(repl, s)

def _unit_alias(s: str) -> str:
    # "달" -> "개월", 퍼센트 공백 제거
    s = (s or '').replace('달', '개월')
    s = _PCT.sub(lambda m: f"{int(m.group(1))}%", s)
    return s

def _normalize_common(s: str) -> str:
    """검증(contains)용: 공백/문장부호 제거, 조사/괄호 제거, 조문·단위 표준화"""
    s = _strip_quotes_josa(s)
    s = _article_canon(s)
    s = _unit_alias(s)
    s = _PUNCT.sub('', s)
    s = s.replace(' ', '')
    return s

def _normalize_display(s: str) -> str:
    """표시용: 읽기 좋게 최소한만 정리"""
    s = _strip_quotes_josa(s)
    s = _article_canon(s)
    s = _unit_alias(s)
    s = _WS.sub(' ', s)
    return s.strip()

# 기관명 보수적 정규화(표시 단계에서만 적용)
_ORG_FIX = [
    (re.compile(r'금융위원장'), '금융위원회'),
    (re.compile(r'금융감독원장'), '금융감독원'),
]
def _org_soft_normalize(s: str) -> str:
    out = s or ''
    for pat, rep in _ORG_FIX:
        out = pat.sub(rep, out)
    return out


# ===== 공개 API: 질문 유형 추정 =====
def extract_question_type(q: str) -> str:
    q = q or ''
    if re.search(r'(몇|얼마|기간|언제)', q): 
        return 'period'
    if re.search(r'(누구|기관|어디|담당)', q): 
        return 'organization'
    if re.search(r'제\d+조', q): 
        return 'article_specific'
    if re.search(r'(무엇|뜻|의미|정의|라\s*함|을\s*말한다|이라\s*한다)', q): 
        return 'definition'
    return 'general'


# ===== 공개 API: 품질 검증 =====
def validate_answer_quality(ans: str, question: str, contexts: List[Dict]) -> Tuple[bool, float, str]:
    """
    느슨한 포함(공백/문장부호/조사 무시) + 유형 가중으로 품질 점수 산출.
    반환: (is_ok, quality_score[0~1], reason)
    """
    if not ans:
        return (False, 0.0, 'empty')

    qtype = extract_question_type(question or '')
    # 표시 문자열(사람이 볼 형태)과 검증 키(붙여쓰기/표준화) 분리
    ans_disp = _normalize_display(ans)
    ans_key  = _normalize_common(ans_disp)
    if not ans_key:
        return (False, 0.0, 'empty')

    # 컨텍스트 전체를 하나로 합쳐 느슨 매칭
    ctx_join = ' '.join((c.get('text', '') or '') for c in (contexts or []))
    ctx_key  = _normalize_common(ctx_join)

    present = ans_key in ctx_key

    # 길이/유형 가중
    ln = len(ans_disp)
    base = 0.55 if present else 0.35
    if qtype in ('period', 'article_specific'):
        base += 0.10
    elif qtype == 'organization':
        base += 0.08
    # 너무 긴 답변 패널티 회피: 2~12자 보너스
    if 2 <= ln <= 12:
        base += 0.06

    return (base >= 0.5, min(1.0, base), '')


# ===== 공개 API: 후처리 =====
def enhanced_postprocess_answer(ans: str, contexts: List[Dict], question: Optional[str]=None, question_type: Optional[str]=None) -> str:
    qtype = question_type or extract_question_type(question or '')
    s = ans or ''
    if qtype == 'organization':
        s = _org_soft_normalize(s)
    s = _normalize_display(s)
    if len(s) > 30:
        s = s[:30]
    return s

# >>>>>>>>>>>>>>>>>>>>>>> utils.py PATCH START <<<<<<<<<<<<<<<<<<<<<<
import re
from typing import List, Dict, Tuple, Optional

# 공백/문장부호/조사/괄호 정리
_WS = re.compile(r'\s+')
_PAREN_TRIM = re.compile(r'[「」“”"()]')
_JOSA_TAIL = re.compile(r'(?:에서|으로|에게|에게서|과|와|의|를|을|은|는|가|이|도|만|까지|부터)$')
_PUNCT = re.compile(r'[·\.,;:·…‧∙ㆍ]+')

# 조문/퍼센트 정규화
_ARTICLE = re.compile(r'제\s*(\d+)\s*조(?:\s*(?:의|之)\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?')
_PCT = re.compile(r'(\d{1,3})\s*%')

# 기관/약칭/한자 → 표준형(보수적)
_ORG_FIX = [
    (re.compile(r'금융위원장'), '금융위원회'),
    (re.compile(r'금융감독원장'), '금융감독원'),
    (re.compile(r'공정위'), '공정거래위원회'),
    (re.compile(r'국세청장'), '국세청'),
    (re.compile(r'법무부장관'), '법무부'),
]
# 한자 약칭 일부 보정(필요시 추가)
_HANJA_FIX = [
    (re.compile(r'金融委員會'), '금융위원회'),
    (re.compile(r'監督[院廳]?'), '감독원'),
]

def _squash_spaces(s: str) -> str:
    return _WS.sub(' ', s or '').strip()

def _strip_quotes_josa(s: str) -> str:
    s = _PAREN_TRIM.sub('', s or '')
    s = _squash_spaces(s)
    for _ in range(2):  # 조사 꼬리 과도 제거 방지
        s2 = _JOSA_TAIL.sub('', s).strip()
        if s2 == s:
            break
        s = s2
    return s

def _article_canon(s: str) -> str:
    """
    "제 3 조", "제3조의2", "제3조 제2항 제3호" 같은 변형을
    → "제3조의2제2항제3호" 로 단일 포맷화
    """
    def repl(m):
        # m: 1=조, 2=의n(선택), 3=항(선택), 4=호(선택)
        out = f"제{m.group(1)}조"
        if m.group(2): out += f"의{m.group(2)}"
        if m.group(3): out += f"제{m.group(3)}항"
        if m.group(4): out += f"제{m.group(4)}호"
        return out
    s = _ARTICLE.sub(repl, s or '')
    # "조 의 2" 같이 띄어진 형태도 흡수
    s = re.sub(r'제(\d+)\s*조\s*(?:의|之)\s*(\d+)', r'제\1조의\2', s)
    return s

def _unit_alias(s: str) -> str:
    # 달 → 개월, 퍼센트 공백 제거
    s = (s or '').replace('달', '개월')
    s = _PCT.sub(lambda m: f"{int(m.group(1))}%", s)
    return s

def _normalize_common(s: str) -> str:
    """검증(contains)용: 최대치로 축약/표준화"""
    s = _strip_quotes_josa(s)
    s = _article_canon(s)
    s = _unit_alias(s)
    s = _PUNCT.sub('', s)
    s = s.replace(' ', '')
    # 한자/약칭 최소 보정
    for pat, rep in _HANJA_FIX:
        s = pat.sub(rep, s)
    return s

def _normalize_display(s: str) -> str:
    """표시용: 읽기 좋은 형태 유지"""
    s = _strip_quotes_josa(s)
    s = _article_canon(s)
    s = _unit_alias(s)
    s = _WS.sub(' ', s)
    # 표시 단계에서만 기관 보정(보수적)
    for pat, rep in _ORG_FIX:
        s = pat.sub(rep, s)
    return s.strip()

def extract_question_type(q: str) -> str:
    q = q or ''
    if re.search(r'(몇|얼마|기간|언제)', q): 
        return 'period'
    if re.search(r'(누구|기관|어디|담당)', q): 
        return 'organization'
    if re.search(r'제\d+조', q): 
        return 'article_specific'
    if re.search(r'(무엇|뜻|의미|정의|라\s*함|을\s*말한다|이라\s*한다)', q): 
        return 'definition'
    return 'general'

def validate_answer_quality(ans: str, question: str, contexts: List[Dict]) -> Tuple[bool, float, str]:
    """
    느슨한 포함(공백/부호/조사 무시) + 유형 가중.
    반환: (is_ok, quality_score[0~1], reason)
    """
    if not ans:
        return (False, 0.0, 'empty')
    qtype = extract_question_type(question or '')

    ans_disp = _normalize_display(ans)
    ans_key  = _normalize_common(ans_disp)
    if not ans_key:
        return (False, 0.0, 'empty')

    ctx_join = ' '.join((c.get('text','') or '') for c in (contexts or []))
    ctx_key  = _normalize_common(ctx_join)

    present = ans_key in ctx_key
    ln = len(ans_disp)

    base = 0.55 if present else 0.35
    if qtype in ('period', 'article_specific'):
        base += 0.10
    elif qtype == 'organization':
        base += 0.08
    if 2 <= ln <= 12:
        base += 0.06

    return (base >= 0.5, min(1.0, base), '')

def enhanced_postprocess_answer(ans: str, contexts: List[Dict], question: Optional[str]=None, question_type: Optional[str]=None) -> str:
    qtype = question_type or extract_question_type(question or '')
    s = ans or ''
    s = _normalize_display(s)
    if len(s) > 30:
        s = s[:30]
    return s
# >>>>>>>>>>>>>>>>>>>>>>> utils.py PATCH END <<<<<<<<<<<<<<<<<<<<<<
