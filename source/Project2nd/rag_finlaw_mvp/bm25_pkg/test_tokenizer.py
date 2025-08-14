#!/usr/bin/env python
"""
bm25_pkg/test_tokenizer.py

재구축된 BM25 인덱스의 토크나이저 테스트
"""

import os
import sys
import pickle
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_rebuilt_index():
    """재구축된 인덱스 테스트"""
    
    current_dir = Path(__file__).parent
    index_path = current_dir / "out" / "bm25_index" / "bm25.pkl"
    
    if not index_path.exists():
        print(f"ERROR: 인덱스 파일 없음: {index_path}")
        return False
    
    # 인덱스 로드
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    
    bm25 = data["bm25"]
    corpus = data["corpus"]
    tokenizer_info = data.get("tokenizer", "unknown")
    
    print(f"인덱스 정보:")
    print(f"   문서 수: {len(corpus):,}개")
    print(f"   토크나이저: {tokenizer_info}")
    print(f"   구축 시간: {data.get('built_at', 'unknown')}")
    
    # 토크나이저 재생성 (동일한 로직)
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        def tokenizer(text):
            tokens = [tok.form for tok in kiwi.tokenize(text)]
            meaningful = []
            stopwords = {'의', '에', '으로', '는', '은', '이', '가', '를', '을', '와', '과'}
            
            for token in tokens:
                if (len(token) >= 2 and 
                    token not in stopwords and
                    any(ord('가') <= ord(c) <= ord('힣') for c in token)):
                    meaningful.append(token)
            return meaningful
        
        print("SUCCESS: Kiwi 토크나이저 로드 성공")
        
    except Exception as e:
        print(f"ERROR: Kiwi 로드 실패: {e}")
        return False
    
    # 문제였던 검색어들 테스트
    problem_queries = [
        "개인금융채권",
        "금융거래지표", 
        "국제금융기구",
        "대통령령"
    ]
    
    print(f"\n검색 테스트:")
    print("-" * 50)
    
    all_fixed = True
    
    for query in problem_queries:
        tokens = tokenizer(query)
        
        if tokens:
            try:
                import numpy as np
                scores = bm25.get_scores(tokens)
                max_score = float(scores.max()) if hasattr(scores, 'max') else max(scores)
                non_zero = int(np.sum(scores > 0)) if hasattr(scores, 'max') else sum(1 for s in scores if s > 0)
                
                status = "SUCCESS: 수정됨" if max_score > 0 else "ERROR: 여전히 문제"
                if max_score == 0:
                    all_fixed = False
                
                print(f"'{query}':")
                print(f"  토큰: {tokens}")
                print(f"  최고점수: {max_score:.3f}")
                print(f"  매칭문서: {non_zero:,}개")
                print(f"  상태: {status}")
                print()
                
            except Exception as e:
                print(f"'{query}': 검색 오류 - {e}")
                all_fixed = False
        else:
            print(f"'{query}': 토큰화 결과 없음 ERROR")
            all_fixed = False
    
    # 추가 테스트: 잘 작동했던 단어들
    print("기존 작동 단어들 테스트:")
    print("-" * 50)
    
    working_queries = ["따른", "또는", "경우", "조제"]
    
    for query in working_queries:
        tokens = tokenizer(query)
        if tokens:
            scores = bm25.get_scores(tokens)
            max_score = float(scores.max()) if hasattr(scores, 'max') else max(scores)
            print(f"'{query}': {max_score:.3f} (토큰: {tokens})")
    
    print("\n" + "="*50)
    if all_fixed:
        print("SUCCESS: 모든 문제 해결됨!")
        print("이제 config.py에서 BM25 가중치를 복원하세요:")
        print("   WEIGHT_BM25 = 0.4")
        print("   WEIGHT_VEC = 0.6")
    else:
        print("WARNING: 일부 문제 남아있음. 추가 조치 필요.")
    
    return all_fixed

if __name__ == "__main__":
    print("재구축된 BM25 인덱스 테스트")
    print("=" * 50)
    
    success = test_rebuilt_index()
    
    if success:
        print("\nSUCCESS: 테스트 통과! 다음 단계로 진행하세요.")
    else:
        print("\nERROR: 테스트 실패. rebuild_current_index.py를 다시 실행하세요.")