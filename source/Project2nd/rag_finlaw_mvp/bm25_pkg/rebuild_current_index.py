#!/usr/bin/env python
"""
bm25_pkg/rebuild_current_index.py

현재 ./bm25_pkg/out/bm25_index/bm25.pkl 인덱스를 
일관된 토크나이저로 재구축하여 토크나이저 불일치 문제 해결
"""

import os
import sys
import pickle
import re
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[ERROR] rank-bm25가 설치되어 있지 않습니다.")
    print("설치: pip install rank-bm25")
    sys.exit(1)

def get_consistent_tokenizer():
    """일관된 토크나이저 - hybrid_retriever.py에서도 동일하게 사용할 것"""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        def kiwi_tokenizer(text):
            tokens = [tok.form for tok in kiwi.tokenize(text)]
            
            # 의미있는 토큰만 필터링
            meaningful_tokens = []
            
            # 불용어 (기존 분석 결과 참고)
            stopwords = {
                '의', '에', '으로', '는', '은', '이', '가', '를', '을', '와', '과', 
                '에서', '부터', '까지', '도', '만', '에게', '한테', '께', '로부터',
                '으며', '이며', '며', '고', '어', '아', '여', '해', '하', 'ᆫ다'
            }
            
            for token in tokens:
                # 조건: 한글 2글자 이상, 불용어 아님
                if (len(token) >= 2 and 
                    token not in stopwords and
                    any(ord('가') <= ord(c) <= ord('힣') for c in token)):
                    meaningful_tokens.append(token)
                    
                # 숫자+단위 패턴도 보존 (3천만원, 6개월 등)
                elif re.match(r'\d+[가-힣]+', token):
                    meaningful_tokens.append(token)
            
            return meaningful_tokens
        
        print("[INFO] Kiwi 토크나이저 로드 성공")
        return kiwi_tokenizer
        
    except Exception as e:
        print(f"[WARN] Kiwi 로드 실패: {e}")
        print("[INFO] 간단한 토크나이저로 폴백")
        
        def simple_tokenizer(text):
            # 한글 2글자 이상 단어 추출
            words = re.findall(r'[가-힣]{2,}', text)
            # 숫자+한글 패턴 추가
            number_units = re.findall(r'\d+[가-힣]+', text)
            return words + number_units
        
        return simple_tokenizer

def rebuild_bm25_index():
    """현재 BM25 인덱스 재구축"""
    
    # 경로 설정
    current_dir = Path(__file__).parent
    index_path = current_dir / "out" / "bm25_index" / "bm25.pkl"
    
    if not index_path.exists():
        print(f"[ERROR] 기존 인덱스를 찾을 수 없습니다: {index_path}")
        print("pipeline_bm25_from_docx.py를 먼저 실행하여 인덱스를 생성하세요.")
        return False
    
    print(f"기존 인덱스 경로: {index_path}")
    print(f"파일 크기: {index_path.stat().st_size:,} bytes")
    
    # 1. 기존 데이터 로드
    print("\n1. 기존 인덱스 로드 중...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    
    corpus = data["corpus"]
    print(f"   문서 수: {len(corpus):,}개")
    print(f"   기존 토크나이저: {data.get('tokenizer', 'unknown')}")
    
    # 2. 새 토크나이저로 재토큰화
    print("\n2. 일관된 토크나이저로 재토큰화...")
    tokenizer = get_consistent_tokenizer()
    
    # 토큰화 테스트 (문제가 되었던 단어들)
    test_texts = [
        "개인금융채권의 관리",
        "금융거래지표의 관리", 
        "국제금융기구에의 가입",
        "대통령령으로 정한다"
    ]
    
    print("   토큰화 테스트:")
    for text in test_texts:
        tokens = tokenizer(text)
        print(f"     '{text}' -> {tokens}")
    
    # 전체 corpus 재토큰화
    print("\n   전체 문서 재토큰화 중...")
    tokenized_corpus = []
    
    for i, doc in enumerate(corpus):
        tokens = tokenizer(doc["text"])
        tokenized_corpus.append(tokens)
        
        if (i + 1) % 5000 == 0:
            print(f"   진행: {i+1:,}/{len(corpus):,}")
    
    print(f"   완료: {len(tokenized_corpus):,}개 문서 토큰화")
    
    # 3. BM25 인덱스 재구축
    print("\n3. BM25 인덱스 재구축 중...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("   BM25Okapi 객체 생성 완료")
    
    # 4. 새 데이터 구성
    new_data = {
        "bm25": bm25,
        "corpus": corpus,
        "tokenizer": "kiwi_consistent_fixed",
        "built_at": datetime.now().isoformat(),
        "rebuild_reason": "tokenizer_consistency_fix"
    }
    
    # 5. 백업 후 저장
    backup_path = index_path.with_suffix('.pkl.backup')
    if index_path.exists():
        index_path.rename(backup_path)
        print(f"   기존 인덱스 백업: {backup_path}")
    
    with open(index_path, "wb") as f:
        pickle.dump(new_data, f)
    
    new_size = index_path.stat().st_size
    print(f"\n4. 새 인덱스 저장 완료")
    print(f"   경로: {index_path}")
    print(f"   크기: {new_size:,} bytes")
    
    # 6. 검색 테스트
    print("\n5. 검색 테스트 (문제였던 단어들)...")
    test_queries = ["개인금융채권", "금융거래지표", "국제금융기구", "대통령령"]
    
    for query in test_queries:
        tokens = tokenizer(query)
        if tokens:
            scores = bm25.get_scores(tokens)
            import numpy as np
            max_score = float(scores.max()) if hasattr(scores, 'max') else max(scores)
            non_zero = int(np.sum(scores > 0)) if hasattr(scores, 'max') else sum(1 for s in scores if s > 0)
            
            print(f"   '{query}':")
            print(f"     토큰: {tokens}")
            print(f"     최고점수: {max_score:.3f} (이전: 0.000)")
            print(f"     매칭문서: {non_zero:,}개")
            
            if max_score > 0:
                print(f"     ✅ 개선됨!")
            else:
                print(f"     ❌ 여전히 문제")
        else:
            print(f"   '{query}': 토큰화 결과 없음")
    
    print("\n" + "="*60)
    print("BM25 인덱스 재구축 완료!")
    print("="*60)
    print("\n다음 단계:")
    print("1. 토크나이저 테스트:")
    print("   cd ..")
    print("   python bm25_pkg/test_tokenizer.py")
    print("\n2. config.py에서 BM25 가중치 복원:")
    print("   WEIGHT_BM25 = 0.4")
    print("   WEIGHT_VEC = 0.6")
    print("\n3. 전체 평가 재실행:")
    print("   python evaluate.py 법령문제_법조문포함.xlsx --mcq 3 --short 3 --debug")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 BM25 인덱스 재구축 (토크나이저 일관성 수정)")
    print("=" * 60)
    
    success = rebuild_bm25_index()
    
    if not success:
        print("\n재구축 실패. 다음을 확인하세요:")
        print("1. bm25_pkg/out/bm25_index/bm25.pkl 파일 존재 여부")
        print("2. kiwipiepy 설치: pip install kiwipiepy")
        print("3. rank-bm25 설치: pip install rank-bm25")