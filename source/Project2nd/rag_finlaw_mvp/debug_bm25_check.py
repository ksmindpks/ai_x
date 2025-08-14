#!/usr/bin/env python
"""
BM25 인덱스 구조 확인 및 테스트 스크립트
"""

import pickle
import os
from typing import List, Dict

def check_bm25_index(bm25_path: str = "./bm25_pkg/out/bm25_index/bm25.pkl"):
    """BM25 인덱스 파일 구조 확인"""
    
    if not os.path.exists(bm25_path):
        print(f"ERROR: BM25 파일 없음: {bm25_path}")
        return False
    
    print(f"BM25 파일: {bm25_path}")
    print(f"파일 크기: {os.path.getsize(bm25_path):,} bytes")
    
    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"데이터 타입: {type(data)}")
        
        if isinstance(data, dict):
            print(f"사용 가능한 키들: {list(data.keys())}")
            
            # 키별 정보 확인
            for key, value in data.items():
                print(f"  - {key}: {type(value)}")
                if key == "corpus" and isinstance(value, list):
                    print(f"    문서 수: {len(value)}")
                    if value:
                        first_doc = value[0]
                        print(f"    첫 번째 문서 구조: {type(first_doc)}")
                        if isinstance(first_doc, dict):
                            print(f"      키들: {list(first_doc.keys())}")
                        print(f"      미리보기: {str(first_doc)[:200]}...")
                
                elif key == "bm25":
                    print(f"    BM25 타입: {type(value)}")
                    if hasattr(value, '__dict__'):
                        print(f"    BM25 속성: {list(value.__dict__.keys())}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 로드 오류: {e}")
        return False

def test_bm25_search(bm25_path: str = "./bm25_pkg/out/bm25_index/bm25.pkl"):
    """BM25 검색 테스트"""
    
    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        
        # BM25와 corpus 추출
        if isinstance(data, dict):
            bm25 = data.get("bm25")
            corpus = data.get("corpus", [])
        else:
            print("ERROR: 예상과 다른 데이터 구조")
            return
        
        if not bm25 or not corpus:
            print("ERROR: BM25 또는 corpus가 없음")
            return
        
        print(f"\nBM25 검색 테스트 시작 (문서 수: {len(corpus)})")
        
        # 테스트 쿼리들
        test_queries = [
            "개인금융채권",
            "대통령령",
            "금융거래지표",
            "국제금융기구"
        ]
        
        # 재구축된 인덱스에 맞는 토크나이저 사용
        def get_proper_tokenizer():
            """재구축된 인덱스와 일치하는 토크나이저"""
            try:
                from kiwipiepy import Kiwi
                kiwi = Kiwi()
                
                def kiwi_tokenizer(text):
                    tokens = [tok.form for tok in kiwi.tokenize(text)]
                    meaningful = []
                    stopwords = {'의', '에', '으로', '는', '은', '이', '가', '를', '을', '와', '과'}
                    
                    for token in tokens:
                        if (len(token) >= 2 and 
                            token not in stopwords and
                            any(ord('가') <= ord(c) <= ord('힣') for c in token)):
                            meaningful.append(token)
                    return meaningful
                
                print("   토크나이저: Kiwi (재구축된 인덱스와 일치)")
                return kiwi_tokenizer
                
            except Exception as e:
                print(f"   WARNING: Kiwi 로드 실패: {e}")
                print("   토크나이저: 간단한 방식 (불일치 가능)")
                
                def simple_tokenize(text):
                    import re
                    words = re.findall(r'[가-힣]{2,}', text)
                    return words[:10]
                    
                return simple_tokenize
        
        tokenizer = get_proper_tokenizer()
        
        for query in test_queries:
            print(f"\n검색어: '{query}'")
            
            tokenized = tokenizer(query)
            print(f"   토큰: {tokenized}")
            
            if tokenized:
                try:
                    import numpy as np
                    scores = bm25.get_scores(tokenized)
                    
                    # numpy 배열 처리
                    if hasattr(scores, 'max'):
                        max_score = float(scores.max())
                        non_zero = int(np.sum(scores > 0))
                    else:
                        max_score = max(scores) if scores else 0
                        non_zero = sum(1 for s in scores if s > 0)
                    
                    print(f"   최고 점수: {max_score:.3f}")
                    print(f"   0점이 아닌 결과: {non_zero}/{len(scores)}")
                    
                    # 상위 3개 결과
                    if max_score > 0:
                        if hasattr(scores, 'argsort'):
                            ranked_idx = scores.argsort()[::-1]  # numpy 방식
                        else:
                            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                        
                        print("   상위 3개 결과:")
                        for i in range(min(3, len(ranked_idx))):
                            idx = ranked_idx[i]
                            score = float(scores[idx])
                            if score > 0:
                                doc_text = corpus[idx].get("text", "")[:100] if isinstance(corpus[idx], dict) else str(corpus[idx])[:100]
                                print(f"      {i+1}. 점수: {score:.3f} | {doc_text}...")
                
                except Exception as e:
                    print(f"   검색 오류: {e}")
                    import traceback
                    print(f"   상세 오류: {traceback.format_exc()}")
            else:
                print("   WARNING: 토큰화 결과 없음")
    
    except Exception as e:
        print(f"ERROR: 테스트 오류: {e}")

def main():
    print("BM25 인덱스 진단 도구")
    print("=" * 50)
    
    # 1. 구조 확인
    success = check_bm25_index()
    
    if success:
        print("\n" + "=" * 50)
        # 2. 검색 테스트
        test_bm25_search()
    
    print("\n진단 완료")

if __name__ == "__main__":
    main()