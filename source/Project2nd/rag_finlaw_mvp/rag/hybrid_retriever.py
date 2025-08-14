# -*- coding: utf-8 -*-
"""
완전한 HybridRetriever 구현 - 디버그 간소화
"""

import os
import pickle
from typing import List, Dict, Callable
from rank_bm25 import BM25Okapi
import re
from config import DEBUG_HYBRID


class PineconeWrapper:
    def __init__(self, index):
        self.index = index

    def search(self, embedding, top_k=10, namespace=None, filter=None):
        return self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=filter
        )


class HybridConfig:
    def __init__(
        self,
        bm25_index_dir: str,
        pinecone_index: PineconeWrapper,
        top_k_bm25: int = 10,
        top_k_vec: int = 10,
        weight_bm25: float = 0.6,
        weight_vec: float = 0.4,
        tokenizer: str = "kiwi"
    ):
        self.bm25_index_dir = bm25_index_dir
        self.pinecone_index = pinecone_index
        self.top_k_bm25 = top_k_bm25
        self.top_k_vec = top_k_vec
        self.weight_bm25 = weight_bm25
        self.weight_vec = weight_vec
        self.tokenizer = tokenizer


class HybridRetriever:
    def __init__(self, cfg, embedder_fn):
        self.cfg = cfg
        self.embedder_fn = embedder_fn
        self._load_bm25()
        self.index = cfg.pinecone_index.index
        self.tokenizer = self._get_kiwi_tokenizer()

    def _load_bm25(self):
        """BM25 인덱스 로드"""
        pkl = os.path.join(self.cfg.bm25_index_dir, "bm25.pkl")
        abs_pkl = os.path.abspath(pkl)
        print(f"[HybridRetriever] BM25 file: {abs_pkl}")
        
        if not os.path.exists(pkl):
            raise FileNotFoundError(f"BM25 인덱스 없음: {abs_pkl}")
        
        print(f"[HybridRetriever] BM25 size: {os.path.getsize(pkl):,} bytes")

        with open(pkl, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if "bm25" in data and "corpus" in data:
                self.bm25 = data["bm25"]
                self.corpus = data["corpus"]
                print(f"[HybridRetriever] Loaded BM25 with {len(self.corpus)} documents")
            else:
                raise ValueError(f"bm25.pkl 구조 확인 필요. 키: {list(data.keys())}")
        else:
            raise ValueError("bm25.pkl에 corpus 정보가 없습니다.")

        print("[HybridRetriever] BM25 loaded OK")

    def _get_kiwi_tokenizer(self):
        """Kiwi 토크나이저 강제 사용 (BM25 인덱스와 일치)"""
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            
            def kiwi_tokenizer(text):
                tokens = [tok.form for tok in kiwi.tokenize(text)]
                meaningful_tokens = []
                stopwords = {'의', '에', '으로', '는', '은', '이', '가', '를', '을', '와', '과', '에서', '부터', '까지', 'ㄴ다'}
                
                for token in tokens:
                    if len(token) >= 2 and token not in stopwords and token.isalpha():
                        meaningful_tokens.append(token)
                
                return meaningful_tokens
            
            print("[INFO] Kiwi 토크나이저 초기화 성공")
            return kiwi_tokenizer
            
        except Exception as e:
            print(f"[ERROR] Kiwi 토크나이저 초기화 실패: {e}")
            import re
            def fallback_tokenizer(text):
                return re.findall(r'[가-힣]{2,}', text)
            return fallback_tokenizer

    def _preprocess_query(self, query):
        """개선된 쿼리 전처리 - 핵심 키워드 보존"""
        import re
        
        # 원본 쿼리 보존
        original_query = query
        
        # 1. 법령명 패턴 제거 (너무 긴 것만)
        query = re.sub(r'「[^」]{20,}」', '', query)
        
        # 2. 괄호 안 내용 제거 (법령 정보만)
        query = re.sub(r'\([^)]*호[^)]*\)', '', query)  # 호수 정보만
        query = re.sub(r'\([^)]*령[^)]*\)', '', query)  # 령 정보만
        
        # 3. "에 관한 법률/시행령" 등은 보존 (중요한 식별자)
        # 기존 코드 제거
        
        # 4. 조문 번호와 핵심 키워드 추출
        article_match = re.search(r'제\d+조', query)
        
        # 핵심 키워드 추출 (3글자 이상 의미있는 단어)
        keywords = re.findall(r'[가-힣]{3,}', query)
        
        # 불필요한 키워드 제거
        exclude_words = {'관한', '따른', '따라', '경우', '사항', '내용', '무엇', '어떤', '얼마'}
        keywords = [k for k in keywords if k not in exclude_words]
        
        if article_match and keywords:
            # 조문 + 상위 5개 키워드
            result = article_match.group() + ' ' + ' '.join(keywords[:5])
        elif keywords:
            # 키워드만 (상위 5개)
            result = ' '.join(keywords[:5])
        else:
            # 폴백: 원본 쿼리 사용
            result = original_query
        
        # 다중 공백 정리
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 간소화된 디버그 출력
        if DEBUG_HYBRID:
            print(f"[Query] '{original_query[:30]}...' -> '{result}'")
        
        return result

    def bm25_search(self, query, top_k=10):
        """Kiwi 토크나이저 사용한 BM25 검색"""
        
        query_clean = self._preprocess_query(query)
        tokenized_q = self.tokenizer(query_clean)
        
        if not tokenized_q:
            if DEBUG_HYBRID:
                print(f"[BM25] No tokens for: '{query}'")
            return []
        
        if DEBUG_HYBRID:
            print(f"[BM25] Tokens: {tokenized_q}")
        
        try:
            import numpy as np
            scores = self.bm25.get_scores(tokenized_q)
            
            max_score = float(scores.max()) if hasattr(scores, 'max') else max(scores)
            
            if max_score == 0:
                if DEBUG_HYBRID:
                    print(f"[BM25] Retry with individual tokens")
                best_results = []
                best_score = 0
                
                for token in tokenized_q:
                    single_scores = self.bm25.get_scores([token])
                    single_max = float(single_scores.max()) if hasattr(single_scores, 'max') else max(single_scores)
                    
                    if single_max > best_score:
                        best_score = single_max
                        ranked_idx = np.argsort(single_scores)[::-1] if hasattr(single_scores, 'argsort') else \
                                     sorted(range(len(single_scores)), key=lambda i: single_scores[i], reverse=True)
                        
                        for idx in ranked_idx[:top_k]:
                            score = float(single_scores[idx])
                            if score > 0:
                                best_results.append({
                                    "score": score,
                                    "text": self.corpus[idx]["text"],
                                    "filename": self.corpus[idx].get("filename", ""),
                                    "chunk_index": self.corpus[idx].get("chunk_index", -1)
                                })
                                if len(best_results) >= top_k:
                                    break
                        
                        if best_results:
                            break
                
                if best_results:
                    if DEBUG_HYBRID:
                        print(f"[BM25] Found: {len(best_results)} (max: {best_score:.1f})")
                    return best_results[:top_k]
                else:
                    if DEBUG_HYBRID:
                        print(f"[BM25] No results for: '{query}'")
                    return []
            
            # 정상 검색 결과 처리
            ranked_idx = np.argsort(scores)[::-1] if hasattr(scores, 'argsort') else \
                         sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            results = []
            for idx in ranked_idx[:top_k]:
                score = float(scores[idx])
                if score > 0:
                    results.append({
                        "score": score,
                        "text": self.corpus[idx]["text"],
                        "filename": self.corpus[idx].get("filename", ""),
                        "chunk_index": self.corpus[idx].get("chunk_index", -1)
                    })
            
            if DEBUG_HYBRID:
                print(f"[BM25] Found: {len(results)} (max: {max_score:.1f})")
            return results
            
        except Exception as e:
            if DEBUG_HYBRID:
                print(f"[BM25] Error: {e}")
            return []

    def search(self, query, k_final=7, namespace=""):
        """개선된 하이브리드 검색"""
        
        if DEBUG_HYBRID:
            print(f"[Search] '{query[:30]}...'")
        
        # BM25 검색
        bm25_results = self.bm25_search(query, top_k=self.cfg.top_k_bm25)
        
        # Vector 검색
        vec_results = []
        try:
            vec = self.embedder_fn(query)
            if vec:
                vec_resp = self.index.query(
                    vector=vec,
                    top_k=self.cfg.top_k_vec,
                    namespace=namespace,
                    include_metadata=True
                )
                
                for m in vec_resp.matches:
                    vec_results.append({
                        "score": m.score,
                        "text": (m.metadata or {}).get("text", ""),
                        "filename": (m.metadata or {}).get("filename", ""),
                        "chunk_index": (m.metadata or {}).get("chunk_index", -1)
                    })
                
                if DEBUG_HYBRID:
                    print(f"[Vector] Found: {len(vec_results)}")
        except Exception as e:
            if DEBUG_HYBRID:
                print(f"[Vector] Error: {e}")
        
        if not bm25_results and not vec_results:
            if DEBUG_HYBRID:
                print("[Search] No results")
            return []
        
        # 점수 정규화
        def safe_normalize(scores):
            if not scores:
                return []
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        bm25_normalized = safe_normalize([r["score"] for r in bm25_results])
        vec_normalized = safe_normalize([r["score"] for r in vec_results])
        
        # 결과 병합
        chunks = {}
        
        for i, r in enumerate(bm25_results):
            key = (r.get("filename", ""), r.get("chunk_index", -1))
            chunks[key] = {
                "text": r.get("text", ""),
                "filename": r.get("filename", ""),
                "chunk_index": r.get("chunk_index", -1),
                "bm25_score_raw": r["score"],
                "bm25_score_norm": bm25_normalized[i] if bm25_normalized else 0.0,
                "vec_score_raw": 0.0,
                "vec_score_norm": 0.0
            }
        
        for i, r in enumerate(vec_results):
            key = (r.get("filename", ""), r.get("chunk_index", -1))
            if key not in chunks:
                chunks[key] = {
                    "text": r.get("text", ""),
                    "filename": r.get("filename", ""),
                    "chunk_index": r.get("chunk_index", -1),
                    "bm25_score_raw": 0.0,
                    "bm25_score_norm": 0.0,
                    "vec_score_raw": 0.0,
                    "vec_score_norm": 0.0
                }
            
            chunks[key]["vec_score_raw"] = r["score"]
            chunks[key]["vec_score_norm"] = vec_normalized[i] if vec_normalized else 0.0
        
        # 최종 점수 계산
        results = []
        for data in chunks.values():
            final_score = (
                self.cfg.weight_bm25 * data["bm25_score_norm"] +
                self.cfg.weight_vec * data["vec_score_norm"]
            )
            
            results.append({
                "text": data["text"],
                "filename": data["filename"],
                "chunk_index": data["chunk_index"],
                "score": final_score,
                "final_score": final_score,
                "bm25_score_raw": data["bm25_score_raw"],
                "bm25_score_norm": data["bm25_score_norm"],
                "vec_score_raw": data["vec_score_raw"],
                "vec_score_norm": data["vec_score_norm"]
            })
        
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        final_results = results[:k_final]
        if final_results and DEBUG_HYBRID:
            print(f"[Result] {len(final_results)} docs (top: {final_results[0]['final_score']:.3f})")
        
        return final_results