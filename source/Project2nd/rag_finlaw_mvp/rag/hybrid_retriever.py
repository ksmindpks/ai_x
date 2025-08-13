# -*- coding: utf-8 -*-
"""
HybridRetriever
BM25(키워드 검색) + Pinecone(벡터 검색) 결합
- BM25 점수와 Vector 점수를 정규화 후 가중합
- 최종 Top-K 결과 반환
"""

import os
import pickle
from typing import List, Dict, Callable
from rank_bm25 import BM25Okapi


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
        self.tokenizer = self._get_tokenizer(cfg.tokenizer)

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

        # data에 bm25와 corpus가 같이 저장돼 있어야 함
        if isinstance(data, dict):
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]
        else:
            raise ValueError("bm25.pkl 구조가 잘못되었습니다. dict 형태여야 하며 'bm25', 'corpus' 키가 필요합니다.")

        print("[HybridRetriever] BM25 loaded OK")

    def _get_tokenizer(self, tokenizer_name):
        """토크나이저 초기화"""
        if tokenizer_name == "kiwi":
            try:
                from kiwipiepy import Kiwi
                kiwi = Kiwi()
                return lambda text: [tok.form for tok in kiwi.tokenize(text)]
            except Exception:
                print("[WARN] Kiwi 로드 실패, 기본 토크나이저 사용")
        
        # 기본 토크나이저
        return lambda text: text.split()

    def bm25_search(self, query, top_k=10):
        """BM25 검색"""
        tokenized_q = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_q)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in ranked_idx[:top_k]:
            results.append({
                "score": float(scores[idx]),
                "text": self.corpus[idx]["text"],
                "filename": self.corpus[idx].get("filename", ""),
                "chunk_index": self.corpus[idx].get("chunk_index", -1)
            })
        return results

    def search(self, query, k_final=7, namespace=""):
        """하이브리드 검색 - 점수 정규화 및 스키마 표준화"""
        
        # ----- BM25 검색 -----
        bm25_results = self.bm25_search(query, top_k=self.cfg.top_k_bm25)
        
        # ----- Vector 검색 -----
        vec_results = []
        try:
            vec = self.embedder_fn(query)
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
        except Exception as e:
            print(f"[Vec ERROR] {str(e)[:120]}")
        
        # ----- Min-Max 정규화 (0~1) -----
        def minmax_normalize(scores):
            if not scores:
                return []
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        bm25_raw_scores = [r["score"] for r in bm25_results]
        vec_raw_scores = [r["score"] for r in vec_results]
        
        bm25_normalized = minmax_normalize(bm25_raw_scores)
        vec_normalized = minmax_normalize(vec_raw_scores)
        
        # ----- (filename, chunk_index) 기준 병합 -----
        chunks = {}  # (filename, chunk_index) -> 통합 데이터
        
        # BM25 결과 저장
        for i, r in enumerate(bm25_results):
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
            
            chunks[key]["bm25_score_raw"] = float(bm25_raw_scores[i])
            chunks[key]["bm25_score_norm"] = float(bm25_normalized[i])
        
        # Vector 결과 저장
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
            
            chunks[key]["vec_score_raw"] = float(vec_raw_scores[i])
            chunks[key]["vec_score_norm"] = float(vec_normalized[i])
        
        # ----- 최종 점수 계산 (0~1 범위) -----
        results = []
        for key, data in chunks.items():
            # 가중합 (0~1 범위 보장)
            final_score = (
                self.cfg.weight_bm25 * data["bm25_score_norm"] +
                self.cfg.weight_vec * data["vec_score_norm"]
            )
            
            # 표준 스키마로 저장
            results.append({
                # 필수 필드
                "text": data["text"],
                "filename": data["filename"],
                "chunk_index": data["chunk_index"],
                "score": final_score,  # retriever.py 호환
                
                # 디버그/분석용 필드
                "final_score": final_score,
                "bm25_score_raw": data["bm25_score_raw"],
                "bm25_score_norm": data["bm25_score_norm"],
                "vec_score_raw": data["vec_score_raw"],
                "vec_score_norm": data["vec_score_norm"]
            })
        
        # final_score 기준 정렬
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return results[:k_final]