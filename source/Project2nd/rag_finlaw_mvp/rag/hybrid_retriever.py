# -*- coding: utf-8 -*-
"""
HybridRetriever
BM25(키워드 검색) + Pinecone(벡터 검색) 결합
- BM25 점수와 Vector 점수를 정규화 후 가중합
- 최종 Top-K 결과 반환
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Callable
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
    def __init__(self, config: HybridConfig, embedder_fn: Callable[[str], List[float]]):
        self.cfg = config
        self.embedder_fn = embedder_fn
        self._load_bm25()

    def _load_bm25(self):
        with open(os.path.join(self.cfg.bm25_index_dir, "bm25.pkl"), "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.ids = data["ids"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.meta = pd.read_parquet(os.path.join(self.cfg.bm25_index_dir, "meta.parquet"))

        if self.cfg.tokenizer == "kiwi":
            try:
                from kiwipiepy import Kiwi
                kiwi = Kiwi()
                self.tokenizer = lambda text: [t.form for t in kiwi.tokenize(text)]
            except Exception:
                print("[WARN] kiwipiepy 미설치 → simple tokenizer 사용")
                self.tokenizer = lambda text: text.lower().split()
        else:
            self.tokenizer = lambda text: text.lower().split()

    def search(self, query: str, k_final: int = 10, namespace: Optional[str] = None, filter: Optional[dict] = None):
        # BM25 검색
        q_tokens = self.tokenizer(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:self.cfg.top_k_bm25]

        bm25_results = []
        for idx in bm25_top_idx:
            meta_row = self.meta.iloc[idx].to_dict()
            bm25_results.append({
                "_id": self.ids[idx],
                "text": meta_row.get("text", ""),
                "bm25_score": float(bm25_scores[idx]),
                "meta": meta_row
            })

        # Pinecone 검색
        emb = self.embedder_fn(query)
        pinecone_resp = self.cfg.pinecone_index.search(emb, top_k=self.cfg.top_k_vec, namespace=namespace, filter=filter)

        vec_results = []
        for match in pinecone_resp.matches:
            vec_results.append({
                "_id": match.id,
                "text": match.metadata.get("text", ""),
                "vec_score": float(match.score),
                "meta": match.metadata
            })

        # 결과 병합
        merged = {}
        for r in bm25_results:
            merged[r["_id"]] = {
                "_id": r["_id"], "text": r["text"], 
                "bm25_score": r["bm25_score"], "vec_score": 0.0, 
                "meta": r["meta"]
            }
        for r in vec_results:
            if r["_id"] in merged:
                merged[r["_id"]]["vec_score"] = r["vec_score"]
            else:
                merged[r["_id"]] = {
                    "_id": r["_id"], "text": r["text"], 
                    "bm25_score": 0.0, "vec_score": r["vec_score"], 
                    "meta": r["meta"]
                }

        # 점수 정규화
        bm25_vals = [v["bm25_score"] for v in merged.values()]
        vec_vals  = [v["vec_score"]  for v in merged.values()]

        def norm(vals):
            min_v, max_v = min(vals), max(vals)
            return [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]

        bm25_norm = norm(bm25_vals)
        vec_norm  = norm(vec_vals)

        # 가중합 + 필드 부착
        for i, key in enumerate(list(merged.keys())):
            merged[key]["bm25_score_raw"] = bm25_vals[i]
            merged[key]["vec_score_raw"]  = vec_vals[i]
            merged[key]["bm25_score_norm"] = bm25_norm[i]
            merged[key]["vec_score_norm"]  = vec_norm[i]
            merged[key]["final_score"] = (
                self.cfg.weight_bm25 * bm25_norm[i] +
                self.cfg.weight_vec  * vec_norm[i]
            )

        # 최종 정렬 후 반환
        final_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)[:k_final]
        return final_results
