# -*- coding: utf-8 -*-
"""
rag/retriever.py
- 단일 진입점: retrieve / retrieve_batch
- HybridRetriever만 사용 (HybridConfig, PineconeWrapper 의존 제거)
"""
from __future__ import annotations

# --- optional .env backup load (lightweight) ---
try:
    if not (os.getenv("UPSTAGE_API_KEY") or os.getenv("PINECONE_API_KEY")):
        from dotenv import load_dotenv, find_dotenv
        p = find_dotenv(usecwd=True)
        if p:
            load_dotenv(p, override=False)
except Exception:
    pass
# -----------------------------------------------

from typing import List, Dict, Any, Optional
import os
from .hybrid_retriever import HybridRetriever

# 환경변수 기본값 (hybrid_retriever 내부 기본과 동일/호환)
_TOP_K_BM25 = int(os.getenv("TOP_K_BM25") or 12)
_TOP_K_VEC  = int(os.getenv("TOP_K_VEC")  or 12)
_FINAL_TOP_K = int(os.getenv("FINAL_TOP_K") or 7)
_WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25") or 0.55)
_WEIGHT_VEC  = float(os.getenv("WEIGHT_VEC") or 0.45)
_DEBUG = (os.getenv("RETRIEVER_DEBUG") or "false").lower() in ("1","true","yes","y")

# 전역 리트리버 싱글톤
_retr = HybridRetriever(
    top_k_bm25=_TOP_K_BM25,
    top_k_vec=_TOP_K_VEC,
    k_final=_FINAL_TOP_K,
    w_bm25=_WEIGHT_BM25,
    w_vec=_WEIGHT_VEC,
    debug=_DEBUG,
)

def retrieve(query: str, top_k: Optional[int] = None, debug: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    단일 질의 검색
    """
    if query is None or not str(query).strip():
        return []
    kf = top_k if top_k is not None else _FINAL_TOP_K
    return _retr.search(query, k_final=kf)

def retrieve_batch(queries: List[str], top_k: Optional[int] = None, debug: Optional[bool] = None, workers: int = 8) -> List[List[Dict[str, Any]]]:
    """
    다중 질의 배치 검색
    """
    if not queries:
        return []
    kf = top_k if top_k is not None else _FINAL_TOP_K
    return _retr.search_batch(queries, k_final=kf, workers=workers)
