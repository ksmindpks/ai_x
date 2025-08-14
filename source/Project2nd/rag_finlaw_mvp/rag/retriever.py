# rag/retriever.py
from typing import List, Dict
from pinecone import Pinecone
from config import (
    PINECONE_API_KEY, PINECONE_NAMESPACE, PINECONE_INDEX_NAME,
    TOP_K_BM25, TOP_K_VEC, WEIGHT_BM25, WEIGHT_VEC, DEBUG_HYBRID
)
from .embedder import embed_texts
from .hybrid_retriever import HybridRetriever, HybridConfig, PineconeWrapper
from rag.utils import preprocess_query  # 전처리 함수 (utils나 별도 모듈에 구현)

# Pinecone 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
pc_wrapper = PineconeWrapper(index)

# HybridRetriever 설정
def embedder_fn(text: str):
    vectors = embed_texts([text])
    return vectors[0] if vectors else []

cfg = HybridConfig(
    bm25_index_dir="./bm25_pkg/out/bm25_index",
    pinecone_index=pc_wrapper,
    top_k_bm25=TOP_K_BM25,
    top_k_vec=TOP_K_VEC,
    weight_bm25=WEIGHT_BM25,
    weight_vec=WEIGHT_VEC,
    tokenizer="kiwi"
)
retriever = HybridRetriever(cfg, embedder_fn)

def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """단일 질의 검색 - final_score 사용"""
    q = preprocess_query(query)
    results = retriever.search(q, k_final=top_k, namespace=PINECONE_NAMESPACE)
    
    # 스키마 확인 및 score 필드 보장
    for r in results:
        if "final_score" in r:
            r["score"] = r["final_score"]  # 명시적 매핑
    
    return results

def retrieve_batch(queries: List[str], top_k: int = 7, debug: bool = DEBUG_HYBRID) -> List[List[Dict]]:
    """배치 검색 - 간소화된 디버그"""
    results_per_query = []
    
    for qi, query in enumerate(queries):
        q = preprocess_query(query)
        results = retriever.search(q, k_final=top_k, namespace=PINECONE_NAMESPACE)
        
        # 스키마 확인
        for r in results:
            if "final_score" in r:
                r["score"] = r["final_score"]
        
        # 간소화된 디버그 출력 (10개마다만)
        if debug and qi % 10 == 0 and results:
            top = results[0]
            final_score = top.get('final_score', 0)
            if DEBUG_HYBRID:
                print(f"[Batch {qi:3d}] Final: {final_score:.3f}")
        
        results_per_query.append(results)
    
    return results_per_query