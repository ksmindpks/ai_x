# rag/retriever.py
from typing import List, Dict
from pinecone import Pinecone
from config import (
    PINECONE_API_KEY, PINECONE_HOST, PINECONE_NAMESPACE,
    BM25_INDEX_DIR, BM25_TOKENIZER,
    TOP_K_BM25, TOP_K_VEC, WEIGHT_BM25, WEIGHT_VEC
)
from .embedder import embed_texts
from .hybrid_retriever import HybridRetriever, HybridConfig, PineconeWrapper

# Pinecone 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)
pc_wrapper = PineconeWrapper(index)

# HybridRetriever 설정
def embedder_fn(text: str):
    vectors = embed_texts([text])
    return vectors[0] if vectors else []

cfg = HybridConfig(
    bm25_index_dir=BM25_INDEX_DIR,
    pinecone_index=pc_wrapper,
    top_k_bm25=TOP_K_BM25,
    top_k_vec=TOP_K_VEC,
    weight_bm25=WEIGHT_BM25,
    weight_vec=WEIGHT_VEC,
    tokenizer=BM25_TOKENIZER
)
retriever = HybridRetriever(cfg, embedder_fn)

def retrieve(query: str, top_k: int = 5):
    results = retriever.search(query, k_final=top_k, namespace=PINECONE_NAMESPACE)
    out = []
    for r in results:
        meta = r.get("meta", {}) or {}
        out.append({
            "score": r.get("final_score", 0.0),
            "bm25_score_raw":  r.get("bm25_score_raw"),
            "vec_score_raw":   r.get("vec_score_raw"),
            "bm25_score_norm": r.get("bm25_score_norm"),
            "vec_score_norm":  r.get("vec_score_norm"),
            "text": r.get("text",""),
            "filename": meta.get("filename",""),
            "chunk_index": meta.get("chunk_index", 0)
        })
    return out

def retrieve_batch(queries: List[str], top_k: int = 5) -> List[List[Dict]]:
    all_results = []
    for q in queries:
        all_results.append(retrieve(q, top_k=top_k))
    return all_results