# -*- coding: utf-8 -*-
"""
rag/retriever.py (boosted v2, 2025-08-18)
- 정의/조문특정 질문 시 상위 컨텍스트 보정 유지
- 첫 문서 고신뢰 시 top_k 축소로 노이즈 억제
- 배치 검색 병렬 안정화
- 반환 컨텍스트 text 필드 누락 시 빈문자열로 보정
"""

from __future__ import annotations
from typing import List, Dict
import os, re, threading
import concurrent.futures as cf

_PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
_PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX", "codedoc-law-index"))
_PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
_BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "./bm25_pkg/out/bm25_index")
_TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "7"))
_TOP_K_VEC  = int(os.getenv("TOP_K_VEC", "15"))
_WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.5"))
_WEIGHT_VEC  = float(os.getenv("WEIGHT_VEC", "0.5"))
_DEBUG = os.getenv("DEBUG_HYBRID", "false").lower() in ("1","true","yes")

from .hybrid_retriever import HybridRetriever, HybridConfig, PineconeWrapper

_index = None
if _PINECONE_API_KEY:
    try:
        from pinecone import Pinecone
        _pc = Pinecone(api_key=_PINECONE_API_KEY)
        _index = _pc.Index(_PINECONE_INDEX_NAME)
    except Exception as e:
        if _DEBUG:
            print(f"[Retriever] Pinecone 초기화 실패: {e}")
        _index = None

try:
    from .utils import preprocess_query
except Exception:
    def preprocess_query(q: str) -> str:
        q = (q or "")
        q = re.sub(r'「[^」]{60,}」', "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

_HYBRID: HybridRetriever | None = None
_INIT_LOCK = threading.Lock()

def _get_hybrid() -> HybridRetriever:
    global _HYBRID
    if _HYBRID is None:
        with _INIT_LOCK:
            if _HYBRID is None:
                cfg = HybridConfig(
                    top_k_bm25=_TOP_K_BM25, top_k_vec=_TOP_K_VEC,
                    weight_bm25=_WEIGHT_BM25, weight_vec=_WEIGHT_VEC,
                    namespace=_PINECONE_NAMESPACE,
                    debug=_DEBUG,
                    bm25_pickle_path=os.path.join(_BM25_INDEX_DIR, "bm25.pkl"),
                )
                pcw = PineconeWrapper(_index) if _index is not None else None
                _HYBRID = HybridRetriever(cfg, pinecone_wrapper=pcw)
    return _HYBRID

_ANCHOR_LAW_RE = re.compile(r'「([^」]{4,80})」')
_ANCHOR_ART_RE = re.compile(r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?')

def _extract_law_and_article(q: str):
    law = ""
    m = _ANCHOR_LAW_RE.search(q or "")
    if m: law = m.group(1)
    arts = [f"제{a}조" + (f"제{b}항" if b else "") + (f"제{c}호" if c else "")
            for a,b,c in _ANCHOR_ART_RE.findall(q or "")]
    return law, arts

def _apply_anchor_boost(query: str, ctxs: List[Dict], law_boost=0.15, art_boost=0.20) -> List[Dict]:
    if not ctxs: return ctxs
    law, arts = _extract_law_and_article(query)
    for c in ctxs:
        t = c.get("text", "") or ""
        base = c.get("final_score", c.get("score", 0.0))
        b = 0.0
        if law and (law in t): b += law_boost
        for a in arts:
            if a and (a in t): b += art_boost
        c["final_score"] = base + b
    ctxs.sort(key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)
    return ctxs

def _boost_definition_when_present(query: str, ctxs: List[Dict]) -> List[Dict]:
    if not ctxs: return ctxs
    if re.search(r'(정의|무엇|라\s*함|을\s*말한다|이라\s*한다)', query or ''):
        for i, c in enumerate(ctxs[:3]):
            base = c.get("final_score", c.get("score", 0.0))
            c["final_score"] = base + 0.10 - 0.02*i
        ctxs.sort(key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)
    return ctxs

def _adapt_top_k(first_score: float, top_k: int) -> int:
    if first_score <= 0.0: return top_k
    if first_score >= 0.92: return min(top_k, 3)
    if first_score >= 0.80: return min(top_k, 5)
    return top_k

def retrieve(query: str, top_k: int = 7, debug: bool = False) -> List[Dict]:
    q0 = (query or "").strip()
    if not q0: return []
    retr = _get_hybrid()
    q = preprocess_query(q0)
    ctxs = retr.search(q, top_k_bm25=retr.cfg.top_k_bm25, top_k_vec=retr.cfg.top_k_vec, debug=debug)
    ctxs = _apply_anchor_boost(q0, ctxs)
    ctxs = _boost_definition_when_present(q0, ctxs)

    # text 필드 보정
    for c in ctxs:
        if "text" not in c or c["text"] is None:
            c["text"] = ""

    first = ctxs[0].get("final_score", ctxs[0].get("score", 0.0)) if ctxs else 0.0
    tk = _adapt_top_k(first, top_k)
    return ctxs[:tk]

def retrieve_batch(queries: List[str], top_k: int = 7, debug: bool = False) -> List[List[Dict]]:
    if not queries: return []
    out: List[List[Dict]] = [[] for _ in range(len(queries))]
    workers = min(max(1, os.cpu_count() or 8), 16, len(queries))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut2i = {ex.submit(retrieve, q, top_k, False): i for i, q in enumerate(queries)}
        for fut in cf.as_completed(fut2i):
            i = fut2i[fut]
            try:
                out[i] = fut.result() or []
            except Exception as e:
                if _DEBUG:
                    print(f"[Retriever] batch item {i} error: {e}")
                out[i] = []
    return out
