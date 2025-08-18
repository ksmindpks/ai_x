# -*- coding: utf-8 -*-
"""
rag/hybrid_retriever.py
- BM25 + Vector 하이브리드
- 개선: 쿼리 전처리(조문/기관 토큰 보존), 점수 정규화 안정화, 제목 부스팅, 가중치 조정, 디버그 지표
"""
from __future__ import annotations
# --- optional .env backup load (lightweight) ---
try:
    import os
    if not (os.getenv("UPSTAGE_API_KEY") or os.getenv("PINECONE_API_KEY")):
        from dotenv import load_dotenv, find_dotenv
        p = find_dotenv(usecwd=True)
        if p:
            load_dotenv(p, override=False)
except Exception:
    pass
# -----------------------------------------------

import os, re, json, math, pickle
from typing import List, Dict, Tuple, Any, Optional
import concurrent.futures

# 환경/가중치 기본값 (config가 있으면 환경변수로 덮어쓰기 가능)
TOP_K_BM25 = int(os.getenv("TOP_K_BM25") or 12)
TOP_K_VEC  = int(os.getenv("TOP_K_VEC")  or 12)
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25") or 0.55)
WEIGHT_VEC  = float(os.getenv("WEIGHT_VEC")  or 0.45)
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K") or 7)
DEBUG = (os.getenv("RETRIEVER_DEBUG") or "false").lower() in ("1","true","yes","y")

# BM25 인덱스 경로
BM25_PKL = os.getenv("BM25_PICKLE") or os.path.join(".", "bm25_pkg", "out", "bm25_index", "bm25.pkl")

# Pinecone (벡터) 선택 사용
USE_PINECONE = (os.getenv("USE_PINECONE") or "true").lower() in ("1","true","yes","y")
PC_INDEX_NAME = os.getenv("PINECONE_INDEX") or "codedoc-law-upstage"
PC_NAMESPACE  = os.getenv("PINECONE_NAMESPACE") or ""
PC_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PC_REGION     = os.getenv("PINECONE_REGION", "us-east-1")

# 임베딩(쿼리용) - Upstage 사용 예시 (langchain_upstage)
_USE_UPSTAGE = (os.getenv("EMBED_PROVIDER") or "upstage").lower() == "upstage"
_UP_MODEL = os.getenv("UPSTAGE_EMBEDDING_MODEL") or "solar-embedding-1-large"

# ----------------------------- 토크나이저/전처리 -----------------------------

def _preprocess_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r'[“”\"\'\(\)\[\]<>·•※]', ' ', q)
    q = re.sub(r'\s+', ' ', q)

    # keep: 제n조(의n), 제n항, 기관명 접미
    keep = re.findall(r'(제\d+조(?:의\d+)?|제\d+항|[가-힣A-Za-z]{2,20}(?:위원회|부|청|원))', q)

    # 말미 조사/어미 제거(가볍게)
    q2 = re.sub(r'(을|를|은|는|이|가|의|에|에서|으로|에게)$', '', q)
    words = [w for w in re.split(r'\s+', q2) if len(w) >= 2]

    merged = []
    seen = set()
    for t in keep + words:
        t = t.strip()
        if t and t not in seen:
            seen.add(t); merged.append(t)
    return ' '.join(merged[:20])


def _safe_minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

# ----------------------------- BM25 로딩/검색 -----------------------------

class _BM25Store:
    def __init__(self, path: str):
        self.ok = False
        self.corpus: List[Dict[str, Any]] = []
        self.model = None
        self.tokenizer = None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # 예상 포맷: dict { bm25, corpus, tokenizer, ... }
            self.model = data.get("bm25")
            self.corpus = data.get("corpus") or []
            self.tokenizer = data.get("tokenizer")
            self.ok = (self.model is not None) and bool(self.corpus)
            if DEBUG:
                print(f"[BM25] loaded docs={len(self.corpus)} ok={self.ok}")
        except Exception as e:
            if DEBUG:
                print(f"[BM25] load failed: {e}")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.ok:
            return []
        q = _preprocess_query(query)
        # 모델이 토큰화까지 내장일 수 있지만, 간단화: model.get_scores(q) 가정 or wrapper 필요
        # 여기서는 저장된 model에 'get_top_n' 스타일이 있다고 가정:
        try:
            # 가정: self.model.get_top_n(query, documents, n=k)
            docs = self.model.get_top_n(q, self.corpus, n=k)
            out = []
            for d in docs:
                # d: corpus 항목 (text, title, filename, chunk_index, score?)
                scr = d.get("score", 0.0)
                out.append({
                    "score": float(scr),
                    "text": d.get("text", ""),
                    "title": d.get("title", ""),
                    "filename": d.get("filename", ""),
                    "chunk_index": d.get("chunk_index", -1),
                })
            return out
        except Exception:
            # 백업: corpus 전체에서 매우 단순 점수(키워드 매칭 수)
            toks = set(q.split())
            scored = []
            for d in self.corpus:
                t = (d.get("text","") or "")
                hit = sum(1 for tk in toks if tk and tk in t)
                if hit > 0:
                    scored.append((hit, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            out = []
            for hit, d in scored[:k]:
                out.append({
                    "score": float(hit),
                    "text": d.get("text",""),
                    "title": d.get("title",""),
                    "filename": d.get("filename",""),
                    "chunk_index": d.get("chunk_index", -1),
                })
            return out


# ----------------------------- 벡터 검색(Pinecone) -----------------------------

_pc = None
_pc_index = None
_embedder = None

def _ensure_pinecone():
    global _pc, _pc_index
    if not USE_PINECONE:
        return
    if _pc is None:
        try:
            from pinecone import Pinecone
            _pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            _pc_index = _pc.Index(PC_INDEX_NAME)
        except Exception as e:
            if DEBUG:
                print(f"[Vec] pinecone init failed: {e}")
            _pc = None
            _pc_index = None

def _ensure_embedder():
    global _embedder
    if _embedder is None:
        try:
            if _USE_UPSTAGE:
                from langchain_upstage import UpstageEmbeddings
                _embedder = UpstageEmbeddings(model=_UP_MODEL)
            else:
                # 필요시 다른 임베딩 프로바이더 연결
                from langchain_upstage import UpstageEmbeddings
                _embedder = UpstageEmbeddings(model=_UP_MODEL)
        except Exception as e:
            if DEBUG:
                print(f"[Vec] embedder init failed: {e}")
            _embedder = None

def _vec_search(query: str, k: int) -> List[Dict[str, Any]]:
    _ensure_pinecone()
    _ensure_embedder()
    if not (_pc_index and _embedder):
        return []
    try:
        qv = _embedder.embed_query(_preprocess_query(query))
        res = _pc_index.query(vector=qv, top_k=k, include_metadata=True, namespace=PC_NAMESPACE)
        out = []
        for m in res.matches or []:
            md = m.metadata or {}
            out.append({
                "score": float(m.score or 0.0),
                "text": md.get("text_preview","") or md.get("text","") or "",
                "title": md.get("title","") or "",
                "filename": md.get("source_path","") or "",
                "chunk_index": int(md.get("chunk_id",-1)),
            })
        return out
    except Exception as e:
        if DEBUG:
            print(f"[Vec] query failed: {e}")
        return []


# ----------------------------- 하이브리드 -----------------------------

class HybridRetriever:
    def __init__(self,
                 top_k_bm25: int = TOP_K_BM25,
                 top_k_vec: int  = TOP_K_VEC,
                 k_final: int    = FINAL_TOP_K,
                 w_bm25: float   = WEIGHT_BM25,
                 w_vec: float    = WEIGHT_VEC,
                 debug: bool     = DEBUG):
        self.k_bm25 = top_k_bm25
        self.k_vec  = top_k_vec
        self.k_final = k_final
        self.w_bm25 = w_bm25
        self.w_vec  = w_vec
        self.debug  = debug

        self.bm25 = _BM25Store(BM25_PKL)

        if self.debug:
            print(f"[HybridRetriever] init: bm25_ok={self.bm25.ok} k_bm25={self.k_bm25} "
                  f"k_vec={self.k_vec} w=({self.w_bm25},{self.w_vec})")

    def search(self, query: str, k_final: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query:
            return []

        kb = self.k_bm25
        kv = self.k_vec
        kf = k_final if k_final is not None else self.k_final

        bm25_res = self.bm25.search(query, kb) if self.bm25.ok else []
        vec_res  = _vec_search(query, kv)

        # 정규화
        bm25_norm = _safe_minmax([r["score"] for r in bm25_res]) if bm25_res else []
        vec_norm  = _safe_minmax([r["score"] for r in vec_res]) if vec_res else []

        chunks: Dict[Tuple[str,int], Dict[str, Any]] = {}

        # BM25 반영
        for i, r in enumerate(bm25_res):
            key = (r.get("filename",""), int(r.get("chunk_index",-1)))
            base = chunks.get(key) or {
                "text": r.get("text",""),
                "title": r.get("title",""),
                "filename": r.get("filename",""),
                "chunk_index": int(r.get("chunk_index",-1)),
                "bm25_score_raw": 0.0,
                "vec_score_raw": 0.0,
                "score": 0.0,
            }
            base["bm25_score_raw"] = float(r["score"])
            base["score"] += self.w_bm25 * (bm25_norm[i] if i < len(bm25_norm) else 0.0)
            chunks[key] = base

        # Vec 반영
        for i, r in enumerate(vec_res):
            key = (r.get("filename",""), int(r.get("chunk_index",-1)))
            base = chunks.get(key) or {
                "text": r.get("text",""),
                "title": r.get("title",""),
                "filename": r.get("filename",""),
                "chunk_index": int(r.get("chunk_index",-1)),
                "bm25_score_raw": 0.0,
                "vec_score_raw": 0.0,
                "score": 0.0,
            }
            base["vec_score_raw"] = float(r["score"])
            base["score"] += self.w_vec * (vec_norm[i] if i < len(vec_norm) else 0.0)
            chunks[key] = base

        # 제목 부스팅(간단): 쿼리 토큰 중 일부가 title에 있으면 +0.05
        qtok = set(_preprocess_query(query).split())
        for k, v in chunks.items():
            title = (v.get("title","") or "")
            hit = any(t in title for t in qtok if t)
            if hit:
                v["score"] += 0.05

        merged = list(chunks.values())
        merged.sort(key=lambda x: x["score"], reverse=True)
        out = merged[:kf]

        if self.debug:
            bm25_n = sum(1 for v in chunks.values() if v["bm25_score_raw"] > 0)
            vec_n  = sum(1 for v in chunks.values() if v["vec_score_raw"]  > 0)
            print(f"[Mix] bm25_hits={bm25_n}, vec_hits={vec_n}, "
                  f"w_bm25={self.w_bm25:.2f}, w_vec={self.w_vec:.2f}, k_final={kf}, out={len(out)}")
        return out

    # 일괄 질의(멀티스레드)
    def search_batch(self, queries: List[str], k_final: Optional[int] = None, workers: int = 8) -> List[List[Dict[str, Any]]]:
        if not queries:
            return []
        out: List[List[Dict[str, Any]]] = [[] for _ in range(len(queries))]
        kf = k_final if k_final is not None else self.k_final
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                fut2i = {ex.submit(self.search, q, kf): i for i, q in enumerate(queries)}
                for fut in concurrent.futures.as_completed(fut2i):
                    i = fut2i[fut]
                    try:
                        out[i] = fut.result() or []
                    except Exception as e:
                        if DEBUG:
                            print(f"[Retriever] batch item {i} error: {e}")
                        out[i] = []
        except Exception as e:
            if DEBUG:
                print(f"[Retriever] batch failed: {e}")
        return out
