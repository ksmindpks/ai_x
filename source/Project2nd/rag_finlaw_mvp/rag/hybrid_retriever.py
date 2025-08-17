# -*- coding: utf-8 -*-
"""
rag/hybrid_retriever.py
- BM25 로더 강화: dict 포맷에서 bm25 없음 → 즉시 재구축 (rank_bm25 우선, 실패 시 MinimalBM25 폴백)
- 디버그와 무관한 1회 경고(치명 상태 가시화)
- Kiwi 토크나이저 우선, 실패 시 간단 토크나이저
- PineconeWrapper는 선택 (없으면 벡터 검색 생략)
- 기존 병합/정규화/인터페이스(search/search_batch, retrieve/retrieve_batch) 유지
"""

from __future__ import annotations
import os, re, pickle, math, threading
from dataclasses import dataclass
from typing import List, Dict, Optional

# ---------- Optional deps ----------
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from kiwipiepy import Kiwi
    _KIWI = Kiwi()
except Exception:
    _KIWI = None

# ---------- Config ----------
try:
    from config import DEBUG_HYBRID
    _DBG = bool(DEBUG_HYBRID)
except Exception:
    _DBG = os.getenv("DEBUG_HYBRID", "false").lower() in ("1", "true", "yes")

# ---------- Tokenizer ----------
def _tokenize_ko(text: str) -> List[str]:
    if not text:
        return []
    # 법조 패턴 보강 토큰
    arts = []
    for a in re.findall(r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?', text):
        t = f"제{a[0]}조"
        if a[1]: t += f"제{a[1]}항"
        if a[2]: t += f"제{a[2]}호"
        arts.append(t)
    if _KIWI is not None:
        try:
            toks = [t.form for t in _KIWI.tokenize(text)]
        except Exception:
            toks = []
    else:
        toks = re.sub(r"[^0-9A-Za-z가-힣%]+", " ", text).split()
    stop = {'의','은','는','이','가','을','를','에','으로','와','과','및','또는','도','만','보다'}
    toks = [t for t in toks if len(t) >= 2 and t not in stop]
    return arts + toks

# ---------- MinimalBM25 (fallback) ----------
class MinimalBM25:
    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        self.doc_len = [len(d) for d in tokenized_corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df: Dict[str, int] = {}
        for d in tokenized_corpus:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def get_scores(self, q: List[str]) -> List[float]:
        if not self.N:
            return []
        scores = [0.0] * self.N
        for qi in q:
            idf = self.idf.get(qi)
            if idf is None:
                continue
            for i, doc in enumerate(self.corpus):
                tf = 0
                for t in doc:
                    if t == qi:
                        tf += 1
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1.0)))
                scores[i] += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        return scores

# ---------- Pinecone wrapper ----------
class PineconeWrapper:
    def __init__(self, index): self.index = index
    def vector_search(self, emb: List[float], top_k: int, namespace: str = "") -> List[Dict]:
        if self.index is None: return []
        try:
            res = self.index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=namespace or None)
            out = []
            for m in getattr(res, "matches", []) or []:
                md = getattr(m, "metadata", {}) or {}
                txt = md.get("text") or md.get("content") or ""
                out.append({"text": txt, "score": float(getattr(m, "score", 0.0)),
                            "filename": md.get("filename"), "chunk_index": md.get("chunk_index", -1)})
            return out
        except Exception as e:
            if _DBG: print(f"[HybridRetriever] Pinecone query error: {e}")
            return []

# ---------- Config dataclass ----------
@dataclass
class HybridConfig:
    top_k_bm25: int = 7
    top_k_vec: int = 15
    weight_bm25: float = 0.5
    weight_vec: float = 0.5
    namespace: str = ""
    debug: bool = False
    bm25_pickle_path: str = os.getenv("BM25_PICKLE_PATH", "./bm25_pkg/out/bm25_index/bm25.pkl")

# ---------- Retriever ----------
class HybridRetriever:
    def __init__(self, cfg: HybridConfig, pinecone_wrapper: Optional[PineconeWrapper] = None):
        self.cfg = cfg
        self._pcw = pinecone_wrapper
        self._bm25 = None
        self._bm25_docs: List[str] = []
        self._bm25_ready = False
        self._bm25_lock = threading.Lock()
        self._warned_once = False

    def _log(self, msg: str):
        if self.cfg.debug:
            print(msg)

    def _warn_once(self, msg: str):
        if not self._warned_once:
            print(msg)  # 디버그 불문 1회 경고
            self._warned_once = True

    def _load_bm25(self):
        if self._bm25_ready: return
        with self._bm25_lock:
            if self._bm25_ready: return
            pkl = os.path.abspath(self.cfg.bm25_pickle_path)
            try:
                self._log(f"[HybridRetriever] BM25 file: {pkl}")
                if os.path.exists(pkl):
                    self._log(f"[HybridRetriever] BM25 size: {os.path.getsize(pkl):,} bytes")
                with open(pkl, "rb") as f:
                    obj = pickle.load(f)

                # dict 포맷 우선: {'bm25': ?, 'corpus': [...]}
                if isinstance(obj, dict) and ("corpus" in obj or "docs" in obj):
                    corpus = obj.get("corpus") or obj.get("docs") or []
                    # 문자열/딕셔너리 모두 허용
                    texts = []
                    for d in corpus:
                        if isinstance(d, str):
                            texts.append(d)
                        elif isinstance(d, dict):
                            texts.append(d.get("text", ""))
                        else:
                            texts.append(str(d))
                    self._bm25_docs = texts
                    bm25_obj = obj.get("bm25", None)
                    if not bm25_obj:
                        toks = [_tokenize_ko(t) for t in self._bm25_docs]
                        if BM25Okapi is not None:
                            try:
                                bm25_obj = BM25Okapi(toks)
                                self._log(f"[HybridRetriever] BM25 rebuilt via rank_bm25 (n={len(texts):,})")
                            except Exception as e:
                                self._log(f"[HybridRetriever] rank_bm25 rebuild failed: {e}")
                                bm25_obj = None
                        if bm25_obj is None:
                            bm25_obj = MinimalBM25(toks)
                            self._warn_once(f"[WARN] BM25 rebuilt via MinimalBM25 (n={len(texts):,})")
                    self._bm25 = bm25_obj

                # rank_bm25 객체 자체가 저장된 경우
                elif (BM25Okapi is not None) and isinstance(obj, BM25Okapi):
                    self._bm25 = obj
                    docs = getattr(obj, "corpus", None)
                    if isinstance(docs, list):
                        self._bm25_docs = [" ".join(d) if isinstance(d, list) else str(d) for d in docs]

                # 덕 타이핑
                elif hasattr(obj, "get_scores"):
                    self._bm25 = obj
                    docs = getattr(obj, "corpus", None) or getattr(obj, "docs", None)
                    if isinstance(docs, list):
                        self._bm25_docs = [" ".join(d) if isinstance(d, list) else str(d) for d in docs]

                # 리스트만 저장된 경우: corpus로 간주
                elif isinstance(obj, list):
                    self._bm25_docs = [str(x) for x in obj]
                    toks = [_tokenize_ko(t) for t in self._bm25_docs]
                    if BM25Okapi is not None:
                        self._bm25 = BM25Okapi(toks)
                        self._log(f"[HybridRetriever] BM25 built via rank_bm25 (n={len(obj):,})")
                    else:
                        self._bm25 = MinimalBM25(toks)
                        self._warn_once(f"[WARN] BM25 built via MinimalBM25 (n={len(obj):,})")
                else:
                    raise RuntimeError("Unknown BM25 pickle format")

                self._bm25_ready = True
                self._log(f"[HybridRetriever] Loaded BM25 with {len(self._bm25_docs):,} documents")
                if not self._bm25:
                    self._warn_once("[WARN] BM25 객체가 준비되지 않음(검색 불가)")
            except Exception as e:
                self._bm25_ready = False
                self._warn_once(f"[WARN] BM25 로드 실패: {e} (경로: {pkl})")

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        self._load_bm25()
        if not self._bm25 or not self._bm25_docs:
            self._warn_once("[WARN] BM25 not ready — 빈 결과 반환")
            return []
        try:
            qtok = _tokenize_ko(query)
            scores = self._bm25.get_scores(qtok)
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [{"text": self._bm25_docs[i], "bm25_score": float(scores[i])} for i in idxs if scores[i] > 0.0]
        except Exception as e:
            self._warn_once(f"[WARN] BM25 search error: {e}")
            return []

    def _vec_search(self, query: str, top_k: int) -> List[Dict]:
        if self._pcw is None:
            return []
        try:
            # 지연 임포트: embedder가 없으면 조용히 폴백
            try:
                from .embedder import embed_texts as _embed
            except Exception:
                from embedder import embed_texts as _embed  # type: ignore
            emb = _embed([query])[0]
            return self._pcw.vector_search(emb, top_k=top_k, namespace=self.cfg.namespace)
        except Exception as e:
            if _DBG: print(f"[HybridRetriever] Vector search error: {e}")
            return []

    @staticmethod
    def _normalize(xs: List[float]) -> List[float]:
        if not xs: return []
        lo, hi = min(xs), max(xs)
        if hi <= lo: return [1.0] * len(xs)
        return [(x - lo) / (hi - lo) for x in xs]

    def _merge(self, bm25_hits: List[Dict], vec_hits: List[Dict], top_k: int) -> List[Dict]:
        pool: Dict[str, Dict] = {}
        for h in bm25_hits:
            t = h.get("text", "")
            pool.setdefault(t, {"text": t, "bm25_score": 0.0, "vec_score": 0.0})
            pool[t]["bm25_score"] = max(pool[t]["bm25_score"], float(h.get("bm25_score", 0.0)))
        for h in vec_hits:
            t = h.get("text", "")
            pool.setdefault(t, {"text": t, "bm25_score": 0.0, "vec_score": 0.0})
            pool[t]["vec_score"] = max(pool[t]["vec_score"], float(h.get("score", 0.0)))
        items = list(pool.values())
        bm = self._normalize([it["bm25_score"] for it in items])
        vc = self._normalize([it["vec_score"] for it in items])
        for i, it in enumerate(items):
            it["final_score"] = self.cfg.weight_bm25 * bm[i] + self.cfg.weight_vec * vc[i]
        items.sort(key=lambda x: x["final_score"], reverse=True)
        return items[:max(1, top_k)]

    # ---------- Public ----------
    def retrieve(self, query: str, top_k: int = 7, debug: bool = False) -> List[Dict]:
        q = re.sub(r'「[^」]{60,}」', "", query or "")
        bm25_hits = self._bm25_search(q, self.cfg.top_k_bm25 if top_k is None else top_k)
        vec_hits  = self._vec_search(q, self.cfg.top_k_vec if top_k is None else max(top_k, 5))
        merged = self._merge(bm25_hits, vec_hits, top_k=top_k or 7)
        if (self.cfg.debug or debug) and merged:
            self._log(f"[HybridRetriever] top score: {merged[0].get('final_score',0):.3f}")
        return merged

    def retrieve_batch(self, queries: List[str], top_k: int = 7, debug: bool = False) -> List[List[Dict]]:
        return [self.retrieve(q, top_k=top_k, debug=(debug and (i % 20 == 0))) for i, q in enumerate(queries)]

    # ---------- Backward compat ----------
    def search(self, query: str, top_k_bm25: int = None, top_k_vec: int = None, debug: bool = False):
        tb = top_k_bm25 or self.cfg.top_k_bm25
        tv = top_k_vec   or self.cfg.top_k_vec
        bm25_hits = self._bm25_search(query, tb)
        vec_hits  = self._vec_search(query, tv)
        return self._merge(bm25_hits, vec_hits, top_k=max(tb, tv, 7))

    def search_batch(self, queries, top_k_bm25: int = None, top_k_vec: int = None, debug: bool = False):
        tb = top_k_bm25 or self.cfg.top_k_bm25
        tv = top_k_vec   or self.cfg.top_k_vec
        return [self.search(q, tb, tv, debug=(debug and (i % 20 == 0))) for i, q in enumerate(queries)]
