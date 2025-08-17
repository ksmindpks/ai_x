# rag/memory_optimized_retriever.py
# -*- coding: utf-8 -*-
"""
rag/memory_optimized_retriever.py - 메모리 효율/속도 개선형 (신규 파일)
- BM25 코퍼스 메모리 매핑
- 임베딩 캐시
- BM25 + 벡터 검색 병렬 실행 및 강건한 점수 병합
- 기존 retriever.retrieve(*) 인터페이스로 래핑하여 투명하게 대체
"""
from __future__ import annotations

import os
import re
import pickle
import time
import hashlib
import threading
import concurrent.futures as cf
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # numpy 없으면 일부 정규화 기능 비활성

# ---------------- Utilities ----------------
def _normalize_score_list(scores: List[float]) -> List[float]:
    if not scores:
        return []
    if np is None:
        # numpy 없으면 간단한 min-max
        mn, mx = min(scores), max(scores)
        if mx <= mn:
            return [1.0] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]
    arr = np.array(scores, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    arr = np.clip(arr, lo, hi)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return [1.0] * len(scores)
    return ((arr - mn) / (mx - mn)).tolist()

_law_re = re.compile(r'「([^」]+)」')
_article_re = re.compile(r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?')

# ---------------- Memory-mapped corpus ----------------
class MemoryMappedCorpus:
    """코퍼스를 디스크 맵 파일로 보관하여 다중 스레드/프로세스에서 재사용"""
    def __init__(self, path: str):
        self.path = path
        self._corpus: Optional[List[Dict]] = None
        self._lock = threading.RLock()

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def load(self) -> List[Dict]:
        with self._lock:
            if self._corpus is not None:
                return self._corpus
            with open(self.path, "rb") as f:
                self._corpus = pickle.load(f)
            return self._corpus

# ---------------- Embedding (optional) ----------------
class _EmbeddingCache:
    def __init__(self, max_items: int = 1000):
        self._cache: Dict[str, List[float]] = {}
        self._order: List[str] = []
        self._max = max_items
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, vec: List[float]):
        with self._lock:
            if key in self._cache:
                return
            self._cache[key] = vec
            self._order.append(key)
            if len(self._order) > self._max:
                k0 = self._order.pop(0)
                self._cache.pop(k0, None)

# ---------------- Retriever core ----------------
@dataclass
class _Cfg:
    bm25_pickle_path: str
    pinecone_api_key: Optional[str]
    pinecone_index_name: str = "codedoc-law-index"
    namespace: str = ""
    top_k_bm25: int = 8
    top_k_vector: int = 20
    weight_bm25: float = 0.35
    weight_vector: float = 0.65

class _BM25:
    def __init__(self, bm25_path: str):
        self.bm25_path = bm25_path
        self.model = None
        self.corpus_file = f"{bm25_path}.corpus"
        self._corpus_mem = MemoryMappedCorpus(self.corpus_file)
        self._tokenizer = None

    def _setup_tokenizer(self):
        if self._tokenizer is not None:
            return
        try:
            from kiwipiepy import Kiwi  # type: ignore
            kiwi = Kiwi()
            def tokenize(text: str) -> List[str]:
                if not text:
                    return []
                article = _article_re.findall(text)
                article_tokens = []
                for a in article:
                    t = f"제{a[0]}조"
                    if a[1]:
                        t += f"제{a[1]}항"
                    if a[2]:
                        t += f"제{a[2]}호"
                    article_tokens.append(t)
                toks = [t.form for t in kiwi.tokenize(text)]
                stop = {'의','은','는','이','가','을','를','에','으로','와','과','및','또는','도','만','보다'}
                toks = [t for t in toks if len(t) >= 2 and t not in stop]
                return article_tokens + toks
            self._tokenizer = tokenize
            print("[INFO] Kiwi 토크나이저 초기화 성공")
        except Exception:
            def tokenize(text: str) -> List[str]:
                if not text:
                    return []
                article = _article_re.findall(text)
                article_tokens = []
                for a in article:
                    t = f"제{a[0]}조"
                    if a[1]:
                        t += f"제{a[1]}항"
                    if a[2]:
                        t += f"제{a[2]}호"
                    article_tokens.append(t)
                text = re.sub(r"[^0-9A-Za-z가-힣%]+"," ",text)
                toks = [w for w in text.split() if len(w) >= 2]
                return article_tokens + toks
            self._tokenizer = tokenize
            print("[WARN] Kiwi 미사용, 기본 토크나이저 사용")

    def load(self):
        if self.model is not None and self._corpus_mem.exists():
            return  # 이미 로드됨
        # bm25.pkl은 dict(bm25, corpus, tokenizer, ...) 구조를 기대
        if not os.path.exists(self.bm25_path):
            print(f"[WARN] BM25 인덱스 없음: {self.bm25_path}")
            return
        size = os.path.getsize(self.bm25_path)
        print(f"[HybridRetriever] BM25 file: {self.bm25_path}")
        print(f"[HybridRetriever] BM25 size: {size:,} bytes")
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "bm25" not in data or "corpus" not in data:
            raise RuntimeError("Unknown BM25 pickle format")
        self.model = data["bm25"]
        corpus: List[Dict] = data["corpus"]
        # 코퍼스를 별도 맵 파일로 생성/유지
        if not self._corpus_mem.exists():
            try:
                with open(self.corpus_file, "wb") as cf:
                    pickle.dump(corpus, cf, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[WARN] 코퍼스 맵 생성 실패: {e}")
        self._setup_tokenizer()

    def search(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        if self.model is None:
            return []
        scores = self.model.get_scores(query_tokens)
        ranked = sorted([(i, float(s)) for i, s in enumerate(scores) if s > 0.0],
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_corpus_doc(self, idx: int) -> Dict:
        corpus = self._corpus_mem.load()
        if 0 <= idx < len(corpus):
            return corpus[idx]
        return {}

class _Vec:
    def __init__(self, api_key: Optional[str], index_name: str):
        self.api_key = api_key
        self.index_name = index_name
        self.index = None
        self._embedder = None
        self._cache = _EmbeddingCache(500)

    def _setup(self):
        if self.index is not None or self.api_key is None:
            return
        try:
            from pinecone import Pinecone  # type: ignore
            pc = Pinecone(api_key=self.api_key)
            self.index = pc.Index(self.index_name)
        except Exception as e:
            print(f"[WARN] Pinecone 연결 실패: {e}")
            self.index = None
        # embedder
        try:
            try:
                from .embedder import embed_texts  # type: ignore
            except Exception:
                from embedder import embed_texts  # type: ignore
            self._embedder = embed_texts
        except Exception:
            self._embedder = None

    def embed(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        self._setup()
        if self._embedder is None:
            return None
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        v = self._cache.get(key)
        if v is not None:
            return v
        vecs = self._embedder([text])
        if not vecs or not vecs[0]:
            return None
        self._cache.put(key, vecs[0])
        return vecs[0]

    def search(self, text: str, top_k: int, namespace: str = "") -> List[Dict]:
        self._setup()
        if self.index is None:
            return []
        vec = self.embed(text)
        if vec is None:
            return []
        try:
            res = self.index.query(vector=vec, top_k=top_k, include_metadata=True, namespace=namespace)
        except Exception as e:
            print(f"[WARN] Pinecone 검색 실패: {e}")
            return []
        out = []
        for m in getattr(res, "matches", []) or []:
            md = getattr(m, "metadata", {}) or {}
            out.append({
                "text": md.get("text",""),
                "filename": md.get("filename",""),
                "chunk_index": md.get("chunk_index",-1),
                "score": float(getattr(m, "score", 0.0))
            })
        return out

# ---------------- Public factory ----------------
class MemoryOptimizedHybridRetriever:
    def __init__(self, cfg: _Cfg):
        self.cfg = cfg
        self._bm25 = _BM25(cfg.bm25_pickle_path)
        self._vec = _Vec(cfg.pinecone_api_key, cfg.pinecone_index_name)
        self._init_lock = threading.Lock()
        self._inited = False

    def _ensure_loaded(self):
        if self._inited:
            return
        with self._init_lock:
            if not self._inited:
                self._bm25.load()
                self._inited = True

    def _analyze(self, q: str) -> Dict:
        q = (q or "").strip()
        law = None; arts: List[str] = []
        m = _law_re.search(q)
        if m:
            law = m.group(1)
        for a in _article_re.findall(q):
            s = f"제{a[0]}조"
            if a[1]: s += f"제{a[1]}항"
            if a[2]: s += f"제{a[2]}호"
            arts.append(s)
        # 키워드
        qq = q
        if law:
            qq = re.sub(r'「[^」]+」','',qq)
        for a in arts:
            qq = re.sub(re.escape(a),'',qq)
        kws = re.findall(r'[가-힣]{2,}', qq)
        stop = {'관한','따른','따라','경우','사항','내용','법률','시행령','어떤','어떻게','무엇'}
        kws = [k for k in kws if k not in stop and len(k)>=2]
        cleaned = ' '.join((arts[:2] + kws[:5]))
        qtype = ("numeric" if re.search(r'(몇|얼마|기간|언제)', q) else
                 "organization" if re.search(r'(누구|누가|기관|담당)', q) else
                 "definition" if re.search(r'(무엇|뜻|의미|정의)', q) else
                 "general")
        return {"law":law, "arts":arts, "kws":kws, "cleaned":cleaned, "qtype":qtype}

    def retrieve(self, query: str, top_k: int = 7) -> List[Dict]:
        if not query or not str(query).strip():
            return []
        self._ensure_loaded()
        analysis = self._analyze(query)

        # 토큰화 한번만
        self._bm25._setup_tokenizer()
        q_tokens = self._bm25._tokenizer(analysis["cleaned"])  # type: ignore
        # 전략
        if analysis["arts"]:
            bm25_k = max(self.cfg.top_k_bm25, top_k + 3)
            vec_k = max(self.cfg.top_k_vector // 2, top_k)
            w_bm25, w_vec = 0.7, 0.3
        else:
            bm25_k = max(self.cfg.top_k_bm25, top_k + 2)
            vec_k = max(self.cfg.top_k_vector, top_k + 5)
            w_bm25, w_vec = self.cfg.weight_bm25, self.cfg.weight_vector

        # 병렬 실행
        bm25_res: List[Tuple[int,float]] = []
        vec_res: List[Dict] = []
        try:
            with cf.ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(self._bm25.search, q_tokens, bm25_k)
                f2 = ex.submit(self._vec.search, analysis["cleaned"], vec_k, self.cfg.namespace)
                bm25_res = f1.result(timeout=30) or []
                vec_res = f2.result(timeout=30) or []
        except Exception:
            bm25_res = self._bm25.search(q_tokens, bm25_k)
            vec_res = self._vec.search(analysis["cleaned"], vec_k, self.cfg.namespace)

        # 결과 매핑
        merged: Dict[Tuple[str,int], Dict] = {}
        # bm25
        bm25_scores = [s for _, s in bm25_res]
        bm25_norm = _normalize_score_list(bm25_scores)
        for (i, raw), norm in zip(bm25_res, bm25_norm):
            doc = self._bm25.get_corpus_doc(i)
            key = (doc.get("filename",""), doc.get("chunk_index", i))
            merged[key] = {
                "text": doc.get("text",""),
                "filename": doc.get("filename",""),
                "chunk_index": doc.get("chunk_index", i),
                "bm25_raw": raw, "bm25_norm": norm,
                "vec_raw": 0.0, "vec_norm": 0.0,
                "q": analysis
            }
        # vec
        vec_scores = [r.get("score",0.0) for r in vec_res]
        vec_norm = _normalize_score_list(vec_scores)
        for r, norm in zip(vec_res, vec_norm):
            key = (r.get("filename",""), r.get("chunk_index",-1))
            item = merged.get(key)
            if not item:
                item = {
                    "text": r.get("text",""),
                    "filename": r.get("filename",""),
                    "chunk_index": r.get("chunk_index",-1),
                    "bm25_raw": 0.0, "bm25_norm": 0.0,
                    "vec_raw": r.get("score",0.0), "vec_norm": norm,
                    "q": analysis
                }
                merged[key] = item
            else:
                item["vec_raw"] = r.get("score",0.0)
                item["vec_norm"] = norm

        # 품질/부스팅
        out = []
        for item in merged.values():
            text = item["text"]
            q = item["q"]
            quality = 0.0
            if q["law"] and q["law"] in text:
                quality += 0.3
            for a in q["arts"]:
                if a in text:
                    quality += 0.5; break
            kws = q["kws"]
            if kws:
                hit = sum(1 for kw in kws if kw in text)
                quality += min(0.2, 0.2 * hit / max(1,len(kws)))
            if 80 <= len(text) <= 2000:
                quality += 0.1
            if q["qtype"] == "numeric" and re.search(r'\d+(?:개월|일|년|월|%)', text):
                quality += 0.15
            elif q["qtype"] == "organization" and re.search(r'[가-힣]{2,}(?:장관|위원회|기관)', text):
                quality += 0.15
            elif q["qtype"] == "definition" and re.search(r'(정의|의미|뜻|내용)', text):
                quality += 0.1

            hybrid = (w_bm25 * item["bm25_norm"] + w_vec * item["vec_norm"])
            if item["bm25_raw"] > 0 and item["vec_raw"] > 0:
                hybrid *= 1.2
            final = hybrid * (1 + min(1.0, quality))
            out.append({
                "text": text,
                "filename": item["filename"],
                "chunk_index": item["chunk_index"],
                "score": final,
                "final_score": final,
            })

        out.sort(key=lambda x: x["final_score"], reverse=True)
        # 다양성 보장
        k_final = max(1, top_k)
        diversified: List[Dict] = []
        per_file = max(2, k_final // 3)
        seen: Dict[str,int] = {}
        for r in out:
            fn = r.get("filename","")
            c = seen.get(fn,0)
            if c < per_file:
                diversified.append(r); seen[fn]=c+1
                if len(diversified) >= k_final:
                    break
        if len(diversified) < k_final:
            for r in out:
                if r not in diversified:
                    diversified.append(r)
                if len(diversified) >= k_final:
                    break
        return diversified[:k_final]

def create_memory_optimized_retriever(config: Dict) -> MemoryOptimizedHybridRetriever:
    bm25_path = config.get("bm25_pickle_path") or os.path.join("./bm25_pkg/out/bm25_index","bm25.pkl")
    pinecone_key = config.get("pinecone_api_key")
    pinecone_index = config.get("pinecone_index_name", "codedoc-law-index")
    namespace = config.get("namespace","")
    top_k_bm25 = int(config.get("top_k_bm25", 8))
    top_k_vector = int(config.get("top_k_vector", 20))
    weight_bm25 = float(config.get("weight_bm25", 0.35))
    weight_vector = float(config.get("weight_vector", 0.65))
    cfg = _Cfg(
        bm25_pickle_path=bm25_path,
        pinecone_api_key=pinecone_key,
        pinecone_index_name=pinecone_index,
        namespace=namespace,
        top_k_bm25=top_k_bm25,
        top_k_vector=top_k_vector,
        weight_bm25=weight_bm25,
        weight_vector=weight_vector,
    )
    return MemoryOptimizedHybridRetriever(cfg)
