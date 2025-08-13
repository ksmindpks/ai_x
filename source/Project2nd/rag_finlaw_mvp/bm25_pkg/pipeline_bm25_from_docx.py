#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.docx → 문장기반 청크 → Parquet 저장 → BM25 인덱스 저장

필요 패키지:
  pip install python-docx rank-bm25 pandas pyarrow kiwipiepy

실행 예:
  python bm25_pkg/pipeline_bm25_from_docx.py
"""

import os
import re
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Optional

import pandas as pd
from rank_bm25 import BM25Okapi

# ====== CONFIG ======
input_dirs = [
    r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/금융법령",
    r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/기업법령",
    r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/보험법령",
    r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/상법투자자산증권주식법령",
    r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/은행법령",
    r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/조합법령",
]
OUT_DIR = Path("./out")
PARQUET_OUT = OUT_DIR / "law_chunks.parquet"
BM25_DIR = OUT_DIR / "bm25_index"
TOKENIZER = "kiwi"   # "kiwi" 권장(미설치 시 자동 fallback), "simple" 선택 가능
TOP_LIMIT = None      # 개발 테스트 시 일부만(예: 200) 처리하고 싶으면 숫자 지정

# ====== CHUNKING PARAMS ======
CHUNK_SIZE = 800      # 청크 최대 길이(문자)
CHUNK_OVERLAP = 80    # 청크 중첩(문자)

# ====== deps: python-docx, rank-bm25, pandas, pyarrow, (권장) kiwipiepy ======

# .docx 파서
try:
    from docx import Document
except Exception as e:
    print("[ERROR] python-docx가 필요합니다. 설치: pip install python-docx", file=sys.stderr)
    raise


# ---------------------------
# Utils
# ---------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_sentences(text: str) -> List[str]:
    """한국어/법령 문장 경계 대략치"""
    text = text.strip()
    # 마침표/물음표/느낌표/닫는 괄호 뒤 공백 등 기준
    sentences = re.split(r"(?<=[\.!?])\s+|(?<=\))\s+(?=[가-힣A-Za-z0-9])", text)
    return [s.strip() for s in sentences if s and s.strip()]


def chunk_by_sentences(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    sents = split_sentences(text)
    out, buf = [], ""
    for sent in sents:
        if not buf:
            buf = sent
            continue
        if len(buf) + 1 + len(sent) <= max_chars:
            buf = f"{buf} {sent}"
        else:
            out.append(buf.strip())
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:] + " " + sent
            else:
                buf = sent
    if buf:
        out.append(buf.strip())
    return out


def iter_docx_files(root_dirs: List[str]) -> List[Path]:
    files, seen = [], set()
    for d in root_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] 입력 폴더 없음: {p}")
            continue
        for fp in p.rglob("*.docx"):
            if fp.name.startswith("~$"):  # 잠금 파일 무시
                continue
            key = str(fp.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
    return sorted(files)


def read_docx_text(path: Path) -> str:
    try:
        doc = Document(str(path))
    except Exception as e:
        print(f"[DOCX ERROR] {path.name}: {e}")
        return ""
    paras = [normalize_text(p.text) for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n".join([p for p in paras if p])
    return normalize_text(text)


# ---------------------------
# Tokenizer
# ---------------------------
def _load_kiwi():
    try:
        from kiwipiepy import Kiwi
        return Kiwi()
    except Exception:
        return None


def _load_okt():
    try:
        from konlpy.tag import Okt
        return Okt()
    except Exception:
        return None


def _load_mecab():
    try:
        from konlpy.tag import Mecab
        return Mecab()
    except Exception:
        return None


def build_tokenizer(name: str) -> Callable[[str], List[str]]:
    name = (name or "kiwi").lower()

    if name == "kiwi":
        kiwi = _load_kiwi()
        if kiwi is not None:
            def tok(text: str) -> List[str]:
                return [t.form for t in kiwi.tokenize(text, normalize_coda=True)]
            return tok
        print("[WARN] kiwi 로드 실패 → simple 토크나이저로 대체")

    if name == "okt":
        okt = _load_okt()
        if okt is not None:
            def tok(text: str) -> List[str]:
                return okt.morphs(text)
            return tok
        print("[WARN] Okt 로드 실패 → simple 토크나이저로 대체")

    if name == "mecab":
        mecab = _load_mecab()
        if mecab is not None:
            def tok(text: str) -> List[str]:
                return mecab.morphs(text)
            return tok
        print("[WARN] Mecab 로드 실패 → simple 토크나이저로 대체")

    # 폴백: 한글/영문/숫자 토큰
    def simple(text: str) -> List[str]:
        return re.findall(r"[가-힣A-Za-z0-9]+", text)
    return simple


# ---------------------------
# Pipeline
# ---------------------------
def make_corpus_from_docx(files: List[Path], chunk_size: int, chunk_overlap: int,
                          top_limit: Optional[int] = None) -> List[Dict]:
    corpus: List[Dict] = []
    use_files = files[:top_limit] if top_limit else files
    for i, fp in enumerate(use_files, 1):
        raw = read_docx_text(fp)
        if not raw:
            continue
        chunks = chunk_by_sentences(raw, max_chars=chunk_size, overlap=chunk_overlap)
        for ci, ch in enumerate(chunks):
            corpus.append({
                "text": ch,
                "filename": fp.name,
                "filepath": str(fp),
                "chunk_index": ci
            })
        if i % 20 == 0:
            print(f"  - 진행: {i}/{len(use_files)} (누적 청크 {len(corpus):,}개) ...")
    return corpus


def build_bm25(corpus: List[Dict], tokenizer: Callable[[str], List[str]]) -> BM25Okapi:
    tokenized_corpus = [tokenizer(c["text"]) for c in corpus]
    return BM25Okapi(tokenized_corpus)


def main():
    print("=" * 70)
    print(" BM25 인덱스 빌드 시작")
    print("=" * 70)
    print(f"[입력 폴더] {', '.join(input_dirs)}")
    print(f"[출력 폴더] {OUT_DIR.resolve()}")
    print(f"[Parquet]   {PARQUET_OUT}")
    print(f"[BM25 DIR]  {BM25_DIR}")
    print(f"[토크나이저] {TOKENIZER}")
    print(f"[청크] size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    if TOP_LIMIT:
        print(f"[LIMIT] 파일 상위 {TOP_LIMIT}개만 처리")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 파일 수집
    files = iter_docx_files(input_dirs)
    print(f"[DOCX] 발견: {len(files):,}개")

    if not files:
        print("[ERROR] .docx 파일이 없습니다. 입력 경로를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    # 2) 청크 생성
    print("[단계] .docx → 텍스트 → 청크")
    corpus = make_corpus_from_docx(files, CHUNK_SIZE, CHUNK_OVERLAP, top_limit=TOP_LIMIT)
    print(f"[결과] 청크: {len(corpus):,}개")

    if not corpus:
        print("[ERROR] 유효한 청크가 없습니다.", file=sys.stderr)
        sys.exit(1)

    # 3) Parquet 저장
    print("[단계] Parquet 저장")
    df = pd.DataFrame(corpus)
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"[저장] {PARQUET_OUT.resolve()}  ({PARQUET_OUT.stat().st_size:,} bytes)")

    # 4) BM25 인덱스 생성 & 저장
    print("[단계] BM25 인덱스 생성")
    tokenizer = build_tokenizer(TOKENIZER)
    bm25 = build_bm25(corpus, tokenizer)
    print("[완료] BM25Okapi 준비")

    payload = {
        "bm25": bm25,
        "corpus": corpus,
        "tokenizer": TOKENIZER.lower(),
        "built_at": datetime.now().isoformat(timespec="seconds")
    }
    out_pkl = BM25_DIR / "bm25.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    print(f"[저장] {out_pkl.resolve()}  ({out_pkl.stat().st_size:,} bytes)")
    print("=" * 70)
    print(" BM25 인덱스 빌드 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
