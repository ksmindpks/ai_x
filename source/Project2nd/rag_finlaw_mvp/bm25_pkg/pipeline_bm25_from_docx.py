#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
지정된 여러 폴더의 .docx 법령 파일을 조문/항 단위로 청크화하여 Parquet로 저장하고,
이어 BM25 인덱스를 생성(bm25.pkl, meta.parquet, vocab.json, index_info.json)합니다.

- 입력: 아래 CONFIG의 input_dirs 폴더들 (재귀 하위 탐색)
- 출력:
  1) ./out/law_chunks.parquet
  2) ./out/bm25_index/ (bm25.pkl, meta.parquet, vocab.json, index_info.json)
"""

import os
import re
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

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

# ====== deps: python-docx, rank-bm25, pandas, pyarrow, (권장) kiwipiepy ======
try:
    from docx import Document
except Exception:
    print("ERROR: python-docx 미설치.  pip install python-docx", file=sys.stderr)
    sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
except Exception:
    print("ERROR: rank-bm25 미설치.  pip install rank-bm25", file=sys.stderr)
    sys.exit(1)


def tokenize_simple(text: str):
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip().split()


def get_kiwi_tokenizer():
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()

        def _tok(text: str):
            return [t.form for t in kiwi.tokenize(text)]
        return _tok
    except Exception:
        print("WARN: kiwipiepy 사용 불가 → simple 토크나이저로 대체", file=sys.stderr)
        return tokenize_simple


def normalize_space(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def guess_effective_from_from_filename(fname: str):
    """
    파일명에서 시행일을 추정 (예: (20250319), 2025-03-19, 2025.03.19 등)
    """
    base = Path(fname).stem
    # 괄호 안 yyyymmdd
    m = re.search(r"\((20\d{2}[.\-]?\d{2}[.\-]?\d{2})\)", base)
    if not m:
        m = re.search(r"(20\d{2}[.\-]?\d{2}[.\-]?\d{2})", base)
    if m:
        raw = re.sub(r"[.\-]", "", m.group(1))
        if len(raw) == 8:
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return None


def guess_law_name_from_filename(fname: str):
    """
    파일명에서 법령명 유추: 괄호류 제거
    """
    base = Path(fname).stem
    # 괄호 블록 제거
    name = re.sub(r"\(.*?\)", "", base)
    return name.strip(" -_")


def detect_law_form(name: str):
    for k in ["시행령", "시행규칙", "법률", "규칙", "총리령", "대통령령", "부칙"]:
        if k in name:
            return k
    return ""


ARTICLE_RE = re.compile(r"^제\s*\d+(?:의\d+)?\s*조")
CLAUSE_RE = re.compile(r"^\(\d+\)")  # (1), (2)...


def iter_chunks_from_docx(docx_path: str):
    """
    .docx → 조문/항 단위 청크 생성.
    반환 dict 필드:
      _id, law_name, law_form, article_no, clause_no, section, text, effective_from, filename, chunk_index
    """
    doc = Document(docx_path)
    fname = Path(docx_path).name
    law_name = guess_law_name_from_filename(fname)
    law_form = detect_law_form(law_name)
    effective_from = guess_effective_from_from_filename(fname)

    article_no = None
    clause_no = None
    section = "본칙"
    buf = []
    chunk_index = 0

    def flush():
        nonlocal buf, chunk_index
        text = normalize_space(" ".join([t for t in buf if t.strip()]))
        if text:
            _id = f"{law_name}:{article_no or '0'}:{clause_no or '0'}:{chunk_index}"
            yield {
                "_id": _id,
                "law_name": law_name,
                "law_form": law_form,
                "article_no": article_no,
                "clause_no": clause_no,
                "section": section,
                "text": text,
                "effective_from": effective_from,
                "filename": fname,
                "chunk_index": chunk_index,
            }
            chunk_index += 1
        buf = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue

        # 섹션 전환(간단 탐지: 부칙/별표/별지 키워드)
        if any(x in t for x in ["부칙", "별표", "별지"]) and len(t) <= 12:
            # 섹션 턴오버 전에 버퍼 flush
            if buf:
                for row in flush():
                    yield row
            section = "부칙" if "부칙" in t else ("별표/별지" if ("별표" in t or "별지" in t) else section)
            article_no = None
            clause_no = None
            continue

        if ARTICLE_RE.match(t):
            if buf:
                for row in flush():
                    yield row
            # 제○조(의○) 추출
            m = re.search(r"제\s*([\d]+(?:의\d+)?)\s*조", t)
            article_no = m.group(1) if m else None
            clause_no = None
            buf = [t]
        elif CLAUSE_RE.match(t):
            if buf:
                for row in flush():
                    yield row
            cm = re.search(r"\((\d+)\)", t)
            clause_no = cm.group(1) if cm else None
            buf = [t]
        else:
            buf.append(t)

    if buf:
        for row in flush():
            yield row


from pathlib import Path

def walk_docx_files(dirs):
    seen = set()
    for d in dirs:
        base = Path(d)
        if not base.exists():
            print(f"[WARN] 폴더 없음: {d}", file=sys.stderr)
            continue
        for f in base.rglob("*"):
            if f.is_file() and not f.name.startswith("~$") and f.suffix.lower() == ".docx":
                key = str(f.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                yield str(f)  # python-docx가 읽기 쉬운 일반 경로 반환
    if not seen:
        print("[ERROR] .docx 파일을 하나도 찾지 못했습니다.", file=sys.stderr)


def build_bm25(index_input: pd.DataFrame, out_dir: Path, tokenizer_name="kiwi"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 토크나이저
    tokenizer = get_kiwi_tokenizer() if tokenizer_name == "kiwi" else tokenize_simple

    texts = list(index_input["text"].astype(str))
    tokenized = [tokenizer(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    vocab = {}
    for toks in tokenized:
        for w in toks:
            vocab[w] = vocab.get(w, 0) + 1

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "ids": list(index_input["_id"]), "tokenized_corpus": tokenized}, f)

    index_input.to_parquet(out_dir / "meta.parquet", index=False)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    info = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rows": len(index_input),
        "tokenizer": tokenizer_name,
        "text_col": "text",
        "id_col": "_id"
    }
    with open(out_dir / "index_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] BM25 인덱스 저장: {out_dir}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    count = 0
    for fp in walk_docx_files(input_dirs):
        try:
            for row in iter_chunks_from_docx(fp):
                rows.append(row)
                count += 1
                if TOP_LIMIT and count >= TOP_LIMIT:
                    break
            if TOP_LIMIT and count >= TOP_LIMIT:
                break
        except Exception as e:
            print(f"[WARN] {fp} 파싱 실패: {e}", file=sys.stderr)

    if not rows:
        print("ERROR: 청크를 하나도 만들지 못했습니다. 경로/권한/문서형식 확인 요망.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows)

    # 텍스트 중복 제거(해시 기준)
    import hashlib
    df["_txhash"] = df["text"].map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    before = len(df)
    df.drop_duplicates(subset=["_txhash"], inplace=True)
    df.drop(columns=["_txhash"], inplace=True)
    after = len(df)
    print(f"[INFO] 중복 제거: {before} -> {after}")

    # 권장 컬럼 순서
    ordered = ["_id","law_name","law_form","article_no","clause_no","section",
               "effective_from","filename","chunk_index","text"]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    df = df[cols]

    # 저장(Parquet)
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"[OK] 청크 저장: {PARQUET_OUT} (rows={len(df)})")

    # BM25 인덱스 빌드
    build_bm25(df, BM25_DIR, tokenizer_name=TOKENIZER)


if __name__ == "__main__":
    main()
