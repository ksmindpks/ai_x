#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_bm25_index.py

입력(법령/조문 청크 데이터)으로부터 BM25 인덱스를 생성해 디스크에 저장합니다.
- 지원 포맷: CSV / JSONL / Parquet
- 필수 컬럼: text (본문)
- 선택 컬럼: id, law_name, article_no, effective_from, effective_to, filename, chunk_index 등
- 한국어 토크나이저: 기본은 간단 토큰화(정규식). --tokenizer kiwi 를 지정하면 kiwipiepy 사용
- 출력: bm25.pkl (BM25Okapi 객체), meta.parquet (문서 메타), vocab.json (토큰 통계), index_info.json (설정)

사용 예:
python build_bm25_index.py \\
  --input ./data/law_chunks.parquet \\
  --text-col text \\
  --id-col id \\
  --output ./bm25_index \\
  --tokenizer kiwi

Windows 경로 예:
python build_bm25_index.py --input C:/path/to/chunks.csv --text-col text --output C:/path/to/bm25 --tokenizer simple

"""
import argparse
import os
import sys
import json
import pickle
import re
from datetime import datetime

import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except Exception:
    print("ERROR: rank-bm25가 설치되어 있지 않습니다.\n  pip install rank-bm25", file=sys.stderr)
    sys.exit(1)

# ---- Tokenizers ----
def tokenize_simple(text: str):
    # 한글/영문/숫자 유지, 나머지 공백으로 대체
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    # 소문자 통일 + 공백 기준 split
    return text.lower().strip().split()

def get_kiwi_tokenizer():
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        def _tok(text: str):
            # 품사 무관 기본형 토큰
            return [t.form for t in kiwi.tokenize(text)]
        return _tok
    except Exception as e:
        print("WARN: kiwipiepy를 사용할 수 없어 simple 토크나이저로 대체합니다. (설치: pip install kiwipiepy)", file=sys.stderr)
        return tokenize_simple

def load_input(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    elif ext in [".jsonl", ".json"]:
        try:
            return pd.read_json(path, lines=True)
        except:
            return pd.read_json(path)
    elif ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"지원하지 않는 입력 포맷입니다: {ext}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="입력 파일 경로(CSV/JSONL/Parquet)")
    ap.add_argument("--text-col", default="text", help="본문 텍스트 컬럼명 (기본: text)")
    ap.add_argument("--id-col", default=None, help="문서 ID 컬럼명 (없으면 자동 생성)")
    ap.add_argument("--output", required=True, help="출력 디렉토리")
    ap.add_argument("--tokenizer", choices=["simple","kiwi"], default="simple", help="토크나이저 선택")
    ap.add_argument("--limit", type=int, default=None, help="(옵션) 상위 N개 행만 사용(테스트용)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_input(args.input)
    if args.limit:
        df = df.head(args.limit)

    if args.text_col not in df.columns:
        raise KeyError(f"입력 데이터에 '{args.text_col}' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")

    # id 부여
    if args.id_col and args.id_col in df.columns:
        ids = list(df[args.id_col].astype(str))
    else:
        ids = [f"doc_{i}" for i in range(len(df))]

    texts = list(df[args.text_col].astype(str))

    # 메타데이터 보존(문서별)
    meta_cols = [c for c in df.columns if c not in [args.text_col]]
    meta_df = df.copy()
    meta_df.insert(0, "_id", ids)

    # 토크나이저 선택
    if args.tokenizer == "kiwi":
        tokenizer = get_kiwi_tokenizer()
    else:
        tokenizer = tokenize_simple

    tokenized_corpus = [tokenizer(t) for t in texts]

    # BM25 빌드
    bm25 = BM25Okapi(tokenized_corpus)

    # Vocab/통계(간단)
    vocab = {}
    for toks in tokenized_corpus:
        for w in toks:
            vocab[w] = vocab.get(w, 0) + 1

    # 저장
    with open(os.path.join(args.output, "bm25.pkl"), "wb") as f:
        import pickle
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "tokenized_corpus": tokenized_corpus
        }, f)

    meta_path = os.path.join(args.output, "meta.parquet")
    meta_df.to_parquet(meta_path, index=False)

    with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    info = {
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "input": os.path.abspath(args.input),
        "rows": len(df),
        "text_col": args.text_col,
        "id_col": args.id_col,
        "tokenizer": args.tokenizer
    }
    with open(os.path.join(args.output, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] BM25 인덱스 저장 완료: {args.output}")
    print(" - bm25.pkl, meta.parquet, vocab.json, index_info.json")


if __name__ == "__main__":
    main()
