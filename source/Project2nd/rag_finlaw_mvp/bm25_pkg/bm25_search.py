#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bm25_search.py

build_bm25_index.py로 생성한 인덱스를 사용하여 쿼리 검색을 수행합니다.

예)
python bm25_search.py --index ./bm25_index --query "전자금융거래법 제8조 고의 중대한 과실" --topk 5 --tokenizer kiwi
"""
import argparse, os, sys, json, pickle, re
import pandas as pd

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
    except Exception as e:
        print("WARN: kiwipiepy를 사용할 수 없어 simple 토크나이저로 대체합니다. (설치: pip install kiwipiepy)", file=sys.stderr)
        return tokenize_simple

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="build_bm25_index.py로 생성한 디렉토리")
    ap.add_argument("--query", required=True, help="검색 질의문")
    ap.add_argument("--topk", type=int, default=5, help="반환 개수")
    ap.add_argument("--tokenizer", choices=["simple","kiwi"], default="simple")
    args = ap.parse_args()

    pkl = os.path.join(args.index, "bm25.pkl")
    meta_path = os.path.join(args.index, "meta.parquet")
    assert os.path.exists(pkl), f"bm25.pkl을 찾을 수 없습니다: {pkl}"
    assert os.path.exists(meta_path), f"meta.parquet을 찾을 수 없습니다: {meta_path}"

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    ids = data["ids"]
    tokenized_corpus = data["tokenized_corpus"]

    meta = pd.read_parquet(meta_path)

    # 쿼리 토크나이즈
    tokenizer = get_kiwi_tokenizer() if args.tokenizer == "kiwi" else tokenize_simple
    q_tokens = tokenizer(args.query)

    # 점수 계산
    scores = bm25.get_scores(q_tokens)
    import numpy as np
    idxs = np.argsort(scores)[::-1][:args.topk]

    # 결과 출력
    rows = []
    for rank, i in enumerate(idxs, start=1):
        row = meta.iloc[i].to_dict()
        row["_rank"] = rank
        row["_score"] = float(scores[i])
        rows.append(row)

    print(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
