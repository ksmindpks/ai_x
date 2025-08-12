#!/usr/bin/env python
# -*- coding: utf-8 -*-

export_bm25_input.py

여러 소스(폴더/파일)의 청크 데이터를 모아 BM25 인덱스 빌드에 바로 쓸 수 있는
단일 Parquet 파일로 내보냅니다.

지원:
- 입력: 디렉토리 또는 파일 경로(복수 가능) — CSV / JSONL / Parquet
- 자동 컬럼 매핑: text 컬럼 후보 중 하나를 자동 감지(text/chunk/content/body)
- 선택 메타 컬럼 동시 보존: id/_id, law_name, article_no, effective_from, effective_to, filename, chunk_index
- 정리:
  * 공백/개행 정규화
  * 중복 제거(텍스트 해시 기준)
- 출력: 지정한 Parquet 파일 1개

예시:
python export_bm25_input.py \
  --src ./data/chunks1.parquet ./data/chunks2.csv ./more/ \
  --out ./data/law_chunks.parquet

특정 컬럼을 강제로 지정하고 싶다면:
python export_bm25_input.py --src ./data --out ./out.parquet --text-col text --id-col _id


import argparse, os, sys, re, hashlib
import pandas as pd
from typing import List

TEXT_CANDIDATES = ["text", "chunk", "content", "body"]
META_CANDIDATES = ["_id","id","law_name","article_no","effective_from","effective_to","filename","chunk_index"]

def is_supported_file(p: str):
    ext = os.path.splitext(p)[1].lower()
    return ext in [".csv",".json",".jsonl",".parquet",".pq"]

def load_any(p: str) -> pd.DataFrame:
    ext = os.path.splitext(p)[1].lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext in [".json",".jsonl"]:
        try:
            return pd.read_json(p, lines=True)
        except:
            return pd.read_json(p)
    if ext in [".parquet",".pq"]:
        return pd.read_parquet(p)
    raise ValueError(f"지원하지 않는 형식: {p}")

def pick_text_col(df: pd.DataFrame, forced: str=None):
    if forced:
        if forced not in df.columns:
            raise KeyError(f"--text-col='{forced}' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")
        return forced
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    # 마지막 시도: 첫 번째 문자열형 컬럼
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    raise KeyError("본문(text) 컬럼을 찾지 못했습니다. --text-col로 지정해주세요.")

def pick_id_col(df: pd.DataFrame, forced: str=None):
    if forced:
        if forced not in df.columns:
            raise KeyError(f"--id-col='{forced}' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")
        return forced
    for c in ["_id","id"]:
        if c in df.columns:
            return c
    return None

def normalize_text(s: str) -> str:
    s = str(s).replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def collect_sources(srcs: List[str]) -> List[str]:
    files = []
    for src in srcs:
        if os.path.isdir(src):
            for root, _, fs in os.walk(src):
                for f in fs:
                    p = os.path.join(root, f)
                    if is_supported_file(p):
                        files.append(p)
        else:
            if is_supported_file(src):
                files.append(src)
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="입력 경로(파일/폴더) — CSV/JSONL/Parquet 지원, 복수 지정 가능")
    ap.add_argument("--out", required=True, help="출력 Parquet 경로")
    ap.add_argument("--text-col", default=None, help="본문 컬럼명 강제 지정(없으면 자동 감지)")
    ap.add_argument("--id-col", default=None, help="ID 컬럼명 강제 지정(없으면 자동 감지)")
    args = ap.parse_args()

    files = collect_sources(args.src)
    if not files:
        print("입력 파일을 찾지 못했습니다.", file=sys.stderr)
        sys.exit(1)

    frames = []
    for p in files:
        try:
            df = load_any(p)
            tcol = pick_text_col(df, args.text_col)
            idcol = pick_id_col(df, args.id_col)
            # 준비
            use_cols = [tcol] + [c for c in META_CANDIDATES if c in df.columns and c != tcol]
            slim = df[use_cols].copy()
            # rename
            slim.rename(columns={tcol:"text"}, inplace=True)
            if idcol and idcol in slim.columns:
                slim.rename(columns={idcol:"_id"}, inplace=True)
            # text normalize
            slim["text"] = slim["text"].astype(str).map(normalize_text)
            slim["_srcfile"] = os.path.basename(p)
            frames.append(slim)
        except Exception as e:
            print(f"[WARN] {p} 처리 중 오류: {e}", file=sys.stderr)

    if not frames:
        print("유효한 입력 데이터를 수집하지 못했습니다.", file=sys.stderr)
        sys.exit(2)

    out_df = pd.concat(frames, ignore_index=True)

    # _id 없으면 해시로 생성
    if "_id" not in out_df.columns:
        out_df["_id"] = [f"auto_{i}" for i in range(len(out_df))]

    # 중복 제거: text 해시 기준
    out_df["_txhash"] = out_df["text"].map(text_hash)
    out_df.drop_duplicates(subset=["_txhash"], inplace=True)
    out_df.drop(columns=["_txhash"], inplace=True)

    # 권장 컬럼 순서
    ordered = ["_id","law_name","article_no","effective_from","effective_to","filename","chunk_index","text","_srcfile"]
    cols = [c for c in ordered if c in out_df.columns] + [c for c in out_df.columns if c not in ordered]
    out_df = out_df[cols]

    # 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[OK] 내보내기 완료: {args.out} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
