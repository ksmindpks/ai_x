# BM25 입력 파일 추출기 (export_bm25_input.py)

여러 데이터 소스(폴더/파일)의 청크 데이터를 모아 **BM25 인덱스 빌드를 위한 Parquet**으로 변환합니다.

## 설치
필요 패키지: pandas, pyarrow
```
pip install pandas pyarrow
```

## 사용 예시
```bash
# 1) 디렉토리와 파일을 혼합 입력
python export_bm25_input.py   --src ./data/chunks1.parquet ./data/chunks2.csv ./more_chunks/   --out ./data/law_chunks.parquet

# 2) 특정 컬럼 강제 (본문이 text가 아닌 content일 때)
python export_bm25_input.py   --src ./data   --out ./out.parquet   --text-col content   --id-col _id
```

## 출력
- 단일 Parquet 파일 1개: `text`(필수), 선택 메타(`_id, law_name, article_no, effective_from, effective_to, filename, chunk_index`) 유지
- 공백 정규화, 텍스트 해시 기준 중복 제거
- 소스 파일명 `_srcfile`로 보관
