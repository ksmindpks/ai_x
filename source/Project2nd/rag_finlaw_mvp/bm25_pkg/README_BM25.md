# BM25 인덱스 빌드 & 검색 툴

이 디렉토리는 금융 법령 RAG에서 **BM25 기반 키워드 검색**을 추가하기 위한
간단하지만 실전용 스크립트를 제공합니다.

## 설치
```
pip install -r requirements.txt
```

## 인덱스 빌드
```
python build_bm25_index.py   --input ./data/law_chunks.parquet   --text-col text   --id-col id   --output ./bm25_index   --tokenizer kiwi
```
- 입력 포맷: CSV / JSONL / Parquet
- 최소 컬럼: `text`
- 선택 컬럼(있으면 함께 저장): `id, law_name, article_no, effective_from, effective_to, filename, chunk_index` 등

## 검색 테스트
```
python bm25_search.py   --index ./bm25_index   --query "전자금융거래법 제8조 고의 중대한 과실"   --topk 5   --tokenizer kiwi
```

## 통합 팁
- RAG retriever에서 Pinecone(임베딩) 결과와 BM25 결과를 **합쳐서** rerank 하세요.
- 한국어 토크나이저는 기본 `simple`로도 동작하지만, 정확도를 위해 `--tokenizer kiwi` 권장(설치 쉬움).
- 대용량이면 Elasticsearch/OpenSearch도 고려하세요.
