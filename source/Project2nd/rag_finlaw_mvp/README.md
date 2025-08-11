# 금융·법령 RAG 챗봇 (MVP)

## 1) 환경 준비
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
copy .env.example .env     # .env 편집 (키/네임스페이스)
```

## 2) .env 설정
```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_HOST=https://codedoc-law-index-1s64sba.svc.aped-4627-b74a.pinecone.io
PINECONE_INDEX=codedoc-law-index
PINECONE_NAMESPACE=<사용할 네임스페이스>
PINECONE_ENV=us-east-1
```

## 3) Streamlit UI 실행
```bash
streamlit run app/ui_streamlit.py
```

## 4) 평가 실행 (단답/사지선다 혼합 JSON)
```bash
python app/evaluate_cli.py --val_json data/val.json --limit 100
```

## 메모
- 검색 결과는 Pinecone 메타데이터의 `filename`, `chunk_index`, `text`를 사용합니다.
- 생성 모델은 OpenAI Chat API (기본 gpt-4o-mini)로 설정되어 있습니다.
- 답변 포맷에 "근거·출처"가 강제되며, 불확실 시 보수적으로 응답합니다.
- 로그/고도화(재랭킹, 필터링, 평가지표 확장)는 추후 단계에서 추가하세요.
