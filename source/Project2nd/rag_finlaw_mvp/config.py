# config.py 최적 설정

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys are required in .env file")

# Pinecone
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_NAMESPACE = ""
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "codedoc-law-index")

# Model 설정 (최적 조합)
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
GENERATION_MODEL = "gpt-4o-mini"  # 균형잡힌 성능

# 검색 설정
DEFAULT_TOP_K = 5  # 상위 5개
MAX_WORKERS = min(os.cpu_count() * 2, 10)  # 적절한 병렬처리
BATCH_SIZE = 30  # 배치 크기

# 하이브리드 설정 (벡터 중심)
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "./bm25_pkg/out/bm25_index")
BM25_TOKENIZER = "kiwi"

# 검색 가중치 (벡터 우선)
TOP_K_BM25 = 8
TOP_K_VEC = 10
WEIGHT_BM25 = 0.2  # BM25 비중 낮춤
WEIGHT_VEC = 0.8   # 벡터 비중 높임

# 디버그
DEBUG_HYBRID = os.getenv("DEBUG_HYBRID", "false").lower() in ("1", "true", "yes")