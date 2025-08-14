# config.py - 간소화된 설정

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

# Model 설정
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
GENERATION_MODEL = "gpt-4o-mini"

# 검색 설정
DEFAULT_TOP_K = 5
MAX_WORKERS = min(os.cpu_count() * 2, 10)
BATCH_SIZE = 30

# BM25 경로
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "./bm25_pkg/out/bm25_index")
BM25_TOKENIZER = "kiwi"

# 하이브리드 검색 설정
TOP_K_BM25 = 5
TOP_K_VEC = 15
WEIGHT_BM25 = 0.4
WEIGHT_VEC = 0.6

# 디버그 (기본 비활성화)
DEBUG_HYBRID = os.getenv("DEBUG_HYBRID", "false").lower() in ("1", "true", "yes")