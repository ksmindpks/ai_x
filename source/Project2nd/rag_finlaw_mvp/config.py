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
PINECONE_NAMESPACE = ""  # 빈 문자열 사용

# Model
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
GENERATION_MODEL = "gpt-3.5-turbo"  # gpt-4는 너무 설명적, gpt-4o-mini → gpt-4o (품질 우선)

# Evaluation
DEFAULT_TOP_K = 7  # 5 → 7 (더 많은 컨텍스트)
MAX_WORKERS = min(os.cpu_count() * 2, 20)
BATCH_SIZE = 50

# BM25 Hybrid 설정
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "./bm25_pkg/out/bm25_index")
BM25_TOKENIZER = os.getenv("BM25_TOKENIZER", "kiwi")
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 12))
TOP_K_VEC = int(os.getenv("TOP_K_VEC", 12))
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", 0.6))
WEIGHT_VEC = float(os.getenv("WEIGHT_VEC", 0.4))