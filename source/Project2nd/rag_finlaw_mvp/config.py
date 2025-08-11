from dotenv import load_dotenv
import os
load_dotenv()

import sys

# API 키 검증
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not set")
    sys.exit(1)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY is not set")
    sys.exit(1)

PINECONE_HOST = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "codedoc-law-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

TOP_K = int(os.getenv("TOP_K", "8"))

# Validation Excel files (2개 파일)
VAL_EXCEL_FILE1 = os.getenv("VAL_EXCEL_FILE1", "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/법령문제_금융은행보험.xlsx")
VAL_EXCEL_FILE2 = os.getenv("VAL_EXCEL_FILE2", "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/법령문제_투자기업조합통합.xlsx")

# Sheet names
SHEET_MCQ = "사지선다형"
SHEET_SHORT = "단답형"