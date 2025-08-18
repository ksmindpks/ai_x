# -*- coding: utf-8 -*-
# rag/embedder_upstage.py

from __future__ import annotations
from typing import List
import os
from langchain_upstage import UpstageEmbeddings

# Upstage Embedding 모델: 4096차원 (중요!)
_MODEL_NAME = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")

# LangChain Upstage 래퍼
_UP_EMB = UpstageEmbeddings(model=_MODEL_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """문서용 임베딩 (여러 개)"""
    if not texts:
        return []
    return _UP_EMB.embed_documents(texts)

def embed_query(text: str) -> List[float]:
    """질의용 임베딩 (1개)"""
    text = text or ""
    return _UP_EMB.embed_query(text)
