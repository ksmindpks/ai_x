from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX, PINECONE_NAMESPACE, TOP_K
from .embedder import embed_texts

pc = Pinecone(api_key=PINECONE_API_KEY)

def retrieve(query: str, filters: Dict[str, Any] | None = None, top_k: int = TOP_K):
    # embed query
    try:
        vec = embed_texts([query])[0]
        index = pc.Index(host=PINECONE_HOST)
        kwargs = {
            "vector": vec,
            "top_k": top_k,
            "namespace": PINECONE_NAMESPACE,
            "include_metadata": True
        }
        if filters:
            kwargs["filter"] = filters
        res = index.query(**kwargs)
        hits = []
        for match in res.matches:
            md = match.metadata or {}
            hits.append({
                "id": match.id,
                "score": match.score,
                "text": md.get("text", ""),
                "filename": md.get("filename"),
                "chunk_index": md.get("chunk_index"),
                "doc_id": md.get("doc_id"),
                "source": md.get("source"),
            })
        return hits
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []