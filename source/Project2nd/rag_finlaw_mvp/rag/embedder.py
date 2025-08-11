from typing import List
from openai import OpenAI
from config import OPENAI_API_KEY, EMBED_MODEL

_client = None

def _client_once():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def embed_texts(texts: List[str]) -> List[List[float]]:
    client = _client_once()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]
