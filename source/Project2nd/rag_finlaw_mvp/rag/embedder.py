from typing import List
from openai import OpenAI
from config import OPENAI_API_KEY, EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """배치 임베딩 처리"""
    if not texts:
        return []
    
    # OpenAI는 최대 2048개 동시 처리
    max_batch = 2048
    all_embeddings = []
    
    for i in range(0, len(texts), max_batch):
        batch = texts[i:i + max_batch]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        all_embeddings.extend([d.embedding for d in response.data])
    
    return all_embeddings