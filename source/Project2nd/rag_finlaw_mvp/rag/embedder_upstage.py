"""
간소화된 Upstage 임베딩 서비스 - 로그 시스템 통합
"""
import os
import requests
import time
from typing import List, Dict, Any

def log_message(log_type, message, module="EMBEDDER"):
    """통합된 로그 함수 - 3단계 분류"""
    # 웹 인터페이스로 전달 시도
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'global_log_callback'):
            callback = st.session_state.global_log_callback
            if callable(callback):
                callback(log_type, message, module, "evaluation")
        else:
            # 웹 환경이 아닐 때만 직접 출력
            print(f"[{module}-{log_type.upper()}] {message}")
    except Exception:
        # 오류 시 직접 출력
        print(f"[{module}-{log_type.upper()}] {message}")

class UpstageEmbedder:
    """로그 시스템 통합된 Upstage 임베딩 서비스"""
    
    def __init__(self, api_key: str = None, model: str = "embedding-query"):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            log_message("FAILURE", "UPSTAGE_API_KEY 환경변수가 설정되지 않음")
            raise ValueError("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다")
            
        self.model = model
        self.api_url = "https://api.upstage.ai/v1/embeddings"
        
        # 차원 자동 감지용
        self._embedding_dim = None
        self._last_request_time = 0
        self._min_interval = 0.05  # 50ms
        
        log_message("SUCCESS", f"Upstage 임베더 초기화 완료 (모델: {model})")
        
    def _rate_limit(self):
        """간단한 rate limiting"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """API 호출"""
        self._rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts if isinstance(texts, list) else [texts]
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # 실패시 바로 exception
        except requests.exceptions.RequestException as e:
            log_message("FAILURE", f"API 호출 실패: {e}")
            raise
        
        try:
            result = response.json()
        except ValueError as e:
            log_message("FAILURE", f"JSON 파싱 실패: {e}")
            raise
        
        embeddings = []
        
        for item in result["data"]:
            embedding = [float(x) for x in item["embedding"]]
            
            # 첫 번째 성공적인 임베딩으로 차원 자동 감지
            if self._embedding_dim is None:
                self._embedding_dim = len(embedding)
                log_message("SUCCESS", f"임베딩 차원 자동 감지: {self._embedding_dim}")
            
            embeddings.append(embedding)
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        if not text or not text.strip():
            # 빈 벡터 반환시 차원 사용
            dim = self._embedding_dim or 1536  # fallback
            log_message("INFO", "빈 텍스트로 인한 제로 벡터 반환")
            return [0.0] * dim
        
        try:
            embeddings = self._call_api([text.strip()])
            return embeddings[0] if embeddings else [0.0] * (self._embedding_dim or 1536)
        except Exception as e:
            log_message("FAILURE", f"단일 쿼리 임베딩 실패: {e}")
            dim = self._embedding_dim or 1536
            return [0.0] * dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서 배치 임베딩"""
        if not texts:
            log_message("INFO", "빈 텍스트 리스트")
            return []
        
        log_message("INFO", f"배치 임베딩 시작: {len(texts)}개 문서")
        
        # 빈 텍스트 처리
        clean_texts = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                clean_texts.append(text.strip())
                text_indices.append(i)
        
        if not clean_texts:
            dim = self._embedding_dim or 1536
            log_message("INFO", f"모든 텍스트가 빈 값, {len(texts)}개 제로 벡터 반환")
            return [[0.0] * dim] * len(texts)
        
        # 배치 API 호출
        try:
            embeddings = self._call_api(clean_texts)
            log_message("SUCCESS", f"배치 임베딩 완료: {len(clean_texts)}개 처리됨")
        except Exception as e:
            log_message("FAILURE", f"배치 임베딩 실패: {e}")
            dim = self._embedding_dim or 1536
            return [[0.0] * dim] * len(texts)
        
        # 원본 순서로 복원
        result = []
        embedding_idx = 0
        dim = self._embedding_dim or (len(embeddings[0]) if embeddings else 1536)
        
        for i in range(len(texts)):
            if i in text_indices:
                result.append(embeddings[embedding_idx])
                embedding_idx += 1
            else:
                result.append([0.0] * dim)
        
        return result

# 전역 인스턴스
_embedder = None

def get_embedder() -> UpstageEmbedder:
    global _embedder
    if _embedder is None:
        try:
            _embedder = UpstageEmbedder()
            log_message("SUCCESS", "전역 임베더 인스턴스 생성 완료")
        except Exception as e:
            log_message("FAILURE", f"임베더 초기화 실패: {e}")
            raise
    return _embedder