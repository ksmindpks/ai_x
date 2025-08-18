# config.py - 성능 최적화된 설정 (최종 개선 버전)
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PerformanceConfig:
    """성능 관련 설정 - 최종 개선"""
    # LLM 동시성 최적화
    openai_concurrency: int = 16  # 20 -> 16 (안정성)
    max_retry_attempts: int = 3   # 재시도 유지
    retry_base_delay: float = 0.1  
    request_timeout: int = 30     # 25 -> 30 (안정성)
    
    # 검색 최적화
    batch_size: int = 32          
    max_workers: int = min(os.cpu_count() * 2, 16)
    
    # 메모리 최적화
    max_context_length: int = 3000  # 2000 -> 3000 (개선된 컨텍스트)
    chunk_size: int = 1000          # 800 -> 1000
    enable_memory_mapping: bool = True
    enable_gpu_acceleration: bool = False  # GPU 없는 환경 고려

@dataclass
class SearchConfig:
    """검색 관련 설정 - 핵심 개선"""
    # 하이브리드 검색 최적화 (중요!)
    top_k_bm25: int = 14           # 8 -> 12 (더 많은 후보)
    top_k_vector: int = 12         # 20 -> 12 (균형)
    weight_bm25: float = 0.6       # 0.3 -> 0.4 (BM25 강화)
    weight_vector: float = 0.4     # 0.7 -> 0.6
    
    # BM25 파라미터 (핵심!)
    bm25_k1: float = 1.5           # 1.2 -> 1.5 (term frequency 중요도 상향)
    bm25_b: float = 0.75           # 유지
    
    # 검색 성능 향상
    enable_parallel_search: bool = True
    cache_embeddings: bool = True
    use_approximate_search: bool = False  # True -> False (정확도 우선)
    enable_query_expansion: bool = False  
    enable_reranking: bool = True        # False -> True (품질 향상)
    
    # 품질 설정
    similarity_threshold: float = 0.1    # 0.15 -> 0.1 (더 많은 결과 허용)
    diversity_penalty: float = 0.05      
    freshness_boost: float = 0.02        
    
    # 최종 검색 결과 수
    final_top_k: int = 10  # 5 -> 10 (더 많은 컨텍스트 제공)

@dataclass
class GenerationConfig:
    """생성 관련 설정 - 핵심 개선"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 80          # 50 -> 80 (단답 추출 여유)
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1

    # 답변 생성 파라미터
    max_answer_length: int = 60    # 50 -> 60 (후처리 60자와 일치)
    min_answer_length: int = 2
    answer_confidence_threshold: float = 0.42  # 0.45 -> 0.42 (임계 미세 완화)

    # 폴백 전략
    use_llm_fallback: bool = True
    llm_fallback_threshold: float = 0.42       # 0.45 -> 0.42

    # 프롬프트 최적화
    enable_few_shot: bool = False
    enable_chain_of_thought: bool = False
    context_window_size: int = 6    # 5 -> 6 (컨텍스트 1개 추가)

@dataclass
class SystemConfig:
    """시스템 전체 설정"""
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    
    # Pinecone
    pinecone_host: str = os.getenv("PINECONE_HOST", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX", "codedoc-law-index")
    pinecone_namespace: str = ""
    
    # 모델 설정
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072
    generation_model: str = "gpt-4o-mini"
    
    # BM25 설정
    bm25_index_dir: str = os.getenv("BM25_INDEX_DIR", "./bm25_pkg/out/bm25_index")
    bm25_tokenizer: str = "kiwi"
    
    # 설정 그룹
    performance: PerformanceConfig = PerformanceConfig()
    search: SearchConfig = SearchConfig()
    generation: GenerationConfig = GenerationConfig()
    
    # 디버그 설정
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
    verbose_logging: bool = os.getenv("VERBOSE_LOGGING", "false").lower() in ("1", "true", "yes")
    enable_profiling: bool = os.getenv("ENABLE_PROFILING", "false").lower() in ("1", "true", "yes")
    
    # 캐싱 설정
    enable_result_caching: bool = True
    cache_ttl: int = 3600  # 1시간
    cache_size: int = 10000  # 캐시 크기
    
    def validate(self) -> list[str]:
        """설정 검증"""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY가 설정되지 않음")
        
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY가 설정되지 않음")
            
        if self.performance.openai_concurrency < 1:
            errors.append("openai_concurrency는 1 이상이어야 함")
            
        if abs(self.search.weight_bm25 + self.search.weight_vector - 1.0) > 1e-6:
            errors.append("BM25와 Vector 가중치 합이 1.0이 아님")
        
        if self.search.top_k_bm25 < 1 or self.search.top_k_vector < 1:
            errors.append("top_k 값들은 1 이상이어야 함")
            
        return errors

# 전역 설정 인스턴스
config = SystemConfig()

# 설정 검증
validation_errors = config.validate()
if validation_errors:
    import warnings
    for error in validation_errors:
        warnings.warn(f"설정 경고: {error}")

# 호환성을 위한 기존 변수들 (수정된 값 반영)
OPENAI_API_KEY = config.openai_api_key
PINECONE_API_KEY = config.pinecone_api_key
PINECONE_HOST = config.pinecone_host
PINECONE_NAMESPACE = config.pinecone_namespace
PINECONE_INDEX_NAME = config.pinecone_index_name
EMBED_MODEL = config.embed_model
EMBED_DIM = config.embed_dim
GENERATION_MODEL = config.generation_model
DEFAULT_TOP_K = config.search.final_top_k  # 7 -> 10
MAX_WORKERS = config.performance.max_workers
BATCH_SIZE = config.performance.batch_size
BM25_INDEX_DIR = config.bm25_index_dir
BM25_TOKENIZER = config.bm25_tokenizer
TOP_K_BM25 = config.search.top_k_bm25
TOP_K_VEC = config.search.top_k_vector
WEIGHT_BM25 = config.search.weight_bm25
WEIGHT_VEC = config.search.weight_vector
DEBUG_HYBRID = config.debug_mode

# 추가 전역 변수 (generator.py와 utils.py에서 사용)
MAX_ANSWER_LENGTH = config.generation.max_answer_length
MIN_ANSWER_LENGTH = config.generation.min_answer_length
ANSWER_CONFIDENCE_THRESHOLD = config.generation.answer_confidence_threshold
USE_LLM_FALLBACK = config.generation.use_llm_fallback
BM25_K1 = config.search.bm25_k1
BM25_B = config.search.bm25_b
SEARCH_TOP_K = config.search.final_top_k
ENABLE_RERANKING = config.search.enable_reranking

# 로깅 설정
if config.debug_mode:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("[CONFIG] 디버그 모드 활성화")
    print(f"[CONFIG] BM25 파라미터: k1={BM25_K1}, b={BM25_B}")
    print(f"[CONFIG] 검색 설정: top_k={SEARCH_TOP_K}, BM25 가중치={WEIGHT_BM25}")
    print(f"[CONFIG] 생성 설정: 최대 길이={MAX_ANSWER_LENGTH}, 임계값={ANSWER_CONFIDENCE_THRESHOLD}")

# config.py 안 VectorConfig 정의 근처에 반영
class VectorConfig:
    provider: str = "pinecone"     # 그대로
    index_name: str = "codedoc-law-upstage"  # 쓰던 이름
    dimension: int = 4096          # ★ 중요: Upstage 임베딩 차원
    metric: str = "cosine"
    namespace: str = ""            # 쓰고 있으면 유지
