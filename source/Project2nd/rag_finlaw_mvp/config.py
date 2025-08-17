# config.py - 성능 최적화된 설정 (개선 버전)
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PerformanceConfig:
    """성능 관련 설정 - 개선됨"""
    # LLM 동시성 최적화
    openai_concurrency: int = 20  # 32 -> 20으로 안정성 향상
    max_retry_attempts: int = 2   # 3 -> 2로 속도 향상
    retry_base_delay: float = 0.1  # 0.15 -> 0.1로 단축
    request_timeout: int = 25      # 20 -> 25로 안정성 향상
    
    # 검색 최적화
    batch_size: int = 32          # 64 -> 32로 메모리 효율성
    max_workers: int = min(os.cpu_count() * 2, 16)  # CPU 코어수 * 2
    
    # 메모리 최적화
    max_context_length: int = 2000  # 2500 -> 2000으로 속도 향상
    chunk_size: int = 800          # 1000 -> 800으로 효율성
    enable_memory_mapping: bool = True
    enable_gpu_acceleration: bool = True

@dataclass
class SearchConfig:
    """검색 관련 설정 최적화 - 개선됨"""
    # 하이브리드 검색 최적화
    top_k_bm25: int = 8           # 10 -> 8
    top_k_vector: int = 20        # 25 -> 20
    weight_bm25: float = 0.3      # 0.3 -> 0.4 (BM25 비중 증가)
    weight_vector: float = 0.7    # 0.7 -> 0.6
    
    # 검색 성능 향상
    enable_parallel_search: bool = True
    cache_embeddings: bool = True
    use_approximate_search: bool = True
    enable_query_expansion: bool = False  # True -> False (성능 우선)
    enable_reranking: bool = False       # True -> False (성능 우선)
    
    # 품질 설정 (더 엄격하게)
    similarity_threshold: float = 0.15   # 0.1 -> 0.15
    diversity_penalty: float = 0.05      # 0.1 -> 0.05
    freshness_boost: float = 0.02        # 0.05 -> 0.02

@dataclass
class GenerationConfig:
    """생성 관련 설정 - 개선됨"""
    # 모델 설정
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0      # 0.1 -> 0.0 (더 일관성 있게)
    max_tokens: int = 80          # 50 -> 80 (적절한 길이)
    top_p: float = 0.9            # 0.8 -> 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    
    # 프롬프트 최적화
    enable_few_shot: bool = False      # True -> False (성능 우선)
    enable_chain_of_thought: bool = False  # True -> False
    context_window_size: int = 3

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
    
    # 캐싱 설정 (개선됨)
    enable_result_caching: bool = True
    cache_ttl: int = 1800  # 3600 -> 1800 (30분)
    
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
    raise ValueError(f"설정 오류: {', '.join(validation_errors)}")

# 호환성을 위한 기존 변수들
OPENAI_API_KEY = config.openai_api_key
PINECONE_API_KEY = config.pinecone_api_key
PINECONE_HOST = config.pinecone_host
PINECONE_NAMESPACE = config.pinecone_namespace
PINECONE_INDEX_NAME = config.pinecone_index_name
EMBED_MODEL = config.embed_model
EMBED_DIM = config.embed_dim
GENERATION_MODEL = config.generation_model
DEFAULT_TOP_K = 7
MAX_WORKERS = config.performance.max_workers
BATCH_SIZE = config.performance.batch_size
BM25_INDEX_DIR = config.bm25_index_dir
BM25_TOKENIZER = config.bm25_tokenizer
TOP_K_BM25 = config.search.top_k_bm25
TOP_K_VEC = config.search.top_k_vector
WEIGHT_BM25 = config.search.weight_bm25
WEIGHT_VEC = config.search.weight_vector
DEBUG_HYBRID = config.debug_mode