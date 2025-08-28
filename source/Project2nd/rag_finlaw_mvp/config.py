# config.py - 정리된 설정 (스레드 안전) + API 키 검증 강화
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import threading

load_dotenv()

@dataclass
class Config:
    """정리된 전역 설정"""
    
    # 프로젝트 루트
    project_root: Path = Path(__file__).parent
    
    # 검색 설정
    mcq_top_k: int = 10
    short_top_k: int = 15
    vector_candidate_k: int = 50
    
    # BM25/Vector 경로
    bm25_index_path: str = field(default_factory=lambda: (
        os.getenv("BM25_PICKLE") 
        or str(Path(__file__).parent / "bm25_pkg" / "bm25_index.pkl")
    ))
    
    # LLM API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    upstage_api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Vector DB
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    
    # LLM 설정
    max_tokens: int = 1024
    llm_timeout: int = 60
    
    def validate(self) -> List[str]:
        """필수 설정 검증 - 강화된 버전"""
        errors = []
        warnings = []
        
        # 1. LLM API 키 검증 (강화)
        has_openai = bool(self.openai_api_key and len(self.openai_api_key.strip()) > 10)
        has_upstage = bool(self.upstage_api_key and len(self.upstage_api_key.strip()) > 10)
        
        if not has_openai and not has_upstage:
            errors.append("최소 하나의 LLM API 키가 필요합니다 (OpenAI 또는 Upstage)")
        else:
            if has_openai:
                if not self.openai_api_key.startswith(('sk-', 'sk-proj-')):
                    warnings.append("OpenAI API 키 형식이 의심스럽습니다")
                else:
                    print("[CONFIG-OK] OpenAI API 키 검증 통과")
            
            if has_upstage:
                if len(self.upstage_api_key.strip()) < 20:
                    warnings.append("Upstage API 키가 너무 짧습니다")
                else:
                    print("[CONFIG-OK] Upstage API 키 검증 통과")
        
        # 2. 검색 시스템 검증 (강화)
        has_bm25 = os.path.exists(self.bm25_index_path)
        has_pinecone = bool(self.pinecone_api_key and self.pinecone_index_name)
        
        if not has_bm25 and not has_pinecone:
            errors.append("BM25 인덱스 또는 Pinecone 중 하나는 필요합니다")
        else:
            if has_bm25:
                try:
                    # BM25 파일 크기 확인
                    bm25_size = os.path.getsize(self.bm25_index_path)
                    if bm25_size < 1024:  # 1KB 미만
                        warnings.append("BM25 인덱스 파일이 너무 작습니다")
                    else:
                        print(f"[CONFIG-OK] BM25 인덱스 검증 통과 ({bm25_size/1024/1024:.1f}MB)")
                except Exception as e:
                    warnings.append(f"BM25 인덱스 파일 검증 실패: {e}")
            
            if has_pinecone:
                if len(self.pinecone_index_name.strip()) < 3:
                    warnings.append("Pinecone 인덱스 이름이 너무 짧습니다")
                else:
                    print("[CONFIG-OK] Pinecone 설정 검증 통과")
        
        # 3. 파일 시스템 검증 (신규)
        if not os.access(self.project_root, os.R_OK):
            errors.append("프로젝트 루트 디렉토리 읽기 권한이 없습니다")
        
        if not os.access(self.project_root, os.W_OK):
            warnings.append("프로젝트 루트 디렉토리 쓰기 권한이 없습니다 (결과 저장 불가)")
        
        # 4. 네트워크 설정 검증 (신규)
        if self.llm_timeout < 10:
            warnings.append("LLM 타임아웃이 너무 짧습니다 (최소 10초 권장)")
        
        if self.max_tokens > 4000:
            warnings.append("max_tokens가 너무 큽니다 (비용 증가 주의)")
        
        # 5. 경고사항 출력
        if warnings:
            print("[CONFIG-WARNING] 설정 경고사항:")
            for warning in warnings:
                print(f"  - {warning}")
        
        return errors
    
    def get_available_llms(self) -> List[str]:
        """사용 가능한 LLM 목록 반환"""
        available = []
        
        if self.openai_api_key and len(self.openai_api_key.strip()) > 10:
            available.append("OpenAI")
        
        if self.upstage_api_key and len(self.upstage_api_key.strip()) > 10:
            available.append("Upstage")
        
        return available
    
    def get_available_retrievers(self) -> List[str]:
        """사용 가능한 검색기 목록 반환"""
        available = []
        
        if os.path.exists(self.bm25_index_path):
            available.append("BM25")
        
        if self.pinecone_api_key and self.pinecone_index_name:
            available.append("Pinecone")
        
        return available
    
    def is_production_ready(self) -> bool:
        """운영 환경 준비 상태 확인"""
        errors = self.validate()
        
        # 기본 요구사항
        if errors:
            return False
        
        # 운영 환경 추가 요구사항
        has_both_retrievers = (
            os.path.exists(self.bm25_index_path) and 
            bool(self.pinecone_api_key and self.pinecone_index_name)
        )
        
        has_multiple_llms = len(self.get_available_llms()) >= 2
        
        return has_both_retrievers and has_multiple_llms

# 전역 설정 인스턴스 (스레드 안전)
_config = None
_config_lock = threading.Lock()

def get_config():
    """스레드 안전한 설정 인스턴스 반환 - 검증 강화"""
    global _config
    
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = Config()
                errors = _config.validate()
                
                if errors:
                    # 오류 메시지 출력
                    print("[CONFIG-ERROR] 설정 검증 실패:")
                    for error in errors:
                        print(f"  - {error}")
                    
                    # 웹 환경 확인
                    try:
                        import streamlit as st
                        # 웹 환경에서도 심각한 오류는 예외 발생 (수정된 부분)
                        if any("API 키" in error for error in errors):
                            print("[CONFIG-CRITICAL] 웹 환경에서도 API 키 오류로 중단")
                            raise RuntimeError(f"필수 설정 누락: {', '.join(errors)}")
                        else:
                            print("[CONFIG-WARNING] 웹 환경에서 경고만 표시")
                    except ImportError:
                        # CLI 환경에서는 모든 오류에 대해 예외 발생
                        print("[CONFIG-CRITICAL] CLI 환경에서 설정 오류로 중단")
                        raise RuntimeError(f"Config 검증 실패: {', '.join(errors)}")
                
                # 설정 요약 출력
                print("[CONFIG-SUMMARY] 설정 요약:")
                print(f"  - 사용 가능한 LLM: {', '.join(_config.get_available_llms())}")
                print(f"  - 사용 가능한 검색기: {', '.join(_config.get_available_retrievers())}")
                print(f"  - 운영 준비 상태: {'OK' if _config.is_production_ready() else 'PARTIAL'}")
    
    return _config

def reset_config():
    """설정 인스턴스 리셋 (테스트용)"""
    global _config
    with _config_lock:
        _config = None

def validate_runtime_environment():
    """런타임 환경 검증 - 추가 검사"""
    try:
        config = get_config()
        
        # 1. 디스크 공간 확인
        import shutil
        free_space = shutil.disk_usage(config.project_root).free
        if free_space < 100 * 1024 * 1024:  # 100MB 미만
            print("[RUNTIME-WARNING] 디스크 여유 공간 부족 (100MB 미만)")
        
        # 2. 메모리 사용량 확인 (가능한 경우)
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                print("[RUNTIME-WARNING] 메모리 사용률 높음 (90% 초과)")
        except ImportError:
            pass
        
        # 3. 네트워크 연결 확인 (간단한 DNS 테스트)
        try:
            import socket
            socket.gethostbyname('google.com')
            print("[RUNTIME-OK] 네트워크 연결 정상")
        except Exception:
            print("[RUNTIME-WARNING] 네트워크 연결 문제 가능성")
        
        return True
        
    except Exception as e:
        print(f"[RUNTIME-ERROR] 런타임 환경 검증 실패: {e}")
        return False

# 모듈 레벨에서 기본 검증 실행
if __name__ == "__main__":
    print("=== Config 모듈 직접 실행 ===")
    config = get_config()
    
    print(f"\n=== 상세 정보 ===")
    print(f"프로젝트 루트: {config.project_root}")
    print(f"BM25 경로: {config.bm25_index_path}")
    print(f"BM25 존재: {os.path.exists(config.bm25_index_path)}")
    print(f"OpenAI 키: {'설정됨' if config.openai_api_key else '없음'}")
    print(f"Upstage 키: {'설정됨' if config.upstage_api_key else '없음'}")
    print(f"Pinecone 키: {'설정됨' if config.pinecone_api_key else '없음'}")
    
    print(f"\n=== 런타임 검증 ===")
    validate_runtime_environment()
    
    print(f"\n=== 최종 상태 ===")
    if config.is_production_ready():
        print("✓ 운영 환경 준비 완료")
    else:
        print("⚠ 부분적 설정만 가능")
else:
    # 모듈 임포트 시에는 기본 검증만
    try:
        get_config()
    except Exception as e:
        print(f"[CONFIG-INIT-ERROR] 설정 초기화 실패: {e}")