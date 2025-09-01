# config.py - 정리된 설정 (스레드 안전) + API 키 검증 강화 + 중복 로그 방지
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import threading

load_dotenv()

# 웹 로그 함수 추가 (단순)
_web_log_func = None
_pending_logs = []  # 웹 로그 함수 설정 전까지 임시 저장

def set_web_log_func(log_func):
    """웹 로그 함수 설정"""
    global _web_log_func, _pending_logs
    _web_log_func = log_func
    
    # 대기 중인 로그들을 모두 전송
    for pending_msg in _pending_logs:
        try:
            _web_log_func(pending_msg)
        except:
            pass
    _pending_logs = []  # 전송 완료 후 비우기

def config_print(message):
    """config 전용 print 함수"""
    global _pending_logs
    
    # 웹 로그로 전송 (설정되었으면 즉시, 아니면 대기열에 추가)
    if _web_log_func:
        try:
            _web_log_func(message)
        except:
            pass
    else:
        # streamlit 환경에서 add_to_mgmt_log 함수 직접 찾기 시도
        try:
            import streamlit as st
            if hasattr(st.session_state, 'mgmt_logs'):
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.mgmt_logs.append(f"[{timestamp}] {message}")
                if len(st.session_state.mgmt_logs) > 50:
                    st.session_state.mgmt_logs.pop(0)
            else:
                _pending_logs.append(message)
        except:
            _pending_logs.append(message)

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
    
    def validate(self, silent=False) -> List[str]:
        """필수 설정 검증 - 강화된 버전 + 조용한 모드 추가"""
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
                elif not silent:  # 조용한 모드가 아닐 때만 출력
                    config_print("[CONFIG-OK] OpenAI API 키 검증 통과")
            
            if has_upstage:
                if len(self.upstage_api_key.strip()) < 20:
                    warnings.append("Upstage API 키가 너무 짧습니다")
                elif not silent:  # 조용한 모드가 아닐 때만 출력
                    config_print("[CONFIG-OK] Upstage API 키 검증 통과")
        
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
                    elif not silent:  # 조용한 모드가 아닐 때만 출력
                        config_print(f"[CONFIG-OK] BM25 인덱스 검증 통과 ({bm25_size/1024/1024:.1f}MB)")
                except Exception as e:
                    warnings.append(f"BM25 인덱스 파일 검증 실패: {e}")
            
            if has_pinecone:
                if len(self.pinecone_index_name.strip()) < 3:
                    warnings.append("Pinecone 인덱스 이름이 너무 짧습니다")
                elif not silent:  # 조용한 모드가 아닐 때만 출력
                    config_print("[CONFIG-OK] Pinecone 설정 검증 통과")
        
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
        
        # 5. 경고사항 출력 (silent 모드가 아닐 때만)
        if warnings and not silent:
            config_print("[CONFIG-WARNING] 설정 경고사항:")
            for warning in warnings:
                config_print(f"  - {warning}")
        
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
        errors = self.validate(silent=True)  # 조용한 모드로 검증
        
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

# 전역 설정 인스턴스 (스레드 안전) + 로그 출력 여부 플래그
_config = None
_config_lock = threading.Lock()
_config_logged = False  # 로그가 이미 출력되었는지 추적

def get_config(silent=False):
    """스레드 안전한 설정 인스턴스 반환 - 중복 로그 방지"""
    global _config, _config_logged
    
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = Config()
                
                # 첫 번째 호출에서만 로그 출력
                should_log = not _config_logged and not silent
                if should_log:
                    _config_logged = True
                
                errors = _config.validate(silent=not should_log)
                
                if errors:
                    if should_log:
                        # 오류 메시지 출력 (앱 중단하지 않고 경고만)
                        config_print("[CONFIG-WARNING] 설정 검증 실패:")
                        for error in errors:
                            config_print(f"  - {error}")
                        config_print("[CONFIG-INFO] 부분적 기능만 사용 가능합니다")
                
                # 설정 요약 출력 (첫 번째 호출에서만)
                if should_log:
                    config_print("[CONFIG-SUMMARY] 설정 요약:")
                    config_print(f"  - 사용 가능한 LLM: {', '.join(_config.get_available_llms())}")
                    config_print(f"  - 사용 가능한 검색기: {', '.join(_config.get_available_retrievers())}")
                    config_print(f"  - 운영 준비 상태: {'OK' if _config.is_production_ready() else 'PARTIAL'}")
    
    return _config

def reset_config():
    """설정 인스턴스 리셋 (테스트용)"""
    global _config, _config_logged
    with _config_lock:
        _config = None
        _config_logged = False

def validate_runtime_environment():
    """런타임 환경 검증 - 추가 검사"""
    try:
        config = get_config(silent=True)  # 조용한 모드로 가져오기
        
        # 1. 디스크 공간 확인
        import shutil
        free_space = shutil.disk_usage(config.project_root).free
        if free_space < 100 * 1024 * 1024:  # 100MB 미만
            config_print("[RUNTIME-WARNING] 디스크 여유 공간 부족 (100MB 미만)")
        
        # 2. 메모리 사용량 확인 (가능한 경우)
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                config_print("[RUNTIME-WARNING] 메모리 사용률 높음 (90% 초과)")
        except ImportError:
            pass
        
        # 3. 네트워크 연결 확인 (간단한 DNS 테스트)
        try:
            import socket
            socket.gethostbyname('google.com')
            config_print("[RUNTIME-OK] 네트워크 연결 정상")
        except Exception:
            config_print("[RUNTIME-WARNING] 네트워크 연결 문제 가능성")
        
        return True
        
    except Exception as e:
        config_print(f"[RUNTIME-ERROR] 런타임 환경 검증 실패: {e}")
        return False

# 모듈 레벨에서 기본 검증 실행a
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
        config_print(f"[CONFIG-INIT-ERROR] 설정 초기화 실패: {e}")