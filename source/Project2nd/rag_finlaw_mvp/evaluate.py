"""
evaluate.py - 웹 호출 가능한 버전 (전역 인스턴스 접근 방식 통일)
CLI와 웹 인터페이스 모두 지원하는 평가 함수
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import traceback

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def get_rag_instances():
    """RAG 인스턴스 가져오기 - 웹/CLI 환경 통합"""
    try:
        # 웹 환경(Streamlit)에서 실행 중인지 확인
        import streamlit as st
        # app.py의 get_rag_system() 사용
        from app import get_rag_system
        return get_rag_system()
    except ImportError:
        # CLI 환경에서는 직접 생성
        print("[CLI-MODE] Streamlit 환경이 아님 - 직접 RAG 인스턴스 생성")
        try:
            from rag.hybrid_retriever import HybridRetriever
            from rag.llm_bridge import HybridLLM
            from config import get_config
            
            config = get_config()
            retriever = HybridRetriever(config)
            llm = HybridLLM(config)
            
            return retriever, llm, config
        except Exception as e:
            print(f"[CLI-ERROR] CLI 모드에서 RAG 생성 실패: {e}")
            return None, None, None
    except Exception as e:
        print(f"[WEB-ERROR] 웹 환경에서 RAG 가져오기 실패: {e}")
        return None, None, None

def run_evaluation(file_path: str, mcq_limit: int = None, short_limit: int = None, 
                  progress_callback=None) -> dict:
    """
    평가 실행 함수 - 웹과 CLI 모두에서 호출 가능 (인스턴스 접근 방식 통일)
    
    Args:
        file_path: 평가할 Excel 파일 경로
        mcq_limit: MCQ 문제 수 제한
        short_limit: 단답형 문제 수 제한  
        progress_callback: 진행상황 콜백 함수 (log_type, message)
    
    Returns:
        dict: 평가 결과 또는 None (실패시)
    """
    
    def log(log_type, message):
        """로그 출력 헬퍼"""
        if progress_callback:
            progress_callback(log_type, message)
        else:
            print(f"[{log_type.upper()}] {message}")
    
    try:
        log("progress", f"평가 시작: {Path(file_path).name}")
        
        # 1. 파일 검증
        log("progress", "파일 검증 중...")
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            log("failure", f"파일을 찾을 수 없습니다: {file_path}")
            return None
        log("progress", "파일 검증 완료")
        
        # 2. 통합된 RAG 인스턴스 가져오기 (수정된 부분)
        log("progress", "RAG 인스턴스 가져오는 중...")
        
        retriever, llm, config = get_rag_instances()
        
        if not retriever or not llm or not config:
            log("failure", "RAG 인스턴스 가져오기 실패")
            return None
        
        log("progress", "RAG 인스턴스 준비 완료")
        
        # 3. 평가기 초기화 (기존 인스턴스 전달)
        log("progress", "평가 시스템 초기화 중...")
        
        try:
            from rag.evaluator import UnifiedEvaluator
            evaluator = UnifiedEvaluator(retriever=retriever, llm=llm, config=config)
            log("progress", "평가 시스템 초기화 완료 (기존 인스턴스 재사용)")
            
        except Exception as e:
            log("failure", f"평가 시스템 초기화 실패: {e}")
            return None
        
        # 4. 평가 실행
        log("progress", "평가 실행 중...")
        
        try:
            # progress_callback 지원하는지 확인
            import inspect
            evaluate_method = getattr(evaluator, 'evaluate_file')
            sig = inspect.signature(evaluate_method)
            
            if 'progress_callback' in sig.parameters:
                # progress_callback 지원
                results = evaluator.evaluate_file(
                    file_path, 
                    mcq_limit, 
                    short_limit,
                    progress_callback=lambda msg: log("progress", msg)
                )
            else:
                # progress_callback 미지원
                log("progress", "progress_callback 미지원 - 기본 평가 실행")
                results = evaluator.evaluate_file(
                    file_path, 
                    mcq_limit, 
                    short_limit
                )
            
            if not results:
                log("failure", "평가할 데이터가 없습니다")
                return None
            
            log("progress", "평가 실행 완료")
            
        except Exception as e:
            log("failure", f"평가 실행 실패: {e}")
            log("failure", f"상세 오류: {traceback.format_exc()}")
            return None
        
        # 5. 결과 저장
        log("progress", "결과 저장 중...")
        
        try:
            saved_file = evaluator.save_results(results, None, file_path)
            
            if saved_file:
                log("success", f"결과 저장 완료: {Path(saved_file).name}")
                
                # 성공/실패 통계
                total_questions = len(results.get('mcq_results', [])) + len(results.get('short_results', []))
                mcq_success = sum(1 for r in results.get('mcq_results', []) if r.get('is_correct', False))
                short_success = sum(1 for r in results.get('short_results', []) if r.get('em_score', 0) > 0)
                total_success = mcq_success + short_success
                
                log("success", f"전체 {total_questions}문제 중 {total_success}문제 성공")
                
                if results.get('mcq_results'):
                    log("success", f"MCQ: {mcq_success}/{len(results['mcq_results'])} 성공")
                
                if results.get('short_results'):
                    log("success", f"단답형: {short_success}/{len(results['short_results'])} 성공")
                
                # 최종 결과에 saved_file 경로 추가
                results['saved_file'] = saved_file
                return results
                
            else:
                log("failure", "결과 저장 실패")
                return None
                
        except Exception as e:
            log("failure", f"결과 저장 실패: {e}")
            return None
            
    except Exception as e:
        log("failure", f"예상치 못한 오류: {e}")
        log("failure", f"상세 오류: {traceback.format_exc()}")
        return None

def run_enhanced_evaluation(file_path: str, mcq_limit: int = None, short_limit: int = None) -> bool:
    """
    CLI용 평가 실행 (기존 함수명 유지)
    
    Returns:
        bool: 성공 여부
    """
    results = run_evaluation(file_path, mcq_limit, short_limit)
    return results is not None

def validate_environment():
    """실행 환경 검증"""
    try:
        # 필수 모듈 확인
        from config import get_config
        config = get_config()
        
        errors = config.validate()
        if errors:
            print("[ENV-ERROR] 설정 검증 실패:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("[ENV-OK] 환경 검증 완료")
        return True
        
    except Exception as e:
        print(f"[ENV-ERROR] 환경 검증 중 오류: {e}")
        return False

def main():
    """메인 함수 - CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="RAG 시스템 평가 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python evaluate.py test.xlsx --mcq 5 --short 5
  python evaluate.py test.xlsx --mcq 10 --debug
  python evaluate.py test.xlsx --short 20
        """
    )
    
    parser.add_argument('file', nargs='?', help='평가할 Excel 파일 경로')
    parser.add_argument('--mcq', type=int, help='MCQ 문제 수 제한')
    parser.add_argument('--short', type=int, help='단답형 문제 수 제한')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    parser.add_argument('--validate-only', action='store_true', help='환경 검증만 실행')
    
    args = parser.parse_args()
    
    # 환경 검증만 실행
    if args.validate_only:
        success = validate_environment()
        sys.exit(0 if success else 1)
    
    if not args.file:
        print("오류: 평가할 파일 경로가 필요합니다.")
        print("사용법: python evaluate.py <excel_file> [--mcq N] [--short N]")
        print("도움말: python evaluate.py --help")
        sys.exit(1)
    
    # 디버그 모드 설정
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("[DEBUG] 디버그 모드 활성화")
    
    # 환경 검증
    print("[CLI] 환경 검증 중...")
    if not validate_environment():
        print("[CLI] 환경 검증 실패 - 실행을 중단합니다.")
        sys.exit(1)
    
    try:
        print(f"[CLI] 평가 시작: {args.file}")
        print(f"[CLI] MCQ 제한: {args.mcq if args.mcq else '제한없음'}")
        print(f"[CLI] 단답형 제한: {args.short if args.short else '제한없음'}")
        
        success = run_enhanced_evaluation(
            file_path=args.file,
            mcq_limit=args.mcq,
            short_limit=args.short
        )
        
        if success:
            print("\n[CLI] 평가가 성공적으로 완료되었습니다!")
            print("[CLI] 결과 파일을 확인하세요.")
        else:
            print("\n[CLI] 평가 실패!")
            print("[CLI] 오류 로그를 확인하고 다시 시도하세요.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n[CLI] 평가가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[CLI] 예상치 못한 오류 발생: {e}")
        if args.debug:
            import traceback
            print("[CLI-DEBUG] 상세 오류:")
            traceback.print_exc()
        sys.exit(1)

# 웹에서 import할 수 있도록 함수들을 모듈 레벨에서 노출
__all__ = ['run_evaluation', 'run_enhanced_evaluation', 'get_rag_instances', 'validate_environment']

if __name__ == "__main__":
    main()