#!/usr/bin/env python
"""
금융·법령 RAG 평가 시스템
"""

import os
# 모든 import 전에 환경변수 설정
os.environ["DEBUG_HYBRID"] = "false"

import argparse
import sys
from pathlib import Path

from rag.utils import load_excel, save_results
from rag.evaluator import Evaluator
from rag import retriever

def debug_retrieve(query: str, top_k: int = 5):
    """디버그용 검색 결과 간단 출력"""
    results = retriever.retrieve(query, top_k=top_k)
    print(f"\n[DEBUG] '{query[:30]}...' -> {len(results)}개 검색")
    
    if results:
        top_score = results[0].get('score', 0)
        print(f"        최고 점수: {top_score:.3f}")
        if top_score < 0.5:
            print(f"        [WARNING] 낮은 검색 점수!")
    
    return results

def main():
    # 하이브리드 디버그 로그 기본적으로 비활성화
    import os
    os.environ["DEBUG_HYBRID"] = "false"
    
    parser = argparse.ArgumentParser(description="RAG 시스템 평가")
    parser.add_argument("file", help="평가할 Excel 파일")
    parser.add_argument("--mcq", type=int, default=None, help="사지선다형 개수")
    parser.add_argument("--short", type=int, default=None, help="단답형 개수")
    parser.add_argument("--workers", type=int, default=10, help="병렬 워커 수")
    parser.add_argument("--output", help="출력 파일명")
    parser.add_argument("--debug", action="store_true", help="검색 디버그 정보 출력")
    parser.add_argument("--verbose", action="store_true", help="하이브리드 검색 상세 로그 활성화")

    args = parser.parse_args()

    # 하이브리드 디버그 로그 제어 (--verbose로 활성화)
    if args.verbose:
        os.environ["DEBUG_HYBRID"] = "true"

    if not Path(args.file).exists():
        print(f"파일을 찾을 수 없습니다: {args.file}")
        sys.exit(1)

    print("\n" + "="*60)
    print(" RAG 평가 시스템")
    print("="*60)

    print(f"\n파일 로드: {args.file}")
    mcq_questions, short_questions = load_excel(args.file)

    if args.mcq is not None:
        mcq_questions = mcq_questions[:args.mcq]
    if args.short is not None:
        short_questions = short_questions[:args.short]

    print(f"사지선다형: {len(mcq_questions)}개")
    print(f"단답형: {len(short_questions)}개")

    evaluator = Evaluator(max_workers=args.workers)

    # 사지선다형 평가
    mcq_results = []
    if mcq_questions:
        print("\n[사지선다형 평가 시작]")
        try:
            result_data = evaluator.evaluate_all(mcq_questions, "mcq")
            if isinstance(result_data, tuple):
                mcq_results, mcq_stats = result_data
            else:
                mcq_results = result_data
        except Exception as e:
            print(f"사지선다형 평가 오류: {e}")
            mcq_results = []

    # 단답형 평가
    short_results = []
    if short_questions:
        print("\n[단답형 평가 시작]")
        
        # 디버그 모드일 때 10개씩 샘플링해서 검색 분석
        if args.debug:
            print("DEBUG 모드: 10개 단위로 검색 분석")
            debug_samples = [short_questions[i] for i in range(0, len(short_questions), 10)]
            for i, q in enumerate(debug_samples):
                print(f"\n--- 질문 {i*10+1}번째 샘플 ---")
                debug_retrieve(q["question"], top_k=5)
        
        try:
            result_data = evaluator.evaluate_all(short_questions, "short")
            if isinstance(result_data, tuple):
                short_results, short_stats = result_data
            else:
                short_results = result_data
        except Exception as e:
            print(f"단답형 평가 오류: {e}")
            short_results = []

    # 결과 저장
    if mcq_results or short_results:
        output_file = save_results(mcq_results, short_results, args.output)
        print(f"\n결과 저장됨: {output_file}")
    else:
        print("\n저장할 결과가 없습니다.")

    print("\n" + "="*60)
    print(" 평가 완료!")
    print("="*60)

if __name__ == "__main__":
    main()