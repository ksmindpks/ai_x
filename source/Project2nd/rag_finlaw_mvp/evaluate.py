#!/usr/bin/env python
"""
금융·법령 RAG 평가 시스템 (하이브리드 검색 점수 디버그 버전)
사용법: python evaluate.py [파일경로] [옵션]
"""

import argparse
import sys
from pathlib import Path

from rag.utils import load_excel, save_results
from rag.evaluator import Evaluator
from rag import retriever  # 하이브리드 retriever
from config import DEBUG_HYBRID

def debug_retrieve(query: str, top_k: int = 5):
    """상세 디버그 정보 출력"""
    results = retriever.retrieve(query, top_k=top_k)
    print(f"\n[DEBUG] Query: '{query[:50]}...'")
    print(f"[Results] Found: {len(results)}")
    
    for i, r in enumerate(results[:3], 1):
        print(f"  [{i}] Score: {r.get('score', 0):.3f}")
        if 'bm25_score_raw' in r:
            print(f"      BM25: raw={r['bm25_score_raw']:.3f}, norm={r.get('bm25_score_norm', 0):.3f}")
        if 'vec_score_raw' in r:
            print(f"      Vec:  raw={r['vec_score_raw']:.3f}, norm={r.get('vec_score_norm', 0):.3f}")
        print(f"      Text: {r['text'][:100]}...")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="RAG 시스템 평가 (디버그)")
    parser.add_argument("file", help="평가할 Excel 파일")
    parser.add_argument("--mcq", type=int, default=None, help="사지선다형 개수")
    parser.add_argument("--short", type=int, default=None, help="단답형 개수")
    parser.add_argument("--workers", type=int, default=10, help="병렬 워커 수")
    parser.add_argument("--output", help="출력 파일명")
    parser.add_argument("--debug", action="store_true", help="하이브리드 점수 로그 출력")

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"파일을 찾을 수 없습니다: {args.file}")
        sys.exit(1)

    print("\n" + "="*60)
    print(" RAG 평가 시스템 (하이브리드 디버그)")
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

    mcq_results = []
    if mcq_questions:
        # 언패킹 오류 수정
        mcq_results, stats = evaluator.evaluate_all(mcq_questions, "mcq")
        # 또는
        result_tuple = evaluator.evaluate_all(mcq_questions, "mcq")
        mcq_results = result_tuple[0]
        stats = result_tuple[1]

    short_results = []
    if short_questions:
        if args.debug:
            for q in short_questions:
                debug_retrieve(q["question"], top_k=7)
        short_results, _ = evaluator.evaluate_all(short_questions, "short")

    output_file = save_results(mcq_results, short_results, args.output)

    print("\n" + "="*60)
    print(" 평가 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
