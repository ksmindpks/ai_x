# evaluate_cli.py
import argparse
import sys
import os
from datetime import datetime
import pandas as pd
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.utils import load_multiple_excels, em_f1, mcq_acc
from rag.retriever import retrieve
from rag.generator import answer_short_extract, answer_mcq
from config import VAL_EXCEL_FILE1, VAL_EXCEL_FILE2

def parse_args():
    ap = argparse.ArgumentParser(description="금융·법령 RAG 시스템 평가 (다중 파일)")
    
    # 파일 지정
    ap.add_argument("--files", nargs='+', default=[VAL_EXCEL_FILE1, VAL_EXCEL_FILE2],
                   help="평가할 Excel 파일들 (기본: config의 FILE1, FILE2)")
    
    # 평가 개수 (파일당)
    ap.add_argument("--mcq_per_file", type=int, default=50,
                   help="각 파일당 사지선다형 평가 개수 (기본: 50)")
    ap.add_argument("--short_per_file", type=int, default=30,
                   help="각 파일당 단답형 평가 개수 (기본: 30)")
    
    # 평가 유형
    ap.add_argument("--type", choices=["mcq", "short", "both"], default="both",
                   help="평가할 문제 유형 (기본: both)")
    
    # 기타 옵션
    ap.add_argument("--verbose", action="store_true",
                   help="상세 출력 모드")
    ap.add_argument("--save_results", action="store_true",
                   help="결과를 Excel 파일로 저장")
    ap.add_argument("--show_source", action="store_true",
                   help="각 문제의 출처 파일 표시")
    
    return ap.parse_args()

def evaluate_questions(questions: List[Dict], question_type: str, 
                       verbose: bool = False, show_source: bool = False):
    """
    문제 리스트 평가
    
    Args:
        questions: 평가할 문제 리스트
        question_type: "mcq" 또는 "short"
        verbose: 상세 출력 여부
        show_source: 출처 파일 표시 여부
    """
    if not questions:
        print(f"[WARNING] 평가할 {question_type} 문제가 없습니다.")
        return []
    
    type_name = "사지선다형" if question_type == "mcq" else "단답형"
    print(f"\n{'='*60}")
    print(f"[{type_name} 평가 시작] 총 {len(questions)}개")
    print(f"{'='*60}")
    
    results = []
    
    if question_type == "mcq":
        s_acc = 0
        
        for i, q in enumerate(questions, 1):
            if verbose:
                source_info = f" [{os.path.basename(q['source_file'])}]" if show_source else ""
                print(f"\n[{i}/{len(questions)}]{source_info}")
                print(f"Q: {q['question'][:50]}...")
                print(f"보기: {[c[:20]+'...' if len(c)>20 else c for c in q['choices']]}")
            
            # 검색 & 답변 생성
            hits = retrieve(q["question"], filters=None, top_k=5)
            out = answer_mcq(q["question"], q.get("choices", []), hits)
            
            pred = ""
            for line in out.splitlines():
                if line.strip().startswith("정답:"):
                    pred = line.split("정답:", 1)[1].strip()
                    break
            
            # 평가
            acc = mcq_acc(pred, q["answer"])
            s_acc += acc
            
            results.append({
                "번호": i,
                "출처": os.path.basename(q['source_file']),
                "질문": q["question"],
                "예측": pred,
                "정답": q["answer"],
                "정확도": "O" if acc else "X",
                "난이도": q.get("meta", {}).get("difficulty", ""),
                "법령": q.get("meta", {}).get("law", "")
            })
            
            if verbose:
                print(f"예측: {pred[:50]}...")
                print(f"정답: {q['answer'][:50]}...")
                print(f"결과: {'[정답]' if acc else '[오답]'}")
        
        # 전체 결과
        n = len(questions)
        print(f"\n[{type_name} 결과]")
        print(f"정확도: {s_acc/n:.3f} ({int(s_acc)}/{n})")
        
    else:  # short
        s_em = s_f1 = 0
        
        for i, q in enumerate(questions, 1):
            if verbose:
                source_info = f" [{os.path.basename(q['source_file'])}]" if show_source else ""
                print(f"\n[{i}/{len(questions)}]{source_info}")
                print(f"Q: {q['question'][:50]}...")
            
            # 검색 & 답변 생성
            hits = retrieve(q["question"], filters=None, top_k=5)
            out = answer_short_extract(q["question"], hits)
            
            pred = ""
            for line in out.splitlines():
                if line.strip().startswith("정답:"):
                    pred = line.split("정답:", 1)[1].strip()
                    break
            
            # 평가
            em, f1 = em_f1(pred, q["answer"])
            s_em += em
            s_f1 += f1
            
            results.append({
                "번호": i,
                "출처": os.path.basename(q['source_file']),
                "질문": q["question"],
                "예측": pred,
                "정답": q["answer"],
                "EM": em,
                "F1": f1,
                "난이도": q.get("meta", {}).get("difficulty", ""),
                "법령": q.get("meta", {}).get("law", "")
            })
            
            if verbose:
                print(f"예측: {pred[:50]}...")
                print(f"정답: {q['answer'][:50]}...")
                print(f"점수: EM={em:.2f}, F1={f1:.3f}")
        
        # 전체 결과
        n = len(questions)
        print(f"\n[{type_name} 결과]")
        print(f"Exact Match: {s_em/n:.3f} ({int(s_em)}/{n})")
        print(f"F1 Score: {s_f1/n:.3f}")
    
    # 파일별 통계 (verbose 모드)
    if verbose and show_source:
        print(f"\n파일별 성능:")
        source_files = list(set(q['source_file'] for q in questions))
        for source in source_files:
            source_results = [r for r in results if r["출처"] == os.path.basename(source)]
            if question_type == "mcq":
                acc = sum(1 for r in source_results if r["정확도"] == "O") / len(source_results)
                print(f"  {os.path.basename(source)}: ACC={acc:.3f} (n={len(source_results)})")
            else:
                em = sum(r["EM"] for r in source_results) / len(source_results)
                f1 = sum(r["F1"] for r in source_results) / len(source_results)
                print(f"  {os.path.basename(source)}: EM={em:.3f}, F1={f1:.3f} (n={len(source_results)})")
    
    # 난이도별 통계
    if verbose:
        print(f"\n난이도별 성능:")
        for difficulty in ["상", "중", "하"]:
            d_results = [r for r in results if r["난이도"] == difficulty]
            if d_results:
                if question_type == "mcq":
                    acc = sum(1 for r in d_results if r["정확도"] == "O") / len(d_results)
                    print(f"  {difficulty}: ACC={acc:.3f} (n={len(d_results)})")
                else:
                    em = sum(r["EM"] for r in d_results) / len(d_results)
                    f1 = sum(r["F1"] for r in d_results) / len(d_results)
                    print(f"  {difficulty}: EM={em:.3f}, F1={f1:.3f} (n={len(d_results)})")
    
    return results

def save_results(mcq_results: List[Dict], short_results: List[Dict]):
    """결과를 Excel 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if mcq_results:
            df_mcq = pd.DataFrame(mcq_results)
            df_mcq.to_excel(writer, sheet_name="사지선다형", index=False)
        
        if short_results:
            df_short = pd.DataFrame(short_results)
            df_short.to_excel(writer, sheet_name="단답형", index=False)
        
        # 요약 통계
        summary = []
        if mcq_results:
            n = len(mcq_results)
            acc = sum(1 for r in mcq_results if r["정확도"] == "O") / n
            summary.append({
                "유형": "사지선다형",
                "총 문제": n,
                "정확도": f"{acc:.3f}",
                "정답": sum(1 for r in mcq_results if r["정확도"] == "O"),
                "오답": sum(1 for r in mcq_results if r["정확도"] == "X")
            })
        
        if short_results:
            n = len(short_results)
            em = sum(r["EM"] for r in short_results) / n
            f1 = sum(r["F1"] for r in short_results) / n
            summary.append({
                "유형": "단답형",
                "총 문제": n,
                "EM": f"{em:.3f}",
                "F1": f"{f1:.3f}",
                "정답": int(sum(r["EM"] for r in short_results))
            })
        
        if summary:
            df_summary = pd.DataFrame(summary)
            df_summary.to_excel(writer, sheet_name="요약", index=False)
    
    print(f"\n[결과 저장 완료] {output_file}")

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print(" 금융·법령 RAG 시스템 평가")
    print("="*60)
    print(f"\n[설정]")
    print(f"  평가 파일: {len(args.files)}개")
    for i, f in enumerate(args.files, 1):
        print(f"    {i}. {os.path.basename(f)}")
    print(f"  파일당 사지선다: {args.mcq_per_file}개")
    print(f"  파일당 단답: {args.short_per_file}개")
    print(f"  평가 유형: {args.type}")
    print(f"  상세 출력: {'Yes' if args.verbose else 'No'}")
    print(f"  결과 저장: {'Yes' if args.save_results else 'No'}")
    
    # 모든 파일에서 문제 로드
    all_mcq, all_short = load_multiple_excels(
        args.files,
        args.mcq_per_file if args.type in ["mcq", "both"] else 0,
        args.short_per_file if args.type in ["short", "both"] else 0,
        verbose=args.verbose
    )
    
    print(f"\n[총 로드된 문제]")
    print(f"  사지선다형: {len(all_mcq)}개")
    print(f"  단답형: {len(all_short)}개")
    
    mcq_results = []
    short_results = []
    
    # 사지선다형 평가
    if args.type in ["mcq", "both"] and all_mcq:
        mcq_results = evaluate_questions(
            all_mcq, "mcq", 
            args.verbose, 
            args.show_source
        )
    
    # 단답형 평가
    if args.type in ["short", "both"] and all_short:
        short_results = evaluate_questions(
            all_short, "short", 
            args.verbose, 
            args.show_source
        )
    
    # 결과 저장
    if args.save_results and (mcq_results or short_results):
        save_results(mcq_results, short_results)
    
    print("\n" + "="*60)
    print(" [평가 완료]")
    print("="*60)

if __name__ == "__main__":
    main()