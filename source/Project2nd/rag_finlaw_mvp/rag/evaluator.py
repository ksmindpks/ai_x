# rag/evaluator.py
import concurrent.futures
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from .retriever import retrieve_batch
from .generator import generate_answer_short, generate_answer_mcq
from .utils import calculate_accuracy, postprocess_answer, score_short
from config import MAX_WORKERS, BATCH_SIZE

class Evaluator:
    """효율적인 평가 엔진"""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.stats = {
            "processed": 0,
            "errors": 0,
            "low_score_queries": 0,
            "start_time": None
        }
    
    def evaluate_batch(self, questions: List[Dict], question_type: str) -> List[Dict]:
        """배치 평가"""
        if not questions:
            return []
        
        # 배치 검색
        queries = [q["question"] for q in questions]
        all_contexts = retrieve_batch(queries, top_k=7, debug=False)
        
        # 검색 품질 체크
        low_score_count = 0
        for contexts in all_contexts:
            if not contexts:
                low_score_count += 1
            elif contexts[0].get('final_score', contexts[0].get('score', 0)) < 0.5:
                low_score_count += 1
                self.stats["low_score_queries"] += 1
        
        if low_score_count > len(questions) * 0.3:
            print(f"[경고] {low_score_count}/{len(questions)} 질문의 검색 품질이 낮음")
        
        # 병렬 답변 생성
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for q, contexts in zip(questions, all_contexts):
                if contexts:
                    best_score = contexts[0].get('final_score', contexts[0].get('score', 0))
                    if best_score < 0.4 and self.stats["processed"] % 50 == 0:
                        print(f"[디버그] 낮은 점수: {best_score:.3f}")
                
                if question_type == "mcq":
                    future = executor.submit(generate_answer_mcq, q["question"], q["choices"], contexts)
                else:
                    future = executor.submit(generate_answer_short, q["question"], contexts)
                futures.append((future, q, contexts))
            
            # 결과 수집
            results = []
            for future, q, contexts in futures:
                try:
                    prediction = future.result(timeout=30)
                    prediction = postprocess_answer(
                        prediction,
                        question_type,
                        contexts[0].get('text', '') if contexts else ""
                    )
                    
                    best_score = 0.0
                    if contexts:
                        best_score = contexts[0].get('final_score', contexts[0].get('score', 0))
                    
                    results.append({
                        "question": q["question"],
                        "prediction": prediction,
                        "answer": q["answer"],
                        "metadata": q.get("meta", {}),
                        "search_score": best_score
                    })
                    self.stats["processed"] += 1
                    
                except Exception as e:
                    results.append({
                        "question": q["question"],
                        "prediction": "(오류)",
                        "answer": q["answer"],
                        "metadata": q.get("meta", {}),
                        "search_score": 0.0
                    })
                    self.stats["errors"] += 1
            
            return results
    
    def evaluate_all(self, questions: List[Dict], question_type: str) -> Tuple[List[Dict], Dict]:
        """전체 평가 실행 - 이 메서드가 있어야 함!"""
        self.stats["start_time"] = time.time()
        
        type_name = "사지선다형" if question_type == "mcq" else "단답형"
        print(f"\n[{type_name} 평가 시작] {len(questions)}개 문제")
        print("-" * 50)
        
        # 배치 처리
        all_results = []
        batches = [questions[i:i+BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]
        
        with tqdm(total=len(questions), desc=f"{type_name} 평가") as pbar:
            for batch_idx, batch in enumerate(batches):
                if batch_idx > 0 and batch_idx % 10 == 0:
                    elapsed = time.time() - self.stats["start_time"]
                    speed = self.stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = (len(questions) - self.stats["processed"]) / speed if speed > 0 else 0
                    print(f"\n[진행] {self.stats['processed']}/{len(questions)} "
                          f"(속도: {speed:.1f}개/초, 예상 잔여: {remaining/60:.1f}분)")
                
                batch_results = self.evaluate_batch(batch, question_type)
                all_results.extend(batch_results)
                pbar.update(len(batch))
        
        # 검색 품질 통계
        avg_search_score = sum(r.get("search_score", 0) for r in all_results) / len(all_results) if all_results else 0
        low_score_results = sum(1 for r in all_results if r.get("search_score", 0) < 0.5)
        
        print("\n[검색 품질 통계]")
        print(f"  평균 검색 점수: {avg_search_score:.3f}")
        print(f"  낮은 점수 질문: {low_score_results}/{len(all_results)} ({low_score_results/len(all_results)*100:.1f}%)")
        
        # 성능 계산
        print("\n[평가 결과]")
        if question_type == "mcq":
            accuracy = calculate_accuracy(all_results)
            print(f"  정확도: {accuracy:.3f}")
            
            # 난이도별 분석
            difficulties = {}
            for r in all_results:
                diff = r.get("metadata", {}).get("difficulty", "미분류")
                if diff not in difficulties:
                    difficulties[diff] = {"correct": 0, "total": 0}
                difficulties[diff]["total"] += 1
                
                # 정규화하여 비교
                import re
                pred_norm = re.sub(r'\s+', ' ', str(r["prediction"]).strip().lower())
                ans_norm = re.sub(r'\s+', ' ', str(r["answer"]).strip().lower())
                if pred_norm == ans_norm:
                    difficulties[diff]["correct"] += 1
            
            if len(difficulties) > 1:
                print("\n  난이도별 정확도:")
                for diff, stats in sorted(difficulties.items()):
                    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"    {diff}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        else:
            # 단답형 평가
            em_scores = []
            f1_scores = []
            for r in all_results:
                em, f1 = score_short(r["prediction"], r["answer"])
                em_scores.append(em)
                f1_scores.append(f1)
            
            em_avg = sum(em_scores) / len(em_scores) if em_scores else 0
            f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
            print(f"  EM: {em_avg:.3f}")
            print(f"  F1: {f1_avg:.3f}")
            
            # 답변 길이 분석
            avg_pred_len = sum(len(str(r["prediction"]).split()) for r in all_results) / len(all_results) if all_results else 0
            avg_gold_len = sum(len(str(r["answer"]).split()) for r in all_results) / len(all_results) if all_results else 0
            
            print(f"\n  평균 답변 길이:")
            print(f"    예측: {avg_pred_len:.1f} 단어")
            print(f"    정답: {avg_gold_len:.1f} 단어")
        
        # 처리 통계
        elapsed = time.time() - self.stats["start_time"]
        print(f"\n[처리 통계]")
        print(f"  소요 시간: {elapsed/60:.1f}분")
        print(f"  처리 속도: {len(questions)/elapsed:.1f}개/초" if elapsed > 0 else "  처리 속도: N/A")
        print(f"  성공: {self.stats['processed']}개")
        print(f"  오류: {self.stats['errors']}개")
        
        return all_results, self.stats
    
    def analyze_errors(self, results: List[Dict], question_type: str, top_n: int = 10):
        """오류 분석"""
        print(f"\n[오류 분석 - 상위 {top_n}개]")
        print("-" * 50)
        
        errors = []
        for r in results:
            if question_type == "mcq":
                is_error = r["prediction"] != r["answer"]
            else:
                em, _ = score_short(r["prediction"], r["answer"])
                is_error = em == 0.0
            if is_error:
                errors.append(r)
        
        errors.sort(key=lambda x: x.get("search_score", 0))
        for i, err in enumerate(errors[:top_n], 1):
            print(f"\n[오류 {i}]")
            print(f"  질문: {err['question'][:60]}...")
            print(f"  예측: {err['prediction'][:50]}...")
            print(f"  정답: {err['answer'][:50]}...")
            print(f"  검색점수: {err.get('search_score', 0):.3f}")
            print(f"  난이도: {err.get('metadata', {}).get('difficulty', 'N/A')}")
