import concurrent.futures
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from .retriever import retrieve_batch
from .generator import generate_answer_short, generate_answer_mcq
from .utils import calculate_accuracy, score_short
from config import MAX_WORKERS, BATCH_SIZE, DEBUG_HYBRID

class Evaluator:
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.stats = {"processed": 0, "errors": 0, "low_score_queries": 0, "start_time": None}

    def postprocess_answer_simple(self, prediction: str, question_type: str, context_text: str = "") -> str:
        if not prediction or prediction == "(오류)":
            return "정보 없음"
        prediction = str(prediction).strip()
        if len(prediction) > 100:
            prediction = prediction[:100]
        return prediction

    def evaluate_batch(self, questions: List[Dict], question_type: str) -> List[Dict]:
        if not questions:
            return []

        queries = [q["question"] for q in questions]
        all_contexts = retrieve_batch(queries, top_k=7, debug=False)

        low_score_count = 0
        for contexts in all_contexts:
            if not contexts:
                low_score_count += 1
            elif contexts[0].get('final_score', contexts[0].get('score', 0)) < 0.5:
                low_score_count += 1
                self.stats["low_score_queries"] += 1
        if low_score_count > len(questions) * 0.3:
            print(f"[경고] {low_score_count}/{len(questions)} 질문의 검색 품질이 낮음")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = []
            for q, contexts in zip(questions, all_contexts):
                if question_type == "mcq":
                    fut = ex.submit(generate_answer_mcq, q["question"], q["choices"], contexts)
                else:
                    fut = ex.submit(generate_answer_short, q["question"], contexts)
                futures.append((fut, q, contexts))

            for fut, q, contexts in futures:
                try:
                    prediction = fut.result(timeout=45)  # 살짝 여유
                except Exception as e:
                    print(f"[ERROR] 답변 생성 실패: {str(e)[:120]}")
                    prediction = "(오류)"
                    self.stats["errors"] += 1

                prediction = self.postprocess_answer_simple(
                    prediction, question_type, contexts[0].get('text', '') if contexts else ""
                )
                best_score = 0.0
                if contexts:
                    best_score = contexts[0].get('final_score', contexts[0].get('score', 0))

                results.append({
                    "question": q["question"],
                    "prediction": prediction,
                    "answer": q.get("answer", ""),
                    "metadata": q.get("meta", {}),
                    "search_score": best_score
                })
                self.stats["processed"] += 1

        return results

    def evaluate_all(self, questions: List[Dict], question_type: str) -> Tuple[List[Dict], Dict]:
        self.stats["start_time"] = time.time()
        type_name = "사지선다형" if question_type == "mcq" else "단답형"
        print(f"\n[{type_name} 평가 시작] {len(questions)}개 문제")
        print("-" * 50)

        all_results = []
        batches = [questions[i:i+BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]

        with tqdm(total=len(questions), desc=f"{type_name} 평가") as pbar:
            for batch_idx, batch in enumerate(batches):
                if batch_idx > 0 and self.stats["processed"] % 10 == 0:
                    elapsed = time.time() - self.stats["start_time"]
                    speed = self.stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = (len(questions) - self.stats["processed"]) / speed if speed > 0 else 0
                    recent = all_results[-10:] if len(all_results) >= 10 else all_results
                    avg_recent_score = sum(r.get("search_score", 0) for r in recent) / len(recent) if recent else 0
                    print(f"\n[진행 {self.stats['processed']:3d}/{len(questions)}] "
                          f"속도: {speed:.1f}/초, 예상잔여: {remaining/60:.1f}분, 최근검색점수: {avg_recent_score:.3f}")

                batch_results = self.evaluate_batch(batch, question_type)
                all_results.extend(batch_results)
                pbar.update(len(batch))

        avg_search = sum(r.get("search_score", 0) for r in all_results) / len(all_results) if all_results else 0
        low_cnt = sum(1 for r in all_results if r.get("search_score", 0) < 0.5)
        print("\n[검색 품질 통계]")
        print(f"  평균 검색 점수: {avg_search:.3f}")
        print(f"  낮은 점수 질문: {low_cnt}/{len(all_results)} ({(low_cnt/len(all_results)*100 if all_results else 0):.1f}%)")

        print("\n[평가 결과]")
        if question_type == "mcq":
            acc = calculate_accuracy(all_results)
            print(f"  정확도: {acc:.3f}")
            difficulties = {}
            for r in all_results:
                diff = r.get("metadata", {}).get("difficulty", "미분류")
                if diff not in difficulties:
                    difficulties[diff] = {"correct": 0, "total": 0}
                difficulties[diff]["total"] += 1
                import re
                pred_norm = re.sub(r'\s+', ' ', str(r["prediction"]).strip().lower())
                ans_norm  = re.sub(r'\s+', ' ', str(r["answer"]).strip().lower())
                if pred_norm == ans_norm:
                    difficulties[diff]["correct"] += 1
            if len(difficulties) > 1:
                print("\n  난이도별 정확도:")
                for diff, st in sorted(difficulties.items()):
                    acc_d = st["correct"]/st["total"] if st["total"]>0 else 0
                    print(f"    {diff}: {acc_d:.3f} ({st['correct']}/{st['total']})")
        else:
            ems, f1s = [], []
            for r in all_results:
                em, f1 = score_short(r["prediction"], r["answer"])
                ems.append(em); f1s.append(f1)
            print(f"  EM: {(sum(ems)/len(ems) if ems else 0):.3f}")
            print(f"  F1: {(sum(f1s)/len(f1s) if f1s else 0):.3f}")
            avg_pred_len = sum(len(str(r["prediction"]).split()) for r in all_results)/len(all_results) if all_results else 0
            avg_gold_len = sum(len(str(r["answer"]).split()) for r in all_results)/len(all_results) if all_results else 0
            print(f"\n  평균 답변 길이:")
            print(f"    예측: {avg_pred_len:.1f} 단어")
            print(f"    정답: {avg_gold_len:.1f} 단어")

        elapsed = time.time() - self.stats["start_time"]
        print(f"\n[처리 통계]")
        print(f"  소요 시간: {elapsed/60:.1f}분")
        print(f"  처리 속도: {len(questions)/elapsed:.1f}개/초" if elapsed>0 else "  처리 속도: N/A")
        print(f"  성공: {self.stats['processed']}개")
        print(f"  오류: {self.stats['errors']}개")

        return all_results, self.stats
