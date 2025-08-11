# evaluate_cli_turbo.py (새 파일)
import sys
import os
from datetime import datetime
import pandas as pd
from typing import List, Dict
import concurrent.futures
from functools import partial
from tqdm import tqdm
import multiprocessing
import asyncio
import aiohttp
from collections import deque
import psutil
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.utils import load_multiple_excels, em_f1, mcq_acc
from rag.retriever import retrieve
from rag.generator import answer_short_extract, answer_mcq
from config import VAL_EXCEL_FILE1, VAL_EXCEL_FILE2


# ===== 최적화 설정 =====
# CPU 코어 수에 따라 자동 조정
CPU_COUNT = multiprocessing.cpu_count()
OPTIMAL_WORKERS = min(CPU_COUNT * 2, 20)  # CPU 코어의 2배, 최대 20

# API Rate Limit 고려
OPENAI_RATE_LIMIT = 500  # 분당 요청 수 (티어에 따라 조정)
BATCH_SIZE = 50  # 한 번에 처리할 배치 크기

print(f"[시스템 정보]")
print(f"  CPU 코어: {CPU_COUNT}개")
print(f"  사용 가능 메모리: {psutil.virtual_memory().available / (1024**3):.1f}GB")
print(f"  최적 워커 수: {OPTIMAL_WORKERS}개")


class RateLimiter:
    """API Rate Limiting 관리"""
    def __init__(self, max_per_minute=500):
        self.max_per_minute = max_per_minute
        self.queue = deque()
        
    async def acquire(self):
        now = time.time()
        # 1분 이전 요청 제거
        while self.queue and self.queue[0] < now - 60:
            self.queue.popleft()
        
        # Rate limit 대기
        if len(self.queue) >= self.max_per_minute:
            sleep_time = 60 - (now - self.queue[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.queue.append(now)


class TurboEvaluator:
    """초고속 평가 클래스"""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or OPTIMAL_WORKERS
        self.rate_limiter = RateLimiter(OPENAI_RATE_LIMIT)
        self.cache = {}  # 결과 캐시
        
    def evaluate_single_cached(self, q, question_type):
        """캐시를 활용한 단일 문제 평가"""
        # 캐시 키 생성
        cache_key = f"{question_type}_{q['question'][:100]}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 검색 (최소화)
            hits = retrieve(q["question"], filters=None, top_k=3)  # top_k 최소화
            
            if question_type == "mcq":
                out = answer_mcq(q["question"], q.get("choices", []), hits)
            else:
                out = answer_short_extract(q["question"], hits)
            
            pred = ""
            for line in out.splitlines():
                if line.strip().startswith("정답:"):
                    pred = line.split("정답:", 1)[1].strip()
                    break
            
            result = {
                "question": q,
                "prediction": pred,
                "success": True
            }
            
            # 캐시 저장
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            return {
                "question": q,
                "prediction": "(오류)",
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, batch, question_type, pbar):
        """배치 처리"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.evaluate_single_cached, q, question_type)
                for q in batch
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
        
        return results
    
    def evaluate_all(self, questions, question_type):
        """전체 평가 - 배치 처리"""
        if not questions:
            return []
        
        type_name = "사지선다형" if question_type == "mcq" else "단답형"
        print(f"\n[{type_name}] {len(questions)}개 처리 시작")
        
        # 배치로 나누기
        batches = [questions[i:i + BATCH_SIZE] 
                  for i in range(0, len(questions), BATCH_SIZE)]
        
        all_results = []
        
        # 진행률 표시
        with tqdm(total=len(questions), desc=f"{type_name} 평가") as pbar:
            for batch_idx, batch in enumerate(batches):
                # 배치 처리
                batch_results = self.process_batch(batch, question_type, pbar)
                all_results.extend(batch_results)
                
                # 메모리 관리
                if batch_idx % 10 == 0:
                    import gc
                    gc.collect()
        
        return self.process_results(all_results, question_type)
    
    def process_results(self, raw_results, question_type):
        """결과 후처리"""
        processed = []
        
        for i, r in enumerate(raw_results, 1):
            if not r['success']:
                continue
                
            q = r['question']
            pred = r['prediction']
            
            if question_type == "mcq":
                acc = mcq_acc(pred, q["answer"])
                processed.append({
                    "번호": i,
                    "질문": q["question"][:100],  # 메모리 절약
                    "예측": pred[:50],
                    "정답": q["answer"][:50],
                    "정확도": "O" if acc else "X",
                    "난이도": q.get("meta", {}).get("difficulty", ""),
                })
            else:
                em, f1 = em_f1(pred, q["answer"])
                processed.append({
                    "번호": i,
                    "질문": q["question"][:100],
                    "예측": pred[:50],
                    "정답": q["answer"][:50],
                    "EM": em,
                    "F1": f1,
                    "난이도": q.get("meta", {}).get("difficulty", ""),
                })
        
        return processed


def save_results_chunked(mcq_results, short_results, chunk_size=1000):
    """대용량 결과를 청크로 나누어 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과가 너무 크면 여러 파일로 분할
    if len(mcq_results) > chunk_size or len(short_results) > chunk_size:
        print(f"\n[대용량 결과 분할 저장]")
        
        # MCQ 분할 저장
        if mcq_results:
            for i in range(0, len(mcq_results), chunk_size):
                chunk = mcq_results[i:i+chunk_size]
                output_file = f"eval_mcq_{timestamp}_part{i//chunk_size+1}.xlsx"
                df = pd.DataFrame(chunk)
                df.to_excel(output_file, index=False)
                print(f"  저장: {output_file} ({len(chunk)}개)")
        
        # Short 분할 저장
        if short_results:
            for i in range(0, len(short_results), chunk_size):
                chunk = short_results[i:i+chunk_size]
                output_file = f"eval_short_{timestamp}_part{i//chunk_size+1}.xlsx"
                df = pd.DataFrame(chunk)
                df.to_excel(output_file, index=False)
                print(f"  저장: {output_file} ({len(chunk)}개)")
    else:
        # 일반 저장
        output_file = f"evaluation_results_{timestamp}.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if mcq_results:
                pd.DataFrame(mcq_results).to_excel(writer, sheet_name="사지선다형", index=False)
            if short_results:
                pd.DataFrame(short_results).to_excel(writer, sheet_name="단답형", index=False)
        print(f"\n[결과 저장] {output_file}")
    
    # 요약 통계 저장
    summary_file = f"eval_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("평가 요약\n")
        f.write("="*60 + "\n\n")
        
        if mcq_results:
            n = len(mcq_results)
            acc = sum(1 for r in mcq_results if r["정확도"] == "O") / n
            f.write(f"사지선다형: {n}개\n")
            f.write(f"  정확도: {acc:.3f}\n\n")
        
        if short_results:
            n = len(short_results)
            em = sum(r["EM"] for r in short_results) / n
            f1 = sum(r["F1"] for r in short_results) / n
            f.write(f"단답형: {n}개\n")
            f.write(f"  EM: {em:.3f}\n")
            f.write(f"  F1: {f1:.3f}\n")
    
    print(f"[요약 저장] {summary_file}")


def main():
    """메인 실행 - 9000개 대규모 처리"""
    
    print("\n" + "="*60)
    print(" 대규모 RAG 평가 시스템 (터보 모드)")
    print("="*60)
    
    # 파일 정보
    files = [VAL_EXCEL_FILE1, VAL_EXCEL_FILE2]
    
    # 전체 문제 로드 (제한 없이)
    print("\n[전체 문제 로드 중...] 시간이 걸릴 수 있습니다.")
    all_mcq, all_short = load_multiple_excels(
        files,
        mcq_limit_per_file=None,  # 제한 없음
        short_limit_per_file=None,  # 제한 없음
        verbose=False
    )
    
    total_questions = len(all_mcq) + len(all_short)
    print(f"\n[로드 완료]")
    print(f"  사지선다형: {len(all_mcq)}개")
    print(f"  단답형: {len(all_short)}개")
    print(f"  총 문제: {total_questions}개")
    
    # 예상 시간 계산
    estimated_time = (total_questions * 2) / (60 * OPTIMAL_WORKERS)
    print(f"\n[예상 소요 시간] {estimated_time:.1f}분 ~ {estimated_time*2:.1f}분")
    
    # 사용자 확인
    response = input("\n계속하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("평가를 취소합니다.")
        return
    
    # 평가 시작
    start_time = time.time()
    evaluator = TurboEvaluator(max_workers=OPTIMAL_WORKERS)
    
    print("\n" + "="*60)
    print(" 평가 시작 (최대 성능 모드)")
    print("="*60)
    
    # 사지선다형 평가
    mcq_results = []
    if all_mcq:
        mcq_results = evaluator.evaluate_all(all_mcq, "mcq")
        print(f"\n사지선다형 완료: {len(mcq_results)}개")
    
    # 단답형 평가
    short_results = []
    if all_short:
        short_results = evaluator.evaluate_all(all_short, "short")
        print(f"단답형 완료: {len(short_results)}개")
    
    # 결과 저장
    save_results_chunked(mcq_results, short_results)
    
    # 소요 시간
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n[총 소요 시간] {elapsed_time:.1f}분")
    print(f"[처리 속도] {total_questions/elapsed_time:.1f}개/분")
    
    # 간단한 통계
    if mcq_results:
        acc = sum(1 for r in mcq_results if r["정확도"] == "O") / len(mcq_results)
        print(f"\n[사지선다형 정확도] {acc:.3f}")
    
    if short_results:
        em = sum(r["EM"] for r in short_results) / len(short_results)
        f1 = sum(r["F1"] for r in short_results) / len(short_results)
        print(f"[단답형 EM] {em:.3f}")
        print(f"[단답형 F1] {f1:.3f}")
    
    print("\n" + "="*60)
    print(" 평가 완료!")
    print("="*60)


if __name__ == "__main__":
    main()