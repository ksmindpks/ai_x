# -*- coding: utf-8 -*-
"""
rag/evaluator.py - 성능 모니터링 강화 버전
주요 개선사항:
1. MCQ 의미있는 성능 지표 추가
2. 난이도별 상세 분석
3. 실시간 성능 추적 개선
"""

from __future__ import annotations
import os, re
import time
import math
import random
import traceback
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterator
import concurrent.futures as cf
from collections import deque, defaultdict
import numpy as np

# 안전 import
from .retriever import retrieve, retrieve_batch
from .generator import generate_answer_mcq, generate_answer_short, get_generation_stats
from .utils import load_excel, save_results, calculate_accuracy, score_short, normalize_for_em

# Config import
try:
    from config import config
    PERFORMANCE_CONFIG = config.performance
    DEBUG_MODE = config.debug_mode
except ImportError:
    class DefaultConfig:
        openai_concurrency = 32
        max_workers = 16
        batch_size = 64
        max_retry_attempts = 3
        retry_base_delay = 0.15
        request_timeout = 20
    
    PERFORMANCE_CONFIG = DefaultConfig()
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

# tqdm 선택적 사용
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    
    class DummyTqdm:
        def __init__(self, total=None, desc="", **kwargs):
            self.n = 0
            self.total = total or 0
            self.desc = desc
        
        def update(self, n=1):
            self.n += n
        
        def close(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    tqdm = DummyTqdm

@dataclass
class EnhancedEvaluationStats:
    """향상된 평가 통계"""
    total_questions: int = 0
    processed_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    
    # MCQ 특화 지표
    mcq_exact_matches: int = 0
    mcq_context_utilizations: int = 0
    mcq_high_confidence_selections: int = 0
    mcq_difficulty_performance: Dict = None
    
    # 단답형 특화 지표
    short_exact_matches: int = 0
    short_partial_matches: int = 0
    info_insufficient: int = 0
    generation_failed: int = 0
    validation_failed: int = 0
    
    # 성능 지표
    total_time: float = 0.0
    search_time: float = 0.0
    generation_time: float = 0.0
    
    avg_search_score: float = 0.0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    
    # 품질 지표
    avg_context_quality: float = 0.0
    context_utilization_rate: float = 0.0
    high_confidence_rate: float = 0.0
    
    # 정확도
    accuracy: float = 0.0
    em_score: float = 0.0
    f1_score: float = 0.0
    
    def __post_init__(self):
        if self.mcq_difficulty_performance is None:
            self.mcq_difficulty_performance = defaultdict(lambda: {'total': 0, 'correct': 0, 'avg_confidence': 0.0})

class EnhancedPerformanceMonitor:
    """향상된 성능 모니터링"""
    
    def __init__(self):
        self.stats = EnhancedEvaluationStats()
        self.lock = threading.Lock()
        
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = 25  # 25초마다 보고
        
        self.recent_times = deque(maxlen=50)
        self.recent_scores = deque(maxlen=50)
        self.recent_results = deque(maxlen=100)
        self.confidence_scores = deque(maxlen=100)
        self.context_quality_scores = deque(maxlen=100)
        
        # MCQ 특화 추적
        self.mcq_choice_analysis = []
        self.mcq_context_matches = []
        
        # 단답형 특화 추적
        self.short_answer_types = defaultdict(int)
        self.short_em_scores = []
    
    def update_mcq_progress(self, processed: int, total: int, 
                           question: str, choices: List[str], selected: str, correct: str,
                           contexts: List[Dict], search_time: float = 0, 
                           generation_time: float = 0, difficulty: str = ""):
        """MCQ 진행상황 업데이트 - 강화된 버전"""
        with self.lock:
            self.stats.processed_questions = processed
            self.stats.total_questions = total
            
            current_time = time.time()
            
            # 기본 시간 추적
            if search_time > 0:
                self.stats.search_time += search_time
            if generation_time > 0:
                self.stats.generation_time += generation_time
            
            # MCQ 특화 분석
            is_correct = (selected == correct) if correct else False
            if is_correct:
                self.stats.mcq_exact_matches += 1
                if difficulty:
                    self.stats.mcq_difficulty_performance[difficulty]['correct'] += 1
            
            if difficulty:
                self.stats.mcq_difficulty_performance[difficulty]['total'] += 1
            
            # 컨텍스트 활용도 분석
            context_text = ' '.join([c.get('text', '') for c in contexts])
            context_utilized = selected in context_text
            if context_utilized:
                self.stats.mcq_context_utilizations += 1
            
            # 검색 품질 분석
            if contexts:
                search_score = contexts[0].get('final_score', 0) if contexts else 0
                self.recent_scores.append(search_score)
                self.context_quality_scores.append(search_score)
                
                # 고품질 컨텍스트 판정 (점수 > 1.5)
                if search_score > 1.5:
                    self.stats.mcq_high_confidence_selections += 1
            
            # 선택 분석 저장
            choice_analysis = {
                'question_type': self._classify_mcq_question(question),
                'selected': selected,
                'correct': correct,
                'is_correct': is_correct,
                'context_utilized': context_utilized,
                'num_choices': len(choices),
                'difficulty': difficulty,
                'search_score': contexts[0].get('final_score', 0) if contexts else 0
            }
            self.mcq_choice_analysis.append(choice_analysis)
            
            self.recent_times.append(current_time)
            
            # 주기적 보고
            if current_time - self.last_report_time >= self.report_interval:
                self._print_enhanced_mcq_progress_report()
                self.last_report_time = current_time
    
    def update_short_progress(self, processed: int, total: int,
                             question: str, prediction: str, answer: str,
                             contexts: List[Dict], search_time: float = 0, 
                             generation_time: float = 0, difficulty: str = ""):
        """단답형 진행상황 업데이트 - 강화된 버전"""
        with self.lock:
            self.stats.processed_questions = processed
            self.stats.total_questions = total
            
            current_time = time.time()
            
            # 기본 시간 추적
            if search_time > 0:
                self.stats.search_time += search_time
            if generation_time > 0:
                self.stats.generation_time += generation_time
            
            # 단답형 특화 분석
            if prediction == "정보 불충분":
                self.stats.info_insufficient += 1
                self.short_answer_types['info_insufficient'] += 1
            elif prediction and prediction.strip():
                # EM/F1 점수 계산
                em, f1 = score_short(prediction, answer)
                self.short_em_scores.append(em)
                
                if em >= 0.9:
                    self.stats.short_exact_matches += 1
                    self.short_answer_types['exact_match'] += 1
                elif em >= 0.3:
                    self.stats.short_partial_matches += 1
                    self.short_answer_types['partial_match'] += 1
                else:
                    self.short_answer_types['low_match'] += 1
                
                self.stats.successful_questions += 1
            else:
                self.stats.generation_failed += 1
                self.short_answer_types['generation_failed'] += 1
            
            # 검색 품질 분석
            if contexts:
                search_score = contexts[0].get('final_score', 0) if contexts else 0
                self.recent_scores.append(search_score)
                self.context_quality_scores.append(search_score)
            
            # 질문 유형 분석
            question_type = self._classify_short_question(question)
            
            self.recent_times.append(current_time)
            
            # 주기적 보고
            if current_time - self.last_report_time >= self.report_interval:
                self._print_enhanced_short_progress_report()
                self.last_report_time = current_time
    
    def _classify_mcq_question(self, question: str) -> str:
        """MCQ 질문 유형 분류"""
        if re.search(r'제\d+조', question):
            return "article_specific"
        elif re.search(r'(기간|얼마|몇)', question):
            return "period"
        elif re.search(r'(누구|누가|기관|담당)', question):
            return "organization"
        elif re.search(r'(무엇|어떤)', question):
            return "definition"
        else:
            return "general"
    
    def _classify_short_question(self, question: str) -> str:
        """단답형 질문 유형 분류"""
        if re.search(r'(기간|얼마|몇.*개월|몇.*년|몇.*일)', question):
            return "period"
        elif re.search(r'(누구|누가|어디|기관|담당)', question):
            return "organization"
        elif re.search(r'(무엇|어떤.*것|정의|의미)', question):
            return "definition"
        elif re.search(r'제\d+조', question):
            return "article_specific"
        else:
            return "general"
    
    def _print_enhanced_mcq_progress_report(self):
        """향상된 MCQ 진행상황 보고"""
        if self.stats.total_questions == 0:
            return
        
        elapsed = time.time() - self.start_time
        progress = self.stats.processed_questions / self.stats.total_questions
        
        # 처리 속도 계산
        if len(self.recent_times) >= 2:
            recent_elapsed = self.recent_times[-1] - self.recent_times[0]
            speed = len(self.recent_times) / max(recent_elapsed, 0.1)
        else:
            speed = 0
        
        # 예상 남은 시간
        remaining = self.stats.total_questions - self.stats.processed_questions
        eta = remaining / max(speed, 0.1)
        
        # 현재까지 정확도
        current_accuracy = self.stats.mcq_exact_matches / max(1, self.stats.processed_questions)
        
        # 평균 검색 점수
        avg_score = np.mean(list(self.recent_scores)) if self.recent_scores else 0
        
        # 컨텍스트 활용률
        context_rate = self.stats.mcq_context_utilizations / max(1, self.stats.processed_questions)
        
        print(f"\n[MCQ 진행상황] {progress:.1%} ({self.stats.processed_questions:,}/{self.stats.total_questions:,})")
        print(f"  속도: {speed:.1f}개/초, 예상잔여: {eta/60:.1f}분")
        print(f"  현재 정확도: {current_accuracy:.1%}, 컨텍스트 활용률: {context_rate:.1%}")
        print(f"  평균 검색점수: {avg_score:.3f}, 경과시간: {elapsed/60:.1f}분")
        
        # 난이도별 성능 (데이터가 있는 경우)
        if any(perf['total'] > 0 for perf in self.stats.mcq_difficulty_performance.values()):
            print(f"  난이도별 정확도:", end="")
            for diff, perf in self.stats.mcq_difficulty_performance.items():
                if perf['total'] > 0:
                    acc = perf['correct'] / perf['total']
                    print(f" {diff}:{acc:.1%}", end="")
            print()
    
    def _print_enhanced_short_progress_report(self):
        """향상된 단답형 진행상황 보고"""
        if self.stats.total_questions == 0:
            return
        
        elapsed = time.time() - self.start_time
        progress = self.stats.processed_questions / self.stats.total_questions
        
        # 처리 속도 계산
        if len(self.recent_times) >= 2:
            recent_elapsed = self.recent_times[-1] - self.recent_times[0]
            speed = len(self.recent_times) / max(recent_elapsed, 0.1)
        else:
            speed = 0
        
        # 예상 남은 시간
        remaining = self.stats.total_questions - self.stats.processed_questions
        eta = remaining / max(speed, 0.1)
        
        # 현재까지 EM 점수
        current_em = np.mean(self.short_em_scores) if self.short_em_scores else 0
        
        # 평균 검색 점수
        avg_score = np.mean(list(self.recent_scores)) if self.recent_scores else 0
        
        # 답변 유형별 분포
        total_processed = self.stats.processed_questions
        exact_rate = self.stats.short_exact_matches / max(1, total_processed)
        partial_rate = self.stats.short_partial_matches / max(1, total_processed)
        info_rate = self.stats.info_insufficient / max(1, total_processed)
        
        print(f"\n[단답형 진행상황] {progress:.1%} ({self.stats.processed_questions:,}/{self.stats.total_questions:,})")
        print(f"  속도: {speed:.1f}개/초, 예상잔여: {eta/60:.1f}분")
        print(f"  현재 EM: {current_em:.1%}, 평균 검색점수: {avg_score:.3f}")
        print(f"  답변 분포: 정확{exact_rate:.1%} 부분{partial_rate:.1%} 불충분{info_rate:.1%}")
        print(f"  경과시간: {elapsed/60:.1f}분")
    
    def finalize(self):
        """최종 통계 계산 - 강화된 버전"""
        with self.lock:
            self.stats.total_time = time.time() - self.start_time
            
            if self.stats.processed_questions > 0:
                self.stats.throughput = self.stats.processed_questions / self.stats.total_time
                self.stats.avg_response_time = self.stats.total_time / self.stats.processed_questions * 1000
            
            if self.recent_scores:
                self.stats.avg_search_score = np.mean(list(self.recent_scores))
            
            if self.context_quality_scores:
                self.stats.avg_context_quality = np.mean(list(self.context_quality_scores))
            
            # MCQ 특화 지표 계산
            if self.stats.processed_questions > 0:
                self.stats.context_utilization_rate = self.stats.mcq_context_utilizations / self.stats.processed_questions
                self.stats.high_confidence_rate = self.stats.mcq_high_confidence_selections / self.stats.processed_questions
            
            # Generator 통계 가져오기
            try:
                gen_stats = get_generation_stats()
                if gen_stats:
                    # 새로운 통계 구조에서 데이터 추출
                    mcq_stats = gen_stats.get('mcq_accuracy', 0)
                    short_stats = gen_stats.get('short_success_rate', 0)
                    self.stats.accuracy = mcq_stats
            except Exception:
                pass
    
    def get_detailed_analysis(self) -> Dict:
        """상세 분석 결과 반환"""
        analysis = {
            "총 처리 문제": self.stats.processed_questions,
            "처리 속도": f"{self.stats.throughput:.1f}개/초",
            "평균 응답시간": f"{self.stats.avg_response_time:.1f}ms",
            "평균 검색 점수": f"{self.stats.avg_search_score:.3f}",
            "평균 컨텍스트 품질": f"{self.stats.avg_context_quality:.3f}"
        }
        
        # MCQ 분석
        if self.mcq_choice_analysis:
            mcq_total = len(self.mcq_choice_analysis)
            mcq_correct = sum(1 for a in self.mcq_choice_analysis if a['is_correct'])
            context_utilized = sum(1 for a in self.mcq_choice_analysis if a['context_utilized'])
            
            analysis["MCQ 분석"] = {
                "정확도": f"{mcq_correct/mcq_total:.1%}" if mcq_total > 0 else "0%",
                "컨텍스트 활용률": f"{context_utilized/mcq_total:.1%}" if mcq_total > 0 else "0%",
                "고품질 선택률": f"{self.stats.high_confidence_rate:.1%}"
            }
            
            # 질문 유형별 분석
            type_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
            for analysis_item in self.mcq_choice_analysis:
                qtype = analysis_item['question_type']
                type_performance[qtype]['total'] += 1
                if analysis_item['is_correct']:
                    type_performance[qtype]['correct'] += 1
            
            analysis["MCQ 질문유형별"] = {
                qtype: f"{perf['correct']}/{perf['total']} ({perf['correct']/perf['total']:.1%})" 
                if perf['total'] > 0 else "0/0"
                for qtype, perf in type_performance.items()
            }
        
        # 단답형 분석
        if self.short_em_scores:
            avg_em = np.mean(self.short_em_scores)
            analysis["단답형 분석"] = {
                "평균 EM": f"{avg_em:.1%}",
                "정확 매칭": f"{self.stats.short_exact_matches}개",
                "부분 매칭": f"{self.stats.short_partial_matches}개",
                "정보 불충분": f"{self.stats.info_insufficient}개"
            }
            
            # 답변 유형 분포
            analysis["단답형 답변유형"] = dict(self.short_answer_types)
        
        return analysis

class HighPerformanceEvaluator:
    """성능 모니터링 강화된 평가기"""
    
    def __init__(self, xlsx_path: str, mcq_n: int, short_n: int, workers: int = None, debug: bool = False):
        self.xlsx_path = xlsx_path
        self.mcq_n = mcq_n
        self.short_n = short_n
        self.workers = workers or PERFORMANCE_CONFIG.max_workers
        self.debug = debug
        
        # 강화된 모니터링
        self.monitor = EnhancedPerformanceMonitor()
        
        # 오류 복구
        self.max_retries = PERFORMANCE_CONFIG.max_retry_attempts
        self.retry_delay = PERFORMANCE_CONFIG.retry_base_delay
        
        # 배치 처리 설정
        self.batch_size = min(PERFORMANCE_CONFIG.batch_size, max(8, self.workers * 2))
    
    def evaluate_all(self, items: List[Dict], mode: str) -> Tuple[List[Dict], Dict]:
        """강화된 전체 평가 실행"""
        if mode == "mcq":
            return self._evaluate_mcq_enhanced(items)
        elif mode == "short":
            return self._evaluate_short_enhanced(items)
        else:
            raise ValueError("mode must be 'mcq' or 'short'")
    
    def _evaluate_mcq_enhanced(self, items: List[Dict]) -> Tuple[List[Dict], Dict]:
        """강화된 MCQ 평가"""
        n = min(self.mcq_n or len(items), len(items))
        items = items[:n]
        
        print(f"\n사지선다형 평가 시작: {n:,}개 문제")
        print(f"  향상된 성능 모니터링 활성화")
        
        # 결과 저장소
        results = [None] * n
        search_scores = []
        difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        self.monitor.stats.total_questions = n
        
        # 배치 검색 실행
        search_start = time.time()
        questions = [item["question"] for item in items]
        
        try:
            contexts_list = retrieve_batch(questions, top_k=7, debug=False)
            if not contexts_list or len(contexts_list) != len(questions):
                raise ValueError("배치 검색 실패")
        except Exception as e:
            if DEBUG_MODE:
                print(f"[MCQ] 배치 검색 실패, 개별 검색으로 전환: {e}")
            contexts_list = []
            for question in questions:
                try:
                    ctx = retrieve(question, top_k=7, debug=False)
                    contexts_list.append(ctx)
                except Exception:
                    contexts_list.append([])
        
        search_time = time.time() - search_start
        
        # 강화된 생성 단계
        generation_start = time.time()
        
        def process_mcq_item_enhanced(i: int) -> bool:
            start_time = time.time()
            success = False
            
            try:
                item = items[i]
                contexts = contexts_list[i] if i < len(contexts_list) else []
                
                search_score = self._extract_top_score(contexts)
                search_scores.append(search_score)
                
                choices = item.get("choices", [])
                question = item["question"]
                correct_answer = item.get("answer", "")
                difficulty = item.get("meta", {}).get("difficulty", "미분류")
                
                # 강화된 MCQ 생성
                prediction = generate_answer_mcq(question, choices, contexts)
                
                if prediction:
                    success = True
                    is_correct = (prediction == correct_answer)
                    if is_correct:
                        difficulty_stats[difficulty]['correct'] += 1
                    difficulty_stats[difficulty]['total'] += 1
                else:
                    prediction = choices[0] if choices else ""
                
                results[i] = {
                    "question": question,
                    "choices": choices,
                    "answer": correct_answer,
                    "prediction": prediction,
                    "metadata": item.get("meta", {}),
                    "search_score": search_score,
                    "difficulty": difficulty,
                    "context_quality": search_score
                }
                
                # 강화된 진행 상황 업데이트
                processed = sum(1 for r in results if r is not None)
                self.monitor.update_mcq_progress(
                    processed, n, question, choices, prediction, correct_answer,
                    contexts, 0, time.time() - start_time, difficulty
                )
                
            except Exception as e:
                if self.debug:
                    print(f"[MCQ-Error] idx={i}: {e}")
                
                # 기본값으로 결과 생성
                item = items[i]
                choices = item.get("choices", [])
                difficulty = item.get("meta", {}).get("difficulty", "미분류")
                difficulty_stats[difficulty]['total'] += 1
                
                results[i] = {
                    "question": item["question"],
                    "choices": choices,
                    "answer": item.get("answer", ""),
                    "prediction": choices[0] if choices else "",
                    "metadata": item.get("meta", {}),
                    "search_score": 0.0,
                    "difficulty": difficulty,
                    "context_quality": 0.0
                }
            
            return success
        
        # 병렬 실행
        with cf.ThreadPoolExecutor(max_workers=self.workers) as executor:
            list(executor.map(process_mcq_item_enhanced, range(n)))
        
        generation_time = time.time() - generation_start
        
        # 강화된 통계 계산
        done_results = [r for r in results if r is not None]
        accuracy = calculate_accuracy(done_results)
        
        self.monitor.finalize()
        
        # 상세 통계 생성
        stats = self._create_enhanced_mcq_stats(done_results, search_time, generation_time, search_scores, difficulty_stats)
        stats['acc'] = accuracy
        
        # 강화된 결과 출력
        self._print_enhanced_mcq_results(done_results, stats, difficulty_stats)
        
        return done_results, stats
    
    def _evaluate_short_enhanced(self, items: List[Dict]) -> Tuple[List[Dict], Dict]:
        """강화된 단답형 평가"""
        n = min(self.short_n or len(items), len(items))
        items = items[:n]
        
        print(f"\n단답형 평가 시작: {n:,}개 문제")
        print(f"  정밀도 향상 모듈 활성화")
        
        # 결과 저장소
        results = [None] * n
        search_scores = []
        em_scores = []
        f1_scores = []
        answer_type_stats = defaultdict(int)
        question_type_stats = defaultdict(lambda: {'total': 0, 'em_sum': 0.0})
        
        self.monitor.stats.total_questions = n
        
        # 배치 검색 실행
        search_start = time.time()
        questions = [item["question"] for item in items]
        
        try:
            contexts_list = retrieve_batch(questions, top_k=7, debug=False)
            if not contexts_list or len(contexts_list) != len(questions):
                raise ValueError("배치 검색 실패")
        except Exception as e:
            if DEBUG_MODE:
                print(f"[Short] 배치 검색 실패, 개별 검색으로 전환: {e}")
            contexts_list = []
            for question in questions:
                try:
                    ctx = retrieve(question, top_k=7, debug=False)
                    contexts_list.append(ctx)
                except Exception:
                    contexts_list.append([])
        
        search_time = time.time() - search_start
        
        # 강화된 생성 단계
        generation_start = time.time()
        
        def process_short_item_enhanced(i: int) -> bool:
            start_time = time.time()
            success = False
            
            try:
                item = items[i]
                contexts = contexts_list[i] if i < len(contexts_list) else []
                
                search_score = self._extract_top_score(contexts)
                search_scores.append(search_score)
                
                question = item["question"]
                correct_answer = item.get("answer", "")
                difficulty = item.get("meta", {}).get("difficulty", "미분류")
                
                # 질문 유형 분류
                question_type = self._classify_question_type(question)
                question_type_stats[question_type]['total'] += 1
                
                # 강화된 단답형 생성
                prediction = generate_answer_short(question, contexts)
                
                # 결과 분석
                if prediction == "정보 불충분":
                    answer_type_stats['info_insufficient'] += 1
                    em, f1 = 0.0, 0.0
                elif prediction and prediction.strip():
                    success = True
                    em, f1 = score_short(prediction, correct_answer)
                    
                    # 답변 유형 분류
                    if em >= 0.9:
                        answer_type_stats['exact_match'] += 1
                    elif em >= 0.3:
                        answer_type_stats['partial_match'] += 1
                    else:
                        answer_type_stats['low_match'] += 1
                else:
                    answer_type_stats['generation_failed'] += 1
                    prediction = "정보 불충분"
                    em, f1 = 0.0, 0.0
                
                em_scores.append(em)
                f1_scores.append(f1)
                question_type_stats[question_type]['em_sum'] += em
                
                results[i] = {
                    "question": question,
                    "answer": correct_answer,
                    "prediction": prediction,
                    "metadata": item.get("meta", {}),
                    "search_score": search_score,
                    "em_score": em,
                    "f1_score": f1,
                    "difficulty": difficulty,
                    "question_type": question_type,
                    "answer_type": self._classify_answer_type(prediction)
                }
                
                # 강화된 진행 상황 업데이트
                processed = sum(1 for r in results if r is not None)
                self.monitor.update_short_progress(
                    processed, n, question, prediction, correct_answer,
                    contexts, 0, time.time() - start_time, difficulty
                )
                
            except Exception as e:
                if self.debug:
                    print(f"[Short-Error] idx={i}: {e}")
                
                # 기본값으로 결과 생성
                item = items[i]
                difficulty = item.get("meta", {}).get("difficulty", "미분류")
                question_type = self._classify_question_type(item["question"])
                
                question_type_stats[question_type]['total'] += 1
                answer_type_stats['generation_failed'] += 1
                
                results[i] = {
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                    "prediction": "정보 불충분",
                    "metadata": item.get("meta", {}),
                    "search_score": 0.0,
                    "em_score": 0.0,
                    "f1_score": 0.0,
                    "difficulty": difficulty,
                    "question_type": question_type,
                    "answer_type": "generation_failed"
                }
                
                em_scores.append(0.0)
                f1_scores.append(0.0)
            
            return success
        
        # 병렬 실행
        with cf.ThreadPoolExecutor(max_workers=self.workers) as executor:
            list(executor.map(process_short_item_enhanced, range(n)))
        
        generation_time = time.time() - generation_start
        
        # 강화된 통계 계산
        done_results = [r for r in results if r is not None]
        
        self.monitor.finalize()
        
        # 상세 통계 생성
        stats = self._create_enhanced_short_stats(
            done_results, search_time, generation_time, search_scores, 
            em_scores, f1_scores, answer_type_stats, question_type_stats
        )
        
        # 강화된 결과 출력
        self._print_enhanced_short_results(done_results, stats, answer_type_stats, question_type_stats)
        
        return done_results, stats
    
    def _create_enhanced_mcq_stats(self, results: List[Dict], search_time: float, 
                                  generation_time: float, search_scores: List[float],
                                  difficulty_stats: Dict) -> Dict:
        """강화된 MCQ 통계 생성"""
        stats = {
            "n": len(results),
            "time_search": search_time,
            "time_gen": generation_time,
            "time_total": search_time + generation_time,
            "speed": len(results) / max(search_time + generation_time, 0.001),
            "avg_search_score": np.mean(search_scores) if search_scores else 0.0,
            "throughput": self.monitor.stats.throughput,
            "avg_response_time": self.monitor.stats.avg_response_time,
            "context_utilization_rate": self.monitor.stats.context_utilization_rate,
            "high_confidence_rate": self.monitor.stats.high_confidence_rate,
            "difficulty_breakdown": dict(difficulty_stats)
        }
        
        # 품질 분석 추가
        if results:
            context_quality_scores = [r.get('context_quality', 0) for r in results]
            stats['avg_context_quality'] = np.mean(context_quality_scores)
            
            # 고품질 컨텍스트 비율
            high_quality_count = sum(1 for score in context_quality_scores if score > 1.5)
            stats['high_quality_context_rate'] = high_quality_count / len(results)
        
        return stats
    
    def _create_enhanced_short_stats(self, results: List[Dict], search_time: float, 
                                    generation_time: float, search_scores: List[float],
                                    em_scores: List[float], f1_scores: List[float],
                                    answer_type_stats: Dict, question_type_stats: Dict) -> Dict:
        """강화된 단답형 통계 생성"""
        stats = {
            "n": len(results),
            "time_search": search_time,
            "time_gen": generation_time,
            "time_total": search_time + generation_time,
            "speed": len(results) / max(search_time + generation_time, 0.001),
            "avg_search_score": np.mean(search_scores) if search_scores else 0.0,
            "throughput": self.monitor.stats.throughput,
            "avg_response_time": self.monitor.stats.avg_response_time,
            "EM": np.mean(em_scores) if em_scores else 0.0,
            "F1": np.mean(f1_scores) if f1_scores else 0.0,
            "answer_type_breakdown": dict(answer_type_stats),
            "question_type_breakdown": dict(question_type_stats)
        }
        
        # 추가 분석
        if results:
            info_insufficient_count = sum(1 for r in results if r.get('prediction') == "정보 불충분")
            stats['info_insufficient_count'] = info_insufficient_count
            stats['info_insufficient_rate'] = info_insufficient_count / len(results)
            
            # EM 점수 분포
            em_distribution = {
                'perfect': sum(1 for em in em_scores if em >= 0.95),
                'excellent': sum(1 for em in em_scores if 0.8 <= em < 0.95),
                'good': sum(1 for em in em_scores if 0.5 <= em < 0.8),
                'fair': sum(1 for em in em_scores if 0.2 <= em < 0.5),
                'poor': sum(1 for em in em_scores if em < 0.2)
            }
            stats['em_distribution'] = em_distribution
            
            # 품질 점수 계산
            context_quality_scores = [r.get('search_score', 0) for r in results]
            stats['avg_context_quality'] = np.mean(context_quality_scores)
        
        return stats
    
    def _print_enhanced_mcq_results(self, results: List[Dict], stats: Dict, difficulty_stats: Dict):
        """강화된 MCQ 결과 출력"""
        print(f"\nMCQ 평가 완료:")
        print(f"  정확도: {stats.get('acc', 0):.1%}")
        print(f"  처리속도: {stats['speed']:.1f}개/초")
        print(f"  평균 검색점수: {stats['avg_search_score']:.3f}")
        print(f"  평균 컨텍스트 품질: {stats.get('avg_context_quality', 0):.3f}")
        print(f"  컨텍스트 활용률: {stats.get('context_utilization_rate', 0):.1%}")
        print(f"  고신뢰도 선택률: {stats.get('high_confidence_rate', 0):.1%}")
        
        # 난이도별 성능
        if difficulty_stats and any(d['total'] > 0 for d in difficulty_stats.values()):
            print(f"  난이도별 정확도:")
            for diff, perf in difficulty_stats.items():
                if perf['total'] > 0:
                    acc = perf['correct'] / perf['total']
                    print(f"    {diff}: {acc:.1%} ({perf['correct']}/{perf['total']})")
        
        # Generator 성능 정보
        try:
            gen_stats = get_generation_stats()
            if gen_stats:
                mcq_accuracy = gen_stats.get('mcq_accuracy', 0)
                context_util = gen_stats.get('mcq_context_utilization', 0)
                avg_confidence = gen_stats.get('mcq_avg_confidence', 0)
                
                print(f"  Generator 통계:")
                print(f"    정확도: {mcq_accuracy:.1%}")
                print(f"    컨텍스트 활용: {context_util:.1%}")
                print(f"    평균 신뢰도: {avg_confidence:.3f}")
        except Exception:
            pass
    
    def _print_enhanced_short_results(self, results: List[Dict], stats: Dict, 
                                     answer_type_stats: Dict, question_type_stats: Dict):
        """강화된 단답형 결과 출력"""
        print(f"\n단답형 평가 완료:")
        print(f"  EM 점수: {stats['EM']:.1%}")
        print(f"  F1 점수: {stats['F1']:.1%}")
        print(f"  처리속도: {stats['speed']:.1f}개/초")
        print(f"  평균 검색점수: {stats['avg_search_score']:.3f}")
        print(f"  평균 컨텍스트 품질: {stats.get('avg_context_quality', 0):.3f}")
        
        # 답변 유형별 분포
        if answer_type_stats:
            total = sum(answer_type_stats.values())
            print(f"  답변 유형 분포:")
            for answer_type, count in answer_type_stats.items():
                print(f"    {answer_type}: {count}개 ({count/total:.1%})")
        
        # EM 점수 분포
        em_dist = stats.get('em_distribution', {})
        if em_dist:
            total = sum(em_dist.values())
            print(f"  EM 점수 분포:")
            for category, count in em_dist.items():
                print(f"    {category}: {count}개 ({count/total:.1%})")
        
        # 질문 유형별 성능
        if question_type_stats:
            print(f"  질문 유형별 평균 EM:")
            for qtype, data in question_type_stats.items():
                if data['total'] > 0:
                    avg_em = data['em_sum'] / data['total']
                    print(f"    {qtype}: {avg_em:.1%} ({data['total']}개)")
        
        # 상세 실패 분석
        info_insufficient_count = stats.get('info_insufficient_count', 0)
        info_rate = stats.get('info_insufficient_rate', 0)
        print(f"  정보 불충분: {info_insufficient_count}개 ({info_rate:.1%})")
        
        # Generator 성능 정보
        try:
            gen_stats = get_generation_stats()
            if gen_stats:
                success_rate = gen_stats.get('short_success_rate', 0)
                validation_rate = gen_stats.get('short_validation_pass_rate', 0)
                avg_time = gen_stats.get('short_avg_generation_time', 0)
                
                print(f"  Generator 통계:")
                print(f"    성공률: {success_rate:.1%}")
                print(f"    검증 통과율: {validation_rate:.1%}")
                print(f"    평균 생성시간: {avg_time:.1f}ms")
        except Exception:
            pass
    
    def _classify_question_type(self, question: str) -> str:
        """질문 유형 분류"""
        if re.search(r'(기간|얼마|몇.*개월|몇.*년|몇.*일)', question):
            return "period"
        elif re.search(r'(누구|누가|어디|기관|담당)', question):
            return "organization"
        elif re.search(r'(무엇|어떤.*것|정의|의미)', question):
            return "definition"
        elif re.search(r'제\d+조', question):
            return "article_specific"
        else:
            return "general"
    
    def _classify_answer_type(self, answer: str) -> str:
        """답변 유형 분류"""
        if answer == "정보 불충분":
            return "insufficient_info"
        elif re.search(r'\d+(?:개월|년|일|월|%)', answer):
            return "numeric"
        elif re.search(r'(장관|위원회|감독원|은행|청|부)', answer):
            return "organization"
        elif len(answer) >= 10:
            return "detailed"
        else:
            return "simple"
    
    def _extract_top_score(self, contexts: List[Dict]) -> float:
        """최고 검색 점수 추출"""
        if not contexts:
            return 0.0
        
        first_ctx = contexts[0]
        return float(first_ctx.get("final_score") or first_ctx.get("score") or 0.0)
    
    def run(self) -> str:
        """강화된 전체 평가 실행"""
        print("\n" + "="*60)
        print(" RAG 성능 강화 평가 시스템")
        print("="*60)
        
        # 데이터 로드
        print(f"\n파일 로드: {self.xlsx_path}")
        mcq_items, short_items = load_excel(self.xlsx_path)
        print(f"사지선다형: {len(mcq_items):,}개 중 {self.mcq_n:,}개 평가")
        print(f"단답형: {len(short_items):,}개 중 {self.short_n:,}개 평가")
        
        # MCQ 평가
        if mcq_items and self.mcq_n > 0:
            mcq_results, mcq_stats = self._evaluate_mcq_enhanced(mcq_items)
        else:
            mcq_results, mcq_stats = [], {}
        
        # Short 평가
        if short_items and self.short_n > 0:
            short_results, short_stats = self._evaluate_short_enhanced(short_items)
        else:
            short_results, short_stats = [], {}
        
        # 결과 저장
        output_file = save_results(mcq_results, short_results)
        
        # 종합 분석 출력
        self._print_comprehensive_analysis(mcq_results, short_results, mcq_stats, short_stats)
        
        print(f"\n결과 저장: {output_file}")
        print("="*60)
        
        return output_file
    
    def _print_comprehensive_analysis(self, mcq_results: List[Dict], short_results: List[Dict],
                                     mcq_stats: Dict, short_stats: Dict):
        """종합 분석 출력"""
        print("\n" + "="*60)
        print(" 종합 성능 분석")
        print("="*60)
        
        total_questions = len(mcq_results) + len(short_results)
        total_time = mcq_stats.get('time_total', 0) + short_stats.get('time_total', 0)
        
        print(f"총 평가 문제: {total_questions:,}개")
        print(f"총 소요 시간: {total_time/60:.1f}분")
        print(f"전체 처리속도: {total_questions/max(total_time, 0.001):.1f}개/초")
        
        if mcq_results:
            mcq_acc = mcq_stats.get('acc', 0)
            context_util = mcq_stats.get('context_utilization_rate', 0)
            high_conf = mcq_stats.get('high_confidence_rate', 0)
            
            print(f"\nMCQ 종합 성능:")
            print(f"  정확도: {mcq_acc:.1%}")
            print(f"  컨텍스트 활용률: {context_util:.1%}")
            print(f"  고신뢰도 선택률: {high_conf:.1%}")
            
            # 목표 달성 평가
            target_mcq = 0.65
            if mcq_acc >= target_mcq:
                print(f"  목표 달성! (목표: {target_mcq:.1%})")
            else:
                improvement_needed = target_mcq - mcq_acc
                print(f"  목표까지: {improvement_needed:.1%}p 개선 필요 (목표: {target_mcq:.1%})")
        
        if short_results:
            short_em = short_stats.get('EM', 0)
            short_f1 = short_stats.get('F1', 0)
            info_rate = short_stats.get('info_insufficient_rate', 0)
            
            print(f"\n단답형 종합 성능:")
            print(f"  EM 점수: {short_em:.1%}")
            print(f"  F1 점수: {short_f1:.1%}")
            print(f"  정보 불충분률: {info_rate:.1%}")
            
            # 목표 달성 평가
            target_em = 0.20
            target_info = 0.05
            
            if short_em >= target_em:
                print(f"  EM 목표 달성! (목표: {target_em:.1%})")
            else:
                improvement_needed = target_em - short_em
                print(f"  EM 목표까지: {improvement_needed:.1%}p 개선 필요 (목표: {target_em:.1%})")
            
            if info_rate <= target_info:
                print(f"  정보 불충분 목표 달성! (목표: ≤{target_info:.1%})")
            else:
                print(f"  정보 불충분률 개선 필요 (목표: ≤{target_info:.1%})")
        
        # 상세 분석 결과
        detailed_analysis = self.monitor.get_detailed_analysis()
        if detailed_analysis:
            print(f"\n상세 분석:")
            for category, data in detailed_analysis.items():
                print(f"  {category}:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"    {key}: {value}")
                else:
                    print(f"    {data}")

# 호환성을 위한 기존 클래스
class Evaluator(HighPerformanceEvaluator):
    """기존 인터페이스 호환성"""
    pass