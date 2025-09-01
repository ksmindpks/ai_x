"""
evaluator.py - 개선된 평가기
주요 개선사항:
1. MCQ/단답형 오류 패턴 분석 강화
2. 개선된 통계 수집 및 리포팅
3. 실패 원인별 세분화된 분석
4. 성능 모니터링 개선
"""
import time
import pandas as pd
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Callable, Optional

from rag.utils import (
    load_excel_data, save_evaluation_results,
    calculate_enhanced_exact_match, calculate_enhanced_f1_score,
    parse_mcq_answer, SearchResult
)

def log_message(log_type, message, module="EVALUATOR"):
    """통합된 로그 함수 - 3단계 분류"""
    # 웹 인터페이스로 전달 시도
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'global_log_callback'):
            callback = st.session_state.global_log_callback
            if callable(callback):
                callback(log_type, message, module, "evaluation")
        else:
            # 웹 환경이 아닐 때만 직접 출력
            print(f"[{module}-{log_type.upper()}] {message}")
    except Exception:
        # 오류 시 직접 출력
        print(f"[{module}-{log_type.upper()}] {message}")

class UnifiedEvaluator:
    """개선된 평가기 - 강화된 오류 분석 및 통계"""
    
    def __init__(self, config=None, retriever=None, llm=None):
        """평가기 초기화 - 중복 로그 방지"""
        
        if retriever and llm:
            # 기존 인스턴스 재사용 (로그 출력 안함)
            self.retriever = retriever
            self.llm = llm
            self.config = config if config else getattr(retriever, 'config', None)
            log_message("SUCCESS", "기존 RAG 인스턴스 재사용")
        else:
            # 새로 생성 (기존 방식)
            log_message("INFO", "평가기 초기화 중...")
            
            if not config:
                from config import get_config
                config = get_config()
            
            self.config = config
            from rag.hybrid_retriever import HybridRetriever
            from rag.llm_bridge import HybridLLM
            
            self.retriever = HybridRetriever(config)
            self.llm = HybridLLM(config)
            log_message("SUCCESS", "새 RAG 인스턴스 생성")
        
        # 법령 용어 사전
        self.legal_synonyms = {
            "중기부": "중소벤처기업부",
            "금위": "금융위원회",
            "금감원": "금융감독원",
            "공정위": "공정거래위원회",
        }
        
        # ★ 강화된 평가 통계 ★
        self.eval_stats = {
            'mcq_error_patterns': {
                'choice_mapping': 0,           # 파싱/매핑 오류
                'context_quality': 0,          # 컨텍스트 품질 문제
                'search_failure': 0,           # 검색 실패
                'negative_detection': 0,       # 부정형 감지 실패
                'llm_reasoning': 0             # LLM 추론 오류 (신규)
            },
            'short_error_patterns': {
                'context_mismatch': 0,         # 검색 문맥 불일치
                'extraction_failure': 0,       # 추출 실패
                'low_bm25_score': 0,          # BM25 점수 부족
                'no_search_results': 0,        # 검색 결과 없음
                'normalization_failure': 0     # 정규화 실패 (신규)
            },
            'search_quality_issues': 0,
            'total_questions': 0,
            'performance_metrics': {           # 성능 메트릭 (신규)
                'avg_mcq_time': 0.0,
                'avg_short_time': 0.0,
                'total_time': 0.0,
                'search_success_rate': 0.0
            }
        }
        
        log_message("SUCCESS", "초기화 완료")

    def evaluate_mcq_batch(self, questions: List[Dict], max_questions: int = None, 
                          progress_callback: Optional[Callable] = None) -> Tuple[float, List[Dict]]:
        """MCQ 배치 평가 - 강화된 오류 분석"""
        if max_questions:
            questions = questions[:max_questions]
        
        if not questions:
            log_message("FAILURE", "MCQ 질문이 없음")
            return 0.0, []
        
        log_message("INFO", f"MCQ 평가 시작: {len(questions)}개 문제")
        start_time = time.time()
        
        correct = 0
        results = []
        mcq_times = []
        
        for i, q in enumerate(questions):
            if progress_callback:
                progress = (i / len(questions)) * 100
                progress_callback(f"MCQ 진행: {i+1}/{len(questions)} ({progress:.1f}%)")
            
            item_start = time.time()
            question_preview = q['question'][:50] + "..." if len(q['question']) > 50 else q['question']
            log_message("INFO", f"MCQ-{i+1} 처리 시작: '{question_preview}'")
            
            # MCQ 검색: BM25(3) + Vector(9) = 10개
            search_results = self.retriever.search(q['question'], question_type="MCQ")
            
            # 검색 품질 분석
            search_quality = self._analyze_search_quality(search_results, "MCQ")
            if search_quality['has_issues']:
                self.eval_stats['search_quality_issues'] += 1
                log_message("FAILURE", f"MCQ-{i+1} 검색 품질 문제: {search_quality['issues']}")
            
            log_message("INFO", f"MCQ-{i+1} 검색 완료: {len(search_results)}개 "
                       f"(BM25최고={search_quality['bm25_max']:.2f}, Vec최고={search_quality['vector_max']:.2f})")
            
            # 부정형 질문 감지
            is_negative_question = self._detect_negative_question(q['question'])
            if is_negative_question:
                log_message("INFO", f"MCQ-{i+1} 부정형 질문 감지")
            
            # 컨텍스트 선택 및 품질 평가
            context = self._select_mcq_context(q['question'], search_results)
            context_quality = self._assess_context_quality(context, q['question'])
            
            # LLM 처리
            choices_dict = {chr(65+i): choice for i, choice in enumerate(q['choices'])}
            llm_response = self.llm.call_mcq(q['question'], choices_dict, context)
            
            # ★ 강화된 답변 파싱 ★
            expected_format = q.get('answer_format', 'number')
            predicted = parse_mcq_answer(llm_response, expected_format)
            correct_answer = str(q.get('answer', '1')).strip()
            
            is_correct = (predicted == correct_answer)
            item_time = time.time() - item_start
            mcq_times.append(item_time)
            
            if is_correct:
                correct += 1
                log_message("SUCCESS", f"MCQ-{i+1} 정답: {predicted}")
            else:
                # ★ 강화된 오답 원인 분석 ★
                error_type = self._analyze_mcq_error_enhanced(
                    search_quality, context_quality, is_negative_question, 
                    q['question'], choices_dict, llm_response, predicted, correct_answer
                )
                self.eval_stats['mcq_error_patterns'][error_type] += 1
                
                log_message("FAILURE", f"MCQ-{i+1} 오답: 예측={predicted}, 정답={correct_answer}, "
                           f"오류유형={error_type}, 컨텍스트품질={context_quality:.2f}")
            
            # 결과 저장 (확장된 정보)
            results.append({
                'question': q['question'][:100],
                'predicted': predicted,
                'correct': correct_answer,
                'is_correct': is_correct,
                'choices': q['choices'],
                'llm_response': llm_response[:100],
                'response_time': item_time,
                'search_results_count': len(search_results),
                'bm25_max_score': search_quality['bm25_max'],
                'vector_max_score': search_quality['vector_max'],
                'context_quality': context_quality,
                'is_negative_question': is_negative_question,
                'error_type': error_type if not is_correct else 'correct',
                'context_preview': context[:200],
                'parsing_succeeded': predicted in ['A', 'B', 'C', 'D', '1', '2', '3', '4']  # 신규
            })
        
        accuracy = correct / len(questions)
        total_time = time.time() - start_time
        
        # 성능 메트릭 업데이트
        self.eval_stats['performance_metrics']['avg_mcq_time'] = sum(mcq_times) / len(mcq_times)
        self.eval_stats['performance_metrics']['total_time'] += total_time
        
        # 오답 패턴 분석 결과 출력
        self._log_error_pattern_analysis("MCQ", len(questions) - correct)
        
        log_message("SUCCESS", f"MCQ 완료: 정확도 {accuracy:.1%} ({correct}/{len(questions)}) - {total_time:.1f}초")
        
        return accuracy, results

    def evaluate_short_batch(self, questions: List[Dict], max_questions: int = None,
                            progress_callback: Optional[Callable] = None) -> Tuple[float, float, List[Dict]]:
        """단답형 배치 평가 - 강화된 오류 분석"""
        if max_questions:
            questions = questions[:max_questions]
        
        if not questions:
            log_message("FAILURE", "단답형 질문이 없음")
            return 0.0, 0.0, []
        
        log_message("INFO", f"단답형 평가 시작: {len(questions)}개 문제")
        start_time = time.time()
        
        em_correct = 0
        f1_total = 0.0
        results = []
        short_times = []
        
        for i, q in enumerate(questions):
            if progress_callback:
                progress = (i / len(questions)) * 100
                progress_callback(f"단답형 진행: {i+1}/{len(questions)} ({progress:.1f}%)")
            
            item_start = time.time()
            question_preview = q['question'][:50] + "..." if len(q['question']) > 50 else q['question']
            log_message("INFO", f"SHORT-{i+1} 처리 시작: '{question_preview}'")
            
            # 단답형 검색: BM25(5) + Vector(7) = 10개
            search_results = self.retriever.search(q['question'], question_type="short")
            
            # 검색 품질 분석
            search_quality = self._analyze_search_quality(search_results, "short")
            
            if len(search_results) == 0:
                log_message("FAILURE", f"SHORT-{i+1} 심각한 문제: 검색 결과 없음")
                self.eval_stats['short_error_patterns']['no_search_results'] += 1
                
                results.append(self._create_failed_short_result(q, "no_search_results"))
                continue
            
            if search_quality['bm25_max'] < 0.4:  # 임계값 0.5→0.4로 조정
                self.eval_stats['short_error_patterns']['low_bm25_score'] += 1
                log_message("FAILURE", f"SHORT-{i+1} BM25 점수 부족: {search_quality['bm25_max']:.2f}")
            
            log_message("INFO", f"SHORT-{i+1} 검색 완료: {len(search_results)}개 "
                       f"(BM25최고={search_quality['bm25_max']:.2f}, Vec최고={search_quality['vector_max']:.2f})")
            
            # 단답형 파이프라인
            predicted, pipeline_info = self._process_short_pipeline_enhanced(q['question'], search_results)
            
            # 정답 비교
            correct_answer = str(q.get('answer', '정보부족')).strip()
            
            # ★ 강화된 법령 용어 정규화 ★
            predicted_normalized = self._normalize_legal_answer_enhanced(predicted)
            correct_normalized = self._normalize_legal_answer_enhanced(correct_answer)
            
            # EM/F1 계산
            em_score = calculate_enhanced_exact_match(predicted_normalized, correct_normalized)
            f1_score = calculate_enhanced_f1_score(predicted_normalized, correct_normalized)
            
            item_time = time.time() - item_start
            short_times.append(item_time)
            
            if em_score:
                em_correct += 1
                log_message("SUCCESS", f"SHORT-{i+1} 정답: '{predicted}' (EM=1.0, F1={f1_score:.2f})")
            else:
                # ★ 강화된 실패 원인 분석 ★
                error_type = self._analyze_short_error_enhanced(
                    predicted, predicted_normalized, correct_normalized, 
                    search_quality, pipeline_info
                )
                self.eval_stats['short_error_patterns'][error_type] += 1
                
                log_message("FAILURE", f"SHORT-{i+1} 오답: 예측='{predicted}', 정답='{correct_answer}', "
                           f"오류유형={error_type}, EM={em_score}, F1={f1_score:.2f}")
            
            f1_total += f1_score
            
            # 결과 저장 (확장된 정보)
            results.append({
                'question': q['question'][:100],
                'predicted': predicted,
                'predicted_normalized': predicted_normalized,
                'correct': correct_answer,
                'correct_normalized': correct_normalized,
                'em_score': 1.0 if em_score else 0.0,
                'f1_score': f1_score,
                'response_time': item_time,
                'search_results_count': len(search_results),
                'bm25_max_score': search_quality['bm25_max'],
                'vector_max_score': search_quality['vector_max'],
                'pipeline_method': pipeline_info.get('method', 'standard'),
                'chunks_used': pipeline_info.get('chunks_used', 0),
                'error_type': error_type if not em_score else 'correct',
                'normalization_changed': predicted != predicted_normalized  # 신규
            })
        
        em_avg = em_correct / len(questions) if questions else 0
        f1_avg = f1_total / len(questions) if questions else 0
        total_time = time.time() - start_time
        
        # 성능 메트릭 업데이트
        if short_times:
            self.eval_stats['performance_metrics']['avg_short_time'] = sum(short_times) / len(short_times)
        self.eval_stats['performance_metrics']['total_time'] += total_time
        
        # 오답 패턴 분석 결과 출력
        self._log_error_pattern_analysis("SHORT", len(questions) - em_correct)
        
        log_message("SUCCESS", f"단답형 완료: EM={em_avg:.1%}, F1={f1_avg:.1%} - {total_time:.1f}초")
        
        return em_avg, f1_avg, results

    def evaluate_file(self, file_path: str, mcq_limit: int = None, short_limit: int = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """전체 파일 평가 - 통계 정보 강화"""
        log_message("INFO", f"파일 평가 시작: {Path(file_path).name}")
        
        eval_start_time = time.time()
        self.eval_stats['total_questions'] = 0
        
        # 데이터 로드
        log_message("INFO", "데이터 로드 중...")
        mcq_questions, short_questions = load_excel_data(file_path, mcq_limit, short_limit)
        
        if not mcq_questions and not short_questions:
            log_message("FAILURE", "평가할 질문이 없습니다.")
            return {}
        
        total_questions = len(mcq_questions) + len(short_questions)
        self.eval_stats['total_questions'] = total_questions
        
        log_message("SUCCESS", f"데이터 로드 완료: MCQ {len(mcq_questions)}개, 단답형 {len(short_questions)}개")
        
        # MCQ 평가
        mcq_accuracy = 0.0
        mcq_results = []
        if mcq_questions:
            log_message("INFO", "MCQ 평가 실행 중...")
            mcq_accuracy, mcq_results = self.evaluate_mcq_batch(
                mcq_questions, mcq_limit, progress_callback
            )
        
        # 단답형 평가
        short_em = short_f1 = 0.0
        short_results = []
        if short_questions:
            log_message("INFO", "단답형 평가 실행 중...")
            short_em, short_f1, short_results = self.evaluate_short_batch(
                short_questions, short_limit, progress_callback
            )
        
        total_eval_time = time.time() - eval_start_time
        self.eval_stats['performance_metrics']['total_time'] = total_eval_time
        
        # ★ 검색 성공률 계산 ★
        total_searches = len(mcq_results) + len(short_results)
        successful_searches = sum(1 for r in mcq_results + short_results 
                                if r.get('search_results_count', 0) > 0)
        search_success_rate = successful_searches / total_searches if total_searches > 0 else 0
        self.eval_stats['performance_metrics']['search_success_rate'] = search_success_rate
        
        # 결과 정리
        evaluation_results = {
            'mcq_accuracy': mcq_accuracy,
            'short_em': short_em,
            'short_f1': short_f1,
            'mcq_total': len(mcq_results),
            'short_total': len(short_results),
            'total_questions': len(mcq_results) + len(short_results),
            'mcq_results': mcq_results,
            'short_results': short_results,
            'source_file': Path(file_path).name,
            'total_time': total_eval_time,
            'evaluation_stats': self.eval_stats.copy(),  # 통계 정보 추가
            'performance_summary': self._generate_performance_summary()  # 신규
        }
        
        self._print_comprehensive_summary(evaluation_results)
        log_message("SUCCESS", "전체 평가 완료")
        return evaluation_results

    def _analyze_mcq_error_enhanced(self, search_quality: Dict, context_quality: float,
                                  is_negative: bool, question: str, choices: Dict, 
                                  llm_response: str, predicted: str, correct: str) -> str:
        """강화된 MCQ 오답 원인 분석"""
        
        # 1. 파싱/매핑 오류 검증
        if predicted not in ['A', 'B', 'C', 'D', '1', '2', '3', '4']:
            return 'choice_mapping'
        
        # 2. LLM 응답 분석 - 파싱은 성공했지만 추론 오류
        if len(llm_response) > 10:  # 설명이 길면
            # 응답에 정답 선택지 내용이 포함되어 있는지 확인
            correct_choice_text = choices.get(correct, '')
            if correct_choice_text and correct_choice_text[:20] in llm_response:
                return 'llm_reasoning'  # LLM이 정답을 알고 있었지만 잘못 매핑
        
        # 3. 검색 실패형
        if search_quality['bm25_max'] < 5.0 and search_quality['vector_max'] < 1.0:
            return 'search_failure'
        
        # 4. 부정형 질문 감지 실패
        if is_negative and not self._check_negative_processing_in_context(choices):
            return 'negative_detection'
        
        # 5. 컨텍스트 품질 문제
        if context_quality < 0.4:
            return 'context_quality'
        
        # 6. 기타 매핑 오류 (기본)
        return 'choice_mapping'

    def _analyze_short_error_enhanced(self, predicted: str, predicted_norm: str, 
                                    correct_norm: str, search_quality: Dict, 
                                    pipeline_info: Dict) -> str:
        """강화된 단답형 오답 원인 분석"""
        
        # 1. 추출 실패 (완전히 실패한 경우)
        if len(predicted.strip()) <= 3 or "정보 부족" in predicted or "처리 실패" in predicted:
            return 'extraction_failure'
        
        # 2. 정규화 실패 (추출은 했지만 정규화에서 손실)
        if predicted != predicted_norm and len(predicted_norm) < len(predicted) * 0.5:
            return 'normalization_failure'
        
        # 3. BM25 점수 부족
        if search_quality['bm25_max'] < 0.4:  # 임계값 0.5→0.4로 조정
            return 'low_bm25_score'
        
        # 4. 컨텍스트 불일치 (기본)
        return 'context_mismatch'

    def _normalize_legal_answer_enhanced(self, text: str) -> str:
        """강화된 법령 답변 정규화"""
        if not text:
            return ""
        
        # 기존 정규화 + 추가 개선
        from rag.utils import enhanced_answer_normalize
        normalized = enhanced_answer_normalize(text)
        
        # 추가 정규화 (평가기 전용)
        additional_synonyms = {
            "규제신속확인절차": "법령적용여부확인절차",
            "신속확인절차": "법령적용여부확인절차",
            "규제확인": "법령적용여부확인",
            "샌드박스": "임시허가",
            "규제샌드박스": "임시허가"
        }
        
        for old, new in additional_synonyms.items():
            normalized = normalized.replace(old, new)
        
        return normalized

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 생성 (신규)"""
        metrics = self.eval_stats['performance_metrics']
        
        return {
            'avg_time_per_question': (metrics['avg_mcq_time'] + metrics['avg_short_time']) / 2,
            'total_evaluation_time': metrics['total_time'],
            'search_success_rate': metrics['search_success_rate'],
            'questions_per_minute': (self.eval_stats['total_questions'] / 
                                   max(1, metrics['total_time'])) * 60,
            'estimated_time_1000q': (metrics['total_time'] / 
                                   max(1, self.eval_stats['total_questions'])) * 1000
        }

    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """포괄적 평가 결과 요약 (강화)"""
        total_q = results.get('total_questions', 0)
        mcq_acc = results.get('mcq_accuracy', 0)
        short_em = results.get('short_em', 0)
        short_f1 = results.get('short_f1', 0)
        total_time = results.get('total_time', 0)
        
        # 통계 정보 출력
        stats = results.get('evaluation_stats', {})
        search_issues = stats.get('search_quality_issues', 0)
        perf_summary = results.get('performance_summary', {})
        
        log_message("SUCCESS", "="*60)
        log_message("SUCCESS", f"최종 결과: 총 {total_q}문제")
        log_message("SUCCESS", f"MCQ 정확도: {mcq_acc:.1%}")
        log_message("SUCCESS", f"단답형 EM: {short_em:.1%}, F1: {short_f1:.1%}")
        log_message("SUCCESS", f"실행 시간: {total_time:.1f}초")
        log_message("INFO", f"검색 품질 문제: {search_issues}건")
        log_message("INFO", f"검색 성공률: {perf_summary.get('search_success_rate', 0):.1%}")
        log_message("INFO", f"분당 처리량: {perf_summary.get('questions_per_minute', 0):.1f}문제/분")
        log_message("SUCCESS", "="*60)

    # 기존 함수들 유지...
    def _analyze_search_quality(self, search_results: List[SearchResult], question_type: str) -> Dict[str, Any]:
        """검색 품질 분석"""
        quality_info = {
            'bm25_max': 0.0,
            'vector_max': 0.0,
            'has_issues': False,
            'issues': []
        }
        
        if not search_results:
            quality_info['has_issues'] = True
            quality_info['issues'].append("검색결과없음")
            return quality_info
        
        bm25_scores = []
        vector_scores = []
        metadata_issues = 0
        
        for r in search_results:
            if hasattr(r, 'metadata') and r.metadata:
                bm25_scores.append(r.metadata.get('bm25_contribution', 0))
                vector_scores.append(r.metadata.get('vector_contribution', 0))
            else:
                bm25_scores.append(0)
                vector_scores.append(0)
                metadata_issues += 1
        
        quality_info['bm25_max'] = max(bm25_scores) if bm25_scores else 0
        quality_info['vector_max'] = max(vector_scores) if vector_scores else 0
        
        # 품질 문제 감지
        if metadata_issues > len(search_results) // 2:
            quality_info['has_issues'] = True
            quality_info['issues'].append(f"메타데이터부족({metadata_issues}개)")
        
        if quality_info['bm25_max'] == 0 and quality_info['vector_max'] == 0:
            quality_info['has_issues'] = True
            quality_info['issues'].append("모든점수0")
        
        return quality_info

    def _log_error_pattern_analysis(self, question_type: str, total_errors: int):
        """오답 패턴 분석 결과 로그"""
        if total_errors <= 0:
            return
        
        if question_type == "MCQ":
            patterns = self.eval_stats['mcq_error_patterns']
        else:
            patterns = self.eval_stats['short_error_patterns']
        
        error_summary = f"{question_type} 오답 패턴 분석 (총 {total_errors}개): "
        error_details = []
        
        for pattern, count in patterns.items():
            if count > 0:
                percentage = (count / total_errors) * 100
                error_details.append(f"{pattern}={count}개({percentage:.1f}%)")
        
        if error_details:
            error_summary += ", ".join(error_details)
            log_message("INFO", error_summary)

    def _create_failed_short_result(self, question: Dict, error_type: str) -> Dict:
        """실패한 단답형 결과 생성"""
        return {
            'question': question['question'][:100],
            'predicted': "검색실패",
            'predicted_normalized': "검색실패",
            'correct': question.get('answer', '정보부족'),
            'correct_normalized': question.get('answer', '정보부족'),
            'em_score': 0.0,
            'f1_score': 0.0,
            'response_time': 0.0,
            'search_results_count': 0,
            'bm25_max_score': 0.0,
            'vector_max_score': 0.0,
            'pipeline_method': "failed",
            'chunks_used': 0,
            'error_type': error_type,
            'normalization_changed': False
        }

    def _detect_negative_question(self, question: str) -> bool:
        """부정형 질문 감지"""
        negative_indicators = [
            "포함되지 않는", "해당하지 않는", "맞지 않는", "아닌 것",
            "제외되는", "틀린 것", "잘못된 것", "예외"
        ]
        
        for indicator in negative_indicators:
            if indicator in question:
                return True
        
        if re.search(r"다음.*?중.*?(?:아닌|않은|없는)", question):
            return True
        
        return False

    def _check_negative_processing_in_context(self, choices: Dict) -> bool:
        """부정형 질문 처리가 제대로 되었는지 확인"""
        # 간단한 휴리스틱: 대부분의 선택지가 언급되었는지 확인
        return len(choices) >= 3  # 임시 구현

    def _select_mcq_context(self, question: str, search_results: List[SearchResult]) -> str:
        """MCQ 컨텍스트 선택"""
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."
        
        selected_chunks = search_results[:3]
        context_parts = []
        
        for i, chunk in enumerate(selected_chunks, 1):
            context_parts.append(f"[참고자료 {i}]\n{chunk.content}")
        
        return "\n\n".join(context_parts)

    def _process_short_pipeline_enhanced(self, question: str, search_results: List[SearchResult]) -> Tuple[str, Dict]:
        """향상된 단답형 파이프라인"""
        if not search_results:
            return "정보 부족", {'method': 'no_results', 'chunks_used': 0}
        
        # BM25 점수 기반 동적 청크 개수 결정
        bm25_scores = []
        for r in search_results:
            if hasattr(r, 'metadata') and r.metadata:
                bm25_scores.append(r.metadata.get('bm25_contribution', 0))
            else:
                bm25_scores.append(0)
        
        bm25_max = max(bm25_scores) if bm25_scores else 0
        
        # 조문 참조 질문 감지
        is_article_ref = "제" in question and "조에서" in question
        
        # 동적 청크 개수 결정
        if bm25_max > 100:
            chunk_count = 5
        elif is_article_ref:
            chunk_count = 4
        else:
            chunk_count = 3
        
        # 상위 청크들로 컨텍스트 구성
        context_parts = []
        for i, result in enumerate(search_results[:chunk_count]):
            context_parts.append(result.content)
        
        full_context = "\n".join(context_parts)
        
        # LLM 호출
        try:
            predicted = self.llm.call_short(question, full_context)
        except Exception as e:
            log_message("FAILURE", f"LLM 호출 실패: {e}")
            predicted = "처리 실패"
        
        pipeline_info = {
            'method': 'enhanced',
            'chunks_used': chunk_count,
            'bm25_max': bm25_max,
            'is_article_ref': is_article_ref
        }
        
        return predicted, pipeline_info

    def _assess_context_quality(self, context: str, question: str) -> float:
        """컨텍스트 품질 평가"""
        if not context or len(context.strip()) < 20:
            return 0.0
        
        # 질문 키워드 추출
        question_keywords = set(re.findall(r'[가-힣]{3,}|\d+(?:년|개월|일)|\d+(?:억|만)?원|제\d+조', question))
        context_keywords = set(re.findall(r'[가-힣]{3,}|\d+(?:년|개월|일)|\d+(?:억|만)?원|제\d+조', context))
        
        if not question_keywords:
            return 0.5
        
        # 키워드 매칭률
        overlap = len(question_keywords & context_keywords)
        keyword_score = overlap / len(question_keywords) if question_keywords else 0
        
        # 컨텍스트 길이 점수
        if 100 <= len(context) <= 1000:
            length_score = 1.0
        elif 50 <= len(context) < 100 or 1000 < len(context) <= 2000:
            length_score = 0.7
        else:
            length_score = 0.3
        
        # 법령 패턴 점수
        legal_count = sum(1 for pattern in ['제', '조', '항', '호', '법률'] if pattern in context)
        legal_score = min(1.0, legal_count / 3)
        
        return min(1.0, keyword_score * 0.5 + length_score * 0.3 + legal_score * 0.2)

    def save_results(self, results: Dict[str, Any], output_file: str = None, source_file: str = None) -> str:
        """결과 저장"""
        try:
            from rag.utils import save_evaluation_results
            saved_file = save_evaluation_results(results, output_file)
            if saved_file:
                log_message("SUCCESS", f"결과 저장 완료: {saved_file}")
            else:
                log_message("FAILURE", "결과 저장 실패")
            return saved_file
        except Exception as e:
            log_message("FAILURE", f"결과 저장 오류: {e}")
            return None