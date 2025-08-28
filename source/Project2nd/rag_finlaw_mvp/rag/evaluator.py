"""
evaluator.py - 로그 시스템 통합 및 디버깅 정보 강화
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
    print(f"[{module}-{log_type.upper()}] {message}")
    
    # 웹 인터페이스로 전달 시도
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'global_log_callback'):
            callback = st.session_state.global_log_callback
            if callable(callback):
                callback(log_type, message, module, "evaluation")
    except Exception:
        pass

class UnifiedEvaluator:
    """디버깅 정보 강화된 평가기"""
    
    def __init__(self, config=None, retriever=None, llm=None):
        log_message("INFO", "평가기 초기화 중...")
        
        if retriever and llm:
            self.retriever = retriever
            self.llm = llm
            self.config = config if config else getattr(retriever, 'config', None)
            log_message("SUCCESS", "기존 RAG 인스턴스 재사용")
        else:
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
        
        # 평가 통계 초기화
        self.eval_stats = {
            'mcq_error_patterns': {
                'search_failure': 0,
                'choice_mapping': 0,
                'context_quality': 0,
                'negative_detection': 0
            },
            'short_error_patterns': {
                'no_search_results': 0,
                'low_bm25_score': 0,
                'extraction_failure': 0,
                'context_mismatch': 0
            },
            'search_quality_issues': 0,
            'total_questions': 0
        }
        
        log_message("SUCCESS", "초기화 완료")

    def evaluate_mcq_batch(self, questions: List[Dict], max_questions: int = None, 
                          progress_callback: Optional[Callable] = None) -> Tuple[float, List[Dict]]:
        """MCQ 배치 평가 - 강화된 디버깅 로그"""
        if max_questions:
            questions = questions[:max_questions]
        
        if not questions:
            log_message("FAILURE", "MCQ 질문이 없음")
            return 0.0, []
        
        log_message("INFO", f"MCQ 평가 시작: {len(questions)}개 문제")
        start_time = time.time()
        
        correct = 0
        results = []
        
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
            
            # 답변 파싱
            expected_format = q.get('answer_format', 'number')
            predicted = parse_mcq_answer(llm_response, expected_format)
            correct_answer = str(q.get('answer', '1')).strip()
            
            is_correct = (predicted == correct_answer)
            if is_correct:
                correct += 1
                log_message("SUCCESS", f"MCQ-{i+1} 정답: {predicted}")
            else:
                # 오답 원인 분석
                error_type = self._analyze_mcq_error(search_quality, context_quality, 
                                                   is_negative_question, q['question'], 
                                                   choices_dict, predicted, correct_answer)
                self.eval_stats['mcq_error_patterns'][error_type] += 1
                
                log_message("FAILURE", f"MCQ-{i+1} 오답: 예측={predicted}, 정답={correct_answer}, "
                           f"오류유형={error_type}, 컨텍스트품질={context_quality:.2f}")
            
            item_time = time.time() - item_start
            
            # 결과 저장
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
                'context_preview': context[:200]
            })
        
        accuracy = correct / len(questions)
        total_time = time.time() - start_time
        
        # 오답 패턴 분석 결과 출력
        self._log_error_pattern_analysis("MCQ", len(questions) - correct)
        
        log_message("SUCCESS", f"MCQ 완료: 정확도 {accuracy:.1%} ({correct}/{len(questions)}) - {total_time:.1f}초")
        
        return accuracy, results

    def evaluate_short_batch(self, questions: List[Dict], max_questions: int = None,
                            progress_callback: Optional[Callable] = None) -> Tuple[float, float, List[Dict]]:
        """단답형 배치 평가 - 강화된 디버깅 로그"""
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
            
            if search_quality['bm25_max'] < 0.5:
                self.eval_stats['short_error_patterns']['low_bm25_score'] += 1
                log_message("FAILURE", f"SHORT-{i+1} BM25 점수 부족: {search_quality['bm25_max']:.2f}")
            
            log_message("INFO", f"SHORT-{i+1} 검색 완료: {len(search_results)}개 "
                       f"(BM25최고={search_quality['bm25_max']:.2f}, Vec최고={search_quality['vector_max']:.2f})")
            
            # 단답형 파이프라인
            predicted, pipeline_info = self._process_short_pipeline_enhanced(q['question'], search_results)
            
            # 정답 비교
            correct_answer = str(q.get('answer', '정보부족')).strip()
            
            # 법령 용어 정규화
            predicted_normalized = self._normalize_legal_answer(predicted)
            correct_normalized = self._normalize_legal_answer(correct_answer)
            
            # EM/F1 계산
            em_score = calculate_enhanced_exact_match(predicted_normalized, correct_normalized)
            f1_score = calculate_enhanced_f1_score(predicted_normalized, correct_normalized)
            
            if em_score:
                em_correct += 1
                log_message("SUCCESS", f"SHORT-{i+1} 정답: '{predicted}' (EM=1.0, F1={f1_score:.2f})")
            else:
                # 실패 원인 분석
                error_type = self._analyze_short_error(predicted, search_quality, pipeline_info)
                self.eval_stats['short_error_patterns'][error_type] += 1
                
                log_message("FAILURE", f"SHORT-{i+1} 오답: 예측='{predicted}', 정답='{correct_answer}', "
                           f"오류유형={error_type}, EM={em_score}, F1={f1_score:.2f}")
            
            f1_total += f1_score
            item_time = time.time() - item_start
            
            # 결과 저장
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
                'error_type': error_type if not em_score else 'correct'
            })
        
        em_avg = em_correct / len(questions) if questions else 0
        f1_avg = f1_total / len(questions) if questions else 0
        total_time = time.time() - start_time
        
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
            'evaluation_stats': self.eval_stats.copy()  # 통계 정보 추가
        }
        
        self._print_final_summary(evaluation_results)
        log_message("SUCCESS", "전체 평가 완료")
        return evaluation_results

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

    def _analyze_mcq_error(self, search_quality: Dict, context_quality: float,
                          is_negative: bool, question: str, choices: Dict, 
                          predicted: str, correct: str) -> str:
        """MCQ 오답 원인 분석"""
        # 1. 검색 실패형
        if search_quality['bm25_max'] < 5.0 and search_quality['vector_max'] < 1.0:
            return 'search_failure'
        
        # 2. 부정형 질문 감지 실패
        if is_negative and not self._check_negative_processing_in_context(choices):
            return 'negative_detection'
        
        # 3. 컨텍스트 품질 문제
        if context_quality < 0.4:
            return 'context_quality'
        
        # 4. 선택지 매핑 오류
        return 'choice_mapping'

    def _analyze_short_error(self, predicted: str, search_quality: Dict, pipeline_info: Dict) -> str:
        """단답형 오답 원인 분석"""
        if len(predicted.strip()) <= 3 or "정보 부족" in predicted:
            return 'extraction_failure'
        
        if search_quality['bm25_max'] < 0.5:
            return 'low_bm25_score'
        
        return 'context_mismatch'

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
            'error_type': error_type
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

    def _normalize_legal_answer(self, text: str) -> str:
        """법령 답변 정규화"""
        if not text:
            return ""
        
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'제\s*(\d+)\s*조', r'제\1조', text)
        text = re.sub(r'(\d+)\s*(년|개월|일)', r'\1\2', text)
        text = re.sub(r'(\d+)\s*(억|만)?\s*원', r'\1\2원', text)
        
        # 동의어 처리
        for old_term, new_term in self.legal_synonyms.items():
            text = text.replace(old_term, new_term)
        
        # 불필요한 문구 제거
        for phrase in ['답변:', '답:', '정답:', '결론:', '따라서']:
            text = text.replace(phrase, '')
        
        return text.strip()

    def _print_final_summary(self, results: Dict[str, Any]):
        """최종 평가 결과 요약"""
        total_q = results.get('total_questions', 0)
        mcq_acc = results.get('mcq_accuracy', 0)
        short_em = results.get('short_em', 0)
        short_f1 = results.get('short_f1', 0)
        total_time = results.get('total_time', 0)
        
        # 통계 정보 출력
        stats = results.get('evaluation_stats', {})
        search_issues = stats.get('search_quality_issues', 0)
        
        log_message("SUCCESS", "="*60)
        log_message("SUCCESS", f"최종 결과: 총 {total_q}문제")
        log_message("SUCCESS", f"MCQ 정확도: {mcq_acc:.1%}")
        log_message("SUCCESS", f"단답형 EM: {short_em:.1%}, F1: {short_f1:.1%}")
        log_message("SUCCESS", f"실행 시간: {total_time:.1f}초")
        log_message("INFO", f"검색 품질 문제: {search_issues}건")
        log_message("SUCCESS", "="*60)

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