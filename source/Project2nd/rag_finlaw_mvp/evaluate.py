#!/usr/bin/env python
"""
evaluate.py - 성능 강화 최종 버전 (오류 수정)
목표:
- MCQ 정확도: 52.6% -> 65%+
- 단답형 EM: 14.1% -> 20%+
- 의미있는 성능 지표 추가
- 실시간 상세 모니터링
"""

import os
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# 환경 변수 설정
os.environ["DEBUG_HYBRID"] = "false"
os.environ["VERBOSE_LOGGING"] = "false"

import gc
gc.set_threshold(700, 10, 10)

# 필수 import를 상단으로 이동 (스코프 문제 해결)
from rag.utils import (
    load_excel, save_results, sample_questions, 
    analyze_dataset_performance, clear_all_caches,
    analyze_failure_patterns, get_performance_insights,
    score_short, calculate_accuracy  # 여기서 미리 import
)
from rag.evaluator import HighPerformanceEvaluator as Evaluator
from rag import retriever

def analyze_comprehensive_results(mcq_results: List[Dict], short_results: List[Dict]) -> Dict:
    """종합적인 결과 분석 - 강화된 버전"""
    analysis = {
        "total_questions": len(mcq_results) + len(short_results),
        "mcq_analysis": {},
        "short_analysis": {},
        "performance_insights": {},
        "improvement_analysis": {},
        "quality_analysis": {},
        "recommendations": []
    }
    
    # MCQ 상세 분석
    if mcq_results:
        mcq_acc = calculate_accuracy(mcq_results)
        
        # 난이도별 성능 분석
        difficulty_performance = {}
        context_quality_scores = []
        confidence_indicators = []
        
        for r in mcq_results:
            diff = r.get('metadata', {}).get('difficulty', '미분류')
            context_quality = r.get('context_quality', 0)
            search_score = r.get('search_score', 0)
            
            context_quality_scores.append(context_quality)
            confidence_indicators.append(search_score)
            
            if diff not in difficulty_performance:
                difficulty_performance[diff] = {
                    'total': 0, 'correct': 0, 'avg_context_quality': 0, 'quality_sum': 0
                }
            
            difficulty_performance[diff]['total'] += 1
            difficulty_performance[diff]['quality_sum'] += context_quality
            
            if r.get('prediction') == r.get('answer'):
                difficulty_performance[diff]['correct'] += 1
        
        # 컨텍스트 활용도 분석
        context_utilized = 0
        high_confidence_selections = 0
        
        for r in mcq_results:
            # 선택된 답이 검색 컨텍스트에서 발견되었는지 추정
            if r.get('search_score', 0) > 1.0:
                context_utilized += 1
            if r.get('search_score', 0) > 1.5:
                high_confidence_selections += 1
        
        analysis["mcq_analysis"] = {
            "accuracy": mcq_acc,
            "total": len(mcq_results),
            "correct": sum(1 for r in mcq_results if r.get('prediction') == r.get('answer')),
            "avg_context_quality": sum(context_quality_scores) / len(context_quality_scores) if context_quality_scores else 0,
            "context_utilization_rate": context_utilized / len(mcq_results),
            "high_confidence_rate": high_confidence_selections / len(mcq_results),
            "difficulty_performance": {
                diff: {
                    'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
                    'avg_context_quality': data['quality_sum'] / data['total'] if data['total'] > 0 else 0
                }
                for diff, data in difficulty_performance.items()
            }
        }
    
    # 단답형 상세 분석 - 대폭 강화
    if short_results:
        ems, f1s = [], []
        info_insufficient_count = 0
        exact_matches = 0
        partial_matches = 0
        generation_failures = 0
        context_quality_scores = []
        
        # 질문 유형별 성능 분석
        question_type_performance = {}
        answer_type_distribution = {}
        
        for r in short_results:
            prediction = r.get('prediction', '')
            answer = r.get('answer', '')
            context_quality = r.get('search_score', 0)
            question_type = r.get('question_type', 'general')
            answer_type = r.get('answer_type', 'unknown')
            
            context_quality_scores.append(context_quality)
            
            # 질문 유형별 통계
            if question_type not in question_type_performance:
                question_type_performance[question_type] = {
                    'total': 0, 'em_sum': 0, 'f1_sum': 0, 'exact_count': 0
                }
            question_type_performance[question_type]['total'] += 1
            
            # 답변 유형별 분포
            answer_type_distribution[answer_type] = answer_type_distribution.get(answer_type, 0) + 1
            
            if prediction == "정보 불충분":
                info_insufficient_count += 1
                ems.append(0.0)
                f1s.append(0.0)
            elif not prediction or not prediction.strip():
                generation_failures += 1
                ems.append(0.0)
                f1s.append(0.0)
            else:
                em, f1 = score_short(prediction, answer)
                ems.append(em)
                f1s.append(f1)
                
                question_type_performance[question_type]['em_sum'] += em
                question_type_performance[question_type]['f1_sum'] += f1
                
                if em >= 0.9:
                    exact_matches += 1
                    question_type_performance[question_type]['exact_count'] += 1
                elif em >= 0.3:
                    partial_matches += 1
        
        avg_em = sum(ems) / len(ems) if ems else 0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0
        avg_context_quality = sum(context_quality_scores) / len(context_quality_scores) if context_quality_scores else 0
        
        # EM 점수 분포 분석
        em_distribution = {
            'perfect': sum(1 for em in ems if em >= 0.95),
            'excellent': sum(1 for em in ems if 0.8 <= em < 0.95),
            'good': sum(1 for em in ems if 0.5 <= em < 0.8),
            'fair': sum(1 for em in ems if 0.2 <= em < 0.5),
            'poor': sum(1 for em in ems if em < 0.2)
        }
        
        analysis["short_analysis"] = {
            "em_score": avg_em,
            "f1_score": avg_f1,
            "total": len(short_results),
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "info_insufficient": info_insufficient_count,
            "generation_failures": generation_failures,
            "info_insufficient_rate": info_insufficient_count / len(short_results),
            "exact_match_rate": exact_matches / len(short_results),
            "success_rate": (exact_matches + partial_matches) / len(short_results),
            "avg_context_quality": avg_context_quality,
            "em_distribution": em_distribution,
            "question_type_performance": {
                qtype: {
                    'avg_em': data['em_sum'] / data['total'] if data['total'] > 0 else 0,
                    'avg_f1': data['f1_sum'] / data['total'] if data['total'] > 0 else 0,
                    'exact_rate': data['exact_count'] / data['total'] if data['total'] > 0 else 0,
                    'total': data['total']
                }
                for qtype, data in question_type_performance.items()
            },
            "answer_type_distribution": answer_type_distribution
        }
        
        # 성능 인사이트 생성
        insights = []
        
        if avg_em >= 0.20:
            insights.append("단답형 EM 목표 달성! (20% 이상)")
        elif avg_em >= 0.15:
            insights.append("단답형 EM 목표에 근접 (15% 이상)")
        else:
            insights.append("단답형 EM 추가 개선 필요")
        
        if info_insufficient_count == 0:
            insights.append("정보 불충분 문제 완전 해결!")
        elif analysis["short_analysis"]["info_insufficient_rate"] <= 0.05:
            insights.append("정보 불충분 비율 매우 낮음 (5% 이하)")
        else:
            insights.append(f"정보 불충분 비율: {analysis['short_analysis']['info_insufficient_rate']:.1%}")
        
        if avg_context_quality >= 1.5:
            insights.append("컨텍스트 품질 우수")
        elif avg_context_quality >= 1.0:
            insights.append("컨텍스트 품질 양호")
        else:
            insights.append("컨텍스트 품질 개선 필요")
        
        analysis["performance_insights"] = insights
        
        # 개선도 분석 (이전 성과 대비)
        baseline_em = 0.05  # 5%
        baseline_info_rate = 0.415  # 41.5%
        
        em_improvement = ((avg_em - baseline_em) / baseline_em) * 100 if baseline_em > 0 else 0
        info_improvement = ((baseline_info_rate - analysis["short_analysis"]["info_insufficient_rate"]) / baseline_info_rate) * 100
        
        analysis["improvement_analysis"] = {
            "em_improvement_percent": em_improvement,
            "info_insufficient_improvement_percent": info_improvement,
            "baseline_em": baseline_em,
            "current_em": avg_em,
            "baseline_info_rate": baseline_info_rate,
            "current_info_rate": analysis["short_analysis"]["info_insufficient_rate"],
            "performance_gain": "매우 우수" if em_improvement > 200 else "우수" if em_improvement > 100 else "양호"
        }
        
        # 품질 분석
        analysis["quality_analysis"] = {
            "avg_mcq_context_quality": analysis["mcq_analysis"].get("avg_context_quality", 0) if mcq_results else 0,
            "avg_short_context_quality": avg_context_quality,
            "overall_quality_score": (analysis["mcq_analysis"].get("avg_context_quality", 0) + avg_context_quality) / 2 if mcq_results else avg_context_quality
        }
        
        # 향상된 추천사항 생성
        recommendations = []
        
        # MCQ 관련 추천
        if mcq_results:
            mcq_acc = analysis["mcq_analysis"]["accuracy"]
            if mcq_acc >= 0.65:
                recommendations.append("MCQ 정확도 목표 달성! (65% 이상)")
            elif mcq_acc >= 0.60:
                recommendations.append("MCQ 정확도 목표에 근접, 미세 조정 필요")
            else:
                recommendations.append(f"MCQ 정확도 개선 필요: 현재 {mcq_acc:.1%} -> 목표 65%")
                
                # 구체적 개선 방안 제시
                context_util = analysis["mcq_analysis"]["context_utilization_rate"]
                if context_util < 0.7:
                    recommendations.append("MCQ 컨텍스트 활용도 개선 필요 (검색 품질 향상)")
                
                high_conf = analysis["mcq_analysis"]["high_confidence_rate"]
                if high_conf < 0.5:
                    recommendations.append("MCQ 고신뢰도 선택률 개선 필요 (선택 로직 강화)")
        
        # 단답형 관련 추천
        if avg_em >= 0.20:
            recommendations.append("단답형 EM 목표 달성!")
        else:
            gap = 0.20 - avg_em
            recommendations.append(f"단답형 EM 개선 필요: {gap:.1%}p 향상 목표")
            
            # 구체적 개선 방안
            if exact_matches / len(short_results) < 0.15:
                recommendations.append("정확한 답변 생성률 개선 필요 (답변 추출 로직 강화)")
            
            if info_insufficient_count > len(short_results) * 0.1:
                recommendations.append("정보 불충분 비율 추가 감소 필요 (검색 품질 향상)")
        
        # 전체적 성능 평가
        total_performance_score = 0
        if mcq_results:
            total_performance_score += analysis["mcq_analysis"]["accuracy"] * 50  # MCQ 50점
        if short_results:
            total_performance_score += avg_em * 50  # 단답형 50점
        
        if total_performance_score >= 80:
            recommendations.append("전체 시스템 성능 우수")
        elif total_performance_score >= 60:
            recommendations.append("전체 시스템 성능 양호, 미세 조정 필요")
        else:
            recommendations.append("전체 시스템 성능 개선 필요")
        
        analysis["recommendations"] = recommendations
    
    return analysis

def compare_with_enhanced_baseline(new_results: Dict, baseline_results: Dict = None):
    """향상된 베이스라인 비교"""
    if not baseline_results:
        baseline_results = {
            'mcq_accuracy': 0.526,         # 52.6%
            'mcq_context_utilization': 0.3, # 추정
            'mcq_high_confidence': 0.2,     # 추정
            'em_score': 0.141,              # 14.1%
            'f1_score': 0.147,              # 14.7%
            'success_rate': 0.141,          # 14.1%
            'info_insufficient_rate': 0.0,  # 0.0%
            'exact_match_rate': 0.1         # 추정
        }
    
    print("\n" + "="*80)
    print(" 성능 개선 비교 분석 (이전 실행 결과 vs 현재)")
    print("="*80)
    
    # MCQ 메트릭
    mcq_metrics = [
        ('MCQ 정확도', 'mcq_accuracy', '{:.1%}', 'accuracy'),
        ('MCQ 컨텍스트 활용률', 'mcq_context_utilization', '{:.1%}', 'context_utilization_rate'),
        ('MCQ 고신뢰도 선택률', 'mcq_high_confidence', '{:.1%}', 'high_confidence_rate')
    ]
    
    # 단답형 메트릭
    short_metrics = [
        ('단답형 EM 점수', 'em_score', '{:.1%}', 'em_score'),
        ('단답형 F1 점수', 'f1_score', '{:.1%}', 'f1_score'),
        ('단답형 정확 매칭률', 'exact_match_rate', '{:.1%}', 'exact_match_rate'),
        ('정보 불충분률', 'info_insufficient_rate', '{:.1%}', 'info_insufficient_rate')
    ]
    
    total_improvements = 0
    significant_improvements = 0
    
    print("MCQ 성능 지표:")
    mcq_analysis = new_results.get('mcq_analysis', {})
    for name, baseline_key, fmt, current_key in mcq_metrics:
        baseline_value = baseline_results.get(baseline_key, 0)
        current_value = mcq_analysis.get(current_key, 0)
        
        if baseline_value > 0:
            improvement = ((current_value - baseline_value) / baseline_value) * 100
            
            if improvement > 0:
                total_improvements += 1
                if improvement >= 50:
                    significant_improvements += 1
                    status = " (대폭 개선!)"
                elif improvement >= 20:
                    status = " (크게 개선)"
                elif improvement >= 10:
                    status = " (개선)"
                else:
                    status = " (소폭 개선)"
            elif improvement < -10:
                status = " (성능 저하)"
            else:
                status = " (유지)"
            
            print(f"  {name:20}: {fmt.format(baseline_value)} -> {fmt.format(current_value)} ({improvement:+.1f}%){status}")
        else:
            print(f"  {name:20}: {fmt.format(baseline_value)} -> {fmt.format(current_value)}")
    
    print("\n단답형 성능 지표:")
    short_analysis = new_results.get('short_analysis', {})
    for name, baseline_key, fmt, current_key in short_metrics:
        baseline_value = baseline_results.get(baseline_key, 0)
        current_value = short_analysis.get(current_key, 0)
        
        if baseline_key == 'info_insufficient_rate':
            # 낮을수록 좋음
            if baseline_value > 0:
                improvement = ((baseline_value - current_value) / baseline_value) * 100
            else:
                improvement = 0
        else:
            # 높을수록 좋음
            if baseline_value > 0:
                improvement = ((current_value - baseline_value) / baseline_value) * 100
            else:
                improvement = 0
        
        if improvement > 0:
            total_improvements += 1
            if improvement >= 50:
                significant_improvements += 1
                status = " (대폭 개선!)"
            elif improvement >= 20:
                status = " (크게 개선)"
            elif improvement >= 10:
                status = " (개선)"
            else:
                status = " (소폭 개선)"
        elif improvement < -10:
            status = " (성능 저하)"
        else:
            status = " (유지)"
        
        print(f"  {name:20}: {fmt.format(baseline_value)} -> {fmt.format(current_value)} ({improvement:+.1f}%){status}")
    
    print("="*80)
    print(f"개선된 지표: {total_improvements}/7개")
    print(f"대폭 개선된 지표: {significant_improvements}개")
    
    # 목표 달성 평가
    mcq_target = 0.65
    em_target = 0.20
    
    current_mcq = mcq_analysis.get('accuracy', 0)
    current_em = short_analysis.get('em_score', 0)
    
    print(f"\n목표 달성 현황:")
    print(f"  MCQ 정확도: {current_mcq:.1%} / {mcq_target:.1%} ({'달성!' if current_mcq >= mcq_target else f'{mcq_target-current_mcq:.1%}p 부족'})")
    print(f"  단답형 EM: {current_em:.1%} / {em_target:.1%} ({'달성!' if current_em >= em_target else f'{em_target-current_em:.1%}p 부족'})")

def print_comprehensive_performance_analysis(analysis: Dict):
    """종합적인 성능 분석 출력"""
    print("\n" + "="*80)
    print(" 종합 성능 분석 (강화된 버전)")
    print("="*80)
    
    # MCQ 분석
    if "mcq_analysis" in analysis and analysis["mcq_analysis"]:
        mcq = analysis["mcq_analysis"]
        print(f"\nMCQ 상세 성능 분석:")
        print(f"  전체 정확도: {mcq['accuracy']:.1%}")
        print(f"  정답 수: {mcq['correct']}/{mcq['total']}")
        print(f"  평균 컨텍스트 품질: {mcq['avg_context_quality']:.3f}")
        print(f"  컨텍스트 활용률: {mcq['context_utilization_rate']:.1%}")
        print(f"  고신뢰도 선택률: {mcq['high_confidence_rate']:.1%}")
        
        if "difficulty_performance" in mcq:
            print(f"  난이도별 성능:")
            for diff, perf in mcq["difficulty_performance"].items():
                print(f"    {diff}: 정확도 {perf['accuracy']:.1%}, 품질 {perf['avg_context_quality']:.3f}")
    
    # 단답형 분석
    if "short_analysis" in analysis and analysis["short_analysis"]:
        short = analysis["short_analysis"]
        print(f"\n단답형 상세 성능 분석:")
        print(f"  EM 점수: {short['em_score']:.1%}")
        print(f"  F1 점수: {short['f1_score']:.1%}")
        print(f"  평균 컨텍스트 품질: {short['avg_context_quality']:.3f}")
        print(f"  정확 매칭: {short['exact_matches']}개 ({short['exact_match_rate']:.1%})")
        print(f"  부분 매칭: {short['partial_matches']}개")
        print(f"  정보 불충분: {short['info_insufficient']}개 ({short['info_insufficient_rate']:.1%})")
        print(f"  생성 실패: {short['generation_failures']}개")
        print(f"  전체 성공률: {short['success_rate']:.1%}")
        
        # EM 점수 분포
        if "em_distribution" in short:
            print(f"  EM 점수 분포:")
            em_dist = short["em_distribution"]
            total = sum(em_dist.values())
            for category, count in em_dist.items():
                percentage = count / total * 100 if total > 0 else 0
                print(f"    {category}: {count}개 ({percentage:.1f}%)")
        
        # 질문 유형별 성능
        if "question_type_performance" in short:
            print(f"  질문 유형별 성능:")
            for qtype, perf in short["question_type_performance"].items():
                print(f"    {qtype}: EM {perf['avg_em']:.1%}, F1 {perf['avg_f1']:.1%}, 정확률 {perf['exact_rate']:.1%} ({perf['total']}개)")
        
        # 답변 유형 분포
        if "answer_type_distribution" in short:
            print(f"  답변 유형 분포:")
            for atype, count in short["answer_type_distribution"].items():
                print(f"    {atype}: {count}개")
    
    # 개선도 분석
    if "improvement_analysis" in analysis and analysis["improvement_analysis"]:
        improvement = analysis["improvement_analysis"]
        print(f"\n성능 개선도 분석:")
        print(f"  EM 점수 개선: {improvement['em_improvement_percent']:+.1f}%")
        print(f"  정보 불충분 개선: {improvement['info_insufficient_improvement_percent']:+.1f}%")
        print(f"  현재 EM: {improvement['current_em']:.1%} (이전: {improvement['baseline_em']:.1%})")
        print(f"  현재 정보불충분률: {improvement['current_info_rate']:.1%} (이전: {improvement['baseline_info_rate']:.1%})")
        print(f"  전체 성능 등급: {improvement['performance_gain']}")
    
    # 품질 분석
    if "quality_analysis" in analysis and analysis["quality_analysis"]:
        quality = analysis["quality_analysis"]
        print(f"\n전체 품질 분석:")
        print(f"  MCQ 평균 품질: {quality['avg_mcq_context_quality']:.3f}")
        print(f"  단답형 평균 품질: {quality['avg_short_context_quality']:.3f}")
        print(f"  전체 품질 점수: {quality['overall_quality_score']:.3f}")
    
    # 성능 인사이트
    if "performance_insights" in analysis and analysis["performance_insights"]:
        print(f"\n성능 인사이트:")
        for i, insight in enumerate(analysis["performance_insights"], 1):
            print(f"  {i}. {insight}")
    
    # 추천사항
    if "recommendations" in analysis and analysis["recommendations"]:
        print(f"\n개선 추천사항:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"  {i}. {rec}")

def main():
    parser = argparse.ArgumentParser(description="성능 강화 RAG 평가 도구")
    
    parser.add_argument("file", help="평가할 Excel 파일")
    parser.add_argument("--mcq", type=int, help="MCQ 샘플 크기")
    parser.add_argument("--short", type=int, help="단답형 샘플 크기")
    parser.add_argument("--workers", type=int, default=8, help="병렬 워커 수")
    parser.add_argument("--output", help="출력 파일명")
    parser.add_argument("--sampling", choices=['random', 'stratified', 'difficulty'], 
                       default='stratified', help="샘플링 방법")
    
    # 사전 정의된 설정들
    parser.add_argument("--quick", action="store_true", help="빠른 테스트 (200/100)")
    parser.add_argument("--standard", action="store_true", help="표준 평가 (800/400)")
    parser.add_argument("--medium", action="store_true", help="중간 평가 (1000/500)")
    parser.add_argument("--large", action="store_true", help="대규모 평가 (2000/1000)")
    parser.add_argument("--test", action="store_true", help="초소형 테스트 (50/50)")
    
    # 분석 옵션들
    parser.add_argument("--analysis", action="store_true", help="상세 분석")
    parser.add_argument("--compare", action="store_true", help="성능 비교")
    parser.add_argument("--no-warmup", action="store_true", help="예열 비활성화")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        os.environ["DEBUG_HYBRID"] = "true"
    
    # 파일 존재 확인
    if not Path(args.file).exists():
        print(f"오류: 파일을 찾을 수 없습니다 - {args.file}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(" 성능 강화 RAG 평가 시스템")
    print("="*80)
    print(f"평가 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 로드
    print(f"\n파일 로드: {args.file}")
    load_start = time.time()
    mcq_questions, short_questions = load_excel(args.file)
    load_time = time.time() - load_start
    
    print(f"로드 완료 ({load_time:.1f}초)")
    print(f"전체 문제 수:")
    print(f"  사지선다형: {len(mcq_questions):,}개")
    print(f"  단답형: {len(short_questions):,}개")
    print(f"  총합: {len(mcq_questions) + len(short_questions):,}개")
    
    # 사전 정의된 설정 적용
    if args.test:
        args.mcq, args.short = 50, 50
        print(f"\n초소형 테스트 모드")
    elif args.quick:
        args.mcq, args.short = 200, 100
        print(f"\n빠른 테스트 모드")
    elif args.standard:
        args.mcq, args.short = 800, 400
        print(f"\n표준 평가 모드")
    elif args.medium:
        args.mcq, args.short = 1000, 500
        print(f"\n중간 평가 모드")
    elif args.large:
        args.mcq, args.short = 2000, 1000
        print(f"\n대규모 평가 모드")
    
    # 샘플링 적용
    if args.mcq is not None or args.short is not None:
        print(f"\n샘플링 적용 ({args.sampling} 방식):")
        sample_start = time.time()
        
        mcq_questions, short_questions = sample_questions(
            mcq_questions, short_questions, 
            args.mcq, args.short, 
            args.sampling
        )
        
        sample_time = time.time() - sample_start
        print(f"샘플링 완료 ({sample_time:.1f}초)")
    
    final_mcq_count = len(mcq_questions)
    final_short_count = len(short_questions)
    
    print(f"\n최종 평가 설정:")
    print(f"  사지선다형: {final_mcq_count:,}개")
    print(f"  단답형: {final_short_count:,}개")
    print(f"  병렬 워커: {args.workers}개")
    print(f"  샘플링: {args.sampling}")
    
    # 시스템 예열
    if not args.no_warmup:
        print("시스템 예열 중...")
        warmup_queries = [
            "금융소비자보호법 제10조",
            "추심연락의 유예 기간은 얼마인가",
            "담당 기관은 어디인가"
        ]
        
        for query in warmup_queries:
            try:
                retriever.retrieve(query, top_k=3, debug=False)
            except Exception:
                pass
        
        print("시스템 예열 완료")
    
    # 평가 실행
    print(f"\n" + "="*60)
    print(" 평가 실행")
    print("="*60)
    
    evaluation_start = time.time()
    
    evaluator = Evaluator(
        xlsx_path=args.file,
        mcq_n=final_mcq_count,
        short_n=final_short_count,
        workers=args.workers,
        debug=args.debug
    )
    
    # MCQ 평가
    mcq_results = []
    mcq_stats = {}
    if mcq_questions:
        print(f"\n사지선다형 평가 실행...")
        mcq_results, mcq_stats = evaluator.evaluate_all(mcq_questions, "mcq")
    
    # 단답형 평가
    short_results = []
    short_stats = {}
    if short_questions:
        print(f"\n단답형 평가 실행...")
        short_results, short_stats = evaluator.evaluate_all(short_questions, "short")
    
    evaluation_time = time.time() - evaluation_start
    
    # 결과 저장
    if mcq_results or short_results:
        output_file = save_results(mcq_results, short_results, args.output)
        print(f"\n결과 저장: {output_file}")
    else:
        print(f"\n저장할 결과가 없습니다.")
        return
    
    # 종합적인 상세 분석
    if args.analysis or args.compare:
        comprehensive_analysis = analyze_comprehensive_results(mcq_results, short_results)
        print_comprehensive_performance_analysis(comprehensive_analysis)
        
        # 성능 인사이트
        insights = get_performance_insights(mcq_results, short_results)
        if insights.get("강점") or insights.get("개선점"):
            print(f"\n추가 성능 인사이트:")
            for strength in insights.get("강점", []):
                print(f"  강점: {strength}")
            for improvement in insights.get("개선점", []):
                print(f"  개선점: {improvement}")
    
    # 향상된 성능 비교
    if args.compare:
        print(f"\n" + "="*60)
        print(" 성능 비교 (수정 전 vs 수정 후)")
        print("="*60)
        
        try:
            # 현재 성능 지표 수집
            current_performance = {}
            
            if mcq_results:
                mcq_acc = calculate_accuracy(mcq_results)
                current_performance['mcq_accuracy'] = mcq_acc
                
                # MCQ 추가 지표
                context_utilized = sum(1 for r in mcq_results if r.get('search_score', 0) > 1.0)
                high_confidence = sum(1 for r in mcq_results if r.get('search_score', 0) > 1.5)
                
                current_performance['mcq_context_utilization'] = context_utilized / len(mcq_results)
                current_performance['mcq_high_confidence'] = high_confidence / len(mcq_results)
            
            if short_results:
                ems, f1s = [], []
                info_insufficient_count = 0
                exact_matches = 0
                
                for r in short_results:
                    pred = r.get('prediction', '')
                    ans = r.get('answer', '')
                    
                    if pred == "정보 불충분":
                        info_insufficient_count += 1
                        ems.append(0.0)
                        f1s.append(0.0)
                    else:
                        em, f1 = score_short(pred, ans)
                        ems.append(em)
                        f1s.append(f1)
                        if em >= 0.9:
                            exact_matches += 1
                
                current_performance['em_score'] = sum(ems) / len(ems) if ems else 0
                current_performance['f1_score'] = sum(f1s) / len(f1s) if f1s else 0
                current_performance['success_rate'] = sum(1 for em in ems if em >= 0.3) / len(ems) if ems else 0
                current_performance['info_insufficient_rate'] = info_insufficient_count / len(short_results) if short_results else 0
                current_performance['exact_match_rate'] = exact_matches / len(short_results) if short_results else 0
            
            # 종합 분석 결과 전달
            comprehensive_analysis = analyze_comprehensive_results(mcq_results, short_results)
            compare_with_enhanced_baseline(comprehensive_analysis)
            
        except Exception as e:
            print(f"성능 비교 중 오류: {e}")
    
    # 최종 요약
    print(f"\n" + "="*80)
    print(" 평가 완료 - 최종 요약")
    print("="*80)
    
    total_questions = len(mcq_results) + len(short_results)
    
    print(f"평가 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 처리 문제: {total_questions:,}개")
    print(f"총 소요 시간: {evaluation_time/60:.1f}분")
    
    if total_questions > 0 and evaluation_time > 0:
        throughput = total_questions / evaluation_time
        print(f"평균 처리 속도: {throughput:.1f}개/초")
    
    # 성능 요약
    if mcq_results:
        mcq_acc = mcq_stats.get('acc', 0)
        mcq_quality = mcq_stats.get('avg_context_quality', 0)
        context_util = mcq_stats.get('context_utilization_rate', 0)
        high_conf = mcq_stats.get('high_confidence_rate', 0)
        
        print(f"MCQ 최종 성능:")
        print(f"  정확도: {mcq_acc:.1%} (목표: 65%)")
        print(f"  컨텍스트 활용률: {context_util:.1%}")
        print(f"  고신뢰도 선택률: {high_conf:.1%}")
        print(f"  평균 품질: {mcq_quality:.3f}")
        
        # 목표 달성 평가
        if mcq_acc >= 0.65:
            print("  MCQ 목표 달성!")
        elif mcq_acc >= 0.60:
            print("  MCQ 목표에 근접")
        else:
            print("  MCQ 추가 개선 필요")
    
    if short_results:
        short_em = short_stats.get('EM', 0)
        short_f1 = short_stats.get('F1', 0)
        short_quality = short_stats.get('avg_context_quality', 0)
        info_rate = short_stats.get('info_insufficient_rate', 0)
        
        print(f"단답형 최종 성능:")
        print(f"  EM 점수: {short_em:.1%} (목표: 20%)")
        print(f"  F1 점수: {short_f1:.1%}")
        print(f"  정보 불충분률: {info_rate:.1%}")
        print(f"  평균 품질: {short_quality:.3f}")
        
        # 핵심 개선 지표
        exact_matches = sum(1 for r in short_results 
                           if r.get('prediction') != '정보 불충분' and 
                           score_short(r.get('prediction', ''), r.get('answer', ''))[0] >= 0.9)
        exact_rate = exact_matches / len(short_results)
        print(f"  정확 매칭률: {exact_rate:.1%}")
        
        # 목표 달성 평가
        if short_em >= 0.20:
            print("  단답형 EM 목표 달성!")
        elif short_em >= 0.15:
            print("  단답형 EM 목표에 근접")
        else:
            print("  단답형 EM 추가 개선 필요")
        
        if info_rate <= 0.05:
            print("  정보 불충분 목표 달성!")
        elif info_rate <= 0.10:
            print("  정보 불충분 목표에 근접")
        
        # 전체 성공률 평가
        success_count = sum(1 for r in short_results 
                           if r.get('prediction') != '정보 불충분' and 
                           score_short(r.get('prediction', ''), r.get('answer', ''))[0] >= 0.3)
        success_rate = success_count / len(short_results)
        print(f"  전체 성공률: {success_rate:.1%}")
    
    # Generator 종합 통계
    try:
        from rag.generator import get_generation_stats
        gen_stats = get_generation_stats()
        if gen_stats:
            print(f"\nGenerator 종합 성능:")
            
            # MCQ 통계
            mcq_accuracy = gen_stats.get('mcq_accuracy', 0)
            mcq_context_util = gen_stats.get('mcq_context_utilization', 0)
            mcq_confidence = gen_stats.get('mcq_avg_confidence', 0)
            
            if mcq_accuracy > 0:
                print(f"  MCQ 생성 정확도: {mcq_accuracy:.1%}")
                print(f"  MCQ 컨텍스트 활용: {mcq_context_util:.1%}")
                print(f"  MCQ 평균 신뢰도: {mcq_confidence:.3f}")
            
            # 단답형 통계
            short_success = gen_stats.get('short_success_rate', 0)
            short_validation = gen_stats.get('short_validation_pass_rate', 0)
            short_gen_time = gen_stats.get('short_avg_generation_time', 0)
            
            if short_success > 0:
                print(f"  단답형 생성 성공률: {short_success:.1%}")
                print(f"  단답형 검증 통과율: {short_validation:.1%}")
                print(f"  단답형 평균 생성시간: {short_gen_time:.1f}ms")
            
            # 질문 유형별 성능
            question_breakdown = gen_stats.get('question_type_breakdown', {})
            if question_breakdown:
                print(f"  질문 유형별 성능:")
                for qtype, data in question_breakdown.items():
                    if data.get('total', 0) > 0:
                        avg_em = data.get('em_sum', 0) / data['total']
                        print(f"    {qtype}: {avg_em:.1%} ({data['total']}개)")
            
    except Exception as e:
        if args.debug:
            print(f"Generator 통계 조회 오류: {e}")
    
    # 전체 시스템 평가
    overall_score = 0
    if mcq_results:
        overall_score += mcq_stats.get('acc', 0) * 50
    if short_results:
        overall_score += short_stats.get('EM', 0) * 50
    
    print(f"\n전체 시스템 성능 점수: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("전체 시스템 성능: 우수")
    elif overall_score >= 60:
        print("전체 시스템 성능: 양호")
    elif overall_score >= 40:
        print("전체 시스템 성능: 보통 (개선 필요)")
    else:
        print("전체 시스템 성능: 미흡 (대폭 개선 필요)")
    
    # 다음 단계 권장사항
    print(f"\n다음 단계 권장사항:")
    next_steps = []
    
    if mcq_results and mcq_stats.get('acc', 0) < 0.65:
        gap = 0.65 - mcq_stats.get('acc', 0)
        next_steps.append(f"MCQ 정확도 {gap:.1%}p 개선 필요")
    
    if short_results and short_stats.get('EM', 0) < 0.20:
        gap = 0.20 - short_stats.get('EM', 0)
        next_steps.append(f"단답형 EM {gap:.1%}p 개선 필요")
    
    if short_results and short_stats.get('info_insufficient_rate', 0) > 0.05:
        next_steps.append("정보 불충분률 추가 감소 필요")
    
    if not next_steps:
        next_steps.append("모든 주요 목표 달성! 시스템 안정성 및 확장성 검토")
    
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    
    # 캐시 정리
    clear_all_caches()
    gc.collect()
    
    print("="*80)
    print("성능 강화 평가가 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()