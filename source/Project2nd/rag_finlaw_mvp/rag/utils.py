# utils.py
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
import os


# ===== 텍스트 정규화 함수들 =====
def normalize_date(text: str) -> str:
    """
    날짜 형식 정규화
    예: '2000.4.1.', '2000-04-01', '2000/4/1' → '2000.4.1.'
    """
    # 공백 제거
    result = text.replace(" ", "")
    # 날짜 패턴 통일 (YYYY.MM.DD 형식으로)
    result = re.sub(r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\.?', r'\1.\2.\3.', result)
    return result


def normalize_text(text: str) -> str:
    """
    텍스트 전체 정규화 (평가용)
    - 소문자 변환
    - 공백 정리
    - 날짜 정규화
    """
    # 소문자 변환 및 공백 정리
    result = " ".join(str(text).strip().lower().split())
    # 날짜 정규화
    result = normalize_date(result)
    return result


# ===== 평가 메트릭 함수들 =====
def em_f1(pred: str, gold: str) -> Tuple[float, float]:
    """
    Exact Match와 F1 점수 계산
    
    Args:
        pred: 예측 답변
        gold: 정답
    
    Returns:
        (exact_match, f1_score) 튜플
    """
    # 정규화
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    
    # Exact Match
    em = 1.0 if pred_norm == gold_norm else 0.0
    
    # F1 Score
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        f1 = 0.0
    else:
        overlap = len(pred_tokens & gold_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    
    return em, f1


def mcq_acc(pred: str, gold: str) -> int:
    """
    사지선다형 정확도 계산
    
    Args:
        pred: 예측 답변
        gold: 정답
    
    Returns:
        1 if correct, 0 otherwise
    """
    return 1 if normalize_text(pred) == normalize_text(gold) else 0


# ===== Excel 데이터 로드 함수들 =====
def load_excel_mcq(path: str, sheet_name: str = "사지선다형") -> List[Dict]:
    """
    사지선다형 문제 로드
    
    Args:
        path: Excel 파일 경로
        sheet_name: 시트 이름
    
    Returns:
        문제 리스트
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as e:
        print(f"[WARNING] 시트 '{sheet_name}'을 찾을 수 없습니다: {e}")
        return []
    
    rows = []
    for idx, r in df.iterrows():
        # 정답 번호를 보기 텍스트로 변환 (1~4 → 보기1~4의 텍스트)
        answer_idx = str(r.get("정답", "")).strip()
        answer_text = ""
        
        if answer_idx.isdigit() and 1 <= int(answer_idx) <= 4:
            answer_text = str(r.get(f"보기{answer_idx}", "")).strip()
        else:
            # 정답이 이미 텍스트인 경우
            answer_text = answer_idx
        
        rows.append({
            "type": "mcq",
            "idx": idx,
            "question": str(r.get("문제내용", "")).strip(),
            "choices": [
                str(r.get(f"보기{i}", "")).strip() 
                for i in range(1, 5) 
                if pd.notna(r.get(f"보기{i}")) and str(r.get(f"보기{i}")).strip()
            ],
            "answer": answer_text,
            "answer_idx": answer_idx,
            "explanation": str(r.get("해설", "")).strip() if pd.notna(r.get("해설")) else "",
            "meta": {
                "law": str(r.get("법령명", "")).strip() if pd.notna(r.get("법령명")) else "",
                "difficulty": str(r.get("난이도", "")).strip() if pd.notna(r.get("난이도")) else "",
                "source_file": os.path.basename(path)
            }
        })
    
    return rows


def load_excel_short(path: str, sheet_name: str = "단답형") -> List[Dict]:
    """
    단답형 문제 로드
    
    Args:
        path: Excel 파일 경로
        sheet_name: 시트 이름
    
    Returns:
        문제 리스트
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as e:
        print(f"[WARNING] 시트 '{sheet_name}'을 찾을 수 없습니다: {e}")
        return []
    
    rows = []
    for idx, r in df.iterrows():
        rows.append({
            "type": "short",
            "idx": idx,
            "question": str(r.get("문제내용", "")).strip(),
            "answer": str(r.get("정답", "")).strip(),
            "explanation": str(r.get("해설", "")).strip() if pd.notna(r.get("해설")) else "",
            "meta": {
                "law": str(r.get("법령명", "")).strip() if pd.notna(r.get("법령명")) else "",
                "difficulty": str(r.get("난이도", "")).strip() if pd.notna(r.get("난이도")) else "",
                "source_file": os.path.basename(path)
            }
        })
    
    return rows


def load_excel_both(path: str, 
                   mcq_limit: Optional[int] = None, 
                   short_limit: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    하나의 Excel 파일에서 사지선다형과 단답형 모두 로드
    
    Args:
        path: Excel 파일 경로
        mcq_limit: 사지선다형 로드 개수 제한
        short_limit: 단답형 로드 개수 제한
    
    Returns:
        (사지선다형 리스트, 단답형 리스트)
    """
    mcq_rows = load_excel_mcq(path, "사지선다형")
    short_rows = load_excel_short(path, "단답형")
    
    # 개수 제한 적용
    if mcq_limit:
        mcq_rows = mcq_rows[:mcq_limit]
    if short_limit:
        short_rows = short_rows[:short_limit]
    
    # 소스 파일 정보 추가
    for row in mcq_rows:
        row["source_file"] = path
    for row in short_rows:
        row["source_file"] = path
    
    return mcq_rows, short_rows


def load_multiple_excels(file_paths: List[str], 
                        mcq_limit_per_file: Optional[int] = None, 
                        short_limit_per_file: Optional[int] = None,
                        verbose: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    여러 Excel 파일에서 문제 로드
    
    Args:
        file_paths: Excel 파일 경로 리스트
        mcq_limit_per_file: 파일당 사지선다형 개수
        short_limit_per_file: 파일당 단답형 개수
        verbose: 상세 출력 여부
    
    Returns:
        (전체 사지선다형 리스트, 전체 단답형 리스트)
    """
    all_mcq = []
    all_short = []
    
    for i, file_path in enumerate(file_paths, 1):
        if not os.path.exists(file_path):
            print(f"[WARNING] 파일을 찾을 수 없습니다: {file_path}")
            continue
        
        if verbose:
            print(f"\n파일 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        mcq, short = load_excel_both(file_path, mcq_limit_per_file, short_limit_per_file)
        
        if verbose:
            print(f"   - 사지선다형: {len(mcq)}개 로드")
            print(f"   - 단답형: {len(short)}개 로드")
        
        all_mcq.extend(mcq)
        all_short.extend(short)
    
    return all_mcq, all_short


# ===== 통계 및 분석 함수들 =====
def analyze_by_difficulty(results: List[Dict], question_type: str = "mcq") -> Dict[str, Dict]:
    """
    난이도별 성능 분석
    
    Args:
        results: 평가 결과 리스트
        question_type: "mcq" 또는 "short"
    
    Returns:
        난이도별 통계 딕셔너리
    """
    analysis = {}
    difficulties = ["상", "중", "하"]
    
    for difficulty in difficulties:
        level_results = [r for r in results if r.get("난이도") == difficulty]
        
        if not level_results:
            continue
        
        if question_type == "mcq":
            correct = sum(1 for r in level_results if r.get("정확도") == "O")
            accuracy = correct / len(level_results)
            analysis[difficulty] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(level_results)
            }
        else:  # short
            em_sum = sum(r.get("EM", 0) for r in level_results)
            f1_sum = sum(r.get("F1", 0) for r in level_results)
            analysis[difficulty] = {
                "em": em_sum / len(level_results),
                "f1": f1_sum / len(level_results),
                "total": len(level_results)
            }
    
    return analysis


def analyze_by_law(results: List[Dict], question_type: str = "mcq") -> Dict[str, Dict]:
    """
    법령별 성능 분석
    
    Args:
        results: 평가 결과 리스트
        question_type: "mcq" 또는 "short"
    
    Returns:
        법령별 통계 딕셔너리
    """
    analysis = {}
    
    # 모든 법령 수집
    laws = list(set(r.get("법령", "") for r in results if r.get("법령")))
    
    for law in laws:
        law_results = [r for r in results if r.get("법령") == law]
        
        if not law_results:
            continue
        
        # 법령명이 너무 길면 축약
        law_short = law[:30] + "..." if len(law) > 30 else law
        
        if question_type == "mcq":
            correct = sum(1 for r in law_results if r.get("정확도") == "O")
            accuracy = correct / len(law_results)
            analysis[law_short] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(law_results)
            }
        else:  # short
            em_sum = sum(r.get("EM", 0) for r in law_results)
            f1_sum = sum(r.get("F1", 0) for r in law_results)
            analysis[law_short] = {
                "em": em_sum / len(law_results),
                "f1": f1_sum / len(law_results),
                "total": len(law_results)
            }
    
    return analysis


def print_statistics(results: List[Dict], question_type: str = "mcq", verbose: bool = False):
    """
    평가 결과 통계 출력
    
    Args:
        results: 평가 결과 리스트
        question_type: "mcq" 또는 "short"
        verbose: 상세 출력 여부
    """
    if not results:
        print("결과가 없습니다.")
        return
    
    type_name = "사지선다형" if question_type == "mcq" else "단답형"
    
    print(f"\n[{type_name} 통계]")
    print("=" * 50)
    
    if question_type == "mcq":
        total = len(results)
        correct = sum(1 for r in results if r.get("정확도") == "O")
        accuracy = correct / total if total > 0 else 0
        
        print(f"전체 정확도: {accuracy:.3f} ({correct}/{total})")
    else:
        total = len(results)
        em_avg = sum(r.get("EM", 0) for r in results) / total if total > 0 else 0
        f1_avg = sum(r.get("F1", 0) for r in results) / total if total > 0 else 0
        
        print(f"평균 EM: {em_avg:.3f}")
        print(f"평균 F1: {f1_avg:.3f}")
        print(f"총 문제: {total}개")
    
    if verbose:
        # 난이도별 분석
        print("\n난이도별 성능:")
        difficulty_stats = analyze_by_difficulty(results, question_type)
        for diff, stats in difficulty_stats.items():
            if question_type == "mcq":
                print(f"  {diff}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
            else:
                print(f"  {diff}: EM={stats['em']:.3f}, F1={stats['f1']:.3f} (n={stats['total']})")
        
        # 상위 5개 법령별 분석
        print("\n주요 법령별 성능 (상위 5개):")
        law_stats = analyze_by_law(results, question_type)
        sorted_laws = sorted(law_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
        
        for law, stats in sorted_laws:
            if question_type == "mcq":
                print(f"  {law}: {stats['accuracy']:.3f} (n={stats['total']})")
            else:
                print(f"  {law}: EM={stats['em']:.3f}, F1={stats['f1']:.3f} (n={stats['total']})")