import pandas as pd
from typing import List, Dict, Tuple
import re

def normalize_text(text: str) -> str:
    """텍스트 정규화"""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def calculate_em_f1(results: List[Dict]) -> Tuple[float, float]:
    """EM과 F1 점수 계산"""
    if not results:
        return 0.0, 0.0
    
    total_em = 0
    total_f1 = 0
    
    for r in results:
        pred = normalize_text(r["prediction"])
        gold = normalize_text(r["answer"])
        
        # Exact Match
        em = 1.0 if pred == gold else 0.0
        total_em += em
        
        # F1 Score
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            f1 = 0.0
        else:
            overlap = len(pred_tokens & gold_tokens)
            precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = overlap / len(gold_tokens) if len(gold_tokens) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total_f1 += f1
    
    return total_em / len(results), total_f1 / len(results)

def calculate_accuracy(results: List[Dict]) -> float:
    """정확도 계산"""
    if not results:
        return 0.0
    
    correct = sum(1 for r in results if normalize_text(r["prediction"]) == normalize_text(r["answer"]))
    return correct / len(results)

def load_excel(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    """Excel 파일에서 문제 로드"""
    mcq_questions = []
    short_questions = []
    
    # 사지선다형 로드
    try:
        df = pd.read_excel(filepath, sheet_name="사지선다형")
        for _, row in df.iterrows():
            answer_idx = str(row.get("정답", "")).strip()
            answer_text = ""
            
            if answer_idx.isdigit() and 1 <= int(answer_idx) <= 4:
                answer_text = str(row.get(f"보기{answer_idx}", "")).strip()
            
            mcq_questions.append({
                "question": str(row.get("문제내용", "")).strip(),
                "choices": [
                    str(row.get(f"보기{i}", "")).strip()
                    for i in range(1, 5)
                    if pd.notna(row.get(f"보기{i}"))
                ],
                "answer": answer_text,
                "meta": {
                    "difficulty": str(row.get("난이도", "")).strip(),
                    "law": str(row.get("법령명", "")).strip()
                }
            })
    except Exception as e:
        print(f"사지선다형 로드 오류: {e}")
    
    # 단답형 로드
    try:
        df = pd.read_excel(filepath, sheet_name="단답형")
        for _, row in df.iterrows():
            short_questions.append({
                "question": str(row.get("문제내용", "")).strip(),
                "answer": str(row.get("정답", "")).strip(),
                "meta": {
                    "difficulty": str(row.get("난이도", "")).strip(),
                    "law": str(row.get("법령명", "")).strip()
                }
            })
    except Exception as e:
        print(f"단답형 로드 오류: {e}")
    
    return mcq_questions, short_questions

def save_results(mcq_results: List[Dict], short_results: List[Dict], output_file: str = None):
    """결과 저장"""
    from datetime import datetime
    
    if not output_file:
        output_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 사지선다형 결과
        if mcq_results:
            df = pd.DataFrame(mcq_results)
            df.to_excel(writer, sheet_name="사지선다형", index=False)
        
        # 단답형 결과
        if short_results:
            df = pd.DataFrame(short_results)
            df.to_excel(writer, sheet_name="단답형", index=False)
        
        # 요약
        summary = []
        if mcq_results:
            acc = calculate_accuracy(mcq_results)
            summary.append({
                "유형": "사지선다형",
                "문제수": len(mcq_results),
                "정확도": f"{acc:.3f}"
            })
        
        if short_results:
            em, f1 = calculate_em_f1(short_results)
            summary.append({
                "유형": "단답형",
                "문제수": len(short_results),
                "EM": f"{em:.3f}",
                "F1": f"{f1:.3f}"
            })
        
        if summary:
            pd.DataFrame(summary).to_excel(writer, sheet_name="요약", index=False)
    
    print(f"결과 저장: {output_file}")
    return output_file

def postprocess_answer(answer: str, question_type: str) -> str:
    """답변 후처리"""
    if question_type == "short":
        # 불필요한 문구 제거
        replacements = [
            ("정해진 금액 없음", ""),
            ("없음.", ""),
            ("문서로 이루어져야 함", "문서"),
            ("채무자의 연체 상황", "연체"),
        ]
        
        for old, new in replacements:
            if old in answer:
                answer = new if new else answer.replace(old, "")
        
        # 첫 단어/숫자만 추출
        import re
        
        # 숫자가 있으면 숫자 우선
        numbers = re.findall(r'\d+[천만억]?[원월년개일]?', answer)
        if numbers:
            return numbers[0]
        
        # 명사만 추출
        words = answer.split()
        if len(words) > 3:
            return ' '.join(words[:3])
    
    return answer.strip()