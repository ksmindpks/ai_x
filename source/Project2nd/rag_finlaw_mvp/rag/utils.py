# rag/utils.py
# -*- coding: utf-8 -*-
import pandas as pd
from typing import List, Dict, Tuple
import re

# ====== 정규화 & 토큰화 유틸 ======
KO_STOPWORDS = {"은","는","이","가","을","를","의","에","에서","에게","으로","와","과","및","또는","도","만","보다"}
PARENS = r"[()\[\]{}<>【】（）〈〉「」『』]"

# --- [추가] 질의 전처리: 법령명/날짜/괄호 제거 등 ---
def preprocess_query(q: str) -> str:
    if not q:
        return ""
    q = q.replace("「", "").replace("」", "")
    # ( ... ) 괄호 안 정보(법령 종류/호수/날짜 등) 제거
    q = re.sub(r'\([^)]*\)', ' ', q)
    # “~에 관한 법률/시행령/시행규칙” 꼬리표 과감히 제거 (키워드만 남기기)
    q = re.sub(r'(에 관한 )?(법률|대통령령|총리령|부령|시행령|시행규칙)\b', ' ', q)
    # 불필요 기호 삭제
    q = re.sub(r'[\"“”\'·]', ' ', q)
    # 다중 공백 정리
    q = re.sub(r'\s+', ' ', q).strip()
    return q

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_punct(s: str) -> str:
    s = re.sub(PARENS, " ", s or "")
    s = re.sub(r"[“”\"'`·•…]", " ", s)
    s = re.sub(r"[,:;!?]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def normalize_units(s: str) -> str:
    s = (s or "")
    s = s.replace(" 만원", "만원").replace(" 억원", "억원")
    s = s.replace("만 원", "만원").replace("억 원", "억원")
    s = s.replace(",", "")
    return s

def normalize_for_em(s: str) -> str:
    """EM용 정규화 (공백/구두점/단위 정리 + 소문자)"""
    return normalize_units(normalize_punct(normalize_spaces(s))).lower()

def extract_digits(s: str) -> str:
    return "".join(re.findall(r"\d+", s or ""))

def soft_numeric_match(a: str, b: str) -> bool:
    """숫자만 비교해 일치하면 True (5,000만원 vs 5000만원)"""
    da, db = extract_digits(a), extract_digits(b)
    return bool(da and db and da == db)

def tokenize_ko(s: str):
    s = normalize_units(normalize_punct(normalize_spaces(s)))
    toks = s.split()
    return [t for t in toks if t not in KO_STOPWORDS]

def f1_token(a: str, b: str) -> float:
    A, B = set(tokenize_ko(a)), set(tokenize_ko(b))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    if inter == 0:
        return 0.0
    prec = inter / len(A)
    rec  = inter / len(B)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

# ====== MCQ 정확도 ======
def calculate_accuracy(results: List[Dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if normalize_for_em(r["prediction"]) == normalize_for_em(r["answer"]))
    return correct / len(results)

# ====== Excel 로드 ======
def load_excel(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    mcq_questions = []
    short_questions = []
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

# ====== 결과 저장 ======
def save_results(mcq_results: List[Dict], short_results: List[Dict], output_file: str = None):
    from datetime import datetime
    if not output_file:
        output_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if mcq_results:
            pd.DataFrame(mcq_results).to_excel(writer, sheet_name="사지선다형", index=False)
        if short_results:
            pd.DataFrame(short_results).to_excel(writer, sheet_name="단답형", index=False)
        summary = []
        if mcq_results:
            acc = calculate_accuracy(mcq_results)
            summary.append({"유형": "사지선다형", "문제수": len(mcq_results), "정확도": f"{acc:.3f}"})
        if short_results:
            em_scores = []
            f1_scores = []
            for r in short_results:
                em, f1 = score_short(r["prediction"], r["answer"])
                em_scores.append(em)
                f1_scores.append(f1)
            em_avg = sum(em_scores) / len(em_scores)
            f1_avg = sum(f1_scores) / len(f1_scores)
            summary.append({"유형": "단답형", "문제수": len(short_results), "EM": f"{em_avg:.3f}", "F1": f"{f1_avg:.3f}"})
        if summary:
            pd.DataFrame(summary).to_excel(writer, sheet_name="요약", index=False)
    print(f"결과 저장: {output_file}")
    return output_file

def postprocess_answer(answer: str, question_type: str, context_text: str = "") -> str:
    answer = answer.strip()

    if question_type == "short":
        # [1] 숫자만 → 컨텍스트에서 숫자+단위 후보 수집
        if re.fullmatch(r'\d+', answer):
            num = answer
            cands = re.findall(rf'{re.escape(num)}\s*([가-힣A-Za-z%]+)', context_text)
            if cands:
                # 우선순위: (1) 단위 길이 짧음, (2) 등장 빈도, (3) 첫 등장
                cands_sorted = sorted(cands, key=lambda u: (len(u) <= 2, -cands.count(u)), reverse=True)
                return f"{num}{cands_sorted[0]}"

        # [2] 불필요 문구 제거
        replacements = [
            ("정해진 금액 없음", ""),
            ("없음.", ""),
            ("없음", ""),
            ("문서로 이루어져야 함", "문서"),
            ("채무자의 연체 상황", "연체"),
        ]
        for old, new in replacements:
            if old in answer:
                answer = new if new else answer.replace(old, "")

        # [3] 동의어 매핑
        synonyms = {
            "시정 요구": "시정명령",
            "임치소": "보관소",
            "금융위원회": "금융위",
            "비공개정보": "비밀정보",
            "소속금융회사": "계열금융회사"
        }
        if answer in synonyms:
            answer = synonyms[answer]

        # [4] 숫자 + 단위 추출
        numbers = re.findall(r'\d+[천만억조]?[원월년개일호조항]?', answer)
        if numbers:
            return numbers[0]

        # [5] 조사 제거 (마지막 조사)
        answer = re.sub(r'(은|는|이|가|을|를|의|에|에서)$', '', answer)

        # [6] 단어 수 제한 완화 (최대 10단어 → 5단어)
        words = answer.split()
        if len(words) > 5:
            answer = ' '.join(words[:5])

    return answer.strip()

def score_short(pred: str, gold: str) -> Tuple[float, float]:
    """EM, F1 반환 (정규화 + 숫자 관용 + 조사 제거)"""
    pred_n = normalize_for_em(pred)
    gold_n = normalize_for_em(gold)

    # 완전일치 또는 숫자만 일치 시 EM=1
    em = 1.0 if (pred_n == gold_n or soft_numeric_match(pred_n, gold_n)) else 0.0

    # 조사 제거 후 토큰 F1
    f1 = f1_token(pred, gold)

    # 가중 평균: EM 비중 높임
    final_f1 = max(f1, em)
    return em, final_f1