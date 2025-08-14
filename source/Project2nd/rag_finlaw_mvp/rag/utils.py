# -*- coding: utf-8 -*-
"""
utils.py
- 전처리/정규화/토큰화
- 의도 분석 및 검증
- 평가 메트릭
- 파일 I/O
- 강화된 후처리(정의/머리말/잡음 차단 강화)
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
import re

KO_STOPWORDS = {"은","는","이","가","을","를","의","에","에서","에게","으로","와","과","및","또는","도","만","보다"}
PARENS = r"[()\[\]{}<>【】（）〈〉「」『』]"

# ---------- 질의 전처리 ----------
def preprocess_query(q: str) -> str:
    if not q:
        return ""
    law_replacements = {
        '개인금융채권의 관리 및 개인금융채무자의 보호': '개인금융채권 관리 개인금융채무자 보호',
        '금융거래지표의 관리': '금융거래지표 관리',
        '국제금융기구에의 가입조치': '국제금융기구 가입조치'
    }
    for a, b in law_replacements.items():
        q = q.replace(a, b)

    keyword_enhancements = {
        '고려해야 하는': '고려 사항',
        '포함되어야 하는': '포함 사항',
        '위법행위를 발견했을 때': '위법행위 발견 조치',
        '추심연락의 유예': '추심연락 유예',
        '담보조달비율': '담보조달 비율'
    }
    for a, b in keyword_enhancements.items():
        q = q.replace(a, b)

    q = re.sub(
        r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?',
        lambda m: f"제{m.group(1)}조" + (f"제{m.group(2)}항" if m.group(2) else "") + (f"제{m.group(3)}호" if m.group(3) else ""),
        q
    )
    q = re.sub(r'\s*에 관한 법률(?:\s*시행령|\s*시행규칙)?$', '', q)
    q = re.sub(r'\([^)]*제?\d+호[^)]*\)', '', q)
    q = re.sub(r'\([^)]*\d{8}[^)]*\)', '', q)
    q = re.sub(r'[\"""\'·]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q

# ---------- 정규화 ----------
def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_punct(s: str) -> str:
    s = re.sub(PARENS, " ", s or "")
    s = re.sub(r"[""\"'`·•…]", " ", s)
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
    return normalize_units(normalize_punct(normalize_spaces(s))).lower()

def extract_digits(s: str) -> str:
    return "".join(re.findall(r"\d+", s or ""))

def soft_numeric_match(a: str, b: str) -> bool:
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

# ---------- 의도 분석/검증 ----------
def analyze_question_intent(question: str) -> dict:
    intent = {'type': 'general', 'context_clues': [], 'target': None, 'priority_patterns': []}
    if any(k in question for k in ['기간','얼마','몇','언제']):
        intent['type'] = 'period'
        if '유예' in question: intent['context_clues'] = ['유예']; intent['priority_patterns'] = ['개월','일','년']
        elif '비율' in question: intent['priority_patterns'] = ['%','퍼센트']
        else: intent['priority_patterns'] = ['개월','일','년','%']
    elif any(k in question for k in ['누구','누가','기관','담당']):
        intent['type'] = 'entity'
        if '정위원' in question: intent['context_clues'] = ['정위원']
        elif '역할' in question: intent['context_clues'] = ['역할']
    elif any(k in question for k in ['무엇','정의','의미','사항','내용']):
        intent['type'] = 'content'
        if '고려' in question and '사항' in question:
            intent['target'] = 'consideration_object'; intent['context_clues'] = ['고려']
        elif '포함' in question and '사항' in question:
            intent['target'] = 'inclusion_content'; intent['context_clues'] = ['포함']
        elif '정의' in question:
            intent['target'] = 'definition'
    elif any(k in question for k in ['조치','방법','절차']):
        intent['type'] = 'action'
        if '위법행위' in question: intent['context_clues'] = ['위법행위']
        elif '통지' in question: intent['context_clues'] = ['통지']
    return intent

def validate_answer_by_intent(answer: str, intent: dict) -> bool:
    if len(answer) < 2 or len(answer) > 60:
        return False
    if intent['type'] == 'period':
        return bool(re.search(r'\d+\s*(?:개월|일|년|%|퍼센트)', answer))
    elif intent['type'] == 'entity':
        return bool(re.search(r'[가-힣]{2,}(?:장관|총재|은행|위원회|금고)', answer))
    elif intent['type'] == 'action':
        return any(k in answer for k in ['지체','없이','요구','서면','전자문서','승인'])
    elif intent['type'] == 'content':
        if intent.get('target') == 'consideration_object':
            return ('권익' in answer) or ('질서' in answer)
    return True

# ---------- 평가 메트릭 ----------
def calculate_accuracy(results: List[Dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if normalize_for_em(r["prediction"]) == normalize_for_em(r["answer"]))
    return correct / len(results)

def score_short(pred: str, gold: str) -> Tuple[float, float]:
    pred_n, gold_n = normalize_for_em(pred), normalize_for_em(gold)
    em = 1.0 if (pred_n == gold_n or soft_numeric_match(pred_n, gold_n)) else 0.0
    f1 = f1_token(pred, gold)
    return em, max(f1, em)

# ---------- 파일 I/O ----------
def load_excel(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    mcq_questions, short_questions = [], []
    try:
        df = pd.read_excel(filepath, sheet_name="사지선다형")
        for _, row in df.iterrows():
            ans_idx = str(row.get("정답", "")).strip()
            ans = ""
            if ans_idx.isdigit() and 1 <= int(ans_idx) <= 4:
                ans = str(row.get(f"보기{ans_idx}", "")).strip()
            mcq_questions.append({
                "question": str(row.get("문제내용","")).strip(),
                "choices": [str(row.get(f"보기{i}","")).strip() for i in range(1,5) if pd.notna(row.get(f"보기{i}"))],
                "answer": ans,
                "meta": {"difficulty": str(row.get("난이도","")).strip(), "law": str(row.get("법령명","")).strip()}
            })
    except Exception as e:
        print(f"사지선다형 로드 오류: {e}")
    try:
        df = pd.read_excel(filepath, sheet_name="단답형")
        for _, row in df.iterrows():
            short_questions.append({
                "question": str(row.get("문제내용","")).strip(),
                "answer": str(row.get("정답","")).strip(),
                "meta": {"difficulty": str(row.get("난이도","")).strip(), "law": str(row.get("법령명","")).strip()}
            })
    except Exception as e:
        print(f"단답형 로드 오류: {e}")
    return mcq_questions, short_questions

def save_results(mcq_results: List[Dict], short_results: List[Dict], output_file: str = None):
    from datetime import datetime
    if not output_file:
        output_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as w:
        if mcq_results: pd.DataFrame(mcq_results).to_excel(w, sheet_name="사지선다형", index=False)
        if short_results: pd.DataFrame(short_results).to_excel(w, sheet_name="단답형", index=False)
        summary = []
        if mcq_results:
            acc = calculate_accuracy(mcq_results)
            summary.append({"유형":"사지선다형","문제수":len(mcq_results),"정확도":f"{acc:.3f}"})
        if short_results:
            ems, f1s = [], []
            for r in short_results:
                em, f1 = score_short(r["prediction"], r["answer"])
                ems.append(em); f1s.append(f1)
            summary.append({
                "유형":"단답형",
                "문제수":len(short_results),
                "EM":f"{(sum(ems)/len(ems)):.3f}" if ems else "0.000",
                "F1":f"{(sum(f1s)/len(f1s)):.3f}" if f1s else "0.000"
            })
        if summary: pd.DataFrame(summary).to_excel(w, sheet_name="요약", index=False)
    print(f"결과 저장: {output_file}")
    return output_file

# ---------- 강화된 후처리 ----------
def enhanced_postprocess_answer(answer: str, contexts: List[Dict], intent: dict = None) -> str:
    if not answer or answer in ["정보 없음", "정보 불충분", "(오류)"]:
        return answer
    answer = " ".join(answer.split())

    # 명백히 잘못된 형식/노이즈 차단
    wrong = [
        r'^제\d+조$',
        r'법률$|시행령$|시행규칙$',
        r'^[가-힣]{1,2}$',
        r'^(다음|위의|해당|상기)',
        r'란\s*다음\s*각\s*호',               # 정의 머리말
        r'다음\s*각\s*호',                    # 조항 머리말
        r'(을|를)\s*(말한다|의미한다)\s*$',   # 정의 종결
        r'임직원으로서\s*다음\s*각\s*호',      # 전형 머리글
    ]
    for p in wrong:
        if re.search(p, answer):
            return "정보 불충분"

    # 숫자-단위 정합성
    if intent and intent.get("type") == "period":
        if not re.search(r'\d+\s*(개월|일|년|%|퍼센트)', answer):
            return "정보 불충분"

    # 문맥 포함성 점검 & 부분 확장
    all_text = " ".join([c.get("text","") for c in contexts[:3]])
    if (answer in all_text) or (answer.replace(" ", "") in all_text.replace(" ", "")):
        pass
    else:
        expanded = try_expand_answer(answer, all_text)
        if expanded and (expanded in all_text or expanded.replace(" ", "") in all_text.replace(" ", "")):
            answer = expanded

    # 최종 품질 검증
    if not is_high_quality_answer(answer):
        return "정보 불충분"
    return answer.strip()

def try_expand_answer(partial: str, full: str) -> Optional[str]:
    pats = [
        rf'{re.escape(partial)}[가-힣\s]{{0,25}}',
        rf'[가-힣\s]{{0,15}}{re.escape(partial)}[가-힣\s]{{0,15}}',
    ]
    cands = []
    for p in pats:
        cands += re.findall(p, full)
    if not cands:
        return None
    best = max(cands, key=len)
    return best.strip() if 2 < len(best) <= 60 else None

def is_high_quality_answer(answer: str) -> bool:
    if len(answer) < 2 or len(answer) > 60:
        return False
    if len(re.findall(r'[가-힣0-9]', answer)) < 2:
        return False
    words = answer.split()
    if len(words) > 1 and len(set(words)) < len(words) * 0.5:
        return False
    return True

# 호환성 래퍼
def postprocess_answer(answer: str, contexts: List[Dict]) -> str:
    return enhanced_postprocess_answer(answer, contexts)

def validate_and_fix_answer(answer: str, contexts: List[Dict]) -> str:
    if re.match(r'^제?\d+조?$', answer):
        return "정보 불충분"
    return answer

def extract_question_type(question: str) -> str:
    if any(k in question for k in ['얼마','기간','몇','언제','날']): return 'numeric'
    elif any(k in question for k in ['누구','누가','기관','담당']): return 'entity'
    elif any(k in question for k in ['무엇','정의','의미']): return 'definition'
    elif any(k in question for k in ['방법','절차','어떻게']): return 'method'
    else: return 'general'
