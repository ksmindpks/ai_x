import streamlit as st
import os
import sys
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import time
import traceback
import uuid
import re
import subprocess

# 스레드 안전성을 위한 추가 import
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = lambda t, ctx=None: None
    get_script_run_ctx = lambda: None

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# ========== 세션 상태 초기화 ==========
def init_session_state():
    """통합 세션 상태 초기화"""
    session_defaults = {
        'evaluation_running': False,
        'evaluation_completed': False,
        
        # MCQ 문제별 상태 + 통계
        'mcq_question_log': [],
        'mcq_stats': {'correct': 0, 'total': 0},
        
        # 단답형 문제별 상태 + 통계  
        'short_question_log': [],
        'short_stats': {'correct': 0, 'total': 0},
        
        # 정답/오답 처리과정 로그
        'correct_process_logs': [],
        'incorrect_process_logs': [],
        
        # 관리 로그
        'mgmt_logs': [],
        'evaluation_progress': {'current': 0, 'total': 0},
        'chat_history': [],
        'session_id': str(uuid.uuid4())[:8]
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# 초기화 실행
init_session_state()

# ========== RAG 시스템 관리 ==========
def get_rag_system():
    """Streamlit 세션 기반 RAG 시스템 관리"""
    if (hasattr(st.session_state, 'rag_retriever') and 
        hasattr(st.session_state, 'rag_llm') and 
        hasattr(st.session_state, 'rag_config')):
        return st.session_state.rag_retriever, st.session_state.rag_llm, st.session_state.rag_config
    
    try:
        from rag.hybrid_retriever import HybridRetriever
        from rag.llm_bridge import HybridLLM
        from config import get_config
        
        config = get_config()
        retriever = HybridRetriever(config)
        llm = HybridLLM(config)
        
        if not retriever or not llm:
            raise RuntimeError("RAG 컴포넌트 초기화 실패")
        
        st.session_state.rag_retriever = retriever
        st.session_state.rag_llm = llm
        st.session_state.rag_config = config
        
        return retriever, llm, config
        
    except Exception as e:
        print(f"RAG 초기화 실패: {e}")
        return None, None, None

def get_rag_available():
    """RAG 사용 가능 여부 확인"""
    retriever, llm, config = get_rag_system()
    return retriever is not None and llm is not None and config is not None

# ========== 질의응답 기능 ==========
def retrieve(question, top_k=5):
    """검색 함수"""
    retriever, llm, config = get_rag_system()
    if not retriever:
        return []
    search_results = retriever.search(question, question_type="short")
    return [{'text': result.content, 'score': result.score} for result in search_results[:top_k]]

def generate_answer_short(question, contexts):
    """질의응답용 자연스러운 답변 생성"""
    retriever, llm, config = get_rag_system()
    if not llm:
        return "시스템 오류"
        
    context_text = "\n".join([ctx['text'] for ctx in contexts])
    
    enhanced_question = f"""참고자료를 바탕으로 질문에 답하세요.

참고자료:
{context_text}

질문: {question}

답변:"""
    
    return call_simple_llm(enhanced_question)

def call_simple_llm(user_prompt, context=""):
    """질의응답용 간단한 LLM 호출"""
    retriever, llm, config = get_rag_system()
    if not config:
        return "시스템 설정 오류"
        
    try:
        if config.openai_api_key:
            import openai
            client = openai.OpenAI(api_key=config.openai_api_key, timeout=config.llm_timeout)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=config.max_tokens
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI 호출 실패: {e}")
    
    try:
        if config.upstage_api_key:
            import openai
            client = openai.OpenAI(
                api_key=config.upstage_api_key,
                base_url="https://api.upstage.ai/v1",
                timeout=config.llm_timeout
            )
            response = client.chat.completions.create(
                model="solar-mini",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=config.max_tokens
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Upstage 호출 실패: {e}")
    
    return "답변 생성에 실패했습니다."

def format_answer_for_chat(question, raw_answer):
    """질의응답용 답변을 완전한 문장으로 포매팅"""
    if not raw_answer or raw_answer.strip() == "":
        return "죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다."
    
    answer = raw_answer.strip()
    
    if answer.endswith(('.', '다', '요', 'ㅂ', '것', '니다', '습니다', '입니다')):
        return answer
    
    if answer.endswith('으로') or answer.endswith('에서') or answer.endswith('는'):
        return f"{answer} 정해져 있습니다."
    
    if '다음' in answer and ('각 호' in answer or '요건' in answer):
        return f"{answer} 모든 조건을 충족해야 합니다."
    
    return f"{answer}."

def add_to_chat_history(question, answer, contexts=None):
    """채팅 기록에 질문과 답변 추가"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    st.session_state.chat_history.append({
        "type": "user",
        "content": question,
        "timestamp": timestamp
    })
    st.session_state.chat_history.append({
        "type": "assistant", 
        "content": answer,
        "timestamp": timestamp,
        "contexts": contexts or []
    })

def display_chat_history():
    """채팅 기록 표시"""
    if st.session_state.chat_history:
        st.subheader("대화 기록")
        
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["type"] == "user":
                    st.markdown(f"**사용자 {chat['timestamp']}:** {chat['content']}")
                else:
                    st.markdown(f"**Assistant {chat['timestamp']}:** {chat['content']}")
                    
                    if chat.get('contexts'):
                        with st.expander(f"검색 결과 보기 ({len(chat['contexts'])}개)"):
                            for j, ctx in enumerate(chat['contexts'][:3], 1):
                                st.write(f"**[{j}]** Score: {ctx['score']:.3f}")
                                st.write(f"{ctx['text'][:200]}...")
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("<hr style='margin: 8px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

# ========== MCQ/SHORT 문제 상태 업데이트 함수들 ==========
def update_mcq_question(question_num, question_text, choices=None, predicted=None, correct=None, is_correct=None, status="processing"):
    """MCQ 문제 상태 업데이트"""
    try:
        existing_idx = -1
        for i, q in enumerate(st.session_state.mcq_question_log):
            if q.get('number') == question_num:
                existing_idx = i
                break
        
        question_data = {
            'number': question_num,
            'text': question_text,
            'choices': choices or [],
            'predicted': predicted,
            'correct': correct,
            'is_correct': is_correct,
            'status': status,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        if existing_idx >= 0:
            st.session_state.mcq_question_log[existing_idx].update(question_data)
        else:
            st.session_state.mcq_question_log.append(question_data)
        
        if status == "completed" and is_correct is not None:
            if is_correct:
                st.session_state.mcq_stats['correct'] += 1
            st.session_state.mcq_stats['total'] += 1
    
    except Exception as e:
        print(f"[MCQ-UPDATE-ERROR] {e}")

def update_short_question(question_num, question_text, predicted=None, correct=None, em_score=None, f1_score=None, status="processing"):
    """단답형 문제 상태 업데이트"""
    try:
        existing_idx = -1
        for i, q in enumerate(st.session_state.short_question_log):
            if q.get('number') == question_num:
                existing_idx = i
                break
        
        question_data = {
            'number': question_num,
            'text': question_text,
            'predicted': predicted,
            'correct': correct,
            'em_score': em_score,
            'f1_score': f1_score,
            'status': status,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        if existing_idx >= 0:
            st.session_state.short_question_log[existing_idx].update(question_data)
        else:
            st.session_state.short_question_log.append(question_data)
        
        if status == "completed" and em_score is not None:
            if em_score > 0:
                st.session_state.short_stats['correct'] += 1
            st.session_state.short_stats['total'] += 1
    
    except Exception as e:
        print(f"[SHORT-UPDATE-ERROR] {e}")

def safe_add_log(log_type, message, category="evaluation"):
    """관리 로그 추가"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    try:
        if category == "management":
            prefix_map = {
                "success": "[OK]", 
                "failure": "[FAIL]", 
                "progress": "[PROG]",
                "info": "[INFO]",
                "warning": "[WARN]",
                "error": "[ERROR]"
            }
            prefix = prefix_map.get(log_type, "[LOG]")
            mgmt_entry = f"{prefix} {log_entry}"
            st.session_state.mgmt_logs.append(mgmt_entry)
            if len(st.session_state.mgmt_logs) > 50:
                st.session_state.mgmt_logs.pop(0)
    except Exception as e:
        print(f"[LOG-ERROR] {e}")

# ========== 로그 라우팅 시스템 ==========
def enhanced_global_log_callback(log_type, message, module="", category="evaluation"):
    """로그 라우팅: 시스템/설정은 관리로, 평가 과정은 정답/오답으로"""
    formatted_msg = f"[{module}] {message}" if module else message
    
    try:
        # 임시 버퍼 초기화
        if 'mcq_temp_logs' not in st.session_state:
            st.session_state.mcq_temp_logs = {}
        if 'short_temp_logs' not in st.session_state:
            st.session_state.short_temp_logs = {}
        if 'current_question' not in st.session_state:
            st.session_state.current_question = None
            
        # 1. Config, 초기화, 시스템 설정은 관리로
        if any(keyword in message.lower() for keyword in [
            "config", "api_key", "temperature", "max_tokens", "timeout",
            "초기화", "로드", "연결", "인스턴스", "모듈", "컴포넌트"
        ]):
            safe_add_log("info", formatted_msg, "management")
            return
            
        # 2. 평가 시작/종료 메시지
        if ("평가 시작:" in message and "개 문제" not in message) or "전체 평가 완료" in message:
            safe_add_log("info", formatted_msg, "management")
            return
            
        # 3. 최종 통계
        if "최종 결과:" in message or "실행 시간:" in message or "============" in message:
            safe_add_log("info", formatted_msg, "management")
            return
            
        # 4. MCQ 완료 통계
        if "MCQ 완료:" in message and "정확도" in message:
            stats_match = re.search(r"정확도 ([0-9.]+)% \((\d+)/(\d+)\)", message)
            if stats_match:
                st.session_state.mcq_final_stats = {
                    'accuracy': float(stats_match.group(1)),
                    'correct': int(stats_match.group(2)), 
                    'total': int(stats_match.group(3))
                }
            safe_add_log("info", formatted_msg, "management")
            return
            
        # 5. SHORT 완료 통계
        if "단답형 완료:" in message and "EM=" in message:
            stats_match = re.search(r"EM=([0-9.]+)%, F1=([0-9.]+)%", message)
            if stats_match:
                st.session_state.short_final_stats = {
                    'em': float(stats_match.group(1)),
                    'f1': float(stats_match.group(2))
                }
            safe_add_log("info", formatted_msg, "management")
            return
            
        # 6. 검색 품질 문제
        if "검색 품질 문제" in message:
            safe_add_log("warning", formatted_msg, "management")
            return
            
        # 7. MCQ 평가 시작
        if "MCQ 평가 시작:" in message and "개 문제" in message:
            num_match = re.search(r"(\d+)개 문제", message)
            if num_match:
                num_questions = int(num_match.group(1))
                for i in range(1, num_questions + 1):
                    st.session_state.mcq_temp_logs[i] = [formatted_msg]
            return
            
        # 8. SHORT 평가 시작
        if "단답형 평가 시작:" in message and "개 문제" in message:
            num_match = re.search(r"(\d+)개 문제", message)
            if num_match:
                num_questions = int(num_match.group(1))
                for i in range(1, num_questions + 1):
                    st.session_state.short_temp_logs[i] = [formatted_msg]
            return
            
        # 9. MCQ 문제 처리
        if "MCQ-" in message:
            match = re.search(r"MCQ-(\d+)", message)
            if match:
                question_num = int(match.group(1))
                
                # 처리 시작
                if "처리 시작" in message:
                    text_match = re.search(r"처리 시작: '(.+?)'", message)
                    if text_match:
                        question_text = text_match.group(1).strip()
                        update_mcq_question(question_num, question_text, status="processing")
                    if question_num not in st.session_state.mcq_temp_logs:
                        st.session_state.mcq_temp_logs[question_num] = []
                    st.session_state.mcq_temp_logs[question_num].append(formatted_msg)
                    st.session_state.current_question = ("MCQ", question_num)
                    
                # 정답 확정
                elif "정답:" in message:
                    correct_match = re.search(r"정답: (\d+)", message)
                    if correct_match:
                        correct_num = int(correct_match.group(1))
                        predicted = correct = chr(64 + correct_num)
                        
                        for q in st.session_state.mcq_question_log:
                            if q.get('number') == question_num:
                                update_mcq_question(question_num, q.get('text', ''), 
                                                   predicted=predicted, correct=correct, 
                                                   is_correct=True, status="completed")
                                break
                    
                    if question_num in st.session_state.mcq_temp_logs:
                        for log in st.session_state.mcq_temp_logs[question_num]:
                            st.session_state.correct_process_logs.append(log)
                        del st.session_state.mcq_temp_logs[question_num]
                    st.session_state.correct_process_logs.append(formatted_msg)
                    st.session_state.correct_process_logs.append("=" * 50)
                    st.session_state.current_question = None
                    
                # 오답 확정
                elif "오답:" in message:
                    error_match = re.search(r"예측=(\d+), 정답=(\d+)", message)
                    if error_match:
                        predicted_num = int(error_match.group(1))
                        correct_num = int(error_match.group(2))
                        predicted = chr(64 + predicted_num)
                        correct = chr(64 + correct_num)
                        
                        for q in st.session_state.mcq_question_log:
                            if q.get('number') == question_num:
                                update_mcq_question(question_num, q.get('text', ''), 
                                                   predicted=predicted, correct=correct, 
                                                   is_correct=False, status="completed")
                                break
                    
                    if question_num in st.session_state.mcq_temp_logs:
                        for log in st.session_state.mcq_temp_logs[question_num]:
                            st.session_state.incorrect_process_logs.append(log)
                        del st.session_state.mcq_temp_logs[question_num]
                    st.session_state.incorrect_process_logs.append(formatted_msg)
                    st.session_state.incorrect_process_logs.append("=" * 50)
                    st.session_state.current_question = None
                    
                # 중간 처리 로그
                else:
                    if question_num in st.session_state.mcq_temp_logs:
                        st.session_state.mcq_temp_logs[question_num].append(formatted_msg)
            return
            
        # 10. SHORT 문제 처리
        if "SHORT-" in message:
            match = re.search(r"SHORT-(\d+)", message)
            if match:
                question_num = int(match.group(1))
                
                # 처리 시작
                if "처리 시작" in message:
                    text_match = re.search(r"처리 시작: '(.+?)'", message)
                    if text_match:
                        question_text = text_match.group(1).strip()
                        update_short_question(question_num, question_text, status="processing")
                    if question_num not in st.session_state.short_temp_logs:
                        st.session_state.short_temp_logs[question_num] = []
                    st.session_state.short_temp_logs[question_num].append(formatted_msg)
                    st.session_state.current_question = ("SHORT", question_num)
                    
                # 정답 확정
                elif "정답:" in message:
                    answer_match = re.search(r"정답: '([^']+)' \(EM=([0-9.]+), F1=([0-9.]+)\)", message)
                    if answer_match:
                        predicted = correct = answer_match.group(1)
                        em_score = float(answer_match.group(2))
                        f1_score = float(answer_match.group(3))
                        
                        for q in st.session_state.short_question_log:
                            if q.get('number') == question_num:
                                update_short_question(question_num, q.get('text', ''), 
                                                     predicted=predicted, correct=correct,
                                                     em_score=em_score, f1_score=f1_score, status="completed")
                                break
                    
                    if question_num in st.session_state.short_temp_logs:
                        for log in st.session_state.short_temp_logs[question_num]:
                            st.session_state.correct_process_logs.append(log)
                        del st.session_state.short_temp_logs[question_num]
                    st.session_state.correct_process_logs.append(formatted_msg)
                    st.session_state.correct_process_logs.append("=" * 50)
                    st.session_state.current_question = None
                    
                # 오답 확정
                elif "오답:" in message:
                    error_match = re.search(r"예측='([^']+)', 정답='([^']+)'.*?EM=([0-9.]+|True|False), F1=([0-9.]+)", message)
                    if error_match:
                        predicted = error_match.group(1)
                        correct = error_match.group(2)
                        em_str = error_match.group(3)
                        em_score = 1.0 if em_str == "True" else (0.0 if em_str == "False" else float(em_str))
                        f1_score = float(error_match.group(4))
                        
                        for q in st.session_state.short_question_log:
                            if q.get('number') == question_num:
                                update_short_question(question_num, q.get('text', ''), 
                                                     predicted=predicted, correct=correct,
                                                     em_score=em_score, f1_score=f1_score, status="completed")
                                break
                    
                    if question_num in st.session_state.short_temp_logs:
                        for log in st.session_state.short_temp_logs[question_num]:
                            st.session_state.incorrect_process_logs.append(log)
                        del st.session_state.short_temp_logs[question_num]
                    st.session_state.incorrect_process_logs.append(formatted_msg)
                    st.session_state.incorrect_process_logs.append("=" * 50)
                    st.session_state.current_question = None
                    
                # 중간 처리 로그
                else:
                    if question_num in st.session_state.short_temp_logs:
                        st.session_state.short_temp_logs[question_num].append(formatted_msg)
            return
            
        # 11. 진행률 업데이트
        if "진행:" in message and re.search(r"(\d+)/(\d+)", message):
            match = re.search(r"(\d+)/(\d+)", message)
            if match:
                st.session_state.evaluation_progress = {
                    'current': int(match.group(1)),
                    'total': int(match.group(2))
                }
            if st.session_state.current_question:
                q_type, q_num = st.session_state.current_question
                if q_type == "MCQ" and q_num in st.session_state.mcq_temp_logs:
                    st.session_state.mcq_temp_logs[q_num].append(formatted_msg)
                elif q_type == "SHORT" and q_num in st.session_state.short_temp_logs:
                    st.session_state.short_temp_logs[q_num].append(formatted_msg)
            return
            
        # 12. 오답 패턴 분석
        if "오답 패턴 분석" in message:
            st.session_state.incorrect_process_logs.append(formatted_msg)
            st.session_state.incorrect_process_logs.append("=" * 50)
            return
            
        # 13. 평가 과정 로그들
        if st.session_state.evaluation_running:
            # MCQ 평가 실행 중
            if "MCQ 평가 실행 중" in message:
                for q_num in st.session_state.mcq_temp_logs:
                    st.session_state.mcq_temp_logs[q_num].append(formatted_msg)
                return
                
            # 단답형 평가 실행 중
            if "단답형 평가 실행 중" in message:
                for q_num in st.session_state.short_temp_logs:
                    st.session_state.short_temp_logs[q_num].append(formatted_msg)
                return
                
            # 평가 과정 관련 키워드
            eval_keywords = [
                "검색", "Vector", "BM25", "임베딩", "LLM", "OpenAI", "Upstage",
                "응답", "처리", "하이브리드", "병합", "매칭", "유사도",
                "호출 성공", "호출 실패", "응답 완료", "후처리", "강화 모드",
                "결과 병합", "RETRIEVER", "EMBEDDER"
            ]
            
            if any(keyword in message for keyword in eval_keywords):
                if st.session_state.current_question:
                    q_type, q_num = st.session_state.current_question
                    if q_type == "MCQ" and q_num in st.session_state.mcq_temp_logs:
                        st.session_state.mcq_temp_logs[q_num].append(formatted_msg)
                    elif q_type == "SHORT" and q_num in st.session_state.short_temp_logs:
                        st.session_state.short_temp_logs[q_num].append(formatted_msg)
                else:
                    for q_num in st.session_state.mcq_temp_logs:
                        st.session_state.mcq_temp_logs[q_num].append(formatted_msg)
                    for q_num in st.session_state.short_temp_logs:
                        st.session_state.short_temp_logs[q_num].append(formatted_msg)
                return
                
        # 14. 나머지는 관리 로그로
        safe_add_log("info", formatted_msg, "management")
        
    except Exception as e:
        print(f"[ROUTING-ERROR] 로그 라우팅 실패: {e}")
        safe_add_log("error", f"라우팅 실패: {formatted_msg}", "management")

st.session_state.global_log_callback = enhanced_global_log_callback

# ========== 로그 포매팅 함수들 ==========
def format_question_log(question_log, question_type):
    """통합 문제 로그 포매팅"""
    if not question_log:
        return f"{question_type} 문제 처리 대기 중..."
    
    log_lines = []
    stats_data = {'total_f1': 0, 'completed_count': 0}
    
    for q in question_log[-10:]:
        question_text = q.get('text', '')[:50] + ('...' if len(q.get('text', '')) > 50 else '')
        
        if q.get('status') == 'processing':
            log_lines.extend([f"문제 {q['number']}: {question_text}", "→ 처리 중...", ""])
            continue
        
        if q.get('status') != 'completed':
            continue
            
        log_lines.append(f"문제 {q['number']}: {question_text}")
        
        if question_type == "MCQ":
            if q.get('choices'):
                choices_str = " ".join([f"{chr(65+i)}) {choice[:20]}{'...' if len(choice) > 20 else ''}" 
                                       for i, choice in enumerate(q['choices'][:4])])
                log_lines.append(f"선택지: {choices_str}")
            
            predicted, correct = q.get('predicted', 'N/A'), q.get('correct', 'N/A')
            status_icon = "✓" if q.get('is_correct') else "✗"
            log_lines.append(f"→ 예측: {predicted}, 정답: {correct} {status_icon}")
        
        else:  # SHORT
            predicted = q.get('predicted', 'N/A')[:30] + ('...' if len(q.get('predicted', '')) > 30 else '')
            correct = q.get('correct', 'N/A')[:30] + ('...' if len(q.get('correct', '')) > 30 else '')
            
            em_score, f1_score = q.get('em_score', 0), q.get('f1_score', 0)
            status_icon = "✓" if em_score > 0 else "✗"
            
            log_lines.extend([
                f"→ 예측: '{predicted}' {status_icon}",
                f"   정답: '{correct}'",
                f"   (EM={em_score:.2f}, F1={f1_score:.2f})"
            ])
            
            stats_data['total_f1'] += f1_score
            stats_data['completed_count'] += 1
        
        log_lines.append("")
    
    # 통계 추가
    log_lines.extend(["=" * 30])
    if question_type == "MCQ":
        add_mcq_stats(log_lines)
    else:
        add_short_stats(log_lines, stats_data)
    
    return "\n".join(log_lines)

def add_mcq_stats(log_lines):
    """MCQ 통계 추가"""
    if hasattr(st.session_state, 'mcq_final_stats'):
        stats = st.session_state.mcq_final_stats
        log_lines.append(f"MCQ 통계: 정확도 {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    else:
        stats = st.session_state.mcq_stats
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            log_lines.append(f"MCQ 통계: 정확도 {accuracy:.1%} ({stats['correct']}/{stats['total']})")

def add_short_stats(log_lines, stats_data):
    """단답형 통계 추가"""
    if hasattr(st.session_state, 'short_final_stats'):
        stats = st.session_state.short_final_stats
        log_lines.append(f"단답형 통계: EM {stats['em']:.1f}%, F1 {stats['f1']:.1f}%")
    else:
        stats = st.session_state.short_stats
        if stats['total'] > 0:
            em_accuracy = stats['correct'] / stats['total']
            avg_f1 = stats_data['total_f1'] / max(1, stats_data['completed_count'])
            log_lines.extend([
                f"단답형 통계: EM {em_accuracy:.1%} ({stats['correct']}/{stats['total']})",
                f"평균 F1: {avg_f1:.2f}"
            ])

def format_mcq_question_log():
    """MCQ 로그 포매팅"""
    return format_question_log(st.session_state.mcq_question_log, "MCQ")

def format_short_question_log():
    """단답형 로그 포매팅"""
    return format_question_log(st.session_state.short_question_log, "SHORT")

def display_real_time_monitoring():
    """실시간 평가 모니터링"""
    
    # 전체 진행률
    progress = st.session_state.evaluation_progress
    if st.session_state.evaluation_running and progress['total'] > 0:
        progress_percent = progress['current'] / progress['total']
        st.progress(progress_percent, f"전체 진행: {progress['current']}/{progress['total']} 문제 완료 ({progress_percent:.1%})")
    elif st.session_state.evaluation_running:
        st.progress(0, "평가 준비 중...")
    
    # MCQ vs 단답형
    col_mcq, col_short = st.columns(2)
    
    with col_mcq:
        st.markdown("**선다형 (MCQ) 문제별 상태**")
        mcq_log_text = format_mcq_question_log()
        st.text_area("MCQ 로그", value=mcq_log_text, height=150, 
                    key="mcq_monitor_area", label_visibility="hidden")
    
    with col_short:
        st.markdown("**단답형 문제별 상태**")
        short_log_text = format_short_question_log()
        st.text_area("단답형 로그", value=short_log_text, height=150, 
                    key="short_monitor_area", label_visibility="hidden")

# ========== 스레드 및 백그라운드 작업 ==========
def get_excel_files():
    """루트 디렉토리의 Excel 파일 목록"""
    excel_files = []
    for pattern in ["*.xlsx", "*.xls"]:
        files = glob.glob(str(project_root / pattern))
        excel_files.extend([Path(f).name for f in files if not Path(f).name.startswith("evaluation_")])
    return sorted(excel_files)

def create_safe_thread(target_func, args=None, name=None):
    """통합 스레드 생성 함수"""
    thread = threading.Thread(
        target=target_func, 
        args=args or (), 
        name=name or "background_task",
        daemon=True
    )
    
    try:
        ctx = get_script_run_ctx()
        if ctx:
            add_script_run_ctx(thread, ctx)
    except:
        pass
    
    return thread

def run_evaluation_task(excel_file, mcq_limit, short_limit):
    """평가 실행 작업"""
    try:
        retriever, llm, config = get_rag_system()
        if not all([retriever, llm, config]):
            print("RAG 인스턴스 초기화 실패")
            return
            
        from rag.evaluator import UnifiedEvaluator
        evaluator = UnifiedEvaluator(retriever=retriever, llm=llm, config=config)
        
        callback = lambda msg: enhanced_global_log_callback("progress", msg, "EVALUATOR", "evaluation")
        
        file_path = project_root / excel_file
        results = evaluator.evaluate_file(str(file_path), mcq_limit, short_limit, progress_callback=callback)
        
        st.session_state.evaluation_running = False
        st.session_state.evaluation_completed = bool(results)
        print("평가 완료!" if results else "평가 실행 실패")
        
    except Exception as e:
        print(f"평가 오류: {str(e)}")
        st.session_state.evaluation_running = False

def run_management_task(task_name):
    """관리 작업 실행"""
    def log_mgmt(log_type, message):
        try:
            safe_add_log(log_type, message, "management")
        except:
            print(f"[MANAGEMENT] {message}")
    
    try:
        log_mgmt("progress", f"{task_name} 시작...")
        
        script_map = {
            "벡터 재생성": ("reindex_upstage_docx.py", 3600),
            "BM25 재생성": ("pipeline_bm25_from_docx.py", 1800), 
            "백업": ("rag_backup.py", 300)
        }
        
        if task_name not in script_map:
            raise ValueError(f"알 수 없는 작업: {task_name}")
        
        script_name, timeout = script_map[task_name]
        
        result = subprocess.run(
            [sys.executable, str(project_root / script_name)], 
            capture_output=True, text=True, timeout=timeout
        )
        
        status = "success" if result.returncode == 0 else "failure"
        message = f"{task_name} 완료" if result.returncode == 0 else f"{task_name} 실패: code {result.returncode}"
        log_mgmt(status, message)
        
    except Exception as e:
        log_mgmt("failure", f"{task_name} 오류: {e}")

def start_evaluation_thread_safe(excel_file, mcq_limit, short_limit):
    """평가 스레드 시작"""
    st.session_state.evaluation_running = True
    st.session_state.evaluation_completed = False
    reset_evaluation_state()
    
    thread = create_safe_thread(
        target_func=run_evaluation_task,
        args=(excel_file, mcq_limit, short_limit),
        name="evaluation_thread"
    )
    thread.start()
    return thread

def run_management_task_safe(task_name):
    """관리 작업 스레드 시작"""
    thread = create_safe_thread(
        target_func=run_management_task,
        args=(task_name,),
        name=f"mgmt_{task_name.replace(' ', '_')}"
    )
    thread.start()

# ========== UI 설정 및 스타일링 ==========
st.set_page_config(page_title="통합 RAG 시스템", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    .element-container { margin-bottom: 0.3rem; }
    h1, h2, h3, h4 { margin: 0.2rem 0; line-height: 1.2; }
    .stTextArea textarea { line-height: 1.3; }
</style>
""", unsafe_allow_html=True)

# ========== UI 핸들러 함수들 ==========
def create_evaluation_controls():
    """평가 시스템 컨트롤 패널"""
    col_title, col_settings = st.columns([1, 2])
    
    with col_title:
        st.markdown("# 평가 시스템")
        st.caption("RAG 시스템 성능 평가 도구")
        
        if st.session_state.evaluation_running:
            if st.button("평가 중지", type="secondary", key="eval_stop", use_container_width=True):
                st.session_state.evaluation_running = False
                st.session_state.evaluation_completed = False
                
        elif st.session_state.evaluation_completed:
            if st.button("새 평가 준비", type="secondary", key="eval_reset", use_container_width=True):
                reset_evaluation_state()
                st.session_state.evaluation_completed = False
                st.rerun()
                
        else:
            excel_files = get_excel_files()
            selected_file = st.session_state.get('selected_file')
            can_start = selected_file and excel_files
            
            if can_start:
                if st.button("평가 시작", type="primary", key="eval_start", use_container_width=True):
                    mcq_limit = st.session_state.get('mcq_limit', 50)
                    short_limit = st.session_state.get('short_limit', 50)
                    handle_evaluation_start(selected_file, mcq_limit, short_limit)
            else:
                st.button("평가 시작", type="primary", disabled=True, key="eval_disabled", use_container_width=True)
    
    with col_settings:
        create_evaluation_settings()

def create_evaluation_settings():
    """평가 설정 패널"""
    st.write("")
    
    excel_files = get_excel_files()
    if not excel_files:
        st.error("루트 디렉토리에 Excel 파일이 없습니다.")
        return
    
    selected_file = st.selectbox(
        "평가할 Excel 파일", excel_files,
        help="루트 디렉토리의 Excel 파일 중 선택"
    )
    st.session_state.selected_file = selected_file
    
    col_mcq, col_short = st.columns(2)
    with col_mcq:
        mcq_limit = st.number_input("선다형 최대 문제 수", 1, 500, 50, 10)
        st.session_state.mcq_limit = mcq_limit
    
    with col_short:
        short_limit = st.number_input("단답형 최대 문제 수", 1, 500, 50, 10)
        st.session_state.short_limit = short_limit

def reset_evaluation_state():
    """평가 상태 초기화"""
    reset_data = {
        'mcq_question_log': [],
        'short_question_log': [],
        'correct_process_logs': [],
        'incorrect_process_logs': [],
        'mcq_stats': {'correct': 0, 'total': 0},
        'short_stats': {'correct': 0, 'total': 0},
        'evaluation_progress': {'current': 0, 'total': 0}
    }
    
    for key, value in reset_data.items():
        st.session_state[key] = value

def create_chat_interface():
    """질의응답 인터페이스 생성"""
    st.markdown("### 금융 법령 질의응답")
    st.caption("AI 기반 법령 질의응답 시스템 (단답형)")
    
    if not get_rag_available():
        st.error("RAG 시스템을 사용할 수 없습니다. 모듈을 확인해주세요.")
        st.stop()
    
    display_chat_history()
    
    st.markdown("#### 새 질문")
    
    with st.form(key="question_form", clear_on_submit=True):
        col_input, col_question, col_clear = st.columns([5, 1, 1])
        
        with col_input:
            question = st.text_input(
                "질문", 
                placeholder="예: 은행법 제1조의 목적은 무엇인가요?",
                label_visibility="hidden"
            )
        
        with col_question:
            st.write("")
            ask_button = st.form_submit_button("질문", use_container_width=True)
            
        with col_clear:
            st.write("")
            clear_button = st.form_submit_button("초기화", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and question.strip():
        process_question(question)
    elif ask_button:
        st.warning("질문을 입력해주세요")

def process_question(question):
    """질문 처리 로직"""
    with st.spinner("검색 중..."):
        try:
            contexts = retrieve(question, top_k=5)
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return
    
    if not contexts:
        answer = "죄송합니다. 관련 문서를 찾을 수 없습니다."
    else:
        with st.spinner("답변 생성 중..."):
            try:
                answer = generate_answer_short(question, contexts)
                answer = format_answer_for_chat(question, answer)
            except Exception as e:
                answer = f"답변 생성 중 오류가 발생했습니다: {e}"
    
    add_to_chat_history(question, answer, contexts)
    st.rerun()

def create_log_display():
    """2개창 로그 표시 시스템"""
    log_col1, log_col2 = st.columns(2)
    
    log_configs = [
        (log_col1, "**정답 처리과정 로그**", st.session_state.correct_process_logs, 50, "정답 처리과정 대기 중...", "correct_logs_fixed"),
        (log_col2, "**오답 처리과정 로그**", st.session_state.incorrect_process_logs, 50, "오답 처리과정 대기 중...", "incorrect_logs_fixed")
    ]
    
    for col, title, log_data, limit, default_msg, key in log_configs:
        with col:
            st.markdown(title)
            try:
                logs = log_data[-limit:] if log_data else []
                text = "\n".join([str(log) for log in logs]) if logs else default_msg
            except:
                text = default_msg
            
            st.text_area("로그 내용", value=text, height=200,
                        key=key, label_visibility="hidden")

def create_management_interface():
    """관리 도구 인터페이스"""
    st.markdown("### 시스템 관리 도구")
    st.caption("데이터 재생성 및 시스템 관리")
    
    management_tasks = [
        ("벡터 데이터베이스", "Pinecone 벡터 인덱스 재생성", 
         "DOCX 문서 → 벡터 임베딩 → Pinecone 업로드\n시간: 약 30분 ~ 2시간",
         "벡터 재생성 시작", "primary", "벡터 재생성"),
        ("BM25 검색 인덱스", "BM25 검색 인덱스 재생성",
         "DOCX 문서 → 텍스트 추출 → BM25 인덱스\n시간: 약 10분 ~ 30분", 
         "BM25 재생성 시작", "secondary", "BM25 재생성"),
        ("프로젝트 백업", "소스코드 백업",
         "모든 .py 파일 압축 → backup 폴더에 저장\n시간: 약 10초 ~ 1분",
         "백업 실행", "secondary", "백업")
    ]
    
    cols = st.columns(3)
    for i, (title, subtitle, desc, btn_text, btn_type, task_name) in enumerate(management_tasks):
        with cols[i]:
            st.markdown(f"#### {title}")
            st.markdown(f"**{subtitle}**")
            
            for line in desc.split('\n'):
                st.markdown(f"- {line}" if not line.startswith('시간:') else f"- {line}")
            
            if st.button(btn_text, type=btn_type, key=f"{task_name.replace(' ', '_')}_btn"):
                handle_management_button(task_name)
    
    st.markdown("#### 관리 로그")
    
    try:
        mgmt_logs = st.session_state.mgmt_logs[-50:] if st.session_state.mgmt_logs else []
        log_text = "\n".join([str(log) for log in mgmt_logs]) if mgmt_logs else "관리 로그가 없습니다."
    except:
        log_text = "관리 로그가 없습니다."
        
    st.text_area("관리 로그 내용", value=log_text, height=400, 
                key=f"management_logs_{st.session_state.session_id}", 
                label_visibility="hidden")
    
    if st.button("로그 초기화", key=f"clear_logs_{st.session_state.session_id}"):
        for log_key in ['correct_process_logs', 'incorrect_process_logs', 'mgmt_logs']:
            st.session_state[log_key] = []
        st.success("로그가 초기화되었습니다.")

def handle_evaluation_start(selected_file, mcq_limit, short_limit):
    """평가 시작 처리"""
    if not selected_file:
        st.error("파일을 선택해주세요.")
        return
    
    try:
        start_evaluation_thread_safe(selected_file, mcq_limit, short_limit)
        st.rerun()
    except Exception as e:
        st.session_state.evaluation_running = False
        st.error(f"평가 시작 실패: {e}")

def handle_management_button(task_name):
    """관리 도구 버튼 처리"""
    try:
        run_management_task_safe(task_name)
        st.success(f"{task_name}이 시작되었습니다. 진행 상황은 로그를 확인하세요.")
    except Exception as e:
        st.error(f"{task_name} 시작 실패: {e}")

# ========== 메인 UI 라우팅 ==========
st.sidebar.title("금융법령 RAG 시스템")
st.sidebar.markdown("---")

tab_selection = st.sidebar.radio(
    "기능 선택",
    ["질의응답", "평가 시스템", "관리 도구"],
    index=0,
    key=f"nav_{st.session_state.session_id}"
)

if tab_selection == "질의응답":
    create_chat_interface()

elif tab_selection == "평가 시스템":
    if not get_rag_available():
        st.error("RAG 시스템을 사용할 수 없습니다.")
        st.stop()
    
    create_evaluation_controls()
    
    st.markdown("#### 실시간 문제 처리 현황")
    display_real_time_monitoring()
    
    create_log_display()
    
    if st.session_state.evaluation_running:
        time.sleep(1.5)
        st.rerun()

elif tab_selection == "관리 도구":
    create_management_interface()

# 푸터
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    f"금융법령 통합 RAG 시스템 v3.2 | 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True
)