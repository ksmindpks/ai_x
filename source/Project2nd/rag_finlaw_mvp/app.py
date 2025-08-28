import streamlit as st
import os
import sys
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import time
import queue
import traceback
import uuid
import re

# 스레드 안전성을 위한 추가 import
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = lambda t, ctx=None: None
    get_script_run_ctx = lambda: None

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def get_rag_system():
    """Streamlit 세션 기반 RAG 시스템 관리"""
    if (hasattr(st.session_state, 'rag_retriever') and 
        hasattr(st.session_state, 'rag_llm') and 
        hasattr(st.session_state, 'rag_config')):
        print("[SESSION-REUSE] 기존 세션의 RAG 인스턴스 재사용")
        return st.session_state.rag_retriever, st.session_state.rag_llm, st.session_state.rag_config
    
    print("[SESSION-NEW] 새 세션을 위한 RAG 인스턴스 생성")
    try:
        from rag.hybrid_retriever import HybridRetriever
        from rag.llm_bridge import HybridLLM
        from config import get_config
        
        config = get_config()
        
        print("[SESSION-NEW] Retriever 초기화 중...")
        retriever = HybridRetriever(config)
        
        print("[SESSION-NEW] LLM Bridge 초기화 중...")
        llm = HybridLLM(config)
        
        if not retriever or not llm:
            raise RuntimeError("RAG 컴포넌트 초기화 실패")
        
        st.session_state.rag_retriever = retriever
        st.session_state.rag_llm = llm
        st.session_state.rag_config = config
        
        print("[SESSION-NEW] RAG 인스턴스 세션에 저장 완료")
        return retriever, llm, config
        
    except Exception as e:
        print(f"[SESSION-ERROR] RAG 초기화 실패: {e}")
        import traceback
        print(f"[SESSION-DEBUG] 상세 오류: {traceback.format_exc()}")
        return None, None, None

def get_rag_available():
    """RAG 사용 가능 여부 확인"""
    retriever, llm, config = get_rag_system()
    return retriever is not None and llm is not None and config is not None

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
    """질의응답용 답변을 완전한 문장으로 포맷팅"""
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
    """채팅 기록에 질문과 답변 추가 - 검색 결과 포함"""
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
    """채팅 기록 표시 - 답변과 검색 결과 함께"""
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

# 세션 상태 초기화 - 5개 창 시스템
def safe_session_init():
    """안전한 세션 상태 초기화 - 5개 창 완전 분리 시스템"""
    try:
        if 'evaluation_running' not in st.session_state:
            st.session_state.evaluation_running = False
        if 'evaluation_completed' not in st.session_state:
            st.session_state.evaluation_completed = False
        
        # 1번창: MCQ 문제별 상태 + 통계
        if 'mcq_question_log' not in st.session_state:
            st.session_state.mcq_question_log = []
        if 'mcq_stats' not in st.session_state:
            st.session_state.mcq_stats = {'correct': 0, 'total': 0}
        
        # 2번창: 단답형 문제별 상태 + 통계  
        if 'short_question_log' not in st.session_state:
            st.session_state.short_question_log = []
        if 'short_stats' not in st.session_state:
            st.session_state.short_stats = {'correct': 0, 'total': 0}
        
        # 3번창: 시스템 진행 로그만 (나머지 로그)
        if 'system_logs' not in st.session_state:
            st.session_state.system_logs = []
        
        # 4번창: 정답 문제들의 전체 처리과정 누적
        if 'correct_process_logs' not in st.session_state:
            st.session_state.correct_process_logs = []
        
        # 5번창: 오답 문제들의 전체 처리과정 누적
        if 'incorrect_process_logs' not in st.session_state:
            st.session_state.incorrect_process_logs = []
        
        # 기타
        if 'mgmt_logs' not in st.session_state:
            st.session_state.mgmt_logs = []
        if 'evaluation_progress' not in st.session_state:
            st.session_state.evaluation_progress = {'current': 0, 'total': 0}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
        
        # 문제별 로그 버퍼
        if 'question_logs_buffer' not in st.session_state:
            st.session_state.question_logs_buffer = {}
            
    except Exception as e:
        print(f"[ERROR] 세션 초기화 실패: {e}")

safe_session_init()

def update_mcq_question(question_num, question_text, choices=None, predicted=None, correct=None, is_correct=None, status="processing"):
    """MCQ 문제 상태 업데이트 - 예측값 포함"""
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
    """단답형 문제 상태 업데이트 - 예측값 포함"""
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

def safe_add_system_log(message):
    """3번창 시스템 로그 추가"""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        if hasattr(st.session_state, 'system_logs'):
            st.session_state.system_logs.append(log_entry)
            if len(st.session_state.system_logs) > 100:
                st.session_state.system_logs.pop(0)
    except Exception as e:
        print(f"[SYSTEM-LOG-ERROR] {e}")

# 5개 창 로그 라우팅 시스템
def enhanced_global_log_callback(log_type, message, module="", category="evaluation"):
    """5개 창 완전 분리 로그 라우팅 시스템"""
    formatted_msg = f"[{module}] {message}" if module else message
    
    # 콘솔 출력 제거 - 각 모듈에서 이미 출력하므로 중복 방지
    # print(f"[{log_type.upper()}] {formatted_msg}")
    
    try:
        # 1,2번창: 문제별 상태 시작
        if ("MCQ-" in message or "SHORT-" in message) and "처리 시작" in message:
            match = re.search(r"(MCQ|SHORT)-(\d+)", message)
            if match:
                question_type = match.group(1)
                question_num = int(match.group(2))
                question_key = f"{question_type}-{question_num}"
                
                # 4,5번창을 위한 로그 버퍼 시작
                st.session_state.question_logs_buffer[question_key] = [formatted_msg]
                
                # 1,2번창 상태 업데이트
                if question_type == "MCQ":
                    text_match = re.search(r"처리 시작: '(.+?)'", message)
                    if text_match:
                        question_text = text_match.group(1).strip()
                        update_mcq_question(question_num, question_text, status="processing")
                elif question_type == "SHORT":
                    text_match = re.search(r"처리 시작: '(.+?)'", message)
                    if text_match:
                        question_text = text_match.group(1).strip()
                        update_short_question(question_num, question_text, status="processing")
                
                return  # 3번창에는 보내지 않음
        
        # 문제 처리 중간 과정들을 버퍼에 수집
        elif any(pattern in message for pattern in ["MCQ-", "SHORT-"]):
            # 현재 처리 중인 문제 찾기 - 버퍼에만 수집하고 즉시 표시하지 않음
            for question_key in list(st.session_state.question_logs_buffer.keys()):
                if question_key.replace("-", "-") in message:
                    st.session_state.question_logs_buffer[question_key].append(formatted_msg)
                    break
            
            # MCQ 완료 패턴 감지 - 새로운 형식 지원
            mcq_complete_match = re.search(r"MCQ-(\d+) 오답: 예측=(\d+), 정답=(\d+)", message)
            mcq_correct_match = re.search(r"MCQ-(\d+) 정답: (\d+)", message)
            
            # SHORT 완료 패턴 감지
            short_complete_match = re.search(r"SHORT-(\d+) (정답|오답):", message)
            
            # 문제 완료 시에만 1,2번창 업데이트 + 4,5번창 라우팅
            if mcq_complete_match or mcq_correct_match or short_complete_match:
                question_type = None
                question_num = None
                is_correct = False
                
                # MCQ 처리
                if mcq_complete_match:
                    question_type = "MCQ"
                    question_num = int(mcq_complete_match.group(1))
                    predicted_num = int(mcq_complete_match.group(2))
                    correct_num = int(mcq_complete_match.group(3))
                    predicted = chr(64 + predicted_num)  # 1->A, 2->B
                    correct = chr(64 + correct_num)
                    is_correct = False
                    
                    for q in st.session_state.mcq_question_log:
                        if q.get('number') == question_num:
                            update_mcq_question(question_num, q.get('text', ''), 
                                               predicted=predicted, correct=correct, 
                                               is_correct=False, status="completed")
                            break
                
                elif mcq_correct_match:
                    question_type = "MCQ"
                    question_num = int(mcq_correct_match.group(1))
                    correct_num = int(mcq_correct_match.group(2))
                    predicted = correct = chr(64 + correct_num)
                    is_correct = True
                    
                    for q in st.session_state.mcq_question_log:
                        if q.get('number') == question_num:
                            update_mcq_question(question_num, q.get('text', ''), 
                                               predicted=predicted, correct=correct, 
                                               is_correct=True, status="completed")
                            break
                
                # SHORT 처리  
                elif short_complete_match:
                    question_type = "SHORT"
                    question_num = int(short_complete_match.group(1))
                    is_correct = short_complete_match.group(2) == "정답"
                    
                    if is_correct:
                        # "SHORT-1 정답: '노동조합의 대표자' (EM=1.0, F1=0.67)" 패턴
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
                    else:
                        # "SHORT-1 오답: 예측='제12조', 정답='관할 노동위원회', 오류유형=context_mismatch, EM=False, F1=0.00" 패턴
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
                
                # 4,5번창으로 전체 처리과정 라우팅 - 완료된 경우에만
                if question_type and question_num is not None:
                    question_key = f"{question_type}-{question_num}"
                    if question_key in st.session_state.question_logs_buffer:
                        question_logs = st.session_state.question_logs_buffer[question_key]
                        
                        # 문제가 완전히 완료된 후에만 4,5번창에 추가
                        if is_correct:
                            # 4번창: 정답 처리과정 누적
                            st.session_state.correct_process_logs.extend(question_logs)
                            st.session_state.correct_process_logs.append("=" * 50)
                        else:
                            # 5번창: 오답 처리과정 누적 (완료된 경우에만)
                            st.session_state.incorrect_process_logs.extend(question_logs)
                            st.session_state.incorrect_process_logs.append("=" * 50)
                        
                        # 버퍼에서 제거
                        del st.session_state.question_logs_buffer[question_key]
                
                return  # 3번창에는 보내지 않음
        
        # 통계 로그 파싱 (로그에서 직접 가져오기)
        elif "MCQ 완료:" in message and "정확도" in message:
            # "MCQ 완료: 정확도 100.0% (3/3) - 6.9초" 파싱
            stats_match = re.search(r"정확도 ([0-9.]+)% \((\d+)/(\d+)\)", message)
            if stats_match:
                accuracy = float(stats_match.group(1))
                correct = int(stats_match.group(2))
                total = int(stats_match.group(3))
                st.session_state.mcq_final_stats = {
                    'accuracy': accuracy,
                    'correct': correct, 
                    'total': total
                }
            safe_add_system_log(formatted_msg)

        elif "단답형 완료:" in message and "EM=" in message:
            # "단답형 완료: EM=100.0%, F1=88.9% - 4.6초" 파싱  
            stats_match = re.search(r"EM=([0-9.]+)%, F1=([0-9.]+)%", message)
            if stats_match:
                em = float(stats_match.group(1))
                f1 = float(stats_match.group(2))
                st.session_state.short_final_stats = {
                    'em': em,
                    'f1': f1
                }
            safe_add_system_log(formatted_msg)
        
        # 진행률 파싱 (3번창으로)
        elif "진행" in message or "/" in message:
            match = re.search(r"(\d+)[/:](\d+)", message)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                st.session_state.evaluation_progress = {
                    'current': current,
                    'total': total
                }
            safe_add_system_log(formatted_msg)
        
        # 시스템 초기화/설정 로그는 관리 로그로 라우팅
        elif any(keyword in message for keyword in [
            "SESSION-", "CONFIG-", "RETRIEVER-", "LLM-", "EMBEDDER-", 
            "초기화", "설정", "검증", "로드", "연결", "완료"
        ]):
            safe_add_log("info", formatted_msg, "management")
        
        # 3번창: 평가 관련 시스템 로그만
        else:
            safe_add_system_log(formatted_msg)
    
    except Exception as e:
        print(f"[ROUTING-ERROR] 로그 라우팅 실패: {e}")
        safe_add_system_log(formatted_msg)

st.session_state.global_log_callback = enhanced_global_log_callback

def format_mcq_question_log():
    """MCQ 문제 로그를 텍스트로 포맷팅 - 예측값 및 통계 포함"""
    if not st.session_state.mcq_question_log:
        return "MCQ 문제 처리 대기 중..."
    
    log_lines = []
    for q in st.session_state.mcq_question_log[-10:]:
        question_text = q.get('text', '')[:50] + ('...' if len(q.get('text', '')) > 50 else '')
        
        if q.get('status') == 'processing':
            log_lines.append(f"문제 {q['number']}: {question_text}")
            log_lines.append("→ 처리 중...")
        
        elif q.get('status') == 'completed':
            log_lines.append(f"문제 {q['number']}: {question_text}")
            
            # 선택지 표시
            if q.get('choices'):
                choices_str = " ".join([f"{chr(65+i)}) {choice[:20]}{'...' if len(choice) > 20 else ''}" 
                                       for i, choice in enumerate(q['choices'][:4])])
                log_lines.append(f"선택지: {choices_str}")
            
            # 예측값과 정답 표시
            predicted = q.get('predicted', 'N/A')
            correct = q.get('correct', 'N/A')
            
            if q.get('is_correct'):
                log_lines.append(f"→ 예측: {predicted}, 정답: {correct} ✓")
            else:
                log_lines.append(f"→ 예측: {predicted}, 정답: {correct} ✗")
        
        log_lines.append("")
    
    # 통계 표시 - 개선된 버전
    if hasattr(st.session_state, 'mcq_final_stats'):
        stats = st.session_state.mcq_final_stats
        log_lines.append("=" * 30)
        log_lines.append(f"MCQ 통계: 정확도 {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    else:
        # 기존 코드 유지
        stats = st.session_state.mcq_stats
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            log_lines.append("=" * 30)
            log_lines.append(f"MCQ 통계: 정확도 {accuracy:.1%} ({stats['correct']}/{stats['total']})")
    
    return "\n".join(log_lines)

def format_short_question_log():
    """단답형 문제 로그를 텍스트로 포맷팅 - 예측값 및 통계 포함"""
    if not st.session_state.short_question_log:
        return "단답형 문제 처리 대기 중..."
    
    log_lines = []
    total_f1 = 0
    completed_count = 0
    
    for q in st.session_state.short_question_log[-10:]:
        question_text = q.get('text', '')[:50] + ('...' if len(q.get('text', '')) > 50 else '')
        
        if q.get('status') == 'processing':
            log_lines.append(f"문제 {q['number']}: {question_text}")
            log_lines.append("→ 처리 중...")
        
        elif q.get('status') == 'completed':
            log_lines.append(f"문제 {q['number']}: {question_text}")
            
            # 예측값과 정답 표시
            predicted = q.get('predicted', 'N/A')[:30] + ('...' if len(q.get('predicted', '')) > 30 else '')
            correct = q.get('correct', 'N/A')[:30] + ('...' if len(q.get('correct', '')) > 30 else '')
            
            em_score = q.get('em_score', 0)
            f1_score = q.get('f1_score', 0)
            
            if em_score > 0:
                log_lines.append(f"→ 예측: '{predicted}' ✓")
                log_lines.append(f"   정답: '{correct}'")
                log_lines.append(f"   (EM={em_score:.2f}, F1={f1_score:.2f})")
            else:
                log_lines.append(f"→ 예측: '{predicted}' ✗")
                log_lines.append(f"   정답: '{correct}'")
                log_lines.append(f"   (EM={em_score:.2f}, F1={f1_score:.2f})")
            
            total_f1 += f1_score
            completed_count += 1
        
        log_lines.append("")
    
    # 통계 표시 - 개선된 버전
    if hasattr(st.session_state, 'short_final_stats'):
        stats_final = st.session_state.short_final_stats
        log_lines.append("=" * 30)
        log_lines.append(f"단답형 통계: EM {stats_final['em']:.1f}%, F1 {stats_final['f1']:.1f}%")
    else:
        # 기존 코드 유지
        stats = st.session_state.short_stats
        if stats['total'] > 0:
            em_accuracy = stats['correct'] / stats['total']
            avg_f1 = total_f1 / max(1, completed_count)
            log_lines.append("=" * 30)
            log_lines.append(f"단답형 통계: EM {em_accuracy:.1%} ({stats['correct']}/{stats['total']})")
            log_lines.append(f"평균 F1: {avg_f1:.2f}")
    
    return "\n".join(log_lines)

def display_real_time_monitoring():
    """5개 창 실시간 평가 모니터링 - 높이 압축 버전"""
    
    # 전체 진행률
    progress = st.session_state.evaluation_progress
    if st.session_state.evaluation_running and progress['total'] > 0:
        progress_percent = progress['current'] / progress['total']
        st.progress(progress_percent, f"전체 진행: {progress['current']}/{progress['total']} 문제 완료 ({progress_percent:.1%})")
    elif st.session_state.evaluation_running:
        st.progress(0, "평가 준비 중...")
    
    # 상단: MCQ vs 단답형 (2컬럼) - 높이 압축
    col_mcq, col_short = st.columns(2)
    
    with col_mcq:
        st.markdown("**1. 선다형 (MCQ) 문제별 상태**")
        mcq_log_text = format_mcq_question_log()
        st.text_area(
            "MCQ 문제 현황", 
            value=mcq_log_text, 
            height=150,
            key="mcq_monitor_area",
            label_visibility="hidden"
        )
    
    with col_short:
        st.markdown("**2. 단답형 문제별 상태**")
        short_log_text = format_short_question_log()
        st.text_area(
            "단답형 문제 현황", 
            value=short_log_text, 
            height=150,
            key="short_monitor_area",
            label_visibility="hidden"
        )

def get_excel_files():
    """루트 디렉토리의 Excel 파일 목록"""
    excel_patterns = ["*.xlsx", "*.xls"]
    excel_files = []
    
    for pattern in excel_patterns:
        files = glob.glob(str(project_root / pattern))
        for f in files:
            filename = Path(f).name
            if not filename.startswith("evaluation_"):
                excel_files.append(filename)
    
    return sorted(excel_files)

def run_evaluation_thread(excel_file, mcq_limit, short_limit):
    """평가 실행 스레드"""
    try:
        print("[EVAL-THREAD] 평가 시작...")
        
        retriever, llm, config = get_rag_system()
        
        if not retriever or not llm:
            print("[EVAL-THREAD] 세션 RAG 인스턴스 초기화 실패")
            return
            
        print("[EVAL-THREAD] 세션 RAG 인스턴스 사용")
        
        from rag.evaluator import UnifiedEvaluator
        evaluator = UnifiedEvaluator(retriever=retriever, llm=llm, config=config)
        
        def thread_progress_callback(message):
            try:
                enhanced_global_log_callback("progress", message, "EVALUATOR", "evaluation")
            except Exception as e:
                print(f"[CALLBACK-ERROR] {e}: {message}")
        
        file_path = project_root / excel_file
        results = evaluator.evaluate_file(
            str(file_path), 
            mcq_limit, 
            short_limit,
            progress_callback=thread_progress_callback
        )
        
        if results:
            print("[EVAL-THREAD] 평가 완료!")
            st.session_state.evaluation_running = False
            st.session_state.evaluation_completed = True
        else:
            print("[EVAL-THREAD] 평가 실행 실패")
            
    except Exception as e:
        print(f"[EVAL-THREAD] 평가 스레드 오류: {str(e)}")
        import traceback
        print(f"[EVAL-THREAD] 상세 오류: {traceback.format_exc()}")
    
    finally:
        print("[EVAL-THREAD] 평가 스레드 종료")

def start_evaluation_thread_safe(excel_file, mcq_limit, short_limit):
    """스레드 안전한 평가 시작"""
    st.session_state.evaluation_running = True
    st.session_state.evaluation_completed = False
    
    # 상태 초기화
    st.session_state.mcq_question_log = []
    st.session_state.short_question_log = []
    st.session_state.mcq_stats = {'correct': 0, 'total': 0}
    st.session_state.short_stats = {'correct': 0, 'total': 0}
    st.session_state.evaluation_progress = {'current': 0, 'total': 0}
    st.session_state.system_logs = []
    st.session_state.correct_process_logs = []
    st.session_state.incorrect_process_logs = []
    
    thread = threading.Thread(
        target=run_evaluation_thread,
        args=(excel_file, mcq_limit, short_limit),
        name="evaluation_thread",
        daemon=True
    )
    
    try:
        ctx = get_script_run_ctx()
        if ctx:
            add_script_run_ctx(thread, ctx)
    except:
        pass
    
    thread.start()
    return thread

def run_management_task_safe(task_name):
    """관리 작업을 위한 스레드 안전한 래퍼"""
    def task_wrapper():
        def mgmt_log(log_type, message):
            try:
                safe_add_log(log_type, message, "management")
            except Exception:
                print(f"[MANAGEMENT] {message}")
        
        try:
            mgmt_log("progress", f"{task_name} 시작...")
            
            if task_name == "벡터 재생성":
                import subprocess
                result = subprocess.run([sys.executable, str(project_root / "reindex_upstage_docx.py")], 
                                      capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    mgmt_log("success", f"{task_name} 완료")
                else:
                    mgmt_log("failure", f"{task_name} 실패: return code {result.returncode}")
            
            elif task_name == "BM25 재생성":
                import subprocess
                result = subprocess.run([sys.executable, str(project_root / "pipeline_bm25_from_docx.py")], 
                                      capture_output=True, text=True, timeout=1800)
                if result.returncode == 0:
                    mgmt_log("success", f"{task_name} 완료")
                else:
                    mgmt_log("failure", f"{task_name} 실패: return code {result.returncode}")
            
            elif task_name == "백업":
                import subprocess
                result = subprocess.run([sys.executable, str(project_root / "rag_backup.py")], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    mgmt_log("success", f"{task_name} 완료")
                else:
                    mgmt_log("failure", f"{task_name} 실패: return code {result.returncode}")
                    
        except Exception as e:
            mgmt_log("failure", f"{task_name} 오류: {e}")
    
    thread = threading.Thread(target=task_wrapper, daemon=True)
    
    try:
        ctx = get_script_run_ctx()
        if ctx:
            add_script_run_ctx(thread, ctx)
    except:
        pass
    
    thread.start()

def safe_add_log(log_type, message, category="evaluation"):
    """관리 로그 추가 (기존 호환성 유지)"""
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
            prefix = prefix_map.get(log_type, "[LOG]")  # 기본값
            mgmt_entry = f"{prefix} {log_entry}"
            st.session_state.mgmt_logs.append(mgmt_entry)
            if len(st.session_state.mgmt_logs) > 50:
                st.session_state.mgmt_logs.pop(0)
    except Exception as e:
        print(f"[LOG-ERROR] {e}")

# 페이지 설정
st.set_page_config(
    page_title="통합 RAG 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 상단 여백 줄이기용 CSS 추가
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    .element-container {
        margin-bottom: 0.05rem;
    }
    h1, h2, h3, h4 {
        margin-top: 0.05rem;
        margin-bottom: 0.05rem;
        line-height: 1.1;
    }
    .stMarkdown {
        margin-bottom: 0.02rem;
        line-height: 1.1;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 0.1rem;
    }
    p {
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .stTextArea textarea {
        line-height: 1.1;
    }
    .stButton {
        margin-bottom: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

def handle_evaluation_start(selected_file, mcq_limit, short_limit):
    """평가 시작 처리"""
    if not selected_file:
        st.error("파일을 선택해주세요.")
        return
    
    try:
        thread = start_evaluation_thread_safe(selected_file, mcq_limit, short_limit)
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

# 사이드바 - 기능 선택
st.sidebar.title("금융법령 RAG 시스템")
st.sidebar.markdown("---")

tab_selection = st.sidebar.radio(
    "기능 선택",
    ["질의응답", "평가 시스템", "관리 도구"],
    index=0,
    key=f"navigation_radio_{st.session_state.session_id}"
)

# 메인 컨텐츠 
if tab_selection == "질의응답":
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
        with st.spinner("검색 중..."):
            try:
                contexts = retrieve(question, top_k=5)
            except Exception as e:
                st.error(f"검색 오류: {e}")
                st.stop()
        
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
        
    elif ask_button and not question.strip():
        st.warning("질문을 입력해주세요")

elif tab_selection == "평가 시스템":
    if not get_rag_available():
        st.error("RAG 시스템을 사용할 수 없습니다.")
        st.stop()
    
    # 상단 좌우 분할: 타이틀 vs 설정
    col_title, col_settings = st.columns([1, 2])
    
    with col_title:
        st.markdown("### 평가 시스템")
        st.caption("RAG 시스템 성능 평가 도구")
        
        # 상태와 버튼을 같은 라인에 배치
        if st.session_state.evaluation_running:
            col_status, col_btn = st.columns([2, 1])
            with col_status:
                st.info("평가 진행 중...")
            with col_btn:
                if st.button("평가 중지", type="secondary", key="stop_eval", use_container_width=True):
                    st.session_state.evaluation_running = False
                    st.session_state.evaluation_completed = False
        elif st.session_state.evaluation_completed:
            col_status, col_btn = st.columns([2, 1])
            with col_status:
                st.success("평가 완료")
            with col_btn:
                if st.button("새 평가 준비", type="secondary", key="new_eval", use_container_width=True):
                    st.session_state.evaluation_completed = False
                    st.session_state.mcq_question_log = []
                    st.session_state.short_question_log = []
                    st.session_state.mcq_stats = {'correct': 0, 'total': 0}
                    st.session_state.short_stats = {'correct': 0, 'total': 0}
                    st.session_state.evaluation_progress = {'current': 0, 'total': 0}
                    st.rerun()
        else:
            col_status, col_btn = st.columns([2, 1])
            with col_status:
                st.success("평가 대기 중")
            with col_btn:
                # 평가 시작 버튼 (설정에서 가져온 변수들 사용)
                excel_files = get_excel_files()
                selected_file = st.session_state.get('selected_file')
                can_start = not st.session_state.evaluation_running and selected_file
                
                if st.button("평가 시작", type="primary", disabled=not can_start, key="start_eval", use_container_width=True):
                    mcq_limit = st.session_state.get('mcq_limit', 50)
                    short_limit = st.session_state.get('short_limit', 50)
                    handle_evaluation_start(selected_file, mcq_limit, short_limit)
    
    with col_settings:
        st.write("")
        
        excel_files = get_excel_files()
        if not excel_files:
            st.error("루트 디렉토리에 Excel 파일이 없습니다.")
            selected_file = None
        else:
            selected_file = st.selectbox(
                "평가할 Excel 파일",
                excel_files,
                help="루트 디렉토리의 Excel 파일 중 선택"
            )
            st.session_state.selected_file = selected_file
        
        col_mcq, col_short = st.columns(2)
        with col_mcq:
            mcq_limit = st.number_input(
                "선다형 최대 문제 수",
                min_value=1,
                max_value=500,
                value=50,
                step=10
            )
            st.session_state.mcq_limit = mcq_limit
        
        with col_short:
            short_limit = st.number_input(
                "단답형 최대 문제 수",
                min_value=1,
                max_value=500,
                value=50,
                step=10
            )
            st.session_state.short_limit = short_limit
    
    # 실시간 모니터링 화면
    st.markdown("#### 실시간 문제 처리 현황")
    display_real_time_monitoring()
    
    # 하단: 5개 창 시스템 (3컬럼) - 안정적인 렌더링
    log_col1, log_col2, log_col3 = st.columns(3)
    
    # 모든 창을 한 번에 안정적으로 렌더링
    with log_col1:
        st.write("**3. 시스템 진행 로그**")
        system_text = "\n".join(st.session_state.system_logs[-25:]) if st.session_state.system_logs else ""
        st.text_area("시스템 로그", value=system_text, height=150, key="system_logs_fixed", label_visibility="hidden")
    
    with log_col2:
        st.write("**4. 정답 처리과정 로그**")
        correct_text = "\n".join(st.session_state.correct_process_logs[-50:]) if st.session_state.correct_process_logs else ""
        st.text_area("정답 로그", value=correct_text, height=150, key="correct_logs_fixed", label_visibility="hidden")
    
    with log_col3:
        st.write("**5. 오답 처리과정 로그**")
        incorrect_text = "\n".join(st.session_state.incorrect_process_logs[-50:]) if st.session_state.incorrect_process_logs else ""
        st.text_area("오답 로그", value=incorrect_text, height=150, key="incorrect_logs_fixed", label_visibility="hidden")
    
    # 자동 새로고침 - 안정성 개선
    if st.session_state.evaluation_running:
        time.sleep(1)
        st.rerun()

elif tab_selection == "관리 도구":
    st.markdown("### 시스템 관리 도구")
    st.caption("데이터 재생성 및 시스템 관리")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 벡터 데이터베이스")
        st.markdown("**Pinecone 벡터 인덱스 재생성**")
        st.markdown("- DOCX 문서 → 벡터 임베딩 → Pinecone 업로드")
        st.markdown("- 시간: 약 30분 ~ 2시간")
        
        if st.button("벡터 재생성 시작", type="primary", key="vector_btn"):
            handle_management_button("벡터 재생성")
    
    with col2:
        st.markdown("#### BM25 검색 인덱스")
        st.markdown("**BM25 검색 인덱스 재생성**")
        st.markdown("- DOCX 문서 → 텍스트 추출 → BM25 인덱스")
        st.markdown("- 시간: 약 10분 ~ 30분")
        
        if st.button("BM25 재생성 시작", type="secondary", key="bm25_btn"):
            handle_management_button("BM25 재생성")
    
    with col3:
        st.markdown("#### 프로젝트 백업")
        st.markdown("**소스코드 백업**")
        st.markdown("- 모든 .py 파일 압축")
        st.markdown("- backup 폴더에 저장")
        st.markdown("- 시간: 약 10초 ~ 1분")
        
        if st.button("백업 실행", type="secondary", key="backup_btn"):
            handle_management_button("백업")
    
    st.markdown("#### 관리 로그")
    
    if st.session_state.mgmt_logs:
        log_text = "\n".join(st.session_state.mgmt_logs[-50:])
        st.text_area("관리 로그", value=log_text, height=400, key="management_logs", label_visibility="hidden")
    else:
        st.text_area("관리 로그", value="관리 로그가 없습니다.", height=400, key="management_empty", label_visibility="hidden")
    
    if st.button("로그 초기화", key=f"clear_logs_{st.session_state.session_id}"):
        st.session_state.system_logs = []
        st.session_state.correct_process_logs = []
        st.session_state.incorrect_process_logs = []
        st.session_state.mgmt_logs = []
        st.success("로그가 초기화되었습니다.")

# 푸터
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    f"금융법령 통합 RAG 시스템 v3.0 | 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True
)