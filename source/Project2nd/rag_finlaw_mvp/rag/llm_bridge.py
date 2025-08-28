"""
llm_bridge.py - 로그 시스템 통합 버전 + Upstage 클라이언트 초기화 수정
"""
import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

def log_message(log_type, message, module="LLM"):
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

class HybridLLM:
    """로그 시스템 통합된 LLM Bridge + Upstage 클라이언트 초기화 수정"""
    
    def __init__(self, config):
        log_message("INFO", "LLM Bridge 초기화 중...")
        self.config = config
        self._init_llms()
        self._init_prompts()
        
        log_message("SUCCESS", "LLM Bridge 초기화 완료")

    def _init_llms(self):
        """LLM 초기화 - Upstage 클라이언트 초기화 추가"""
        self.clients = {}
        
        # OpenAI 초기화
        if self.config.openai_api_key:
            try:
                import openai
                self.clients['openai'] = openai.OpenAI(
                    api_key=self.config.openai_api_key,
                    timeout=self.config.llm_timeout
                )
                log_message("SUCCESS", "OpenAI 초기화 완료")
            except Exception as e:
                log_message("FAILURE", f"OpenAI 초기화 실패: {e}")
        
        # Upstage 초기화 추가 (기존 누락된 부분)
        if self.config.upstage_api_key:
            try:
                import openai
                self.clients['upstage'] = openai.OpenAI(
                    api_key=self.config.upstage_api_key,
                    base_url="https://api.upstage.ai/v1",
                    timeout=self.config.llm_timeout
                )
                log_message("SUCCESS", "Upstage 초기화 완료")
            except Exception as e:
                log_message("FAILURE", f"Upstage 초기화 실패: {e}")
        
        if not self.clients:
            log_message("FAILURE", "사용 가능한 LLM이 없음")

    def _init_prompts(self):
        """프롬프트 시스템 초기화"""
        self.prompts = {
            # MCQ용 프롬프트
            'mcq_system': """한국 법령 전문가로서 주어진 컨텍스트에서 정답을 선택하세요.

**필수 검증 절차:**
1. 각 선택지를 컨텍스트와 비교하여 언급 여부 확인
2. 질문에 가장 정확히 부합하는 선택지 선택
3. 컨텍스트에 명확한 근거가 있는지 확인

**핵심 원칙:**
- 컨텍스트에 없는 내용은 절대 선택 금지
- 추측하지 말고 명시된 내용만 근거로 사용
- 애매하면 가장 구체적인 근거가 있는 선택지 선택

최종 답변은 반드시 A, B, C, D 중 하나만 출력하세요.""",

            # 부정형용 프롬프트
            'mcq_negative': """이는 부정형 질문입니다.

**부정형 질문 해결법:**
1. 컨텍스트에서 관련 내용을 모두 찾으세요
2. 각 선택지를 하나씩 확인하세요:
   - 컨텍스트에 언급되어 있으면 → 틀린 답
3. **컨텍스트에 언급되지 않은 유일한 선택지가 정답입니다**

부정형에서는 "포함되지 않는" 것을 찾으므로, 컨텍스트에 없는 선택지를 선택해야 합니다.

반드시 A, B, C, D 중 하나만 출력하세요.""",

            'mcq_user': """컨텍스트:
{context}

질문: {question}

선택지:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

답:""",

            # 단답형용 프롬프트
            'short_system': """법령 문서에서 질문에 직접 답하는 핵심만 추출하세요.

**추출 방법:**
1. 컨텍스트에서 관련 문장을 찾으세요
2. 그 문장에서 질문에 직접 답하는 핵심 부분만 추출하세요
3. 설명 부분은 제외하세요

**질문 유형별 추출:**
▶ 조문 질문 ("어느 조", "몇 조"): "제X조"만
▶ 담당자 질문 ("누가", "담당"): 기관명/직책명만  
▶ 기간 질문 ("언제", "기간"): 날짜/기간만
▶ 방식 질문 ("어떻게", "방식"): 핵심 방법만

질문에 대한 직접적인 답변 부분만 추출하세요.""",

            'short_user': """컨텍스트:
{context}

질문: {question}

위 질문에 직접 답하는 핵심만 컨텍스트에서 추출:"""
        }

    def _detect_negative_question(self, question: str) -> bool:
        """부정형 질문 감지"""
        negative_indicators = [
            "포함되지 않는", "해당하지 않는", "맞지 않는", "아닌 것",
            "제외되는", "틀린 것", "잘못된 것", "예외"
        ]
        
        for indicator in negative_indicators:
            if indicator in question:
                log_message("INFO", f"부정형 표현 감지: '{indicator}'")
                return True
        
        # 문맥적 패턴
        if re.search(r"다음.*?중.*?(?:아닌|않은|없는)", question):
            log_message("INFO", "부정형 패턴 감지")
            return True
        
        return False

    def call_mcq(self, question: str, choices: Dict[str, str], context: str) -> str:
        """MCQ 처리 - 디버깅 정보 포함"""
        log_message("INFO", "MCQ 처리 시작")
        
        # 입력 데이터 검증
        if not context or len(context.strip()) < 10:
            log_message("FAILURE", "MCQ 컨텍스트가 너무 짧음 또는 비어있음")
        
        if len(choices) != 4:
            log_message("FAILURE", f"MCQ 선택지 개수 이상 - {len(choices)}개 (예상: 4개)")
        
        # 부정형 질문 감지
        is_negative = self._detect_negative_question(question)
        
        # 프롬프트 선택
        if is_negative:
            system_prompt = self.prompts['mcq_negative']
            log_message("INFO", "부정형 질문으로 감지 - 특수 프롬프트 사용")
        else:
            system_prompt = self.prompts['mcq_system']
        
        user_prompt = self.prompts['mcq_user'].format(
            context=context,
            question=question,
            choice_a=choices.get('A', ''),
            choice_b=choices.get('B', ''),
            choice_c=choices.get('C', ''),
            choice_d=choices.get('D', '')
        )
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            # 응답 검증
            if not response or len(response.strip()) == 0:
                log_message("FAILURE", "LLM이 빈 응답 반환")
                return "A"  # 기본값
            
            cleaned_response = response.strip()
            
            # 예상치 못한 응답 형식 감지
            if len(cleaned_response) > 10:
                log_message("FAILURE", f"MCQ 응답이 예상보다 김 - '{cleaned_response[:20]}...'")
            
            if not any(char in cleaned_response for char in ['A', 'B', 'C', 'D', '1', '2', '3', '4']):
                log_message("FAILURE", f"MCQ 응답에 유효한 선택지 없음 - '{cleaned_response}'")
            
            log_message("SUCCESS", f"MCQ 응답 완료: '{cleaned_response}'")
            return cleaned_response
            
        except Exception as e:
            log_message("FAILURE", f"MCQ 처리 오류: {e}")
            # 디버그를 위해 상세 정보 출력
            import traceback
            log_message("FAILURE", f"MCQ 오류 상세: {traceback.format_exc()}")
            return "A"

    def call_short(self, question: str, context: str) -> str:
        """단답형 처리 - 디버깅 정보 포함"""
        log_message("INFO", "단답형 처리 시작")
        
        if not context or len(context.strip()) < 20:
            log_message("FAILURE", "단답형 컨텍스트 부족으로 처리 불가")
            return "정보 부족"
        
        # 컨텍스트 품질 검증
        if len(context) > 5000:
            log_message("FAILURE", f"단답형 컨텍스트가 매우 김 - {len(context)}자")
        elif len(context) < 100:
            log_message("FAILURE", f"단답형 컨텍스트가 매우 짧음 - {len(context)}자")
        
        system_prompt = self.prompts['short_system']
        
        user_prompt = self.prompts['short_user'].format(
            context=context,
            question=question
        )
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            # 응답 품질 검증
            if not response:
                log_message("FAILURE", "LLM이 빈 응답 반환")
                return "정보 부족"
            
            if not self._validate_response(response):
                log_message("FAILURE", f"단답형 응답 검증 실패 - '{response[:30]}...'")
                return "정보 부족"
            
            processed = self._post_process_response(response, question)
            
            # 후처리 결과 검증
            if processed == response:
                log_message("INFO", "단답형 후처리: 변경사항 없음")
            else:
                log_message("INFO", f"단답형 후처리: '{response[:20]}...' → '{processed}'")
            
            log_message("SUCCESS", f"단답형 처리 완료: '{processed}'")
            return processed
            
        except Exception as e:
            log_message("FAILURE", f"단답형 처리 오류: {e}")
            # 디버그를 위해 상세 정보 출력
            import traceback
            log_message("FAILURE", f"단답형 오류 상세: {traceback.format_exc()}")
            return "처리 실패"

    def _post_process_response(self, response: str, question: str) -> str:
        """답변 후처리 - 질문 유형별 최적화된 추출"""
        if not response or "정보 부족" in response:
            return response
        
        response = response.strip()
        
        # 1. 기본 정리 - 접두어 제거
        prefixes = ["답변:", "답:", "정답:", "결론:", "따라서", "그러므로", "추출:", "결과:", "정답은"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # 2. 따옴표 제거
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1].strip()
        
        # 3. 질문 유형별 특화 추출
        
        # 조문 질문 ("어느 조", "몇 조", "제X조")
        if any(keyword in question for keyword in ["어느 조", "몇 조", "조에서", "조의"]):
            article_match = re.search(r'제\s*\d+\s*조(?:제\s*\d+\s*항)?(?:제\s*\d+\s*호)?', response)
            if article_match:
                return article_match.group().replace(" ", "")
        
        # 담당자/기관 질문 ("누가", "담당", "기관", "관할", "소관")
        elif any(keyword in question for keyword in ["누가", "담당", "기관", "관할", "소관"]):
            person_patterns = [
                r'([가-힣]+(?:위원회|청|부|처|원))(?:장관?|위원장)?',
                r'([가-힣]+장관)',
                r'([가-힣]+위원장)',
                r'([가-힣]+(?:부|청|원|처))',
            ]
            for pattern in person_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group()
        
        # 기간 질문 ("언제", "기간", "시점", "때", "일 이내", "개월")
        elif any(keyword in question for keyword in ["언제", "기간", "시점", "때", "일 이내", "개월"]):
            period_patterns = [
                r'\d+(?:년|개월|일)(?:\s*(?:이내|이상|미만|전|후))?',
                r'매년\s*\d+월\s*\d+일',
                r'회계연도\s*개시일',
                r'사업연도\s*종료일',
            ]
            for pattern in period_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group().replace(" ", "")
        
        # 금액 질문 ("얼마", "금액", "한도", "원", "비용")
        elif any(keyword in question for keyword in ["얼마", "금액", "한도", "원", "비용"]):
            amount_patterns = [
                r'\d+(?:,\d+)*(?:억|만)?\s*원(?:\s*(?:이상|이하|미만))?',
                r'\d+(?:\.\d+)?%',
            ]
            for pattern in amount_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group().replace(" ", "").replace(",", "")
        
        # 서식/양식 질문 ("서식", "양식", "별지")
        elif any(keyword in question for keyword in ["서식", "양식", "별지", "제출서류"]):
            form_match = re.search(r'별지\s*제\s*\d+\s*호(?:서식|양식)?', response)
            if form_match:
                return form_match.group().replace(" ", "")
        
        # 방법/절차 질문 ("어떻게", "방법", "절차")
        elif any(keyword in question for keyword in ["어떻게", "방법", "절차", "과정"]):
            # 핵심 동사구 추출
            method_patterns = [
                r'(?:신청|허가|승인|등록|신고|접수|처리|발급|제출)(?:하여야|해야|한다)',
                r'(?:온라인|서면|직접|우편)(?:으로|로)\s*(?:신청|제출)',
            ]
            for pattern in method_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group()
        
        # 4. 일반적인 후처리
        
        # 핵심 키워드 추출 (법령 특화)
        important_words = re.findall(
            r'제\d+조|[가-힣]+(?:위원회|청|부|처|원)|' + 
            r'\d+(?:년|개월|일)|' +
            r'\d+(?:억|만)?원|' +
            r'별지\s*제\s*\d+\s*호서식|' +
            r'[가-힣]{2,}', 
            response
        )
        
        # 8개 단어로 제한
        if len(important_words) > 8:
            response = " ".join(important_words[:8])
        elif important_words:
            response = " ".join(important_words)
        
        # 5. 불필요한 접미어 제거
        suffix_patterns = [
            r'(?:라\s*한다|를\s*말한다|에\s*따라|로\s*정한다|고\s*한다)$',
            r'(?:이다|이며|이고|하다|한다|함)$',
            r'[.!?]+$'
        ]
        
        for pattern in suffix_patterns:
            response = re.sub(pattern, '', response)
        
        # 6. 최종 검증 및 반환
        response = response.strip()
        
        # 너무 짧거나 의미없는 응답 필터링
        if len(response) <= 1 or response in ["다음", "해당", "관련", "기타"]:
            return "정보 부족"
        
        # 너무 긴 응답 자르기
        if len(response) > 100:
            response = response[:97] + "..."
        
        return response if response else "정보 부족"

    def _validate_response(self, response: str) -> bool:
        """응답 품질 검증 - 법령 특화"""
        if not response or len(response.strip()) <= 1:
            return False
        
        response = response.strip()
        
        # "정보 부족"은 유효한 응답으로 간주
        if "정보 부족" in response:
            return True
        
        # 너무 짧은 응답 거부
        if len(response) <= 2:
            return False
        
        # 의미없는 패턴들 - 법령 특화
        meaningless_patterns = [
            r'^[가-힣]{1,2}$',           # 1-2글자 단답
            r'^[가-힣]로$',              # "~로"로 끝나는 불완전한 답
            r'^[가-힣]부$',              # "~부"만 있는 경우
            r'^[가-힣]에$',              # "~에"로 끝나는 불완전한 답
            r'^다음$',                    # "다음"만 있는 경우
            r'^해당$',                    # "해당"만 있는 경우  
            r'^관련$',                    # "관련"만 있는 경우
            r'^기타$',                    # "기타"만 있는 경우
            r'^상기$',                    # "상기"만 있는 경우
            r'^위$',                      # "위"만 있는 경우
            r'^본$',                      # "본"만 있는 경우
            r'^각$',                      # "각"만 있는 경우
            r'^모든?$',                   # "모든" 또는 "모"만 있는 경우
            r'^전체$',                    # "전체"만 있는 경우
            r'^일반$',                    # "일반"만 있는 경우
            r'^특별$',                    # "특별"만 있는 경우
            r'^기본$',                    # "기본"만 있는 경우
            r'^필요$',                    # "필요"만 있는 경우
            r'^가능$',                    # "가능"만 있는 경우
            r'^불가$',                    # "불가"만 있는 경우
        ]
        
        # 패턴 매칭 검사
        for pattern in meaningless_patterns:
            if re.match(pattern, response):
                return False
        
        # 반복 문자 패턴 거부 (예: "가가가", "111")
        if len(set(response)) == 1 and len(response) > 1:
            return False
        
        # 숫자만으로 이루어진 경우 (조문번호 제외) 검증
        if re.match(r'^\d+$', response):
            # 조문 번호로 보이는 경우는 허용 (1-1000 범위)
            try:
                num = int(response)
                if 1 <= num <= 1000:
                    return True
                else:
                    return False
            except ValueError:
                return False
        
        # 영어만 있는 경우 거부
        if re.match(r'^[A-Za-z\s]+$', response):
            return False
        
        # 특수문자만 있는 경우 거부
        if re.match(r'^[^\w\s가-힣]+$', response):
            return False
        
        # 의미있는 법령 용어가 포함되어 있는지 검사
        meaningful_patterns = [
            r'제\d+조',                   # 조문 번호
            r'\d+(?:년|개월|일)',         # 기간
            r'\d+(?:억|만)?원',           # 금액
            r'[가-힣]+(?:위원회|청|부|처|원)', # 기관명
            r'별지\s*제\s*\d+\s*호',      # 서식
            r'[가-힣]{3,}',               # 3글자 이상 한글 (일반 명사)
        ]
        
        # 의미있는 패턴이 하나라도 있으면 유효
        for pattern in meaningful_patterns:
            if re.search(pattern, response):
                return True
        
        # 여기까지 왔으면 의미있는 내용이 없는 것으로 판단
        return False

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """LLM 호출 - 우선순위 기반 + 향상된 오류 처리"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 사용 가능한 클라이언트가 있는지 확인
        if not self.clients:
            log_message("FAILURE", "사용 가능한 LLM 클라이언트가 없음")
            raise RuntimeError("LLM 서비스를 사용할 수 없습니다")
        
        # OpenAI 우선 시도
        if 'openai' in self.clients:
            try:
                result = self._call_openai(messages)
                log_message("SUCCESS", "OpenAI 호출 성공")
                return result
            except Exception as e:
                error_msg = str(e)
                # 구체적인 오류 유형 로깅
                if "rate_limit" in error_msg.lower():
                    log_message("FAILURE", f"OpenAI 요청 한도 초과: {e}")
                elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                    log_message("FAILURE", f"OpenAI 인증 오류: {e}")
                elif "timeout" in error_msg.lower():
                    log_message("FAILURE", f"OpenAI 타임아웃: {e}")
                else:
                    log_message("FAILURE", f"OpenAI 호출 실패: {e}")
        
        # Upstage 대안 (수정된 부분 - 이제 정상 작동함)
        if 'upstage' in self.clients:
            try:
                result = self._call_upstage(messages)
                log_message("SUCCESS", "Upstage 호출 성공")
                return result
            except Exception as e:
                error_msg = str(e)
                # 구체적인 오류 유형 로깅
                if "rate_limit" in error_msg.lower():
                    log_message("FAILURE", f"Upstage 요청 한도 초과: {e}")
                elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                    log_message("FAILURE", f"Upstage 인증 오류: {e}")
                elif "timeout" in error_msg.lower():
                    log_message("FAILURE", f"Upstage 타임아웃: {e}")
                else:
                    log_message("FAILURE", f"Upstage 호출 실패: {e}")
        
        # 모든 서비스 실패
        log_message("FAILURE", "모든 LLM 서비스 호출 실패")
        
        # 사용 가능한 서비스 목록 로깅
        available_services = list(self.clients.keys())
        log_message("FAILURE", f"시도한 서비스: {', '.join(available_services)}")
        
        raise RuntimeError("모든 LLM 서비스 사용 불가")

    def _call_openai(self, messages: List[Dict]) -> str:
        """OpenAI API 호출 - 향상된 오류 처리"""
        try:
            response = self.clients['openai'].chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=self.config.max_tokens
            )
            
            # 응답 내용 검증
            if not response.choices or not response.choices[0].message:
                raise ValueError("OpenAI 응답이 비어있음")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("OpenAI 응답 내용이 None")
            
            return content.strip()
            
        except Exception as e:
            # OpenAI 특정 오류들을 더 구체적으로 처리
            error_type = type(e).__name__
            log_message("FAILURE", f"OpenAI API 오류 ({error_type}): {e}")
            raise

    def _call_upstage(self, messages: List[Dict]) -> str:
        """Upstage API 호출 - 향상된 오류 처리"""
        try:
            response = self.clients['upstage'].chat.completions.create(
                model="solar-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=self.config.max_tokens
            )
            
            # 응답 내용 검증
            if not response.choices or not response.choices[0].message:
                raise ValueError("Upstage 응답이 비어있음")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Upstage 응답 내용이 None")
            
            return content.strip()
            
        except Exception as e:
            # Upstage 특정 오류들을 더 구체적으로 처리
            error_type = type(e).__name__
            log_message("FAILURE", f"Upstage API 오류 ({error_type}): {e}")
            raise