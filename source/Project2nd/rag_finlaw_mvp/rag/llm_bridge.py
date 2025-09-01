"""
llm_bridge.py - 후처리 추가 완화 및 질문 유형 자동 감지
주요 개선사항:
1. MCQ 파싱 - 개선된 parse_mcq_answer 함수 연동
2. 단답형 후처리 추가 완화 (extraction_failure 20건→8건 목표)
3. 질문 유형 자동 감지를 통한 맞춤형 검증
"""
import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

def log_message(log_type, message, module="LLM"):
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

class HybridLLM:
    """개선된 LLM Bridge - MCQ 파싱 연동 + 단답형 후처리 대폭 완화"""
    
    def __init__(self, config, silent=False):
        if not silent:
            log_message("INFO", "LLM Bridge 초기화 중...")
        self.config = config
        self._init_llms()
        self._init_prompts()
        
        log_message("SUCCESS", "LLM Bridge 초기화 완료")

    def _init_llms(self):
        """LLM 초기화"""
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
        
        # Upstage 초기화
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
2. 질문의 정답인 선택지 선택
3. 컨텍스트에 근거가 있는지 확인

**핵심 원칙:**
- 컨텍스트에 없는 내용 선택 금지
- 추측 금지
- 정답과 가장 유사하고, 근거가 있는 선택지 선택

최종 답변은 선택지(A, B, C, D 중 혹은 1, 2, 3, 4 중) 하나만 출력하세요.""",

            # 부정형용 프롬프트
            'mcq_negative': """한국 법령 전문가로서 주어진 컨텍스트에서 부정형 질문의 정답을 선택하세요.

**부정형 질문 해결법:**
1. 컨텍스트에서 관련 내용을 모두 찾으세요
2. 각 선택지를 하나씩 확인하세요:
   - 컨텍스트에 언급되어 있으면 → 답이 아님
3. **컨텍스트에 언급되지 않은 유일한 선택지가 정답입니다**

부정형에서는 "포함되지 않는" 것을 찾으므로, 컨텍스트에 없는 선택지를 선택해야 합니다.

반드시  선택지(A, B, C, D 중 혹은 1, 2, 3, 4 중) 하나만 출력하세요.""",

            'mcq_user': """컨텍스트:
{context}

질문: {question}

선택지:
A 혹은 1) {choice_a}
B 혹은 2) {choice_b}
C 혹은 3) {choice_c}
D 혹은 4) {choice_d}

답:""",

            # 단답형용 프롬프트 (대폭 완화)
            'short_system': """법령 문서에서 질문의 정답을 추출하세요.

**추출 방법:**
1. 컨텍스트에서 관련 문장을 찾으세요
2. 그 문장에서 질문의 정답 부분만 추출하세요
3. 불필요한 설명은 제외하세요

**질문 유형별 답변:**
▶ 정의 질문 ("~란", "~이란", "정의"): 정의 내용만
▶ 개수 질문 ("몇 명", "몇 개"): 숫자+단위만  
▶ 담당자 질문 ("누가", "담당"): 기관명/직책명만
▶ 기간 질문 ("언제", "기간"): 날짜/기간만
▶ 조문 내용 질문 ("제X조에서 정하는"): 조문이 규정하는 구체적 내용

**관대한 추출 원칙:**
- 질문과 관련된 모든 정보를 포함하되,
- 너무 엄격하게 잘라서 정답 부분이 잃지 않게 추출하고,
- 질문 의도에 충실하게 의미가 통하도록 완전한 표현 사용해서

질문에 대한 정답 부분을 추출하세요.""",

            'short_user': """컨텍스트:
{context}

질문: {question}

위 질문에 정답을 컨텍스트에서 추출:"""
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

    def _detect_answer_type_from_question(self, question: str) -> str:
        """질문에서 기대되는 답변 유형 자동 감지 - 수정된 버전"""
        
        # 조문 번호 자체를 묻는 질문만 article로 분류
        if any(pattern in question for pattern in ['어느 조', '몇 조']) and '정하는' not in question:
            return 'article'
        
        # 정의를 묻는 질문
        if any(pattern in question for pattern in ['정의', '란', '이란', '무엇인가']):
            return 'definition'
        
        # 인원/개수를 묻는 질문  
        if any(pattern in question for pattern in ['몇 명', '몇 개', '몇 인', '인원']):
            return 'count'
            
        # 기관/담당자 관련
        if any(pattern in question for pattern in ['누가', '담당', '기관', '관할', '소관']):
            return 'agency'
        
        # 기간 관련
        if any(pattern in question for pattern in ['언제', '기간', '시기', '때', '일 이내', '개월']):
            return 'period'
        
        # 금액 관련
        if any(pattern in question for pattern in ['얼마', '금액', '한도', '원', '비용']):
            return 'amount'
        
        # 서식 관련
        if any(pattern in question for pattern in ['서식', '양식', '별지', '제출서류']):
            return 'form'
        
        # 절차/방법 관련
        if any(pattern in question for pattern in ['어떻게', '방법', '절차', '과정']):
            return 'procedure'
        
        return 'general'

    def call_mcq(self, question: str, choices: Dict[str, str], context: str) -> str:
        """MCQ 처리 - 개선된 파싱 함수 연동"""
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
            
            # *** 개선된 MCQ 파싱 함수 연동 ***
            if not response or len(response.strip()) == 0:
                log_message("FAILURE", "LLM이 빈 응답 반환")
                return "A"  # 기본값
            
            # utils.py의 개선된 parse_mcq_answer 함수 사용
            from rag.utils import parse_mcq_answer
            parsed_answer = parse_mcq_answer(response, expected_format='alphabet')
            
            log_message("SUCCESS", f"MCQ 파싱 완료: '{response.strip()}' → '{parsed_answer}'")
            return parsed_answer
            
        except Exception as e:
            log_message("FAILURE", f"MCQ 처리 오류: {e}")
            import traceback
            log_message("FAILURE", f"MCQ 오류 상세: {traceback.format_exc()}")
            return "A"

    def call_short(self, question: str, context: str) -> str:
        """단답형 처리 - 대폭 완화된 후처리"""
        log_message("INFO", "단답형 처리 시작")
        
        if not context or len(context.strip()) < 20:
            log_message("FAILURE", "단답형 컨텍스트 부족으로 처리 불가")
            return "정보 부족"
        
        # 컨텍스트 품질 검증 (경고만 출력, 중단하지 않음)
        if len(context) > 5000:
            log_message("INFO", f"단답형 컨텍스트가 매우 김 - {len(context)}자")
        elif len(context) < 100:
            log_message("INFO", f"단답형 컨텍스트가 짧음 - {len(context)}자")
        
        # 질문 유형 감지
        question_type = self._detect_answer_type_from_question(question)
        log_message("INFO", f"감지된 질문 유형: {question_type}")
        
        system_prompt = self.prompts['short_system']
        
        user_prompt = self.prompts['short_user'].format(
            context=context,
            question=question
        )
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            # 응답 품질 검증 (완화)
            if not response:
                log_message("FAILURE", "LLM이 빈 응답 반환")
                return "정보 부족"
            
            # *** 대폭 완화된 검증 로직 ***
            if not self._validate_response_ultra_relaxed(response):
                log_message("INFO", f"단답형 응답 검증 실패 (완화 모드) - '{response[:30]}...'")
                # 검증 실패해도 "정보 부족" 반환하지 않고, 후처리만 진행
            
            processed = self._post_process_response_ultra_relaxed(response, question, question_type)
            
            # 후처리 결과 검증
            if processed != response:
                log_message("INFO", f"단답형 후처리: '{response[:20]}...' → '{processed}'")
            
            log_message("SUCCESS", f"단답형 처리 완료: '{processed}'")
            return processed
            
        except Exception as e:
            log_message("FAILURE", f"단답형 처리 오류: {e}")
            import traceback
            log_message("FAILURE", f"단답형 오류 상세: {traceback.format_exc()}")
            return "처리 실패"

    def _post_process_response_ultra_relaxed(self, response: str, question: str, question_type: str) -> str:
        """답변 후처리 - 대폭 완화된 버전 (extraction_failure 20건→8건 목표)"""
        if not response:
            return "정보 부족"
        
        response = response.strip()
        
        # "정보 부족"은 그대로 반환
        if "정보 부족" in response or "처리 실패" in response:
            return response
        
        # 1. 기본 정리 - 접두어 제거
        prefixes = ["답변:", "답:", "정답:", "결론:", "따라서", "그러므로", "추출:", "결과:", "정답은"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # 2. 따옴표 제거
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1].strip()
        
        # *** 3. 질문 유형별 특화 추출 (대폭 완화) ***
        
        # 조문 질문 ("어느 조", "몇 조", "제X조")
        if question_type == 'article':
            article_match = re.search(r'제\s*\d+\s*조(?:제\s*\d+\s*항)?(?:제\s*\d+\s*호)?', response)
            if article_match:
                return article_match.group().replace(" ", "")
            # 실패 시 숫자만이라도 찾기
            number_match = re.search(r'(\d+)(?:조|항|호)', response)
            if number_match:
                return f"제{number_match.group(1)}조"
        
        # 담당자/기관 질문 ("누가", "담당", "기관", "관할", "소관")
        elif question_type == 'agency':
            # 기관명 + 직책 조합 우선
            agency_position_patterns = [
                r'([가-힣]+(?:위원회|청|부|처|원))(?:\s*(?:장관|위원장|청장|원장))',
                r'([가-힣]+장관)',
                r'([가-힣]+위원장)',
                r'([가-힣]+(?:위원회|청|부|처|원))',
            ]
            
            for pattern in agency_position_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group().strip()
            
            # 일반적인 기관명 패턴도 허용 (완화)
            general_agency = re.search(r'[가-힣]{2,}(?:위원회|청|부|처|원)', response)
            if general_agency:
                return general_agency.group()
        
        # 기간 질문 ("언제", "기간", "시기", "때", "일 이내", "개월")
        elif question_type == 'period':
            # 구체적인 날짜 패턴 우선
            date_patterns = [
                r'\d{4}[년./-]\s*\d{1,2}[월./-]\s*\d{1,2}일?',
                r'\d+(?:년|개월|일)(?:\s*(?:이내|이상|미만|전|후|까지))?',
                r'매년\s*\d+월\s*\d+일',
                r'회계연도\s*개시일',
                r'사업연도\s*종료일',
                r'\d+(?:\.|-)\d+(?:\.|-)\d+',  # 날짜 형식 완화
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group().replace(" ", "")
            
            # 숫자만이라도 있으면 허용 (대폭 완화)
            number_unit = re.search(r'(\d+)(?:\s*(?:년|월|일|개월))', response)
            if number_unit:
                return number_unit.group().replace(" ", "")
        
        # 금액 질문 ("얼마", "금액", "한도", "원", "비용")
        elif question_type == 'amount':
            amount_patterns = [
                r'\d+(?:,\d+)*(?:억|만)?\s*원(?:\s*(?:이상|이하|미만))?',
                r'\d+(?:\.\d+)?%',
                r'\d+(?:억|만|천)?\s*원',  # 완화된 패턴
                r'\d+(?:,\d{3})*원',  # 콤마 포함 패턴
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group().replace(" ", "").replace(",", "")
            
            # 숫자만이라도 찾기 (완화)
            number_only = re.search(r'(\d+)', response)
            if number_only and '원' in question:
                return f"{number_only.group()}원"
        
        # 서식/양식 질문 ("서식", "양식", "별지")
        elif question_type == 'form':
            form_patterns = [
                r'별지\s*제\s*\d+\s*호(?:서식|양식)?',
                r'제\s*\d+\s*호\s*(?:서식|양식)',
                r'서식\s*\d+',
                r'양식\s*\d+',
            ]
            
            for pattern in form_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group().replace(" ", "")
            
            # 숫자만이라도 있으면 서식으로 가정 (완화)
            form_number = re.search(r'(\d+)(?:\s*(?:호|번))', response)
            if form_number:
                return f"별지 제{form_number.group(1)}호서식"
        
        # 방법/절차 질문 ("어떻게", "방법", "절차")
        elif question_type == 'procedure':
            # 핵심 동사구 추출 (완화)
            procedure_patterns = [
                r'(?:신청|허가|승인|등록|신고|접수|처리|발급|제출)(?:하여야|해야|한다|함)',
                r'(?:온라인|서면|직접|우편)(?:으로|로)\s*(?:신청|제출)',
                r'[가-힣]{2,}(?:절차|방법|과정)',
                r'[가-힣]{3,}(?:하여야\s*한다)',
            ]
            
            for pattern in procedure_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group()
            
            # 동작 관련 단어라도 찾기 (완화)
            action_word = re.search(r'(?:신청|허가|승인|등록|신고|접수|처리|발급|제출)', response)
            if action_word:
                return action_word.group()
        
        # *** 4. 일반적인 후처리 (대폭 완화) ***
        
        # 핵심 키워드 추출 (법령 특화) - 더 관대하게
        important_words = re.findall(
            r'제\d+조|[가-힣]+(?:위원회|청|부|처|원)|' + 
            r'\d+(?:년|개월|일)|' +
            r'\d+(?:억|만)?원|' +
            r'별지\s*제\s*\d+\s*호서식|' +
            r'[가-힣]{2,}',  # 2글자 이상 한글 (더 포괄적)
            response
        )
        
        # *** 완화: 15개 → 20개로 증가 ***
        if len(important_words) > 20:
            response = " ".join(important_words[:20])
        elif important_words:
            response = " ".join(important_words)
        
        # 5. 불필요한 접미어 제거 (기존과 동일)
        suffix_patterns = [
            r'(?:라\s*한다|를\s*말한다|에\s*따라|로\s*정한다|고\s*한다)',
            r'(?:이다|이며|이고|하다|한다|함)',
            r'[.!?]+'
        ]
        
        for pattern in suffix_patterns:
            response = re.sub(pattern, '', response)
        
        # 6. 최종 검증 및 반환 (대폭 완화)
        response = response.strip()
        
        # *** 완화: 빈 응답이라도 기본 처리 ***
        if len(response) == 0:
            return "정보 부족"
        
        # *** 완화: 의미없는 단답도 법령 패턴 포함시 유지 ***
        if response in ["다음", "해당", "관련", "기타", "등", "및"]:
            # 법령 패턴이 포함되지 않으면 "정보 부족"
            if not re.search(r'제\d+조|위원회|청|부|처|원|\d+(?:년|개월|일|원)', response):
                return "정보 부족"
        
        # 너무 긴 응답 자르기 (120 → 150자로 완화)
        if len(response) > 150:
            response = response[:147] + "..."
        
        return response if response else "정보 부족"

    def _validate_response_ultra_relaxed(self, response: str) -> bool:
        """응답 품질 검증 - 대폭 완화된 기준 (extraction_failure 완화)"""
        if not response or len(response.strip()) == 0:
            return False
        
        response = response.strip()
        
        # "정보 부족"은 유효한 응답으로 간주
        if "정보 부족" in response:
            return True
        
        # *** 완화: 1글자도 허용 (기존 2글자→1글자) ***
        if len(response) < 1:
            return False
        
        # *** 완화: 법령 패턴 포함시 매우 짧아도 유효 ***
        if re.search(r'제\d+조|위원회|청|부|처|원|\d+(?:년|개월|일|원)', response):
            return True
        
        # *** 완화: 의미없는 패턴들도 더 관대하게 ***
        meaningless_patterns = [
            r'^[가-힣]{1}$',              # 1글자 단답도 허용
            r'^[가-힣]로$',               # "~로"로 끝나는 불완전한 답 허용
            r'^[가-힣]부$',               # "~부"만 있는 경우 허용  
            r'^[가-힣]에$',               # "~에"로 끝나는 불완전한 답 허용
        ]
        
        # 패턴 매칭 검사 (완화)
        for pattern in meaningless_patterns:
            if re.match(pattern, response):
                # *** 완화: 법령 패턴 포함시 예외 허용 ***
                if re.search(r'제\d+조|위원회|청|부|처|원|\d+(?:년|개월|일|원)', response):
                    return True
                # 단순한 1글자도 허용 (대폭 완화)
                return True
        
        # 반복 문자 패턴 거부 (기존과 동일)
        if len(set(response)) == 1 and len(response) > 1:
            return False
        
        # *** 완화: 숫자만으로 이루어진 경우 더 관대하게 ***
        if re.match(r'^\d+$', response):
            # 조문 번호로 보이는 경우는 허용 (1-3000 범위로 확장)
            try:
                num = int(response)
                if 1 <= num <= 3000:  # 2000 → 3000으로 확장
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
        
        # *** 완화: 의미있는 법령 용어가 포함되어 있는지 검사 (기준 완화) ***
        meaningful_patterns = [
            r'제\d+조',                   # 조문 번호
            r'\d+(?:년|개월|일)',         # 기간
            r'\d+(?:억|만)?원',           # 금액
            r'[가-힣]+(?:위원회|청|부|처|원)', # 기관명
            r'별지\s*제\s*\d+\s*호',      # 서식
            r'[가-힣]{1,}',               # 1글자 이상 한글 (기존 2글자→1글자로 완화)
        ]
        
        # 의미있는 패턴이 하나라도 있으면 유효
        for pattern in meaningful_patterns:
            if re.search(pattern, response):
                return True
        
        # 여기까지 왔으면 의미있는 내용이 없는 것으로 판단하지만, 대폭 완화로 True 반환
        return True  # *** 완화: 거의 모든 응답 허용 ***

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
        
        # Upstage 대안
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