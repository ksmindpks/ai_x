# 자동 완성이 안될 경우 (가상환경 설정 안되어 있는 경우) : ctrl+shift+p -> 가상환경 선택
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import warnings
warnings.filterwarnings('ignore')

# 1. client 생성
load_dotenv('.env')
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
# 2. assistant 생성
assistant_cs = client.beta.assistants.create(
    name='CustomerSupportBot',
    instructions='당신은 고객 지원 챗봇임. 사용자 문의에 대해 200자 이내로 친절한 답변을 함',
    model='gpt-4o-mini',
#     tools=[]
)
# thread 생성: 기억담당
thread_cs = client.beta.threads.create()
print("챗봇이 시작. 종료를 원하면 '종료'나 'exit'를 입력, 모든 대화 이력은 저장")

while True:
    user_input = input('user:').strip()
    if user_input.lower() in ('종료', 'exit'):
        print('챗봇 종료. 이용에 감사!')
        break
    if user_input=='':
        continue
    # 4~6 : user_input을 thread_cs에 추가하고 실행한 후 최종 답변 출력
    # 4. 스레드에 user_input을 추가
    client.beta.threads.messages.create(
        thread_id=thread_cs.id,
        role='user',
        content=user_input
    )
    # 5. 실행
    client.beta.threads.runs.create_and_poll(
        thread_id=thread_cs.id,
        assistant_id=assistant_cs.id
    )
    # 6. 최종답변 출력
    messages = client.beta.threads.messages.list(thread_id=thread_cs.id)
    assistant_reply = messages.data[0]
    reply_text = assistant_reply.content[0].text.value
    print(f'user : {user_input}')
    print(f'assistant : {reply_text}')
    
# 7. 대화 이력 뽑아, 파일 출력
sorted_messages = sorted(messages.data, # 정렬 list
                        key=lambda msg: msg.created_at) # 정렬 기준

with open('data/ch07_chat_history.txt', 'wt', encoding='utf-8') as f:
    for message in sorted_messages:
        # 생성 시각(message.created_at)을 datetime으로 변환
        datetime_info = time.localtime(message.created_at)
        # 보기 좋은 문자열 형식으로 변환
        output_str    = time.strftime('%y-%m-%d %H:%M:%S', datetime_info)
        # 파일 출력
        f.write('{:9}({}) : {}\n'.format(message.role,
                                        output_str,
                                        message.content[0].text.value))