import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

def askGpt(prompt):
    "매개변수로 받은 prompt요청 결과 반환"
    load_dotenv()
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role":"system", "content":"당신은 한국어로 잘 요약하는 전문가입니다."},
            {"role":"user", "content":prompt}]
    )
    return response.choices[0].message.content

# 기능 구현
def main():
    st.header("요약프로그램")
    st.markdown("---")
    message = st.text_area("요약할 글을 입력하세요")
    if st.button("요약"):
        prompt = f"""다음 텍스트를 두 줄로 요약. 글머리 기호 형식을 사용
                텍스트 : {message}"""
        result = askGpt(prompt)
        st.info(result)

if __name__ == "__main__":
    main()