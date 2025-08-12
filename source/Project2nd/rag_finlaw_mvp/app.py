import streamlit as st
from rag.retriever import retrieve
from rag.generator import generate_answer_short, generate_answer_mcq

st.set_page_config(page_title="RAG 챗봇", layout="wide")

st.title("금융·법령 RAG 챗봇")
st.caption("AI 기반 법령 질의응답 시스템")

# 입력
question = st.text_input("질문을 입력하세요")

# 사지선다형 옵션
is_mcq = st.checkbox("사지선다형")
choices = []
if is_mcq:
    cols = st.columns(2)
    for i in range(4):
        with cols[i % 2]:
            choice = st.text_input(f"보기 {i+1}", key=f"choice_{i}")
            if choice:
                choices.append(choice)

# 검색 및 답변
if st.button("질문하기", type="primary"):
    if not question:
        st.warning("질문을 입력해주세요")
    else:
        with st.spinner("검색 중..."):
            contexts = retrieve(question, top_k=5)
        
        if not contexts:
            st.error("관련 문서를 찾을 수 없습니다")
        else:
            # 검색 결과 표시
            with st.expander("검색 결과", expanded=False):
                for i, ctx in enumerate(contexts[:3], 1):
                    st.write(f"**[{i}]** Score: {ctx['score']:.3f}")
                    st.write(f"{ctx['text'][:200]}...")
            
            # 답변 생성
            with st.spinner("답변 생성 중..."):
                if is_mcq and choices:
                    answer = generate_answer_mcq(question, choices, contexts)
                else:
                    answer = generate_answer_short(question, contexts)
            
            st.success("답변")
            st.write(answer)