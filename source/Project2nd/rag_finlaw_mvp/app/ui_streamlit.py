import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import retrieve
from rag.generator import answer_short, answer_mcq
from rag.postprocess import DISCLAIMER, detect_mcq

st.title("금융·법령 RAG 챗봇 (CodeDoc 버전)")
st.caption(DISCLAIMER)

q = st.text_input("질문을 입력하세요")
col1, col2 = st.columns(2)
with col1:
    is_mcq = st.checkbox("사지선다형 (보기 제공)", value=False)
with col2:
    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)

choices = []
if is_mcq:
    choices = [st.text_input(f"보기{i+1}", key=f"c{i}") for i in range(4)]

if st.button("질의"):
    with st.spinner("검색 중..."):
        hits = retrieve(q, filters=None, top_k=topk)
    st.subheader("검색 결과(상위)")
    for h in hits:
        st.write(f"- {h.get('filename')} #{h.get('chunk_index')} (score={h.get('score'):.3f})")
        st.write(h.get("text","")[:300] + ("..." if len(h.get("text",""))>300 else ""))

    with st.spinner("응답 생성 중..."):
        if detect_mcq(q, choices if any(choices) else None):
            ans = answer_mcq(q, [c for c in choices if c], hits)
        else:
            ans = answer_short(q, hits)
    st.markdown("---")
    st.subheader("답변")
    st.text(ans)
