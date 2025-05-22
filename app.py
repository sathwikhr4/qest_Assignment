import streamlit as st
from agents.query_agent import get_relevant_legal_text
from agents.summarization_agent import summarize_legal_text

st.set_page_config(page_title="Legal Chatbot", layout="centered")
st.title("🇮🇳 Indian Legal Chatbot (Multi-Agent)")

query = input("Ask a legal question: ")

if query:
    with st.spinner("Fetching legal information..."):
        context = get_relevant_legal_text(query)
        simplified_answer = summarize_legal_text(context)

    st.subheader("Answer:")
    st.write(simplified_answer)

    with st.expander("Show Extracted Legal Text"):
        for section in context:
            st.write(section)