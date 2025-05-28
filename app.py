import streamlit as st
from utils import process_pdf_and_save_to_vectorstore
from rag_chain import get_rag_chain
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="📄 Q&A Chatbot PDF")
st.title("GenAI Chatbot from PDF document by nhuttran")

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    process_pdf_and_save_to_vectorstore("temp.pdf")
    st.session_state.chain = get_rag_chain()
    st.success("✅ Document processed successfully")

query = st.text_input("💬 Enter question:")

if query and st.session_state.chain:
    response = st.session_state.chain.invoke(query)
    st.write("🧠 Answer:")
    st.write(response)