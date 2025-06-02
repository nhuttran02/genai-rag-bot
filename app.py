import streamlit as st
from utils import process_pdf_and_save_to_vectorstore
from rag_chain import get_rag_chain
from dotenv import load_dotenv
import os
import time
import uuid
import fitz

TEXTS = {
    "en": {
        "title": "🤖 AI PDF Assistant by nhuttran",
        "upload_header": "📄 Upload Your PDF",
        "upload_label": "Choose a PDF file",
        "invalid_pdf": "❌ The uploaded file is not a valid PDF.",
        "file_info": "**File:** `{}` — {:.2f} MB",
        "processing": "🧠 Analyzing and embedding document...",
        "done": "✅ Document successfully processed. You can start chatting!",
        "question_placeholder": "e.g., What is this document about?",
        "ask_button": "🚀 Ask",
        "clear_button": "🗑️ Clear",
        "answer_header": "### 🤖 Answer:",
        "source_ref": "📚 Source References",
        "conversation": "### 📜 Conversation History",
        "status_ready": "✅ Ready",
        "status_wait": "⏳ Waiting for document",
        "instructions": "1. Upload PDF\n2. Wait for processing\n3. Ask questions\n4. Get smart answers",
        "model_info": "LLM: DeepSeek R1\nEmbedding: MiniLM-L6-v2\nVector DB: ChromaDB",
        "developer": "Developed by: nhuttran\nVersion: 1.0.0\nTech: RAG + Streamlit",
        "start_prompt": "📄 Please upload a PDF file to start chatting."
    },
    "vi": {
        "title": "🤖 Trợ lý PDF AI by nhuttran",
        "upload_header": "📄 Tải lên tài liệu PDF",
        "upload_label": "Chọn tệp PDF",
        "invalid_pdf": "❌ Tệp đã tải lên không phải là PDF hợp lệ.",
        "file_info": "**Tệp:** `{}` — {:.2f} MB",
        "processing": "🧠 Đang phân tích và embedding tài liệu...",
        "done": "✅ Tài liệu đã được xử lý. Bạn có thể bắt đầu trò chuyện!",
        "question_placeholder": "VD: Tài liệu này nói về điều gì?",
        "ask_button": "🚀 Hỏi",
        "clear_button": "🗑️ Xóa",
        "answer_header": "### 🤖 Trả lời:",
        "source_ref": "📚 Nguồn tham khảo",
        "conversation": "### 📜 Lịch sử trò chuyện",
        "status_ready": "✅ Sẵn sàng",
        "status_wait": "⏳ Chờ tài liệu",
        "instructions": "1. Tải lên PDF\n2. Chờ xử lý\n3. Đặt câu hỏi\n4. Nhận câu trả lời thông minh",
        "model_info": "LLM: DeepSeek R1\nEmbedding: MiniLM-L6-v2\nVector DB: ChromaDB",
        "developer": "Phát triển bởi: nhuttran\nPhiên bản: 1.0.0\nCông nghệ: RAG + Streamlit",
        "start_prompt": "📄 Vui lòng tải lên file PDF để bắt đầu trò chuyện."
    }
}

load_dotenv()

if "language" not in st.session_state:
    st.session_state.language = "en"
if "chain" not in st.session_state:
    st.session_state.chain = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "collection_id" not in st.session_state:
    st.session_state.collection_id = None

st.set_page_config(page_title="AI PDF Assistant", page_icon="📄", layout="wide")

# Restore previous session if available
if st.session_state.collection_id and not st.session_state.chain:
    try:
        st.session_state.chain = get_rag_chain(st.session_state.collection_id)
    except Exception as e:
        st.warning("⚠️ Unable to restore previous document session.")

with st.sidebar:
    st.selectbox("🌐 Language / Ngôn ngữ", options=["en", "vi"], index=0 if st.session_state.language == "en" else 1, key="language")
    st.markdown("### 📊 System Info")
    status = TEXTS[st.session_state.language]["status_ready"] if st.session_state.chain else TEXTS[st.session_state.language]["status_wait"]
    st.success(f"Status: {status}")
    st.markdown("### 📋 Instructions")
    st.info(TEXTS[st.session_state.language]["instructions"])
    st.markdown("### 🧠 Model Info")
    st.code(TEXTS[st.session_state.language]["model_info"])
    st.markdown("### 👨‍💼 Developer")
    st.markdown(TEXTS[st.session_state.language]["developer"])

st.title(TEXTS[st.session_state.language]["title"])
st.header(TEXTS[st.session_state.language]["upload_header"])

uploaded_file = st.file_uploader(TEXTS[st.session_state.language]["upload_label"], type="pdf")
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_size = len(file_bytes) / (1024 * 1024)
    temp_path = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    st.markdown(TEXTS[st.session_state.language]["file_info"].format(uploaded_file.name, file_size))
    try:
        fitz.open(temp_path).close()
        with st.spinner(TEXTS[st.session_state.language]["processing"]):
            collection_id = process_pdf_and_save_to_vectorstore(temp_path)
            st.session_state.collection_id = collection_id
            st.session_state.chain = get_rag_chain(collection_id)
            st.success(TEXTS[st.session_state.language]["done"])
    except Exception as e:
        st.error(f"❌ {str(e)}")
    finally:
        os.remove(temp_path)

if st.session_state.chain:
    st.header("💬 " + TEXTS[st.session_state.language]["title"])
    query = st.text_input("🔍", placeholder=TEXTS[st.session_state.language]["question_placeholder"])
    col1, col2 = st.columns([4, 1])
    ask = col1.button(TEXTS[st.session_state.language]["ask_button"])
    clear = col2.button(TEXTS[st.session_state.language]["clear_button"])
    if clear:
        st.session_state.conversation_history.clear()
        st.rerun()

    if ask and query.strip():
        with st.spinner("🧠 Thinking..."):
            try:
                response = st.session_state.chain.invoke({"query": query})
                answer = response.get("result", "No answer.")
                st.session_state.conversation_history.append((query, answer))
            except Exception as e:
                st.error(f"❌ {str(e)}")

    if st.session_state.conversation_history:
        st.markdown(TEXTS[st.session_state.language]["conversation"])
        for q, a in reversed(st.session_state.conversation_history):
            with st.expander(f"❓ {q}"):
                st.markdown(f"**{TEXTS[st.session_state.language]['answer_header']}**\n\n{a}")

    if 'response' in locals() and response.get("source_documents"):
        with st.expander(TEXTS[st.session_state.language]["source_ref"]):
            for i, doc in enumerate(response["source_documents"][:3]):
                page = doc.metadata.get("page", "Unknown")
                st.info(f"**{i+1}. Page {page}:**\n{doc.page_content[:300]}...")
else:
    st.info(TEXTS[st.session_state.language]["start_prompt"])

st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#888;padding:10px;">
    🤖 <strong>{TEXTS[st.session_state.language]["title"]}</strong><br>
    Powered by DeepSeek, ChromaDB, and Streamlit
</div>
""", unsafe_allow_html=True)
