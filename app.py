
import streamlit as st
from utils import process_pdf_and_save_to_vectorstore
from rag_chain import get_rag_chain
from dotenv import load_dotenv
import os
import time

def clear_chroma_locks():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
    lock_file = os.path.join(db_path, "chroma.sqlite.lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except:
            pass

load_dotenv()
clear_chroma_locks()

# Configure page
st.set_page_config(page_title="📄 Q&A Chatbot PDF")
st.title("GenAI Chatbot from PDF document by nhuttran")

def ensure_db_permissions(db_path):
    """Ensure proper permissions on database directory"""
    try:
        os.makedirs(db_path, exist_ok=True)
        os.chmod(db_path, 0o777)  # Wide permissions for debugging
        return True
    except Exception as e:
        st.error(f"Permission error on DB directory: {str(e)}")
        return False

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    temp_file_path = "temp.pdf"
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")

    if not ensure_db_permissions(db_path):
        st.stop()

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Clear any existing ChromaDB locks
        for _ in range(3):  # Retry mechanism
            try:
                process_pdf_and_save_to_vectorstore(temp_file_path)
                st.session_state.chain = get_rag_chain()
                st.success("✅ Document processed successfully")
                break
            except Exception as e:
                if "readonly database" in str(e):
                    time.sleep(1)  # Wait before retry
                    continue
                raise

    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                st.warning(f"⚠️ Could not delete temporary file: {str(e)}")

query = st.text_input("💬 Enter question:")

if query and st.session_state.chain:
    try:
        response = st.session_state.chain.invoke({"query": query})  # Note the dictionary input
        st.write("🧠 Answer:")
        st.write(response["result"])  # Access the result key
    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")
