import os
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
import chromadb

def get_rag_chain(collection_id):
    db_path = os.path.join("db", collection_id)
    if not os.path.exists(db_path):
        raise FileNotFoundError("Vector DB not found. Upload PDF first.")

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=db_path)

    vectordb = Chroma(
        client=client,
        collection_name="pdf_docs",
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )