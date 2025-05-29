import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma
import chromadb

def process_pdf_and_save_to_vectorstore(pdf_path):
    db_path = "db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_path)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=client,
        collection_name="pdf_docs"
    )

    return vectordb
