# import os
# import shutil
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
# from langchain_community.vectorstores import Chroma
# import chromadb

# def process_pdf_and_save_to_vectorstore(pdf_path):
#     db_path = "db"
#     if os.path.exists(db_path):
#         shutil.rmtree(db_path)

#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     splits = splitter.split_documents(documents)

#     embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     client = chromadb.PersistentClient(path=db_path)

#     vectordb = Chroma.from_documents(
#         documents=splits,
#         embedding=embedding,
#         client=client,
#         collection_name="pdf_docs"
#     )

#     return vectordb

import os
import shutil
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

def process_pdf_and_save_to_vectorstore(pdf_path):
    base_db_path = "db"
    collection_id = str(uuid.uuid4())
    db_path = os.path.join(base_db_path, collection_id)
    os.makedirs(db_path, exist_ok=True)

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

    # Clean up: keep only 5 newest collections
    all_dirs = sorted(
        [d for d in os.listdir(base_db_path) if os.path.isdir(os.path.join(base_db_path, d))],
        key=lambda d: os.path.getctime(os.path.join(base_db_path, d))
    )
    for old in all_dirs[:-5]:
        shutil.rmtree(os.path.join(base_db_path, old), ignore_errors=True)

    return collection_id