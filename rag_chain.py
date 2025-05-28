import os
from langchain_community.vertorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_rag_chain():
    llm = ChatOpenAI(
        model = "gryphe/mythomist-7b",
        openai_api_base = "https://openrouter.ai/api/v1",
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        temperature = 0.3
    )

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    vectordb = Chroma(persist_directory = "db", embedding_function = embedding)
    retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs ={"k": 3})
    llm = ChatOpenAI(temperature=0.2)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

