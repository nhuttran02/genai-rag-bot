from langchain.vertorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings

def get_rag_chain():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0.2)

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return rag_chain

