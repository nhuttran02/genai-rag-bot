
# 🧠 GenAI RAG Bot from PDF using LangChain + Streamlit

This is an AI-powered chatbot that performs **Retrieval-Augmented Generation (RAG)** from uploaded PDF documents. The project is built using:

- 🧱 LangChain
- 📄 ChromaDB (Vector Store)
- 🤗 HuggingFace Embeddings
- 🗣️ OpenRouter (LLM API)
- 🌐 Streamlit (Frontend UI)
- 📦 Deployed on VPS with Nginx + Domain Integration

---

## 🚀 Features

- 📥 Upload any PDF document for processing
- 🔍 Extracts, splits, and **embeds PDF content** into a vector database
- 💬 Interact with the document via natural language questions
- 🧩 Supports free LLMs via OpenRouter (e.g., `deepseek`, `devin`, `mistral`, ...)
- 🌐 Accessible via subdomain: `https://ragbot.nhuttran.id.vn`

---

## 📁 Project Structure

```
genai-rag-bot/
├── app.py               # Main Streamlit interface
├── rag_chain.py         # Initializes the chain (LLM + retriever)
├── utils.py             # PDF processing and vectorstore storage
├── requirements.txt     # Required dependencies
└── .env                 # Environment variables (API key, etc.)
```

---

## ⚙️ Manual Setup

```bash
git clone https://github.com/yourusername/genai-rag-bot.git
cd genai-rag-bot

python3 -m venv rag_env
source rag_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Create a .env file with the following content:
# OPENAI_API_KEY=your_openrouter_api_key

streamlit run app.py
```

---


## 📌 Technical Notes

- **LLM**: Set via OpenRouter API in `rag_chain.py`
- **Embeddings**: Uses HuggingFace `"all-MiniLM-L6-v2"` (no OpenAI cost)
- **Vector Store**: ChromaDB v0.4.13 (compatible with LangChain)
- **UI**: Built with Streamlit for simplicity and rapid testing
- **Hosting**: VPS Ubuntu 20.04+ with Docker/Nginx (optional)

---

## 📚 References

- [LangChain Documentation](https://docs.langchain.com)
- [OpenRouter API](https://openrouter.ai/docs)
- [Streamlit Docs](https://docs.streamlit.io)
- [ChromaDB](https://docs.trychroma.com/)

---

## 👤 Author

**Tran Hong Nhut**  
AI Engineer  
📧 nhuttran230902@gmail.com  

---

> This project was built to showcase practical experience deploying a GenAI chatbot using open-source and free-tier AI technologies.
