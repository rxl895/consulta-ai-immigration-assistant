# 🧠 Consulta AI Immigration Assistant

A working prototype of an AI-powered assistant that answers U.S. immigration-related queries using **Retrieval-Augmented Generation (RAG)**. Built with `LangChain`, `FAISS`, and `Streamlit`, the assistant helps users get accurate, grounded responses from official sources like USCIS.

> ⚡ Designed in alignment with Consulta's mission to merge law, automation, and AI for impactful immigration solutions.

---

## 🚀 Features

- 🔍 **Query-Answering**: Ask natural language questions about immigration
- 📚 **Knowledge Base**: Uses USCIS public content (scraped or uploaded)
- 🧠 **RAG Architecture**: Retrieval + LLM generation pipeline
- 🛠️ **LangChain + FAISS**: Efficient vector search and retrieval
- 🌐 **Streamlit Interface**: Interactive and deployable UI

---

## 📸 Demo (Optional)
> *(Insert screenshots or a Loom demo link here if available)*

---

## 🛠️ Tech Stack

| Component         | Tool/Library                      |
|------------------|-----------------------------------|
| LLM              | HuggingFaceHub (`zephyr-7b-beta`) |
| Retrieval        | FAISS vector DB                   |
| Framework        | LangChain                         |
| UI               | Streamlit                         |
| Web Scraping     | BeautifulSoup                     |
| Deployment-ready?| ✅ Yes (Replit, HuggingFace, etc.)|

---

## 📦 Project Structure

```bash
consulta-ai-immigration-assistant/
├── data/                        # Stores scraped content & FAISS index
│   ├── uscis_content.txt        # Raw USCIS content
│   └── uscis_faiss_index/       # Saved FAISS vector database
├── docs/                        # Assets for README/demo
│   └── README_assets/           # Screenshots or GIFs
├── src/
│   ├── app.py                   # Streamlit frontend & QA pipeline
│   ├── ingest.py                # Web scraper for USCIS site
│   └── embed.py                 # Embeds scraped data into FAISS index
├── .env                         # Hugging Face / OpenAI API keys (not committed)
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
