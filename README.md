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

## 🔧 Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/rxl895/consulta-ai-immigration-assistant.git
cd consulta-ai-immigration-assistant
2. Install Python Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Set Environment Variables
Create a .env file:

bash
Copy
Edit
touch .env
Paste your Hugging Face API token inside:

env
Copy
Edit
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
🔐 Never commit this file. It’s already included in .gitignore.

4. Run the Full Pipeline
Step 1: Scrape USCIS Content
bash
Copy
Edit
python3 src/ingest.py
Step 2: Embed the Content with Sentence Transformers
bash
Copy
Edit
python3 src/embed.py
Step 3: Run the Chatbot Interface
bash
Copy
Edit
streamlit run src/app.py
Visit the generated localhost link to chat with your AI Assistant.

## 💬 Example Questions
“How to apply for an H1B visa?”

“What documents do I need for OPT?”

“What is I-130 used for?”

“How to change visa status inside the U.S.?”

## 🧠 Powered By
LangChain for chaining and context injection

FAISS for similarity search over USCIS content

HuggingFaceHub for hosted open LLMs (e.g. Zephyr)

Streamlit for interactive front-end

BeautifulSoup for scraping USCIS.gov content

## 📌 Future Improvements
 Add citation links for retrieved USCIS content

 Multilingual support (Spanish, Hindi)

 Add user feedback for answer helpfulness

 Public deployment on HuggingFace Spaces

 Add token usage monitoring and cost display

## ✨ Credits
Created with ❤️ by Ritika Lamba
Inspired by Consulta’s mission to revolutionize immigration 

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

