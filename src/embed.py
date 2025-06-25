# src/embed.py

import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load text from file
with open("data/uscis_content.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 1: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.create_documents([raw_text])

# Step 2: Use FREE HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create FAISS index
db = FAISS.from_documents(docs, embeddings)

# Step 4: Save vector DB locally
db.save_local("data/uscis_faiss_index")

print("✅ Text embedded using HuggingFace and saved to data/uscis_faiss_index")
