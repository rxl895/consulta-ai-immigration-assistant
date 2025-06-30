# src/embed.py

import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables (if needed for embeddings or future use)
load_dotenv()

# Step 1: Load USCIS content from the text file
with open("data/uscis_content.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 2: Split text into chunks using LangChain's RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(raw_text)

# Step 3: Add 'source' metadata to each chunk for traceability in responses
docs = [
    Document(page_content=chunk, metadata={"source": f"chunk-{i+1}"})
    for i, chunk in enumerate(chunks)
]

# Step 4: Load Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create a FAISS vector index from the document chunks
db = FAISS.from_documents(docs, embeddings)

# Step 6: Save the FAISS index locally
db.save_local("data/uscis_faiss_index")

print("✅ Text embedded and saved to data/uscis_faiss_index with source metadata.")
