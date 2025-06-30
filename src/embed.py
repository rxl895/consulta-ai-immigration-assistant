import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Load the USCIS content from file
with open("data/uscis_content.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_text(raw_text)

# Step 3: Add source metadata to each chunk
docs = [
    Document(page_content=chunk, metadata={"source": f"chunk-{i+1}"})
    for i, chunk in enumerate(texts)
]

# Step 4: Use HuggingFace embeddings (free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create FAISS index
db = FAISS.from_documents(docs, embeddings)

# Step 6: Save index to disk
db.save_local("data/uscis_faiss_index")

print("✅ Text embedded and saved to data/uscis_faiss_index with source metadata.")
