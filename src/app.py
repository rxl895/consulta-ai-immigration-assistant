# src/app.py

import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

import os

# Load API key
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official sources.")

# Load FAISS index
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("data/uscis_faiss_index", embeddings)
    retriever = db.as_retriever()
    return retriever

retriever = load_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Input
query = st.text_input("❓ Your question:")
if query:
    with st.spinner("Generating response..."):
        response = qa_chain.run(query)
        st.success(response)
