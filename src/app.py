import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official sources like USCIS.")

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Missing OpenAI API key. Please set `OPENAI_API_KEY` in your .env file.")
    st.stop()

# Load FAISS index and embeddings
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        "data/uscis_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # <-- Important fix
    )
    retriever = db.as_retriever()
    return retriever

retriever = load_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Input field
query = st.text_input("❓ Your question:")
if query:
    with st.spinner("Generating response..."):
        response = qa_chain.run(query)
        st.success(response)
