import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official USCIS sources.")

# Sidebar: Model selection
model_choice = st.sidebar.selectbox(
    "Choose LLM backend:",
    [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "microsoft/phi-2"
    ]
)
st.sidebar.markdown(f"🔍 Using model: `{model_choice}`")

# Load FAISS retriever with cached embedding
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/uscis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Load selected model
llm = HuggingFaceHub(
    repo_id=model_choice,
    model_kwargs={"temperature": 0.5, "max_new_tokens": 200}
)

# QA chain setup
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# User input
query = st.text_input("❓ Your question:")
if query:
    with st.spinner("Generating response..."):
        try:
            response = qa_chain.run(query)
            st.success(response)
            st.caption(f"🧠 Powered by: `{model_choice}`")
        except Exception as e:
            st.error("⚠️ The selected model failed to generate a response. Try switching models.")
            st.exception(e)
