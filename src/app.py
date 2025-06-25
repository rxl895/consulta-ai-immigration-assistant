import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables (.env with HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official USCIS sources.")

# Load FAISS index and embeddings
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/uscis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return retriever

retriever = load_retriever()

# Load the Hugging Face model for answering questions
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",  # You can replace this with other small models
    model_kwargs={"temperature": 0.5, "max_length": 200}
)

# Create the QA chain using retrieval
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Input box for user's question
query = st.text_input("❓ Your question:")
if query:
    with st.spinner("Generating response..."):
        response = qa_chain.run(query)
        st.success(response)
