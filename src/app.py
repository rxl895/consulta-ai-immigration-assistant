import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load Hugging Face token from .env
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official USCIS sources.")

# Load FAISS vector store and embeddings
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/uscis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return retriever

retriever = load_retriever()

# Load Hugging Face model
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 200}
)

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Input field
query = st.text_input("❓ Your question:")

if query:
    with st.spinner("Generating response..."):
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)

        # Generate response using the chain
        response = qa_chain.combine_documents_chain.run(input_documents=docs, question=query)
        st.success(response)

        # Show source documents
        st.markdown("#### 📄 Sources used:")
        for i, doc in enumerate(docs):
            content_preview = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"**Chunk {i+1}:** {content_preview[:300]}{'...' if len(content_preview) > 300 else ''}")

        # Feedback section
        st.markdown("#### 🙋 Was this helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", key="yes_feedback"):
                st.success("Thanks for your feedback!")
                st.session_state.setdefault("feedback_log", []).append({"query": query, "feedback": "yes"})
        with col2:
            if st.button("👎 No", key="no_feedback"):
                st.warning("Thanks — your feedback helps us improve.")
                st.session_state.setdefault("feedback_log", []).append({"query": query, "feedback": "no"})
