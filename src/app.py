import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official USCIS sources.")

# Sidebar: LLM model selector
model_choice = st.sidebar.selectbox(
    "Choose LLM backend:",
    [
        "google/flan-t5-base",  # ✅ Free
        "google/flan-t5-small",  # ✅ Free
        "HuggingFaceH4/zephyr-7b-beta",  # ⚠️ Paid
        "mistralai/Mistral-7B-Instruct-v0.1"  # ⚠️ Paid
    ],
    index=0  # default to free model
)
st.sidebar.markdown(f"🔍 Using model: `{model_choice}`")

# Load FAISS retriever
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/uscis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Load LLM with fallback
def load_llm(repo_id):
    try:
        return HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.5, "max_new_tokens": 200}
        )
    except Exception as e:
        st.warning(f"⚠️ Failed to load `{repo_id}`. Falling back to `google/flan-t5-base`.")
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5, "max_new_tokens": 200}
        )

llm = load_llm(model_choice)

# QA chain with source doc support
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# Main input
query = st.text_input("❓ Your question:")
if query:
    with st.spinner("Generating response..."):
        try:
            result = qa_chain(query)
            answer = result.get("answer", "Sorry, I couldn't find an answer.")
            sources = result.get("sources", "")

            st.markdown("### 🤖 Answer")
            st.success(answer)

            if sources:
                st.markdown("### 📄 Sources used:")
                for idx, chunk in enumerate(sources.split(","), start=1):
                    st.markdown(f"**Chunk {idx}:** {chunk.strip()}")

            st.caption(f"🧠 Powered by: `{model_choice}`")

            # Feedback
            st.markdown("---")
            st.markdown("🐵 **Was this helpful?**")
            col1, col2 = st.columns(2)
            with col1:
                st.button("👍 Yes")
            with col2:
                st.button("👎 No")

        except Exception as e:
            st.error("⚠️ Model failed to generate a response. Try switching to a different one.")
            st.exception(e)
