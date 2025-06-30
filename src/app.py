import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official USCIS sources.")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Choose LLM backend:",
    [
        "google/flan-t5-base",  # ✅ FREE
        "google/flan-t5-small",  # ✅ FREE
        "mistralai/Mistral-7B-Instruct-v0.1"  # ⚠️ Might also require PRO
    ]
)

st.sidebar.markdown(f"🔍 Using model: `{model_choice}`")

# Load retriever with cache
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/uscis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Load selected LLM
llm = HuggingFaceHub(
    repo_id=model_choice,
    model_kwargs={"temperature": 0.5, "max_new_tokens": 200}
)

# Custom prompt compatible with 'summaries' input
custom_prompt = PromptTemplate(
    template="""
You are an AI assistant. Answer the question using ONLY the provided summaries.
If the answer cannot be found in the summaries, say "I don't know".

Question: {question}

Summaries:
{summaries}

Answer:
""",
    input_variables=["summaries", "question"]
)

# Setup QA chain
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# User input
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

            # Feedback section
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
