import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Immigration AI Assistant", layout="centered")
st.title("🧠 Immigration AI Assistant 🇺🇸")
st.markdown("Ask any U.S. immigration question and get an AI-generated response, powered by official USCIS sources.")

# Sidebar model selection
model_choice = st.sidebar.selectbox(
    "Choose LLM backend:",
    [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "microsoft/phi-2"
    ]
)
st.sidebar.markdown(f"🔍 Using model: `{model_choice}`")

# Load retriever
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/uscis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Custom prompt for better answers
custom_prompt = PromptTemplate(
    template="""You are an AI assistant. Answer the question using ONLY the provided context. 
If the answer cannot be found, say "I don't know".

Question: {question}

Context:
{context}

Answer:""",
    input_variables=["context", "question"],
)

qa_chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=custom_prompt)

# Load model
llm = HuggingFaceHub(
    repo_id=model_choice,
    model_kwargs={"temperature": 0.5, "max_new_tokens": 200}
)

# Chain with custom prompt
qa_chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=custom_prompt)

# Input box
query = st.text_input("❓ Your question:")
if query:
    with st.spinner("Generating response..."):
        try:
            docs = retriever.get_relevant_documents(query)
            result = qa_chain({"input_documents": docs, "question": query})

            st.markdown("### 🤖 Answer")
            st.success(result["answer"])

            # Show sources
            if result.get("sources"):
                st.markdown("### 📄 Sources used:")
                for idx, chunk in enumerate(result["sources"].split(","), start=1):
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
