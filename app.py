import os
import streamlit as st
from pypdf import PdfReader

# Text splitting from your original RAG notebook
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + Vector DB from your original RAG notebook
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Cloud Free LLM Engine
from langchain_groq import ChatGroq

# Modern LangChain chain components (Updated for v1.0+)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG PDF Assistant", layout="wide")
st.title("📄 PDF Chatbot (100% Free & Online)")

# Safe secret loading fallback to prevent Streamlit Secret missing crashes
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# 1. Initialize your exact HuggingFace embedding model from the notebook
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 2. Initialize the Free Cloud LLM via Groq
@st.cache_resource
def load_llm(api_key):
    if api_key:
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.3)
    return None

# Sidebar: Document Ingestion UI & API key backup input
with st.sidebar:
    st.header("Setup & Upload")
    
    # If the secret isn't configured, provide an on-screen password box input
    if not GROQ_API_KEY:
        input_key = st.text_input("Enter Groq API Key", type="password", help="Get a free key from console.groq.com")
        if input_key:
            GROQ_API_KEY = input_key
            st.success("API key registered!")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None and "vector_store" not in st.session_state:
        with st.spinner("Processing PDF and generating embeddings..."):
            pdf_reader = PdfReader(uploaded_file)
            transcript = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    transcript += text + "\n"
            
            if not transcript.strip():
                st.error("Could not extract any text from this PDF file.")
            else:
                # Exact Splitter strategy from your original notebook
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.create_documents([transcript])
                
                # Build exact FAISS index from your notebook
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state["vector_store"] = vector_store
                st.success(f"Indexed {len(chunks)} chunks successfully!")

# Chat Window Setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! Please enter your API Key (if not in settings) and upload a PDF to start chatting."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask a question about your file..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    if not GROQ_API_KEY:
        st.chat_message("assistant").write("❌ Please enter your Groq API Key in the sidebar first.")
    elif "vector_store" not in st.session_state:
        st.chat_message("assistant").write("❌ Please upload a file to the sidebar first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Dynamically instantiate LLM with the provided key
                    llm = load_llm(GROQ_API_KEY)
                    
                    retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 7})
                    
                    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know.\n\n"
                        "Context:\n{context}"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error connecting to cloud engine: {str(e)}")
