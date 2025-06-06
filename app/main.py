
"""
Main Streamlit application for an AI Assistant using Retrieval-Augmented Generation (RAG).

This module initializes the Streamlit UI, manages user interactions, and connects to the RAG backend
for document retrieval and question answering. It provides functionality to create a vector store from
documents, check its existence, and process user queries using the RAG pipeline.

Functions:
    create_logger(): Initializes and caches a logger for the application.
    create_rag(): Initializes and caches the RAG object with configuration parameters.
    check_vector_store(): Checks if the vector store exists and displays a warning if not.

UI Components:
    - Title: "AI Assistant"
    - Button: "Create Documents Store" to build the vector store from documents.
    - Text area: For user to input questions.
    - Button: "Submit" to process the question and display the answer.
    - Text area: Displays the RAG-generated answer.
    - Markdown: Displays relevant documents retrieved for the answer.

Session State:
    - "rag_answer": Stores the answer generated by the RAG pipeline.
    - "relevant_docs": Stores the list of relevant documents for display.

Exceptions are logged and displayed to the user as error messages.
"""

import logging
import streamlit as st

from rag_utils import RAG
from config import (
    LLM_MODEL_NAME,
    EMBEDDINGS_MODEL_NAME,
    OLLAMA_MODELS_BASE_URL,
    DOCS_DIR,
    DOCS_METADATA_FILE_PATH,
    VECTOR_STORE_DIR,
    PROMPT_TEMPLATE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIVIER_TOP_K,
)

# Set up logger
@st.cache_resource
def create_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    return logger

@st.cache_resource
def create_rag():
    return RAG(
        prompt_template=PROMPT_TEMPLATE,
        llm_name=LLM_MODEL_NAME,
        embeddings_model_name=EMBEDDINGS_MODEL_NAME,
        models_base_url=OLLAMA_MODELS_BASE_URL,
        docs_dir=DOCS_DIR,
        docs_metadata_file_path=DOCS_METADATA_FILE_PATH,
        vector_store_dir=VECTOR_STORE_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        retriever_top_k=RETRIVIER_TOP_K
    )

def check_vector_store():
    exists = rag.is_vector_store_exists()
    if not exists:
        st.warning("Documents Store does not exist. Please create it first.")
    return exists

# Initialize Streamlit app
st.title("AI Assistant")

logger = create_logger()

rag = create_rag()
check_vector_store()

# Submit button
if st.button("Create Documents Store"):
    try:
        st.spinner("Creating vector store...")
        rag.create_vector_store()      
        rag.init_rag_chain()
        st.success("Documents Store created successfully!")
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        st.error(f"Failed to create Documents Store.")

if "rag_answer" not in st.session_state:
    st.session_state["rag_answer"] = ""

if "relevant_docs" not in st.session_state:
    st.session_state["relevant_docs"] = ""

# Input field for user question
user_question = st.text_area("Enter your question:", "", height=100)

# Submit button
if st.button("Submit") and check_vector_store():
    if user_question.strip():
        # Process the question using RAG chain
        response = rag.invoke(user_question)
        st.session_state["rag_answer"] = response

        # Retrieve relevant documents
        docs = rag.found_docs
        docs_display = "\n".join(
            [f"{doc_num+1}. {doc['file_path']}\n{doc['source']}\n" for doc_num, doc in enumerate(docs)]
        )
        st.session_state["relevant_docs"] = docs_display
    else:
        st.warning("Please enter a question.")

if not user_question.strip():
    st.session_state["rag_answer"] = ""

# Text areas for RAG answer and relevant documents
rag_answer = st.text_area(
    label="Answer", 
    value=st.session_state.get("rag_answer", ""), 
    height=150
)

st.text("Relevant Documents")
st.markdown(st.session_state.get("relevant_docs", ""), unsafe_allow_html=True)
