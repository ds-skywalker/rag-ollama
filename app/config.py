"""
Configuration module for the application.

This module loads environment variables and defines constants used throughout the app,
including directory paths, model names, API endpoints, prompt templates, and chunking parameters.
"""

import os

DOCS_DIR = os.environ.get("DOCS_DIR")
DOCS_METADATA_FILE_NAME = os.environ.get("DOCS_METADATA_FILE_NAME")
DOCS_METADATA_FILE_PATH = os.path.join(DOCS_DIR, DOCS_METADATA_FILE_NAME)
VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR")

LLM_MODEL_NAME = "qwen2.5:7b"
EMBEDDINGS_MODEL_NAME = "bge-m3"
OLLAMA_MODELS_BASE_URL = "http://ollama:11434"

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIVIER_TOP_K = 5
