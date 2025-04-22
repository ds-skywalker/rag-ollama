# AI Assistant

> **Key Feature:** This project enables you to launch Retrieval-Augmented Generation (RAG) locally on your own computer using [Ollama](https://ollama.com/) for both language models and embeddings. No cloud APIs required.

AI Assistant is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions and receive answers based on the content of a collection of PDF documents. It leverages modern language models and vector search to provide accurate, context-aware responses.

## Features

- **Document Ingestion:** Loads and processes PDF documents from a specified directory.
- **Vector Store:** Builds a FAISS-based vector store for efficient similarity search over document chunks.
- **Retrieval-Augmented Generation:** Uses retrieved document context to generate concise, relevant answers.
- **Streamlit UI:** User-friendly web interface for asking questions and viewing relevant source documents.
- **Docker Support:** Easily deployable with Docker and Docker Compose.

## Project Structure

```
.
├── app/
│   ├── main.py           # Streamlit app entry point
│   ├── rag_utils.py      # RAG logic
│   └── config.py         # Configuration and environment loading
├── docs/                 # Directory for your PDF documents (you can modify it in the .env file)
├── vector_store/         # Directory for the FAISS vector store, will be created automatically (.env file)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
├── compose.yaml          # Docker Compose setup
├── .env                  # Environment variables
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- PDF documents placed in the `docs/` directory
- Documents metadata file docs_metadata.csv (you can use other name, need to modify .env file)
  also placed in the `docs/` directory. This is the simple csv file with 2 columns "file name" and "link".
  Where the "file name" - name of the pdf file with some information and "link" - link to the webpage with
  this information. If you have not such links, you can put some dummy links.
  For example:
  "file name", "link" 
  llm.pdf,     https://aws.amazon.com/what-is/large-language-model/  
  rag.pdf,     https://aws.amazon.com/what-is/retrieval-augmented-generation/

### Setup

1. **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rag_ollama
    ```

2. **Configure environment variables:**
    - Edit `.env` to set document and vector store directories if needed.

3. **Build and run with Docker Compose:**
    ```bash
    docker compose up --build
    ```

4. **Pull models**
    ```bash
    ollama pull bge-m3
    ollama pull qwen2.5:7b
    ```    

5. **Access the app:**
    - Open [http://localhost:8501](http://localhost:8501) in your browser.

### Usage

- Click **"Create Documents Store"** to build the vector store from your PDFs (only needed once or when documents change).
- Enter your question in the text area and click **"Submit"**.
- The answer and relevant source documents will be displayed.

## Configuration

- All main settings (model names, directories, chunk size, etc.) are managed in `app/config.py` and via environment variables in `.env`.

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://faiss.ai/)
- [Ollama](https://ollama.com/) (for LLM and embeddings)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

## License

This project is provided for educational and research purposes.

---
