"""
This module provides a Retrieval Augmented Generation (RAG) system that integrates
document processing, vector search, and language model query generation to answer
user questions based on a set of PDF documents. The module leverages several key components:

    • Document Loading: Uses PyMuPDFLoader to load and extract text from PDF files found in
      a specified directory.

    • Text Splitting: Applies RecursiveCharacterTextSplitter to break documents into manageable
      chunks based on configurable chunk size and overlap.

    • Vector Store Management: Utilizes FAISS to create, load, and save a vector store.
      The vector store is built using embeddings generated via the OllamaEmbeddings model,
      facilitating efficient similarity searches.

    • Retrieval Augmented Generation (RAG) Chain: Constructs a processing chain that:
          - Retrieves relevant text chunks from the vector store using similarity search.
          - Consolidates retrieved texts, deduplicates document metadata, and formats context.
          - Prompts a ChatLLM (ChatOllama) to generate answers based on the retrieved information.
          - Parses and outputs the model's answer appropriately.
"""

import os
import logging

import pandas as pd
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAG:
    """
    RAG (Retrieval-Augmented Generation) class for managing document ingestion, vector store creation, 
    and retrieval-augmented question answering using language models and embeddings.
    Attributes:
        _prompt_template (str): Template for the prompt used in the RAG chain.
        _llm_name (str): Name of the language model to use.
        _embeddings_model_name (str): Name of the embeddings model to use.
        _models_base_url (str): Base URL for the models.
        _docs_dir (str): Directory containing the documents (PDFs).
        _docs_metadata_file_path (str): Path to the metadata CSV file for documents.
        _vector_store_dir (str): Directory to store the vector database.
        _chunk_size (int): Size of text chunks for splitting documents.
        _chunk_overlap (int): Overlap between text chunks.
        _retriever_top_k (int): Number of top documents to retrieve.
        _vector_store_db_path (str): Path to the vector store database.
        _llm: Language model instance.
        _embeddings_model: Embeddings model instance.
        _vector_store: Vector store instance.
        _rag_chain: RAG chain instance.
        found_docs (list): List of found documents in the last retrieval.
    """
    def __init__(
        self,
        prompt_template,
        llm_name,
        embeddings_model_name,
        models_base_url,
        docs_dir,
        docs_metadata_file_path,
        vector_store_dir,
        chunk_size,
        chunk_overlap,
        retriever_top_k
    ):
        self._prompt_template = prompt_template
        self._llm_name = llm_name
        self._embeddings_model_name = embeddings_model_name
        self._models_base_url = models_base_url
        self._docs_dir = docs_dir
        self._docs_metadata_file_path = docs_metadata_file_path
        self._vector_store_dir = vector_store_dir
        self.print_info()
        self._vector_store_db_path = os.path.join(
            vector_store_dir, 
            os.path.basename(embeddings_model_name)
        )
        self._chunk_size = chunk_size
        self._retriever_top_k = retriever_top_k
        self._chunk_overlap = chunk_overlap     
        self._llm = ChatOllama(
            model=self._llm_name, 
            base_url=self._models_base_url, 
            temperature=0.0
        )
        self._embeddings_model = OllamaEmbeddings(
            model=self._embeddings_model_name, 
            base_url=self._models_base_url
        )
        self.init_rag_chain()
        self.found_docs = []

    def print_info(self):
        """
        Logs the current configuration details of the instance, including LLM name, embeddings model name,
        models base URL, documents directory, documents metadata file path, and vector store directory.
        """
        logger.info(f"LLM name: {self._llm_name}")
        logger.info(f"Embeddings model name: {self._embeddings_model_name}")
        logger.info(f"Models base URL: {self._models_base_url}")
        logger.info(f"Docs directory: {self._docs_dir}")
        logger.info(f"Docs metadata file path: {self._docs_metadata_file_path}")
        logger.info(f"Vector store directory: {self._vector_store_dir}")

    def invoke(self, user_question):
        """
        Processes the user's question by removing leading and trailing whitespace,
        then invokes the underlying RAG (Retrieval-Augmented Generation) chain with the cleaned question.
        Args:
            user_question (str): The question input provided by the user.
        Returns:
            str: The result returned by the RAG chain after processing the user's question.
        """        
        return self._rag_chain.invoke(user_question.strip())

    def init_rag_chain(self):
        """
        Initializes the RAG (Retrieval-Augmented Generation) chain for the application.
        This method loads the vector store and configures it as a retriever using similarity search
        with a specified number of top results (`self._retriever_top_k`). If the vector store is
        successfully loaded, it creates the RAG chain and assigns it to `self._rag_chain`.
        If the vector store cannot be loaded, `self._rag_chain` is set to None.
        """
        self._vector_store = self._load_vector_store()
        if self._vector_store is not None:
            self._vector_store = self._vector_store.as_retriever(
                search_type='similarity', 
                search_kwargs={'k': self._retriever_top_k}        
            )            
            self._rag_chain = self._create_rag_chain()
        else:
            self._rag_chain = None

    def create_vector_store(self):
        """
        Creates a FAISS-based vector store from PDF documents.

        This method performs the following steps:
            1. Retrieves a list of PDF files.
            2. Loads the PDF files into document objects.
            3. Updates document metadata with source links and file paths.
            4. Splits the documents into text chunks.
            5. Initializes a FAISS vector store with the appropriate embedding dimension.
            6. Adds the text chunks to the vector store.
            7. Saves the vector store to a local path.

        Logging is used to provide information about the number of PDFs, documents, chunks, and documents added to the vector store.

        Raises:
            Any exceptions raised by the underlying PDF loading, embedding, or FAISS operations.
        """
        logger.info("Creating vector store...")
        pdfs = self._get_pdfs_list()
        docs = self._load_pdfs(pdfs)

        docs_links_dict = self._get_docs_links_dict()
        for i in range(len(docs)):
            source = os.path.basename(docs[i].metadata["source"])
            docs[i].metadata["source"] = docs_links_dict[source]
            docs[i].metadata["file_path"] = source

        chunks = self._split_text(docs)
        logger.info(f"Number of PDFs: {len(pdfs)}")
        logger.info(f"Number of documents: {len(docs)}")
        logger.info(f"Number of chunks: {len(chunks)}")

        embed_dimension = len(self._embeddings_model.embed_query("test"))
        index = faiss.IndexFlatL2(embed_dimension)
        self.vector_store = FAISS(
            embedding_function=self._embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        ids = self.vector_store.add_documents(chunks, verbose=True)
        logger.info(f"Number of documents added: {len(ids)}")  
        self.vector_store.save_local(self._vector_store_db_path)
        logger.info("Vector store saved.")

    def is_vector_store_exists(self):
        """
        Checks whether the vector store exists.

        Returns:
            bool: True if the vector store is initialized (not None), otherwise False.
        """
        return True if self._vector_store is not None else False    

    def _update_found_docs(self, docs):
        """
        Updates the list of found documents by filtering out duplicates based on the 'file_path' metadata.

        Args:
            docs (list): A list of document objects, each expected to have a 'metadata' attribute containing 'file_path' and 'source', 
                         and a 'page_content' attribute.

        Returns:
            list: The original list of document objects.
        """
        docs_names = []
        self.found_docs = []
        for doc in docs:
            if doc.metadata["file_path"] not in docs_names:
                self.found_docs.append(
                    {
                        "page_content": doc.page_content, 
                        "file_path": doc.metadata["file_path"],
                        "source": doc.metadata["source"]
                    }
                )
                docs_names.append(doc.metadata["file_path"])
        return docs  

    def _get_context(self, docs):
        """
        Concatenates the page content of a list of document objects into a single string, separated by double newlines.

        Args:
            docs (list): A list of document objects, each expected to have a 'page_content' attribute.

        Returns:
            str: A single string containing the concatenated page contents of all documents, separated by two newline characters.
        """
        return '\n\n'.join([doc.page_content for doc in docs])

    def _create_rag_chain(self):
        """
        Creates and returns a RAG (Retrieval-Augmented Generation) chain for processing input questions.

        The chain performs the following steps:
            1. Retrieves relevant context documents using the vector store and context update methods.
            2. Passes the retrieved context and the input question to a chat prompt template.
            3. Sends the prompt to the language model (LLM) for response generation.
            4. Parses the LLM output into a string format.

        Returns:
            A composed chain object that takes a question as input and outputs a generated response string.
        """
        prompt = ChatPromptTemplate.from_template(self._prompt_template)
        rag_chain = (
            {
                "context": self._vector_store | self._update_found_docs | self._get_context, 
                "question": RunnablePassthrough()
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )
        return rag_chain

    def _get_docs_links_dict(self):
        """
        Reads a CSV file containing document metadata and returns a dictionary mapping file names to their corresponding links.

        Returns:
            dict: A dictionary where the keys are file names (str) and the values are links (str) extracted from the metadata CSV file.
        """
        df = pd.read_csv(self._docs_metadata_file_path)
        docs_links_dicts_list = df[["file name", "link"]].to_dict(orient="records")
        result = {d["file name"]: d["link"] for d in docs_links_dicts_list}
        return result

    def _get_pdfs_list(self):
        """
        Recursively searches the documents directory for PDF files.

        Returns:
            list: A list of absolute file paths to all PDF files found within the documents directory and its subdirectories.
        """
        pdfs = []
        for root, dirs, files in os.walk(self._docs_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdfs.append(os.path.join(root, file))
        return pdfs

    def _load_pdfs(self, pdfs):
        """
        Loads and processes a list of PDF files into document objects.

        Args:
            pdfs (list of str): List of file paths to PDF documents.

        Returns:
            list: A list of document objects extracted from the provided PDFs.
        """
        docs = []
        for pdf in pdfs:
            loader = PyMuPDFLoader(pdf)
            doc = loader.load()
            docs.extend(doc)
        return docs

    def _split_text(self, docs):
        """
        Splits a list of documents into smaller text chunks using the RecursiveCharacterTextSplitter.

        Args:
            docs (List[Document]): A list of Document objects to be split into chunks.

        Returns:
            List[Document]: A list of Document objects, each representing a chunk of the original documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, 
            chunk_overlap=self._chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)
        return chunks

    def _save_vector_store(self):
        """
        Saves the current state of the vector store to the local file system.

        This method ensures that the directory specified by `self._vector_store_dir` exists,
        creating it if necessary, and then saves the vector store to the path specified by
        `self._vector_store_db_path`.

        Raises:
            OSError: If the directory cannot be created.
            Exception: If saving the vector store fails.
        """
        if not os.path.exists(self._vector_store_dir):
            os.makedirs(self._vector_store_dir)
        self.vector_store.save_local(self._vector_store_db_path)

    def _load_vector_store(self, allow_dangerous_deserialization=True):
        """
        Loads the vector store from the specified database path if it exists.

        Args:
            allow_dangerous_deserialization (bool, optional): 
                Whether to allow potentially unsafe deserialization when loading the vector store. 
                Defaults to True.

        Returns:
            object or None: The loaded vector store object if successful, otherwise None.
        """
        logger.info("Loading vector store...")
        if not os.path.exists(self._vector_store_db_path):
            logger.info("Vector store does not exist.")
            return None
        try:
            vector_store = FAISS.load_local(
                self._vector_store_db_path, 
                self._embeddings_model, 
                allow_dangerous_deserialization=allow_dangerous_deserialization
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
        logger.info("Vector store loaded.")        
        return vector_store
