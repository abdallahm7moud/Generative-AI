# rag/embeddings.py
from typing import List, Dict, Any, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

class EmbeddingManager:
    """Class for managing embeddings and vector database operations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: Optional[str] = None):
        """
        Initialize the EmbeddingManager.
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
            persist_directory: Directory to persist the vector store (None for in-memory)
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Initialize the embedding model (using lightweight model suitable for local use)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Dict[str, Any]]) -> None:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of document chunks to embed and store
        """
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist if a directory is specified
        if self.persist_directory:
            self.vectorstore.persist()
            
        print(f"Created vector store with {len(documents)} documents")

    def load_vectorstore(self) -> bool:
        """
        Load a persisted vector store.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            print("No persist directory specified or directory doesn't exist")
            return False
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Loaded vector store from {self.persist_directory}")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of document chunks to add
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        self.vectorstore.add_documents(documents)
        
        # Persist if a directory is specified
        if self.persist_directory:
            self.vectorstore.persist()
            
        print(f"Added {len(documents)} documents to vector store")

    def get_vectorstore(self):
        """
        Get the vector store.
        
        Returns:
            The vector store object
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore