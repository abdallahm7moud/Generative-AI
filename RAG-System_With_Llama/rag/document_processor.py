# rag/document_processor.py
import os
from typing import List, Dict, Any, Optional
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Class for processing documents for a RAG system."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        """
        Initialize the DocumentProcessor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with text and metadata
        """
        _, file_extension = os.path.splitext(file_path)
        
        # Select appropriate loader based on file extension
        if file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ['.txt', '.md', '.html']:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load the document
        documents = loader.load()
        
        # Add source information to metadata
        for doc in documents:
            doc.metadata['source'] = file_path
            
        return documents

    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory containing documents
            
        Returns:
            List of document chunks with text and metadata
        """
        all_documents = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                except (ValueError, Exception) as e:
                    print(f"Error loading {file_path}: {e}")
        
        return all_documents

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        return self.text_splitter.split_documents(documents)

    def process_documents(self, directory: str) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory (load and split).
        
        Args:
            directory: Directory containing documents
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_documents(directory)
        chunks = self.split_documents(documents)
        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        return chunks