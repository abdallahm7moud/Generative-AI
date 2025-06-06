# main.py
from rag.document_processor import DocumentProcessor
from rag.embeddings import EmbeddingManager
from rag.retriever import Retriever
from rag.generator import OllamaGenerator
import os
from typing import Optional
 
class OllamaRAGSystem:
    """Complete RAG System using Llama 3.2 1B via Ollama."""

    def __init__(
        self, 
        data_dir: str = "data",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: Optional[str] = "vectorstore",
        ollama_model: str = "llama3.2:1b",
        top_k: int = 2
    ):
        """Initialize the RAG System with all components."""
        # Initialize document processor
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            persist_directory=persist_dir
        )
        
        # Load or create vector store
        if persist_dir and os.path.exists(persist_dir):
            success = self.embedding_manager.load_vectorstore()
            if not success:
                self._create_new_vectorstore(data_dir)
        else:
            self._create_new_vectorstore(data_dir)
        
        # Initialize retriever
        self.vectorstore = self.embedding_manager.get_vectorstore()
        self.retriever = Retriever(self.vectorstore, top_k=top_k)
        
        # Initialize Ollama generator
        self.generator = OllamaGenerator(model_name=ollama_model)
        
        print("Ollama RAG system initialized successfully!")

    def _create_new_vectorstore(self, data_dir: str):
        """Process documents and create a new vector store."""
        chunks = self.processor.process_documents(data_dir)
        self.embedding_manager.create_vectorstore(chunks)

    def add_documents(self, directory: str):
        """Add new documents to the system."""
        chunks = self.processor.process_documents(directory)
        self.embedding_manager.add_documents(chunks)

    def query(self, query: str, with_sources: bool = False, use_mmr: bool = False):
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            with_sources: Whether to include source citations
            use_mmr: Whether to use MMR for diverse retrieval
            
        Returns:
            Generated response with metadata
        """
        # Retrieve relevant documents
        if use_mmr:
            documents = self.retriever.retrieve_with_mmr(query)
        else:
            documents = self.retriever.retrieve(query)
        
        # Generate response
        if with_sources:
            return self.generator.generate_response_with_sources(query, documents)
        else:
            return self.generator.generate_response(query, documents)

    def direct_query(self, query: str, use_mmr: bool = False):
        """
        Process a query using direct Ollama API call.
        
        Args:
            query: User query
            use_mmr: Whether to use MMR for diverse retrieval
            
        Returns:
            Generated response with metadata
        """
        # Retrieve relevant documents
        if use_mmr:
            documents = self.retriever.retrieve_with_mmr(query)
        else:
            documents = self.retriever.retrieve(query)
        
        # Format the context
        context = self.generator.format_documents(documents)
        
        # Generate response using direct API call
        response = self.generator.direct_ollama_call(query, context)
        
        return {
            "query": query,
            "response": response,
            "context_documents": documents,
            "model": self.generator.model_name
        }
        
        
# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = OllamaRAGSystem(
        data_dir="data",
        ollama_model="llama3.2:1b"
    )

    # Process a query
    query = "What is RAG and what are its key components?"
    result = rag.query(query, with_sources=True)

    # Print the response
    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{result['response']}")