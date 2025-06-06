# rag/retriever.py
from typing import List, Dict, Any, Optional
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
 
class Retriever:
    """Class for retrieving relevant documents based on queries."""

    def __init__(self, vectorstore: VectorStore, top_k: int = 3):
        """
        Initialize the Retriever.
        
        Args:
            vectorstore: Vector store containing document embeddings
            top_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        # Create a retriever from the vector store
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            
        Returns:
            List of retrieved documents
        """
        documents = self.retriever.get_relevant_documents(query)
        return documents

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: Query text
            
        Returns:
            List of (document, score) tuples
        """
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        return docs_and_scores

    def retrieve_with_mmr(self, query: str, diversity: float = 0.3) -> List[Document]:
        """
        Retrieve documents using Maximum Marginal Relevance for diversity.
        
        Args:
            query: Query text
            diversity: Diversity parameter (0-1, higher means more diverse results)
            
        Returns:
            List of retrieved documents
        """
        if hasattr(self.vectorstore, "max_marginal_relevance_search"):
            documents = self.vectorstore.max_marginal_relevance_search(
                query, k=self.top_k, fetch_k=self.top_k*3, lambda_mult=diversity
            )
            return documents
        else:
            print("Vector store doesn't support MMR, falling back to standard retrieval")
            return self.retrieve(query)

    def hybrid_search(self, query: str) -> List[Document]:
        """
        Perform hybrid search combining vector search with keyword search if available.
        
        Args:
            query: Query text
            
        Returns:
            List of retrieved documents
        """
        if hasattr(self.vectorstore, "hybrid_search"):
            documents = self.vectorstore.hybrid_search(query, k=self.top_k)
            return documents
        else:
            print("Vector store doesn't support hybrid search, falling back to standard retrieval")
            return self.retrieve(query)