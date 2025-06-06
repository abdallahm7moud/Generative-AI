# rag/evaluator.py
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain.schema import Document

class SimpleRAGEvaluator:
    """Class for evaluating RAG system performance without requiring an external evaluation LLM."""

    def __init__(self):
        """Initialize the Simple RAG Evaluator."""
        pass

    def evaluate_retrieval(self, query: str, retrieved_docs: List[Document], ground_truth_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval performance.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            ground_truth_ids: List of ground truth document IDs
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not ground_truth_ids:
            return {
                "num_docs_retrieved": len(retrieved_docs)
            }

        # Extract document IDs from retrieved docs
        retrieved_ids = [doc.metadata.get('source', '') for doc in retrieved_docs]

        # Calculate precision, recall, F1
        relevant_retrieved = set(retrieved_ids).intersection(set(ground_truth_ids))
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_docs_retrieved": len(retrieved_docs)
        }

    def evaluate_response_basic(self, query: str, response: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Perform basic statistical evaluation of the generated response.
        
        Args:
            query: User query
            response: Generated response
            context_docs: Retrieved context documents
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        results = {
            "response_length": len(response.split()),
            "response_char_length": len(response)
        }

        # Context utilization (simple heuristic)
        context_text = " ".join([doc.page_content for doc in context_docs]).lower()
        response_words = set(response.lower().split())

        # Count words from response found in context
        words_from_context = sum(1 for word in response_words if word in context_text)
        context_utilization = words_from_context / len(response_words) if response_words else 0
        results["context_utilization"] = context_utilization

        # Context relevance based on query terms
        query_terms = set(query.lower().split())
        term_overlap = sum(1 for term in query_terms if term in context_text) / len(query_terms) if query_terms else 0
        results["context_relevance"] = term_overlap

        # Source citation check (for responses with sources)
        has_citations = "[document" in response.lower() or "document " in response.lower()
        results["has_citations"] = has_citations

        return results

    def run_evaluation(self, rag_system, test_queries: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Run a comprehensive evaluation on a set of test queries.
        
        Args:
            rag_system: RAG system to evaluate
            test_queries: List of dictionaries with query, [ground_truth], [relevant_docs]
            
        Returns:
            DataFrame of evaluation results
        """
        results = []

        for test_case in test_queries:
            query = test_case["query"]
            relevant_docs = test_case.get("relevant_docs", None)

            # Run query through the RAG system
            rag_result = rag_system.query(query, with_sources=True)

            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                query, 
                rag_result["context_documents"],
                relevant_docs
            )

            # Evaluate response
            response_metrics = self.evaluate_response_basic(
                query,
                rag_result["response"],
                rag_result["context_documents"]
            )

            # Combine results
            combined_metrics = {
                "query": query,
                "response": rag_result["response"],
                **retrieval_metrics,
                **response_metrics
            }

            results.append(combined_metrics)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df
