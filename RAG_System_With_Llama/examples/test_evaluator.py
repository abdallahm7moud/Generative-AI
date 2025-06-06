# examples/test_evaluator.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from main import OllamaRAGSystem
from rag.evaluator import SimpleRAGEvaluator
import pandas as pd

# Initialize the RAG system
rag = OllamaRAGSystem(
    data_dir="data",
    ollama_model="llama3.2:1b",
    top_k=2
)

# Create test cases
test_queries = [
    {
        "query": "What is RAG and when was it introduced?",
        "relevant_docs": ["data/rag_explanation.txt"]
    },
    {
        "query": "What are the main components of a RAG system?",
        "relevant_docs": ["data/rag_explanation.txt"]
    },
    {
        "query": "What are the advantages of using RAG?",
        "relevant_docs": ["data/rag_explanation.txt"]
    }
]
 
# Initialize evaluator
evaluator = SimpleRAGEvaluator()
# Run evaluation
results = evaluator.run_evaluation(rag, test_queries)
# Display results
pd.set_option('display.max_colwidth', None)
print(results[["query", "response", "precision", "recall", "f1_score", "context_utilization", "has_citations"]])