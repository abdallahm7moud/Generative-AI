# examples/test_ollama_rag.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from main import OllamaRAGSystem

# Make sure we have test data
os.makedirs("data", exist_ok=True)
test_file_path = "data/rag_explanation.txt"

if not os.path.exists(test_file_path):
    with open(test_file_path, "w") as f:
        f.write("""
        # Retrieval-Augmented Generation (RAG)
        
        Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language models
        by incorporating external knowledge retrieval into the generation process. It was introduced
        by researchers at Facebook AI in 2020.
        
        ## Core Components
        
        1. **Document Store**: A collection of documents containing domain-specific knowledge.
        2. **Retriever**: A system that finds relevant documents or passages based on a query.
        3. **Generator**: A language model that produces responses using the retrieved information.
        4. **Embedding Model**: Converts text into vector representations for similarity matching.
        5. **Vector Database**: Efficiently stores and indexes embeddings for quick retrieval.
        
        ## Advantages of RAG
        
        - Reduces hallucinations by grounding responses in factual information
        - Enables access to up-to-date information beyond the model's training data
        - Allows incorporation of domain-specific knowledge
        - Provides transparency through explicit source attribution
        - More cost-effective than continuous model retraining
        
        ## Implementation Approaches
        
        There are several ways to implement RAG systems:
        
        - **Basic RAG**: Simple retrieval followed by generation
        - **Advanced RAG**: Includes query reformulation, multi-step retrieval, and reranking
        - **Hybrid Approaches**: Combines fine-tuning with retrieval for specialized domains
        """)
        
# Initialize the RAG system with Llama 3.2 1B
rag = OllamaRAGSystem(
    data_dir="data",
    ollama_model="llama3.2:1b",
    top_k=2  # Using a smaller top_k value due to Llama 3.2's limited context window
)
 
# Test queries
queries = [
    "What is RAG and when was it introduced?",
    "What are the main components of a RAG system?",
    "What are the advantages of using RAG over traditional LLMs?",
    "How can RAG be implemented in practice?"
]
 
# Process each query and display results
for query in queries:
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)

    # Get response with sources
    result = rag.query(query, with_sources=True)

    print(f"\nResponse:\n{result['response']}")

    # Test MMR retrieval for diverse results
    mmr_result = rag.query(query, with_sources=True, use_mmr=True)

    print(f"\nMMR Response:\n{mmr_result['response']}")