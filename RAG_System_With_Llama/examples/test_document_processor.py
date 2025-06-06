# examples/test_document_processor.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from rag.document_processor import DocumentProcessor


# Create a test document
os.makedirs("data", exist_ok=True)
with open("data/test_document.txt", "w") as f:
    f.write("""
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
    by allowing them to access external knowledge. This approach combines the strengths of 
    retrieval-based and generation-based methods in natural language processing.

    The key components of a RAG system include:
    1. A document store containing knowledge
    2. A retrieval system to find relevant information
    3. A language model to generate responses

    RAG addresses the limitations of traditional language models, such as outdated knowledge
    and hallucinations, by grounding responses in factual information from external sources.
    """)
# Initialize the document processor
# Note: Using smaller chunks for Llama 3.2 1B to accommodate its smaller context window
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
# Process the test document
chunks = processor.process_documents("data")
# Print the chunks
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"Text: {chunk.page_content[:100]}...")
    print(f"Metadata: {chunk.metadata}")