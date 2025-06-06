# rag/generator.py
from typing import List, Dict, Any, Optional
import requests
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
 
class OllamaGenerator:
    """Class for generating responses using Llama 3.2 1B via Ollama and retrieved documents."""

    def __init__(self, model_name: str = "llama3.2:1b", temperature: float = 0.1):
        """
        Initialize the OllamaGenerator.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature parameter for generation (0.0 = deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the Ollama LLM
        try:
            self.llm = Ollama(model=model_name, temperature=temperature)
            print(f"Connected to Ollama with model: {model_name}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running and the model is downloaded.")
            raise
        # Create default system and user templates optimized for Llama 3.2
        self.system_template = """You are a helpful AI assistant. Answer the user's question based on the provided information.
        If the answer cannot be determined from the context, say "I don't know based on the provided information" but still offer any pertinent insights you might have.
        Include citations from the context in your answer using [Document X] notation where X is the document number.
        Keep your responses concise and focused."""
        
        self.user_template = """Context information:
    {context}
    Question: {question}
    Answer based on the context information:"""

    def format_documents(self, documents: List[Document]) -> str:
        """
        Format a list of documents into a string for the context.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown source')
            content = doc.page_content.strip()
            formatted_docs.append(f"[Document {i+1}] {content}")
        
        return "\n\n".join(formatted_docs)

    def generate_response(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate a response to a query using retrieved documents as context.
        
        Args:
            query: User query
            documents: List of retrieved documents
            
        Returns:
            Dictionary containing response and metadata
        """
        # Format the documents into a context string
        context = self.format_documents(documents)
        
        # Create the prompt
        prompt = f"{self.system_template}\n\n{self.user_template.format(context=context, question=query)}"
        
        # Generate response using Ollama
        response = self.llm.invoke(prompt)
        
        # Return response with metadata
        return {
            "query": query,
            "response": response,
            "context_documents": documents,
            "model": self.model_name
        }

    def generate_response_with_sources(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate a response with explicit source citations.
        
        Args:
            query: User query
            documents: List of retrieved documents
            
        Returns:
            Dictionary containing response with sources and metadata
        """
        # Create a source-citing system prompt
        source_system_template = """You are a helpful AI assistant. Answer the user's question based on the provided information.
        If the answer cannot be determined from the context, say "I don't know based on the provided information" but still offer any pertinent insights you might have.
        Include citations from the context in your answer using [Document X] notation where X is the document number.
        Keep your responses concise and focused."""
        
        # Format the documents into a context string with clear document markers
        context = self.format_documents(documents)
        
        # Create the prompt
        prompt = f"{source_system_template}\n\n{self.user_template.format(context=context, question=query)}"
        
        # Generate response using Ollama
        response = self.llm.invoke(prompt)
        
        # Return response with metadata
        return {
            "query": query,
            "response": response,
            "context_documents": documents,
            "model": self.model_name
        }

    def direct_ollama_call(self, query: str, context: str) -> str:
        """
        Make a direct call to Ollama API for more control.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        prompt = f"{self.system_template}\n\n{self.user_template.format(context=context, question=query)}"
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                }
            )
            result = response.json()
            return result["response"]
        except Exception as e:
            print(f"Error in direct Ollama call: {e}")
            return "Error generating response: " + str(e)