# config.py
from crewai import Agent
import os
from langchain_community.chat_models import ChatOllama

# Set a dummy API key (required but not used with Ollama)
os.environ["OPENAI_API_KEY"] = "dummy-key"
# Initialize Ollama LLM for CrewAI
def get_llama_llm(temperature=0.2, system_prompt=None):
    """
    Configure and return an LLM instance using Llama 3.2-1b via Ollama
    Args:
    temperature: Controls randomness (lower = more deterministic)
    system_prompt: Optional system prompt to guide model behavior
    Returns:
    Configured LLM instance for CrewAI
    """

    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant working as part of a
        multi-agent system. You are precise, factual, and helpful. You complete
        tasks thoroughly and report your results clearly."""
        # Configure the Ollama LLM
    llm = ChatOllama(
        model="ollama/codellama:34b",
        temperature=temperature,
        base_url="http://localhost:11434",
        system=system_prompt,
        )
    return llm

