# test_ollama.py
from crewai import Agent , Task
from config import get_llama_llm
import time
def test_ollama_integration():
    """Test the Ollama integration with CrewAI"""
    print("Testing Ollama integration with Llama 3.2-1b...")
    # Get the LLM
    llm = get_llama_llm()
    # Create a simple agent with the LLM
    agent = Agent(
        role="Tester",
        goal="Test Ollama integration with CrewAI",
        backstory="You are a test agent designed to verify that Ollama is working correctly with CrewAI.",
        llm=llm,
        verbose=True,
        max_iterations=1, # Limit iterations to prevent hanging
        max_execution_time=60 # Set maximum execution time in seconds
        )
    # Test with a simple task
    print("\nAsking agent to perform a simple task...")
    start_time = time.time()
    try:
        # Use a timeout to avoid hanging
        response = agent.execute_task(
            task= Task(
                description="Explain what agentic AI is in 3 sentences. Keep it simple and direct.",
                expected_output="3 sentences about agentic AI"
            )
        )
        print(f"\nResponse (in {time.time() - start_time:.2f} seconds):")
        print(response)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
if __name__ == "__main__":
    test_ollama_integration()
