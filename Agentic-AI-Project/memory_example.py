# memory_example.py

import os
from crewai import Agent, Task, Crew, Process
from tools.memory_tool import get_memory_tools
from config import get_llama_llm

# Set a dummy API key (required but not used with Ollama)
os.environ["OPENAI_API_KEY"] = "dummy-key"

def run_memory_example():
    """Demonstrate memory persistence across tasks and runs"""

    # Initialize memory tools
    memory_tools = get_memory_tools()

    # Create an agent with memory capabilities
    memory_agent = Agent(
        role="Knowledge Manager",
        goal="Build and maintain knowledge across multiple sessions",
        backstory=(
            "You are an expert at collecting, organizing, and retrieving "
            "information. You carefully document important facts and ensure they are "
            "available for future reference."
        ),
        verbose=True,
        llm=get_llama_llm(),
        tools=memory_tools,
        max_iterations=3,
        max_execution_time=300
    )

    # Task 1: Gather and store information
    gather_task = Task(
        description=(
            "Research information about the latest AI trends in 2025.\n"
            "For each important fact you discover:\n"
            "1. Use the store_fact tool to save it to memory\n"
            "2. For major entities (people, companies, technologies), use store_entity_information\n"
            "3. Log your research progress using log_task\n"
            "Make sure to store at least 5 important facts and 3 entity descriptions."
        ),
        expected_output=(
            "A report of all information stored in memory, including:\n"
            "1. The facts that were stored\n"
            "2. The entities that were documented\n"
            "3. A reflection on the most important trends identified"
        ),
        agent=memory_agent
    )

    # Task 2: Retrieve and analyze stored information
    retrieve_task = Task(
        description=(
            "Retrieve the information you previously stored about AI trends in 2025.\n"
            "Use the retrieve_facts tool to access the information.\n"
            "Based on this retrieved information, create a comprehensive report that:\n"
            "1. Summarizes the key trends\n"
            "2. Identifies connections between different facts and entities\n"
            "3. Provides recommendations based on the stored knowledge"
        ),
        expected_output=(
            "A comprehensive analysis report that demonstrates effective use of "
            "stored information, showing how persistent memory enhances analysis."
        ),
        agent=memory_agent
    )

    # Run the crew with sequential memory flow
    crew = Crew(
        agents=[memory_agent],
        tasks=[gather_task, retrieve_task],
        verbose=True,
        process=Process.sequential
    )

    try:
        result = crew.kickoff()
        print("\n✅ Memory Example Result:")
        print(result)
        return result
    except Exception as e:
        print(f"❌ Memory example error: {str(e)}")
        return f"Memory example failed: {str(e)}"


if __name__ == "__main__":
    run_memory_example()
