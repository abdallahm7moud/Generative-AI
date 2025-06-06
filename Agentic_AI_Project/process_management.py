# process_management.py

import os
from crewai import Agent, Task, Crew, Process
from config import get_llama_llm

# Set a dummy API key (required by LangChain even if using Ollama)
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Human feedback function
def get_human_feedback(agent_name, task_description, output):
    print("\n" + "=" * 50)
    print(f"Human Feedback Required for {agent_name}")
    print("=" * 50)
    print(f"\nTask: {task_description}")
    print(f"\nCurrent Output:\n{output}")
    feedback = input("\nPlease provide feedback or guidance (press Enter to approve): ")
    if feedback.strip():
        return False, feedback
    else:
        return True, "Output approved"


# Main process runner
def run_process_example():
    """Demonstrate different process management approaches in CrewAI"""

    llm = get_llama_llm()

    # Define agents
    manager = Agent(
        role="Manager",
        goal="Coordinate the team",
        backstory="You're an experienced project manager.",
        llm=llm,
        max_iterations=3,
        max_execution_time=300
    )

    researcher = Agent(
        role="Researcher",
        goal="Find information",
        backstory="You're an expert researcher.",
        llm=llm,
        max_iterations=3,
        max_execution_time=300
    )

    analyst = Agent(
        role="Analyst",
        goal="Analyze information",
        backstory="You're a skilled data analyst.",
        llm=llm,
        max_iterations=3,
        max_execution_time=300
    )

    # Define tasks
    task1 = Task(
        description="Research the latest advancements in AI agents",
        expected_output="A comprehensive research report on AI agents",
        agent=researcher,
        human_input_mode="ALWAYS"
    )

    task2 = Task(
        description="Analyze the research findings to identify key trends",
        expected_output="An analytical report with insights on AI agent trends",
        agent=analyst,
        context=[task1],
        human_input_mode="NEVER"
    )

    # ---------------------------
    # 1. Sequential Process
    # ---------------------------
    print("\n🔄 Running Sequential Process...")
    crew_sequential = Crew(
        agents=[manager, researcher, analyst],
        tasks=[task1, task2],
        verbose=True,
        process=Process.sequential
    )

    try:
        result_sequential = crew_sequential.kickoff()
        print("\n✅ Sequential Process Result:")
        print(result_sequential)
    except Exception as e:
        print(f"❌ Sequential process error: {str(e)}")

    # ---------------------------
    # 2. Hierarchical Process
    # ---------------------------
    print("\n🧠 Running Hierarchical Process...")
    crew_hierarchical = Crew(
        agents=[manager, researcher, analyst],
        tasks=[task1, task2],
        verbose=True,
        process=Process.hierarchical,
        manager_agent=manager
    )

    try:
        result_hierarchical = crew_hierarchical.kickoff()
        print("\n✅ Hierarchical Process Result:")
        print(result_hierarchical)
        return result_hierarchical
    except Exception as e:
        print(f"❌ Hierarchical process error: {str(e)}")
        return f"Process example failed: {str(e)}"


# Entry point
if __name__ == "__main__":
    run_process_example()
