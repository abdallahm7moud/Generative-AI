# main.py

import os
from crewai import Agent, Task, Crew, Process

# Agent creators
from agents.manager import create_manager_agent
from agents.researcher import create_researcher_agent
from agents.analyst import create_analyst_agent
from agents.developer import create_developer_agent

# Tools
from tools.web_search import WebSearchTool
from tools.document_tool import DocumentAnalyzerTool
from tools.code_tool import get_code_tools
from tools.memory_tool import get_memory_tools

# Helpers
from utils.helpers import create_delegated_tasks

# Set a dummy API key for OpenAI (required by LangChain, not used with Ollama)
os.environ["OPENAI_API_KEY"] = "dummy-key"


def run_hierarchical_crew():
    """Run a hierarchical CrewAI workflow with task delegation"""

    # 🧠 Initialize tools
    web_search_tool = WebSearchTool()
    document_analyzer = DocumentAnalyzerTool()
    code_tools = get_code_tools()
    memory_tools = get_memory_tools()

    # 🤖 Create agents
    manager = create_manager_agent(tools=memory_tools)
    researcher = create_researcher_agent(tools=[web_search_tool, document_analyzer] + memory_tools)
    analyst = create_analyst_agent(tools=[document_analyzer] + memory_tools)
    developer = create_developer_agent(tools=code_tools + memory_tools)

    # 🔄 Group agents for task delegation
    worker_agents = {
        "researcher": researcher,
        "analyst": analyst,
        "developer": developer
    }

    # 📌 Define the main task for the manager to delegate
    main_task = create_delegated_tasks(
        manager,
        worker_agents,
        """
        Research the current state of Retrieval-Augmented Generation (RAG) systems in 2025.
        Analyze their strengths and limitations compared to fine-tuning approaches.
        Then, develop a basic implementation of a RAG pipeline with documentation.
        """
    )

    # 🧩 Setup the hierarchical crew
    crew = Crew(
        agents=[manager, researcher, analyst, developer],
        tasks=[main_task],
        verbose=True,               # Show step-by-step logs
        process=Process.sequential  # Ensure tasks run in order
    )

    # 🚀 Run the crew
    try:
        result = crew.kickoff()
        print("\n✅ Final Result:")
        print(result)
        return result
    except Exception as e:
        print(f"❌ Error in crew execution: {str(e)}")
        return f"Crew execution failed: {str(e)}"


# Entry point
if __name__ == "__main__":
    run_hierarchical_crew()
