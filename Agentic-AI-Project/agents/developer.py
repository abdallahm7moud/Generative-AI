# Agent: Developer
from crewai import Agent
from config import get_llama_llm
def create_developer_agent(tools=None):
    """
    Create a developer agent specialized in code implementation.
    Args:
    tools: List of tools available to the developer
    Returns:
    CrewAI Agent configured as a developer
    """
    # Developer-specific system prompt
    developer_prompt = """You are an expert software developer who excels at
        translating requirements into high-quality code. You're detail-oriented,
        logical, and focus on creating clean, maintainable solutions.
        When developing code:
        1. Understand the requirements thoroughly
        2. Plan your implementation approach
        3. Write clean, well-documented code
        4. Include error handling and edge cases
        5. Provide clear usage examples
        Your code should be elegant, efficient, and well-documented.
        """
    # Get the LLM with developer-specific configuration
    llm = get_llama_llm(temperature=0.2, system_prompt=developer_prompt)
    # Create the developer agent
    developer = Agent(
        role="Software Developer",
        goal="Create high-quality, well-documented code based on specifications",
        backstory="""You're a seasoned software developer with expertise across
        multiple programming languages and frameworks. You write clean, efficient
        code that's easy to maintain and extend. You have a talent for translating
        complex requirements into elegant technical solutions.""",
        verbose=True,
        allow_delegation=False, # Worker agents don't delegate
        llm=llm,
        tools=tools or [],
        max_iterations=5,
        max_execution_time=600
        )
    return developer