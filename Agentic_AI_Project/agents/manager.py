# Agent: Manager (supervisor)
from crewai import Agent
from config import get_llama_llm
def create_manager_agent(tools=None):
    """
    Create a manager agent that oversees the entire operation.
    Args:
    tools: List of tools available to the manager
    Returns:
    CrewAI Agent configured as a manager
    """
    # Manager-specific system prompt
    manager_prompt = """You are a skilled project manager who coordinates a team of AI agents.
    Your job is to:
    1. Break down complex tasks into manageable steps
    2. Assign tasks to the appropriate specialists
    3. Review their work and provide feedback
    4. Ensure the final deliverable meets the requirements
    5. Request human input when necessary
    
    You make clear, decisive plans and communicate expectations clearly.
    You excel at synthesizing information from multiple sources and team members.
    """
    # Get the LLM with manager-specific configuration
    llm = get_llama_llm(temperature=0.2, system_prompt=manager_prompt)
    # Create the manager agent
    manager = Agent(
        role="Project Manager",
        goal="Coordinate the team to deliver high-quality results efficiently",
        backstory="""You have years of experience managing complex projects.
        You're known for your ability to keep teams on track and deliver
        exceptional results on time. You understand each team member's
        strengths and utilize them effectively.""",
        verbose=True,
        allow_delegation=True, # Manager can delegate tasks
        llm=llm,
        tools=tools or [],
        max_iterations=5, # Prevent infinite loops
        max_execution_time=600 # 10 minute timeout
        )
    return manager