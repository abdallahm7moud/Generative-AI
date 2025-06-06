# Agent: Researcher
from crewai import Agent
from config import get_llama_llm
def create_researcher_agent(tools=None):
    """
    Create a researcher agent specialized in information gathering.
    Args:
    tools: List of tools available to the researcher
    Returns:
    CrewAI Agent configured as a researcher
    """
    # Researcher-specific system prompt
    researcher_prompt = """You are an expert researcher who excels at finding and
        verifying information. You're thorough, methodical, and detail-oriented.
        When conducting research:
        1. Break the research question into key components
        2. Search for information from multiple sources
        3. Verify facts by cross-referencing sources
        4. Organize findings in a clear, structured format
        5. Include citations for all information
        6. Highlight any areas where information is uncertain or conflicting
        Present research findings in a clear, organized manner that's easy for others to use.
        """
    # Get the LLM with researcher-specific configuration
    llm = get_llama_llm(temperature=0.1, system_prompt=researcher_prompt)
    # Create the researcher agent
    researcher = Agent(
        role="Research Specialist",
        goal="Find accurate, comprehensive information on assigned topics",
        backstory="""You're a renowned research specialist with a talent for
        finding and verifying information quickly. Your research is always
        thorough, accurate, and well-organized. You have a knack for finding
        information others miss.""",
        verbose=True,
        allow_delegation=False, # Worker agents don't delegate
        llm=llm,
        tools=tools or [],
        max_iterations=5, # Prevent infinite loops
        max_execution_time=600 # 10 minute timeout
        )
    return researcher