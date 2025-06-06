# Agent: Analyst
from crewai import Agent
from config import get_llama_llm
def create_analyst_agent(tools=None):
    """
    Create an analyst agent specialized in data processing and insights.
    Args:
    tools: List of tools available to the analyst
    Returns:
    CrewAI Agent configured as an analyst
    """
    # Analyst-specific system prompt
    analyst_prompt = """You are an expert data analyst who excels at processing
        information and extracting meaningful insights. You're analytical,
        logical, and have excellent pattern recognition skills.
        When analyzing information:
        1. Organize and structure the data logically
        2. Identify key patterns, trends, and relationships
        3. Draw evidence-based conclusions
        4. Provide clear, actionable insights
        5. Highlight limitations and uncertainties in your analysis
        Present your analysis in a clear, structured format with supporting evidence.
        """
    # Get the LLM with analyst-specific configuration
    llm = get_llama_llm(temperature=0.1, system_prompt=analyst_prompt)
    # Create the analyst agent
    analyst = Agent(
        role="Data Analyst",
        goal="Process information to extract valuable insights and patterns",
        backstory="""You're a highly skilled analyst with years of experience
        turning raw information into actionable insights. You have a talent
        for seeing patterns others miss and explaining complex findings in
        clear, accessible terms.""",
        verbose=True,
        allow_delegation=False, # Worker agents don't delegate
        llm=llm,
        tools=tools or [],
        max_iterations=5,
        max_execution_time=600
        )
    return analyst
