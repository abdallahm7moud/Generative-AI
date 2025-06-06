# Utility: Helper functions
from crewai import Task

def create_delegated_tasks(manager_agent, worker_agents, main_task_description):
    """
    Helper function to set up a task that can be delegated to worker agents.
    Args:
    manager_agent: The manager agent who will delegate tasks
    worker_agents: Dict of worker agents keyed by role
    main_task_description: The high-level task description
    Returns:
    A CrewAI Task configured for delegation
    """
    # Create the main task for the manager
    manager_task = Task(
        description=f"""
            Coordinate the completion of this project: {main_task_description}
            Your responsibilities:
            1. Analyze the requirements and break them down into subtasks
            2. Assign each subtask to the appropriate specialist:
            - Research tasks to the Research Specialist
            - Analysis tasks to the Data Analyst
            - Implementation tasks to the Software Developer
            3. Review the work of each specialist and provide feedback
            4. Integrate the results into a cohesive final deliverable
            5. Ensure all requirements are met
            You can communicate with these specialists:
            - Research Specialist: for information gathering and verification
            - Data Analyst: for processing information and extracting insights
            - Software Developer: for implementing code solutions
            Start by creating a clear project plan with subtasks for each specialist.
            """,
        expected_output="""
            A complete, high-quality solution that fully addresses the original request,
            including all necessary research, analysis, and implementation components.
            The solution should include:
            1. A summary of the project and approach
            2. Research findings with citations
            3. Analysis and insights derived from the research
            4. Implementation details (code, documentation, etc.)
            5. Final recommendations or conclusions
            """,
            agent=manager_agent
            )
    return manager_task
