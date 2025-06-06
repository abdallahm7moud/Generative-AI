# Tool: Memory management
from crewai.tools import BaseTool
import json
import os
import time
from typing import Dict, Any, Optional, List
from pydantic import Field , PrivateAttr


class StoreFactTool(BaseTool):
    """Tool for storing important facts in memory."""

    name: str = Field(default="store_fact", description="What the tool does")
    description: str = Field(default="Store an important fact in memory.")
    _memory_file: str = PrivateAttr()
    _memory: list = PrivateAttr()
    
    def __init__(self, memory_file: str = "agent_memory.json"):
        """
        Initialize the memory tool.
        Args:
            memory_file: File to store agent memories
        """
        super().__init__(name="store_fact",description="Store an important fact in memory.")
        self._memory_file = memory_file
        self._memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """
        Load memory from file or initialize if it doesn't exist.
        Returns:
            Dictionary containing agent memories
        """
        if os.path.exists(self._memory_file):
            try:
                with open(self._memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._initialize_memory()
        else:
            return self._initialize_memory()

    def _initialize_memory(self) -> Dict[str, Any]:
        """
        Initialize a new memory structure.
        Returns:
            New memory dictionary
        """
        memory = {
            "facts": [],
            "entities": {},
            "conversations": [],
            "tasks": {},
            "reflections": [],
            "last_updated": time.time()
        }
        self._save_memory(memory)
        return memory

    def _save_memory(self, memory: Optional[Dict[str, Any]] = None) -> None:
        """
        Save memory to file.
        Args:
            memory: Memory dictionary to save (uses self._memory if None)
        """
        if memory is None:
            memory = self._memory
        memory["last_updated"] = time.time()
        with open(self._memory_file, 'w') as f:
            json.dump(memory, f, indent=2)

    def _run(self, fact: str) -> str:
        """
        Store an important fact in memory.
        Args:
            fact: The fact to remember
        Returns:
            Confirmation message
        """
        self._memory["facts"].append({
            "content": fact,
            "timestamp": time.time()
        })
        self._save_memory()
        return f"Fact stored in memory: '{fact}'"


class RetrieveFactsTool(BaseTool):
    """Tool for retrieving facts from memory."""

    name: str = Field(default="retrieve_facts", description="What the tool does")
    description: str = Field(default="Retrieve facts from memory, optionally filtered by query.")
    _memory_file: str = PrivateAttr()
    _memory: list = PrivateAttr()

    def __init__(self, memory_file: str = "agent_memory.json"):
        """
        Initialize the memory tool.
        Args:
            memory_file: File to store agent memories
        """
        super().__init__(name="retrieve_facts",description="Retrieve facts from memory, optionally filtered by query.")
        self._memory_file = memory_file
        self._memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """
        Load memory from file or initialize if it doesn't exist.
        Returns:
            Dictionary containing agent memories
        """
        if os.path.exists(self._memory_file):
            try:
                with open(self._memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"facts": [], "last_updated": time.time()}
        else:
            return {"facts": [], "last_updated": time.time()}

    def _run(self, query: str = "") -> str:
        """
        Retrieve facts from memory, optionally filtered by query.
        Args:
            query: Optional search term to filter facts
        Returns:
            String containing retrieved facts
        """
        facts = self._memory.get("facts", [])
        if not facts:
            return "No facts stored in memory."

        if query:
            filtered_facts = [
                f for f in facts if query.lower() in f["content"].lower()
            ]
            if not filtered_facts:
                return f"No facts found matching query: '{query}'"
            result = f"Facts related to '{query}':\n\n"
            for i, fact in enumerate(filtered_facts, 1):
                result += f"{i}. {fact['content']}\n"
            return result
        else:
            result = "All stored facts:\n\n"
            for i, fact in enumerate(facts, 1):
                result += f"{i}. {fact['content']}\n"
            return result


def get_memory_tools(memory_file: str = "agent_memory.json") -> List[BaseTool]:
    """
    Get a collection of memory-related tools.
    Args:
        memory_file: File path for shared memory
    Returns:
        List of memory tools
    """
    return [
        StoreFactTool(memory_file=memory_file),
        RetrieveFactsTool(memory_file=memory_file)
    ]
