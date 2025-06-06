import os
import json
import time
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool

# ------------------ Input Schemas ------------------

class StoreFactArgs(BaseModel):
    fact: str = Field(description="The fact to store in memory")


class RetrieveFactsArgs(BaseModel):
    query: Optional[str] = Field(default="", description="Optional text to filter facts")


# ------------------ StoreFactTool ------------------

class StoreFactTool(BaseTool):
    name: str = "store_fact"
    description: str = "Store an important fact in memory."
    args_schema: Type[StoreFactArgs] = StoreFactArgs
    _memory_file: str = PrivateAttr("agent_memory.json")

    def __init__(self, memory_file: str = "agent_memory.json"):
        super().__init__()
        self._memory_file = memory_file

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self._memory_file):
            try:
                with open(self._memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._initialize_memory()
        else:
            return self._initialize_memory()

    def _initialize_memory(self) -> Dict[str, Any]:
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
        if memory is None:
            memory = self._load_memory()
        memory["last_updated"] = time.time()
        with open(self._memory_file, 'w') as f:
            json.dump(memory, f, indent=2)

    def _run(self, fact: str) -> str:
        try:
            memory = self._load_memory()
            memory["facts"].append({
                "content": fact,
                "timestamp": time.time()
            })
            self._save_memory(memory)
            return f"Fact stored in memory: '{fact}'"
        except Exception as ex:
            return f"Error storing fact: {str(ex)}"


# ------------------ RetrieveFactsTool ------------------

class RetrieveFactsTool(BaseTool):
    name: str = "retrieve_facts"
    description: str = "Retrieve facts from memory, optionally filtered by query."
    args_schema: Type[RetrieveFactsArgs] = RetrieveFactsArgs
    _memory_file: str = PrivateAttr("agent_memory.json")

    def __init__(self, memory_file: str = "agent_memory.json"):
        super().__init__()
        self._memory_file = memory_file

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self._memory_file):
            try:
                with open(self._memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"facts": [], "last_updated": time.time()}
        else:
            return {"facts": [], "last_updated": time.time()}

    def _run(self, query: str = "") -> str:
        try:
            memory = self._load_memory()
            facts = memory.get("facts", [])
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
        except Exception as ex:
            return f"Error retrieving facts: {str(ex)}"


# ------------------ Tool Factory ------------------

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
