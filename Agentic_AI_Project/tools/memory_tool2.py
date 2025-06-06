import json
import os
import time
from langchain.tools import tool

MEMORY_FILE = "agent_memory.json"

def _load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return _initialize_memory()
    return _initialize_memory()

def _initialize_memory():
    memory = {
        "facts": [],
        "entities": {},
        "conversations": [],
        "tasks": {},
        "reflections": [],
        "last_updated": time.time()
    }
    _save_memory(memory)
    return memory

def _save_memory(memory):
    memory["last_updated"] = time.time()
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)


@tool("store_fact", return_direct=True)
def store_fact(fact: str) -> str:
    """
    Store an important fact in memory.
    
    Args:
        fact: A string representing the fact to be stored.
    """
    memory = _load_memory()
    memory["facts"].append({
        "content": fact,
        "timestamp": time.time()
    })
    _save_memory(memory)
    return f"Fact stored in memory: '{fact}'"


@tool("retrieve_facts", return_direct=True)
def retrieve_facts(query: str = "") -> str:
    """
    Retrieve facts from memory, optionally filtered by a search query.
    
    Args:
        query: Optional keyword to filter the facts.
    """
    memory = _load_memory()
    facts = memory.get("facts", [])

    if not facts:
        return "No facts stored in memory."

    if query:
        filtered_facts = [
            f for f in facts if query.lower() in f["content"].lower()
        ]
        if not filtered_facts:
            return f"No facts found matching query: '{query}'"
        return "\n".join(f"{i+1}. {fact['content']}" for i, fact in enumerate(filtered_facts))
    
    return "\n".join(f"{i+1}. {fact['content']}" for i, fact in enumerate(facts))