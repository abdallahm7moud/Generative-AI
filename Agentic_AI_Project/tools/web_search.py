import os
import time
import json
import hashlib
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, PrivateAttr
from crewai.tools import BaseTool

# --------- Input Schema ---------
class WebSearchArgs(BaseModel):
    query: str

# --------- Tool Implementation ---------
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Use this tool to perform a web search and return the most relevant result."
    args_schema: Type[WebSearchArgs] = WebSearchArgs
    _cache_dir: str = PrivateAttr(default="./.cache")

    def __init__(self, cache_dir: Optional[str] = "./.cache"):
        super().__init__()
        self._cache_dir = cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def _get_cache_path(self, query: str) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(self._cache_dir, f"search_{query_hash}.json")

    def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            cache_time = cached_data.get('timestamp', 0)
            if time.time() - cache_time < 86400:  # 24 hours in seconds
                return cached_data.get('results')
        return None

    def _cache_result(self, query: str, results: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path(query)
        cache_data = {
            'timestamp': time.time(),
            'query': query,
            'results': results
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

    def _format_results(self, results: Dict[str, Any], query: str) -> str:
        items = results.get("items", [])
        if not items:
            return f"No results found for: {query}"
        formatted = f"Search results for '{query}':\n\n"
        for i, item in enumerate(items, 1):
            formatted += f"Result {i}:\n"
            formatted += f"Title: {item.get('title', 'No title')}\n"
            formatted += f"Summary: {item.get('snippet', 'No summary available')}\n"
            formatted += f"URL: {item.get('link', 'No link available')}\n\n"
        return formatted

    def _run(self, query: str) -> str:
        try:
            cached_results = self._get_cached_result(query)
            if cached_results:
                return self._format_results(cached_results, query)
            # Simulated search results
            results = {
                "items": [
                    {
                        "title": f"Example Result 1 for {query}",
                        "snippet": f"This is a detailed information about {query} with relevant facts and figures.",
                        "link": f"https://example.com/1?q={query.replace(' ', '+')}"
                    },
                    {
                        "title": f"Example Result 2 for {query}",
                        "snippet": f"Additional information about {query} including recent developments and analysis.",
                        "link": f"https://example.com/2?q={query.replace(' ', '+')}"
                    },
                    {
                        "title": f"Example Result 3 for {query}",
                        "snippet": f"Comprehensive guide to {query} with step-by-step instructions and explanations.",
                        "link": f"https://example.com/3?q={query.replace(' ', '+')}"
                    }
                ]
            }
            self._cache_result(query, results)
            return self._format_results(results, query)
        except Exception as e:
            return f"Error performing search: {str(e)}"
