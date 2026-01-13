"""Web search implementations."""

from threads_hype_agent.search.base import BaseWebSearch, SearchResult
from threads_hype_agent.search.duckduckgo import DuckDuckGoSearch

__all__ = ["BaseWebSearch", "SearchResult", "DuckDuckGoSearch"]
