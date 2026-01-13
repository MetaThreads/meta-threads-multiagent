"""Web search implementations."""

from threads_multiagent.search.base import BaseWebSearch, SearchResult
from threads_multiagent.search.duckduckgo import DuckDuckGoSearch

__all__ = ["BaseWebSearch", "SearchResult", "DuckDuckGoSearch"]
