"""Base web search abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a web search result."""

    title: str
    url: str
    snippet: str
    source: str | None = None


class BaseWebSearch(ABC):
    """Abstract base class for web search implementations."""

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search the web for the given query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of search results.
        """
        ...
