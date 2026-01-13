"""DuckDuckGo web search implementation."""

import asyncio
from functools import partial
from typing import Any

from ddgs import DDGS

from threads_multiagent.logging import get_logger
from threads_multiagent.search.base import BaseWebSearch, SearchResult

logger = get_logger(__name__)


class DuckDuckGoSearch(BaseWebSearch):
    """Web search using DuckDuckGo."""

    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate"):
        """Initialize DuckDuckGo search.

        Args:
            region: Region for search results (default: worldwide).
            safesearch: Safe search level (off, moderate, strict).
        """
        self._region = region
        self._safesearch = safesearch

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search DuckDuckGo for the given query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of search results.
        """
        logger.info(f"Searching DuckDuckGo for: {query}")

        # Run sync search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            partial(self._search_sync, query, limit),
        )

        logger.info(f"Found {len(results)} search results")

        # Log results for debugging
        for i, r in enumerate(results):
            logger.debug(f"Result {i+1}: {r.title} - {r.source}")

        return results

    def _search_sync(self, query: str, limit: int) -> list[SearchResult]:
        """Synchronous search implementation.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of search results.
        """
        try:
            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(
                        query,
                        region=self._region,
                        safesearch=self._safesearch,
                        max_results=limit,
                    )
                )

            logger.debug(f"Raw DuckDuckGo results: {len(raw_results)}")

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source=self._extract_source(r.get("href", "")),
                )
                for r in raw_results
            ]

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _extract_source(self, url: str) -> str:
        """Extract domain from URL as source.

        Args:
            url: Full URL.

        Returns:
            Domain name.
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return ""


async def test_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Test function to run search independently.

    Args:
        query: Search query.
        limit: Max results.

    Returns:
        List of result dicts.
    """
    search = DuckDuckGoSearch()
    results = await search.search(query, limit)

    print(f"\n=== Search Results for: {query} ===")
    print(f"Found {len(results)} results\n")

    result_dicts = []
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
        print(f"   URL: {r.url}")
        print(f"   Source: {r.source}")
        print(f"   Snippet: {r.snippet[:150]}...")
        print()
        result_dicts.append({
            "title": r.title,
            "url": r.url,
            "source": r.source,
            "snippet": r.snippet,
        })

    return result_dicts
