"""Tests for web search implementations."""

import pytest
from unittest.mock import MagicMock, patch

from threads_hype_agent.search.base import BaseWebSearch, SearchResult
from threads_hype_agent.search.duckduckgo import DuckDuckGoSearch


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="example.com",
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.source == "example.com"


class TestBaseWebSearch:
    """Tests for BaseWebSearch abstract class."""

    def test_cannot_instantiate(self):
        """Test that BaseWebSearch cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseWebSearch()


class TestDuckDuckGoSearch:
    """Tests for DuckDuckGoSearch implementation."""

    def test_initialization(self):
        """Test DuckDuckGoSearch initialization."""
        search = DuckDuckGoSearch()
        assert search._region == "wt-wt"
        assert search._safesearch == "moderate"

    def test_initialization_custom_params(self):
        """Test DuckDuckGoSearch with custom parameters."""
        search = DuckDuckGoSearch(region="us-en", safesearch="strict")
        assert search._region == "us-en"
        assert search._safesearch == "strict"

    def test_extract_source(self):
        """Test URL source extraction."""
        search = DuckDuckGoSearch()

        assert search._extract_source("https://www.example.com/page") == "example.com"
        assert search._extract_source("https://news.example.com/article") == "news.example.com"
        assert search._extract_source("http://test.org/path") == "test.org"

    def test_extract_source_invalid(self):
        """Test source extraction with invalid URL."""
        search = DuckDuckGoSearch()
        assert search._extract_source("invalid-url") == ""

    @pytest.mark.asyncio
    async def test_search_with_mock(self):
        """Test search with mocked DDGS."""
        search = DuckDuckGoSearch()

        mock_results = [
            {
                "title": "Test Article",
                "href": "https://example.com/article",
                "body": "Test snippet content",
            }
        ]

        with patch.object(search, '_search_sync', return_value=[
            SearchResult(
                title="Test Article",
                url="https://example.com/article",
                snippet="Test snippet content",
                source="example.com",
            )
        ]):
            results = await search.search("test query", limit=5)

            assert len(results) == 1
            assert results[0].title == "Test Article"
            assert results[0].url == "https://example.com/article"

    def test_search_sync_exception_handling(self):
        """Test that _search_sync handles exceptions gracefully."""
        search = DuckDuckGoSearch()

        with patch('threads_hype_agent.search.duckduckgo.DDGS') as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("API Error")
            results = search._search_sync("test", 5)
            assert results == []

    def test_search_sync_success(self):
        """Test successful sync search."""
        search = DuckDuckGoSearch()

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Test", "href": "https://test.com", "body": "snippet"}
        ]

        with patch('threads_hype_agent.search.duckduckgo.DDGS') as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
            results = search._search_sync("test", 5)

            assert len(results) == 1
            assert results[0].title == "Test"
            assert results[0].url == "https://test.com"
