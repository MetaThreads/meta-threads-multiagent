"""Pytest fixtures for tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables before any tests run."""
    os.environ.setdefault("OPENROUTER_API_KEY", "test-api-key")
    os.environ.setdefault("THREADS_BEARER_TOKEN", "test-token:12345")
    os.environ.setdefault("THREADS_MCP_URL", "https://test.mcp.example.com/mcp")
    os.environ.setdefault("LANGFUSE_ENABLED", "false")
    yield


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before each test."""
    from threads_multiagent.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Find trending AI news and post to Threads"},
    ]


@pytest.fixture
def mock_news_articles():
    """Sample news articles for testing."""
    return [
        {
            "title": "OpenAI releases GPT-5",
            "url": "https://example.com/gpt5",
            "summary": "OpenAI has announced GPT-5 with enhanced capabilities.",
            "source": "TechNews",
            "published_at": "2025-01-12T10:00:00Z",
        },
    ]
