"""Pytest fixtures for tests."""

import pytest


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
