"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from threads_hype_agent.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_from_env(self):
        """Test loading settings from environment."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-key",
            "THREADS_BEARER_TOKEN": "test-token:12345",
        }):
            # Clear the cache to get fresh settings
            get_settings.cache_clear()
            settings = Settings()
            assert settings.openrouter_api_key == "test-key"
            assert settings.threads_bearer_token == "test-token:12345"

    def test_default_values(self):
        """Test default setting values."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-key",
            "THREADS_BEARER_TOKEN": "test-token:12345",
        }):
            settings = Settings()
            assert settings.openrouter_model == "google/gemini-2.5-flash-lite"
            assert settings.api_port == 8000
            assert settings.log_level == "INFO"
            assert settings.max_agent_iterations == 30
            assert settings.langfuse_host == "https://cloud.langfuse.com"
            # Note: langfuse_enabled may be overridden by test fixtures

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
