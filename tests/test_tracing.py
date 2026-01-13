"""Tests for Langfuse tracing module."""

from unittest.mock import MagicMock, patch

import pytest

from threads_hype_agent.tracing.langfuse_tracer import LangfuseTracer, get_tracer


class TestLangfuseTracer:
    """Tests for LangfuseTracer class."""

    def test_tracer_disabled_without_keys(self):
        """Test tracer is disabled when keys are not provided."""
        tracer = LangfuseTracer(
            secret_key=None,
            public_key=None,
        )
        assert tracer.enabled is False

    def test_tracer_disabled_with_partial_keys(self):
        """Test tracer is disabled with only partial keys."""
        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key=None,
        )
        assert tracer.enabled is False

    def test_tracer_disabled_explicitly(self):
        """Test tracer can be explicitly disabled."""
        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
            enabled=False,
        )
        assert tracer.enabled is False

    def test_callback_handler_none_when_disabled(self):
        """Test callback handler returns None when disabled."""
        tracer = LangfuseTracer(enabled=False)
        handler = tracer.get_callback_handler()
        assert handler is None

    @patch("threads_hype_agent.tracing.langfuse_tracer.Langfuse")
    def test_tracer_enabled_with_keys(self, mock_langfuse):
        """Test tracer is enabled when keys are provided."""
        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
            host="https://test.langfuse.com",
        )
        assert tracer.enabled is True
        mock_langfuse.assert_called_once_with(
            secret_key="sk-test",
            public_key="pk-test",
            host="https://test.langfuse.com",
        )

    @patch("threads_hype_agent.tracing.langfuse_tracer.CallbackHandler")
    @patch("threads_hype_agent.tracing.langfuse_tracer.Langfuse")
    def test_get_callback_handler(self, mock_langfuse, mock_handler):
        """Test getting callback handler."""
        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
        )
        handler = tracer.get_callback_handler()

        assert handler is not None
        mock_handler.assert_called_once_with()

    @patch("threads_hype_agent.tracing.langfuse_tracer.Langfuse")
    def test_build_config_metadata(self, mock_langfuse):
        """Test building config metadata for SDK v3."""
        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
        )
        metadata = tracer.build_config_metadata(
            user_id="test-user",
            session_id="test-session",
            tags=["custom-tag"],
            metadata={"key": "value"},
        )

        assert metadata["langfuse_user_id"] == "test-user"
        assert metadata["langfuse_session_id"] == "test-session"
        assert "custom-tag" in metadata["langfuse_tags"]
        assert "langgraph" in metadata["langfuse_tags"]
        assert "threads-hype-agent" in metadata["langfuse_tags"]
        assert metadata["key"] == "value"

    @patch("threads_hype_agent.tracing.langfuse_tracer.Langfuse")
    def test_build_config_metadata_minimal(self, mock_langfuse):
        """Test building config metadata with minimal args."""
        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
        )
        metadata = tracer.build_config_metadata()

        assert "langfuse_user_id" not in metadata
        assert "langfuse_session_id" not in metadata
        assert "langgraph" in metadata["langfuse_tags"]
        assert "threads-hype-agent" in metadata["langfuse_tags"]

    @patch("threads_hype_agent.tracing.langfuse_tracer.get_client")
    @patch("threads_hype_agent.tracing.langfuse_tracer.Langfuse")
    def test_flush(self, mock_langfuse, mock_get_client):
        """Test flushing traces."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
        )
        tracer.flush()

        mock_client.flush.assert_called_once()

    @patch("threads_hype_agent.tracing.langfuse_tracer.get_client")
    @patch("threads_hype_agent.tracing.langfuse_tracer.Langfuse")
    def test_shutdown(self, mock_langfuse, mock_get_client):
        """Test shutting down tracer."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tracer = LangfuseTracer(
            secret_key="sk-test",
            public_key="pk-test",
        )
        tracer.shutdown()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_returns_instance(self):
        """Test get_tracer returns a tracer instance."""
        # Clear cache
        get_tracer.cache_clear()
        tracer = get_tracer()
        assert isinstance(tracer, LangfuseTracer)

    def test_get_tracer_cached(self):
        """Test get_tracer returns cached instance."""
        get_tracer.cache_clear()
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2
