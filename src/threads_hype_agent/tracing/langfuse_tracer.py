"""Langfuse tracing for LangGraph workflows (SDK v3 compatible)."""

from functools import lru_cache
from typing import Any

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from threads_hype_agent.config import get_settings
from threads_hype_agent.logging import get_logger

logger = get_logger(__name__)


class LangfuseTracer:
    """Langfuse tracer for LangGraph workflow execution.

    Uses the native Langfuse CallbackHandler for LangChain/LangGraph
    integration to automatically capture traces of workflow execution.

    In SDK v3, trace attributes (user_id, session_id, tags) are passed
    via config metadata rather than constructor arguments.
    """

    def __init__(
        self,
        secret_key: str | None = None,
        public_key: str | None = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
    ):
        """Initialize the Langfuse tracer.

        Args:
            secret_key: Langfuse secret key.
            public_key: Langfuse public key.
            host: Langfuse host URL.
            enabled: Whether tracing is enabled.
        """
        self._enabled = enabled and secret_key is not None and public_key is not None
        self._client: Langfuse | None = None

        if self._enabled:
            # Initialize the Langfuse singleton
            self._client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host,
            )
            logger.info("Langfuse tracing enabled")
        else:
            logger.info("Langfuse tracing disabled")

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def get_callback_handler(self) -> CallbackHandler | None:
        """Get a Langfuse CallbackHandler for LangGraph tracing.

        Returns:
            CallbackHandler for use with LangGraph, or None if disabled.
        """
        if not self._enabled:
            return None

        handler = CallbackHandler()
        logger.debug("Created Langfuse callback handler")
        return handler

    def build_config_metadata(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build config metadata for LangGraph invocation.

        In SDK v3, trace attributes are passed via config metadata
        using special langfuse_ prefixed keys.

        Args:
            user_id: Optional user identifier.
            session_id: Optional session identifier.
            tags: Optional list of tags.
            metadata: Optional additional metadata.

        Returns:
            Metadata dict for use in LangGraph config.
        """
        default_tags = ["langgraph", "threads-hype-agent"]
        all_tags = (tags or []) + default_tags

        config_metadata: dict[str, Any] = metadata.copy() if metadata else {}

        if user_id:
            config_metadata["langfuse_user_id"] = user_id
        if session_id:
            config_metadata["langfuse_session_id"] = session_id
        if all_tags:
            config_metadata["langfuse_tags"] = all_tags

        return config_metadata

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._enabled:
            client = get_client()
            client.flush()
            logger.debug("Flushed Langfuse client")

    def shutdown(self) -> None:
        """Shutdown the tracer and flush pending data."""
        if self._enabled:
            client = get_client()
            client.flush()
            client.shutdown()
            logger.info("Langfuse tracer shutdown")


@lru_cache
def get_tracer() -> LangfuseTracer:
    """Get cached Langfuse tracer instance.

    Returns:
        Configured LangfuseTracer instance.
    """
    settings = get_settings()
    return LangfuseTracer(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
        enabled=settings.langfuse_enabled,
    )
