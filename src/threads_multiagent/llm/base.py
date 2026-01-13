"""Base LLM client abstraction."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from threads_multiagent.models.messages import Message


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers.

    All LLM client implementations must inherit from this class
    and implement the required methods.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name/identifier of the model being used."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a completion from messages."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from messages."""
        ...

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[Message, list[dict[str, Any]]]:
        """Generate a completion with tool calling support."""
        ...

    @abstractmethod
    async def responses_create(
        self,
        input: str,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a response using OpenAI Responses API with MCP support."""
        ...
