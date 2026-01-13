"""OpenRouter LLM client implementation using OpenAI interface.

Uses Langfuse's OpenAI wrapper for automatic token usage and cost tracking.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from langfuse.openai import AsyncOpenAI  # type: ignore[attr-defined]
from openai import APIConnectionError, APIStatusError, RateLimitError

from threads_multiagent.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from threads_multiagent.llm.base import BaseLLMClient
from threads_multiagent.logging import get_logger
from threads_multiagent.models.messages import Message

logger = get_logger(__name__)


class OpenRouterClient(BaseLLMClient):
    """OpenRouter LLM client using OpenAI-compatible interface.

    OpenRouter provides access to multiple LLM providers through
    a unified API compatible with OpenAI's interface.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            model: Model identifier (e.g., "anthropic/claude-sonnet-4").
            base_url: OpenRouter API base URL.
        """
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/MetaThreads/meta-threads-hype-agent",
                "X-Title": "Threads Hype Agent",
            },
        )

        logger.info(f"Initialized OpenRouter client with model: {model}")

    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def client(self) -> AsyncOpenAI:
        """Get the async OpenAI client for direct access."""
        return self._client

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a completion from messages."""
        try:
            openai_messages = [msg.to_openai_format() for msg in messages]

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            logger.debug(f"Completion generated: {len(content)} chars")
            return Message(role="assistant", content=content)

        except APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise LLMConnectionError(f"Failed to connect to OpenRouter: {e}") from e
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise LLMRateLimitError(f"OpenRouter rate limit exceeded: {e}") from e
        except APIStatusError as e:
            logger.error(f"API error: {e}")
            raise LLMResponseError(f"OpenRouter API error: {e}") from e

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from messages."""
        try:
            openai_messages = [msg.to_openai_format() for msg in messages]

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            # Response is AsyncStream when stream=True
            async for chunk in response:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except APIConnectionError as e:
            logger.error(f"Connection error during streaming: {e}")
            raise LLMConnectionError(f"Failed to connect to OpenRouter: {e}") from e
        except RateLimitError as e:
            logger.error(f"Rate limit error during streaming: {e}")
            raise LLMRateLimitError(f"OpenRouter rate limit exceeded: {e}") from e
        except APIStatusError as e:
            logger.error(f"API error during streaming: {e}")
            raise LLMResponseError(f"OpenRouter API error: {e}") from e

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[Message, list[dict[str, Any]]]:
        """Generate a completion with tool calling support."""
        try:
            openai_messages = [msg.to_openai_format() for msg in messages]

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,  # type: ignore
                tools=tools,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            message = Message(role="assistant", content=content)

            # Extract tool calls if present
            tool_calls: list[dict[str, Any]] = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    # Only process function-type tool calls
                    if hasattr(tc, "function") and tc.function is not None:
                        tool_calls.append(
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments),
                            }
                        )
                logger.debug(f"Tool calls extracted: {[tc['name'] for tc in tool_calls]}")

            return message, tool_calls

        except APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise LLMConnectionError(f"Failed to connect to OpenRouter: {e}") from e
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise LLMRateLimitError(f"OpenRouter rate limit exceeded: {e}") from e
        except APIStatusError as e:
            logger.error(f"API error: {e}")
            raise LLMResponseError(f"OpenRouter API error: {e}") from e

    async def responses_create(
        self,
        input: str,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a response using OpenAI Responses API with MCP support.

        This is the FastMCP-style API that supports MCP tools directly.

        Args:
            input: The user input/prompt.
            tools: List of tools including MCP server configs.
            **kwargs: Additional parameters.

        Returns:
            Response from the API.
        """
        try:
            logger.info(f"Creating response with input: {input[:50]}...")

            response = await self._client.responses.create(
                model=self._model,
                input=input,
                tools=tools,  # type: ignore[arg-type]
                **kwargs,
            )

            logger.debug("Response created successfully")
            return response

        except APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise LLMConnectionError(f"Failed to connect to OpenRouter: {e}") from e
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise LLMRateLimitError(f"OpenRouter rate limit exceeded: {e}") from e
        except APIStatusError as e:
            logger.error(f"API error: {e}")
            raise LLMResponseError(f"OpenRouter API error: {e}") from e
