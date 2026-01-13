"""Tests for LLM module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from threads_multiagent.llm.base import BaseLLMClient
from threads_multiagent.llm.openrouter import OpenRouterClient
from threads_multiagent.models.messages import Message


class TestBaseLLMClient:
    """Tests for BaseLLMClient abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseLLMClient cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLMClient()


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = OpenRouterClient(
            api_key="test-key",
            model="test-model",
        )
        assert client.model_name == "test-model"

    def test_model_name_property(self):
        """Test model_name property."""
        client = OpenRouterClient(
            api_key="test-key",
            model="anthropic/claude-sonnet-4",
        )
        assert client.model_name == "anthropic/claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test completion method."""
        client = OpenRouterClient(api_key="test-key", model="test-model")

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            messages = [Message(role="user", content="Hello")]
            result = await client.complete(messages)

            assert result.role == "assistant"
            assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        """Test completion with tools."""
        client = OpenRouterClient(api_key="test-key", model="test-model")

        # Mock response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            messages = [Message(role="user", content="Use a tool")]
            tools = [{"type": "function", "function": {"name": "test_tool"}}]

            message, tool_calls = await client.complete_with_tools(messages, tools)

            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "test_tool"
            assert tool_calls[0]["arguments"] == {"arg": "value"}
