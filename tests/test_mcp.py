"""Tests for MCP client implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from threads_multiagent.mcp.client import BearerAuth, MCPClient


class TestBearerAuth:
    """Tests for BearerAuth class."""

    def test_auth_flow(self):
        """Test that auth flow adds bearer token to request."""
        auth = BearerAuth("test-token")
        request = httpx.Request("GET", "https://example.com")

        # Get the generator
        flow = auth.auth_flow(request)

        # First next() should yield the modified request
        modified_request = next(flow)

        assert "Authorization" in modified_request.headers
        assert modified_request.headers["Authorization"] == "Bearer test-token"


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_initialization(self):
        """Test client initialization."""
        with patch("threads_multiagent.mcp.client.Client"):
            client = MCPClient(
                server_url="https://mcp.example.com",
                bearer_token="test-token",
                timeout=60.0,
            )

            assert client._server_url == "https://mcp.example.com"
            assert client._bearer_token == "test-token"
            assert client._timeout == 60.0
            assert client._tools is None

    def test_initialization_without_token(self):
        """Test client initialization without bearer token."""
        with patch("threads_multiagent.mcp.client.Client"):
            client = MCPClient(
                server_url="https://mcp.example.com",
            )

            assert client._bearer_token is None

    def test_get_tools_for_openai_empty(self):
        """Test get_tools_for_openai when no tools loaded."""
        with patch("threads_multiagent.mcp.client.Client"):
            client = MCPClient(server_url="https://mcp.example.com")
            tools = client.get_tools_for_openai()
            assert tools == []

    def test_get_tools_for_openai_with_tools(self):
        """Test get_tools_for_openai with loaded tools."""
        with patch("threads_multiagent.mcp.client.Client"):
            client = MCPClient(server_url="https://mcp.example.com")
            client._tools = [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {"type": "object", "properties": {"arg": {"type": "string"}}},
                }
            ]

            tools = client.get_tools_for_openai()

            assert len(tools) == 1
            assert tools[0]["type"] == "function"
            assert tools[0]["function"]["name"] == "test_tool"
            assert tools[0]["function"]["description"] == "A test tool"

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools from MCP server."""
        mock_tool = MagicMock()
        mock_tool.name = "threads_post"
        mock_tool.description = "Post to Threads"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        mock_client_instance = MagicMock()
        mock_client_instance.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("threads_multiagent.mcp.client.Client", return_value=mock_client_instance):
            client = MCPClient(server_url="https://mcp.example.com")
            tools = await client.list_tools()

            assert len(tools) == 1
            assert tools[0]["name"] == "threads_post"
            assert tools[0]["description"] == "Post to Threads"

    @pytest.mark.asyncio
    async def test_list_tools_cached(self):
        """Test that tools are cached after first call."""
        mock_client_instance = MagicMock()
        mock_client_instance.list_tools = AsyncMock()

        with patch("threads_multiagent.mcp.client.Client", return_value=mock_client_instance):
            client = MCPClient(server_url="https://mcp.example.com")
            client._tools = [{"name": "cached_tool", "description": "Cached", "inputSchema": {}}]

            tools = await client.list_tools()

            assert tools[0]["name"] == "cached_tool"
            # Should not call list_tools again
            mock_client_instance.list_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling a tool on MCP server."""
        mock_content_item = MagicMock()
        mock_content_item.text = '{"id": "123", "text": "Posted!"}'

        mock_result = MagicMock()
        mock_result.content = [mock_content_item]

        mock_client_instance = MagicMock()
        mock_client_instance.call_tool = AsyncMock(return_value=mock_result)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("threads_multiagent.mcp.client.Client", return_value=mock_client_instance):
            client = MCPClient(server_url="https://mcp.example.com")
            result = await client.call_tool("threads_post", {"text": "Hello"})

            assert '{"id": "123"' in result
            mock_client_instance.call_tool.assert_called_once_with(
                "threads_post", {"text": "Hello"}
            )

    @pytest.mark.asyncio
    async def test_call_tool_no_content(self):
        """Test calling a tool that returns no content."""
        mock_result = MagicMock()
        mock_result.content = []

        mock_client_instance = MagicMock()
        mock_client_instance.call_tool = AsyncMock(return_value=mock_result)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("threads_multiagent.mcp.client.Client", return_value=mock_client_instance):
            client = MCPClient(server_url="https://mcp.example.com")
            result = await client.call_tool("some_tool", {})

            # Should return string representation of result
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_call_tool_without_arguments(self):
        """Test calling a tool without arguments."""
        mock_result = MagicMock()
        mock_result.content = []

        mock_client_instance = MagicMock()
        mock_client_instance.call_tool = AsyncMock(return_value=mock_result)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("threads_multiagent.mcp.client.Client", return_value=mock_client_instance):
            client = MCPClient(server_url="https://mcp.example.com")
            await client.call_tool("get_user_info")

            mock_client_instance.call_tool.assert_called_once_with("get_user_info", {})
