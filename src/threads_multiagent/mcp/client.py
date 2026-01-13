"""MCP client for FastMCP server connection."""

from collections.abc import Generator
from typing import Any

import httpx
from fastmcp import Client

from threads_multiagent.logging import get_logger

logger = get_logger(__name__)


class BearerAuth(httpx.Auth):
    """Bearer token authentication for httpx."""

    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class MCPClient:
    """Client for connecting to MCP servers.

    Uses FastMCP Client for proper protocol handling.
    """

    def __init__(
        self,
        server_url: str,
        bearer_token: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize the MCP client.

        Args:
            server_url: URL of the MCP server.
            bearer_token: Optional bearer token for authentication.
            timeout: Request timeout in seconds.
        """
        self._server_url = server_url
        self._bearer_token = bearer_token
        self._timeout = timeout
        self._tools: list[dict[str, Any]] | None = None

        # Create auth if bearer token provided
        auth = BearerAuth(bearer_token) if bearer_token else None

        self._client = Client(server_url, auth=auth, timeout=timeout)

        logger.info(f"Initialized MCP client for: {server_url}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server.

        Returns:
            List of tool definitions.
        """
        if self._tools is not None:
            return self._tools

        async with self._client:
            tools = await self._client.list_tools()
            self._tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {"type": "object", "properties": {}},
                }
                for tool in tools
            ]
            logger.info(f"Listed {len(self._tools)} tools from MCP server")
            return self._tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        async with self._client:
            result = await self._client.call_tool(tool_name, arguments or {})
            logger.info(f"Called tool: {tool_name}")

            # Extract content from CallToolResult
            if hasattr(result, "content") and result.content:
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                    else:
                        texts.append(str(item))
                return "\n".join(texts)

            return str(result)

    def get_tools_for_openai(self) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format.

        Returns:
            Tools in OpenAI format.
        """
        if self._tools is None:
            return []

        openai_tools = []
        for tool in self._tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return openai_tools
