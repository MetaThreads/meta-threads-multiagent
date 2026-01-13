"""Threads agent implementation using MCP client with tool-based paradigm."""

import json
from typing import TYPE_CHECKING, Any, cast

from langfuse import observe

from threads_multiagent.agents.base import BaseAgent
from threads_multiagent.exceptions import AgentError
from threads_multiagent.logging import get_logger
from threads_multiagent.mcp.client import MCPClient
from threads_multiagent.models.agents import Plan
from threads_multiagent.models.messages import Message
from threads_multiagent.prompts.threads import THREADS_PROMPT

if TYPE_CHECKING:
    from threads_multiagent.graph.state import AgentState, PlanDict

logger = get_logger(__name__)


class ThreadsAgent(BaseAgent):
    """Threads agent that autonomously interacts with Meta Threads via MCP.

    Uses MCP client to connect to Threads MCP server and
    LLM function calling to decide which tools to use.
    All operations are exposed as tools with Langfuse tracing.
    """

    def __init__(
        self,
        llm: Any,
        mcp_server_url: str,
        bearer_token: str | None = None,
    ):
        """Initialize the Threads agent.

        Args:
            llm: LLM client (OpenRouterClient).
            mcp_server_url: URL of the Threads MCP server.
            bearer_token: Optional bearer token for MCP auth.
        """
        super().__init__(llm)
        self.mcp_client = MCPClient(
            server_url=mcp_server_url,
            bearer_token=bearer_token,
        )
        self._tools_loaded = False

    @property
    def name(self) -> str:
        return "threads"

    @property
    def description(self) -> str:
        return "Interacts with Meta Threads to create posts, replies, and manage content"

    def get_system_prompt(self) -> str:
        return THREADS_PROMPT

    @observe(as_type="tool", name="load_mcp_tools")
    async def _tool_load_mcp_tools(self) -> list[dict[str, Any]]:
        """Tool: Load available MCP tools from server.

        Returns:
            List of available tools in OpenAI format.
        """
        if not self._tools_loaded:
            await self.mcp_client.list_tools()
            self._tools_loaded = True

        tools = self.mcp_client.get_tools_for_openai()
        logger.info(f"Loaded {len(tools)} MCP tools")
        return tools

    def _get_tools_description(self) -> str:
        """Build dynamic tools description from loaded MCP tools."""
        tools = self.mcp_client._tools
        if not tools:
            return ""

        lines = ["\nAvailable tools:"]
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "").split("\n")[0]  # First line only
            schema = tool.get("inputSchema", {})
            params = schema.get("properties", {})

            # Build parameter list
            required = schema.get("required", [])
            param_strs = []
            for param_name in params:
                is_required = param_name in required
                suffix = "" if is_required else "?"
                param_strs.append(f"{param_name}{suffix}")

            params_str = ", ".join(param_strs) if param_strs else ""
            lines.append(f"- {name}({params_str}): {desc}")

        return "\n".join(lines)

    @observe(as_type="tool", name="decide_threads_action")
    async def _tool_decide_action(
        self,
        user_request: str,
        goal: str,
        current_action: str,
        web_results: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Message, list[dict[str, Any]]]:
        """Tool: Use LLM to decide which Threads action to take.

        Args:
            user_request: Original user request.
            goal: Overall goal from plan.
            current_action: Current task direction.
            web_results: Web search results if available.
            tools: Available MCP tools.

        Returns:
            Tuple of (response message, tool calls).
        """
        # Build context parts
        context_parts = [f"User's request: {user_request}"]

        if goal:
            context_parts.append(f"Overall goal: {goal}")

        if current_action:
            context_parts.append(f"Current task direction: {current_action}")

        # Include web search results if available
        if web_results:
            context_parts.append("\nResearch findings from web search:")
            for result in web_results[:3]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                context_parts.append(f"- {title}: {snippet[:150]}...")

        context_parts.append("\nPlease accomplish the user's goal using the available Threads tools.")

        user_content = "\n".join(context_parts)

        # Build system prompt with dynamic tools
        system_prompt = self.get_system_prompt() + self._get_tools_description()

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]

        # Call LLM with function calling
        response_msg, tool_calls = await self.llm.complete_with_tools(
            messages=messages,
            tools=tools,
            temperature=0.7,
        )

        logger.info(f"LLM decided on {len(tool_calls)} tool call(s)")
        return response_msg, tool_calls

    @observe(as_type="tool", name="execute_mcp_tool")
    async def _tool_execute_mcp(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Tool: Execute an MCP tool call.

        Args:
            tool_name: Name of the MCP tool.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        logger.info(f"Executing MCP tool: {tool_name}")

        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)

            # Format result
            if isinstance(result, dict):
                content = result.get("content", [])
                if content and isinstance(content, list):
                    text_items = [
                        item.get("text", str(item))
                        for item in content
                        if isinstance(item, dict)
                    ]
                    return "\n".join(text_items) if text_items else json.dumps(result, indent=2)
                return json.dumps(result, indent=2)
            return str(result)

        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return f"Error calling {tool_name}: {e}"

    async def invoke(self, state: "AgentState") -> "AgentState":
        """Execute Threads operations using MCP.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with Threads results.

        Raises:
            AgentError: If Threads operation fails.
        """
        logger.info("Threads agent invoked")

        plan_data = state.get("plan")

        try:
            # Extract context
            user_request = ""
            for msg in state.get("messages", []):
                if msg["role"] == "user":
                    user_request = msg["content"]
                    break

            goal = plan_data.get("goal", "") if plan_data else ""
            current_action = state.get("current_action", "")
            web_results = state.get("web_search_results", [])

            # Tool 1: Load MCP tools
            tools = await self._tool_load_mcp_tools()

            # Tool 2: Decide action using LLM
            response_msg, tool_calls = await self._tool_decide_action(
                user_request, goal, current_action, web_results, tools
            )

            # Tool 3: Execute MCP tool calls
            results = []
            for call in tool_calls:
                tool_name = call.get("name", "")
                arguments = call.get("arguments", {})
                result = await self._tool_execute_mcp(tool_name, arguments)
                results.append(result)

            # Combine results
            final_result = "\n\n".join(results) if results else response_msg.content or "No action taken"
            logger.info(f"Threads operation completed: {final_result[:100]}...")

            # Update state
            new_state = state.copy()

            if plan_data:
                plan = Plan(**plan_data)
                plan.mark_current_step_completed(result=final_result)
                new_state["plan"] = cast("PlanDict", plan.model_dump())

            # Add result to threads_results
            threads_results = state.get("threads_results", [])
            threads_results.append({"action": current_action, "result": final_result})
            new_state["threads_results"] = threads_results

            new_state["messages"] = state["messages"] + [
                {"role": "assistant", "content": final_result}
            ]

            return new_state

        except Exception as e:
            logger.error(f"Threads agent failed: {e}")
            raise AgentError(f"Threads agent failed: {e}") from e
