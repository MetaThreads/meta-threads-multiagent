"""Response agent implementation with tool-based paradigm."""

import json
from typing import TYPE_CHECKING

from langfuse import observe

from threads_hype_agent.agents.base import BaseAgent
from threads_hype_agent.exceptions import AgentError
from threads_hype_agent.logging import get_logger
from threads_hype_agent.models.messages import Message
from threads_hype_agent.prompts.response import RESPONSE_PROMPT

if TYPE_CHECKING:
    from threads_hype_agent.graph.state import AgentState

logger = get_logger(__name__)


class ResponseAgent(BaseAgent):
    """Response agent that generates human-readable responses.

    Takes workflow results and creates a clear, conversational
    response for the user.
    All operations are exposed as tools with Langfuse tracing.
    """

    @property
    def name(self) -> str:
        return "response"

    @property
    def description(self) -> str:
        return "Generates human-readable responses from workflow results"

    def get_system_prompt(self) -> str:
        return RESPONSE_PROMPT

    @observe(as_type="tool", name="build_response_context")
    def _tool_build_context(self, state: "AgentState") -> str:
        """Tool: Build context for response generation.

        Args:
            state: Current workflow state.

        Returns:
            Context string for LLM.
        """
        parts = []

        user_request = ""
        for msg in state.get("messages", []):
            if msg["role"] == "user":
                user_request = msg["content"]
                break

        parts.append(f"User's request: {user_request}")

        plan = state.get("plan")
        if plan:
            parts.append(f"\nGoal: {plan.get('goal', 'Unknown')}")

        web_results = state.get("web_search_results", [])
        if web_results:
            search_summary = "\n".join([
                f"- {result['title']} ({result['source']}): {result['snippet'][:100]}..."
                for result in web_results[:5]
            ])
            parts.append(f"\nWeb search results:\n{search_summary}")

        threads_results = state.get("threads_results", [])
        if threads_results:
            parts.append("\nThreads operations:")
            for result in threads_results:
                action = result.get("action", "Unknown action")
                data = result.get("result", "")
                try:
                    if isinstance(data, str) and data.startswith("{"):
                        parsed = json.loads(data)
                        data = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
                parts.append(f"\nAction: {action}\nResult:\n{data}")

        parts.append("\n\nPlease create a clear, human-readable response for the user.")

        return "\n".join(parts)

    @observe(as_type="tool", name="generate_response")
    async def _tool_generate_response(self, context: str) -> str:
        """Tool: Generate human-readable response using LLM.

        Args:
            context: Response context.

        Returns:
            Generated response.
        """
        messages = [
            Message(role="system", content=self.get_system_prompt()),
            Message(role="user", content=context),
        ]

        response = await self.llm.complete(messages, temperature=0.7)
        return response.content

    async def invoke(self, state: "AgentState") -> "AgentState":
        """Generate a human-readable response.

        Args:
            state: Current workflow state with results from other agents.

        Returns:
            Updated state with human-readable output.

        Raises:
            AgentError: If response generation fails.
        """
        logger.info("Response agent invoked")

        try:
            # Tool 1: Build context from state
            context = self._tool_build_context(state)

            # Tool 2: Generate response using LLM
            output = await self._tool_generate_response(context)

            logger.info("Generated human-readable response")

            # Update state
            new_state = state.copy()
            new_state["output"] = output
            new_state["messages"] = state["messages"] + [
                {"role": "assistant", "content": output}
            ]

            return new_state

        except Exception as e:
            logger.error(f"Response agent failed: {e}")
            raise AgentError(f"Response agent failed: {e}") from e
