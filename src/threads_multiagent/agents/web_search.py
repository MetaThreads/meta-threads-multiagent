"""Web search agent implementation with tool-based paradigm."""

from typing import TYPE_CHECKING, Any, cast

from langfuse import observe

from threads_multiagent.agents.base import BaseAgent
from threads_multiagent.exceptions import AgentError
from threads_multiagent.logging import get_logger
from threads_multiagent.models.agents import Plan
from threads_multiagent.models.messages import Message
from threads_multiagent.prompts.web_search import QUERY_GENERATION_PROMPT, SYNTHESIS_PROMPT
from threads_multiagent.search.base import BaseWebSearch

if TYPE_CHECKING:
    from threads_multiagent.graph.state import AgentState, PlanDict

logger = get_logger(__name__)


class WebSearchAgent(BaseAgent):
    """Web search agent that autonomously searches and synthesizes information.

    Uses LLM to generate optimal search queries and synthesize results.
    All operations are exposed as tools with Langfuse tracing.
    """

    def __init__(
        self,
        llm: Any,
        web_search: BaseWebSearch,
    ):
        """Initialize the web search agent.

        Args:
            llm: LLM client for processing.
            web_search: Web search implementation.
        """
        super().__init__(llm)
        self.web_search = web_search

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for information and synthesizes results"

    def get_system_prompt(self) -> str:
        return SYNTHESIS_PROMPT

    @observe(as_type="tool", name="generate_search_query")
    async def _tool_generate_query(
        self,
        user_request: str,
        goal: str,
        current_action: str,
    ) -> str:
        """Tool: Generate an optimal search query using LLM.

        Args:
            user_request: Original user request.
            goal: Overall goal from plan.
            current_action: Current task direction.

        Returns:
            Generated search query.
        """
        context_parts = [f"User's request: {user_request}"]
        if goal:
            context_parts.append(f"Overall goal: {goal}")
        if current_action:
            context_parts.append(f"Current task direction: {current_action}")

        context = "\n".join(context_parts)

        messages = [
            Message(role="system", content=QUERY_GENERATION_PROMPT),
            Message(role="user", content=context),
        ]

        response = await self.llm.complete(messages, temperature=0.3)
        query = response.content.strip().strip('"').strip("'")
        logger.info(f"Generated search query: {query}")
        return query

    @observe(as_type="tool", name="web_search")
    async def _tool_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Tool: Execute web search.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of search result dicts.
        """
        results = await self.web_search.search(query=query, limit=limit)
        logger.info(f"Found {len(results)} search results")

        return [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "source": r.source,
            }
            for r in results
        ]

    @observe(as_type="tool", name="synthesize_results")
    async def _tool_synthesize(
        self,
        results: list[dict[str, Any]],
        user_request: str,
        goal: str,
    ) -> str:
        """Tool: Synthesize search results into a summary.

        Args:
            results: List of search result dicts.
            user_request: Original user request.
            goal: Overall goal from plan.

        Returns:
            Synthesized summary.
        """
        if not results:
            return "No relevant search results found for this query."

        # Build results text
        results_text = "\n\n".join(
            f"**{r['title']}** ({r['source']})\n{r['snippet']}\nURL: {r['url']}" for r in results
        )

        # Build context
        context_parts = [f"User's request: {user_request}"]
        if goal:
            context_parts.append(f"Goal: {goal}")
        context_parts.append(f"\nSearch results:\n{results_text}")
        context_parts.append("\nPlease synthesize these results to address the user's needs.")

        messages = [
            Message(role="system", content=SYNTHESIS_PROMPT),
            Message(role="user", content="\n".join(context_parts)),
        ]

        response = await self.llm.complete(messages, temperature=0.5)
        return response.content

    async def invoke(self, state: "AgentState") -> "AgentState":
        """Search the web based on user's request and context.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with search results.

        Raises:
            AgentError: If web search fails.
        """
        logger.info("Web search agent invoked")

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

            # Tool 1: Generate search query
            query = await self._tool_generate_query(user_request, goal, current_action)

            # Tool 2: Execute web search
            results_data = await self._tool_search(query, limit=5)

            # Tool 3: Synthesize results
            synthesis = await self._tool_synthesize(results_data, user_request, goal)

            # Update state
            new_state = state.copy()
            new_state["web_search_results"] = results_data

            if plan_data:
                plan = Plan(**plan_data)
                plan.mark_current_step_completed(result=synthesis)
                new_state["plan"] = cast("PlanDict", plan.model_dump())

            new_state["messages"] = state["messages"] + [
                {"role": "assistant", "content": synthesis}
            ]

            return new_state

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise AgentError(f"Web search agent failed: {e}") from e
