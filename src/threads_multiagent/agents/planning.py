"""Planning agent implementation with tool-based paradigm."""

import json
from typing import TYPE_CHECKING, cast

from langfuse import observe

from threads_multiagent.agents.base import BaseAgent
from threads_multiagent.exceptions import PlanningError
from threads_multiagent.logging import get_logger
from threads_multiagent.models.agents import Plan, PlanStep
from threads_multiagent.models.messages import Message
from threads_multiagent.prompts.planning import PLANNING_PROMPT

if TYPE_CHECKING:
    from threads_multiagent.graph.state import AgentState, PlanDict

logger = get_logger(__name__)


class PlanningAgent(BaseAgent):
    """Planning agent that analyzes user input and creates execution plans.

    This agent is called first in the workflow to break down
    user requests into actionable steps for other agents.
    All operations are exposed as tools with Langfuse tracing.
    """

    @property
    def name(self) -> str:
        return "planning"

    @property
    def description(self) -> str:
        return "Analyzes user requests and creates step-by-step execution plans"

    def get_system_prompt(self) -> str:
        return PLANNING_PROMPT

    @observe(as_type="tool", name="generate_plan")
    async def _tool_generate_plan(self, user_message: str) -> str:
        """Tool: Generate a plan using LLM.

        Args:
            user_message: User's request.

        Returns:
            Raw LLM response with plan.
        """
        messages = [
            Message(role="system", content=self.get_system_prompt()),
            Message(role="user", content=user_message),
        ]

        response = await self.llm.complete(messages, temperature=0.3)
        return response.content

    @observe(as_type="tool", name="parse_plan")
    def _tool_parse_plan(self, response: str, user_message: str) -> Plan:
        """Tool: Parse LLM response into a Plan object.

        Args:
            response: Raw LLM response.
            user_message: Original user request.

        Returns:
            Parsed Plan object.
        """
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                steps = []
                for step_data in data.get("steps", []):
                    agent = step_data.get("agent", "threads")
                    if agent not in ("web_search", "threads"):
                        agent = "threads"

                    steps.append(
                        PlanStep(
                            agent=agent,
                            action=step_data.get("action", ""),
                            completed=False,
                        )
                    )

                return Plan(
                    goal=data.get("goal", user_message),
                    steps=steps,
                    current_step_index=0,
                )

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response, using fallback")

        return self._create_fallback_plan(user_message, response)

    def _create_fallback_plan(self, user_message: str, _response: str) -> Plan:
        """Create a fallback plan when JSON parsing fails."""
        steps: list[PlanStep] = []

        search_keywords = [
            "news",
            "trending",
            "latest",
            "search",
            "find",
            "research",
            "information",
        ]
        if any(kw in user_message.lower() for kw in search_keywords):
            steps.append(
                PlanStep(
                    agent="web_search",
                    action="Search for relevant information",
                    completed=False,
                )
            )

        post_keywords = ["post", "share", "publish", "thread", "create"]
        if any(kw in user_message.lower() for kw in post_keywords):
            steps.append(
                PlanStep(
                    agent="threads",
                    action="Create and publish post on Threads",
                    completed=False,
                )
            )

        if not steps:
            steps.append(
                PlanStep(
                    agent="threads",
                    action="Process user request on Threads",
                    completed=False,
                )
            )

        return Plan(
            goal=user_message,
            steps=steps,
            current_step_index=0,
        )

    async def invoke(self, state: "AgentState") -> "AgentState":
        """Create an execution plan from user input.

        Args:
            state: Current workflow state with user messages.

        Returns:
            Updated state with execution plan.

        Raises:
            PlanningError: If plan creation fails.
        """
        logger.info("Planning agent invoked")

        user_message = None
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        if not user_message:
            raise PlanningError("No user message found in state")

        try:
            # Tool 1: Generate plan using LLM
            response = await self._tool_generate_plan(user_message)

            # Tool 2: Parse the response
            plan = self._tool_parse_plan(response, user_message)

            logger.info(f"Created plan with {len(plan.steps)} steps")

            # Update state
            new_state = state.copy()
            new_state["plan"] = cast("PlanDict", plan.model_dump())
            new_state["messages"] = state["messages"] + [
                {"role": "assistant", "content": f"Plan created: {plan.goal}"}
            ]

            return new_state

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise PlanningError(f"Failed to create plan: {e}") from e
