"""Orchestrator agent implementation with tool-based paradigm."""

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from langfuse import observe

from threads_hype_agent.agents.base import BaseAgent
from threads_hype_agent.exceptions import OrchestrationError
from threads_hype_agent.logging import get_logger
from threads_hype_agent.models.agents import Plan, PlanStep
from threads_hype_agent.models.messages import Message
from threads_hype_agent.prompts.orchestrator import ORCHESTRATOR_PROMPT

if TYPE_CHECKING:
    from threads_hype_agent.graph.state import AgentState, PlanDict

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that uses LLM to evaluate outputs and route tasks.

    This agent evaluates the results from previous steps, decides whether
    to continue, modify the plan, or complete the workflow.
    All operations are exposed as tools with Langfuse tracing.
    """

    @property
    def name(self) -> str:
        return "orchestrator"

    @property
    def description(self) -> str:
        return "Evaluates agent outputs and routes tasks based on LLM decisions"

    def get_system_prompt(self) -> str:
        return ORCHESTRATOR_PROMPT

    @observe(as_type="tool", name="build_evaluation_context")
    def _tool_build_context(self, state: "AgentState", plan: Plan) -> str:
        """Tool: Build context for LLM evaluation.

        Args:
            state: Current workflow state.
            plan: Current execution plan.

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
        parts.append(f"\nGoal: {plan.goal}")

        parts.append("\nExecution Plan:")
        for i, step in enumerate(plan.steps):
            status = "COMPLETED" if step.completed else "PENDING"
            parts.append(f"  {i+1}. [{status}] {step.agent}: {step.action}")
            if step.result:
                parts.append(f"      Result: {step.result[:200]}...")

        last_completed = self._get_last_completed_step(plan)
        if last_completed:
            parts.append(f"\nLast completed step: {last_completed.agent} - {last_completed.action}")

            if last_completed.agent == "web_search":
                web_results = state.get("web_search_results", [])
                if web_results:
                    parts.append("Web search results:")
                    for result in web_results[:3]:
                        parts.append(f"  - {result.get('title', 'No title')}: {result.get('snippet', '')[:100]}...")
                else:
                    parts.append("Web search results: No results found")

            elif last_completed.agent == "threads":
                threads_results = state.get("threads_results", [])
                if threads_results:
                    parts.append("Threads results:")
                    for result in threads_results[-1:]:
                        action = result.get("action", "Unknown")
                        data = result.get("result", "")
                        if isinstance(data, str) and len(data) > 300:
                            data = data[:300] + "..."
                        parts.append(f"  Action: {action}")
                        parts.append(f"  Result: {data}")
                else:
                    parts.append("Threads results: No results")
        else:
            parts.append("\nNo steps completed yet - this is the first evaluation.")

        error = state.get("error")
        if error:
            parts.append(f"\nError occurred: {error}")

        next_step = plan.get_next_incomplete_step()
        if next_step:
            parts.append(f"\nNext pending step: {next_step.agent} - {next_step.action}")
        else:
            parts.append("\nAll steps completed.")

        return "\n".join(parts)

    def _get_last_completed_step(self, plan: Plan) -> PlanStep | None:
        """Get the last completed step from the plan."""
        last_completed = None
        for step in plan.steps:
            if step.completed:
                last_completed = step
        return last_completed

    @observe(as_type="tool", name="evaluate_with_llm")
    async def _tool_evaluate(self, context: str) -> str:
        """Tool: Use LLM to evaluate the current state.

        Args:
            context: Evaluation context.

        Returns:
            Raw LLM response.
        """
        messages = [
            Message(role="system", content=self.get_system_prompt()),
            Message(role="user", content=context),
        ]

        response = await self.llm.complete(messages, temperature=0.3)
        return response.content

    @observe(as_type="tool", name="parse_decision")
    def _tool_parse_decision(self, response: str) -> dict[str, Any]:
        """Tool: Parse LLM decision response.

        Args:
            response: Raw LLM response.

        Returns:
            Parsed decision dictionary.
        """
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            decision: dict[str, Any] = json.loads(response)
            return decision
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse orchestrator decision: {e}")
            return {
                "evaluation": "Failed to parse LLM response",
                "decision": "continue",
                "reasoning": "Defaulting to continue due to parse error",
                "modifications": [],
            }

    @observe(as_type="tool", name="process_decision")
    def _tool_process_decision(
        self, state: "AgentState", plan: Plan, decision: dict[str, Any]
    ) -> "AgentState":
        """Tool: Process the LLM decision and update state.

        Args:
            state: Current workflow state.
            plan: Current execution plan.
            decision: Parsed LLM decision.

        Returns:
            Updated state.
        """
        new_state = state.copy()
        decision_type = decision.get("decision", "continue")

        if decision_type == "complete":
            logger.info("Orchestrator decided to complete workflow")
            new_state["next_agent"] = "response"
            new_state["plan"] = cast("PlanDict", plan.model_dump())
            return new_state

        if decision_type == "modify":
            logger.info("Orchestrator decided to modify plan")
            modifications = decision.get("modifications", [])

            if modifications:
                new_steps = []
                for mod in modifications:
                    agent = mod.get("agent")
                    action = mod.get("action", "").lower()

                    communication_keywords = ["inform the user", "tell the user", "explain to", "notify the user"]
                    is_communication = any(kw in action for kw in communication_keywords)

                    if is_communication:
                        logger.warning(
                            f"Skipping communication modification: '{mod.get('action')}'. "
                            f"User communication is handled by response agent."
                        )
                        continue

                    if agent in ("web_search", "threads") and mod.get("action"):
                        new_steps.append(PlanStep(agent=agent, action=mod.get("action")))

                if new_steps:
                    insert_idx = 0
                    for i, step in enumerate(plan.steps):
                        if step.completed:
                            insert_idx = i + 1

                    for i, new_step in enumerate(new_steps):
                        plan.steps.insert(insert_idx + i, new_step)

                    logger.info(f"Added {len(new_steps)} new steps to plan")
                elif modifications:
                    logger.info("All modifications were user-communication. Completing workflow.")
                    new_state["next_agent"] = "response"
                    new_state["plan"] = cast("PlanDict", plan.model_dump())
                    return new_state

        next_step = plan.get_next_incomplete_step()
        if next_step:
            new_state["next_agent"] = next_step.agent
            new_state["current_action"] = next_step.action
            logger.info(f"Routing to agent: {next_step.agent} (from plan)")
        else:
            new_state["next_agent"] = "response"
            logger.info("No more steps, routing to response")

        new_state["plan"] = cast("PlanDict", plan.model_dump())
        return new_state

    async def invoke(self, state: "AgentState") -> "AgentState":
        """Use LLM to evaluate outputs and determine next action.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with next_agent set.

        Raises:
            OrchestrationError: If orchestration fails.
        """
        logger.info("Orchestrator agent invoked")

        plan_data = state.get("plan")
        if not plan_data:
            raise OrchestrationError("No plan found in state")

        plan = Plan(**plan_data)

        # Tool 1: Build evaluation context
        context = self._tool_build_context(state, plan)

        # Tool 2: Evaluate with LLM
        response = await self._tool_evaluate(context)

        # Tool 3: Parse decision
        decision = self._tool_parse_decision(response)

        logger.info(f"Orchestrator decision: {decision.get('decision')}")
        logger.debug(f"Evaluation: {decision.get('evaluation')}")
        logger.debug(f"Reasoning: {decision.get('reasoning')}")

        # Tool 4: Process decision
        return self._tool_process_decision(state, plan, decision)

    def get_next_agent(
        self, state: "AgentState"
    ) -> Literal["threads", "web_search", "response"]:
        """Get the next agent from state."""
        next_agent = state.get("next_agent", "response")
        if next_agent in ("threads", "web_search", "response"):
            return next_agent
        return "response"
