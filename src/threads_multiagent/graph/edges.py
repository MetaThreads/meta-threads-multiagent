"""Conditional edge functions for LangGraph workflow."""

from typing import Literal

from threads_multiagent.graph.state import AgentState
from threads_multiagent.logging import get_logger

logger = get_logger(__name__)


def route_from_orchestrator(
    state: AgentState,
) -> Literal["threads", "web_search", "response"]:
    """Route from orchestrator to the next agent.

    This function is called after the orchestrator node
    to determine which agent should execute next.

    Args:
        state: Current workflow state.

    Returns:
        Next agent identifier or "response" to generate final output.
    """
    next_agent = state.get("next_agent")

    if next_agent == "threads":
        logger.info("Routing to threads agent")
        return "threads"
    elif next_agent == "web_search":
        logger.info("Routing to web_search agent")
        return "web_search"
    else:
        logger.info("Routing to response agent")
        return "response"


def should_continue(state: AgentState) -> bool:
    """Check if the workflow should continue.

    Args:
        state: Current workflow state.

    Returns:
        True if workflow should continue, False otherwise.
    """
    # Check for error
    if state.get("error"):
        logger.warning(f"Workflow stopping due to error: {state['error']}")
        return False

    # Check if plan exists and is complete
    plan = state.get("plan")
    if plan:
        all_complete = all(step["completed"] for step in plan.get("steps", []))
        if all_complete:
            logger.info("Plan complete, workflow should end")
            return False

    return True
