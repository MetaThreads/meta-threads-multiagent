"""Graph state definitions for LangGraph workflow."""

from typing import Any, Literal, TypedDict


class PlanStepDict(TypedDict):
    """Dictionary representation of a plan step."""

    agent: Literal["threads", "web_search"]
    action: str
    completed: bool
    result: str | None


class PlanDict(TypedDict):
    """Dictionary representation of an execution plan."""

    goal: str
    steps: list[PlanStepDict]
    current_step_index: int


class AgentState(TypedDict, total=False):
    """State passed between agents in the LangGraph workflow.

    This TypedDict defines the shape of state that flows
    through the workflow graph.
    """

    # Conversation messages in standard format
    messages: list[dict[str, str]]

    # Execution plan created by planning agent
    plan: PlanDict | None

    # Web search results fetched by web_search agent
    web_search_results: list[dict[str, Any]]

    # Results from Threads MCP operations
    threads_results: list[dict[str, Any]]

    # Current action being executed
    current_action: str

    # Next agent to route to (set by orchestrator)
    next_agent: Literal["threads", "web_search", "response"] | None

    # Error information if something fails
    error: str | None

    # Final output summary (populated when workflow completes)
    output: str | None


def create_initial_state(messages: list[dict[str, str]]) -> AgentState:
    """Create initial state for a new workflow run.

    Args:
        messages: Initial conversation messages.

    Returns:
        Initialized AgentState.
    """
    return AgentState(
        messages=messages,
        plan=None,
        web_search_results=[],
        threads_results=[],
        current_action="",
        next_agent=None,
        error=None,
        output=None,
    )
