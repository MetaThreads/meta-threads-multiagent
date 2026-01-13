"""Graph node functions for LangGraph workflow."""

from typing import Any

from threads_hype_agent.agents.orchestrator import OrchestratorAgent
from threads_hype_agent.agents.planning import PlanningAgent
from threads_hype_agent.agents.response import ResponseAgent
from threads_hype_agent.agents.threads import ThreadsAgent
from threads_hype_agent.agents.web_search import WebSearchAgent
from threads_hype_agent.graph.state import AgentState
from threads_hype_agent.logging import get_logger

logger = get_logger(__name__)


async def planning_node(
    state: AgentState,
    planning_agent: PlanningAgent,
) -> AgentState:
    """Execute the planning agent.

    Args:
        state: Current workflow state.
        planning_agent: Planning agent instance.

    Returns:
        Updated state with execution plan.
    """
    logger.info("Executing planning node")
    return await planning_agent.invoke(state)


async def orchestrator_node(
    state: AgentState,
    orchestrator_agent: OrchestratorAgent,
) -> AgentState:
    """Execute the orchestrator agent.

    Args:
        state: Current workflow state.
        orchestrator_agent: Orchestrator agent instance.

    Returns:
        Updated state with next_agent set.
    """
    logger.info("Executing orchestrator node")
    return await orchestrator_agent.invoke(state)


async def threads_node(
    state: AgentState,
    threads_agent: ThreadsAgent,
) -> AgentState:
    """Execute the threads agent.

    Args:
        state: Current workflow state.
        threads_agent: Threads agent instance.

    Returns:
        Updated state with threads results.
    """
    logger.info("Executing threads node")
    return await threads_agent.invoke(state)


async def response_node(
    state: AgentState,
    response_agent: ResponseAgent,
) -> AgentState:
    """Execute the response agent.

    Args:
        state: Current workflow state.
        response_agent: Response agent instance.

    Returns:
        Updated state with human-readable output.
    """
    logger.info("Executing response node")
    return await response_agent.invoke(state)


async def web_search_node(
    state: AgentState,
    web_search_agent: WebSearchAgent,
) -> AgentState:
    """Execute the web search agent.

    Args:
        state: Current workflow state.
        web_search_agent: Web search agent instance.

    Returns:
        Updated state with web search results.
    """
    logger.info("Executing web_search node")
    return await web_search_agent.invoke(state)


def create_node_functions(
    planning_agent: PlanningAgent,
    orchestrator_agent: OrchestratorAgent,
    threads_agent: ThreadsAgent,
    response_agent: ResponseAgent,
    web_search_agent: WebSearchAgent,
) -> dict[str, Any]:
    """Create node functions with injected agent dependencies.

    Args:
        planning_agent: Planning agent instance.
        orchestrator_agent: Orchestrator agent instance.
        threads_agent: Threads agent instance.
        response_agent: Response agent instance.
        web_search_agent: Web search agent instance.

    Returns:
        Dictionary of node name to node function.
    """

    async def _planning(state: AgentState) -> AgentState:
        return await planning_node(state, planning_agent)

    async def _orchestrator(state: AgentState) -> AgentState:
        return await orchestrator_node(state, orchestrator_agent)

    async def _threads(state: AgentState) -> AgentState:
        return await threads_node(state, threads_agent)

    async def _response(state: AgentState) -> AgentState:
        return await response_node(state, response_agent)

    async def _web_search(state: AgentState) -> AgentState:
        return await web_search_node(state, web_search_agent)

    return {
        "planning": _planning,
        "orchestrator": _orchestrator,
        "threads": _threads,
        "response": _response,
        "web_search": _web_search,
    }
