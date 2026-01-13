"""LangGraph workflow assembly."""

from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph

from threads_multiagent.agents.orchestrator import OrchestratorAgent
from threads_multiagent.agents.planning import PlanningAgent
from threads_multiagent.agents.response import ResponseAgent
from threads_multiagent.agents.threads import ThreadsAgent
from threads_multiagent.agents.web_search import WebSearchAgent
from threads_multiagent.graph.edges import route_from_orchestrator
from threads_multiagent.graph.nodes import create_node_functions
from threads_multiagent.graph.state import AgentState, create_initial_state
from threads_multiagent.logging import get_logger
from threads_multiagent.tracing import LangfuseTracer, get_tracer

logger = get_logger(__name__)


def build_workflow(
    planning_agent: PlanningAgent,
    orchestrator_agent: OrchestratorAgent,
    threads_agent: ThreadsAgent,
    response_agent: ResponseAgent,
    web_search_agent: WebSearchAgent,
) -> StateGraph:
    """Build the LangGraph workflow.

    Creates a workflow with the following structure:
    1. Planning agent (entry point)
    2. Orchestrator agent (routes in a loop)
    3. Web search and Threads agents (execute based on routing)
    4. Response agent (generates human-readable output before end)

    Args:
        planning_agent: Planning agent instance.
        orchestrator_agent: Orchestrator agent instance.
        threads_agent: Threads agent instance.
        response_agent: Response agent instance.
        web_search_agent: Web search agent instance.

    Returns:
        Compiled StateGraph workflow.
    """
    # Create node functions with injected dependencies
    nodes = create_node_functions(
        planning_agent,
        orchestrator_agent,
        threads_agent,
        response_agent,
        web_search_agent,
    )

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planning", nodes["planning"])
    workflow.add_node("orchestrator", nodes["orchestrator"])
    workflow.add_node("threads", nodes["threads"])
    workflow.add_node("response", nodes["response"])
    workflow.add_node("web_search", nodes["web_search"])

    # Set entry point: always start with planning
    workflow.set_entry_point("planning")

    # Planning → Orchestrator (always)
    workflow.add_edge("planning", "orchestrator")

    # Orchestrator routes conditionally
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "threads": "threads",
            "web_search": "web_search",
            "response": "response",
        },
    )

    # Agents loop back to orchestrator
    workflow.add_edge("threads", "orchestrator")
    workflow.add_edge("web_search", "orchestrator")

    # Response leads to END
    workflow.add_edge("response", END)

    logger.info("Built workflow graph")
    return workflow.compile()


class WorkflowRunner:
    """Runner for executing the LangGraph workflow.

    Provides methods for running the workflow with
    optional streaming of intermediate states.
    """

    def __init__(
        self,
        planning_agent: PlanningAgent,
        orchestrator_agent: OrchestratorAgent,
        threads_agent: ThreadsAgent,
        response_agent: ResponseAgent,
        web_search_agent: WebSearchAgent,
        max_iterations: int = 10,
        tracer: LangfuseTracer | None = None,
    ):
        """Initialize the workflow runner.

        Args:
            planning_agent: Planning agent instance.
            orchestrator_agent: Orchestrator agent instance.
            threads_agent: Threads agent instance.
            response_agent: Response agent instance.
            web_search_agent: Web search agent instance.
            max_iterations: Maximum workflow iterations.
            tracer: Optional Langfuse tracer for observability.
        """
        self._workflow = build_workflow(
            planning_agent,
            orchestrator_agent,
            threads_agent,
            response_agent,
            web_search_agent,
        )
        self._max_iterations = max_iterations
        self._tracer = tracer or get_tracer()
        logger.info("Initialized workflow runner")

    async def run(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentState:
        """Run the workflow to completion.

        Args:
            messages: Initial conversation messages.
            user_id: Optional user ID for tracing.
            session_id: Optional session ID for tracing.

        Returns:
            Final workflow state.
        """
        initial_state = create_initial_state(messages)
        logger.info("Starting workflow run")

        # Build config with optional Langfuse callback
        config: dict[str, Any] = {"recursion_limit": self._max_iterations}

        callback_handler = self._tracer.get_callback_handler()

        if callback_handler is not None:
            config["callbacks"] = [callback_handler]
            # In SDK v3, trace attributes are passed via metadata
            config["metadata"] = self._tracer.build_config_metadata(
                user_id=user_id,
                session_id=session_id,
                metadata={"message_count": len(messages)},
            )

        # Run workflow
        final_state = await self._workflow.ainvoke(initial_state, config=config)

        # Flush traces
        self._tracer.flush()

        logger.info("Workflow run completed")
        return final_state

    async def stream(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream workflow execution states.

        Args:
            messages: Initial conversation messages.
            user_id: Optional user ID for tracing.
            session_id: Optional session ID for tracing.

        Yields:
            Intermediate state updates with node information.
        """
        initial_state = create_initial_state(messages)
        logger.info("Starting workflow stream")

        # Build config with optional Langfuse callback
        config: dict[str, Any] = {"recursion_limit": self._max_iterations}

        callback_handler = self._tracer.get_callback_handler()

        if callback_handler is not None:
            config["callbacks"] = [callback_handler]
            # In SDK v3, trace attributes are passed via metadata
            config["metadata"] = self._tracer.build_config_metadata(
                user_id=user_id,
                session_id=session_id,
                metadata={"message_count": len(messages), "streaming": True},
            )

        async for event in self._workflow.astream(initial_state, config=config):
            # Event contains node name and state
            for node_name, node_state in event.items():
                yield {
                    "node": node_name,
                    "state": node_state,
                }

        # Flush traces
        self._tracer.flush()

        logger.info("Workflow stream completed")
