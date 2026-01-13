"""FastAPI dependencies for dependency injection."""

from functools import lru_cache

from threads_multiagent.agents.orchestrator import OrchestratorAgent
from threads_multiagent.agents.planning import PlanningAgent
from threads_multiagent.agents.response import ResponseAgent
from threads_multiagent.agents.threads import ThreadsAgent
from threads_multiagent.agents.web_search import WebSearchAgent
from threads_multiagent.config import Settings, get_settings
from threads_multiagent.graph.workflow import WorkflowRunner
from threads_multiagent.llm.openrouter import OpenRouterClient
from threads_multiagent.logging import get_logger
from threads_multiagent.search.duckduckgo import DuckDuckGoSearch

logger = get_logger(__name__)


@lru_cache
def get_llm_client() -> OpenRouterClient:
    """Get cached LLM client instance."""
    settings = get_settings()
    return OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
    )


@lru_cache
def get_web_search() -> DuckDuckGoSearch:
    """Get cached web search instance."""
    return DuckDuckGoSearch()


def get_workflow_runner() -> WorkflowRunner:
    """Create workflow runner with all dependencies.

    Returns:
        Configured workflow runner.
    """
    settings = get_settings()
    llm = get_llm_client()
    web_search = get_web_search()

    # Create agents
    planning_agent = PlanningAgent(llm)
    orchestrator_agent = OrchestratorAgent(llm)
    threads_agent = ThreadsAgent(
        llm=llm,
        mcp_server_url=settings.threads_mcp_url,
        bearer_token=settings.threads_bearer_token,
    )
    response_agent = ResponseAgent(llm)
    web_search_agent = WebSearchAgent(llm, web_search)

    # Create workflow runner
    return WorkflowRunner(
        planning_agent=planning_agent,
        orchestrator_agent=orchestrator_agent,
        threads_agent=threads_agent,
        response_agent=response_agent,
        web_search_agent=web_search_agent,
        max_iterations=settings.max_agent_iterations,
    )
