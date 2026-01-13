"""Test script for MCP integration."""

import asyncio

from threads_multiagent.agents.orchestrator import OrchestratorAgent
from threads_multiagent.agents.planning import PlanningAgent
from threads_multiagent.agents.response import ResponseAgent
from threads_multiagent.agents.threads import ThreadsAgent
from threads_multiagent.agents.web_search import WebSearchAgent
from threads_multiagent.config import get_settings
from threads_multiagent.graph.workflow import WorkflowRunner
from threads_multiagent.llm.openrouter import OpenRouterClient
from threads_multiagent.search.duckduckgo import DuckDuckGoSearch


async def main():
    settings = get_settings()

    print(f"LLM Model: {settings.openrouter_model}")
    print(f"MCP URL: {settings.threads_mcp_url}")

    llm = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
    )
    web_search = DuckDuckGoSearch()

    runner = WorkflowRunner(
        planning_agent=PlanningAgent(llm),
        orchestrator_agent=OrchestratorAgent(llm),
        threads_agent=ThreadsAgent(
            llm=llm,
            mcp_server_url=settings.threads_mcp_url,
            bearer_token=settings.threads_bearer_token,
        ),
        response_agent=ResponseAgent(llm),
        web_search_agent=WebSearchAgent(llm, web_search),
    )

    print("\n=== Testing: What is my latest threads post? ===\n")

    result = await runner.run([
        {"role": "user", "content": "What is my latest threads post?"}
    ])

    print("=== OUTPUT ===")
    print(result.get("output", "No output"))


if __name__ == "__main__":
    asyncio.run(main())
