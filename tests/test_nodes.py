"""Tests for graph node functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from threads_multiagent.graph.nodes import (
    planning_node,
    orchestrator_node,
    threads_node,
    response_node,
    web_search_node,
    create_node_functions,
)


@pytest.fixture
def sample_state():
    """Create a sample workflow state."""
    return {
        "messages": [{"role": "user", "content": "Test"}],
        "plan": None,
        "web_search_results": [],
        "threads_results": [],
        "current_action": "",
        "next_agent": None,
        "error": None,
        "output": None,
    }


@pytest.fixture
def mock_planning_agent():
    """Create a mock planning agent."""
    agent = MagicMock()
    agent.invoke = AsyncMock(return_value={"plan": {"goal": "Test", "steps": []}})
    return agent


@pytest.fixture
def mock_orchestrator_agent():
    """Create a mock orchestrator agent."""
    agent = MagicMock()
    agent.invoke = AsyncMock(return_value={"next_agent": "response"})
    return agent


@pytest.fixture
def mock_threads_agent():
    """Create a mock threads agent."""
    agent = MagicMock()
    agent.invoke = AsyncMock(return_value={"threads_results": [{"action": "post"}]})
    return agent


@pytest.fixture
def mock_response_agent():
    """Create a mock response agent."""
    agent = MagicMock()
    agent.invoke = AsyncMock(return_value={"output": "Final response"})
    return agent


@pytest.fixture
def mock_web_search_agent():
    """Create a mock web search agent."""
    agent = MagicMock()
    agent.invoke = AsyncMock(return_value={"web_search_results": [{"title": "Test"}]})
    return agent


class TestPlanningNode:
    """Tests for planning_node function."""

    @pytest.mark.asyncio
    async def test_planning_node_calls_agent(self, sample_state, mock_planning_agent):
        """Test that planning_node calls the agent's invoke method."""
        result = await planning_node(sample_state, mock_planning_agent)

        mock_planning_agent.invoke.assert_called_once_with(sample_state)
        assert "plan" in result


class TestOrchestratorNode:
    """Tests for orchestrator_node function."""

    @pytest.mark.asyncio
    async def test_orchestrator_node_calls_agent(self, sample_state, mock_orchestrator_agent):
        """Test that orchestrator_node calls the agent's invoke method."""
        result = await orchestrator_node(sample_state, mock_orchestrator_agent)

        mock_orchestrator_agent.invoke.assert_called_once_with(sample_state)
        assert result["next_agent"] == "response"


class TestThreadsNode:
    """Tests for threads_node function."""

    @pytest.mark.asyncio
    async def test_threads_node_calls_agent(self, sample_state, mock_threads_agent):
        """Test that threads_node calls the agent's invoke method."""
        result = await threads_node(sample_state, mock_threads_agent)

        mock_threads_agent.invoke.assert_called_once_with(sample_state)
        assert "threads_results" in result


class TestResponseNode:
    """Tests for response_node function."""

    @pytest.mark.asyncio
    async def test_response_node_calls_agent(self, sample_state, mock_response_agent):
        """Test that response_node calls the agent's invoke method."""
        result = await response_node(sample_state, mock_response_agent)

        mock_response_agent.invoke.assert_called_once_with(sample_state)
        assert result["output"] == "Final response"


class TestWebSearchNode:
    """Tests for web_search_node function."""

    @pytest.mark.asyncio
    async def test_web_search_node_calls_agent(self, sample_state, mock_web_search_agent):
        """Test that web_search_node calls the agent's invoke method."""
        result = await web_search_node(sample_state, mock_web_search_agent)

        mock_web_search_agent.invoke.assert_called_once_with(sample_state)
        assert "web_search_results" in result


class TestCreateNodeFunctions:
    """Tests for create_node_functions function."""

    def test_create_node_functions_returns_all_nodes(
        self,
        mock_planning_agent,
        mock_orchestrator_agent,
        mock_threads_agent,
        mock_response_agent,
        mock_web_search_agent,
    ):
        """Test that create_node_functions returns all expected nodes."""
        nodes = create_node_functions(
            mock_planning_agent,
            mock_orchestrator_agent,
            mock_threads_agent,
            mock_response_agent,
            mock_web_search_agent,
        )

        assert "planning" in nodes
        assert "orchestrator" in nodes
        assert "threads" in nodes
        assert "response" in nodes
        assert "web_search" in nodes

    @pytest.mark.asyncio
    async def test_created_planning_node_works(
        self,
        sample_state,
        mock_planning_agent,
        mock_orchestrator_agent,
        mock_threads_agent,
        mock_response_agent,
        mock_web_search_agent,
    ):
        """Test that created planning node function works."""
        nodes = create_node_functions(
            mock_planning_agent,
            mock_orchestrator_agent,
            mock_threads_agent,
            mock_response_agent,
            mock_web_search_agent,
        )

        result = await nodes["planning"](sample_state)
        mock_planning_agent.invoke.assert_called_once()
        assert "plan" in result

    @pytest.mark.asyncio
    async def test_created_orchestrator_node_works(
        self,
        sample_state,
        mock_planning_agent,
        mock_orchestrator_agent,
        mock_threads_agent,
        mock_response_agent,
        mock_web_search_agent,
    ):
        """Test that created orchestrator node function works."""
        nodes = create_node_functions(
            mock_planning_agent,
            mock_orchestrator_agent,
            mock_threads_agent,
            mock_response_agent,
            mock_web_search_agent,
        )

        await nodes["orchestrator"](sample_state)
        mock_orchestrator_agent.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_created_threads_node_works(
        self,
        sample_state,
        mock_planning_agent,
        mock_orchestrator_agent,
        mock_threads_agent,
        mock_response_agent,
        mock_web_search_agent,
    ):
        """Test that created threads node function works."""
        nodes = create_node_functions(
            mock_planning_agent,
            mock_orchestrator_agent,
            mock_threads_agent,
            mock_response_agent,
            mock_web_search_agent,
        )

        await nodes["threads"](sample_state)
        mock_threads_agent.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_created_response_node_works(
        self,
        sample_state,
        mock_planning_agent,
        mock_orchestrator_agent,
        mock_threads_agent,
        mock_response_agent,
        mock_web_search_agent,
    ):
        """Test that created response node function works."""
        nodes = create_node_functions(
            mock_planning_agent,
            mock_orchestrator_agent,
            mock_threads_agent,
            mock_response_agent,
            mock_web_search_agent,
        )

        await nodes["response"](sample_state)
        mock_response_agent.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_created_web_search_node_works(
        self,
        sample_state,
        mock_planning_agent,
        mock_orchestrator_agent,
        mock_threads_agent,
        mock_response_agent,
        mock_web_search_agent,
    ):
        """Test that created web_search node function works."""
        nodes = create_node_functions(
            mock_planning_agent,
            mock_orchestrator_agent,
            mock_threads_agent,
            mock_response_agent,
            mock_web_search_agent,
        )

        await nodes["web_search"](sample_state)
        mock_web_search_agent.invoke.assert_called_once()
