"""Tests for agent implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from threads_multiagent.agents.planning import PlanningAgent
from threads_multiagent.agents.orchestrator import OrchestratorAgent
from threads_multiagent.agents.web_search import WebSearchAgent
from threads_multiagent.agents.response import ResponseAgent
from threads_multiagent.models.agents import Plan, PlanStep
from threads_multiagent.search.base import SearchResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def mock_web_search():
    """Create a mock web search."""
    search = MagicMock()
    search.search = AsyncMock(return_value=[
        SearchResult(
            title="Test Article",
            url="https://example.com/test",
            snippet="Test snippet content",
            source="example.com",
        )
    ])
    return search


@pytest.fixture
def sample_state():
    """Create a sample workflow state."""
    return {
        "messages": [{"role": "user", "content": "Find latest crypto news"}],
        "plan": None,
        "web_search_results": [],
        "threads_results": [],
        "current_action": "",
        "next_agent": None,
        "error": None,
        "output": None,
    }


class TestPlanningAgent:
    """Tests for PlanningAgent."""

    def test_agent_name(self, mock_llm):
        """Test agent name property."""
        agent = PlanningAgent(mock_llm)
        assert agent.name == "planning"

    def test_agent_description(self, mock_llm):
        """Test agent description property."""
        agent = PlanningAgent(mock_llm)
        assert "plan" in agent.description.lower()

    def test_get_system_prompt(self, mock_llm):
        """Test system prompt is not empty."""
        agent = PlanningAgent(mock_llm)
        prompt = agent.get_system_prompt()
        assert prompt is not None
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_invoke_creates_plan(self, mock_llm, sample_state):
        """Test that invoke creates a plan."""
        mock_response = MagicMock()
        mock_response.content = '''{"goal": "Find crypto news", "steps": [{"agent": "web_search", "action": "Search for news"}]}'''
        mock_llm.complete.return_value = mock_response

        agent = PlanningAgent(mock_llm)
        result = await agent.invoke(sample_state)

        assert result["plan"] is not None
        assert result["plan"]["goal"] == "Find crypto news"
        assert len(result["plan"]["steps"]) == 1

    @pytest.mark.asyncio
    async def test_invoke_fallback_on_invalid_json(self, mock_llm, sample_state):
        """Test fallback plan when JSON parsing fails."""
        mock_response = MagicMock()
        mock_response.content = "Invalid response"
        mock_llm.complete.return_value = mock_response

        agent = PlanningAgent(mock_llm)
        result = await agent.invoke(sample_state)

        assert result["plan"] is not None
        # Should create fallback plan based on keywords in user message
        assert len(result["plan"]["steps"]) > 0

    @pytest.mark.asyncio
    async def test_invoke_no_user_message_raises(self, mock_llm):
        """Test that invoke raises when no user message."""
        from threads_multiagent.exceptions import PlanningError

        agent = PlanningAgent(mock_llm)
        state = {"messages": [{"role": "assistant", "content": "Hello"}]}

        with pytest.raises(PlanningError):
            await agent.invoke(state)


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent."""

    def test_agent_name(self, mock_llm):
        """Test agent name property."""
        agent = OrchestratorAgent(mock_llm)
        assert agent.name == "orchestrator"

    def test_agent_description(self, mock_llm):
        """Test agent description property."""
        agent = OrchestratorAgent(mock_llm)
        assert len(agent.description) > 0

    def test_get_next_agent_threads(self, mock_llm):
        """Test get_next_agent returns threads."""
        agent = OrchestratorAgent(mock_llm)
        state = {"next_agent": "threads"}
        assert agent.get_next_agent(state) == "threads"

    def test_get_next_agent_web_search(self, mock_llm):
        """Test get_next_agent returns web_search."""
        agent = OrchestratorAgent(mock_llm)
        state = {"next_agent": "web_search"}
        assert agent.get_next_agent(state) == "web_search"

    def test_get_next_agent_response(self, mock_llm):
        """Test get_next_agent returns response."""
        agent = OrchestratorAgent(mock_llm)
        state = {"next_agent": "response"}
        assert agent.get_next_agent(state) == "response"

    def test_get_next_agent_default(self, mock_llm):
        """Test get_next_agent returns response as default."""
        agent = OrchestratorAgent(mock_llm)
        state = {"next_agent": None}
        assert agent.get_next_agent(state) == "response"

    def test_get_next_agent_invalid(self, mock_llm):
        """Test get_next_agent returns response for invalid value."""
        agent = OrchestratorAgent(mock_llm)
        state = {"next_agent": "invalid"}
        assert agent.get_next_agent(state) == "response"

    @pytest.mark.asyncio
    async def test_invoke_no_plan_raises(self, mock_llm, sample_state):
        """Test invoke raises when no plan in state."""
        from threads_multiagent.exceptions import OrchestrationError

        agent = OrchestratorAgent(mock_llm)
        with pytest.raises(OrchestrationError):
            await agent.invoke(sample_state)

    @pytest.mark.asyncio
    async def test_invoke_continue_decision(self, mock_llm, sample_state):
        """Test invoke with continue decision."""
        mock_response = MagicMock()
        mock_response.content = '{"evaluation": "Good", "decision": "continue", "reasoning": "Next step", "modifications": []}'
        mock_llm.complete.return_value = mock_response

        sample_state["plan"] = {
            "goal": "Test",
            "steps": [
                {"agent": "web_search", "action": "Search", "completed": False, "result": None}
            ],
            "current_step_index": 0,
        }

        agent = OrchestratorAgent(mock_llm)
        result = await agent.invoke(sample_state)

        assert result["next_agent"] == "web_search"

    @pytest.mark.asyncio
    async def test_invoke_complete_decision(self, mock_llm, sample_state):
        """Test invoke with complete decision."""
        mock_response = MagicMock()
        mock_response.content = '{"evaluation": "Done", "decision": "complete", "reasoning": "All done", "modifications": []}'
        mock_llm.complete.return_value = mock_response

        sample_state["plan"] = {
            "goal": "Test",
            "steps": [
                {"agent": "web_search", "action": "Search", "completed": True, "result": "Done"}
            ],
            "current_step_index": 1,
        }

        agent = OrchestratorAgent(mock_llm)
        result = await agent.invoke(sample_state)

        assert result["next_agent"] == "response"


class TestWebSearchAgent:
    """Tests for WebSearchAgent."""

    def test_agent_name(self, mock_llm, mock_web_search):
        """Test agent name property."""
        agent = WebSearchAgent(mock_llm, mock_web_search)
        assert agent.name == "web_search"

    def test_agent_description(self, mock_llm, mock_web_search):
        """Test agent description property."""
        agent = WebSearchAgent(mock_llm, mock_web_search)
        assert "search" in agent.description.lower()

    @pytest.mark.asyncio
    async def test_tool_generate_query(self, mock_llm, mock_web_search):
        """Test query generation tool."""
        mock_response = MagicMock()
        mock_response.content = "crypto market trends 2024"
        mock_llm.complete.return_value = mock_response

        agent = WebSearchAgent(mock_llm, mock_web_search)
        query = await agent._tool_generate_query("Find crypto news", "Search news", "Search")

        assert query == "crypto market trends 2024"
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_search(self, mock_llm, mock_web_search):
        """Test search tool."""
        agent = WebSearchAgent(mock_llm, mock_web_search)
        results = await agent._tool_search("test query", limit=3)

        assert len(results) == 1
        assert results[0]["title"] == "Test Article"
        mock_web_search.search.assert_called_once_with(query="test query", limit=3)

    @pytest.mark.asyncio
    async def test_tool_synthesize(self, mock_llm, mock_web_search):
        """Test synthesis tool."""
        mock_response = MagicMock()
        mock_response.content = "Summary of search results"
        mock_llm.complete.return_value = mock_response

        agent = WebSearchAgent(mock_llm, mock_web_search)
        results = [{"title": "Test", "url": "http://test.com", "snippet": "Test", "source": "test.com"}]
        synthesis = await agent._tool_synthesize(results, "user request", "goal")

        assert synthesis == "Summary of search results"

    @pytest.mark.asyncio
    async def test_tool_synthesize_no_results(self, mock_llm, mock_web_search):
        """Test synthesis tool with no results."""
        agent = WebSearchAgent(mock_llm, mock_web_search)
        synthesis = await agent._tool_synthesize([], "user request", "goal")

        assert "No relevant search results" in synthesis

    @pytest.mark.asyncio
    async def test_invoke(self, mock_llm, mock_web_search, sample_state):
        """Test full invoke flow."""
        mock_response = MagicMock()
        mock_response.content = "Generated content"
        mock_llm.complete.return_value = mock_response

        sample_state["plan"] = {
            "goal": "Find news",
            "steps": [{"agent": "web_search", "action": "Search", "completed": False, "result": None}],
            "current_step_index": 0,
        }

        agent = WebSearchAgent(mock_llm, mock_web_search)
        result = await agent.invoke(sample_state)

        assert len(result["web_search_results"]) == 1
        assert result["plan"]["steps"][0]["completed"] is True


class TestResponseAgent:
    """Tests for ResponseAgent."""

    def test_agent_name(self, mock_llm):
        """Test agent name property."""
        agent = ResponseAgent(mock_llm)
        assert agent.name == "response"

    def test_agent_description(self, mock_llm):
        """Test agent description property."""
        agent = ResponseAgent(mock_llm)
        assert "response" in agent.description.lower()

    def test_tool_build_context(self, mock_llm, sample_state):
        """Test context building tool."""
        sample_state["plan"] = {"goal": "Test goal", "steps": [], "current_step_index": 0}
        sample_state["web_search_results"] = [
            {"title": "Test", "source": "test.com", "snippet": "Test snippet here"}
        ]

        agent = ResponseAgent(mock_llm)
        context = agent._tool_build_context(sample_state)

        assert "Test goal" in context
        assert "user" in context.lower()

    def test_tool_build_context_with_threads_results(self, mock_llm, sample_state):
        """Test context building with threads results."""
        sample_state["threads_results"] = [
            {"action": "post", "result": '{"id": "123"}'}
        ]

        agent = ResponseAgent(mock_llm)
        context = agent._tool_build_context(sample_state)

        assert "Threads operations" in context
        assert "post" in context

    @pytest.mark.asyncio
    async def test_tool_generate_response(self, mock_llm):
        """Test response generation tool."""
        mock_response = MagicMock()
        mock_response.content = "Generated human response"
        mock_llm.complete.return_value = mock_response

        agent = ResponseAgent(mock_llm)
        response = await agent._tool_generate_response("Some context")

        assert response == "Generated human response"
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke(self, mock_llm, sample_state):
        """Test full invoke flow."""
        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_llm.complete.return_value = mock_response

        agent = ResponseAgent(mock_llm)
        result = await agent.invoke(sample_state)

        assert result["output"] == "Final response"
        assert len(result["messages"]) > len(sample_state["messages"])
