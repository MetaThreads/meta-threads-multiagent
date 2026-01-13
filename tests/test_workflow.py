"""Tests for workflow module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threads_multiagent.graph.state import create_initial_state
from threads_multiagent.graph.workflow import WorkflowRunner, build_workflow


@pytest.fixture
def mock_agents():
    """Create mock agent instances."""
    planning = MagicMock()
    planning.invoke = AsyncMock()

    orchestrator = MagicMock()
    orchestrator.invoke = AsyncMock()

    threads = MagicMock()
    threads.invoke = AsyncMock()

    response = MagicMock()
    response.invoke = AsyncMock()

    web_search = MagicMock()
    web_search.invoke = AsyncMock()

    return {
        "planning": planning,
        "orchestrator": orchestrator,
        "threads": threads,
        "response": response,
        "web_search": web_search,
    }


@pytest.fixture
def mock_tracer():
    """Create a mock tracer."""
    tracer = MagicMock()
    tracer.get_callback_handler.return_value = None
    tracer.build_config_metadata.return_value = {}
    tracer.flush = MagicMock()
    return tracer


class TestBuildWorkflow:
    """Tests for build_workflow function."""

    def test_build_workflow_returns_compiled_graph(self, mock_agents):
        """Test that build_workflow returns a compiled StateGraph."""
        workflow = build_workflow(
            mock_agents["planning"],
            mock_agents["orchestrator"],
            mock_agents["threads"],
            mock_agents["response"],
            mock_agents["web_search"],
        )

        # Should be a compiled graph (CompiledGraph)
        assert workflow is not None
        assert hasattr(workflow, "ainvoke")
        assert hasattr(workflow, "astream")


class TestWorkflowRunner:
    """Tests for WorkflowRunner class."""

    def test_initialization(self, mock_agents, mock_tracer):
        """Test runner initialization."""
        with patch("threads_multiagent.graph.workflow.get_tracer", return_value=mock_tracer):
            runner = WorkflowRunner(
                planning_agent=mock_agents["planning"],
                orchestrator_agent=mock_agents["orchestrator"],
                threads_agent=mock_agents["threads"],
                response_agent=mock_agents["response"],
                web_search_agent=mock_agents["web_search"],
                max_iterations=20,
            )

            assert runner._max_iterations == 20
            assert runner._workflow is not None

    def test_initialization_with_custom_tracer(self, mock_agents, mock_tracer):
        """Test runner initialization with custom tracer."""
        runner = WorkflowRunner(
            planning_agent=mock_agents["planning"],
            orchestrator_agent=mock_agents["orchestrator"],
            threads_agent=mock_agents["threads"],
            response_agent=mock_agents["response"],
            web_search_agent=mock_agents["web_search"],
            tracer=mock_tracer,
        )

        assert runner._tracer == mock_tracer

    @pytest.mark.asyncio
    async def test_run(self, mock_agents, mock_tracer):
        """Test workflow run method."""
        expected_state = {
            "messages": [{"role": "user", "content": "test"}],
            "output": "Final output",
        }

        with patch("threads_multiagent.graph.workflow.build_workflow") as mock_build:
            mock_workflow = MagicMock()
            mock_workflow.ainvoke = AsyncMock(return_value=expected_state)
            mock_build.return_value = mock_workflow

            runner = WorkflowRunner(
                planning_agent=mock_agents["planning"],
                orchestrator_agent=mock_agents["orchestrator"],
                threads_agent=mock_agents["threads"],
                response_agent=mock_agents["response"],
                web_search_agent=mock_agents["web_search"],
                tracer=mock_tracer,
            )

            messages = [{"role": "user", "content": "test"}]
            result = await runner.run(messages)

            assert result["output"] == "Final output"
            mock_tracer.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_user_session_ids(self, mock_agents, mock_tracer):
        """Test workflow run with user and session IDs."""
        mock_tracer.get_callback_handler.return_value = MagicMock()

        with patch("threads_multiagent.graph.workflow.build_workflow") as mock_build:
            mock_workflow = MagicMock()
            mock_workflow.ainvoke = AsyncMock(return_value={})
            mock_build.return_value = mock_workflow

            runner = WorkflowRunner(
                planning_agent=mock_agents["planning"],
                orchestrator_agent=mock_agents["orchestrator"],
                threads_agent=mock_agents["threads"],
                response_agent=mock_agents["response"],
                web_search_agent=mock_agents["web_search"],
                tracer=mock_tracer,
            )

            await runner.run(
                [{"role": "user", "content": "test"}],
                user_id="user-123",
                session_id="session-456",
            )

            mock_tracer.build_config_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream(self, mock_agents, mock_tracer):
        """Test workflow stream method."""

        async def mock_astream(*_args, **_kwargs):
            yield {"planning": {"messages": []}}
            yield {"orchestrator": {"next_agent": "response"}}
            yield {"response": {"output": "Done"}}

        with patch("threads_multiagent.graph.workflow.build_workflow") as mock_build:
            mock_workflow = MagicMock()
            mock_workflow.astream = mock_astream
            mock_build.return_value = mock_workflow

            runner = WorkflowRunner(
                planning_agent=mock_agents["planning"],
                orchestrator_agent=mock_agents["orchestrator"],
                threads_agent=mock_agents["threads"],
                response_agent=mock_agents["response"],
                web_search_agent=mock_agents["web_search"],
                tracer=mock_tracer,
            )

            messages = [{"role": "user", "content": "test"}]
            events = []

            async for event in runner.stream(messages):
                events.append(event)

            assert len(events) == 3
            assert events[0]["node"] == "planning"
            assert events[1]["node"] == "orchestrator"
            assert events[2]["node"] == "response"
            mock_tracer.flush.assert_called_once()


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_state_with_messages(self):
        """Test creating initial state with messages."""
        messages = [{"role": "user", "content": "Hello"}]
        state = create_initial_state(messages)

        assert state["messages"] == messages
        assert state["plan"] is None
        assert state["web_search_results"] == []
        assert state["threads_results"] == []
        assert state["next_agent"] is None
        assert state["error"] is None
        assert state["output"] is None

    def test_create_state_with_multiple_messages(self):
        """Test creating initial state with multiple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Search for news"},
        ]
        state = create_initial_state(messages)

        assert len(state["messages"]) == 3
