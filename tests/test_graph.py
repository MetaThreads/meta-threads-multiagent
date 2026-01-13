"""Tests for graph module."""

import pytest

from threads_hype_agent.graph.state import AgentState, create_initial_state
from threads_hype_agent.graph.edges import route_from_orchestrator, should_continue


class TestAgentState:
    """Tests for AgentState."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        messages = [{"role": "user", "content": "Hello"}]
        state = create_initial_state(messages)

        assert state["messages"] == messages
        assert state["plan"] is None
        assert state["web_search_results"] == []
        assert state["threads_results"] == []
        assert state["next_agent"] is None
        assert state["output"] is None


class TestRouting:
    """Tests for routing functions."""

    def test_route_to_web_search(self):
        """Test routing to web_search."""
        state: AgentState = {
            "messages": [],
            "plan": None,
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": "web_search",
            "error": None,
            "output": None,
        }
        result = route_from_orchestrator(state)
        assert result == "web_search"

    def test_route_to_threads(self):
        """Test routing to threads."""
        state: AgentState = {
            "messages": [],
            "plan": None,
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": "threads",
            "error": None,
            "output": None,
        }
        result = route_from_orchestrator(state)
        assert result == "threads"

    def test_route_to_response(self):
        """Test routing to response."""
        state: AgentState = {
            "messages": [],
            "plan": None,
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": "response",
            "error": None,
            "output": None,
        }
        result = route_from_orchestrator(state)
        assert result == "response"

    def test_route_default_to_response(self):
        """Test default routing to response."""
        state: AgentState = {
            "messages": [],
            "plan": None,
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": None,
            "error": None,
            "output": None,
        }
        result = route_from_orchestrator(state)
        assert result == "response"


class TestShouldContinue:
    """Tests for should_continue function."""

    def test_continue_with_incomplete_plan(self):
        """Test continuing with incomplete plan."""
        state: AgentState = {
            "messages": [],
            "plan": {
                "goal": "Test",
                "steps": [
                    {"agent": "threads", "action": "Post", "completed": False, "result": None}
                ],
                "current_step_index": 0,
            },
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": None,
            "error": None,
            "output": None,
        }
        assert should_continue(state) is True

    def test_stop_with_complete_plan(self):
        """Test stopping with complete plan."""
        state: AgentState = {
            "messages": [],
            "plan": {
                "goal": "Test",
                "steps": [
                    {"agent": "threads", "action": "Post", "completed": True, "result": "Done"}
                ],
                "current_step_index": 1,
            },
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": None,
            "error": None,
            "output": None,
        }
        assert should_continue(state) is False

    def test_stop_with_error(self):
        """Test stopping with error."""
        state: AgentState = {
            "messages": [],
            "plan": None,
            "web_search_results": [],
            "threads_results": [],
            "current_action": "",
            "next_agent": None,
            "error": "Something went wrong",
            "output": None,
        }
        assert should_continue(state) is False
