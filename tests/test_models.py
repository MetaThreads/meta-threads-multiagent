"""Tests for Pydantic models."""

import pytest

from threads_multiagent.models.agents import Plan, PlanStep
from threads_multiagent.models.api import ChatRequest, StreamEvent
from threads_multiagent.models.messages import Conversation, Message


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        msg = Message(role="assistant", content="Hi there")
        result = msg.to_openai_format()
        assert result == {"role": "assistant", "content": "Hi there"}

    def test_invalid_role(self):
        """Test that invalid role raises error."""
        with pytest.raises(ValueError):
            Message(role="invalid", content="test")


class TestConversation:
    """Tests for Conversation model."""

    def test_empty_conversation(self):
        """Test creating empty conversation."""
        conv = Conversation()
        assert conv.messages == []

    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ]
        )
        result = conv.to_openai_format()
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_get_last_user_message(self):
        """Test getting last user message."""
        conv = Conversation(
            messages=[
                Message(role="user", content="First"),
                Message(role="assistant", content="Response"),
                Message(role="user", content="Second"),
            ]
        )
        assert conv.get_last_user_message() == "Second"

    def test_get_last_user_message_empty(self):
        """Test getting last user message when none exists."""
        conv = Conversation(
            messages=[
                Message(role="assistant", content="Hello"),
            ]
        )
        assert conv.get_last_user_message() is None


class TestPlan:
    """Tests for Plan model."""

    def test_create_plan(self):
        """Test creating a plan."""
        plan = Plan(
            goal="Test goal",
            steps=[
                PlanStep(agent="web_search", action="Fetch news"),
                PlanStep(agent="threads", action="Post content"),
            ],
        )
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 2

    def test_get_current_step(self):
        """Test getting current step."""
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(agent="web_search", action="Fetch"),
                PlanStep(agent="threads", action="Post"),
            ],
        )
        step = plan.get_current_step()
        assert step is not None
        assert step.agent == "web_search"

    def test_get_next_incomplete_step(self):
        """Test getting next incomplete step."""
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(agent="web_search", action="Fetch", completed=True),
                PlanStep(agent="threads", action="Post", completed=False),
            ],
        )
        step = plan.get_next_incomplete_step()
        assert step is not None
        assert step.agent == "threads"

    def test_mark_step_completed(self):
        """Test marking step as completed."""
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(agent="web_search", action="Fetch"),
                PlanStep(agent="threads", action="Post"),
            ],
        )
        plan.mark_current_step_completed(result="Done")
        assert plan.steps[0].completed is True
        assert plan.steps[0].result == "Done"
        assert plan.current_step_index == 1

    def test_is_complete(self):
        """Test checking if plan is complete."""
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(agent="web_search", action="Fetch", completed=True),
                PlanStep(agent="threads", action="Post", completed=True),
            ],
        )
        assert plan.is_complete() is True

    def test_is_not_complete(self):
        """Test checking if plan is incomplete."""
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(agent="web_search", action="Fetch", completed=True),
                PlanStep(agent="threads", action="Post", completed=False),
            ],
        )
        assert plan.is_complete() is False


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_create_request(self):
        """Test creating a chat request."""
        request = ChatRequest(
            messages=[
                Message(role="user", content="Hello"),
            ]
        )
        assert len(request.messages) == 1


class TestStreamEvent:
    """Tests for StreamEvent model."""

    def test_token_event(self):
        """Test creating token event."""
        event = StreamEvent(type="token", content="Hello")
        assert event.type == "token"
        assert event.content == "Hello"

    def test_agent_event(self):
        """Test creating agent event."""
        event = StreamEvent(type="agent", agent_name="planning", status="started")
        assert event.type == "agent"
        assert event.agent_name == "planning"

    def test_to_sse_data(self):
        """Test conversion to SSE data."""
        event = StreamEvent(type="done", content="Complete")
        data = event.to_sse_data()
        assert "done" in data
        assert "Complete" in data
