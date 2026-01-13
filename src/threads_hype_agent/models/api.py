"""API request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from threads_hype_agent.models.messages import Message


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    messages: list[Message] = Field(description="Conversation messages")


class StreamEvent(BaseModel):
    """A single event in the SSE stream."""

    type: Literal["token", "agent", "tool_call", "error", "done"] = Field(
        description="Type of stream event"
    )
    content: str | None = Field(default=None, description="Text content for token events")
    agent_name: str | None = Field(default=None, description="Agent name for agent events")
    status: Literal["started", "completed"] | None = Field(
        default=None, description="Agent status"
    )
    tool_name: str | None = Field(default=None, description="Tool name for tool_call events")
    tool_args: dict[str, Any] | None = Field(default=None, description="Tool arguments")
    tool_result: Any | None = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message for error events")

    def to_sse_data(self) -> str:
        """Convert to SSE data format."""
        return self.model_dump_json(exclude_none=True)


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    content: str = Field(description="Final response content")
    agent_trace: list[dict[str, Any]] = Field(
        default_factory=list, description="Trace of agent executions"
    )
