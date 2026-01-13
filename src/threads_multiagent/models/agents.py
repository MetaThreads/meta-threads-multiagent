"""Agent response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call made by an agent."""

    tool_name: str = Field(description="Name of the tool being called")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: Any | None = Field(default=None, description="Result of the tool call")


class AgentResponse(BaseModel):
    """Response from an agent invocation."""

    agent_name: str = Field(description="Name of the agent that produced this response")
    content: str = Field(description="Text content of the response")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tools called during execution"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    agent: Literal["threads", "web_search"] = Field(description="Agent to execute this step")
    action: str = Field(description="Description of the action to perform")
    completed: bool = Field(default=False, description="Whether this step is completed")
    result: str | None = Field(default=None, description="Result of the step execution")


class Plan(BaseModel):
    """Execution plan created by the planning agent."""

    goal: str = Field(description="The overall goal to achieve")
    steps: list[PlanStep] = Field(default_factory=list, description="Steps to execute")
    current_step_index: int = Field(default=0, description="Index of the current step")

    def get_current_step(self) -> PlanStep | None:
        """Get the current step to execute."""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_next_incomplete_step(self) -> PlanStep | None:
        """Get the next incomplete step."""
        for step in self.steps:
            if not step.completed:
                return step
        return None

    def mark_current_step_completed(self, result: str | None = None) -> None:
        """Mark the current step as completed and advance."""
        if self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].completed = True
            self.steps[self.current_step_index].result = result
            self.current_step_index += 1

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.completed for step in self.steps)
