"""Base agent abstraction with Langfuse tracing support."""

from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from langfuse import observe

from threads_hype_agent.config import get_settings
from threads_hype_agent.llm.base import BaseLLMClient
from threads_hype_agent.logging import get_logger

if TYPE_CHECKING:
    from threads_hype_agent.graph.state import AgentState

logger = get_logger(__name__)

T = TypeVar("T")


def traced_tool(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace a tool call with Langfuse.

    Args:
        name: Optional name for the tool. Defaults to function name.

    Returns:
        Decorated function with Langfuse tracing.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tool_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            settings = get_settings()
            if settings.langfuse_enabled:
                # Use observe decorator dynamically
                observed_func = observe(as_type="tool", name=tool_name)(func)
                return await observed_func(*args, **kwargs)
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            settings = get_settings()
            if settings.langfuse_enabled:
                observed_func = observe(as_type="tool", name=tool_name)(func)
                return observed_func(*args, **kwargs)
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class BaseAgent(ABC):
    """Abstract base class for all agents with Langfuse tracing.

    All agent implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, llm: BaseLLMClient):
        """Initialize the agent.

        Args:
            llm: LLM client for generating responses.
        """
        self.llm = llm
        self._settings = get_settings()
        logger.debug(f"Initialized agent: {self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent identifier.

        Returns:
            Unique name for this agent.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the agent description.

        Returns:
            Description of what this agent does.
        """
        ...

    @abstractmethod
    async def invoke(self, state: "AgentState") -> "AgentState":
        """Process state and return updated state.

        Args:
            state: Current workflow state.

        Returns:
            Updated workflow state.

        Raises:
            AgentError: If agent execution fails.
        """
        ...

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            System prompt string.
        """
        # Default implementation - subclasses should override
        return f"You are {self.name}. {self.description}"

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Helper method to call the LLM.

        Args:
            messages: Messages in OpenAI format.
            **kwargs: Additional arguments for the LLM.

        Returns:
            LLM response content.
        """
        from threads_hype_agent.models.messages import Message

        msg_objects = [Message(**m) for m in messages]
        response = await self.llm.complete(msg_objects, **kwargs)
        return response.content

    async def _execute_tool(
        self,
        tool_name: str,
        tool_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a tool with Langfuse tracing.

        Args:
            tool_name: Name of the tool for tracing.
            tool_func: The tool function to execute.
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Tool execution result.
        """
        if self._settings.langfuse_enabled:
            # Wrap with observe decorator
            observed_func = observe(as_type="tool", name=tool_name)(tool_func)
            import asyncio
            if asyncio.iscoroutinefunction(tool_func):
                return await observed_func(*args, **kwargs)
            return observed_func(*args, **kwargs)
        else:
            import asyncio
            if asyncio.iscoroutinefunction(tool_func):
                return await tool_func(*args, **kwargs)
            return tool_func(*args, **kwargs)

