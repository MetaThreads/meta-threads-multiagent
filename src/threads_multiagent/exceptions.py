"""Custom exceptions for the Threads Hype Agent."""


class ThreadsHypeAgentError(Exception):
    """Base exception for all Threads Hype Agent errors."""

    pass


# LLM Errors
class LLMError(ThreadsHypeAgentError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""

    pass


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""

    pass


class LLMResponseError(LLMError):
    """Invalid or unexpected response from LLM."""

    pass


# MCP Errors
class MCPError(ThreadsHypeAgentError):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""

    pass


class MCPToolError(MCPError):
    """Error executing MCP tool."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


# News Fetcher Errors
class NewsFetcherError(ThreadsHypeAgentError):
    """Base exception for news fetcher errors."""

    pass


class NewsSourceError(NewsFetcherError):
    """Error fetching from news source."""

    def __init__(self, source: str, message: str):
        self.source = source
        super().__init__(f"News source '{source}' error: {message}")


# Agent Errors
class AgentError(ThreadsHypeAgentError):
    """Base exception for agent errors."""

    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    pass


class PlanningError(AgentError):
    """Error during planning phase."""

    pass


class OrchestrationError(AgentError):
    """Error during orchestration."""

    pass


# Workflow Errors
class WorkflowError(ThreadsHypeAgentError):
    """Base exception for workflow errors."""

    pass


class MaxIterationsError(WorkflowError):
    """Maximum workflow iterations exceeded."""

    pass
