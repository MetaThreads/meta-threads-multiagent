"""FastAPI middleware for error handling and logging."""

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from threads_hype_agent.exceptions import (
    AgentError,
    LLMError,
    MCPError,
    ThreadsHypeAgentError,
)
from threads_hype_agent.logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Log request and response details."""
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.3f}s"
        )

        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Handle exceptions and return appropriate responses."""
        try:
            return await call_next(request)

        except LLMError as e:
            logger.error(f"LLM error: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": "LLM service error",
                    "detail": str(e),
                    "type": "llm_error",
                },
            )

        except MCPError as e:
            logger.error(f"MCP error: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": "Threads MCP service error",
                    "detail": str(e),
                    "type": "mcp_error",
                },
            )

        except AgentError as e:
            logger.error(f"Agent error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Agent execution error",
                    "detail": str(e),
                    "type": "agent_error",
                },
            )

        except ThreadsHypeAgentError as e:
            logger.error(f"Application error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Application error",
                    "detail": str(e),
                    "type": "application_error",
                },
            )

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(e),
                    "type": "internal_error",
                },
            )
