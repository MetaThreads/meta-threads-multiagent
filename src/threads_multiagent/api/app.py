"""FastAPI application factory and entry point."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from threads_multiagent.api.middleware import ErrorHandlerMiddleware, LoggingMiddleware
from threads_multiagent.api.routes import chat_router, health_router
from threads_multiagent.config import get_settings
from threads_multiagent.logging import get_logger, setup_logging

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()

    # Setup logging
    setup_logging(level=settings.log_level)

    # Create app
    app = FastAPI(
        title="Threads Hype Agent API",
        description="LangGraph-based AI agent for automated Threads posting with news aggregation",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Include routers
    app.include_router(health_router)
    app.include_router(chat_router)

    logger.info("FastAPI application created")
    return app


def main() -> None:
    """Run the application with uvicorn."""
    settings = get_settings()

    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")

    uvicorn.run(
        "threads_multiagent.api.app:create_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    main()
