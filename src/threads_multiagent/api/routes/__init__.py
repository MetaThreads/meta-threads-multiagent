"""API route handlers."""

from threads_multiagent.api.routes.chat import router as chat_router
from threads_multiagent.api.routes.health import router as health_router

__all__ = [
    "chat_router",
    "health_router",
]
