"""Health check route."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy"}


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        API information.
    """
    return {
        "name": "Threads Hype Agent API",
        "version": "0.1.0",
        "status": "running",
    }
