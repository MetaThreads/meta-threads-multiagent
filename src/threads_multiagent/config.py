"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenRouter Configuration
    openrouter_api_key: str
    openrouter_model: str = "google/gemini-2.5-flash-lite"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Threads MCP Configuration
    threads_mcp_url: str = "https://uncertain-crimson-rodent.fastmcp.app/mcp"
    threads_bearer_token: str

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Agent Configuration
    max_agent_iterations: int = 30
    agent_timeout_seconds: int = 120

    # Langfuse Tracing
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
