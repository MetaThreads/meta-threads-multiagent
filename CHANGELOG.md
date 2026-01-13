# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-13

### Added

- Initial release of Meta Threads Multiagent system
- LangGraph-based multi-agent workflow architecture
- **Planning Agent**: Analyzes user requests and creates execution plans
- **Orchestrator Agent**: Routes tasks and evaluates agent outputs using LLM
- **Web Search Agent**: Autonomous web search with DuckDuckGo integration
- **Threads Agent**: Interacts with Meta Threads API via MCP
- **Response Agent**: Generates human-readable responses from workflow results
- MCP client integration for Threads API operations
- OpenRouter LLM client for multi-model support
- Langfuse tracing integration for observability
- FastAPI-based REST API with streaming support
- Comprehensive test suite with 71%+ coverage
- GitHub Actions CI/CD pipeline
