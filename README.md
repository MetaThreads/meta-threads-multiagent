# Meta Threads Multiagent

LangGraph-based AI agent for automated Threads posting with web search capabilities. Built on top of [Meta Threads MCP](https://github.com/MetaThreads/meta-threads-mcp).

## Features

- **Multi-agent architecture** using LangGraph for orchestrated workflows
- **Planning agent** that breaks down user requests into actionable steps
- **Web search agent** for autonomous information retrieval via DuckDuckGo
- **Threads agent** for posting to Meta Threads via MCP
- **Response agent** for generating human-readable summaries
- **OpenRouter integration** for LLM access through OpenAI-compatible interface
- **Langfuse tracing** for observability and debugging
- **FastAPI server** with SSE streaming support

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│   Planning  │  ← Creates execution plan
│    Agent    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Orchestrator│  ← Routes to agents in loop
│    Agent    │◄────────────────────┐
└──────┬──────┘                     │
       │                            │
   ┌───┴───┐                        │
   ▼       ▼                        │
┌─────┐ ┌─────────┐                 │
│ Web │ │ Threads │─────────────────┘
│Search│ │  Agent  │
└─────┘ └────┬────┘
             │
             ▼
      ┌────────────┐
      │Threads MCP │
      └────────────┘
             │
             ▼
      ┌────────────┐
      │  Response  │  ← Generates final output
      │   Agent    │
      └────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/MetaThreads/meta-threads-multiagent.git
cd meta-threads-multiagent

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:

```env
# OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-xxx

# Threads MCP configuration
THREADS_MCP_URL=https://your-mcp-server.fastmcp.app/mcp
THREADS_BEARER_TOKEN=your_access_token:your_user_id

# Optional: Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Usage

### Start the API server

```bash
# Using the CLI
threads-hype-agent

# Or directly with Python
python -m threads_hype_agent.api.app
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### `POST /chat` - Streaming chat

Stream responses via Server-Sent Events:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Post about the latest AI news"}]}'
```

#### `POST /chat/sync` - Synchronous chat

Get a complete response:

```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Post about the latest AI news"}]}'
```

#### `GET /health` - Health check

```bash
curl http://localhost:8000/health
```

### Python Client Example

```python
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat/sync",
            json={
                "messages": [
                    {"role": "user", "content": "Find trending tech news and post about it"}
                ]
            },
            timeout=120.0,
        )
        print(response.json())

asyncio.run(main())
```

See `examples/client_example.py` for a complete streaming example.

## Project Structure

```
src/threads_hype_agent/
├── agents/           # Agent implementations
│   ├── base.py       # BaseAgent ABC with Langfuse tracing
│   ├── planning.py   # Planning agent
│   ├── orchestrator.py # Orchestrator agent
│   ├── web_search.py # Web search agent
│   ├── threads.py    # Threads MCP agent
│   └── response.py   # Response generation agent
├── api/              # FastAPI application
│   ├── app.py        # App factory
│   ├── routes/       # API routes
│   └── middleware.py
├── graph/            # LangGraph workflow
│   ├── state.py      # AgentState TypedDict
│   ├── nodes.py      # Node functions
│   ├── edges.py      # Routing logic
│   └── workflow.py   # Workflow builder
├── llm/              # LLM abstraction
│   ├── base.py       # BaseLLMClient ABC
│   └── openrouter.py # OpenRouter implementation
├── mcp/              # MCP client
│   └── client.py     # FastMCP client
├── search/           # Web search
│   ├── base.py       # BaseWebSearch ABC
│   └── duckduckgo.py # DuckDuckGo implementation
├── tracing/          # Observability
│   └── langfuse_tracer.py
├── models/           # Pydantic models
├── prompts/          # Agent prompts
├── config.py         # Settings
├── exceptions.py     # Custom exceptions
└── logging.py        # Logging setup
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src tests

# Run type checking
uv run mypy src
```

## License

MIT License - see [LICENSE](LICENSE) for details.
