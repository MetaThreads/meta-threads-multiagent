"""Chat route with SSE streaming."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from threads_multiagent.api.dependencies import get_workflow_runner
from threads_multiagent.logging import get_logger
from threads_multiagent.models.api import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


async def generate_sse_events(
    request: ChatRequest,
) -> AsyncIterator[dict[str, Any]]:
    """Generate SSE events from workflow execution.

    Args:
        request: Chat request with messages.

    Yields:
        SSE event data dictionaries.
    """
    try:
        workflow_runner = get_workflow_runner()

        # Convert messages to dict format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Stream workflow execution
        async for event in workflow_runner.stream(messages):
            node_name = event.get("node", "")
            state = event.get("state", {})

            # Emit agent status event
            yield {
                "event": "agent",
                "data": json.dumps(
                    {
                        "type": "agent",
                        "agent_name": node_name,
                        "status": "completed",
                    }
                ),
            }

            # Emit content if there are new messages
            messages_in_state = state.get("messages", [])
            if messages_in_state:
                last_msg = messages_in_state[-1]
                if last_msg.get("role") == "assistant":
                    yield {
                        "event": "token",
                        "data": json.dumps(
                            {
                                "type": "token",
                                "content": last_msg.get("content", ""),
                            }
                        ),
                    }

            # Emit tool calls if threads agent executed
            if node_name == "threads":
                threads_results = state.get("threads_results", [])
                for result in threads_results:
                    yield {
                        "event": "tool_call",
                        "data": json.dumps(
                            {
                                "type": "tool_call",
                                "tool_name": "threads_action",
                                "tool_result": result.get("result"),
                            }
                        ),
                    }

        # Emit done event
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "type": "done",
                    "content": "Workflow completed successfully",
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error in SSE stream: {e}")
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "type": "error",
                    "error": str(e),
                }
            ),
        }


@router.post("/chat")
async def chat_stream(
    request: ChatRequest,
) -> EventSourceResponse:
    """Stream chat responses via Server-Sent Events.

    Args:
        request: Chat request with conversation messages.

    Returns:
        SSE response stream.
    """
    logger.info(f"Chat request received with {len(request.messages)} messages")
    return EventSourceResponse(generate_sse_events(request))


@router.post("/chat/sync")
async def chat_sync(
    request: ChatRequest,
) -> ChatResponse:
    """Non-streaming chat endpoint.

    Args:
        request: Chat request with conversation messages.

    Returns:
        Complete chat response.
    """
    logger.info(f"Sync chat request received with {len(request.messages)} messages")

    workflow_runner = get_workflow_runner()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    final_state = await workflow_runner.run(messages)

    # Extract final content
    final_messages = final_state.get("messages", [])
    content = ""
    if final_messages:
        last_msg = final_messages[-1]
        if last_msg.get("role") == "assistant":
            content = last_msg.get("content", "")

    # Build agent trace
    agent_trace = []
    plan = final_state.get("plan")
    if plan is not None:
        for step in plan.get("steps", []):
            agent_trace.append(
                {
                    "agent": step.get("agent"),
                    "action": step.get("action"),
                    "completed": step.get("completed"),
                    "result": step.get("result"),
                }
            )

    return ChatResponse(content=content, agent_trace=agent_trace)
