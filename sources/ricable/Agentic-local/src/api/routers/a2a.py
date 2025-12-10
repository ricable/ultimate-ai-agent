"""
=============================================================================
Edge-Native AI - A2A Protocol Router
Agent-to-Agent protocol endpoints (Google A2A specification)
=============================================================================
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any
import json
import asyncio

from ..models.schemas import A2ATaskRequest, A2ATaskResponse

router = APIRouter()


@router.post("/a2a")
async def a2a_endpoint(request: Request):
    """
    A2A JSON-RPC 2.0 endpoint.

    Supports the following methods:
    - tasks/send: Send a task to this agent
    - tasks/get: Get task status
    - tasks/cancel: Cancel a running task
    - tasks/sendSubscribe: Subscribe to task updates (streaming)
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            },
        )

    # Validate JSON-RPC
    if body.get("jsonrpc") != "2.0":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "Invalid JSON-RPC version"},
            }
        )

    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    # Route to appropriate handler
    handlers = {
        "tasks/send": handle_task_send,
        "tasks/get": handle_task_get,
        "tasks/cancel": handle_task_cancel,
        "tasks/sendSubscribe": handle_task_subscribe,
    }

    handler = handlers.get(method)
    if not handler:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        )

    try:
        result = await handler(params)
        return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": result})
    except Exception as e:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)},
            }
        )


async def handle_task_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tasks/send request.

    Receives a task from another agent and processes it using
    the local agent orchestration.
    """
    task_id = params.get("id", f"task-{hash(str(params))}")
    message = params.get("message", {})
    metadata = params.get("metadata", {})

    # Extract text from message parts
    parts = message.get("parts", [])
    text_content = "\n".join(
        part.get("text", "") for part in parts if part.get("type") == "text"
    )

    # Process the task (placeholder - would use orchestrator)
    result = f"Processed task: {text_content[:100]}..."

    return {
        "id": task_id,
        "status": {"state": "completed", "timestamp": "2024-01-01T00:00:00Z"},
        "artifacts": [{"parts": [{"type": "text", "text": result}]}],
    }


async def handle_task_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tasks/get request.

    Returns the current status of a task.
    """
    task_id = params.get("id")
    if not task_id:
        raise ValueError("Task ID required")

    # Would look up task in AgentDB
    return {
        "id": task_id,
        "status": {"state": "pending", "timestamp": "2024-01-01T00:00:00Z"},
    }


async def handle_task_cancel(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tasks/cancel request.

    Cancels a running task.
    """
    task_id = params.get("id")
    if not task_id:
        raise ValueError("Task ID required")

    # Would cancel task
    return {"success": True}


async def handle_task_subscribe(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tasks/sendSubscribe request.

    Sets up SSE streaming for task updates.
    """
    task_id = params.get("id", f"task-{hash(str(params))}")

    return {"streamUrl": f"/a2a/stream/{task_id}", "protocol": "sse"}


@router.get("/a2a/stream/{task_id}")
async def a2a_stream(task_id: str):
    """
    SSE streaming endpoint for task updates.

    Streams task progress updates as Server-Sent Events.
    """

    async def event_generator():
        # Initial connection
        yield f"event: connected\ndata: {json.dumps({'taskId': task_id})}\n\n"

        # Simulate task progress (would be real updates from orchestrator)
        states = ["working", "working", "completed"]
        for i, state in enumerate(states):
            await asyncio.sleep(1)
            yield f"event: status\ndata: {json.dumps({'state': state, 'progress': (i + 1) / len(states)})}\n\n"

        # Final result
        yield f"event: artifact\ndata: {json.dumps({'type': 'text', 'text': 'Task completed successfully'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/a2a/agents")
async def list_known_agents():
    """
    List agents discovered via A2A protocol.

    Returns agents found through Kubernetes service discovery,
    DNS, or static configuration.
    """
    return {
        "agents": [],
        "discovery_method": "kubernetes",
        "last_refresh": "2024-01-01T00:00:00Z",
    }


@router.post("/a2a/discover")
async def trigger_discovery():
    """
    Manually trigger agent discovery.
    """
    return {"status": "discovery_started"}


@router.post("/a2a/send")
async def send_to_agent(
    agent_url: str,
    task: A2ATaskRequest,
):
    """
    Send a task to another agent via A2A protocol.
    """
    # Would use A2AProtocol to send task
    return {"task_id": task.id or "generated-id", "status": "sent", "target": agent_url}
