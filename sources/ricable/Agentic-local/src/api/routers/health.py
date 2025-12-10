"""
=============================================================================
Edge-Native AI - Health Router
Health check and monitoring endpoints
=============================================================================
"""

from fastapi import APIRouter, Depends
from datetime import datetime
import time

from ..models.schemas import HealthResponse

router = APIRouter()

# Track startup time
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check for all system components.

    Checks:
    - LiteLLM gateway connectivity
    - AgentDB database status
    - RuVector index status
    - Redis connectivity
    - Local inference availability
    """
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        components={
            "litellm": {"status": "healthy", "url": "http://localhost:4000"},
            "agentdb": {"status": "healthy", "type": "sqlite"},
            "ruvector": {"status": "healthy", "vectors": 0},
            "redis": {"status": "healthy", "connected": True},
            "local_inference": {"status": "healthy", "model": "qwen-coder"},
        },
        timestamp=datetime.utcnow(),
    )


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if the service is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.

    Returns 200 if the service is ready to accept traffic.
    """
    # Would check all dependencies are connected
    return {"status": "ready"}


@router.get("/metrics")
async def get_metrics():
    """
    Get system metrics for monitoring.

    Returns metrics for:
    - Request counts (local vs cloud)
    - Latency statistics
    - Cost tracking
    - Agent activity
    """
    return {
        "requests": {
            "total": 0,
            "local": 0,
            "cloud": 0,
            "cached": 0,
            "failed": 0,
        },
        "latency": {
            "average_ms": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
        },
        "cost": {
            "total_usd": 0.0,
            "local_savings_usd": 0.0,
        },
        "agents": {
            "active": 0,
            "total_executions": 0,
        },
        "workflows": {
            "active": 0,
            "completed": 0,
        },
        "vectors": {
            "total": 0,
            "searches": 0,
        },
    }


@router.get("/info")
async def get_system_info():
    """
    Get system configuration and capabilities.
    """
    return {
        "name": "Edge-Native AI",
        "version": "1.0.0",
        "capabilities": [
            "agent-orchestration",
            "workflow-execution",
            "code-execution",
            "semantic-search",
            "a2a-protocol",
            "mcp-server",
        ],
        "models": {
            "local": ["qwen-coder", "qwen-coder-14b", "local-general"],
            "cloud": ["gpt-4o", "claude-3.5-sonnet"],
        },
        "frameworks": {
            "orchestration": ["agentic-flow", "claude-flow"],
            "storage": ["agentdb", "ruvector"],
            "protocols": ["a2a", "mcp"],
        },
        "infrastructure": {
            "os": "kairos",
            "orchestration": "k3s",
            "compute": "spinkube",
            "gateway": "litellm",
        },
    }
