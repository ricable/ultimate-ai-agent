"""
=============================================================================
Edge-Native AI - FastAPI Backend
Main application entry point with A2A protocol and MCP server support
=============================================================================
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import os
import logging

from .routers import agents, workflows, a2a, health, mcp
from .services.orchestrator import EdgeAIOrchestrator
from .models.schemas import AgentCard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: EdgeAIOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator

    logger.info("Starting Edge-Native AI Backend...")

    # Initialize orchestrator
    orchestrator = EdgeAIOrchestrator(
        litellm_url=os.getenv("LITELLM_URL", "http://localhost:4000"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        db_url=os.getenv("DATABASE_URL", "sqlite:///./data/edgeai.db"),
    )
    await orchestrator.initialize()

    logger.info("Edge-Native AI Backend started successfully")

    yield

    # Cleanup
    logger.info("Shutting down Edge-Native AI Backend...")
    await orchestrator.shutdown()


# Create FastAPI application
app = FastAPI(
    title="Edge-Native AI API",
    description="""
    Decentralized Edge-Native AI Backend with:
    - Agent orchestration via ruvnet ecosystem (agentic-flow, claude-flow, agentdb, ruvector)
    - A2A (Agent-to-Agent) protocol for cross-agent communication
    - MCP (Model Context Protocol) server support
    - LiteLLM gateway with local-first routing
    - E2B secure sandbox integration
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get orchestrator
def get_orchestrator() -> EdgeAIOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return orchestrator


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["Workflows"])
app.include_router(a2a.router, tags=["A2A Protocol"])
app.include_router(mcp.router, prefix="/mcp", tags=["MCP"])


# =============================================================================
# A2A Protocol Endpoints (Well-Known)
# =============================================================================


@app.get("/.well-known/agent.json")
async def get_agent_card(orch: EdgeAIOrchestrator = Depends(get_orchestrator)):
    """
    A2A Agent Card endpoint for agent discovery.
    Returns the agent's capabilities, skills, and metadata.
    """
    return AgentCard(
        name="Edge-Native-AI",
        description="Decentralized AI agent platform with local-first inference",
        url=os.getenv("BASE_URL", "http://localhost:8000"),
        version="1.0.0",
        capabilities={
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            {
                "id": "code-generation",
                "name": "Code Generation",
                "description": "Generate code using local Qwen Coder models",
            },
            {
                "id": "code-execution",
                "name": "Code Execution",
                "description": "Execute code securely via E2B or local sandbox",
            },
            {
                "id": "semantic-search",
                "name": "Semantic Search",
                "description": "Search knowledge base using vector embeddings",
            },
            {
                "id": "workflow-orchestration",
                "name": "Workflow Orchestration",
                "description": "Execute SPARC methodology workflows",
            },
        ],
        provider={"organization": "Edge-Native-AI", "url": "https://edge-ai.local"},
    ).model_dump()


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Edge-Native AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "agent_card": "/.well-known/agent.json",
            "health": "/health",
            "agents": "/api/v1/agents",
            "workflows": "/api/v1/workflows",
            "a2a": "/a2a",
            "mcp": "/mcp",
            "docs": "/docs",
        },
    }


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development",
    )
