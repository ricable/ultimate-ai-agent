"""
Edge-Native AI SaaS Backend
FastAPI Control Plane for Distributed Agent Orchestration

This module provides the central API gateway for the decentralized
edge-native AI architecture, handling:
- User authentication and authorization
- Agent lifecycle management
- Request routing to SpinKube/Wasm agents
- A2A protocol coordination
- E2B sandbox integration
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
import asyncio
import httpx
import uuid
import json
import os

# Local imports
from .agents import AgentOrchestrator, AgentConfig
from .protocols import A2AProtocol, MCPServer
from .sandbox import E2BSandbox, LocalWasmSandbox
from .gateway import LiteLLMRouter
from .database import Database, VectorStore
from .auth import get_current_user, User


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    # Startup
    app.state.db = await Database.connect()
    app.state.vector_store = VectorStore(
        backend="ruvector",
        dimension=384,
        index_type="hnsw"
    )
    app.state.llm_router = LiteLLMRouter()
    app.state.a2a_protocol = A2AProtocol()
    app.state.agent_orchestrator = AgentOrchestrator()

    # Initialize sandbox environments
    app.state.e2b_sandbox = E2BSandbox(api_key=os.getenv("E2B_API_KEY"))
    app.state.wasm_sandbox = LocalWasmSandbox()

    yield

    # Shutdown
    await app.state.db.disconnect()
    await app.state.e2b_sandbox.cleanup()


app = FastAPI(
    title="Edge-Native AI SaaS",
    description="Decentralized AI Agent Platform with Kairos/SpinKube/E2B",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Data Models
# ============================================================================

class AgentCreateRequest(BaseModel):
    """Request to create a new AI agent."""
    name: str = Field(..., description="Unique agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(default_factory=list)
    model: str = Field(default="gpt-4o", description="LLM model to use")
    tools: List[str] = Field(default_factory=list)
    runtime: str = Field(default="wasm", description="Runtime: wasm, container, e2b")


class AgentTaskRequest(BaseModel):
    """Request to execute a task on an agent."""
    agent_id: str
    task: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False
    sandbox_mode: str = Field(default="auto", description="auto, e2b, local, wasm")


class A2AMessage(BaseModel):
    """Agent-to-Agent communication message."""
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None


class CodeExecutionRequest(BaseModel):
    """Request to execute code in a sandbox."""
    code: str
    language: str = "python"
    timeout: int = 30
    sandbox_type: str = Field(default="e2b", description="e2b, wasm, docker")
    dependencies: List[str] = Field(default_factory=list)


# ============================================================================
# Health and Discovery Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Edge-Native AI SaaS",
        "version": "1.0.0",
        "status": "healthy",
        "protocols": ["a2a", "mcp"],
        "runtimes": ["spinkube", "e2b", "docker"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes probes."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "connected",
            "vector_store": "connected",
            "llm_router": "ready"
        }
    }


@app.get("/.well-known/agent.json")
async def agent_card():
    """A2A Protocol Agent Card for service discovery."""
    return {
        "name": "edge-native-saas",
        "version": "1.0.0",
        "description": "Decentralized AI Agent Platform",
        "capabilities": [
            "code-execution",
            "data-analysis",
            "document-processing",
            "agent-orchestration"
        ],
        "endpoints": {
            "tasks": "/api/v1/agents/tasks",
            "a2a": "/api/v1/a2a",
            "mcp": "/api/v1/mcp"
        },
        "authentication": {
            "type": "bearer",
            "oauth2_url": "/oauth2/token"
        },
        "protocols": {
            "a2a": "1.0",
            "mcp": "2024-11-05"
        }
    }


# ============================================================================
# Agent Management Endpoints
# ============================================================================

@app.post("/api/v1/agents", status_code=201)
async def create_agent(
    request: AgentCreateRequest,
    user: User = Depends(get_current_user)
):
    """Create a new AI agent instance."""
    agent_id = str(uuid.uuid4())

    config = AgentConfig(
        id=agent_id,
        name=request.name,
        description=request.description,
        capabilities=request.capabilities,
        model=request.model,
        tools=request.tools,
        runtime=request.runtime,
        owner_id=user.id
    )

    # Deploy agent based on runtime
    orchestrator = app.state.agent_orchestrator
    if request.runtime == "wasm":
        await orchestrator.deploy_spinapp(config)
    elif request.runtime == "container":
        await orchestrator.deploy_container(config)

    # Store agent metadata
    await app.state.db.create_agent(config)

    return {
        "id": agent_id,
        "name": request.name,
        "status": "deployed",
        "runtime": request.runtime,
        "endpoint": f"/api/v1/agents/{agent_id}"
    }


@app.get("/api/v1/agents")
async def list_agents(
    user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """List all agents owned by the user."""
    agents = await app.state.db.list_agents(
        owner_id=user.id,
        limit=limit,
        offset=offset
    )
    return {"agents": agents, "total": len(agents)}


@app.post("/api/v1/agents/tasks")
async def execute_agent_task(
    request: AgentTaskRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Execute a task on an agent."""
    agent = await app.state.db.get_agent(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Determine sandbox mode
    sandbox_mode = request.sandbox_mode
    if sandbox_mode == "auto":
        # Use E2B for complex code execution, Wasm for simple logic
        if "execute_code" in request.task.lower():
            sandbox_mode = "e2b"
        else:
            sandbox_mode = "wasm"

    # Route to appropriate LLM via LiteLLM
    llm_response = await app.state.llm_router.chat(
        model=agent.model,
        messages=[
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": request.task}
        ],
        stream=request.stream
    )

    if request.stream:
        return StreamingResponse(
            stream_response(llm_response),
            media_type="text/event-stream"
        )

    return {
        "task_id": str(uuid.uuid4()),
        "agent_id": request.agent_id,
        "status": "completed",
        "result": llm_response,
        "sandbox_used": sandbox_mode
    }


async def stream_response(response) -> AsyncGenerator[str, None]:
    """Stream SSE responses from LLM."""
    async for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# Code Execution Endpoints (E2B Integration)
# ============================================================================

@app.post("/api/v1/execute")
async def execute_code(
    request: CodeExecutionRequest,
    user: User = Depends(get_current_user)
):
    """Execute code in a secure sandbox environment."""
    execution_id = str(uuid.uuid4())

    if request.sandbox_type == "e2b":
        # Use E2B Firecracker microVM for full library support
        sandbox = app.state.e2b_sandbox
        result = await sandbox.execute(
            code=request.code,
            language=request.language,
            timeout=request.timeout,
            dependencies=request.dependencies
        )
    elif request.sandbox_type == "wasm":
        # Use local Wasm sandbox for simple execution
        sandbox = app.state.wasm_sandbox
        result = await sandbox.execute(
            code=request.code,
            timeout=request.timeout
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sandbox type: {request.sandbox_type}"
        )

    return {
        "execution_id": execution_id,
        "status": "completed",
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "artifacts": result.get("artifacts", []),
        "execution_time_ms": result.get("execution_time_ms", 0)
    }


# ============================================================================
# A2A Protocol Endpoints
# ============================================================================

@app.post("/api/v1/a2a/send")
async def send_a2a_message(
    message: A2AMessage,
    user: User = Depends(get_current_user)
):
    """Send a message to another agent using A2A protocol."""
    a2a = app.state.a2a_protocol

    # Discover target agent
    target_agent = await a2a.discover_agent(message.to_agent)
    if not target_agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {message.to_agent} not found in network"
        )

    # Send message via JSON-RPC 2.0
    response = await a2a.send_message(
        target=target_agent,
        message_type=message.message_type,
        payload=message.payload,
        correlation_id=message.correlation_id
    )

    return {
        "status": "delivered",
        "correlation_id": response.get("correlation_id"),
        "response": response.get("result")
    }


@app.get("/api/v1/a2a/discover")
async def discover_agents(
    capability: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Discover agents in the network by capability."""
    a2a = app.state.a2a_protocol

    agents = await a2a.discover_agents(capability=capability)

    return {
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "capabilities": agent.capabilities,
                "endpoint": agent.endpoint
            }
            for agent in agents
        ]
    }


# ============================================================================
# MCP (Model Context Protocol) Endpoints
# ============================================================================

@app.get("/api/v1/mcp/tools")
async def list_mcp_tools(user: User = Depends(get_current_user)):
    """List available MCP tools."""
    return {
        "tools": [
            {
                "name": "database_query",
                "description": "Query the PostgreSQL database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            },
            {
                "name": "vector_search",
                "description": "Semantic search in vector store",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10}
                    }
                }
            },
            {
                "name": "file_read",
                "description": "Read a file from storage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }
            }
        ]
    }


@app.post("/api/v1/mcp/tools/{tool_name}")
async def call_mcp_tool(
    tool_name: str,
    request: Request,
    user: User = Depends(get_current_user)
):
    """Execute an MCP tool."""
    body = await request.json()

    if tool_name == "database_query":
        result = await app.state.db.execute_query(body.get("query"))
    elif tool_name == "vector_search":
        result = await app.state.vector_store.search(
            query=body.get("query"),
            k=body.get("k", 10)
        )
    else:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")

    return {"result": result}


# ============================================================================
# Vector Store Endpoints (RuVector Integration)
# ============================================================================

@app.post("/api/v1/vectors/insert")
async def insert_vectors(
    request: Request,
    user: User = Depends(get_current_user)
):
    """Insert vectors into the RuVector store."""
    body = await request.json()

    await app.state.vector_store.insert(
        vectors=body.get("vectors"),
        metadata=body.get("metadata", [])
    )

    return {"status": "inserted", "count": len(body.get("vectors", []))}


@app.post("/api/v1/vectors/search")
async def search_vectors(
    request: Request,
    user: User = Depends(get_current_user)
):
    """Search for similar vectors."""
    body = await request.json()

    results = await app.state.vector_store.search(
        query_vector=body.get("query"),
        k=body.get("k", 10),
        filter=body.get("filter")
    )

    return {"results": results}


# ============================================================================
# LLM Gateway Endpoints
# ============================================================================

@app.post("/api/v1/chat/completions")
async def chat_completions(
    request: Request,
    user: User = Depends(get_current_user)
):
    """OpenAI-compatible chat completions endpoint via LiteLLM."""
    body = await request.json()

    response = await app.state.llm_router.chat(
        model=body.get("model", "gpt-4o"),
        messages=body.get("messages", []),
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens"),
        stream=body.get("stream", False)
    )

    if body.get("stream"):
        return StreamingResponse(
            stream_response(response),
            media_type="text/event-stream"
        )

    return response


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
