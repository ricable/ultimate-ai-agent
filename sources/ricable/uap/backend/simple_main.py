#!/usr/bin/env python3
"""
Simple FastAPI backend for UAP platform
Minimal version to get the basic functionality running
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="UAP - Unified Agentic Platform",
    description="A simplified version of the UAP backend for initial testing",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class ChatMessage(BaseModel):
    message: str
    agent_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    agent_id: str
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="3.0.0"
    )

# Basic chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    # Simple echo response for now
    return ChatResponse(
        response=f"Echo from {message.agent_id}: {message.message}",
        agent_id=message.agent_id,
        timestamp=datetime.now().isoformat()
    )

# Agent status endpoint
@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "agents": {
            "copilot": {"status": "active", "type": "mock"},
            "agno": {"status": "active", "type": "mock"},
            "mastra": {"status": "active", "type": "mock"}
        },
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time communication
@app.websocket("/ws/agents/{agent_id}")
async def websocket_endpoint(websocket, agent_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo the message back
            response = f"Agent {agent_id} received: {data}"
            await websocket.send_text(response)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)