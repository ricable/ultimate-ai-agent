#!/usr/bin/env python3
"""
Docker MCP HTTP/SSE Bridge
Provides HTTP and Server-Sent Events transport for Docker MCP Gateway
Enables remote access for Claude Code and Gemini clients
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Docker MCP HTTP Bridge", 
    description="HTTP/SSE transport bridge for Docker MCP Gateway",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MCPBridge:
    def __init__(self):
        self.gateway_process: Optional[subprocess.Popen] = None
        self.connected_clients: Dict[str, Any] = {}
        
    async def start_gateway(self):
        """Start the Docker MCP Gateway process"""
        try:
            cmd = ["docker", "mcp", "gateway", "run", "--verbose", "--log-calls"]
            self.gateway_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            logger.info("Docker MCP Gateway started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start gateway: {e}")
            return False
    
    async def stop_gateway(self):
        """Stop the Docker MCP Gateway process"""
        if self.gateway_process:
            self.gateway_process.terminate()
            self.gateway_process.wait()
            logger.info("Docker MCP Gateway stopped")
    
    async def send_to_gateway(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to gateway and return response"""
        if not self.gateway_process:
            raise HTTPException(status_code=503, detail="Gateway not running")
        
        try:
            # Send message to gateway
            message_str = json.dumps(message) + "\n"
            self.gateway_process.stdin.write(message_str)
            self.gateway_process.stdin.flush()
            
            # Read response
            response_line = self.gateway_process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                raise HTTPException(status_code=500, detail="No response from gateway")
                
        except Exception as e:
            logger.error(f"Gateway communication error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global bridge instance
bridge = MCPBridge()

@app.on_event("startup")
async def startup_event():
    """Initialize the MCP bridge on startup"""
    logger.info("Starting Docker MCP HTTP Bridge...")
    await bridge.start_gateway()

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Docker MCP HTTP Bridge...")
    await bridge.stop_gateway()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Docker MCP HTTP Bridge",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "gateway_running": bridge.gateway_process is not None and bridge.gateway_process.poll() is None,
        "transport": ["http", "sse"],
        "clients": ["claude-code", "gemini", "cursor", "vscode"]
    }

@app.post("/mcp")
async def mcp_http(request: Request):
    """HTTP transport endpoint for MCP messages"""
    try:
        message = await request.json()
        logger.info(f"Received HTTP MCP message: {message.get('method', 'unknown')}")
        
        response = await bridge.send_to_gateway(message)
        return response
        
    except Exception as e:
        logger.error(f"HTTP transport error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/sse")
async def mcp_sse():
    """Server-Sent Events transport endpoint"""
    async def event_generator():
        client_id = f"sse_{datetime.now().timestamp()}"
        bridge.connected_clients[client_id] = {"transport": "sse", "connected_at": datetime.now()}
        
        try:
            logger.info(f"SSE client {client_id} connected")
            
            # Send initial connection event
            yield {
                "event": "connection",
                "data": json.dumps({
                    "type": "connection_established",
                    "client_id": client_id,
                    "transport": "sse",
                    "capabilities": ["tools", "resources", "prompts"]
                })
            }
            
            # Keep connection alive and handle events
            while True:
                # In a real implementation, this would handle bidirectional communication
                # For now, send periodic heartbeats
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                }
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            logger.info(f"SSE client {client_id} disconnected")
            bridge.connected_clients.pop(client_id, None)
            raise
        except Exception as e:
            logger.error(f"SSE error for client {client_id}: {e}")
            bridge.connected_clients.pop(client_id, None)
            raise

    return EventSourceResponse(event_generator())

@app.get("/clients")
async def list_clients():
    """List connected clients"""
    return {
        "connected_clients": len(bridge.connected_clients),
        "clients": bridge.connected_clients
    }

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    try:
        # Query gateway for available tools
        message = {
            "jsonrpc": "2.0",
            "id": "list_tools",
            "method": "tools/list"
        }
        
        response = await bridge.send_to_gateway(message)
        return response
        
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Call a specific MCP tool"""
    try:
        tool_input = await request.json()
        
        message = {
            "jsonrpc": "2.0", 
            "id": f"call_{tool_name}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input
            }
        }
        
        response = await bridge.send_to_gateway(message)
        return response
        
    except Exception as e:
        logger.error(f"Failed to call tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker MCP HTTP Bridge")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Docker MCP HTTP Bridge on {args.host}:{args.port}")
    logger.info("Transport modes: HTTP POST (/mcp) and SSE (/mcp/sse)")
    logger.info("Supported clients: Claude Code, Gemini, Cursor, VS Code")
    
    uvicorn.run(
        "mcp-http-bridge:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )