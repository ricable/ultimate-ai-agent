#!/usr/bin/env python3
"""
SAFLA MCP Server Launcher
========================

This script launches the SAFLA MCP server that communicates with the remote
SAFLA instance on Fly.io using the Model Context Protocol.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAFLAMCPServer:
    """MCP Server that proxies requests to remote SAFLA instance."""
    
    def __init__(self, base_url: str = "https://safla.fly.dev"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def handle_stdio(self):
        """Handle MCP communication over stdio."""
        logger.info(f"SAFLA MCP Server starting (remote: {self.base_url})")
        
        # Main message loop
        while True:
            try:
                line = await self.read_line()
                if not line:
                    break
                    
                request = json.loads(line)
                response = await self.handle_request(request)
                await self.send_response(response)
                
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                await self.send_error_response(str(e))
    
    async def read_line(self) -> Optional[str]:
        """Read a line from stdin."""
        loop = asyncio.get_event_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        return line.strip() if line else None
    
    async def send_response(self, response: Dict[str, Any]):
        """Send response to stdout."""
        print(json.dumps(response))
        sys.stdout.flush()
    
    async def send_error_response(self, error_message: str, request_id: str = None):
        """Send error response."""
        await self.send_response({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": error_message
            }
        })
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "0.1.0",
                    "serverInfo": {
                        "name": "safla-mcp-server",
                        "version": "0.1.3"
                    },
                    "capabilities": {
                        "tools": {
                            "list": True
                        }
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "generate_embeddings",
                            "description": "Generate embeddings using SAFLA's optimized neural engine",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "texts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of texts to generate embeddings for"
                                    },
                                    "batch_size": {
                                        "type": "integer",
                                        "default": 256,
                                        "description": "Batch size for processing"
                                    }
                                },
                                "required": ["texts"]
                            }
                        },
                        {
                            "name": "store_memory",
                            "description": "Store data in SAFLA's hybrid memory system",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "Content to store in memory"
                                    },
                                    "memory_type": {
                                        "type": "string",
                                        "enum": ["episodic", "semantic", "procedural"],
                                        "default": "episodic",
                                        "description": "Type of memory storage"
                                    }
                                },
                                "required": ["content"]
                            }
                        },
                        {
                            "name": "retrieve_memories",
                            "description": "Retrieve memories from SAFLA's hybrid memory system",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Query to search memories"
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "default": 5,
                                        "description": "Maximum number of memories to retrieve"
                                    }
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "get_performance_metrics",
                            "description": "Get current SAFLA performance metrics",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "run_benchmark",
                            "description": "Run performance benchmark on SAFLA",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "benchmark_type": {
                                        "type": "string",
                                        "enum": ["embedding_performance", "memory_operations", "full_system"],
                                        "default": "embedding_performance",
                                        "description": "Type of benchmark to run"
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            # Call remote SAFLA API
            result = await self.call_remote_safla(tool_name, tool_args)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def call_remote_safla(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call remote SAFLA API and return result."""
        try:
            # Map tool names to API methods
            method_map = {
                "generate_embeddings": "generate_embeddings",
                "store_memory": "store_memory",
                "retrieve_memories": "retrieve_memories",
                "get_performance_metrics": "get_performance_metrics",
                "run_benchmark": "run_benchmark"
            }
            
            if tool_name not in method_map:
                return {
                    "error": f"Unknown tool: {tool_name}"
                }
            
            # Create MCP request for remote SAFLA
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"tool_{tool_name}",
                "method": method_map[tool_name],
                "params": arguments
            }
            
            async with self.session.post(
                f"{self.base_url}/api/safla",
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result.get("result", result), indent=2)
                            }
                        ]
                    }
                else:
                    error_text = await response.text()
                    return {
                        "error": f"Remote API error: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Error calling remote SAFLA: {e}")
            return {
                "error": str(e)
            }

async def main():
    """Main entry point for MCP server."""
    async with SAFLAMCPServer() as server:
        await server.handle_stdio()

if __name__ == "__main__":
    asyncio.run(main())