#!/usr/bin/env python3
"""
SAFLA MCP Server - Simple Version
=================================

A minimal MCP server that connects to the remote SAFLA instance.
"""

import json
import sys
import os
import urllib.request
import urllib.parse

# Get configuration from environment
SAFLA_URL = os.environ.get("SAFLA_REMOTE_URL", "https://safla.fly.dev")

def read_json_line():
    """Read a line of JSON from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())

def write_json_line(data):
    """Write JSON to stdout."""
    print(json.dumps(data), flush=True)

def call_safla_api(method, params):
    """Call the remote SAFLA API using urllib."""
    mcp_request = {
        "jsonrpc": "2.0",
        "id": f"safla_{method}",
        "method": method,
        "params": params
    }
    
    try:
        req = urllib.request.Request(
            f"{SAFLA_URL}/api/safla",
            data=json.dumps(mcp_request).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("result", {"status": "success", "data": result})
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main MCP server loop."""
    # Read messages from stdin and respond
    while True:
        try:
            message = read_json_line()
            if not message:
                break
            
            method = message.get("method")
            params = message.get("params", {})
            msg_id = message.get("id")
            
            # Handle initialize
            if method == "initialize":
                write_json_line({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": "safla",
                            "version": "0.1.3"
                        },
                        "capabilities": {
                            "tools": {}
                        }
                    }
                })
            
            # Handle tools/list
            elif method == "tools/list":
                write_json_line({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [
                            {
                                "name": "generate_embeddings",
                                "description": "Generate embeddings using SAFLA's extreme-optimized engine (1.75M+ ops/sec)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "texts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Texts to embed"
                                        }
                                    },
                                    "required": ["texts"]
                                }
                            },
                            {
                                "name": "store_memory",
                                "description": "Store information in SAFLA's hybrid memory system",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "content": {
                                            "type": "string",
                                            "description": "Content to store"
                                        },
                                        "memory_type": {
                                            "type": "string",
                                            "enum": ["episodic", "semantic", "procedural"],
                                            "default": "episodic"
                                        }
                                    },
                                    "required": ["content"]
                                }
                            },
                            {
                                "name": "retrieve_memories",
                                "description": "Search and retrieve from SAFLA's memory system",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Search query"
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "default": 5
                                        }
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "get_performance",
                                "description": "Get SAFLA performance metrics",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        ]
                    }
                })
            
            # Handle tools/call
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                # Map tool names to API methods
                api_method = {
                    "generate_embeddings": "generate_embeddings",
                    "store_memory": "store_memory",
                    "retrieve_memories": "retrieve_memories",
                    "get_performance": "get_performance_metrics"
                }.get(tool_name)
                
                if api_method:
                    result = call_safla_api(api_method, tool_args)
                    
                    write_json_line({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                }
                            ]
                        }
                    })
                else:
                    write_json_line({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    })
            
            # Handle unknown methods
            else:
                if msg_id:  # Only respond if it's a request (has an id)
                    write_json_line({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    })
                    
        except json.JSONDecodeError:
            # Invalid JSON, ignore
            continue
        except EOFError:
            # End of input
            break
        except Exception as e:
            # Log error but continue
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    main()