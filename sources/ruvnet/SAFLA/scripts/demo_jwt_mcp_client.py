#!/usr/bin/env python3
"""Demo MCP client with JWT authentication."""

import json
import sys


def send_request(request):
    """Send request to MCP server via stdio."""
    print(json.dumps(request), flush=True)
    # In a real client, you would read the response from stdin
    print("(Response would be read from stdin in actual usage)", file=sys.stderr)


def demo_jwt_flow():
    """Demonstrate JWT authentication flow with MCP server."""
    
    print("=== MCP JWT Authentication Demo ===", file=sys.stderr)
    print("This demo shows the JSON-RPC requests for JWT auth", file=sys.stderr)
    print("Run the MCP server with: python -m safla.mcp_stdio_server", file=sys.stderr)
    print("\n", file=sys.stderr)
    
    # 1. Initialize connection
    print("1. Initialize MCP connection:", file=sys.stderr)
    send_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "demo-client",
                "version": "1.0.0"
            }
        }
    })
    
    # 2. Login to get tokens
    print("\n2. Login with credentials:", file=sys.stderr)
    send_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "auth/login",
        "params": {
            "username": "developer",
            "password": "dev123"
        }
    })
    print("Expected response: access_token, refresh_token, user info", file=sys.stderr)
    
    # 3. List tools with authentication
    print("\n3. List tools (authenticated request):", file=sys.stderr)
    # In practice, you'd use the token from the login response
    example_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    send_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/list",
        "params": {
            "headers": {
                "Authorization": f"Bearer {example_token}"
            }
        }
    })
    
    # 4. Call a tool with authentication
    print("\n4. Call a tool (authenticated request):", file=sys.stderr)
    send_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "get_system_info",
            "arguments": {},
            "headers": {
                "Authorization": f"Bearer {example_token}"
            }
        }
    })
    
    # 5. Refresh token
    print("\n5. Refresh access token:", file=sys.stderr)
    example_refresh_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    send_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "auth/refresh",
        "params": {
            "refresh_token": example_refresh_token
        }
    })
    
    # 6. Attempt unauthorized request
    print("\n6. Attempt request without token (should fail):", file=sys.stderr)
    send_request({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/list",
        "params": {}
    })
    print("Expected: Authentication error", file=sys.stderr)
    
    print("\n=== Demo Users ===", file=sys.stderr)
    print("admin/admin123 - Full access", file=sys.stderr)
    print("developer/dev123 - Read/write access", file=sys.stderr)
    print("reader/read123 - Read-only access", file=sys.stderr)


if __name__ == "__main__":
    demo_jwt_flow()