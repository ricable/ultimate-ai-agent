#!/usr/bin/env python3
"""
Example MCP client with JWT authentication support.

This demonstrates how to authenticate and interact with the SAFLA MCP server.
"""

import json
import sys
import asyncio
from typing import Dict, Any, Optional


class AuthenticatedMCPClient:
    """MCP client with JWT authentication support."""
    
    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.request_id = 0
    
    def _next_request_id(self) -> int:
        """Generate next request ID."""
        self.request_id += 1
        return self.request_id
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a request to the MCP server via stdio."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params or {}
        }
        
        # Add authentication token if available and not a public method
        public_methods = ["initialize", "auth/login", "auth/refresh", "health/check"]
        if self.access_token and method not in public_methods:
            request["params"]["auth_token"] = self.access_token
        
        # Send request
        print(json.dumps(request), flush=True)
        
        # Read response (in real implementation, this would be async)
        # For demo purposes, we'll simulate the response
        return self._simulate_response(method, params)
    
    def _simulate_response(self, method: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate server responses for demonstration."""
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "safla-mcp-server", "version": "2.0.0"}
                }
            }
        elif method == "auth/login":
            # Demo login response
            if params.get("username") == "developer" and params.get("password") == "dev123":
                return {
                    "jsonrpc": "2.0",
                    "id": self.request_id,
                    "result": {
                        "success": True,
                        "access_token": "demo-access-token",
                        "refresh_token": "demo-refresh-token",
                        "expires_in": 3600,
                        "user": {
                            "id": "user_002",
                            "username": "developer",
                            "roles": ["developer"],
                            "permissions": ["tools:read", "tools:execute"]
                        }
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": self.request_id,
                    "error": {
                        "code": -32000,
                        "message": "Invalid username or password"
                    }
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "result": {"status": "success", "method": method}
            }
    
    async def initialize(self) -> bool:
        """Initialize connection to MCP server."""
        response = await self.send_request("initialize")
        return "result" in response
    
    async def login(self, username: str, password: str) -> bool:
        """Authenticate with the MCP server."""
        response = await self.send_request("auth/login", {
            "username": username,
            "password": password
        })
        
        if "result" in response and response["result"].get("success"):
            result = response["result"]
            self.access_token = result["access_token"]
            self.refresh_token = result["refresh_token"]
            print(f"✓ Logged in as {result['user']['username']}")
            print(f"  Roles: {', '.join(result['user']['roles'])}")
            print(f"  Token expires in: {result['expires_in']} seconds")
            return True
        else:
            error = response.get("error", {}).get("message", "Unknown error")
            print(f"✗ Login failed: {error}")
            return False
    
    async def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self.refresh_token:
            print("✗ No refresh token available")
            return False
        
        response = await self.send_request("auth/refresh", {
            "refresh_token": self.refresh_token
        })
        
        if "result" in response and response["result"].get("success"):
            self.access_token = response["result"]["access_token"]
            print("✓ Access token refreshed")
            return True
        else:
            print("✗ Token refresh failed")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        response = await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if "result" in response:
            return response["result"]
        else:
            error = response.get("error", {})
            raise Exception(f"Tool call failed: {error.get('message', 'Unknown error')}")
    
    async def list_tools(self) -> list:
        """List available tools."""
        response = await self.send_request("tools/list")
        return response.get("result", {}).get("tools", [])
    
    async def logout(self) -> bool:
        """Logout and invalidate tokens."""
        if not self.access_token:
            return True
        
        response = await self.send_request("auth/logout", {
            "token": self.access_token
        })
        
        if "result" in response and response["result"].get("success"):
            self.access_token = None
            self.refresh_token = None
            print("✓ Logged out successfully")
            return True
        else:
            print("✗ Logout failed")
            return False


async def main():
    """Demonstrate MCP client usage with authentication."""
    print("=== SAFLA MCP Client with JWT Authentication ===\n")
    
    # Create client
    client = AuthenticatedMCPClient()
    
    # Initialize connection
    print("1. Initializing MCP connection...")
    if await client.initialize():
        print("✓ Connected to MCP server\n")
    else:
        print("✗ Failed to connect\n")
        return
    
    # Login
    print("2. Authenticating...")
    username = input("Username (try 'developer'): ") or "developer"
    password = input("Password (try 'dev123'): ") or "dev123"
    
    if not await client.login(username, password):
        return
    
    print("\n3. Calling authenticated endpoints...")
    
    try:
        # List tools
        print("\n- Listing available tools...")
        tools = await client.list_tools()
        print(f"✓ Found {len(tools)} tools")
        
        # Call a tool
        print("\n- Validating installation...")
        result = await client.call_tool("validate_installation", {})
        print(f"✓ Installation status: {result.get('status', 'unknown')}")
        
        # Simulate token refresh
        print("\n4. Refreshing access token...")
        if await client.refresh_access_token():
            print("✓ Ready for more requests")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    finally:
        # Logout
        print("\n5. Logging out...")
        await client.logout()
    
    print("\n=== Session complete ===")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())