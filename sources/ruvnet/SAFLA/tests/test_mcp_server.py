#!/usr/bin/env python3
"""Test the SAFLA MCP server locally."""

import json
import subprocess
import asyncio

async def test_mcp_server():
    """Test the MCP server with sample requests."""
    
    # Start the MCP server
    proc = await asyncio.create_subprocess_exec(
        'python3', '/workspaces/SAFLA/safla_mcp_server.py',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={'SAFLA_REMOTE_URL': 'https://safla.fly.dev'}
    )
    
    async def send_request(request):
        """Send a request and get response."""
        proc.stdin.write((json.dumps(request) + '\n').encode())
        await proc.stdin.drain()
        
        response_line = await proc.stdout.readline()
        return json.loads(response_line.decode())
    
    try:
        print("Testing MCP Server...")
        
        # Test 1: Initialize
        print("\n1. Testing initialize...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Test 2: List tools
        print("\n2. Testing tools/list...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Test 3: Call a tool
        print("\n3. Testing tools/call (get_performance)...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_performance",
                "arguments": {}
            }
        })
        print(f"Response: {json.dumps(response, indent=2)}")
        
        print("\n✅ MCP Server tests completed!")
        
    finally:
        proc.terminate()
        await proc.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())