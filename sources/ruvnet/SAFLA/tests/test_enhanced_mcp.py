#!/usr/bin/env python3
"""Test the enhanced SAFLA MCP server."""

import json
import subprocess
import asyncio
import os

async def test_enhanced_mcp():
    """Test the enhanced MCP server with new tools."""
    
    # Start the MCP server
    env = os.environ.copy()
    env['SAFLA_REMOTE_URL'] = 'https://safla.fly.dev'
    
    proc = await asyncio.create_subprocess_exec(
        'python3', '/workspaces/SAFLA/safla_mcp_enhanced.py',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    async def send_request(request):
        """Send a request and get response."""
        proc.stdin.write((json.dumps(request) + '\n').encode())
        await proc.stdin.drain()
        
        response_line = await proc.stdout.readline()
        return json.loads(response_line.decode())
    
    try:
        print("Testing Enhanced MCP Server...")
        
        # Test 1: Initialize
        print("\n1. Testing initialize...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        print(f"Version: {response['result']['serverInfo']['version']}")
        
        # Test 2: List tools
        print("\n2. Testing tools/list...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        tools = response['result']['tools']
        print(f"Found {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool['name']}: {tool['description'][:60]}...")
        
        # Test 3: Test new analyze_text tool
        print("\n3. Testing analyze_text tool...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_text",
                "arguments": {
                    "text": "SAFLA is an advanced self-aware AI system with extreme optimization.",
                    "analysis_type": "all"
                }
            }
        })
        print(f"Analysis result: {response.get('result', {}).get('content', [{}])[0].get('text', 'No result')[:200]}...")
        
        # Test 4: Test batch_process tool
        print("\n4. Testing batch_process tool...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "batch_process",
                "arguments": {
                    "data": ["item1", "item2", "item3"],
                    "operation": "embed",
                    "batch_size": 256
                }
            }
        })
        print(f"Batch result: {response.get('result', {}).get('content', [{}])[0].get('text', 'No result')[:200]}...")
        
        # Test 5: Test monitor_health tool
        print("\n5. Testing monitor_health tool...")
        response = await send_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "monitor_health",
                "arguments": {
                    "include_metrics": True,
                    "include_predictions": True
                }
            }
        })
        print(f"Health status: {response.get('result', {}).get('content', [{}])[0].get('text', 'No result')[:200]}...")
        
        print("\n✅ Enhanced MCP Server tests completed!")
        
    finally:
        proc.terminate()
        await proc.wait()

if __name__ == "__main__":
    asyncio.run(test_enhanced_mcp())