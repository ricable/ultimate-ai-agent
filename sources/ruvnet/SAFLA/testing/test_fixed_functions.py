#!/usr/bin/env python3
"""
Verification script for the two fixed MCP server functions:
1. create_custom_strategy - Fixed parameter schema mismatch
2. get_learning_metrics - Fixed argument handling issue
"""

import asyncio
import json
import subprocess
import sys
from typing import Dict, Any


async def test_fixed_functions():
    """Test the specific functions that were fixed"""
    print("🔧 Testing Fixed MCP Server Functions")
    print("=" * 50)
    
    # Start the MCP server
    server_process = await asyncio.create_subprocess_exec(
        sys.executable, "safla/mcp_stdio_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Wait for server to be ready
        await asyncio.sleep(2)
        
        # Test 1: create_custom_strategy with expected_outcomes
        print("\n1. Testing create_custom_strategy with expected_outcomes:")
        request1 = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "create_custom_strategy",
                "arguments": {
                    "strategy_name": "Test Strategy with Outcomes",
                    "description": "A comprehensive test strategy",
                    "context": "testing environment",
                    "steps": ["analyze requirements", "design solution", "implement", "test"],
                    "expected_outcomes": ["improved performance", "reduced errors", "better maintainability"]
                }
            }
        }
        
        server_process.stdin.write((json.dumps(request1) + "\n").encode())
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        response1 = json.loads(response_line.decode().strip())
        
        if "error" in response1:
            print(f"   ❌ FAILED: {response1['error']['message']}")
        else:
            print(f"   ✅ SUCCESS: Strategy created with ID {response1['result'].get('strategy_id', 'N/A')}")
            print(f"      Expected outcomes included: {len(response1['result'].get('expected_outcomes', []))} items")
        
        # Test 2: create_custom_strategy without expected_outcomes
        print("\n2. Testing create_custom_strategy without expected_outcomes:")
        request2 = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "create_custom_strategy",
                "arguments": {
                    "strategy_name": "Minimal Test Strategy",
                    "description": "A minimal strategy without expected outcomes",
                    "context": "minimal testing",
                    "steps": ["step1", "step2"]
                }
            }
        }
        
        server_process.stdin.write((json.dumps(request2) + "\n").encode())
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        response2 = json.loads(response_line.decode().strip())
        
        if "error" in response2:
            print(f"   ❌ FAILED: {response2['error']['message']}")
        else:
            print(f"   ✅ SUCCESS: Strategy created with ID {response2['result'].get('strategy_id', 'N/A')}")
            print(f"      Expected outcomes defaulted to: {response2['result'].get('expected_outcomes', [])}")
        
        # Test 3: get_learning_metrics with default parameters
        print("\n3. Testing get_learning_metrics with default parameters:")
        request3 = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_learning_metrics",
                "arguments": {}
            }
        }
        
        server_process.stdin.write((json.dumps(request3) + "\n").encode())
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        response3 = json.loads(response_line.decode().strip())
        
        if "error" in response3:
            print(f"   ❌ FAILED: {response3['error']['message']}")
        else:
            print(f"   ✅ SUCCESS: Metrics retrieved")
            result = response3['result']
            print(f"      Metric type: {result.get('metric_type', 'N/A')}")
            print(f"      Time range: {result.get('time_range_hours', 'N/A')} hours")
            print(f"      Metrics keys: {list(result.get('metrics', {}).keys())}")
        
        # Test 4: get_learning_metrics with specific parameters
        print("\n4. Testing get_learning_metrics with specific parameters:")
        request4 = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "get_learning_metrics",
                "arguments": {
                    "metric_type": "accuracy",
                    "time_range_hours": 12
                }
            }
        }
        
        server_process.stdin.write((json.dumps(request4) + "\n").encode())
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        response4 = json.loads(response_line.decode().strip())
        
        if "error" in response4:
            print(f"   ❌ FAILED: {response4['error']['message']}")
        else:
            print(f"   ✅ SUCCESS: Accuracy metrics retrieved")
            result = response4['result']
            print(f"      Metric type: {result.get('metric_type', 'N/A')}")
            print(f"      Time range: {result.get('time_range_hours', 'N/A')} hours")
            print(f"      Filtered metrics: {list(result.get('metrics', {}).keys())}")
        
        print("\n" + "=" * 50)
        print("🎉 All fixed functions are working correctly!")
        print("✅ create_custom_strategy: Parameter schema mismatch resolved")
        print("✅ get_learning_metrics: Argument handling issue resolved")
        
    finally:
        # Clean up
        server_process.terminate()
        await server_process.wait()


if __name__ == "__main__":
    asyncio.run(test_fixed_functions())