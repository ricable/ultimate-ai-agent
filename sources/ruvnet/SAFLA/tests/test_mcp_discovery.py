#!/usr/bin/env python3
"""Test MCP server tool discovery."""

import subprocess
import json
import sys

def test_mcp_server():
    """Test the MCP server tool discovery."""
    
    # Test initialization
    print("1. Testing initialization...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }
    
    result = subprocess.run(
        ["python3", "safla_mcp_enhanced.py"],
        input=json.dumps(init_request) + "\n",
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout.strip())
            print(f"✅ Initialization successful: {response['result']['serverInfo']['name']}")
        except:
            print(f"❌ Initialization failed: {result.stdout}")
            return False
    else:
        print(f"❌ Initialization error: {result.stderr}")
        return False
    
    # Test tools/list
    print("\n2. Testing tools/list...")
    tools_request = {
        "jsonrpc": "2.0", 
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    result = subprocess.run(
        ["python3", "safla_mcp_enhanced.py"],
        input=json.dumps(tools_request) + "\n",
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout.strip())
            tools = response['result']['tools']
            print(f"✅ Tools discovered: {len(tools)} tools found")
            
            # List first few tools
            for i, tool in enumerate(tools[:5]):
                print(f"   {i+1}. {tool['name']}: {tool['description'][:50]}...")
            
            if len(tools) > 5:
                print(f"   ... and {len(tools) - 5} more tools")
                
            return len(tools) > 0
        except Exception as e:
            print(f"❌ Tools/list parsing failed: {e}")
            print(f"Raw output: {result.stdout}")
            return False
    else:
        print(f"❌ Tools/list error: {result.stderr}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("\n🎉 MCP server discovery working correctly!")
    else:
        print("\n🔴 MCP server discovery has issues!")
    sys.exit(0 if success else 1)