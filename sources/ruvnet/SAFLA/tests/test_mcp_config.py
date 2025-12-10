#!/usr/bin/env python3
"""Test MCP configuration by simulating Claude Code MCP client."""

import subprocess
import json
import os
import tempfile

def test_mcp_config():
    """Test the MCP configuration like Claude Code would."""
    
    print("Testing MCP Configuration (.roo/mcp.json)")
    print("=" * 50)
    
    # Read the MCP configuration
    try:
        with open('/workspaces/SAFLA/.roo/mcp.json', 'r') as f:
            config = json.load(f)
        
        safla_config = config['mcpServers']['safla']
        print(f"✅ MCP config loaded")
        print(f"Command: {safla_config['command']}")
        print(f"Args: {safla_config['args']}")
        print(f"Env: {safla_config['env']}")
        
    except Exception as e:
        print(f"❌ Failed to load MCP config: {e}")
        return False
    
    # Set up environment
    env = os.environ.copy()
    env.update(safla_config.get('env', {}))
    
    print(f"\n🔧 Environment variables:")
    for key, value in safla_config.get('env', {}).items():
        print(f"   {key} = {value}")
    
    # Test initialization (like Claude Code would do)
    print(f"\n1. Testing MCP server initialization...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    try:
        result = subprocess.run(
            [safla_config['command']] + safla_config['args'],
            input=json.dumps(init_request) + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=15
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout.strip())
            server_info = response['result']['serverInfo']
            print(f"✅ Server initialized: {server_info['name']} v{server_info['version']}")
        else:
            print(f"❌ Initialization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False
    
    # Test tool discovery (like Claude Code would do)
    print(f"\n2. Testing tool discovery...")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    try:
        result = subprocess.run(
            [safla_config['command']] + safla_config['args'],
            input=json.dumps(tools_request) + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=15
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout.strip())
            tools = response['result']['tools']
            print(f"✅ Discovered {len(tools)} tools:")
            
            # Group tools by category
            original_tools = []
            enhanced_tools = []
            
            for tool in tools:
                if tool['name'] in ['generate_embeddings', 'store_memory', 'retrieve_memories', 'get_performance']:
                    original_tools.append(tool['name'])
                else:
                    enhanced_tools.append(tool['name'])
            
            print(f"   📦 Original SAFLA tools ({len(original_tools)}): {', '.join(original_tools)}")
            print(f"   🚀 Enhanced tools ({len(enhanced_tools)}): {', '.join(enhanced_tools[:5])}")
            if len(enhanced_tools) > 5:
                print(f"       ... and {len(enhanced_tools) - 5} more")
            
            return len(tools) >= 10  # Should have at least 10 tools
        else:
            print(f"❌ Tool discovery failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Tool discovery error: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_config()
    print(f"\n{'='*50}")
    if success:
        print("🎉 MCP Configuration working perfectly!")
        print("✅ Claude Code should be able to discover and use all SAFLA tools")
    else:
        print("🔴 MCP Configuration has issues!")
        print("❌ Claude Code may not be able to use SAFLA tools")
    
    print(f"\n📋 To use in Claude Code:")
    print(f"1. Ensure the .roo/mcp.json file is in your project root")
    print(f"2. Restart Claude Code to pick up the new MCP server")
    print(f"3. The SAFLA tools should appear in the tool palette")