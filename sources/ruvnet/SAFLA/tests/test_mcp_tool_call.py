#!/usr/bin/env python3
"""Test MCP server tool calling."""

import subprocess
import json
import sys

def test_tool_call():
    """Test calling a tool via MCP."""
    
    # Test analyze_text tool
    print("Testing analyze_text tool via MCP...")
    
    tool_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "analyze_text",
            "arguments": {
                "text": "SAFLA is an amazing AI system!",
                "analysis_type": "sentiment"
            }
        }
    }
    
    result = subprocess.run(
        ["python3", "safla_mcp_enhanced.py"],
        input=json.dumps(tool_request) + "\n",
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout.strip())
            if "result" in response:
                content = response["result"]["content"]
                if content and len(content) > 0:
                    print(f"✅ Tool call successful!")
                    print(f"Response type: {content[0].get('type', 'unknown')}")
                    result_text = content[0].get('text', '')
                    if len(result_text) > 200:
                        result_text = result_text[:200] + "..."
                    print(f"Result: {result_text}")
                    return True
                else:
                    print(f"❌ Empty result content")
                    return False
            else:
                print(f"❌ No result in response: {response}")
                return False
        except Exception as e:
            print(f"❌ Tool call parsing failed: {e}")
            print(f"Raw output: {result.stdout}")
            return False
    else:
        print(f"❌ Tool call error: {result.stderr}")
        return False

if __name__ == "__main__":
    success = test_tool_call()
    if success:
        print("\n🎉 MCP tool calling working correctly!")
    else:
        print("\n🔴 MCP tool calling has issues!")
    sys.exit(0 if success else 1)