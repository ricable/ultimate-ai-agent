#!/usr/bin/env python3
"""
Gemini MCP Client Configuration
Connects Gemini AI to Docker MCP Gateway via HTTP/SSE transport
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional

import aiohttp
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiMCPClient:
    """Gemini client that connects to Docker MCP Gateway"""
    
    def __init__(self, gateway_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.gateway_url = gateway_url
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            logger.warning("No Gemini API key provided. Tool calling will be limited.")
            self.model = None
            
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        await self.fetch_available_tools()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_available_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from Docker MCP Gateway"""
        try:
            async with self.session.get(f"{self.gateway_url}/tools") as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data and "tools" in data["result"]:
                        self.available_tools = data["result"]["tools"]
                        logger.info(f"Fetched {len(self.available_tools)} tools from MCP Gateway")
                        return self.available_tools
                    else:
                        logger.warning("Unexpected response format from MCP Gateway")
                        return []
                else:
                    logger.error(f"Failed to fetch tools: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching tools: {e}")
            return []
    
    def convert_mcp_tools_to_gemini(self) -> List[Tool]:
        """Convert MCP tools to Gemini function declarations"""
        gemini_tools = []
        
        for tool in self.available_tools:
            try:
                # Convert MCP tool schema to Gemini function declaration
                function_declaration = FunctionDeclaration(
                    name=tool.get("name", "unknown_tool"),
                    description=tool.get("description", "MCP tool"),
                    parameters={
                        "type": "object",
                        "properties": tool.get("inputSchema", {}).get("properties", {}),
                        "required": tool.get("inputSchema", {}).get("required", [])
                    }
                )
                gemini_tools.append(Tool(function_declarations=[function_declaration]))
                
            except Exception as e:
                logger.error(f"Failed to convert tool {tool.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Converted {len(gemini_tools)} tools for Gemini")
        return gemini_tools
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via the gateway"""
        try:
            async with self.session.post(
                f"{self.gateway_url}/tools/{tool_name}",
                json=arguments,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully called tool {tool_name}")
                    return result
                else:
                    error_msg = f"Tool call failed: HTTP {response.status}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                    
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def chat_with_tools(self, message: str) -> str:
        """Chat with Gemini using MCP tools"""
        if not self.model:
            return "Gemini API key not configured. Cannot process requests."
        
        try:
            gemini_tools = self.convert_mcp_tools_to_gemini()
            
            # Start chat with tools
            chat = self.model.start_chat(tools=gemini_tools)
            
            # Send message
            response = chat.send_message(message)
            
            # Handle function calls
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Execute the function call
                        function_name = part.function_call.name
                        function_args = dict(part.function_call.args)
                        
                        logger.info(f"Executing function: {function_name} with args: {function_args}")
                        
                        # Call the MCP tool
                        tool_result = await self.call_mcp_tool(function_name, function_args)
                        
                        # Send the result back to Gemini
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_name,
                                        response={"result": tool_result}
                                    )
                                )]
                            )
                        )
            
            return response.text
            
        except Exception as e:
            error_msg = f"Chat error: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def get_gateway_status(self) -> Dict[str, Any]:
        """Get Docker MCP Gateway status"""
        try:
            async with self.session.get(f"{self.gateway_url}/") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Gateway unreachable: HTTP {response.status}"}
        except Exception as e:
            return {"error": f"Gateway connection failed: {e}"}

async def main():
    """Main function for testing Gemini MCP integration"""
    print("🤖 Gemini MCP Client - Docker Gateway Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("🔧 Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    async with GeminiMCPClient() as client:
        # Check gateway status
        status = await client.get_gateway_status()
        print(f"🌐 Gateway Status: {status.get('status', 'unknown')}")
        print(f"🔧 Available Tools: {len(client.available_tools)}")
        
        if client.available_tools:
            print("\n📋 Available MCP Tools:")
            for tool in client.available_tools[:5]:  # Show first 5 tools
                print(f"  • {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}")
            
            if len(client.available_tools) > 5:
                print(f"  ... and {len(client.available_tools) - 5} more")
        
        # Interactive mode
        print("\n💬 Interactive Mode (type 'quit' to exit)")
        print("📝 Example: 'List files in the current directory'")
        print("📝 Example: 'Search for information about Docker MCP'")
        
        while True:
            try:
                user_input = input("\n🤖 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    print("🤔 Gemini: ", end="", flush=True)
                    response = await client.chat_with_tools(user_input)
                    print(response)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())