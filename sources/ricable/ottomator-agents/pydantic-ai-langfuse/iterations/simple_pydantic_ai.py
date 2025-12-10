from __future__ import annotations
from typing import Any, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import nest_asyncio
import asyncio
import logfire
import base64
import os

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent, RunContext

load_dotenv()

def configure_langfuse():
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3002")
    LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    # Configure Logfire to work with Langfuse
    nest_asyncio.apply()
    logfire.configure(service_name='pydantic_ai_agent', send_to_logfire=False)

configure_langfuse()

# Brave Search MCP server
brave_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-brave-search'],
    env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
)

# Brave search agent
brave_agent = Agent(
    OpenAIModel("gpt-4.1-mini"),
    system_prompt="You are a web search specialist using Brave Search. Find relevant information on the web.",
    mcp_servers=[brave_server],
    instrument=True
)

# ========== Main execution function ==========

async def main():
    """Run the primary agent with a given query."""
    """Example query: What are the latest offerings for the Grok 3 LLM?"""
    print("Brave MCP Pydantic AI Agent")
    print("Enter 'exit' to quit the program.")

    # Use AsyncExitStack to manage all MCP servers in one context
    async with brave_agent.run_mcp_servers():
        console = Console()
        messages = []        
        
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                curr_message = ""
                with Live('', console=console, vertical_overflow='visible') as live:
                    async with brave_agent.run_stream(
                        user_input, message_history=messages
                    ) as result:
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                # Add the new messages to the chat history
                messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
