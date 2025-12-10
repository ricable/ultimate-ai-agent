from dotenv import load_dotenv
import pathlib
import asyncio
import sys
import os

from pydantic_ai import Agent
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIModel

import mcp_client

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

load_dotenv()

def get_model():
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    return OpenAIModel(
        llm,
        base_url=base_url,
        api_key=api_key
    )

async def get_pydantic_ai_agent():
    client = mcp_client.MCPClient()
    client.load_servers(str(CONFIG_FILE))
    tools = await client.start()
    return client, Agent(model=get_model(), tools=tools)

async def main():
    client, agent = await get_pydantic_ai_agent()
    while True:
        # Example: Search the web to find the newest local LLMs.
        user_input = input("\n[You] ")
        
        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("Goodbye!")
            break

        result = await agent.run(user_input)
        print('[Assistant] ', result.data)


if __name__ == '__main__':
    asyncio.run(main())    