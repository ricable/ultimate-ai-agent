from dotenv import load_dotenv
import asyncio
import os

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent, RunContext

load_dotenv()

# ========== Helper function to get model configuration ==========
def get_model():
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

server = MCPServerHTTP(url='http://localhost:8060/sse')
agent = Agent(get_model(), mcp_servers=[server])

async def main():
    async with agent.run_mcp_servers():  
        result = await agent.run('What memories do I have??')
    print(result.data)

if __name__ == "__main__":
    asyncio.run(main())