from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import pathlib
import asyncio
import sys
import os
import logfire
from httpx import AsyncClient

from pydantic_ai import Agent
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIModel

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

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

agent = Agent(model=get_model(), deps_type=Deps)

@agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    with logfire.span('calling Brave search API', query=web_query) as span:
        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': web_query,
                'count': 5,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    results = []
    
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."

async def main():
    while True:
        # Example: Search the web to find the newest local LLMs.
        user_input = input("\n[You] ")
        
        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("Goodbye!")
            break

        async with AsyncClient() as client:
            brave_api_key = os.getenv('BRAVE_API_KEY', None)
            deps = Deps(client=client, brave_api_key=brave_api_key)

            result = await agent.run(
                user_input, deps=deps
            )
            
            print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())