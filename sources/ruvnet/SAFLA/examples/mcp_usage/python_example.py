#!/usr/bin/env python3
"""
Example: Using SAFLA via MCP in Python
"""

import asyncio
import aiohttp

async def use_safla_mcp():
    async with aiohttp.ClientSession() as session:
        # Generate embeddings
        mcp_request = {
            "jsonrpc": "2.0",
            "id": "embed_001",
            "method": "generate_embeddings",
            "params": {
                "texts": ["Hello SAFLA!", "This is a test"],
                "config": {"batch_size": 256, "cache_embeddings": True}
            }
        }
        
        async with session.post(
            "https://safla.fly.dev/api/safla",
            json=mcp_request
        ) as response:
            result = await response.json()
            print(f"Embeddings generated: {result}")

if __name__ == "__main__":
    asyncio.run(use_safla_mcp())
