#!/usr/bin/env python3
"""Debug connectivity to UI server"""

import asyncio
import os

import aiohttp


async def test_connectivity():
    # Determine if we're in Docker
    in_docker = os.path.exists("/.dockerenv")

    # Test different URLs
    urls_to_test = []

    if in_docker:
        print("Running inside Docker container")
        urls_to_test = [
            "http://host.docker.internal:3738",
            "http://host.docker.internal:3737",
            "http://frontend:5173",
            "http://Archon-UI:5173",
        ]
    else:
        print("Running on host machine")
        urls_to_test = [
            "http://localhost:3738",
            "http://localhost:3737",
        ]

    print("\nTesting connectivity to UI server...")

    async with aiohttp.ClientSession() as session:
        for url in urls_to_test:
            try:
                print(f"\nTrying {url}...")
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    print(f"  Status: {response.status}")
                    if response.status == 200:
                        content = await response.text()
                        print(f"  Success! Response length: {len(content)} chars")
                        has_root = 'id="root"' in content
                        print(f"  Contains 'root' element: {has_root}")
                    else:
                        print("  Non-200 status code")
            except Exception as e:
                print(f"  Failed: {type(e).__name__}: {e}")

    print("\nDone testing connectivity")


if __name__ == "__main__":
    asyncio.run(test_connectivity())
