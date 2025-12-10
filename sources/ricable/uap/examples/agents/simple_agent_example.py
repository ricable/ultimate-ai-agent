"""
Simple Agent Example

This example demonstrates how to create and use a simple agent using the UAP SDK.
"""

import asyncio
from uap_sdk import UAPAgent, CustomAgentBuilder, Configuration, SimpleAgent


async def main():
    """Main function demonstrating simple agent usage."""
    
    print("=== UAP Simple Agent Example ===\n")
    
    # Create configuration
    config = Configuration({
        "backend_url": "http://localhost:8000",
        "websocket_url": "ws://localhost:8000",
        "log_level": "INFO"
    })
    
    # Method 1: Using the built-in SimpleAgent
    print("1. Creating agent with SimpleAgent framework...")
    
    agent = (CustomAgentBuilder("simple-demo")
             .with_simple_framework()
             .with_config(config)
             .build())
    
    # Start the agent
    await agent.start()
    print(f"Agent started: {agent.agent_id}")
    
    # Get agent status
    status = agent.get_status()
    print(f"Agent status: {status['framework']['status']}")
    print()
    
    # Interact with the agent
    test_messages = [
        "Hello there!",
        "How are you doing?", 
        "What can you help me with?",
        "Tell me about yourself",
        "Goodbye!"
    ]
    
    for message in test_messages:
        print(f"User: {message}")
        response = await agent.process_message(message)
        print(f"Agent: {response.get('content', '')}")
        print()
    
    # Stop the agent
    await agent.stop()
    print("Agent stopped successfully.")


if __name__ == "__main__":
    asyncio.run(main())