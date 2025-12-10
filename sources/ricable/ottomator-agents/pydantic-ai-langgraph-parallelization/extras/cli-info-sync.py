from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart, PartDeltaEvent, PartStartEvent, TextPartDelta
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text
from pydantic_ai import Agent
from dotenv import load_dotenv
from typing import List
import asyncio
import logfire
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.info_gathering_agent import info_gathering_agent

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

class CLI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.console = Console()

    async def chat(self):
        print("Flight Agent CLI (type 'quit' to exit)")
        print("Enter your message:")
        
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                break

            # Run the agent with streaming
            with Live('', console=self.console, vertical_overflow='visible') as live:
                output_messages = []
                result = await info_gathering_agent.run(user_input, message_history=self.messages)
                live.update(Markdown(result.data.response))

            # Store the user message, tool calls and results, and the AI response
            self.messages += result.all_messages()

async def main():
    cli = CLI()
    await cli.chat()

if __name__ == "__main__":
    asyncio.run(main())
