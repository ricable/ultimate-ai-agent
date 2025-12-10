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
from agents.flight_agent import flight_agent, FlightDeps

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

class CLI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = FlightDeps(
            preferred_airlines=["OceanAir"]
        )
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
                async with flight_agent.iter(user_input, deps=self.deps, message_history=self.messages) as run:
                    async for node in run:
                        ai_response = ""
                        if Agent.is_model_request_node(node):
                            # A model request node => We can stream tokens from the model's request
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                            ai_response = event.part.content
                                            live.update(Markdown(ai_response))
                                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                            ai_response += event.delta.content_delta
                                            live.update(Markdown(ai_response))                       

            # Store the user message, tool calls and results, and the AI response
            self.messages += run.result.all_messages()

async def main():
    cli = CLI()
    await cli.chat()

if __name__ == "__main__":
    asyncio.run(main())
