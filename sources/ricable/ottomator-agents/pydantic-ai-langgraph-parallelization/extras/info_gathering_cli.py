from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from rich.console import Console, ConsoleOptions, RenderResult
from pydantic import ValidationError
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
                async with info_gathering_agent.run_stream(user_input, message_history=self.messages) as result:
                    async for message, last in result.stream_structured(debounce_by=0.01):  
                        try:
                            if last and not travel_details.response:
                                raise Exception("Incorrect travel details returned by the agent.")
                            travel_details = await result.validate_structured_result(  
                                message,
                                allow_partial=not last,
                            )
                        except ValidationError as e:
                            continue

                        if travel_details.response:
                            live.update(Markdown(travel_details.response))

            print(travel_details.all_details_given)           

            # Store the user message
            self.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )

            # Add the final response from the agent
            self.messages.append(
                ModelResponse(parts=[TextPart(content=travel_details.response)])
            )

async def main():
    cli = CLI()
    await cli.chat()

if __name__ == "__main__":
    asyncio.run(main())
