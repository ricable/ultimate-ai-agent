from dotenv import load_dotenv
from typing import List
import asyncio
import logfire
import sys
import os

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

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

    async def chat(self):
        print("Flight Agent CLI (type 'quit' to exit)")
        print("Enter your message:")
        
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                break

            # Run the agent with streaming
            result = await flight_agent.run(
                user_input,
                deps=self.deps,
                message_history=self.messages
            )

            # Store the user message
            self.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )

            # Store itermediatry messages like tool calls and responses
            filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' or part.part_kind == 'text' for part in msg.parts))]
            self.messages.extend(filtered_messages)

            # Optional if you want to print out tool calls and responses
            # print(filtered_messages + "\n\n")

            print(result.data)

            # Add the final response from the agent
            self.messages.append(
                ModelResponse(parts=[TextPart(content=result.data)])
            )

async def main():
    cli = CLI()
    await cli.chat()

if __name__ == "__main__":
    asyncio.run(main())
