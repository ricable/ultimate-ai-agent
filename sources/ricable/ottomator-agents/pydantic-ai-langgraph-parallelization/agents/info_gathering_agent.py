from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Any, List, Dict
from dataclasses import dataclass
import logfire
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model

logfire.configure(send_to_logfire='if-token-present')

model = get_model()

class TravelDetails(BaseModel):
    """Details for the current trip."""
    response: str = Field(description='The response to give back to the user if they did not give all the necessary details for their trip')
    destination: str
    origin: str
    max_hotel_price: int
    date_leaving: str = Field(description='Date in format MM-DD')
    date_returning: str = Field(description='Date in format MM-DD')
    all_details_given: bool = Field(description='True if the user has given all the necessary details, otherwise false')

system_prompt = """
You are a travel planning assistant who helps users plan their trips.

Your goal is to gather all the necessary details from the user for their trip, including:
- Where they are going
- Where they are flying from
- Date they are leaving (month and day)
- Date they are returning
- Max price for a hotel per night

Output all the information for the trip you have in the required format, and also
ask the user for any missing information if necessary. Tell the user what information they need to provide still.
"""

info_gathering_agent = Agent(
    model,
    result_type=TravelDetails,
    system_prompt=system_prompt,
    retries=2
)