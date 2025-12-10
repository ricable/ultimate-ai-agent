from pydantic_ai import Agent, RunContext
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

system_prompt = """
You are a travel planning assistant who helps users plan their trips.

You can provide personalized activity recommendations based on the user's destination, duration, budget, and preferences.

Use the get_weather_forecast tool to get the weather based on the location to aid in recommending the right activities.

Format your response in a clear, organized way with activities you recommend based on the weather and your reason for each.

Never ask for clarification on any piece of information before recommending activities, just make
your best guess for any parameters that you aren't sure of.
"""

activity_agent = Agent(
    model,
    system_prompt=system_prompt,
    retries=2
)

@activity_agent.tool_plain
async def get_weather_forecast(city: str, date: str) -> str:
    """Get the weather forecast for a city on a specific date."""
    # In a real implementation, this would call a weather API
    weather_data = {
        "New York": {"sunny": 0.3, "rainy": 0.4, "cloudy": 0.3},
        "Los Angeles": {"sunny": 0.8, "rainy": 0.1, "cloudy": 0.1},
        "Chicago": {"sunny": 0.4, "rainy": 0.3, "cloudy": 0.3},
        "Miami": {"sunny": 0.7, "rainy": 0.2, "cloudy": 0.1},
        "London": {"sunny": 0.2, "rainy": 0.5, "cloudy": 0.3},
        "Paris": {"sunny": 0.4, "rainy": 0.3, "cloudy": 0.3},
        "Tokyo": {"sunny": 0.5, "rainy": 0.3, "cloudy": 0.2},
    }
    
    if city in weather_data:
        conditions = weather_data[city]
        # Simple simulation based on probabilities
        highest_prob = max(conditions, key=conditions.get)
        temp_range = {
            "New York": "15-25°C",
            "Los Angeles": "20-30°C",
            "Chicago": "10-20°C",
            "Miami": "25-35°C",
            "London": "10-18°C",
            "Paris": "12-22°C",
            "Tokyo": "15-25°C",
        }
        return f"The weather in {city} on {date} is forecasted to be {highest_prob} with temperatures around {temp_range.get(city, '15-25°C')}."
    else:
        return f"Weather forecast for {city} is not available."