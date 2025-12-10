from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from constants.constants import (MODEL_RETRIES, PYDANTIC_AI_MODEL)
from constants.prompts import BUZZ_INTERN_AGENT_SYSTEM_PROMPT

# Create Agent Instance with System Prompt and Result Type
buzz_intern_agent = Agent(
    model=PYDANTIC_AI_MODEL,
    name="buzz_intern_agent",
    end_strategy="early",
    model_settings=ModelSettings(temperature=0.3),
    system_prompt=BUZZ_INTERN_AGENT_SYSTEM_PROMPT,
    result_type=str,
    result_retries=MODEL_RETRIES,
    deps_type=str,
)
