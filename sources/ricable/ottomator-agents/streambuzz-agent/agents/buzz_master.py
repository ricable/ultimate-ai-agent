from constants.constants import (CHAT_WRITE_INTERVAL, MODEL_RETRIES,
                                 PYDANTIC_AI_MODEL)
from constants.enums import StateEnum
from constants.prompts import BUZZ_MASTER_SYSTEM_PROMPT
from exceptions.user_error import UserError
from models.youtube_models import StreamMetadataDB, WriteChatModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from utils import supabase_util

# Create Agent Instance with System Prompt and Result Type
buzz_master_agent = Agent(
    model=PYDANTIC_AI_MODEL,
    name="buzz_master_agent",
    end_strategy="early",
    model_settings=ModelSettings(temperature=0.0),
    system_prompt=BUZZ_MASTER_SYSTEM_PROMPT,
    result_type=str,
    result_tool_name="execute_task",
    result_tool_description="execute tasks and return user friendly response",
    result_retries=MODEL_RETRIES,
    deps_type=str,
)
"""
An agent designed to manage and respond to "buzz" prompts within a live stream context.

This agent uses a system prompt to guide its behavior and interacts with a Supabase database
to retrieve and store information about active streams and chat replies. It is configured to use
a specific language model, end its execution early, and maintain a low temperature for consistent responses.

Attributes:
    model (str): The name of the language model to be used by the agent.
        This is typically set by the `PYDANTIC_AI_MODEL` constant.
    name (str): The name of the agent, which is "buzz_master_agent".
    end_strategy (str): The strategy used to end the agent's execution.
        "early" indicates that the agent will stop as soon as a result is available.
    model_settings (ModelSettings): Configuration settings for the language model.
        Includes parameters like `temperature`, set to 0.0 for deterministic output.
    system_prompt (str): The system prompt that guides the agent's behavior.
       This is defined by the `BUZZ_MASTER_SYSTEM_PROMPT` constant and provides context for the agent's tasks.
    result_type (type): The expected data type of the agent's output, set to `str`.
    result_tool_name (str): The name of the tool used to return the final result, "execute_task".
    result_tool_description (str): A description of the result tool, "execute tasks and return user friendly response".
    result_retries (int): The number of times to retry the result tool if it fails, set by `MODEL_RETRIES`.
    deps_type (type): The expected data type of the agent's dependencies, set to `str`.
"""

async def get_active_stream(session_id):
    active_stream: StreamMetadataDB = await supabase_util.get_active_stream(
        session_id=session_id
    )
    if not active_stream:
        raise UserError(
            "You are not moderating any YouTube live stream currently. "
            "Start a stream by sending a YouTube Live Stream URL"
            )
    return active_stream

@buzz_master_agent.tool
async def get_current_buzz(ctx: RunContext[str]) -> str:
    """
    Retrieves the current active buzz for a given session ID.

    This tool queries the Supabase database to find the currently active buzz associated with the provided session ID.
    The buzz is typically a text prompt or question that the agent is meant to respond to.

    Args:
        ctx (RunContext[str]): The context of the agent run, containing the session ID as a dependency.
            The session ID is used to identify the specific stream or context.

    Returns:
        str: The current buzz associated with the session ID.
            Returns an empty string if no active buzz is found.

    Raises:
        Exception: If there is an issue fetching data from the database,
            such as a network error or database query failure.
    """
    _ = await get_active_stream(session_id=ctx.deps)
    return await supabase_util.get_current_buzz(session_id=ctx.deps)


@buzz_master_agent.tool
async def get_next_buzz(ctx: RunContext[str]) -> str:
    """
    Marks the current buzz as inactive and retrieves the next buzz for a given session ID.

    This tool first deactivates the current active buzz associated with the provided session ID in the Supabase database.
    It then retrieves the next available buzz, which becomes the new active buzz for the session.

    Args:
        ctx (RunContext[str]): The context of the agent run, containing the session ID as a dependency.
            The session ID is used to identify the specific stream or context.

    Returns:
        str: The next buzz associated with the session ID.
            Returns an empty string if no next buzz is found.

    Raises:
        Exception: If there is an issue interacting with the database,
            such as problems marking the current buzz inactive or fetching the next one.
    """
    _ = await get_active_stream(session_id=ctx.deps)
    await supabase_util.mark_current_buzz_inactive(session_id=ctx.deps)
    return await supabase_util.get_current_buzz(session_id=ctx.deps)


@buzz_master_agent.tool
async def store_reply(ctx: RunContext[str], reply: str) -> str:
    """
    Stores a user's reply for a given session ID, associated with the active stream.

    This tool stores a user's reply in the Supabase database, associating it with the active live stream
    for the given session ID. The reply is initially marked as not yet written to the live chat.
    Replies are cumulated within a time slot defined by `CHAT_WRITE_INTERVAL`.

    Args:
        ctx (RunContext[str]): The context of the agent run, containing the session ID as a dependency.
            The session ID is used to identify the specific stream or context.
        reply (str): The user's reply to be stored.

    Returns:
        str: A confirmation message indicating that the reply has been acknowledged and will be posted
            to the live chat within the specified time slot.

    Raises:
        UserError: If no active stream is found for the given session ID, or if there is an error
            storing the reply in the database. The error message provides details about the failure.
    """
    try:
        active_stream = await get_active_stream(session_id=ctx.deps)
        await supabase_util.store_reply(
            WriteChatModel(
                session_id=active_stream.session_id,
                live_chat_id=active_stream.live_chat_id,
                retry_count=0,
                reply=reply,
                is_written=StateEnum.NO.value,
            )
        )
        return f"Your reply is acknowledged. Replies within time slot of {CHAT_WRITE_INTERVAL} seconds will be cumulated and posted to live chat."
    except Exception as e:
        raise UserError(f"Error storing reply: {str(e)}")
