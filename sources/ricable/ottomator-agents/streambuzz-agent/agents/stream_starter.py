from typing import Any, Dict

from constants.constants import MODEL_RETRIES, PYDANTIC_AI_MODEL
from constants.prompts import STREAM_STARTER_AGENT_SYSTEM_PROMPT
from exceptions.user_error import UserError
from models.youtube_models import StreamMetadata, StreamMetadataDB
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from utils import supabase_util
from utils.youtube_util import (deactivate_session, get_stream_metadata,
                                validate_and_extract_youtube_id)

# Create Agent Instance with System Prompt and Result Type
stream_starter_agent = Agent(
    model=PYDANTIC_AI_MODEL,
    name="stream_starter_agent",
    end_strategy="early",
    model_settings=ModelSettings(temperature=0.0),
    system_prompt=STREAM_STARTER_AGENT_SYSTEM_PROMPT,
    result_type=str,
    result_tool_name="start_stream",
    result_tool_description="get url from user query, validate the url and start the "
    "stream.",
    result_retries=MODEL_RETRIES,
    deps_type=str,
)


def get_live_chat_id(metadata: Dict[str, Any]) -> str:
    """Extract the live chat ID from the provided metadata.

    This function attempts to retrieve the active live chat ID from the
    'liveStreamingDetails' section of the given metadata.

    Args:
        metadata: A dictionary containing live-streaming details, expected to have a
          'liveStreamingDetails' key with a nested 'activeLiveChatId' key.

    Returns:
        The live chat ID as a string.

    Raises:
        UserError: If the 'liveStreamingDetails' or 'activeLiveChatId' keys
          are missing or if any error occurs while accessing them, indicating an
          inactive or invalid stream link.
    """
    try:
        return metadata.get("liveStreamingDetails").get("activeLiveChatId")
    except Exception as e:
        print(f"Error fetching live chat id: {str(e)}")
        raise UserError("Inactive or invalid stream link.")


def populate_metadata_class(snippet: Dict[str, Any]) -> StreamMetadata:
    """Populate a StreamMetadata object with video details from a snippet.

    This function extracts relevant information such as the title, channel title,
    and thumbnail URL from the provided snippet dictionary and uses it to
    create and return a `StreamMetadata` instance.

    Args:
        snippet: A dictionary containing video details, expected to have
            'title', 'channelTitle', and nested 'thumbnails' keys with a
            'high' key containing a 'url'.

    Returns:
        A `StreamMetadata` instance populated with the extracted video details.
    """
    return StreamMetadata(
        title=snippet.get("title").strip(),
        channel_title=snippet.get("channelTitle"),
        thumbnail_url=snippet.get("thumbnails").get("high").get("url"),
    )


@stream_starter_agent.tool
async def start_stream(ctx: RunContext[str], url: str) -> Dict[str, Any]:
    """Start a stream by validating the URL and fetching stream metadata.

    This asynchronous function is decorated as a tool for the
    `stream_starter_agent`. It takes a YouTube URL, validates it, retrieves
    the stream metadata, and stores the relevant information in a database. It
    also deactivates the current session.

    Args:
        ctx: The run context containing dependencies, specifically the session ID.
        url: The YouTube URL to validate and start the stream.

    Returns:
        A dictionary containing the stream metadata, including title, channel
        title, and thumbnail URL.

    Raises:
        UserError: If the provided URL is invalid, if the stream is not live,
          or if there is an issue retrieving metadata from the YouTube API.
        Exception: If any other unexpected error occurs during the process.
    """
    session_id = ctx.deps
    try:
        video_id = await validate_and_extract_youtube_id(url=url)
        video_metadata = await get_stream_metadata(
            video_id=video_id, session_id=session_id
        )
        metadata_items = video_metadata.get("items")

        if not metadata_items:
            raise UserError("Invalid YouTube stream link.")

        video_id = metadata_items[0].get("id")
        snippet = metadata_items[0].get("snippet")
        live_chat_id = get_live_chat_id(metadata_items[0])
        stream_metadata = populate_metadata_class(snippet)

        # Update flag for session_id
        await deactivate_session(session_id)

        # Store metadata in DB
        stream_metadata_db = StreamMetadataDB(
            **stream_metadata.model_dump(),
            video_id=video_id,
            live_chat_id=live_chat_id,
            session_id=session_id,
            next_chat_page="",
            is_active=1,
        )
        await supabase_util.start_stream(stream_metadata_db)
        return stream_metadata.model_dump()
    except UserError as ue:
        print(f"Error>> start_stream: {str(ue)}")
        raise
    except Exception as e:
        print(f"Error>> start_stream: {str(e)}")
        raise
