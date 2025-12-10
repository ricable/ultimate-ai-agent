from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic_ai.messages import (ModelRequest, ModelResponse, TextPart,
                                  UserPromptPart)

from constants.constants import (CONVERSATION_CONTEXT, MESSAGES, MODEL_RETRIES,
                                 STREAMER_KB, SUPABASE_CLIENT, YT_BUZZ,
                                 YT_REPLY, YT_STREAMS)
from constants.enums import BuzzStatusEnum, StateEnum
from models.agent_models import ProcessedChunk
from models.youtube_models import (StreamBuzzModel, StreamMetadataDB,
                                   WriteChatModel)

# Load environment variables
load_dotenv()


# MESSAGES table queries
async def fetch_human_session_history(session_id: str, limit: int = 10) -> list[str]:
    """Fetches the most recent human conversation history for a given session.

    This function retrieves a specified number of the most recent messages from the
    `MESSAGES` table in Supabase, filtering for messages of type "human" and
    ordering them by creation time in descending order. It then returns a list of
    message contents, limited by the `CONVERSATION_CONTEXT` constant.

    Args:
        session_id: The unique identifier of the session.
        limit: The maximum number of messages to retrieve. Defaults to 10.

    Returns:
        A list of strings, where each string is the content of a human message.
        The list is limited by the `CONVERSATION_CONTEXT` constant.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(MESSAGES)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        messages = []
        for msg in response.data:
            if msg["message"]["type"] == "human":
                messages.append(msg["message"]["content"])
        return messages[:CONVERSATION_CONTEXT]
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch_human_session_history: {str(e)}"
        )


async def fetch_conversation_history(
    session_id: str, limit: int = 10
) -> list[ModelRequest | ModelResponse]:
    """Fetches the most recent conversation history for a given session.

    This function retrieves a specified number of the most recent messages from the
    `MESSAGES` table in Supabase, ordering them by creation time in descending
    order. It then converts the messages into a list of `ModelRequest` or
    `ModelResponse` objects, based on the message type ("human" or other). The
    list is returned in chronological order (oldest to newest).

    Args:
        session_id: The unique identifier of the session.
        limit: The maximum number of messages to retrieve. Defaults to 10.

    Returns:
        A list of `ModelRequest` or `ModelResponse` objects, representing the
        conversation history. The list is in chronological order.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(MESSAGES)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        # Convert to list and reverse to get chronological order
        conversation_history = response.data[::-1]

        # Convert conversation history to format expected by Pydantic AI
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]
            msg = (
                ModelRequest(parts=[UserPromptPart(content=msg_content)])
                if msg_type == "human"
                else ModelResponse(parts=[TextPart(content=msg_content)])
            )
            messages.append(msg)
        return messages
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch_conversation_history: {str(e)}"
        )


async def store_message(
    session_id: str, message_type: str, content: str, data: Optional[Dict] = None
):
    """Stores a message in the Supabase `MESSAGES` table.

    This function inserts a new message into the `MESSAGES` table, including the
    session ID, message type, content, and any optional data.

    Args:
        session_id: The unique identifier of the session.
        message_type: The type of the message (e.g., "human", "ai").
        content: The content of the message.
        data: An optional dictionary containing additional message data. Defaults to None.

    Raises:
        HTTPException: If an error occurs during the database insertion, with a 500
        status code and error details.
    """
    message_obj = {"type": message_type, "content": content}
    if data:
        message_obj["data"] = data

    try:
        SUPABASE_CLIENT.table(MESSAGES).insert(
            {"session_id": session_id, "message": message_obj}
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store_message: {str(e)}"
        )


# YT_STREAMS table queries
async def get_active_streams() -> List[Dict[str, Any]]:
    """Retrieves all active streams from the `YT_STREAMS` table.

    This function queries the `YT_STREAMS` table to find all rows where the
    `is_active` flag is set to `StateEnum.YES.value`.

    Returns:
        A list of dictionaries, where each dictionary represents an active stream.
        Each dictionary contains the 'session_id', 'live_chat_id', and
        'next_chat_page' keys.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(YT_STREAMS)
            .select("session_id, live_chat_id, next_chat_page")
            .eq("is_active", StateEnum.YES.value)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_active_streams: {str(e)}"
        )


async def get_active_stream(session_id: str) -> Optional[StreamMetadataDB]:
    """Retrieves the active stream metadata for a given session.

    This function queries the `YT_STREAMS` table to find the active stream
    associated with the provided `session_id`. A stream is considered active if
    its `is_active` flag is set to `StateEnum.YES.value`.

    Args:
        session_id: The unique identifier of the session.

    Returns:
        A `StreamMetadataDB` object containing the metadata of the active stream
        if found, otherwise None.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(YT_STREAMS)
            .select("*")
            .eq("session_id", session_id)
            .eq("is_active", StateEnum.YES.value)
            .execute()
        )
        if not response.data:
            return None
        active_stream = response.data[0]
        return StreamMetadataDB(
            session_id=active_stream["session_id"],
            video_id=active_stream["video_id"],
            live_chat_id=active_stream["live_chat_id"],
            next_chat_page=active_stream["next_chat_page"],
            is_active=active_stream["is_active"],
        )
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_active_stream: {str(e)}"
        )


async def start_stream(stream_metadata_db: StreamMetadataDB):
    """Stores stream metadata for a given session in the `YT_STREAMS` table.

    This function inserts a new row into the `YT_STREAMS` table with the provided
    stream metadata.

    Args:
        stream_metadata_db: A `StreamMetadataDB` object containing the stream metadata.

    Raises:
        HTTPException: If an error occurs during the database insertion, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_STREAMS).insert(
            {
                "session_id": stream_metadata_db.session_id,
                "video_id": stream_metadata_db.video_id,
                "live_chat_id": stream_metadata_db.live_chat_id,
                "next_chat_page": stream_metadata_db.next_chat_page,
                "is_active": stream_metadata_db.is_active,
            }
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start_stream: {str(e)}")


async def deactivate_existing_streams(session_id: str):
    """Deactivates all streams associated with a given session.

    This function updates the `is_active` flag to `StateEnum.NO.value` for all
    rows in the `YT_STREAMS` table that match the provided `session_id`.

    Args:
        session_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_STREAMS).update({"is_active": StateEnum.NO.value}).eq(
            "session_id", session_id
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to deactivate_existing_streams: {str(e)}"
        )


async def update_next_chat_page(live_chat_id: str, next_chat_page: str):
    """Updates the `next_chat_page` for an active stream of a given session.

    This function updates the `next_chat_page` column in the `YT_STREAMS` table
    for the active stream associated with the provided `live_chat_id`. A stream is
    considered active if its `is_active` flag is set to `StateEnum.YES.value`.

    Args:
        live_chat_id: The unique identifier of the sessionstream.
        next_chat_page: The new value for the `next_chat_page` column.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_STREAMS).update({"next_chat_page": next_chat_page}).eq(
            "live_chat_id", live_chat_id
        ).eq("is_active", StateEnum.YES.value).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update_next_chat_page: {str(e)}"
        )


# YT_BUZZ table queries
async def store_buzz(buzz: StreamBuzzModel):
    """Stores a buzz event in the `YT_BUZZ` table.

    This function inserts a new row into the `YT_BUZZ` table with the provided
    buzz details.

    Args:
        buzz: A `StreamBuzzModel` object containing the buzz details.

    Raises:
        HTTPException: If an error occurs during the database insertion, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_BUZZ).insert(
            {
                "buzz_type": buzz.buzz_type,
                "session_id": buzz.session_id,
                "original_chat": buzz.original_chat,
                "author": buzz.author,
                "generated_response": buzz.generated_response,
                "buzz_status": BuzzStatusEnum.FOUND.value,
            }
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store_buzz: {str(e)}")


async def get_current_buzz(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent active buzz for a given session.

    This function queries the `YT_BUZZ` table to find the most recent active buzz
    associated with the provided `session_id`. A buzz is considered active if its
    `buzz_status` is set to `BuzzStatusEnum.ACTIVE.value`. It returns the
    `buzz_type`, `original_chat`, `author`, and `generated_response` of the
    found buzz.

    Args:
        session_id: The unique identifier of the session.

    Returns:
        A dictionary containing the `buzz_type`, `original_chat`, `author`, and
        `generated_response` of the current active buzz, or None if no active
        buzz is found.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(YT_BUZZ)
            .select("buzz_type, original_chat, author, generated_response")
            .eq("session_id", session_id)
            .eq("buzz_status", BuzzStatusEnum.ACTIVE.value)
            .order("created_at")
            .order("id")
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_current_buzz: {str(e)}"
        )


async def mark_current_buzz_inactive(session_id: str):
    """Marks the most recent active buzz as inactive for a given session.

    This function updates the `buzz_status` to `BuzzStatusEnum.INACTIVE.value` for
    the most recent active buzz in the `YT_BUZZ` table that matches the provided
    `session_id`. A buzz is considered active if its `buzz_status` is set to
    `BuzzStatusEnum.ACTIVE.value`.

    Args:
        session_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(YT_BUZZ)
            .select("id")
            .eq("session_id", session_id)
            .eq("buzz_status", BuzzStatusEnum.ACTIVE.value)
            .order("created_at")
            .order("id")
            .limit(1)
            .execute()
        )
        if response.data:
            SUPABASE_CLIENT.table(YT_BUZZ).update(
                {"buzz_status": BuzzStatusEnum.INACTIVE.value}
            ).eq("id", response.data[0]["id"]).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update_current_buzz_inactive: {str(e)}"
        )


async def get_found_buzz() -> list[Dict[str, Any]]:
    """Retrieves all buzz events with a status of "FOUND".

    This function queries the `YT_BUZZ` table to retrieve all rows where the
    `buzz_status` is set to `BuzzStatusEnum.FOUND.value`.

    Returns:
        A list of dictionaries, where each dictionary contains the `id`,
        `buzz_type`, and `original_chat` of a found buzz event.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(YT_BUZZ)
            .select("id, session_id, author, buzz_type, original_chat")
            .eq("buzz_status", BuzzStatusEnum.FOUND.value)
            .order("created_at")
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_found_buzz: {str(e)}"
        )


async def update_buzz_status_by_id(buzz_id: int, buzz_status: str):
    """Updates the status of a specific buzz event by its ID.

    This function updates the `buzz_status` column in the `YT_BUZZ` table for the
    row that matches the provided `id`.

    Args:
        buzz_id: The unique identifier of the buzz event to update.
        buzz_status: The new status to set for the buzz event.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_BUZZ).update({"buzz_status": buzz_status}).eq(
            "id", buzz_id
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update_buzz_status: {str(e)}"
        )


async def update_buzz_status_by_session_id(session_id: str, buzz_status: int):
    """Updates the status of all buzz events for a given session.

    This function updates the `buzz_status` column in the `YT_BUZZ` table for all
    rows that match the provided `session_id`.

    Args:
        session_id: The unique identifier of the session.
        buzz_status: The new status to set for the buzz events.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_BUZZ).update({"buzz_status": buzz_status}).eq(
            "session_id", session_id
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to read_existing_buzz: {str(e)}"
        )


async def update_buzz_status_batch_by_id(id_list: list[int], buzz_status: int):
    """Updates the status of multiple buzz events by their IDs.

    This function updates the `buzz_status` column in the `YT_BUZZ` table for all
    rows that match the provided list of `id_list`.

    Args:
        id_list: A list of unique identifiers of the buzz events to update.
        buzz_status: The new status to set for the buzz events.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_BUZZ).update({"buzz_status": buzz_status}).in_(
            "id", id_list
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update_buzz_status: {str(e)}"
        )


async def update_buzz_response_by_id(buzz_id: int, generated_response: str):
    """Updates the response of a specific buzz event by its ID.

    This function updates the `buzz_status` to `BuzzStatusEnum.ACTIVE.value` and
    the `generated_response` column in the `YT_BUZZ` table for the row that
    matches the provided `id`.

    Args:
        buzz_id: The unique identifier of the buzz event to update.
        generated_response: The new generated response to set for the buzz event.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_BUZZ).update(
            {
                "buzz_status": BuzzStatusEnum.ACTIVE.value,
                "generated_response": generated_response,
            }
        ).eq("id", buzz_id).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update_buzz_response: {str(e)}"
        )


# YT_REPLY table queries
async def store_reply(reply: WriteChatModel):
    """Stores a chat reply in the `YT_REPLY` table.

    This function inserts a new row into the `YT_REPLY` table with the provided
    reply details.

    Args:
        reply: A `WriteChatModel` object containing the reply details.

    Raises:
        HTTPException: If an error occurs during the database insertion, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_REPLY).insert(
            {
                "session_id": reply.session_id,
                "live_chat_id": reply.live_chat_id,
                "retry_count": reply.retry_count,
                "reply": reply.reply,
                "is_written": StateEnum.NO.value,
            }
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store_reply: {str(e)}")


async def get_unwritten_replies() -> list[Dict[str, Any]]:
    """Retrieves unwritten chat replies from the `YT_REPLY` table.

    This function queries the `YT_REPLY` table to retrieve all rows where the
    `is_written` flag is set to `StateEnum.NO.value` and the `retry_count` is less
    than `MODEL_RETRIES`.

    Returns:
        A list of dictionaries, where each dictionary represents an unwritten
        chat reply, containing 'session_id', 'live_chat_id', and 'reply' keys.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(YT_REPLY)
            .select("session_id, live_chat_id, reply")
            .eq("is_written", StateEnum.NO.value)
            .lt("retry_count", MODEL_RETRIES)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_unwritten_chats: {str(e)}"
        )


async def mark_replies_pending(live_chat_id: str):
    """Marks unwritten replies as pending for a given session.

    This function updates the `is_written` flag to `StateEnum.PENDING.value` for
    all rows in the `YT_REPLY` table that match the provided `live_chat_id` and
    have an `is_written` flag set to `StateEnum.NO.value`.

    Args:
        live_chat_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_REPLY).update(
            {"is_written": StateEnum.PENDING.value}
        ).eq("live_chat_id", live_chat_id).eq("is_written", StateEnum.NO.value).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to read_existing_buzz: {str(e)}"
        )


async def mark_replies_success(live_chat_id: str):
    """Marks pending replies as successfully written for a given session.

    This function updates the `is_written` flag to `StateEnum.YES.value` for all
    rows in the `YT_REPLY` table that match the provided `live_chat_id` and have an
    `is_written` flag set to `StateEnum.PENDING.value`.

    Args:
        live_chat_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_REPLY).update({"is_written": StateEnum.YES.value}).eq(
            "live_chat_id", live_chat_id
        ).eq("is_written", StateEnum.PENDING.value).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to read_existing_buzz: {str(e)}"
        )


async def mark_replies_failed(live_chat_id: str):
    """Marks pending replies as failed and increments the retry count for a given session.

    This function updates the `is_written` flag to `StateEnum.NO.value` and
    increments the `retry_count` by 1 for all rows in the `YT_REPLY` table that
    match the provided `live_chat_id` and have an `is_written` flag set to
    `StateEnum.PENDING.value`.

    Args:
        live_chat_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_REPLY).update(
            {"is_written": StateEnum.NO.value}
        ).inc({"retry_count": 1}).eq("live_chat_id", live_chat_id).eq(
            "is_written", StateEnum.PENDING.value
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update replies: {str(e)}"
        )


async def deactivate_replies(session_id: str):
    """Deactivates all replies for a given session.

    This function updates the `is_written` flag to `StateEnum.YES.value` for all
    rows in the `YT_REPLY` table that match the provided `session_id`.

    Args:
        session_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database update, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(YT_REPLY).update({"is_written": StateEnum.YES.value}).eq(
            "session_id", session_id
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to read_existing_buzz: {str(e)}"
        )


# STREAMER_KB table queries
async def get_kb_file_name(session_id: str) -> Optional[str]:
    """Retrieves the file name of the knowledge base for a given session.

    This function queries the `STREAMER_KB` table to find the `file_name`
    associated with the provided `session_id`.

    Args:
        session_id: The unique identifier of the session.

    Returns:
        The file name of the knowledge base if found, otherwise None.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = (
            SUPABASE_CLIENT.table(STREAMER_KB)
            .select("file_name")
            .eq("session_id", session_id)
            .execute()
        )
        return response.data[0]["file_name"] if response.data else None
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_previous_kb_file_name: {str(e)}"
        )


async def delete_previous_kb_entries(session_id: str):
    """Deletes all knowledge base entries for a given session.

    This function deletes all rows in the `STREAMER_KB` table that match the
    provided `session_id`.

    Args:
        session_id: The unique identifier of the session.

    Raises:
        HTTPException: If an error occurs during the database deletion, with a 500
        status code and error details.
    """
    try:
        SUPABASE_CLIENT.table(STREAMER_KB).delete().eq(
            "session_id", session_id
        ).execute()
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete_previous_kb_entries: {str(e)}"
        )


async def insert_chunk(chunk: ProcessedChunk):
    """Inserts a processed chunk into the `STREAMER_KB` table.

    This function inserts a new row into the `STREAMER_KB` table with the
    details of the provided processed chunk.

    Args:
        chunk: A `ProcessedChunk` object containing the chunk details.

    Returns:
        The result of the insertion operation.

    Raises:
        HTTPException: If an error occurs during the database insertion, with a 500
        status code and error details.
    """
    try:
        data = {
            "session_id": chunk.session_id,
            "file_name": chunk.file_name,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "embedding": chunk.embedding,
        }

        result = SUPABASE_CLIENT.table(STREAMER_KB).insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.session_id}")
        return result
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to insert_chunk: {chunk=}\n{str(e)}"
        )


async def get_matching_chunks(
    query_embedding: List[float], session_id: str
) -> list[Dict[str, Any]]:
    """Retrieves matching knowledge base chunks for a given query embedding.

    This function uses the `match_streamer_knowledge` RPC function in Supabase to
    find the most relevant chunks in the `STREAMER_KB` table based on the
    provided `query_embedding` and `session_id`. The number of matching chunks
    returned is limited by the `CONVERSATION_CONTEXT` constant.

    Args:
        query_embedding: A list of floats representing the query embedding.
        session_id: The unique identifier of the session.

    Returns:
        A list of dictionaries, where each dictionary represents a matching
        knowledge base chunk.

    Raises:
        HTTPException: If an error occurs during the database query, with a 500
        status code and error details.
    """
    try:
        response = SUPABASE_CLIENT.rpc(
            "match_streamer_knowledge",
            {
                "query_embedding": query_embedding,
                "user_session_id": session_id,
                "match_count": CONVERSATION_CONTEXT,
            },
        ).execute()

        return response.data
    except Exception as e:
        print(f"Error>> Failed at supabase_util: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get_matching_chunks: {str(e)}"
        )
