from typing import Optional

from constants.enums import BuzzStatusEnum
from pydantic import BaseModel


class StreamMetadata(BaseModel):
    """
    Represents metadata associated with a live stream.

    This model holds basic information about a live stream, such as its title,
    the channel title, and the URL of its thumbnail image.

    Attributes:
        title (Optional[str]): The title of the live stream. Defaults to an empty string.
        channel_title (Optional[str]): The title of the channel hosting the live stream.
            Defaults to an empty string.
        thumbnail_url (Optional[str]): URL of the thumbnail image for the live stream.
            Defaults to an empty string.
    """

    title: Optional[str] = ""
    channel_title: Optional[str] = ""
    thumbnail_url: Optional[str] = ""


class StreamMetadataDB(StreamMetadata):
    """
    Represents stream metadata stored in a database, extending `StreamMetadata`.

    This model includes all the attributes from `StreamMetadata` and adds
    database-specific fields, such as session IDs, video IDs, chat IDs,
    and the stream's activity status.

    Attributes:
        session_id (str): A unique identifier for the stream session.
        video_id (str): The unique identifier of the video associated with the stream.
        live_chat_id (str): The unique identifier for the live chat associated with the stream.
        next_chat_page (Optional[str]): A token or URL for fetching the next page of
            chat messages. Defaults to an empty string.
        is_active (Optional[int]): An integer representing whether the stream is currently
            active. 1 indicates active, 0 indicates inactive. Defaults to 1.
    """

    session_id: str
    video_id: str
    live_chat_id: str
    next_chat_page: Optional[str] = ""
    is_active: Optional[int] = 1


class BuzzModel(BaseModel):
    """
    Represents a buzz, which is a generated response to a user interaction.

    This model captures the essential information about a buzz, including its
    type and the generated response text.

    Attributes:
        buzz_type (str): The type or category of the buzz.
        generated_response (str): The generated response text.
    """

    buzz_type: str
    generated_response: str


class StreamBuzzModel(BuzzModel):
    """
    Represents a buzz associated with a specific stream, extending `BuzzModel`.

    This model includes all attributes from `BuzzModel` and adds stream-specific
    details, such as the session ID, the original chat message that triggered
    the buzz, the author of that message, and the buzz status.

    Attributes:
        session_id (str): The unique identifier for the stream session.
        original_chat (str): The original chat message that triggered the buzz.
        author (str): The author of the original chat message.
        buzz_status (Optional[int]): An integer representing the status of the buzz.
            Defaults to 0, which is equivalent to `BuzzStatusEnum.FOUND.value`.
    """

    session_id: str
    original_chat: str
    author: str
    buzz_status: Optional[int] = BuzzStatusEnum.FOUND.value


class StreamBuzzDisplay(BaseModel):
    """
    Represents a buzz for display purposes, containing the original chat, author, and generated response.

    This model is designed to provide all necessary information for displaying a buzz
    in a user-friendly format, including the original chat, author, and the generated
    response.

    Attributes:
        buzz_type (str): The type or category of the buzz.
        original_chat (str): The original chat message that triggered the buzz.
        author (str): The author of the original chat message.
        generated_response (str): The generated response text.
    """

    buzz_type: str
    original_chat: str
    author: str
    generated_response: str


class WriteChatModel(BaseModel):
    """
    Represents data related to writing a chat message to a live stream.

    This model holds all the necessary information for writing a chat message
    to a live stream, including the session and chat IDs, retry count, the
    reply text, an optional reply summary, and a flag indicating if the
    message has been successfully written.

    Attributes:
        session_id (str): The unique identifier for the stream session.
        live_chat_id (str): The unique identifier for the live chat associated
            with the stream.
        retry_count (Optional[int]): The number of times the message has been
            attempted to be written. Defaults to 0.
        reply (str): The text of the reply to be written.
        reply_summary (Optional[str]): An optional summary of the reply.
            Defaults to an empty string.
        is_written (Optional[int]): An integer representing whether the message
            has been successfully written. 1 indicates success, 0 indicates failure.
            Defaults to 0.
    """

    session_id: str
    live_chat_id: str
    retry_count: Optional[int] = 0
    reply: str
    reply_summary: Optional[str] = ""
    is_written: Optional[int] = 0

class ChatIntent(BaseModel):
    original_chat: str
    author: str
    intent: str