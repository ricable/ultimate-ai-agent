# Request/Response Models
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class ProcessedChunk:
    """Represents a processed chunk of a document.

    Attributes:
        session_id (str): The ID of the session this chunk belongs to. This typically
            identifies the context of the processing, allowing for grouping of related
            chunks.
        file_name (str): The name of the file the chunk originates from. This is used
            for tracking the source of the information.
        chunk_number (int): The sequential number of the chunk within the file. This
            is useful for maintaining the original order of the text within the file.
        title (str): The title of the chunk. This is often a short, descriptive
            summary of the chunk's topic.
        summary (str): A brief summary of the chunk's content. This provides a concise
            overview of the main points of the chunk.
        content (str): The actual text content of the chunk. This is the raw
            text extracted from the document.
        embedding (List[float]): A numerical representation (embedding) of the chunk's
            content. This is used for semantic search and other machine learning tasks.
            Embeddings capture the meaning of the text in a vector space.
    """

    session_id: str
    file_name: str
    chunk_number: int
    title: str
    summary: str
    content: str
    embedding: List[float]


@dataclass
class ProcessFoundBuzz:
    """Represents a buzz (a specific piece of information or finding) identified during processing.

    Attributes:
        id (int): A unique identifier for the buzz. This ensures that each buzz can
            be tracked and referenced individually.
        buzz_type (str): The type or category of the buzz. This provides context about
            what kind of information the buzz represents (e.g., "key takeaway",
            "action item", "problem identified").
        original_chat (str): The original text where the buzz was identified. This
            allows for easy reference to the source of the buzz.
    """

    id: int
    session_id: str
    author: str
    buzz_type: str
    original_chat: str


class AgentRequest(BaseModel):
    """Represents a request sent to an agent.

    This model encapsulates all necessary information for an agent to process a user's
    request, including the query, user details, session context, and optional file
    metadata.

    Attributes:
        query (str): The user's query or request. This is the main instruction or
            question that the agent needs to address.
        user_id (str): The ID of the user making the request. This identifies the
            originator of the request, allowing for user-specific handling.
        request_id (str): A unique ID for this specific request. This is used to track
            and manage individual requests throughout the system.
        session_id (str): The ID of the current session. This provides context for the
            request, allowing agents to access and utilize relevant session data.
        files (Optional[List[Dict[str, Any]]], optional): An optional list of file
            metadata associated with the request. Each dictionary within the list
            represents a file and may contain details such as file name, size,
            and type. Defaults to None if no files are included in the request.
    """

    query: str
    user_id: str
    request_id: str
    session_id: str
    files: Optional[List[Dict[str, Any]]] = None


class AgentResponse(BaseModel):
    """Represents a response from an agent.

    This model provides a basic structure for agent responses, indicating the
    success or failure of the request processing. More complex responses might
    include additional fields to provide more detailed information.

    Attributes:
        success (bool): A boolean indicating whether the request was processed
            successfully. A value of `True` indicates success, while `False`
            indicates failure.
    """

    success: bool
