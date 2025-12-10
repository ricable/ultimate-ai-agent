import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from supabase import create_client

load_dotenv()

# Files
TITLE = "title"
SUMMARY = "summary"
ACCEPTED_FILE_QUANTITY = 1
"""The number of files accepted for processing."""
ACCEPTED_FILE_MIME = "text/plain"
"""The accepted MIME type of the file."""
ACCEPTED_FILE_EXTENSION = ".txt"
"""The accepted file extension."""
MAX_FILE_SIZE_MB = 5
"""Maximum file size in megabytes."""
MAX_FILE_SIZE_B = MAX_FILE_SIZE_MB * 1024 * 1024  # 5MB
"""Maximum file size in bytes."""


# Intents
CHAT_READ_INTERVAL = 30
"""Interval in seconds to read chat messages."""
CHAT_WRITE_INTERVAL = 60
"""Interval in seconds to write chat messages."""
CONVERSATION_CONTEXT = 3
"""Number of previous messages to include in the conversation context."""
START_STREAM_APPEND = f"\n\nFetching buzz in {CHAT_READ_INTERVAL} seconds..."
"""Message appended to start of stream."""
CONFIDENCE_THRESHOLD = 0.35
STREAMER_INTENT_EXAMPLES = {
    "START_STREAM": [
        "Start the stream",
        "Begin streaming",
        "Initiate the live stream",
        "Go live",
        "Launch the stream",
        "Start broadcasting",
        "Begin live transmission",
        "Kick off the stream",
        "Turn on the stream",
        "Start moderating",
        "Begin moderating the stream",
        "Activate live moderation",
        "Enable stream moderation",
        "Start monitoring the stream",
    ],
    "GET_CURRENT_CHAT": [
        "Get now",
        "Get current",
        "Display current",
        "What’s happening now?",
        "Show me the current buzz",
        "Fetch latest update",
        "What’s trending now?",
        "Show the current status",
        "What is happening at the moment?",
        "Give me the latest buzz",
        "Retrieve the latest topic",
        "What’s live right now?",
    ],
    "GET_NEXT_CHAT": [
        "Get next",
        "Display next",
        "What’s next?",
        "Tell me the next one",
        "Show the upcoming buzz",
        "What’s coming up?",
        "After this, what’s next?",
        "Show me the next trending topic",
        "Give me the next item",
        "What’s the next update?",
        "What follows this?",
        "Move to the next topic",
    ],
    "REPLY_CHAT": [
        "Respond to the message",
        "Write a reply",
        "Reply as",
        "Post reply",
        "Post a response",
        "Send an answer",
        "Write a message back",
        "Craft a response",
        "Respond appropriately",
        "Answer the user",
        "Type a reply",
        "React to this message",
        "Post a reply",
        "Reply with the right words",
        "Post a reply about this",
        "Post a reply AI agents are amazing",
        "Post a reply ...",
        "Write a reply now",
        "Reply with a comment",
        "Reply to this statement",
        "Write a quick reply",
        "Post your reply to this",
        "Draft a reply",
        "Post a reply regarding this",
        "Send a reply about this",
        "Type your reply",
        "Post a relevant reply",
        "Post a reply and share thoughts",
        "Send a reply, AI agents are amazing",
        "Post a response about AI agents",
        "Draft a reply based on the topic",
        "Write a follow-up reply",
        "Reply to this with context",
        "Post a meaningful reply",
        "Reply to this, it’s amazing",
        "Craft a reply for this statement",
        "Reply now to this amazing post",
        "Reply to the message, amazing AI",
        "Post a proper reply to this query",
        "Write a thoughtful reply",
        "Post a reply and expand on this",
        "Share your reply about AI"
    ],
    "UNKNOWN": [
        "Hi",
        "Hello",
        "Whatsup",
        "What is pydantic ai?",
        "Use knowledge base to answer",
        "Use RAG to answer",
        "Random query",
        "Unrelated question",
        "This doesn't fit any category",
        "Tell me a joke",
        "Who is the president of Mars?",
        "Define an impossible concept",
        "What’s the meaning of life?",
        "Completely off-topic question",
    ],
}

"""Examples of streamer intents and their corresponding phrases.

    This dictionary maps intent names (keys) to lists of example phrases (values)
    that a streamer might use to trigger that intent.
"""

# Model Constants
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
"""Name of the embedding model.

    This string represents the specific model name used for generating embeddings.
"""
EMBEDDING_DIMENSIONS = 768
"""Dimensionality of the generated embeddings.

    This integer represents the number of dimensions in the vector space where text
    is embedded.
"""
CHUNK_SIZE = 1000
"""Size of text chunks used for processing.

    This integer defines the maximum number of characters to consider when processing
    text in chunks.
"""
MODEL_RETRIES = 3
"""Number of retries for model calls.

    This integer specifies the maximum number of times to retry a model call in case
    of failure.
"""

# Open Router Pydantic-AI model related constants
OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"
"""Base URL for Open Router API.

    This string defines the base URL used to make API requests to Open Router.
"""
# OPEN_ROUTER_MODEL_NAME = "google/gemini-2.0-flash-thinking-exp:free"
OPEN_ROUTER_MODEL_NAME = "openai/gpt-4o-mini"
"""Name of the Open Router model.

    This string specifies the name of the model to be used from the Open Router API.
"""
OPEN_ROUTER_CLIENT = OpenAI(
    base_url=OPEN_ROUTER_BASE_URL,
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)
"""OpenAI client configured for Open Router.

    This instance of the OpenAI client is configured to use the Open Router API.
"""
OPEN_ROUTER_MODEL = OpenAIModel(
    model_name=OPEN_ROUTER_MODEL_NAME,
    base_url=OPEN_ROUTER_BASE_URL,
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)
"""Pydantic-AI OpenAI model configured for Open Router.

    This instance of the Pydantic-AI OpenAI model is configured to use the Open Router API.
"""

# Gemini Pydantic-AI model related constants - testing
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
"""Name of the Gemini model.

    This string specifies the name of the Gemini model to be used.
"""
GEMINI_MODEL = GeminiModel(
    model_name=GEMINI_MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY")
)
"""Pydantic-AI Gemini model.

   This instance of the Pydantic-AI Gemini model is configured with the API key.
"""

# Model used in project
# PYDANTIC_AI_MODEL = GEMINI_MODEL
PYDANTIC_AI_MODEL = OPEN_ROUTER_MODEL
"""The Pydantic-AI model used in the project.

    This is the main model used for generating text responses.
"""

# YouTube API related constants
YOUTUBE_URL_REGEX = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/[a-zA-Z0-9_\-]+"
"""Regex for matching YouTube URLs.

    This regular expression is used to validate YouTube URLs.
"""
ALLOWED_DOMAINS = ["youtube.com", "www.youtube.com", "youtu.be"]
"""Allowed domains for YouTube URLs.

    This list specifies the valid domains for YouTube URLs.
"""
YOUTUBE_API_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
"""Endpoint for YouTube API videos.

    This string defines the API endpoint for retrieving video information from YouTube.
"""
YOUTUBE_LIVE_API_ENDPOINT = "https://www.googleapis.com/youtube/v3/liveChat/messages"
"""Endpoint for YouTube API live chat messages.

    This string defines the API endpoint for retrieving live chat messages from YouTube.
"""
OAUTH_TOKEN_URI = "https://oauth2.googleapis.com/token"
YOUTUBE_SSL = "https://www.googleapis.com/auth/youtube.force-ssl"

# Supabase setup constants
SUPABASE_CLIENT = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
)
"""Supabase client.

    This instance of the Supabase client is configured to interact with the Supabase database.
"""

# Table names
MESSAGES = "messages"
"""Name of the messages table.

    This string represents the name of the table storing chat messages.
"""
YT_STREAMS = "youtube_streams"
"""Name of the youtube streams table.

    This string represents the name of the table storing youtube stream information.
"""
YT_BUZZ = "youtube_buzz"
"""Name of the youtube buzz table.

    This string represents the name of the table storing youtube buzz information.
"""
YT_REPLY = "youtube_reply"
"""Name of the youtube reply table.

    This string represents the name of the table storing youtube reply information.
"""
STREAMER_KB = "streamer_knowledge"
"""Name of the streamer knowledge table.

    This string represents the name of the table storing streamer knowledge base.
"""
