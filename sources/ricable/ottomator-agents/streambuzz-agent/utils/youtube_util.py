import json
import os
import re
import time
from urllib.parse import parse_qs, urlparse

import requests
from cachetools.func import ttl_cache
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from requests import HTTPError

from constants.constants import (ALLOWED_DOMAINS, OAUTH_TOKEN_URI, YOUTUBE_API_ENDPOINT,
                                 YOUTUBE_LIVE_API_ENDPOINT, YOUTUBE_SSL)
from constants.enums import BuzzStatusEnum
from exceptions.user_error import UserError
from logger import log_method
from utils import supabase_util

# Load environment variables from .env file
load_dotenv()

# Retrieve and parse the dictionary
YOUTUBE_API_KEY_BUNCHES_ENV = json.loads(os.getenv("YOUTUBE_API_KEY_BUNCHES"))


@ttl_cache(ttl=3600)
def get_youtube_api_key_bunches():
    youtube_api_key_bunches = []
    for key_bunch in YOUTUBE_API_KEY_BUNCHES_ENV:
        creds = Credentials(
            None,  # No initial access token
            refresh_token=key_bunch["refresh_token"],
            token_uri=OAUTH_TOKEN_URI,
            client_id=key_bunch["client_id"],
            client_secret=key_bunch["client_secret"],
            scopes=[YOUTUBE_SSL]
        )
        if not creds.valid:
            creds.refresh(Request())
            print("Token refreshed successfully.")
        key_bunch["access_token"] = creds.token
        youtube_api_key_bunches.append(key_bunch)

    return youtube_api_key_bunches


@log_method
async def validate_and_extract_youtube_id(url: str) -> str:
    """
    Validates a YouTube URL and extracts the video ID if the URL is valid.

    This function parses the provided URL, checks if the domain is valid
    (either 'youtu.be' or 'youtube.com'), and extracts the 11-character
    alphanumeric video ID from the URL. It supports standard, shortened,
    and embed URL formats.

    Args:
        url (str): The YouTube URL to validate and parse.

    Returns:
        str: The extracted video ID if the URL is valid.

    Raises:
        UserError: If the provided URL is invalid due to an invalid domain,
                   invalid path, or invalid video ID format.
        Exception: If there's any other unexpected error during the process.
    """
    try:
        url_pattern = r"(https?://|www\.)\S+"
        match = re.search(url_pattern, url)
        if not match:
            raise UserError("No URL found in the text.")

        # Parse the YouTube URL
        url = match.group()
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # Check if the domain is valid
        if domain not in ALLOWED_DOMAINS:
            raise UserError(f"Invalid YouTube domain: {domain}")

        # Extract video ID based on URL format
        if "youtu.be" in domain:
            # Shortened URL
            video_id = path[1:]  # Skip leading '/'
        elif "youtube.com" in domain:
            # Standard URL or embed URL
            if path == "/watch" and "v" in query:
                video_id = query["v"][0]
            elif path.startswith("/embed/"):
                video_id = path.split("/embed/")[1]
            else:
                raise UserError(f"Invalid YouTube video URL path: {path}")
        else:
            raise UserError("Unrecognized YouTube URL format.")

        # Validate video ID format (11-character alphanumeric)
        if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
            raise UserError(f"Invalid YouTube video ID: {video_id}")
        return video_id
    except UserError as ue:
        print(f"Error validating and extracting YouTube ID: {str(ue)}")
        raise
    except Exception as e:
        print(f"Error validating and extracting YouTube ID: {str(e)}")
        raise


@log_method
async def get_stream_metadata(video_id: str, session_id: str) -> dict:
    """
    Retrieves stream metadata for a given YouTube video ID using the YouTube API.

    This function takes a YouTube video ID, calls the YouTube API to get
    live-streaming details and snippet information, and returns the API response.
    It uses a retry mechanism with multiple API keys if the initial request fails.

    Args:
        video_id (str): The YouTube video ID to fetch metadata for.
        session_id (str): The session ID associated with the request.

    Returns:
        dict: The JSON response from the YouTube API containing stream metadata.

    Raises:
        UserError: If there's an error related to the user input or API usage.
        Exception: If there's any other unexpected error during the process.
    """
    try:
        params = {
            "part": "liveStreamingDetails,snippet",
            "id": video_id,
        }
        response = await get_request_with_retries(
            YOUTUBE_API_ENDPOINT, params, session_id, use_keys=True
        )

        return response
    except UserError as ue:
        print(f"Error>> get_stream_metadata: {str(ue)}")
        raise
    except Exception as e:
        print(f"Error>> get_stream_metadata: {str(e)}")
        raise


@log_method
async def post_request_with_retries(
        url: str, params: dict, payload: dict, use_keys: bool = False
) -> dict:
    """
    Makes a POST request with retries using multiple API keys.

    This function iterates through a list of API keys, attempting to make a POST
    request to the specified URL. If a request fails, it retries with the next key
    after a short delay. The function handles both API key authentication and
    bearer token authentication based on the `use_keys` flag.

    Args:
        url (str): The URL to make the POST request to.
        params (dict): The parameters to include in the POST request.
        payload (dict): The payload to include in the POST request.
        use_keys (bool): If True, uses 'api_key' for authentication;
                        otherwise, uses 'access_token'. Defaults to False.

    Returns:
        dict: The JSON response from the POST request if successful.

    Raises:
        HTTPError: If all API keys fail or the maximum number of retries is reached.
    """
    api_key_bunches = get_youtube_api_key_bunches()
    for attempt, key_dict in enumerate(api_key_bunches):
        if use_keys:
            params["key"] = key_dict["api_key"]
            response = requests.post(url, params=params, data=payload, timeout=10)
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key_dict['access_token']}",
            }
            response = requests.post(
                url, headers=headers, params=params, data=payload, timeout=10
            )

        try:
            if response.status_code == 200:
                return response.json()

            # Log the failure
            print(
                f"Attempt {attempt + 1}: {response.status_code=}\nBody="
                f"{response.json()}. Retrying..."
            )

        except requests.exceptions.RequestException as e:
            # Log the exception
            print(
                f"Error>> {str(e)}\nAttempt {attempt + 1}: {response.status_code=}\n"
                f"body={response.json()}. Retrying..."
            )

        # Retry after a short delay
        time.sleep(2)

    # If all attempts fail
    raise HTTPError("All API keys failed or maximum retries reached.")


@log_method
async def get_request_with_retries(
        url: str, params: dict, session_id: str, use_keys: bool = True
) -> dict:
    """Makes a GET request with retries using multiple API keys.

    This function iterates through a list of API keys, attempting to make a GET
    request to the specified URL. If a request fails, it retries with the next key
    after a short delay. The function handles both API key authentication and
    bearer token authentication based on the `use_keys` flag.

    Args:
        url (str): The URL to make the GET request to.
        params (dict): The parameters to include in the GET request.
        session_id (str): The session ID associated with the request.
        use_keys (bool): If True, uses 'api_key' for authentication;
                        otherwise, uses 'access_token'. Defaults to True.

    Returns:
        dict: The JSON response from the GET request if successful.

    Raises:
        HTTPError: If all API keys fail or the maximum number of retries is reached.
    """
    api_key_bunches = get_youtube_api_key_bunches()
    for attempt, key_dict in enumerate(api_key_bunches):
        if use_keys:
            params["key"] = key_dict["api_key"]
            response = requests.get(url, params=params, timeout=10)
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key_dict['access_token']}",
            }
            response = requests.get(url, headers=headers, params=params, timeout=10)

        try:
            if response.status_code == 200:
                return response.json()
            error_reason = None
            if 400 <= response.status_code < 500:
                error_reason = (
                    response.json()
                    .get("error", {})
                    .get("errors", [{}])[0]
                    .get("reason")
                )
            if response.status_code == 400:
                print(f"Bad Request (400): {error_reason}\nBreaking...")
                break
            elif response.status_code == 403 and error_reason == "liveChatEnded":
                print("Live Chat Ended: Deactivating stream and breaking...")
                await deactivate_stream(
                    session_id=session_id,
                    message="The current YouTube Live Stream has ended. You can explore the buzz so far, but replies are disabled. Start a new stream anytime!",
                )
                break
            else:
                print(
                    f"Attempt {attempt + 1}: {response.status_code=}\nBody="
                    f"{response.json()}. Retrying..."
                )

        except requests.exceptions.RequestException as e:
            # Log the exception
            print(
                f"Error>> {str(e)}\nAttempt {attempt + 1}: {response.status_code=}\n"
                f"body={response.json()}. Retrying..."
            )

        # Retry after a short delay
        time.sleep(2)

    # If all attempts fail
    raise HTTPError("All API keys failed, maximum retries reached or bad request.")


@log_method
async def deactivate_stream(session_id: str, message: str = None) -> None:
    """
    Deactivates all streams and marks all replies as inactive for a given session.

    This function uses the provided session ID to deactivate existing streams
    associated with the session and marks all replies as inactive. This is
    typically done when a new stream is being processed for the same session.

    Args:
        session_id (str): The session ID for which to mark streams as unavailable
                          and replies as inactive.
        message (str, optional): A message to store as an agent response.
                                 Defaults to None.

    Raises:
        Exception: If there is an error during the deactivation process.
    """
    try:
        await supabase_util.deactivate_existing_streams(session_id)
        await supabase_util.deactivate_replies(session_id)
        if message:
            # Store agent's response
            await supabase_util.store_message(
                session_id=session_id,
                message_type="ai",
                content=message,
                data={"session_id": f"Deactivated {session_id}"},
            )
    except Exception as e:
        print(f"Error>> deactivate_stream: {session_id=}\n{str(e)}")
        raise


@log_method
async def deactivate_session(session_id: str) -> None:
    """Deactivates all streams and marks all buzz as inactive for a given session.

    This function deactivates streams associated with the session, and updates
    the buzz status to inactive for all buzz entries associated with the
    given session ID. This is typically used when a user prompts a new link
    in the session.

    Args:
        session_id (str): The session ID for which to deactivate streams and
                         mark buzz as inactive.

    Raises:
        Exception: If there is an error during the deactivation or update process.
    """
    try:
        await deactivate_stream(session_id)
        await supabase_util.update_buzz_status_by_session_id(
            session_id, BuzzStatusEnum.INACTIVE.value
        )
    except Exception as e:
        print(f"Error>> deactivate_session: {session_id=}\n{str(e)}")
        raise


@log_method
async def get_live_chat_messages(session_id: str, live_chat_id: str,
                                 next_chat_page: str) -> list:
    """
    Retrieves live chat messages from YouTube using the Live Chat API.

    This function fetches live chat messages from a specified YouTube live chat,
    formats the messages, and updates the next page token.

    Args:
        session_id (str): The session ID associated with the request.
        live_chat_id (str): The YouTube live chat ID to fetch messages from.
        next_chat_page (str): The token for the next page of chat messages.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chat
              message with 'original_chat' and 'author' keys.

    Raises:
        Exception: If there's an error during the API request or data processing.
    """
    params = {"part": "snippet, authorDetails", "liveChatId": live_chat_id}
    if next_chat_page and next_chat_page.strip():
        params["pageToken"] = next_chat_page.strip()
    # Fetch live chat messages
    live_chat_response = await get_request_with_retries(
        url=YOUTUBE_LIVE_API_ENDPOINT,
        params=params,
        session_id=session_id,
        use_keys=True,
    )

    # Extract chats as a list of dictionaries with updated displayName formatting
    chats = [
        {
            "original_chat": item.get("snippet").get("displayMessage"),
            "author": (
                f"@{item.get('authorDetails').get('displayName')}"
                + (
                    " (owner)"
                    if item.get("authorDetails").get("isChatOwner", False)
                    else ""
                )
                + (
                    " (sponsor)"
                    if item.get("authorDetails").get("isChatSponsor", False)
                    else ""
                )
                + (
                    " (verified)"
                    if item.get("authorDetails").get("isVerified", False)
                    else ""
                )
                + (
                    " (moderator)"
                    if item.get("authorDetails").get("isChatModerator", False)
                    else ""
                )
            ),
        }
        for item in live_chat_response.get("items", [])
    ]

    # Extract next_chat_page and update the next chat page token
    next_chat_page = live_chat_response.get("nextPageToken", "")
    await supabase_util.update_next_chat_page(live_chat_id, next_chat_page)

    return chats
