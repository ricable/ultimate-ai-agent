import os
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agents import orchestrator
from agents.buzz_intern import buzz_intern_agent
from constants.constants import (CHAT_READ_INTERVAL, CHAT_WRITE_INTERVAL,
                                 CONVERSATION_CONTEXT)
from exceptions.user_error import UserError
from models.agent_models import AgentRequest, AgentResponse
from routers import chat_worker
from routers.chat_worker import read_live_chats, write_live_chats
from utils.supabase_util import (fetch_conversation_history,
                                 fetch_human_session_history, store_message)

# Load environment variables
load_dotenv()

# Create the scheduler instance
scheduler = AsyncIOScheduler()


# Define lifespan context manager
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Manages the application's lifespan, specifically starting and stopping the scheduler.

    This context manager is used by FastAPI to handle startup and shutdown events.
    It initializes the background scheduler with jobs for reading and writing live chats,
    starts the scheduler, and ensures it's properly shut down when the application exits.

    Args:
        _: The FastAPI application instance (unused).

    Yields:
        None: The context manager yields control back to FastAPI after starting the scheduler
            and when the application is shutting down.
    """
    # Prevent duplicate jobs if app restarts
    if not scheduler.get_job("read_live_chats"):
        scheduler.add_job(
            read_live_chats,
            "interval",
            seconds=CHAT_READ_INTERVAL,
            id="read_live_chats",
        )

    if not scheduler.get_job("write_live_chats"):
        scheduler.add_job(
            write_live_chats,
            "interval",
            seconds=CHAT_WRITE_INTERVAL,
            id="write_live_chats",
        )

    # Start the scheduler
    scheduler.start()
    print("Scheduler started...")

    # Yield control back to FastAPI
    yield

    # Shutdown the scheduler when the app stops
    scheduler.shutdown()
    print("Scheduler shut down...")


# Create FastAPI app and pass the lifespan function
app = FastAPI(lifespan=lifespan)
app.include_router(chat_worker.router)
security = HTTPBearer()

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> bool:
    """
    Verifies the provided bearer token against the configured API token.

    This function is a dependency for FastAPI routes that require authentication.
    It retrieves the expected token from the environment variable `API_BEARER_TOKEN` and
    compares it to the token provided in the `Authorization` header.

    Args:
        credentials: The authentication credentials extracted from the request header.

    Returns:
        bool: True if the token is valid.

    Raises:
        HTTPException:
            - 500: If the `API_BEARER_TOKEN` environment variable is not set.
            - 401: If the provided token does not match the expected token.
    """
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500, detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return True


@app.get("/")
async def root():
    """
    Root endpoint for the API.

    Returns:
        str: A message indicating that the Chat Worker is running and tasks are scheduled.
    """
    return "Chat Worker is up and running! Tasks have been Scheduled!"


# noinspection PyUnusedLocal
@app.post("/api/v1/streambuzz", response_model=AgentResponse)
async def sample_supabase_agent(
    request: AgentRequest, authenticated: bool = Depends(verify_token)
):
    """
    Processes a user's request using a conversational agent, interacting with Supabase.

    This endpoint receives a user's query, retrieves relevant conversation history
    from Supabase,
    sends the query to the buzz_intern_agent, stores the user's query and the agent's
    response
    in Supabase, and returns a success indicator. It also handles potential
    `UserError` exceptions
    by generating a polite error message using the buzz_intern_agent and stores that
    in Supabase.
    General exceptions are caught and an error message is stored in Supabase.

    Args:
        request: The user's request, including the query, session ID, request ID,
        and files.
        authenticated: A boolean indicating if the user is authenticated, derived
        from the `verify_token` dependency.

    Returns:
        AgentResponse: An object indicating the success or failure of the request.

    Raises:
        HTTPException: If an error occurs during the agent interaction or database
        operations.
    """
    try:
        # Fetch conversation history from the DB
        messages = await fetch_conversation_history(
            request.session_id, CONVERSATION_CONTEXT
        )

        human_messages = await fetch_human_session_history(request.session_id)

        # Store user's query with files if present
        message_data = {"request_id": request.request_id}
        if request.files:
            message_data["files"] = request.files

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data=message_data,
        )

        # Get agent's response
        agent_response = await orchestrator.get_response(
            request=request, human_messages=human_messages, messages=messages
        )

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=agent_response,
            data={"request_id": request.request_id},
        )

        return AgentResponse(success=True)
    except UserError as ue:
        user_error_string = str(ue)
        print(f"Error>> get_response: {user_error_string}")
        exception_response = await buzz_intern_agent.run(
            user_prompt=f"Respond with a short polite message within 100 words to "
                        f"convey the following error.\n{user_error_string}. You can "
                        f"use emojis sparingly to express yourself.",
            result_type=str,
            deps=request.session_id,
        )

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=exception_response.data,
            data={"request_id": request.request_id},
        )

        return AgentResponse(success=True)
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id},
        )
        return AgentResponse(success=False)

if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
