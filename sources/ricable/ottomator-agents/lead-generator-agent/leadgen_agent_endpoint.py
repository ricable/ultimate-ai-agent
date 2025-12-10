from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import os

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)

from leadgen_agent import hunter_agent, HunterDeps

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Lead Generator API")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Request/Response Models
class LeadRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class LeadResponse(BaseModel):
    success: bool

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True    

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.post("/api/lead-generator", response_model=LeadResponse)
async def lead_generator_endpoint(
    request: LeadRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        # Store user's query immediately
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query
        )            

        # Immediately store processing message
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="Extracting leads info...\nPlease wait for few minutes.",
            data={"request_id": f"{request.request_id}_processing"}
        )

        # Return success immediately
        # Create background task for processing
        async def process_request():
            try:
                async with httpx.AsyncClient() as client:
                    deps = HunterDeps(
                        client=client,
                        hunter_api_key=os.getenv("HUNTER_API_KEY")
                    )

                    # Run the agent
                    result = await hunter_agent.run(
                        request.query,
                        deps=deps
                    )

                # Store agent's final response
                await store_message(
                    session_id=request.session_id,
                    message_type="ai",
                    content=result.data,
                    data={"request_id": request.request_id}
                )
            except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                print(error_message)
                await store_message(
                    session_id=request.session_id,
                    message_type="ai",
                    content="I apologize, but I encountered an error processing your request.",
                    data={"error": error_message, "request_id": request.request_id}
                )

        # Start background task
        import asyncio
        asyncio.create_task(process_request())

        # Return immediately
        return LeadResponse(success=True)

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        print(error_message)
        return LeadResponse(success=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
