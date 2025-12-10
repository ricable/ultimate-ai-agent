from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from supabase import create_client, Client
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)
from dataclasses import dataclass
from dotenv import load_dotenv
from httpx import AsyncClient
import os
import json

# Load environment variables
load_dotenv()

# Global HTTP client
http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global http_client
    http_client = AsyncClient()

    yield

    # Shutdown
    await http_client.aclose()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    chatInput: str
    sessionId: str

class ChatResponse(BaseModel):
    output: str

# Agent dependencies
@dataclass
class AgentDeps:
    http_client: AsyncClient
    searxng_base_url: str

# Get model configuration
def get_model():
    llm = os.getenv('LLM_CHOICE', 'qwen3:14b')
    base_url = os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')
    api_key = os.getenv('LLM_API_KEY', 'ollama')
    
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

# Create the main agent with system prompt
agent = Agent(
    get_model(),
    system_prompt="You are a personal assistant who helps answer questions. You have access to web search to find current information when needed.",
    deps_type=AgentDeps,
    retries=2
)

# Create metadata agent (no tools, no deps)
metadata_agent = Agent(
    get_model(),
    system_prompt="You are a helpful assistant.",
    retries=1
)

# Web search tool
@agent.tool
async def web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the web with a specific query and get information from the top search results.
    
    Args:
        query: The search query
        
    Returns:
        A summary of the top search results
    """
    try:
        searxng_url = ctx.deps.searxng_base_url
        params = {'q': query, 'format': 'json'}
        
        response = await ctx.deps.http_client.get(f"{searxng_url}/search", params=params)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Get top 3 results
        for i, result in enumerate(data.get('results', [])[:1], 1):
            # Fetch the actual content from each URL
            try:
                page_response = await ctx.deps.http_client.get(
                    result.get('url', ''),
                    timeout=5.0,
                    follow_redirects=True
                )
                if page_response.status_code == 200:
                    # In production, you'd want to extract text from HTML properly
                    # For now, we'll use the snippet from search results
                    content = result.get('content', '')[:500]
                else:
                    content = result.get('content', '')[:500]
            except:
                content = result.get('content', '')[:500]
            
            results.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': content
            })
        
        if results:
            return json.dumps(results, indent=2)
        else:
            return "No search results found."
            
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# Bearer token verification
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="BEARER_TOKEN environment variable not set"
        )
    
    # Ensure the token is not empty or just whitespace
    expected_token = expected_token.strip()
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="BEARER_TOKEN environment variable is empty"
        )
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True

# Database operations
async def fetch_conversation_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch conversation history from Supabase."""
    try:
        response = supabase.table("n8n_chat_histories") \
            .select("*") \
            .eq("session_id", session_id) \
            .limit(limit) \
            .execute()
        
        # Reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        print(f"Error fetching conversation history: {e}")
        return []

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in Supabase."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data
    
    try:
        supabase.table("n8n_chat_histories").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        print(f"Error storing message: {e}")

# Main endpoint
@app.post("/invoke-python-agent", response_model=ChatResponse)
async def invoke_agent(
    request: ChatRequest,
    authenticated: bool = Depends(verify_token)
):
    """Main endpoint that handles chat requests with web search capability."""
    
    # Check if this is a metadata request (starting with "### Task")
    if request.chatInput.startswith("### Task"):
        # For metadata requests, use the metadata agent without history
        result = await metadata_agent.run(request.chatInput)
        print(result.output)
        return ChatResponse(output=result.output)
    
    try:
        # Fetch conversation history
        history = await fetch_conversation_history(request.sessionId)
        
        # Convert conversation history to Pydantic AI message format
        messages = []
        for msg in history:
            msg_data = msg.get("message", {})
            msg_type = msg_data.get("type")
            msg_content = msg_data.get("content", "")
            
            if msg_type == "human":
                messages.append(ModelRequest(parts=[UserPromptPart(content=msg_content)]))
            else:
                messages.append(ModelResponse(parts=[TextPart(content=msg_content)]))
        
        # Store user's message
        await store_message(
            session_id=request.sessionId,
            message_type="human",
            content=request.chatInput
        )
        
        # Create agent dependencies
        deps = AgentDeps(
            http_client=http_client,
            searxng_base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
        )
        
        # Run the agent with message history
        result = await agent.run(request.chatInput, message_history=messages, deps=deps)
        print(result.output)
        
        # Store agent's response
        await store_message(
            session_id=request.sessionId,
            message_type="ai",
            content=result.output
        )
        
        return ChatResponse(output=result.output)
        
    except Exception as e:
        error_message = f"I encountered an error: {str(e)}"
        
        # Store error response
        await store_message(
            session_id=request.sessionId,
            message_type="ai",
            content=error_message
        )
        
        return ChatResponse(output=error_message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8055)