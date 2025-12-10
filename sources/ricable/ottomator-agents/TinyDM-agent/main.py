from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from google import genai
from database import store_message, get_session_messages
import base64
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="TinyDM")

# Initialize Gemini with new SDK
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    client = genai.Client(api_key=api_key)
    
    # Test the model
    test_response = client.models.generate_content(model='gemini-2.0-flash-exp', contents='Hello')
    if test_response and test_response.text:
        logger.info("Gemini model initialized and tested successfully")
    else:
        raise ValueError("Model test failed - empty response")
except Exception as e:
    logger.error(f"Error initializing Gemini: {str(e)}")
    raise

class FileUpload:
    def __init__(self, name: str, type: str, base64: str):
        self.name = name
        self.type = type
        self.base64 = base64

    def validate(self) -> bool:
        # Check file size (5MB limit)
        file_size = len(self.base64) * 3 / 4  # Approximate size of decoded base64
        return file_size <= 5 * 1024 * 1024  # 5MB in bytes

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def handle_dm_request(request: Dict[str, Any], authorization: str = None):
    # Validate bearer token
    if authorization != f"Bearer {os.getenv('BEARER_TOKEN')}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Extract request information
        query = request.get("query")
        session_id = request.get("session_id")
        user_id = request.get("user_id")
        request_id = request.get("request_id")
        files = request.get("files", [])
        
        logger.info(f"Received request - Session: {session_id}, User: {user_id}, Request: {request_id}")
        
        if not all([query, session_id, user_id, request_id]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Check if this is the first message in the session
        session_messages = await get_session_messages(session_id)
        is_first_message = len(session_messages) == 0

        # If this is the first message, send a welcome message
        if is_first_message:
            welcome_message = {
                "type": "ai",
                "content": """Greetings, brave adventurer! 🎲✨ I am TinyDM, your companion in the realms of imagination. How may I assist you today?

I can help you with:
🗡️ Creating combat encounters
🏰 Generating vivid location descriptions
👥 Crafting memorable NPCs
🐉 Detailing fearsome monsters
✨ Explaining spells and magic
📖 Clarifying rules and mechanics
🌍 Building entire worlds and campaigns

Or perhaps you seek something else? Whatever your quest, I'm here to guide you through the infinite possibilities of D&D!"""
            }
            # This first message will be the description for the agent and displayed as the first message in the studio
            # await store_message(session_id, welcome_message)

        # Validate files if present
        if files:
            if len(files) > 5:  # Max 5 files per message
                raise HTTPException(status_code=400, detail="Too many files")
            
            for file in files:
                file_obj = FileUpload(**file)
                if not file_obj.validate():
                    raise HTTPException(status_code=400, detail=f"File {file_obj.name} exceeds size limit")

        # Store user message
        user_message = {
            "type": "human",
            "content": query
        }
        await store_message(session_id, user_message)
        logger.info("Stored user message")

        # Check if user is asking for help
        help_keywords = ["help", "what can you do", "capabilities", "what do you do", "assist"]
        is_help_request = any(keyword in query.lower() for keyword in help_keywords)
        
        if is_help_request:
            ai_response = """Greetings, brave adventurer! 🎲✨ I am TinyDM, your companion in the realms of imagination. How may I assist you today?

I can help you with:
🗡️ Creating combat encounters
🏰 Generating vivid location descriptions
👥 Crafting memorable NPCs
🐉 Detailing fearsome monsters
✨ Explaining spells and magic
📖 Clarifying rules and mechanics
🌍 Building entire worlds and campaigns

Or perhaps you seek something else? Whatever your quest, I'm here to guide you through the infinite possibilities of D&D!"""
        else:
            # Generate response using Gemini
            system_prompt = """You are TinyDM, a Dungeons & Dragons assistant. You help players and DMs with rules, 
            character creation, story ideas, and general D&D knowledge. Be concise but helpful, and maintain the fantasy atmosphere 
            in your responses. If you're not sure about something, say so rather than making up incorrect information."""
            
            prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
            
            try:
                logger.info("Generating response with Gemini")
                response = client.models.generate_content(model='gemini-2.0-flash-exp', contents=prompt)
                if not response:
                    raise ValueError("Empty response from Gemini")
                ai_response = response.text
                logger.info("Generated AI response successfully")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

        # Store AI response
        ai_message = {
            "type": "ai",
            "content": ai_response
        }
        await store_message(session_id, ai_message)
        logger.info("Stored AI response")
        
        return {
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error in handle_dm_request: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Root endpoint for Agent0
@app.post("/")
async def root(request: Dict[str, Any], authorization: str = Header(None)):
    return await handle_dm_request(request, authorization)

# Keep the original endpoint as an alias
@app.post("/api/dm-assistant")
async def dm_assistant(request: Dict[str, Any], authorization: str = Header(None)):
    return await handle_dm_request(request, authorization)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )
