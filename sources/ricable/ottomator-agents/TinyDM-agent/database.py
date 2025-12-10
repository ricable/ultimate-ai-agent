from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client with service role key for admin access
supabase = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")  # Use the anon key
)

async def store_message(session_id: str, message: Dict[str, Any]) -> bool:
    """Store a message in the database."""
    try:
        data = supabase.table("messages").insert({
            "session_id": session_id,
            "message": message
        }).execute()
        
        if hasattr(data, 'error') and data.error:
            logger.error(f"Error storing message: {data.error}")
            return False
            
        logger.info(f"Successfully stored message for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing message: {e}")
        return False

async def get_session_messages(session_id: str) -> list:
    """Retrieve all messages for a given session."""
    try:
        response = supabase.table("messages")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at")\
            .execute()
            
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error retrieving messages: {response.error}")
            return []
            
        return response.data
    except Exception as e:
        logger.error(f"Error retrieving messages: {e}")
        return []
