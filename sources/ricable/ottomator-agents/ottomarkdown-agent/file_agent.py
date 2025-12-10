from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import base64
from openai import OpenAI
from markitdown import MarkItDown
import hashlib
from datetime import datetime
import logging
import imghdr
import io
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('markdown_results/file_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)  # Force reload environment variables

# Initialize FastAPI app and OpenRouter client
app = FastAPI()
security = HTTPBearer()
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:8001",  # Required for OpenRouter
        "X-Title": "MarkItDown App",  # Optional, for OpenRouter analytics
    }
)

# Initialize MarkItDown globally
md = MarkItDown(
    llm_client=openai_client,
    llm_model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
)

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str
    files: Optional[List[Dict[str, Any]]] = None

class AgentResponse(BaseModel):
    success: bool
    markdown: str = ""
    error: Optional[str] = ""

class FileRequest(BaseModel):
    file: Dict[str, Any]  # Should include name, base64, type, and optionally model

class MarkdownResponse(BaseModel):
    success: bool
    markdown: str = ""
    error: str = ""

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
    # Truncate content if it's too large (100KB limit)
    max_content_size = 100000  # 100KB
    if len(content) > max_content_size:
        content = content[:max_content_size] + "\n...(truncated)"
        logger.warning(f"Message content truncated to {max_content_size} characters")

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
            
    except Exception as e:
        logger.error(f"Failed to store message: {e}")
        # Don't raise the exception, just log it
        # This prevents message storage failures from breaking the main functionality

async def generate_summary(text: str) -> str:
    """Generate a summary using the OpenRouter API with Mistral model."""
    try:
        model = os.getenv("OPENROUTER_MODEL")
        if not model:
            raise ValueError("OPENROUTER_MODEL environment variable not set")
            
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise summaries."},
                {"role": "user", "content": f"Please provide a brief summary of the following content:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Summary generation failed"

async def save_markdown_file(filename: str, content: str, summary: str = None) -> str:
    """Save markdown content to a file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs('markdown_results', exist_ok=True)
        
        # Clean filename and create full path
        clean_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        base, ext = os.path.splitext(clean_filename)
        markdown_path = os.path.join('markdown_results', f"{base}.md")
        
        # Combine summary and content
        full_content = ""
        if summary:
            full_content = f"# Summary\n{summary}\n\n# Content\n{content}"
        else:
            full_content = content
            
        # Save to file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
            
        return markdown_path
    except Exception as e:
        logger.error(f"Error saving markdown file: {str(e)}")
        return ""

async def process_files_to_string(files: Optional[List[Dict[str, Any]]], query: str = "") -> str:
    """Convert a list of files with base64 content into a formatted string using MarkItDown."""
    if not files:
        return ""
        
    file_content = "File content to use as context:\n\n"
    
    for i, file in enumerate(files, 1):
        try:
            # Skip system files
            if file['name'].startswith('.'):
                logger.info(f"Skipping system file: {file['name']}")
                continue
                
            # Save base64 content to a temporary file
            decoded_content = base64.b64decode(file['base64'])
            
            # Detect if the content is an image using imghdr
            content_stream = io.BytesIO(decoded_content)
            image_type = imghdr.what(content_stream)
            is_image = image_type is not None
            
            temp_file_path = f"/tmp/temp_file_{file['name']}"
            with open(temp_file_path, "wb") as f:
                f.write(decoded_content)
            
            # Create appropriate MarkItDown instance based on file type
            if is_image:
                vlm_model = os.getenv("OPENROUTER_VLM_MODEL")
                if not vlm_model:
                    raise ValueError("OPENROUTER_VLM_MODEL environment variable not set")
                    
                logger.info(f"Detected image type: {image_type}, using vision model: {vlm_model}")
                temp_md = MarkItDown(
                    llm_client=openai_client,
                    llm_model=vlm_model
                )
            else:
                model = os.getenv("OPENROUTER_MODEL")
                if not model:
                    raise ValueError("OPENROUTER_MODEL environment variable not set")
                    
                temp_md = MarkItDown(
                    llm_client=openai_client,
                    llm_model=model
                )
            
            # Convert file to markdown using MarkItDown
            result = temp_md.convert(temp_file_path, use_llm=True)
            markdown_content = result.text_content
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
            # If query is provided, use it with LLM
            if query:
                response = openai_client.chat.completions.create(
                    model=os.getenv("OPENROUTER_MODEL"),
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes text based on user queries."},
                        {"role": "user", "content": f"{query}\n\nText to process:\n{markdown_content}"}
                    ]
                )
                processed_content = response.choices[0].message.content
                file_content += f"{i}. {file['name']}:\n\n{processed_content}\n\n"
            else:
                file_content += f"{i}. {file['name']}:\n\n{markdown_content}\n\n"
                
            logger.info(f"Successfully processed {file['name']}")
            
        except Exception as e:
            logger.error(f"Error processing file {file['name']}: {str(e)}")
            # Fallback to direct text conversion if markdown conversion fails
            try:
                if is_image:
                    file_content += f"{i}. {file['name']} (image file - processing failed)\n\n"
                else:
                    text_content = decoded_content.decode('utf-8')
                    file_content += f"{i}. {file['name']} (plain text):\n\n{text_content}\n\n"
            except:
                file_content += f"{i}. {file['name']} (failed to process)\n\n"
    
    return file_content

async def get_document_hash(file_data: Dict[str, Any]) -> str:
    """Generate a unique hash for a document based on its content"""
    content = file_data.get('base64', '') or file_data.get('content', '')
    name = file_data.get('name', '')
    file_type = file_data.get('type', '')
    
    # Combine all fields to create a unique hash
    hash_input = f"{content}{name}{file_type}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()

async def store_document_markdown(
    supabase_client,
    doc_hash: str,
    markdown: str,
    file_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Store document markdown in Supabase"""
    doc_data = {
        'doc_hash': doc_hash,
        'file_name': file_data.get('name'),
        'file_type': file_data.get('type'),
        'markdown_content': markdown,
        'created_at': datetime.utcnow().isoformat(),
        'last_accessed': datetime.utcnow().isoformat()
    }
    
    result = supabase_client.table('document_cache').upsert(doc_data).execute()
    return result.data[0] if result.data else None

async def get_cached_markdown(
    supabase_client,
    doc_hash: str
) -> Optional[str]:
    """Retrieve cached markdown from Supabase"""
    result = supabase_client.table('document_cache')\
        .select('markdown_content')\
        .eq('doc_hash', doc_hash)\
        .execute()
    
    if result.data:
        # Update last accessed timestamp
        supabase_client.table('document_cache')\
            .update({'last_accessed': datetime.utcnow().isoformat()})\
            .eq('doc_hash', doc_hash)\
            .execute()
        return result.data[0]['markdown_content']
    return None

async def process_file_cached(name: str, file_type: str, base64_content: str, model: str, use_cache: bool = True) -> Optional[str]:
    """Process a single file with caching."""
    try:
        # Create file data
        file_data = {
            'name': name,
            'type': file_type,
            'base64': base64_content,
            'model': model
        }
        
        # Get document hash
        doc_hash = await get_document_hash(file_data)
        
        if use_cache:
            # Try to get cached markdown
            cached_markdown = await get_cached_markdown(supabase, doc_hash)
            if cached_markdown:
                return cached_markdown
        
        # Convert file if not in cache
        markdown = await process_files_to_string([file_data])
        if markdown:
            # Store in cache
            await store_document_markdown(supabase, doc_hash, markdown, file_data)
            return markdown
            
    except Exception as e:
        logger.error(f"Error processing file {name}: {str(e)}")
        return None

@app.post("/api/file-agent", response_model=AgentResponse)
async def file_agent(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        logger.info(f"Received request: {request}")
        
        # Fetch conversation history from the DB
        conversation_history = await fetch_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = "user" if msg_data["type"] == "human" else "assistant"
            msg_content = msg_data["content"]
            
            messages.append({"role": msg_type, "content": msg_content})

        # Store user's query with files if present
        message_data = {"request_id": request.request_id}
        if request.files:
            message_data["files"] = request.files

        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data=message_data
        )

        # Get markdown content from files using query as context
        markdown_content = "No markdown generated from your request. Upload a file and I'll convert it to Markdown!"
        if request.files:
            try:
                markdown_content = await process_files_to_string(request.files, query=request.query)
                logger.info(f"Successfully processed {len(request.files)} files with query: {request.query}")
            except Exception as e:
                logger.error(f"Error processing files: {str(e)}")
                raise e

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=markdown_content,
            data={"request_id": request.request_id}
        )

        return AgentResponse(success=True, markdown=markdown_content)

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False, markdown="", error=str(e))

@app.post("/api/convert-to-markdown", response_model=MarkdownResponse)
async def convert_to_markdown(
    request: FileRequest,
    authenticated: bool = Depends(verify_token)
):
    """Convert a single file to markdown format."""
    try:
        logger.info(f"Processing file: {request.file['name']}")
        
        # Save base64 content to a temporary file
        decoded_content = base64.b64decode(request.file['base64'])
        
        # Detect if the content is an image using imghdr
        content_stream = io.BytesIO(decoded_content)
        image_type = imghdr.what(content_stream)
        is_image = image_type is not None
        
        temp_file_path = f"/tmp/temp_file_{request.file['name']}"
        try:
            with open(temp_file_path, "wb") as f:
                f.write(decoded_content)
            logger.info(f"Saved content to temporary file: {temp_file_path}")
            
            if is_image:
                logger.info(f"Detected image type: {image_type}, using vision model: {os.getenv('OPENROUTER_VLM_MODEL')}")
                try:
                    # Create a new MarkItDown instance with the vision model
                    temp_md = MarkItDown(
                        llm_client=openai_client,
                        llm_model=os.getenv("OPENROUTER_VLM_MODEL")
                    )
                    result = temp_md.convert(temp_file_path, use_llm=True)
                    if not result.text_content:
                        raise Exception("Vision model returned empty response")
                    logger.info("Successfully used vision model")
                except Exception as vision_error:
                    if "401" in str(vision_error):
                        logger.error(f"Vision model access unauthorized: {str(vision_error)}")
                        return MarkdownResponse(
                            success=False,
                            error=f"API key does not have access to vision model {os.getenv('OPENROUTER_VLM_MODEL')}"
                        )
                    logger.error(f"Vision model error: {str(vision_error)}")
                    return MarkdownResponse(
                        success=False,
                        error=f"Error using vision model: {str(vision_error)}"
                    )
            else:
                # Use default model for non-image files
                temp_md = MarkItDown(
                    llm_client=openai_client,
                    llm_model=os.getenv("OPENROUTER_MODEL")
                )
                result = temp_md.convert(temp_file_path, use_llm=True)
            
            markdown_content = result.text_content
            if not markdown_content:
                raise Exception("No markdown content generated")
            
            logger.info(f"Successfully converted file. Output length: {len(markdown_content)}")
            
            # Clean up temporary file
            os.remove(temp_file_path)
            logger.info("Cleaned up temporary file")
            
            return MarkdownResponse(
                success=True,
                markdown=markdown_content
            )
            
        except Exception as e:
            logger.error(f"Error converting file: {str(e)}")
            # Fallback to direct text conversion if markdown conversion fails
            try:
                text_content = decoded_content.decode('utf-8')
                logger.info("Fallback: Using direct text conversion")
                return MarkdownResponse(
                    success=True,
                    markdown=text_content
                )
            except:
                error_msg = f"Failed to process file {request.file['name']}: {str(e)}"
                logger.error(error_msg)
                return MarkdownResponse(
                    success=False,
                    error=error_msg
                )
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        return MarkdownResponse(
            success=False,
            error=error_msg
        )

@app.post("/api/file-agent-cached")
async def process_files_cached(
    request: Request,
    query: str = "",
    files: List[Dict[str, Any]] = [],
    session_id: str = "",
    user_id: str = "",
    request_id: str = "",
    use_cache: bool = True,
    authenticated: bool = Depends(verify_token)
):
    """Process files with AI agent, utilizing cached markdown when available."""
    try:
        # Validate input
        if not files:
            return {
                "success": False,
                "error": "No files provided",
                "markdown": ""
            }

        # Process each file
        results = []
        for file_data in files:
            try:
                # Extract file info
                name = file_data.get('name', '')
                file_type = file_data.get('type', '')
                base64_content = file_data.get('base64', '')
                model = file_data.get('model', os.getenv("OPENROUTER_MODEL"))
                
                if not all([name, file_type, base64_content]):
                    continue

                # Process the file
                result = await process_file_cached(
                    name=name,
                    file_type=file_type,
                    base64_content=base64_content,
                    model=model,
                    use_cache=use_cache
                )
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing file {name}: {str(e)}")
                continue
        
        # Handle case where no files were successfully processed
        if not results:
            return {
                "success": False,
                "error": "No valid files were processed",
                "markdown": ""
            }
        
        # Store conversation messages
        try:
            await store_message(
                session_id=session_id,
                message_type="user",
                content=query if query else "Process files",
                data={"files": [f["name"] for f in files]}
            )
            
            await store_message(
                session_id=session_id,
                message_type="assistant",
                content="\n\n".join(results),
                data={"files": [f["name"] for f in files]}
            )
        except Exception as e:
            logger.error(f"Error storing messages: {str(e)}")
            # Continue even if message storage fails
        
        return {
            "success": True,
            "markdown": "\n\n".join(results)
        }
        
    except Exception as e:
        logger.error(f"Error in process_files_cached: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "markdown": ""
        }

if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
