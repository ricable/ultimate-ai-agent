# File Processing Agent for Live Agent Studio

Author: [Loic Baconnier](https://deeplearning.fr/)

This is a specialized Python FastAPI agent that demonstrates how to handle file uploads in the Live Agent Studio. It shows how to process, store, and leverage file content in conversations with AI models.

This agent builds upon the foundation laid out in [`~sample-python-agent~/sample_supabase_agent.py`](../~sample-python-agent~/sample_supabase_agent.py), extending it with file handling capabilities.

Not all agents need file handling which is why the sample Python agent is kept simple and this one is available to help you build agents with file handling capabilities. The Live Agent Studio has file uploading built in and the files are sent in the exact format shown in this agent.

## Overview

This agent extends the base Python agent template to showcase file handling capabilities:
- Process uploaded files in base64 format
- Store file content with conversation history
- Integrate file content into AI model context
- Maintain conversation continuity with file references
- Handle multiple files in a single conversation

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Supabase account (for conversation storage)
- OpenRouter API key (for LLM access)
- Basic understanding of:
  - FastAPI and async Python
  - Base64 encoding/decoding
  - OpenRouter API
  - Supabase

## Core Components

### 1. File Processing

The agent includes robust file handling:
- Base64 decoding of uploaded files
- Text extraction and formatting using MarkItDown
- Persistent storage of file data in Supabase
- Document caching for faster subsequent queries

### 2. Conversation Management

Built on the sample Supabase agent template, this agent adds:
- File metadata storage with messages
- File content integration in conversation history
- Contextual file reference handling
- Document caching for improved performance

### 3. AI Integration

Seamless integration with OpenRouter's LLM models:
- File content as conversation context
- Maintained context across multiple messages
- Intelligent responses based on file content
- Efficient caching of document conversions

## API Routes

### 1. File to Markdown Conversion
```bash
POST /api/convert-to-markdown
```
Converts a single file to markdown format. Simple and direct conversion.

Request:
```json
{
    "file": {
        "name": "document.pdf",
        "type": "application/pdf",
        "base64": "base64_encoded_content"
    }
}
```

Response:
```json
{
    "success": true,
    "markdown": "# Converted Content\n\nYour markdown content here...",
    "error": ""
}
```

### 2. AI Agent Processing
```bash
POST /api/file-agent
```
Processes files with an AI agent, providing enhanced responses and analysis. Supports both text documents and images.

Request:
```json
{
    "query": "Please analyze this document and summarize key points",
    "files": [{
        "name": "document.pdf",
        "type": "application/pdf",
        "base64": "base64_encoded_content"
    }],
    "session_id": "unique_session_id",
    "user_id": "user_123",
    "request_id": "request_456"
}
```

Response:
```json
{
    "success": true,
    "markdown": "Analysis and summary in markdown format...",
    "error": ""
}
```

### 3. Cached File Agent
```bash
POST /api/file-agent-cached
```
Same as `/api/file-agent` but with document caching for improved performance.

### Image Processing

The agent now supports image processing capabilities:
- Automatic image type detection using `imghdr`
- Integration with OpenRouter's Vision Language Models (VLM)
- Image description and analysis in markdown format
- Contextual understanding of images based on queries

For image processing to work, make sure to set the `OPENROUTER_VLM_MODEL` environment variable to a compatible vision model (e.g., `meta-llama/llama-3.2-11b-vision-instruct`).

Example image processing request:
```json
{
    "query": "Describe this image",
    "files": [{
        "name": "photo.jpg",
        "type": "image/jpeg",
        "base64": "base64_encoded_content"
    }],
    "session_id": "unique_session_id",
    "user_id": "user_123",
    "request_id": "request_456"
}
```

| Field | Type | Description |
|-------|------|-------------|
| query | string | Instructions for the AI agent |
| files | array | List of files to process |
| files[].name | string | Original filename with extension |
| files[].type | string | MIME type of the file |
| files[].base64 | string | Base64-encoded file content |
| session_id | string | Unique session identifier for conversation context |
| user_id | string | User identifier |
| request_id | string | Unique request identifier |

Response:
```json
{
    "success": true,
    "markdown": "Image description and analysis in markdown format..."
}
```

Features:
- Stores converted markdown in Supabase for reuse
- Faster responses for subsequent queries on the same document
- Updates last_accessed timestamp for cache management
- Option to bypass cache with use_cache=false

Example curl:
```bash
curl -X POST http://localhost:8001/api/file-agent-cached \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main points?",
    "files": [{
      "name": "document.pdf",
      "type": "application/pdf",
      "base64": "'$(base64 -i document.pdf)'"
    }],
    "session_id": "session_123",
    "user_id": "user_456",
    "request_id": "req_789",
    "use_cache": true
  }'
```

## Detailed API Documentation

### Authentication
All endpoints require Bearer token authentication:
```http
Authorization: Bearer your_token_here
```

### 1. File to Markdown Conversion
```http
POST /api/convert-to-markdown
```

#### Request Body
```json
{
    "file": {
        "name": "example.pdf",
        "type": "application/pdf",
        "base64": "base64_encoded_content"
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| file.name | string | Original filename with extension |
| file.type | string | MIME type of the file |
| file.base64 | string | Base64-encoded file content |

#### Response
```json
{
    "success": true,
    "markdown": "# Converted Content\n\nMarkdown content here...",
    "error": ""
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether the conversion was successful |
| markdown | string | Converted markdown content |
| error | string | Error message if conversion failed |

#### Supported File Types
- Documents: `.pdf`, `.docx`, `.txt`
- Spreadsheets: `.xlsx`, `.csv`
- Presentations: `.pptx`
- Web: `.html`
- Images: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`

### 2. AI Agent File Processing
```http
POST /api/file-agent
```

#### Request Body
```json
{
    "query": "Please analyze this document and extract key information",
    "files": [{
        "name": "document.pdf",
        "type": "application/pdf",
        "base64": "base64_encoded_content"
    }],
    "session_id": "session_123",
    "user_id": "user_456",
    "request_id": "req_789"
}
```

| Field | Type | Description |
|-------|------|-------------|
| query | string | Instructions for the AI agent |
| files | array | List of files to process |
| files[].name | string | Original filename with extension |
| files[].type | string | MIME type of the file |
| files[].base64 | string | Base64-encoded file content |
| session_id | string | Unique session identifier for conversation context |
| user_id | string | User identifier |
| request_id | string | Unique request identifier |

#### Response
```json
{
    "success": true,
    "markdown": "# Analysis Results\n\nAI-generated analysis..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether the processing was successful |
| markdown | string | AI-generated response in markdown format |

#### Features
- Maintains conversation context using session_id
- Supports multiple files in a single request
- Converts files to markdown before AI processing
- Uses OpenRouter's LLM models for enhanced analysis
- Stores conversation history in Supabase

### Error Responses
Both endpoints return similar error structures:

```json
{
    "success": false,
    "error": "Detailed error message",
    "markdown": ""
}
```

Common HTTP Status Codes:
- 200: Success
- 400: Bad Request (invalid input)
- 401: Unauthorized (invalid token)
- 500: Internal Server Error

### Rate Limiting
- Maximum file size: 10MB per file
- Maximum files per request: 5
- Maximum requests per minute: 60

## Error Handling

The API implements robust error handling:

1. Missing Files
- Returns 422 Unprocessable Entity when no files are provided
- Clear error message indicating "No files provided"

2. File Processing Errors
- Graceful handling of conversion failures
- Detailed error messages for debugging
- Continues processing remaining files if one fails

3. Cache Management
- Handles missing cache entries gracefully
- Falls back to fresh conversion if cache retrieval fails
- Updates cache timestamps on successful access

4. Schema Validation
- Automatic validation of request parameters
- Clear error messages for missing required fields
- Type checking of input values

## Database Setup

The application requires two main tables in Supabase:

1. Messages Table
```sql
create table messages (
    id uuid default uuid_generate_v4() primary key,
    session_id text,
    message_type text,
    content text,
    data jsonb default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now())
);
```

2. Document Cache Table
```sql
create table document_cache (
    doc_hash text primary key,
    file_name text,
    file_type text,
    markdown_content text,
    created_at timestamp with time zone default timezone('utc'::text, now()),
    last_accessed timestamp with time zone default timezone('utc'::text, now())
);
```

## Environment Variables

Create a `.env` file with the following variables:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_default_model
OPENROUTER_VLM_MODEL=your_vision_model  # Required for image processing
API_BEARER_TOKEN=your_api_token
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key
```

## Docker Setup

Build and run the container:
```bash
docker build -t file-agent .
docker run -p 8001:8001 --env-file .env file-agent
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ottomarkdown.git
cd ottomarkdown
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Create required directories:
```bash
mkdir -p test_files markdown_results
```

6. Run the application:
```bash
uvicorn file_agent:app --reload --port 8001
```

## Testing

1. Add test files to `test_files/` directory

2. Run conversion tests:
```bash
python test_markdown.py
```

3. Test the API with curl:
```bash
# Test markdown conversion
curl -X POST http://localhost:8001/api/convert-to-markdown \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d @test_payload.json

# Test AI agent with caching
curl -X POST http://localhost:8001/api/file-agent-cached \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

## Running the Agent

Start the agent with:
```bash
python file_agent.py
```

The agent will be available at `http://localhost:8001`.

## API Usage

Send requests to `/api/file-agent` with:
- `query`: Your question or prompt
- `files`: Array of file objects with:
  - `name`: Filename
  - `type`: MIME type
  - `base64`: Base64-encoded file content

Example request:
```json
{
  "query": "What does this file contain?",
  "files": [{
    "name": "example.txt",
    "type": "text/plain",
    "base64": "VGhpcyBpcyBhIHRlc3QgZmlsZS4="
  }],
  "session_id": "unique-session-id",
  "user_id": "user-id",
  "request_id": "request-id"
}
```

## Curl Examples

### Quick Reference

```bash
# 1. Convert a file to markdown
curl -X POST http://localhost:8001/api/convert-to-markdown \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "file": {
      "name": "document.pdf",
      "type": "application/pdf",
      "base64": "'$(base64 -i document.pdf)'"
    }
  }'

# 2. Process with AI agent
curl -X POST http://localhost:8001/api/file-agent \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Please analyze this document",
    "files": [{
      "name": "document.pdf",
      "type": "application/pdf",
      "base64": "'$(base64 -i document.pdf)'"
    }],
    "session_id": "session_123",
    "user_id": "user_456",
    "request_id": "req_789"
  }'
```

### Helper Script
Save this as `call_api.sh`:

```bash
#!/bin/bash

TOKEN="your_token_here"
API_URL="http://localhost:8001"

# Function to convert file to markdown
convert_to_markdown() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    local mime_type=$(file --mime-type -b "$file_path")
    
    curl -X POST "$API_URL/api/convert-to-markdown" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "file": {
                "name": "'$file_name'",
                "type": "'$mime_type'",
                "base64": "'$(base64 -i "$file_path")'"
            }
        }'
}

# Function to process with AI agent
process_with_agent() {
    local file_path=$1
    local query=$2
    local file_name=$(basename "$file_path")
    local mime_type=$(file --mime-type -b "$file_path")
    
    curl -X POST "$API_URL/api/file-agent" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "'$query'",
            "files": [{
                "name": "'$file_name'",
                "type": "'$mime_type'",
                "base64": "'$(base64 -i "$file_path")'"
            }],
            "session_id": "session_'$(date +%s)'",
            "user_id": "user_test",
            "request_id": "req_'$(date +%s)'"
        }'
}

# Usage examples:
# ./call_api.sh convert document.pdf
# ./call_api.sh process document.pdf "Summarize this document"

case "$1" in
    "convert")
        convert_to_markdown "$2"
        ;;
    "process")
        process_with_agent "$2" "$3"
        ;;
    *)
        echo "Usage:"
        echo "  Convert to markdown: $0 convert <file_path>"
        echo "  Process with agent: $0 process <file_path> \"query\""
        exit 1
        ;;
esac
```

### Example Usage

1. Make the script executable:
```bash
chmod +x call_api.sh
```

2. Convert a file to markdown:
```bash
./call_api.sh convert path/to/document.pdf
```

3. Process a file with AI:
```bash
./call_api.sh process path/to/document.pdf "Analyze this document and summarize key points"
```

4. Process multiple files (raw curl):
```bash
curl -X POST http://localhost:8001/api/file-agent \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare these documents",
    "files": [
      {
        "name": "document1.pdf",
        "type": "application/pdf",
        "base64": "'$(base64 -i document1.pdf)'"
      },
      {
        "name": "document2.pdf",
        "type": "application/pdf",
        "base64": "'$(base64 -i document2.pdf)'"
      }
    ],
    "session_id": "session_123",
    "user_id": "user_456",
    "request_id": "req_789"
  }'
```

5. Convert HTML (raw curl):
```bash
curl -X POST http://localhost:8001/api/convert-to-markdown \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "file": {
      "name": "page.html",
      "type": "text/html",
      "base64": "'$(base64 -i page.html)'"
    }
  }'

```

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.

