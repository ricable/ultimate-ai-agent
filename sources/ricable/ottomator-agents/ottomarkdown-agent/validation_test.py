import os
from openai import OpenAI
import json
from dotenv import load_dotenv, find_dotenv
import base64
from markitdown import MarkItDown
from pathlib import Path
import io
import requests
import mimetypes
import time
import re
import httpx

# Load environment variables from .env file
print("Loading .env file from:", find_dotenv())
load_dotenv(find_dotenv())

# Set environment variables
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENROUTER_MODEL"] = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
os.environ["OPENROUTER_VLM_MODEL"] = os.getenv("OPENROUTER_VLM_MODEL", "meta-llama/llama-3.2-11b-vision-instruct:free")

# Print loaded environment variables
print("\nEnvironment variables loaded:")
print(f"OPENROUTER_API_KEY = {os.getenv('OPENROUTER_API_KEY')}")
print(f"OPENROUTER_MODEL = {os.getenv('OPENROUTER_MODEL')}")
print(f"OPENROUTER_VLM_MODEL = {os.getenv('OPENROUTER_VLM_MODEL')}")

# Set up API URL with protocol
API_URL = "http://" + os.getenv("API_URL", "localhost:8001")

# Get API token
API_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_TOKEN:
    raise ValueError("API_BEARER_TOKEN not set in environment")

# Common headers
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Get and clean environment variables
api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
model = os.getenv("OPENROUTER_MODEL", "").strip()

print("\nEnvironment variables loaded:")
print("OPENROUTER_API_KEY =", api_key)
print("OPENROUTER_MODEL =", model)

# Remove any duplicated API key
if len(api_key) > 100:  # API keys are typically around 73 characters
    api_key = api_key[:73]  # Take only the first part

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
if not model:
    raise ValueError("OPENROUTER_MODEL not found in environment variables")

# Validate API key format
if not api_key.startswith("sk-or-v1-"):
    raise ValueError("Invalid API key format. Should start with 'sk-or-v1-'")

def ensure_dir(directory):
    """Ensure a directory exists, create it if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def test_openrouter_api():
    """Test OpenRouter API connection with Llama vision model."""
    print("\nStarting OpenRouter API test...")
    
    # Load environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_VLM_MODEL")
    
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables")
        return
        
    print("Using API key:", api_key)
    print("Using VLM model:", model)
    
    try:
        # Initialize OpenRouter client
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "http://localhost:8001",
                "X-Title": "MarkItDown Test",
            }
        )
        
        print("\nSending request to OpenRouter API...")
        
        # Simple test prompt
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello and confirm you can process images!"}
            ]
        )
        
        print("\nRaw API Response:")
        print(json.dumps(response.model_dump(), indent=2))
        
        print("\nProcessed Response:")
        print("Content:", response.choices[0].message.content)
        print("Model used:", response.model)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print("\nError occurred during API test:")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

def test_file_processing():
    """Test file processing capabilities without LLM calls"""
    print("\nStarting file processing test...")
    
    # Initialize OpenRouter client with VLM model for images
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "MarkItDown Test",
        }
    )
    
    # Initialize MarkItDown with VLM model for images
    md = MarkItDown(
        llm_client=openai_client,
        llm_model=os.getenv("OPENROUTER_VLM_MODEL")
    )
    
    # Test directory path
    test_dir = Path(__file__).parent / "test_files"
    results_dir = Path(__file__).parent / "markdown_results"
    ensure_dir(results_dir)
    
    for file_path in test_dir.glob("*"):
        try:
            print(f"\nProcessing {file_path.name}...")
            result = md.convert(str(file_path))
            
            if result and hasattr(result, 'text_content'):
                print(f"✓ Successfully processed {file_path.name}")
                
                # Create markdown file name
                file_type = file_path.suffix.lower()[1:]
                base_name = file_path.stem
                md_file = results_dir / f"local_convert_{base_name}_{file_type}.md"
                
                # Save markdown content for all files
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(result.text_content)
                
                print(f"  Output length: {len(result.text_content)} characters")
                print(f"  Saved markdown to: {md_file}")
            else:
                print(f"✗ Failed to process {file_path.name}")
                print(f"  Result: {result}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path.name}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")

def test_file_processing_with_llm():
    """Test processing all files with LLM"""
    print("\nTesting file processing with LLM...")
    
    # Initialize OpenRouter clients
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "MarkItDown Test",
        }
    )

    # Initialize MarkItDown with appropriate models
    md_text = MarkItDown(
        llm_client=openai_client,
        llm_model=os.getenv("OPENROUTER_MODEL")
    )
    md_vlm = MarkItDown(
        llm_client=openai_client,
        llm_model=os.getenv("OPENROUTER_VLM_MODEL")
    )
    
    test_dir = Path(__file__).parent / "test_files"
    
    for file_path in test_dir.glob("*"):
        if file_path.name.startswith('.'):
            continue
            
        try:
            print(f"\nProcessing {file_path.name}...")
            
            # Use appropriate model and always set use_llm=True
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                result = md_vlm.convert(str(file_path), use_llm=True)
            else:
                result = md_text.convert(str(file_path), use_llm=True)
            
            # Save results
            if result and hasattr(result, 'text_content'):
                output_path = f'markdown_results/api_openrouter_{file_path.stem}_{file_path.suffix[1:]}.md'
                with open(output_path, 'w') as f:
                    if hasattr(result, 'title') and result.title:
                        f.write(f"# {result.title}\n\n")
                    f.write(result.text_content)
                
                print(f"Successfully processed {file_path.name}")
                print(f"Output saved to: {output_path}")
                print(f"First 100 characters: {result.text_content[:100]}...")
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

def test_image_processing_with_llm():
    """Test processing image files with LLM integration for descriptions."""
    print("\nTesting image processing with LLM...")
    
    # Get all image files from test_files directory
    test_dir = Path(__file__).parent / "test_files"
    image_files = [f for f in test_dir.glob("*") if f.is_file() 
                   and not f.name.startswith('.') 
                   and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']]
    
    # Initialize OpenRouter client with VLM model for images
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "MarkItDown Test",
        }
    )

    # Initialize MarkItDown with VLM model
    md = MarkItDown(
        llm_client=openai_client,
        llm_model=os.getenv("OPENROUTER_VLM_MODEL")
    )
    
    for image_path in image_files:
        try:
            print(f"\nProcessing image: {image_path.name}")
            result = md.convert(str(image_path), use_llm=True)
            
            # Use consistent naming pattern matching other test functions
            output_path = f'markdown_results/api_openrouter_vision_{image_path.stem}_{image_path.suffix[1:]}.md'
            with open(output_path, 'w') as f:
                f.write(result.text_content)
                
            print(f"Successfully processed {image_path.name}")
            print(f"Output saved to: {output_path}")
            print(f"Generated description: {result.text_content[:200]}...")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(f"Error type: {type(e).__name__}")

def test_file_agent_openrouter():
    """Test file agent with OpenRouter LLM using query on markdown output"""
    print("\nTesting file agent with OpenRouter...")
    
    # Initialize OpenRouter client
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "MarkItDown Test",
        }
    )
    
    # Initialize MarkItDown with appropriate models
    md_text = MarkItDown(
        llm_client=openai_client,
        llm_model=os.getenv("OPENROUTER_MODEL")
    )
    md_vlm = MarkItDown(
        llm_client=openai_client,
        llm_model=os.getenv("OPENROUTER_VLM_MODEL")
    )
    
    test_dir = Path(__file__).parent / "test_files"
    results_dir = Path(__file__).parent / "markdown_results"
    ensure_dir(results_dir)
    
    for file_path in test_dir.glob("*"):
        if file_path.name.startswith('.'):
            continue
            
        try:
            print(f"\nProcessing {file_path.name}...")
            
            # First get markdown content using appropriate model
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                markdown_result = md_vlm.convert(str(file_path), use_llm=True)
            else:
                markdown_result = md_text.convert(str(file_path), use_llm=True)
            
            if markdown_result and hasattr(markdown_result, 'text_content'):
                # Now process the markdown with LLM query for summary
                query = "Give me a concise summary of this content in 3-4 sentences."
                response = openai_client.chat.completions.create(
                    model=os.getenv("OPENROUTER_MODEL"),
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear and concise summaries."},
                        {"role": "user", "content": f"{markdown_result.text_content}\n\n{query}"}
                    ]
                )
                
                # Save results
                output_path = results_dir / f"agent_openrouter_summary_{file_path.stem}_{file_path.suffix[1:]}.md"
                with open(output_path, 'w', encoding='utf-8') as f:
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                        f.write(f"![{file_path.stem}](../test_files/{file_path.name})\n\n")
                    f.write("# Original Content\n\n")
                    f.write(markdown_result.text_content)
                    f.write("\n\n# Summary\n\n")
                    f.write(response.choices[0].message.content)
                
                print(f"✓ Successfully processed {file_path.name}")
                print(f"  Saved to: {output_path}")
                
        except Exception as e:
            print(f"✗ Error processing {file_path.name}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")

# def test_api_file_agent_cached():
#     print("\nTesting /api/file-agent-cached endpoint...")
    
#     # Get list of test files
#     test_files = os.listdir('test_files')
#     print(f"Processing {len(test_files)} files:")
#     for f in test_files:
#         print(f"- {f}")
#     print()
    
#     # Process one file at a time to avoid timeouts
#     batch_size = 1
#     max_retries = 5
#     content_size_limit = 100000
    
#     for i in range(0, len(test_files), batch_size):
#         batch = test_files[i:i + batch_size]
#         files_data = []
        
#         for file_name in batch:
#             file_path = os.path.join('test_files', file_name)
#             if os.path.getsize(file_path) > content_size_limit:
#                 print(f"⚠️ {file_name} exceeds size limit ({os.path.getsize(file_path)} bytes). Content will be truncated.")
            
#             try:
#                 with open(file_path, 'rb') as f:
#                     content = f.read(content_size_limit)
#                     base64_content = base64.b64encode(content).decode('utf-8')
#                     print(f"✓ Successfully encoded {file_name}")
#                     files_data.append({
#                         'name': file_name,
#                         'base64': base64_content,
#                         'type': os.path.splitext(file_name)[1][1:]
#                     })
#             except Exception as e:
#                 print(f"✗ Failed to encode {file_name}: {str(e)}")
#                 continue
        
#         if not files_data:
#             continue
            
#         # Try the request with retries and exponential backoff
#         success = False
#         for retry in range(max_retries):
#             try:
#                 response = requests.post(
#                     'http://localhost:8001/api/file-agent',
#                     params={
#                         'query': 'summarize',
#                         'session_id': 'test_session_123',
#                         'user_id': 'test_user_123',
#                         'request_id': 'test_request_123',
#                         'use_cache': 'true'
#                     },
#                     json=files_data,
#                     headers={'Authorization': f'Bearer {os.getenv("API_BEARER_TOKEN")}'}
#                 )
                
#                 print(f"\nStatus Code: {response.status_code}")
                
#                 if response.status_code == 200:
#                     success = True
#                     break
#                 else:
#                     print(f"Attempt {retry + 1} failed. Status: {response.status_code}")
#                     if retry < max_retries - 1:
#                         delay = min(30, (2 ** retry) * 5)
#                         print(f"Waiting {delay} seconds before retry...")
#                         time.sleep(delay)
#             except Exception as e:
#                 print(f"Request failed: {str(e)}")
#                 if retry < max_retries - 1:
#                     delay = min(30, (2 ** retry) * 5)
#                     print(f"Waiting {delay} seconds before retry...")
#                     time.sleep(delay)
#     # ... rest of the function ...

async def test_file_agent():
    """Test the /api/file-agent endpoint with various file types."""
    print("\nStarting file processing test...\n")
    
    # Process each file type
    test_files = os.listdir('test_files')
    for file in test_files:
        if file.startswith('.'):
            print(f"Skipping hidden file: {file}")
            continue
            
        print(f"\nProcessing {file}...")
        
        # Read file content
        file_path = os.path.join('test_files', file)
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        # Convert to base64
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        # Create request data
        request_data = {
            "query": "Please process this file and provide insights",
            "user_id": "test_user",
            "request_id": "test_request",
            "session_id": "test_session",
            "files": [{
                "name": file,
                "base64": encoded_content,
                "type": os.path.splitext(file)[1][1:]  # Get file extension without dot
            }]
        }
        
        # First get markdown conversion
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/file-agent",
                headers={
                    "Authorization": f"Bearer {os.getenv('API_BEARER_TOKEN')}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            
        if response.status_code == 200:
            print(f"✓ Successfully processed {file}")
            result = response.json()
            markdown_content = result.get("markdown", "")
            
            # Save markdown content with original filename
            base_name = os.path.splitext(file)[0]
            output_path = 'markdown_results/api_openrouter_connection_test.md'
            with open(output_path, 'w') as f:
                f.write(f"# OpenRouter API Connection Test\n\n")
                f.write(f"Model: {model}\n")
                f.write(f"Response: {response.choices[0].message.content}")
            markdown_content = result.get("markdown", "")
            
            # Save markdown content with original filename
            base_name = os.path.splitext(file)[0]
            output_file = os.path.join('markdown_results', f"{base_name}.md")
            with open(output_file, 'w') as f:
                f.write(markdown_content)
                    
            print(f"  Output length: {len(markdown_content)} characters")
            print(f"  First 100 characters: {markdown_content[:100]}...")
            print(f"  Saved to: {output_file}")
        else:
            print(f"✗ Failed to process {file}")
            print(f"  Status: {response.status_code}")
            try:
                error_json = response.json()
                if isinstance(error_json, dict):
                    print(f"  Error details: {json.dumps(error_json, indent=2)}")
                    if 'detail' in error_json:
                        print(f"  Validation error: {error_json['detail']}")
                else:
                    print(f"  Error response: {error_json}")
            except json.JSONDecodeError:
                print(f"  Raw response: {response.text}")
            except Exception as e:
                print(f"  Error parsing response: {str(e)}")
                print(f"  Raw response: {response.text}")
            
async def test_convert_to_markdown():
    """Test the /api/convert-to-markdown endpoint with all test files."""
    print("\nTesting /api/convert-to-markdown endpoint...")
    
    test_files = [f for f in os.listdir('test_files') if not f.startswith('.')]
    for file in test_files:
        print(f"\nConverting {file}...")
        
        # Read and encode file
        file_path = os.path.join('test_files', file)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        # Test conversion
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/convert-to-markdown",
                headers=HEADERS,
                json={
                    "file": {
                        "name": file,
                        "base64": encoded_content
                    }
                }
            )
        
        if response.status_code == 200:
            print(f"✓ Successfully converted {file}")
            result = response.json()
            markdown_content = result.get("markdown", "")
            
            print(f"  Output length: {len(markdown_content)} characters")
            if markdown_content:
                print(f"  First 100 characters: {markdown_content[:100]}...")
                
                # Save for verification
                clean_name = re.sub(r'[^\w\-_\.]', '_', file)
                base_name = os.path.splitext(clean_name)[0]
                markdown_path = os.path.join('markdown_results', f"{base_name}.md")
                os.makedirs('markdown_results', exist_ok=True)
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"  Saved to: {markdown_path}")
        else:
            print(f"✗ Failed to convert {file}")
            print(f"  Error: {response.text}")

async def test_file_agent_api():
    """Test the /api/file-agent endpoint with all test files."""
    print("\nTesting /api/file-agent endpoint...")
    
    test_files = [f for f in os.listdir('test_files') if not f.startswith('.')]
    for file in test_files:
        print(f"\nProcessing {file} with file-agent...")
        
        # Read and encode file
        file_path = os.path.join('test_files', file)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        # Test without query first
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/file-agent",
                headers=HEADERS,
                json={
                    "files": [{
                        "name": file,
                        "base64": encoded_content
                    }]
                }
            )
        
        if response.status_code == 200:
            print(f"✓ Successfully processed {file}")
            result = response.json()
            markdown_content = result.get("markdown", "")
            
            print(f"  Output length: {len(markdown_content)} characters")
            if markdown_content:
                print(f"  First 100 characters: {markdown_content[:100]}...")
                
                # Save markdown result
                clean_name = re.sub(r'[^\w\-_\.]', '_', file)
                base_name = os.path.splitext(clean_name)[0]
                markdown_path = os.path.join('markdown_results', f"{base_name}.md")
                os.makedirs('markdown_results', exist_ok=True)
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"  Saved to: {markdown_path}")
                
                # Now test with a query
                query = "Please provide a concise summary of the following content"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{API_URL}/api/file-agent",
                        headers=HEADERS,
                        params={"query": query},
                        json={
                            "files": [{
                                "name": f"{base_name}.md",
                                "base64": base64.b64encode(markdown_content.encode()).decode()
                            }]
                        }
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result.get("markdown", "")
                    
                    # Save summary result
                    summary_path = os.path.join('markdown_results', f"{base_name}_summary.md")
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    print(f"  Summary saved to: {summary_path}")
                else:
                    print(f"✗ Failed to get summary for {file}")
                    print(f"  Error: {response.text}")
        else:
            print(f"✗ Failed to process {file}")
            print(f"  Error: {response.text}")

async def test_file_agent_cached_api():
    """Test the /api/file-agent-cached endpoint."""
    print("\nTesting /api/file-agent-cached endpoint...")
    
    # Use a test file for cache testing
    test_file = next(f for f in os.listdir('test_files') if f.endswith('.docx'))
    print(f"Using {test_file} for cache testing...")
    
    # Read and encode file
    file_path = os.path.join('test_files', test_file)
    with open(file_path, 'rb') as f:
        file_content = f.read()
    encoded_content = base64.b64encode(file_content).decode('utf-8')
    
    files = [{
        "name": test_file,
        "base64": encoded_content
    }]
    
    # Test 1: First request (should cache)
    print("\nTest 1: First request (should cache)...")
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        response = await client.post(
            f"{API_URL}/api/file-agent-cached",
            headers=HEADERS,
            params={
                "query": "What are the main topics?",
                "session_id": "test_session_123",
                "user_id": "test_user_123",
                "request_id": "test_request_123",
                "use_cache": True
            },
            json={"files": files}
        )
    
    if response.status_code == 200:
        print(f"✓ First request successful")
        print(f"  Time taken: {time.time() - start_time:.2f}s")
        
        # Test 2: Second request (should use cache)
        print("\nTest 2: Second request (should use cache)...")
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            response = await client.post(
                f"{API_URL}/api/file-agent-cached",
                headers=HEADERS,
                params={
                    "query": "Give me a different perspective",
                    "session_id": "test_session_123",
                    "user_id": "test_user_123",
                    "request_id": "test_request_456",
                    "use_cache": True
                },
                json={"files": files}
            )
        
        if response.status_code == 200:
            print(f"✓ Second request successful")
            print(f"  Time taken: {time.time() - start_time:.2f}s")
        else:
            print(f"✗ Second request failed")
            print(f"  Error: {response.text}")
    else:
        print(f"✗ First request failed")
        print(f"  Error: {response.text}")

async def main():
    """Run all tests in sequence."""
    print("\nStarting tests...")
    
    # Run OpenRouter tests
    test_openrouter_api()
    test_file_processing()
    test_file_processing_with_llm()
    test_image_processing_with_llm()
    test_file_agent_openrouter()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
