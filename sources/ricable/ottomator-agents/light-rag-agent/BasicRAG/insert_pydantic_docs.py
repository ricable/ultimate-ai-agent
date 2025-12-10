"""Script to insert Pydantic AI documentation from URL into ChromaDB."""

import argparse
import hashlib
import os
import re
from typing import List, Dict, Any, Tuple
import httpx

from utils import get_chroma_client, get_or_create_collection, add_documents_to_collection

# URL of the Pydantic AI documentation
PYDANTIC_DOCS_URL = "https://ai.pydantic.dev/llms.txt"

def fetch_pydantic_docs() -> str:
    """Fetch the Pydantic AI documentation from the URL.
    
    Returns:
        The content of the documentation
    """
    try:
        response = httpx.get(PYDANTIC_DOCS_URL)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Error fetching Pydantic AI documentation: {e}")

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split the text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: The size of each chunk
        overlap: The overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position for this chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text, try to find a good break point
        if end < len(text):
            # Look for a newline character to break at
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start + chunk_size // 2:  # Only use if it's not too close to the start
                end = newline_pos + 1  # Include the newline
        
        # Add the chunk to our list
        chunks.append(text[start:end])
        
        # Move the start position for the next chunk, considering overlap
        # Ensure we always make progress by moving at least 1 character forward
        start = max(end - overlap, start + 1)
        
        # Safety check to ensure we're making progress
        if start >= len(text):
            break
    
    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extract section information from a chunk.
    
    Args:
        chunk: The text chunk
        
    Returns:
        Dictionary with section information
    """
    # Try to extract section headers
    headers = re.findall(r'headers:\{type:MARKDOWN_NODE_TYPE_HEADER_\d+ text:"([^"]+)"\}', chunk)
    
    # Extract content type if available
    content_type_match = re.search(r'type:"([^"]+)"', chunk)
    content_type = content_type_match.group(1) if content_type_match else "unknown"
    
    return {
        "headers": "; ".join(headers) if headers else "",  
        "content_type": content_type,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def process_pydantic_docs(chunk_size: int = 1000, overlap: int = 200) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Process the Pydantic AI documentation.
    
    Args:
        chunk_size: The size of each chunk
        overlap: The overlap between chunks
        
    Returns:
        Tuple containing lists of IDs, documents, and metadatas
    """
    print("Fetching Pydantic AI documentation...")
    content = fetch_pydantic_docs()
    
    print(f"Splitting documentation into chunks (size: {chunk_size}, overlap: {overlap})...")
    chunks = split_into_chunks(content, chunk_size, overlap)
    
    ids = []
    documents = []
    metadatas = []
    
    print(f"Processing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        # Generate a unique ID for the chunk
        chunk_id = f"pydantic-docs-chunk-{i}"
        
        # Extract metadata
        metadata = extract_section_info(chunk)
        metadata["chunk_index"] = i
        metadata["source"] = PYDANTIC_DOCS_URL
        
        # Add to our lists
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(metadata)
        
        if i % 10 == 0:
            print(f"Processed {i} chunks...")
    
    return ids, documents, metadatas

def main():
    """Main function to parse arguments and insert Pydantic docs into ChromaDB."""
    parser = argparse.ArgumentParser(description="Insert Pydantic AI documentation into ChromaDB")
    parser.add_argument("--collection", default="pydantic_docs", help="Name of the ChromaDB collection")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory to store ChromaDB data")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                        help="Name of the embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of each text chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for adding documents")
    
    args = parser.parse_args()
    
    # Process Pydantic AI documentation
    print("Processing Pydantic AI documentation...")
    ids, documents, metadatas = process_pydantic_docs(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if not documents:
        print("No documents found to process.")
        return
    
    print(f"Found {len(documents)} chunks.")
    
    # Get ChromaDB client and collection
    print(f"Connecting to ChromaDB at {args.db_dir}...")
    client = get_chroma_client(args.db_dir)
    collection = get_or_create_collection(
        client, 
        args.collection,
        embedding_model_name=args.embedding_model
    )
    
    # Add documents to the collection
    print(f"Adding chunks to collection '{args.collection}'...")
    add_documents_to_collection(
        collection,
        ids,
        documents,
        metadatas,
        batch_size=args.batch_size
    )
    
    print(f"Successfully added {len(documents)} chunks to ChromaDB collection '{args.collection}'.")

if __name__ == "__main__":
    main()
