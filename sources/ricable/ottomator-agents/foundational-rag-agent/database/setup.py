"""
Database setup and connection utilities for Supabase with pgvector.
"""
import os
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client, Client

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

class SupabaseClient:
    """
    Client for interacting with Supabase and pgvector.
    
    Args:
        supabase_url: URL for Supabase instance. Defaults to SUPABASE_URL env var.
        supabase_key: API key for Supabase. Defaults to SUPABASE_KEY env var.
    """
    
    def __init__(
        self, 
        supabase_url: Optional[str] = None, 
        supabase_key: Optional[str] = None
    ):
        """
        Initialize the Supabase client.
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and key must be provided either as arguments or environment variables."
            )
        
        self.client = create_client(self.supabase_url, self.supabase_key)
    
    def store_document_chunk(
        self, 
        url: str, 
        chunk_number: int, 
        content: str, 
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store a document chunk with its embedding in Supabase.
        
        Args:
            url: Source URL or identifier for the document
            chunk_number: Chunk number within the document
            content: Text content of the chunk
            embedding: Vector embedding of the chunk
            metadata: Additional metadata about the chunk
            
        Returns:
            Dictionary containing the inserted record
        """
        if metadata is None:
            metadata = {}
            
        data = {
            "url": url,
            "chunk_number": chunk_number,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }
        
        result = self.client.table("rag_pages").insert(data).execute()
        return result.data[0] if result.data else {}
    
    def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for document chunks by vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of matching document chunks with similarity scores
        """
        # Prepare parameters for the RPC call
        params = {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
        
        # Add filter if provided
        if filter_metadata:
            params["filter"] = filter_metadata
        
        # Call the match_rag_pages function
        result = self.client.rpc("match_rag_pages", params).execute()
        return result.data if result.data else []
    
    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        """
        Get a document chunk by its ID.
        
        Args:
            doc_id: ID of the document chunk
            
        Returns:
            Document chunk data
        """
        result = self.client.table("rag_pages").select("*").eq("id", doc_id).execute()
        return result.data[0] if result.data else {}
    
    def get_all_document_sources(self) -> List[str]:
        """
        Get a list of all unique document sources.
        
        Returns:
            List of unique source URLs/identifiers
        """
        result = self.client.table("rag_pages").select("url").execute()
        urls = set(item["url"] for item in result.data if result.data)
        return list(urls)
        
    def count_documents(self) -> int:
        """
        Count the total number of unique documents in the database.
        
        Returns:
            Number of unique documents (based on unique URLs)
        """
        return len(self.get_all_document_sources())


def setup_database_tables() -> None:
    """
    Set up the necessary database tables and functions for the RAG system.
    This should be run once to initialize the database.
    
    Note: This is typically done through the Supabase MCP server in production.
    """
    # This is a placeholder for the actual implementation
    # In a real application, you would use the Supabase MCP server to run the SQL
    pass
