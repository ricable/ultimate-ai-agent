"""
Knowledge base search tool for the RAG AI agent.
"""
import os
import sys
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.setup import SupabaseClient
from document_processing.embeddings import EmbeddingGenerator

class KnowledgeBaseSearchParams(BaseModel):
    """
    Parameters for the knowledge base search tool.
    """
    query: str = Field(..., description="The search query to find relevant information in the knowledge base")
    max_results: int = Field(5, description="Maximum number of results to return (default: 5)")
    source_filter: Optional[str] = Field(None, description="Optional filter to search only within a specific source")


class KnowledgeBaseSearchResult(BaseModel):
    """
    Result from the knowledge base search.
    """
    content: str = Field(..., description="Content of the document chunk")
    source: str = Field(..., description="Source of the document chunk")
    source_type: str = Field(..., description="Type of source (e.g., 'pdf', 'txt')")
    similarity: float = Field(..., description="Similarity score between the query and the document")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the document")


class KnowledgeBaseSearch:
    """
    Tool for searching the knowledge base using vector similarity.
    """
    
    def __init__(
        self,
        supabase_client: Optional[SupabaseClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize the knowledge base search tool.
        
        Args:
            supabase_client: SupabaseClient instance for database operations
            embedding_generator: EmbeddingGenerator instance for creating embeddings
        """
        self.supabase_client = supabase_client or SupabaseClient()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
    
    async def search(self, params: KnowledgeBaseSearchParams) -> List[KnowledgeBaseSearchResult]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            params: Search parameters
            
        Returns:
            List of search results
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.embed_text(params.query)
        
        # Prepare filter metadata if source filter is provided
        filter_metadata = None
        if params.source_filter:
            filter_metadata = {"source": params.source_filter}
        
        # Search for documents
        results = self.supabase_client.search_documents(
            query_embedding=query_embedding,
            match_count=params.max_results,
            filter_metadata=filter_metadata
        )
        
        # Convert results to KnowledgeBaseSearchResult objects
        search_results = []
        for result in results:
            search_results.append(
                KnowledgeBaseSearchResult(
                    content=result["content"],
                    source=result["metadata"].get("source", "Unknown"),
                    source_type=result["metadata"].get("source_type", "Unknown"),
                    similarity=result["similarity"],
                    metadata=result["metadata"]
                )
            )
        
        return search_results
    
    async def get_available_sources(self) -> List[str]:
        """
        Get a list of all available sources in the knowledge base.
        
        Returns:
            List of source identifiers
        """
        return self.supabase_client.get_all_document_sources()
