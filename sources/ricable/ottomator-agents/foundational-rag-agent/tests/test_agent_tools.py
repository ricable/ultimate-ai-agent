"""
Unit tests for the agent tools module.
"""
import os
import sys
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import KnowledgeBaseSearch, KnowledgeBaseSearchParams, KnowledgeBaseSearchResult


class TestKnowledgeBaseSearch:
    """
    Test cases for the KnowledgeBaseSearch class.
    """
    
    @pytest.mark.asyncio
    async def test_search_with_results(self):
        """
        Test that search returns results when documents are found.
        """
        # Mock the SupabaseClient and EmbeddingGenerator
        mock_supabase = MagicMock()
        mock_embedding_generator = MagicMock()
        
        # Set up mock return values
        mock_embedding = [0.1] * 1536  # Mock embedding vector
        mock_embedding_generator.embed_text.return_value = mock_embedding
        
        mock_search_results = [
            {
                "id": 1,
                "url": "local://test1.txt",
                "chunk_number": 0,
                "content": "This is test content 1.",
                "metadata": {
                    "source": "test1.txt",
                    "source_type": "txt"
                },
                "similarity": 0.95
            },
            {
                "id": 2,
                "url": "local://test2.txt",
                "chunk_number": 1,
                "content": "This is test content 2.",
                "metadata": {
                    "source": "test2.txt",
                    "source_type": "txt"
                },
                "similarity": 0.85
            }
        ]
        mock_supabase.search_documents.return_value = mock_search_results
        
        # Create the KnowledgeBaseSearch instance with mocks
        kb_search = KnowledgeBaseSearch(
            supabase_client=mock_supabase,
            embedding_generator=mock_embedding_generator
        )
        
        # Create search parameters
        params = KnowledgeBaseSearchParams(
            query="test query",
            max_results=2
        )
        
        # Call the search method
        results = await kb_search.search(params)
        
        # Check that the mocks were called correctly
        mock_embedding_generator.embed_text.assert_called_once_with("test query")
        mock_supabase.search_documents.assert_called_once_with(
            query_embedding=mock_embedding,
            match_count=2,
            filter_metadata=None
        )
        
        # Check the results
        assert len(results) == 2
        assert isinstance(results[0], KnowledgeBaseSearchResult)
        assert results[0].content == "This is test content 1."
        assert results[0].source == "test1.txt"
        assert results[0].source_type == "txt"
        assert results[0].similarity == 0.95
        
        assert results[1].content == "This is test content 2."
        assert results[1].source == "test2.txt"
        assert results[1].source_type == "txt"
        assert results[1].similarity == 0.85
    
    @pytest.mark.asyncio
    async def test_search_with_source_filter(self):
        """
        Test that search applies source filter correctly.
        """
        # Mock the SupabaseClient and EmbeddingGenerator
        mock_supabase = MagicMock()
        mock_embedding_generator = MagicMock()
        
        # Set up mock return values
        mock_embedding = [0.1] * 1536  # Mock embedding vector
        mock_embedding_generator.embed_text.return_value = mock_embedding
        
        mock_search_results = [
            {
                "id": 1,
                "url": "local://test1.txt",
                "chunk_number": 0,
                "content": "This is test content 1.",
                "metadata": {
                    "source": "test1.txt",
                    "source_type": "txt"
                },
                "similarity": 0.95
            }
        ]
        mock_supabase.search_documents.return_value = mock_search_results
        
        # Create the KnowledgeBaseSearch instance with mocks
        kb_search = KnowledgeBaseSearch(
            supabase_client=mock_supabase,
            embedding_generator=mock_embedding_generator
        )
        
        # Create search parameters with source filter
        params = KnowledgeBaseSearchParams(
            query="test query",
            max_results=5,
            source_filter="test1.txt"
        )
        
        # Call the search method
        results = await kb_search.search(params)
        
        # Check that the mocks were called correctly
        mock_embedding_generator.embed_text.assert_called_once_with("test query")
        mock_supabase.search_documents.assert_called_once_with(
            query_embedding=mock_embedding,
            match_count=5,
            filter_metadata={"source": "test1.txt"}
        )
        
        # Check the results
        assert len(results) == 1
        assert results[0].source == "test1.txt"
    
    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """
        Test that search returns empty list when no documents are found.
        """
        # Mock the SupabaseClient and EmbeddingGenerator
        mock_supabase = MagicMock()
        mock_embedding_generator = MagicMock()
        
        # Set up mock return values
        mock_embedding = [0.1] * 1536  # Mock embedding vector
        mock_embedding_generator.embed_text.return_value = mock_embedding
        
        # Return empty results
        mock_supabase.search_documents.return_value = []
        
        # Create the KnowledgeBaseSearch instance with mocks
        kb_search = KnowledgeBaseSearch(
            supabase_client=mock_supabase,
            embedding_generator=mock_embedding_generator
        )
        
        # Create search parameters
        params = KnowledgeBaseSearchParams(
            query="test query",
            max_results=5
        )
        
        # Call the search method
        results = await kb_search.search(params)
        
        # Check the results
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_get_available_sources(self):
        """
        Test that get_available_sources returns the list of sources.
        """
        # Mock the SupabaseClient
        mock_supabase = MagicMock()
        
        # Set up mock return values
        mock_sources = ["test1.txt", "test2.pdf", "test3.txt"]
        mock_supabase.get_all_document_sources.return_value = mock_sources
        
        # Create the KnowledgeBaseSearch instance with mock
        kb_search = KnowledgeBaseSearch(supabase_client=mock_supabase)
        
        # Call the get_available_sources method
        sources = await kb_search.get_available_sources()
        
        # Check that the mock was called correctly
        mock_supabase.get_all_document_sources.assert_called_once()
        
        # Check the results
        assert sources == mock_sources
