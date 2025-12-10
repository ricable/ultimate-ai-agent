"""
Unit tests for the RAG agent module.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import RAGAgent
from agent.tools import KnowledgeBaseSearchResult


class TestRAGAgent:
    """
    Test cases for the RAGAgent class.
    """
    
    @pytest.mark.asyncio
    async def test_query_with_results(self):
        """
        Test that the agent query returns results when documents are found.
        """
        # Mock the Pydantic AI Agent and KnowledgeBaseSearch
        mock_agent = MagicMock()
        mock_kb_search = MagicMock()
        
        # Set up mock return values for the agent
        mock_result = MagicMock()
        mock_result.output = "This is the agent's response based on the knowledge base."
        mock_result.tool_calls = [MagicMock()]
        mock_result.tool_calls[0].tool.name = "search"
        
        # Create mock search results
        mock_search_results = [
            KnowledgeBaseSearchResult(
                content="This is test content 1.",
                source="test1.txt",
                source_type="txt",
                similarity=0.95,
                metadata={"source": "test1.txt", "source_type": "txt"}
            ),
            KnowledgeBaseSearchResult(
                content="This is test content 2.",
                source="test2.txt",
                source_type="txt",
                similarity=0.85,
                metadata={"source": "test2.txt", "source_type": "txt"}
            )
        ]
        mock_result.tool_calls[0].result = mock_search_results
        
        # Set the agent's run method to return the mock result
        mock_agent.run.return_value = mock_result
        
        # Create the RAGAgent with mocks
        with patch('agent.agent.Agent', return_value=mock_agent):
            rag_agent = RAGAgent(
                model="gpt-4.1-mini",
                api_key="test_api_key",
                kb_search=mock_kb_search
            )
            
            # Call the query method
            result = await rag_agent.query("What is the test content?")
            
            # Check that the agent was called correctly
            mock_agent.run.assert_called_once()
            
            # Check the result
            assert result["response"] == "This is the agent's response based on the knowledge base."
            assert len(result["kb_results"]) == 2
            assert result["kb_results"][0].content == "This is test content 1."
            assert result["kb_results"][1].content == "This is test content 2."
    
    @pytest.mark.asyncio
    async def test_query_no_kb_results(self):
        """
        Test that the agent query works when no knowledge base results are found.
        """
        # Mock the Pydantic AI Agent and KnowledgeBaseSearch
        mock_agent = MagicMock()
        mock_kb_search = MagicMock()
        
        # Set up mock return values for the agent
        mock_result = MagicMock()
        mock_result.output = "I don't have specific information about that in my knowledge base."
        mock_result.tool_calls = [MagicMock()]
        mock_result.tool_calls[0].tool.name = "search"
        mock_result.tool_calls[0].result = []  # Empty results
        
        # Set the agent's run method to return the mock result
        mock_agent.run.return_value = mock_result
        
        # Create the RAGAgent with mocks
        with patch('agent.agent.Agent', return_value=mock_agent):
            rag_agent = RAGAgent(
                model="gpt-4.1-mini",
                api_key="test_api_key",
                kb_search=mock_kb_search
            )
            
            # Call the query method
            result = await rag_agent.query("What is something not in the knowledge base?")
            
            # Check that the agent was called correctly
            mock_agent.run.assert_called_once()
            
            # Check the result
            assert result["response"] == "I don't have specific information about that in my knowledge base."
            assert len(result["kb_results"]) == 0
    
    @pytest.mark.asyncio
    async def test_get_available_sources(self):
        """
        Test that get_available_sources returns the list of sources.
        """
        # Mock the KnowledgeBaseSearch
        mock_kb_search = MagicMock()
        
        # Set up mock return values
        mock_sources = ["test1.txt", "test2.pdf", "test3.txt"]
        mock_kb_search.get_available_sources.return_value = mock_sources
        
        # Create the RAGAgent with mock
        with patch('agent.agent.Agent'):
            rag_agent = RAGAgent(
                model="gpt-4.1-mini",
                api_key="test_api_key",
                kb_search=mock_kb_search
            )
            
            # Call the get_available_sources method
            sources = await rag_agent.get_available_sources()
            
            # Check that the mock was called correctly
            mock_kb_search.get_available_sources.assert_called_once()
            
            # Check the results
            assert sources == mock_sources
