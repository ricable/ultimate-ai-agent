"""
Unit tests for UI components.
"""

import pytest
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from src.ui.components.memory_demo import MemoryDemo
from src.ui.themes.safla_theme import SaflaTheme, get_custom_css


@pytest.mark.unit
class TestMemoryDemo:
    """Test suite for Memory Demo component."""
    
    def test_initialization(self, mock_safla_manager):
        """Test memory demo initialization."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        assert memory_demo.safla_manager == mock_safla_manager
        assert memory_demo.last_search_results == []
    
    def test_create_interface_structure(self, mock_safla_manager):
        """Test that interface creates all expected components."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        # Mock Gradio components
        with patch('gradio.Markdown') as mock_markdown, \
             patch('gradio.Tabs') as mock_tabs, \
             patch('gradio.TabItem') as mock_tab_item:
            
            memory_demo.create_interface()
            
            # Check that main components were created
            assert mock_markdown.call_count >= 2  # Title and description
            mock_tabs.assert_called_once()
            assert mock_tab_item.call_count >= 4  # 4 memory types
    
    @pytest.mark.asyncio
    async def test_perform_vector_search_valid(self, mock_safla_manager):
        """Test vector search with valid input."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        # Setup test data
        query = "test query"
        threshold = 0.8
        max_results = 5
        
        expected_results = [
            {"content": "result 1", "similarity": 0.9},
            {"content": "result 2", "similarity": 0.85}
        ]
        
        mock_safla_manager.search_vector_memory.return_value = expected_results
        
        # Run async function using asyncio
        results, fig, analytics = await memory_demo.perform_vector_search(
            query, threshold, max_results
        )
        
        # Verify results
        assert results == expected_results
        assert isinstance(fig, go.Figure)
        assert isinstance(analytics, pd.DataFrame)
        assert len(analytics) > 0
        
        # Verify search was called correctly
        mock_safla_manager.search_vector_memory.assert_called_once_with(
            query=query,
            threshold=threshold,
            max_results=max_results
        )
    
    @pytest.mark.asyncio
    async def test_perform_vector_search_empty_query(self, mock_safla_manager):
        """Test vector search with empty query."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        # Test with empty query
        results, fig, analytics = await memory_demo.perform_vector_search(
            "", 0.7, 5
        )
        
        # Should return empty results
        assert results == []
        assert isinstance(fig, go.Figure)
        assert isinstance(analytics, pd.DataFrame)
        assert len(analytics) == 0
    
    def test_validate_search_input_valid(self, mock_safla_manager):
        """Test input validation with valid inputs."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        # Should not raise any exception
        try:
            memory_demo.validate_search_input("valid query", 0.7)
        except ValueError:
            pytest.fail("Valid input should not raise ValueError")
    
    def test_validate_search_input_empty_query(self, mock_safla_manager):
        """Test input validation with empty query."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            memory_demo.validate_search_input("", 0.7)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            memory_demo.validate_search_input("   ", 0.7)
    
    def test_validate_search_input_invalid_threshold(self, mock_safla_manager):
        """Test input validation with invalid threshold."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        # Threshold too low
        with pytest.raises(ValueError, match="Threshold must be between"):
            memory_demo.validate_search_input("query", -0.1)
        
        # Threshold too high
        with pytest.raises(ValueError, match="Threshold must be between"):
            memory_demo.validate_search_input("query", 1.5)
    
    def test_create_similarity_visualization_with_results(self, mock_safla_manager):
        """Test similarity visualization with results."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        results = [
            {"content": "First result content", "similarity": 0.95},
            {"content": "Second result content", "similarity": 0.87},
            {"content": "Third result content", "similarity": 0.75}
        ]
        
        fig = memory_demo._create_similarity_visualization(results, "test query")
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "bar"
    
    def test_create_similarity_visualization_no_results(self, mock_safla_manager):
        """Test similarity visualization with no results."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        fig = memory_demo._create_similarity_visualization([], "test query")
        
        assert isinstance(fig, go.Figure)
        # Check for "No results" annotation
        assert len(fig.layout.annotations) > 0
        assert "No results" in fig.layout.annotations[0].text
    
    def test_generate_search_analytics(self, mock_safla_manager):
        """Test search analytics generation."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        results = [
            {"content": "result 1", "similarity": 0.9},
            {"content": "result 2", "similarity": 0.8},
            {"content": "result 3", "similarity": 0.7}
        ]
        
        analytics = memory_demo._generate_search_analytics(results, "test query")
        
        assert isinstance(analytics, pd.DataFrame)
        assert "Metric" in analytics.columns
        assert "Value" in analytics.columns
        assert "Status" in analytics.columns
        assert len(analytics) > 0
        
        # Check specific metrics
        metrics = analytics["Metric"].tolist()
        assert "Results Found" in metrics
        assert "Average Similarity" in metrics
        assert "Max Similarity" in metrics
    
    def test_recall_episodic_memories(self, mock_safla_manager):
        """Test episodic memory recall."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        fig, details = memory_demo._recall_episodic_memories(
            "Last Day",
            "Learning",
            "Positive"
        )
        
        assert isinstance(fig, go.Figure)
        assert isinstance(details, dict)
        assert "experiences" in details
        assert len(details["experiences"]) <= 5
    
    def test_explore_semantic_connections(self, mock_safla_manager):
        """Test semantic connection exploration."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        fig = memory_demo._explore_semantic_connections(
            "artificial intelligence",
            depth=2
        )
        
        assert isinstance(fig, go.Figure)
        # Check for annotation with graph info
        assert len(fig.layout.annotations) > 0
    
    def test_get_working_memory_status(self, mock_safla_manager):
        """Test working memory status retrieval."""
        memory_demo = MemoryDemo(mock_safla_manager)
        
        status, fig = memory_demo._get_working_memory_status()
        
        assert isinstance(status, dict)
        assert "capacity_used" in status
        assert "active_items" in status
        assert "attention_weights" in status
        
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "pie"


@pytest.mark.unit
class TestSaflaTheme:
    """Test suite for SAFLA theme."""
    
    def test_theme_initialization(self):
        """Test SAFLA theme initialization."""
        theme = SaflaTheme()
        
        assert theme is not None
        # Theme should have set custom colors
        assert hasattr(theme, 'primary_hue')
        assert hasattr(theme, 'secondary_hue')
    
    def test_get_custom_css(self):
        """Test custom CSS generation."""
        css = get_custom_css()
        
        assert isinstance(css, str)
        assert ".gr-button-primary" in css
        assert "linear-gradient" in css
        assert ".gr-tab-item" in css
        assert "@media" in css  # Responsive styles