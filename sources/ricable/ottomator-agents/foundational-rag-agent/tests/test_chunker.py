"""
Unit tests for the text chunker module.
"""
import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.chunker import TextChunker


class TestTextChunker:
    """
    Test cases for the TextChunker class.
    """
    
    def test_init_with_default_values(self):
        """
        Test that TextChunker initializes with default values.
        """
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
    
    def test_init_with_custom_values(self):
        """
        Test that TextChunker initializes with custom values.
        """
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
    
    def test_init_with_large_overlap(self):
        """
        Test that TextChunker adjusts overlap when it's too large.
        """
        # In our new implementation, we automatically adjust the overlap
        # to be at most half of the chunk size
        chunker = TextChunker(chunk_size=500, chunk_overlap=500)
        assert chunker.chunk_overlap == 250  # Should be adjusted to chunk_size // 2
        
        chunker = TextChunker(chunk_size=500, chunk_overlap=600)
        assert chunker.chunk_overlap == 250  # Should be adjusted to chunk_size // 2
    
    def test_chunk_text_short_text(self):
        """
        Test chunking text that is shorter than chunk_size.
        """
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        text = "This is a short text."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_long_text(self):
        """
        Test chunking text that is longer than chunk_size.
        """
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a longer text that should be split into multiple chunks. " * 5
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        
        # Check that the chunks cover the entire text
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed += chunk
            else:
                # Account for overlap
                overlap_start = len(reconstructed) - chunker.chunk_overlap
                if overlap_start > 0:
                    reconstructed = reconstructed[:overlap_start] + chunk
                else:
                    reconstructed += chunk
        
        # The reconstructed text might be slightly different due to splitting at sentence boundaries
        assert len(reconstructed) >= len(text) * 0.9
    
    def test_chunk_by_separator(self):
        """
        Test splitting text by a separator.
        """
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        paragraphs = [
            "This is the first paragraph.",
            "This is the second paragraph.",
            "This is the third paragraph.",
            "This is the fourth paragraph."
        ]
        text = "\n\n".join(paragraphs)
        
        chunks = chunker.chunk_by_separator(text, separator="\n\n")
        
        assert len(chunks) == 4
        for i, paragraph in enumerate(paragraphs):
            assert paragraph in chunks[i]
    
    def test_chunk_by_separator_large_paragraph(self):
        """
        Test splitting text by a separator with a paragraph larger than chunk_size.
        """
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        paragraphs = [
            "This is a short paragraph.",
            "This is a very long paragraph that exceeds the chunk size and should be split into multiple chunks.",
            "This is another short paragraph."
        ]
        text = "\n\n".join(paragraphs)
        
        chunks = chunker.chunk_by_separator(text, separator="\n\n")
        
        assert len(chunks) > 3  # The long paragraph should be split
        assert paragraphs[0] in chunks[0]
        assert paragraphs[2] in chunks[-1]
