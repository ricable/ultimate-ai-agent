"""
Text chunking functionality for document processing.
"""
import os
from typing import List
from pathlib import Path

class TextChunker:
    """
    Simple text chunker that splits documents into manageable pieces.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker with size and overlap settings.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)  # Ensure overlap isn't too large
        
        print(f"Initialized TextChunker with size={chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using a simple sliding window approach.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Handle empty or very short text
        if not text or not text.strip():
            print("Warning: Empty text provided to chunker")
            return [""]
            
        if len(text) <= self.chunk_size:
            print(f"Text is only {len(text)} chars, returning as single chunk")
            return [text]
        
        # Simple sliding window chunking
        chunks = []
        step_size = self.chunk_size - self.chunk_overlap
        
        # Ensure step size is at least 100 characters to prevent infinite loops
        if step_size < 100:
            step_size = 100
            print(f"Warning: Adjusted step size to {step_size} to ensure progress")
        
        # Create chunks with a sliding window
        position = 0
        text_length = len(text)
        
        while position < text_length:
            # Calculate end position for current chunk
            end = min(position + self.chunk_size, text_length)
            
            # Extract the chunk
            chunk = text[position:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move position forward by step_size
            position += step_size
            
            # Safety check
            if position >= text_length:
                break
                
            # Progress indicator for large texts
            if text_length > 100000 and len(chunks) % 10 == 0:
                print(f"Chunking progress: {min(position, text_length)}/{text_length} characters")
        
        print(f"Created {len(chunks)} chunks from {text_length} characters of text")
        return chunks
    
    def chunk_by_separator(self, text: str, separator: str = "\n\n") -> List[str]:
        """
        Split text by separator first, then ensure chunks are within size limits.
        
        Args:
            text: The text to split
            separator: The separator to split on (default: paragraph breaks)
            
        Returns:
            List of text chunks
        """
        # Handle empty text
        if not text or not text.strip():
            return [""]
            
        # Handle short text
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by separator
        parts = text.split(separator)
        print(f"Split text into {len(parts)} parts using separator '{separator}'")
        
        # Filter out empty parts
        parts = [part for part in parts if part.strip()]
        
        # Handle case where there are no meaningful parts
        if not parts:
            return [""]
            
        # Handle case where each part is already small enough
        if all(len(part) <= self.chunk_size for part in parts):
            print("All parts are within chunk size limit")
            return parts
        
        # Combine parts into chunks that fit within chunk_size
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # If this part alone exceeds chunk size, we need to split it further
            if len(part) > self.chunk_size:
                # First add any accumulated chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    
                # Then split the large part using the regular chunker
                part_chunks = self.chunk_text(part)
                chunks.extend(part_chunks)
                continue
                
            # If adding this part would exceed chunk size, start a new chunk
            if current_chunk and len(current_chunk) + len(separator) + len(part) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = part
            # Otherwise add to current chunk
            else:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
        
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(current_chunk)
            
        print(f"Created {len(chunks)} chunks using separator-based chunking")
        return chunks
