"""
Embeddings generation with OpenAI for document processing.
"""
import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
from pathlib import Path

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

class EmbeddingGenerator:
    """
    Simple and reliable embedding generator using OpenAI's API.
    """
    
    def __init__(self):
        """Initialize the embedding generator with API key from environment variables."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided as OPENAI_API_KEY environment variable.")
        
        # Set up the OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Default embedding dimension for text-embedding-3-small
        self.embedding_dim = 1536
        
        print(f"Initialized EmbeddingGenerator with model: {self.model}")
    
    def _create_zero_embedding(self) -> List[float]:
        """Create a zero vector with the correct dimension."""
        return [0.0] * self.embedding_dim
    
    def embed_text(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate an embedding for a single text with retry logic.
        
        Args:
            text: The text to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            Embedding vector
        """
        # Handle empty text
        if not text or not text.strip():
            print("Warning: Empty text provided, returning zero embedding")
            return self._create_zero_embedding()
        
        # Truncate very long text to avoid API limits
        max_length = 8000
        if len(text) > max_length:
            print(f"Warning: Text exceeds {max_length} characters, truncating")
            text = text[:max_length]
        
        # Try to generate embedding with retries
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Embedding error (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    print("All retry attempts failed, returning zero embedding")
                    return self._create_zero_embedding()
    
    def embed_batch(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in small batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            print("No valid texts to embed")
            return []
        
        results = []
        
        # Process in small batches to avoid memory issues
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(valid_texts)-1)//batch_size + 1} with {len(batch)} texts")
            
            # Process each text individually for better error isolation
            batch_results = []
            for text in batch:
                embedding = self.embed_text(text)
                batch_results.append(embedding)
            
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(valid_texts):
                time.sleep(0.5)
        
        print(f"Successfully embedded {len(results)} texts")
        return results
