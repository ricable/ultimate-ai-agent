"""Unified document chunking using Docling and fallback methods."""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..utils.config import DoclingConfig
from ..utils.logging import get_logger
from ..utils.deduplication import DatasetDeduplicator
from .serialization import SerializationFramework, SerializationConfig, Flow4ChunkingSerializerProvider

logger = get_logger(__name__)

# Advanced tokenization imports
try:
    from transformers import AutoTokenizer
    import tiktoken
    HAS_TOKENIZERS = True
    logger.info("Advanced tokenizers available (HuggingFace, OpenAI)")
except ImportError:
    logger.warning("Advanced tokenizers not available. Using simple tokenization.")
    HAS_TOKENIZERS = False
    AutoTokenizer = None
    tiktoken = None


# ===== ADVANCED TOKENIZATION STRATEGIES =====

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    def __init__(self, cache_tokenizers: bool = True):
        self.cache_tokenizers = cache_tokenizers
        self._cached_tokenizer = None
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    def split_oversized_chunk(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Split text that exceeds max_tokens into smaller chunks."""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if self.count_tokens(potential_chunk) <= max_tokens:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Add overlap if specified
                    if overlap_tokens > 0 and chunks:
                        overlap_text = self._extract_overlap(current_chunk, overlap_tokens)
                        current_chunk = overlap_text + (" " if overlap_text else "") + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    word_chunks = self._split_by_words(sentence, max_tokens)
                    chunks.extend(word_chunks[:-1])  # Add all but last
                    current_chunk = word_chunks[-1] if word_chunks else ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_overlap(self, text: str, overlap_tokens: int) -> str:
        """Extract the last N tokens from text for overlap."""
        tokens = self.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        overlap_token_ids = tokens[-overlap_tokens:]
        return self.decode(overlap_token_ids)
    
    def _split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """Split text by words when sentences are too long."""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            potential_chunk = current_chunk + (" " if current_chunk else "") + word
            if self.count_tokens(potential_chunk) <= max_tokens:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace transformer tokenizer for sentence-transformers and other models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_tokenizers: bool = True):
        super().__init__(cache_tokenizers)
        if not HAS_TOKENIZERS:
            raise ImportError("HuggingFace transformers not available. Install with: pip install transformers")
        
        self.model_name = model_name
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the HuggingFace tokenizer."""
        if self.cache_tokenizers and self._cached_tokenizer is not None:
            return
        
        try:
            self._cached_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded HuggingFace tokenizer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace tokenizer {self.model_name}: {e}")
            # Fallback to a known working tokenizer
            self._cached_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            logger.info("Using fallback tokenizer: bert-base-uncased")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer."""
        if not text.strip():
            return 0
        try:
            tokens = self._cached_tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using word approximation: {e}")
            return len(text.split()) + int(len(text.split()) * 0.3)  # Rough approximation
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._cached_tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._cached_tokenizer.decode(token_ids, skip_special_tokens=True)


class OpenAITokenizer(BaseTokenizer):
    """OpenAI tiktoken tokenizer for GPT models."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", cache_tokenizers: bool = True):
        super().__init__(cache_tokenizers)
        if not HAS_TOKENIZERS:
            raise ImportError("tiktoken not available. Install with: pip install tiktoken")
        
        self.model_name = model_name
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tiktoken encoder."""
        if self.cache_tokenizers and self._cached_tokenizer is not None:
            return
        
        try:
            self._cached_tokenizer = tiktoken.encoding_for_model(self.model_name)
            logger.info(f"Loaded tiktoken encoder for: {self.model_name}")
        except Exception:
            # Fallback to cl100k_base encoding
            self._cached_tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Using fallback tiktoken encoding: cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text.strip():
            return 0
        try:
            tokens = self._cached_tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using word approximation: {e}")
            return len(text.split()) + int(len(text.split()) * 0.3)  # Rough approximation
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._cached_tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._cached_tokenizer.decode(token_ids)


class SimpleTokenizer(BaseTokenizer):
    """Simple word-based tokenizer as fallback."""
    
    def __init__(self, cache_tokenizers: bool = True):
        super().__init__(cache_tokenizers)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using simple word splitting with approximation for subwords."""
        if not text.strip():
            return 0
        words = text.split()
        # Rough approximation: 1.3x word count to account for subword tokenization
        return int(len(words) * 1.3)
    
    def encode(self, text: str) -> List[int]:
        """Simple encoding - return word indices."""
        words = text.split()
        return list(range(len(words)))  # Simple indexing
    
    def decode(self, token_ids: List[int]) -> str:
        """Simple decoding - not actually reversible."""
        return f"[{len(token_ids)} tokens]"


class TokenizerFactory:
    """Factory for creating appropriate tokenizers."""
    
    @staticmethod
    def create_tokenizer(
        tokenizer_type: str = "auto",
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        cache_tokenizers: bool = True
    ) -> BaseTokenizer:
        """Create a tokenizer based on configuration.
        
        Args:
            tokenizer_type: Type of tokenizer ("auto", "huggingface", "openai", "simple")
            model_name: Specific model name for the tokenizer
            embedding_model: Embedding model to align with
            cache_tokenizers: Whether to cache tokenizer instances
            
        Returns:
            Appropriate tokenizer instance
        """
        
        # Auto-detect based on embedding model
        if tokenizer_type == "auto":
            if embedding_model:
                if "openai" in embedding_model.lower() or "gpt" in embedding_model.lower():
                    tokenizer_type = "openai"
                elif "sentence-transformers" in embedding_model.lower() or "bert" in embedding_model.lower():
                    tokenizer_type = "huggingface"
                    model_name = embedding_model
                else:
                    tokenizer_type = "huggingface"
                    model_name = embedding_model
            else:
                # Default to HuggingFace if available
                tokenizer_type = "huggingface" if HAS_TOKENIZERS else "simple"
        
        # Create appropriate tokenizer
        if tokenizer_type == "huggingface" and HAS_TOKENIZERS:
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceTokenizer(model_name, cache_tokenizers)
        elif tokenizer_type == "openai" and HAS_TOKENIZERS:
            model_name = model_name or "gpt-3.5-turbo"
            return OpenAITokenizer(model_name, cache_tokenizers)
        else:
            logger.warning(f"Falling back to simple tokenizer (requested: {tokenizer_type})")
            return SimpleTokenizer(cache_tokenizers)


# ===== ENHANCED CONFIGURATION =====

@dataclass
class TokenizerConfig:
    """Configuration for tokenization strategies."""
    tokenizer_type: str = "auto"  # auto, huggingface, openai, simple
    model_name: Optional[str] = None  # Specific tokenizer model
    embedding_model: Optional[str] = None  # Embedding model to align with
    max_tokens: int = 500  # Maximum tokens per chunk
    overlap_tokens: int = 50  # Token overlap between chunks
    cache_tokenizers: bool = True  # Cache tokenizer instances
    enable_token_aware_splitting: bool = True  # Use token-aware splitting
    enable_merge_peers: bool = True  # Merge undersized chunks
    min_tokens: int = 100  # Minimum tokens per chunk
    merge_threshold: float = 0.7  # Merge if chunk is < threshold * max_tokens


# Try to import docling for advanced chunking
try:
    from docling.document_converter import DocumentConverter as DoclingConverter
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import DocElement
    from docling.datamodel.document import DoclingDocument
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import (
        MarkdownTableSerializer,
        MarkdownParams,
    )
    from transformers import AutoTokenizer
    import tiktoken
    HAS_DOCLING = True
    HAS_ADVANCED_TOKENIZERS = True
    logger.info("Docling available for advanced hybrid chunking with context enrichment and custom tokenizers")
except ImportError as e:
    logger.warning(f"Docling advanced features not available: {e}. Falling back to simple chunking.")
    HAS_DOCLING = False
    HAS_ADVANCED_TOKENIZERS = False
    DoclingConverter = None
    HybridChunker = None
    DocElement = None
    DoclingDocument = None
    HuggingFaceTokenizer = None
    ChunkingDocSerializer = None
    ChunkingSerializerProvider = None
    MarkdownTableSerializer = None
    MarkdownParams = None
    AutoTokenizer = None
    tiktoken = None


class DocumentChunk:
    """Represents a document chunk with content and metadata."""
    
    def __init__(
        self,
        text: str,
        chunk_id: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a document chunk.
        
        Args:
            text: The chunk content
            chunk_id: Unique identifier for the chunk
            source: Source document name
            metadata: Additional metadata about the chunk
        """
        self.text = text
        self.chunk_id = chunk_id
        self.source = source
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata
        }


class DocumentChunker:
    """Unified document chunker with Docling and fallback methods."""
    
    def __init__(self, config: Optional[DoclingConfig] = None, tokenizer_config: Optional[TokenizerConfig] = None):
        """Initialize the document chunker.
        
        Args:
            config: Chunking configuration
            tokenizer_config: Tokenization configuration
        """
        self.config = config or DoclingConfig()
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self._docling_converter = None
        self._hybrid_chunker = None
        self._tokenizer = None
        self._serializer_provider = None
        self._serialization_framework = None
        
        # Enhanced chunking parameters
        self.semantic_min_size = max(self.config.min_chunk_size, 600)  # From improved chunker
        self.semantic_max_size = min(self.config.max_chunk_size, 2000)  # From improved chunker
        self.semantic_overlap = min(self.config.chunk_overlap, 100)  # Controlled overlap
        
        # Initialize tokenizer
        self._tokenizer = self._create_tokenizer()
        
        # Token-aware parameters
        self.max_tokens = self.tokenizer_config.max_tokens
        self.min_tokens = self.tokenizer_config.min_tokens
        self.overlap_tokens = self.tokenizer_config.overlap_tokens
        self.merge_threshold = self.tokenizer_config.merge_threshold
        
        logger.info(f"DocumentChunker initialized with tokenizer: {type(self._tokenizer).__name__}")
        logger.info(f"Token limits: min={self.min_tokens}, max={self.max_tokens}, overlap={self.overlap_tokens}")
        
        # Initialize serialization framework
        self._initialize_serialization_framework()
    
    def _create_tokenizer(self) -> BaseTokenizer:
        """Create appropriate tokenizer based on configuration."""
        return TokenizerFactory.create_tokenizer(
            tokenizer_type=self.tokenizer_config.tokenizer_type,
            model_name=self.tokenizer_config.model_name,
            embedding_model=self.tokenizer_config.embedding_model,
            cache_tokenizers=self.tokenizer_config.cache_tokenizers
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using configured tokenizer."""
        return self._tokenizer.count_tokens(text)
    
    def split_oversized_chunk(self, text: str) -> List[str]:
        """Split text that exceeds max tokens."""
        return self._tokenizer.split_oversized_chunk(text, self.max_tokens, self.overlap_tokens)
    
    def should_merge_chunks(self, chunk1: str, chunk2: str) -> bool:
        """Determine if two chunks should be merged based on token limits."""
        if not self.tokenizer_config.enable_merge_peers:
            return False
        
        chunk1_tokens = self.count_tokens(chunk1)
        chunk2_tokens = self.count_tokens(chunk2)
        
        # Don't merge if either chunk is above merge threshold
        if (chunk1_tokens > self.max_tokens * self.merge_threshold or 
            chunk2_tokens > self.max_tokens * self.merge_threshold):
            return False
        
        # Check if combined chunk would fit
        combined_tokens = self.count_tokens(chunk1 + " " + chunk2)
        return combined_tokens <= self.max_tokens
    
    def optimize_chunk_sizes(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Optimize chunk sizes using token-aware merging and splitting."""
        if not chunks:
            return chunks
        
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_tokens = self.count_tokens(current_chunk.text)
            
            # Handle oversized chunks
            if current_tokens > self.max_tokens:
                if self.tokenizer_config.enable_token_aware_splitting:
                    split_texts = self.split_oversized_chunk(current_chunk.text)
                    for j, split_text in enumerate(split_texts):
                        new_chunk = DocumentChunk(
                            text=split_text,
                            chunk_id=current_chunk.chunk_id + j * 0.1,  # Fractional IDs for splits
                            source=current_chunk.source,
                            metadata={
                                **current_chunk.metadata,
                                "split_from_oversized": True,
                                "split_index": j,
                                "original_chunk_id": current_chunk.chunk_id,
                                "token_count": self.count_tokens(split_text)
                            }
                        )
                        optimized_chunks.append(new_chunk)
                else:
                    # Just add the oversized chunk with warning
                    current_chunk.metadata["oversized_warning"] = True
                    current_chunk.metadata["token_count"] = current_tokens
                    optimized_chunks.append(current_chunk)
                i += 1
                continue
            
            # Try to merge with next chunk if undersized
            merged = False
            if (current_tokens < self.min_tokens and 
                i + 1 < len(chunks) and 
                self.tokenizer_config.enable_merge_peers):
                
                next_chunk = chunks[i + 1]
                if self.should_merge_chunks(current_chunk.text, next_chunk.text):
                    # Merge chunks
                    merged_text = current_chunk.text + "\n\n" + next_chunk.text
                    merged_chunk = DocumentChunk(
                        text=merged_text,
                        chunk_id=current_chunk.chunk_id,
                        source=current_chunk.source,
                        metadata={
                            **current_chunk.metadata,
                            "merged_with_next": True,
                            "merged_chunk_ids": [current_chunk.chunk_id, next_chunk.chunk_id],
                            "token_count": self.count_tokens(merged_text)
                        }
                    )
                    optimized_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    merged = True
            
            if not merged:
                # Add current chunk with token count
                current_chunk.metadata["token_count"] = current_tokens
                optimized_chunks.append(current_chunk)
                i += 1
        
        logger.info(f"Chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks")
        return optimized_chunks
    
    def _initialize_serialization_framework(self):
        """Initialize the serialization framework for enhanced content handling."""
        try:
            # Create serialization config based on DoclingConfig
            serialization_config = SerializationConfig(
                table_format="markdown",
                picture_format="annotation",
                include_table_captions=True,
                include_picture_descriptions=True,
                preserve_markup=True,
                include_source_metadata=True
            )
            
            # Initialize the serialization framework
            self._serialization_framework = SerializationFramework(serialization_config)
            
            # Create serializer provider for chunking
            self._serializer_provider = self._serialization_framework.create_serializer_provider()
            
            logger.info("Serialization framework initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize serialization framework: {e}")
            self._serialization_framework = None
            self._serializer_provider = None
    
    def _initialize_advanced_chunking(self):
        """Initialize advanced Docling chunking components with custom tokenizers and serializers."""
        try:
            # Initialize custom tokenizer based on configuration
            self._tokenizer = self._create_custom_tokenizer()
            
            # Initialize custom serializer provider for enhanced table and figure handling
            self._serializer_provider = self._create_serializer_provider()
            
            # Initialize advanced HybridChunker with custom components
            self._hybrid_chunker = self._create_hybrid_chunker()
            
            logger.info("Advanced Docling chunking initialized with custom tokenizer and serializers")
            
        except Exception as e:
            logger.warning(f"Failed to initialize advanced chunking components: {e}")
            self._tokenizer = None
            self._serializer_provider = None
            self._hybrid_chunker = None
    
    def _create_custom_tokenizer(self):
        """Create custom tokenizer based on configuration."""
        tokenizer_name = getattr(self.config, 'tokenizer', 'cl100k_base')
        max_tokens = getattr(self.config, 'chunk_size', 500)
        
        try:
            # Try HuggingFace tokenizer first for better semantic understanding
            if tokenizer_name.startswith('sentence-transformers/') or '/' in tokenizer_name:
                logger.info(f"Using HuggingFace tokenizer: {tokenizer_name}")
                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                return HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer,
                    max_tokens=max_tokens,
                )
            
            # Try OpenAI/tiktoken tokenizer for specific models
            elif tokenizer_name in ['cl100k_base', 'p50k_base', 'r50k_base', 'gpt2']:
                logger.info(f"Using tiktoken tokenizer: {tokenizer_name}")
                # Note: Direct tiktoken integration would require custom wrapper
                # Fall back to HuggingFace for now
                default_model = "sentence-transformers/all-MiniLM-L6-v2"
                hf_tokenizer = AutoTokenizer.from_pretrained(default_model)
                return HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer,
                    max_tokens=max_tokens,
                )
            
            else:
                # Default to sentence transformers for semantic chunking
                logger.info("Using default sentence transformers tokenizer")
                default_model = "sentence-transformers/all-MiniLM-L6-v2"
                hf_tokenizer = AutoTokenizer.from_pretrained(default_model)
                return HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer,
                    max_tokens=max_tokens,
                )
                
        except Exception as e:
            logger.warning(f"Failed to create custom tokenizer {tokenizer_name}: {e}")
            return None
    
    def _create_serializer_provider(self):
        """Create custom serializer provider for enhanced table and image handling."""
        try:
            # Use Flow4's advanced serialization framework if available
            if self._serialization_framework:
                return self._serialization_framework.create_serializer_provider()
            
            # Fallback to basic Docling serializer if Flow4 framework not available
            class EnhancedSerializerProvider(ChunkingSerializerProvider):
                def __init__(self, config):
                    self.config = config
                
                def get_serializer(self, doc):
                    # Create enhanced markdown parameters for better formatting
                    markdown_params = MarkdownParams(
                        image_placeholder="<!-- [Image: {}] -->",
                        strict_json_figure_handling=True,
                    )
                    
                    # Use MarkdownTableSerializer for better table handling
                    table_serializer = MarkdownTableSerializer()
                    
                    return ChunkingDocSerializer(
                        doc=doc,
                        table_serializer=table_serializer,
                        params=markdown_params,
                    )
            
            return EnhancedSerializerProvider(self.config)
            
        except Exception as e:
            logger.warning(f"Failed to create custom serializer provider: {e}")
            return None
    
    def _create_hybrid_chunker(self):
        """Create advanced HybridChunker with custom configuration."""
        try:
            kwargs = {}
            
            # Add tokenizer if available
            if self._tokenizer:
                kwargs['tokenizer'] = self._tokenizer
            
            # Add serializer provider if available
            if self._serializer_provider:
                kwargs['serializer_provider'] = self._serializer_provider
            
            # Enable peer merging for better chunk coherence
            kwargs['merge_peers'] = True
            
            return HybridChunker(**kwargs)
            
        except Exception as e:
            logger.warning(f"Failed to create advanced HybridChunker: {e}")
            return None
    
    def _get_docling_converter(self):
        """Get or create Docling DocumentConverter instance."""
        if not HAS_DOCLING:
            return None
        
        if self._docling_converter is None:
            self._docling_converter = DoclingConverter()
        return self._docling_converter
    
    def _extract_metadata_from_markdown(self, markdown_content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter metadata from markdown.
        
        Args:
            markdown_content: Markdown content with potential frontmatter
            
        Returns:
            Tuple of (metadata dict, content without frontmatter)
        """
        metadata = {}
        content = markdown_content
        
        if markdown_content.startswith('---'):
            parts = markdown_content.split('---', 2)
            if len(parts) >= 3:
                yaml_text = parts[1].strip()
                content = parts[2].strip()
                
                # Simple YAML parsing
                for line in yaml_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        return metadata, content
    
    def _chunk_with_docling(self, markdown_path: str) -> List[DocumentChunk]:
        """Chunk document using advanced Docling hybrid chunking with custom tokenizers and serializers.
        
        Args:
            markdown_path: Path to the markdown file
            
        Returns:
            List of DocumentChunk objects
        """
        if not HAS_DOCLING:
            raise ImportError("Docling is required for advanced chunking.")
        
        logger.info(f"Chunking {markdown_path} with advanced Docling hybrid chunker")
        
        try:
            # Initialize DocumentConverter
            converter = self._get_docling_converter()
            
            # Convert the markdown file to a Docling document
            logger.debug(f"Converting {markdown_path} to Docling document format")
            result = converter.convert(source=markdown_path)
            doc = result.document
            
            # Use advanced HybridChunker if available, otherwise fall back to default
            if self._hybrid_chunker:
                chunker = self._hybrid_chunker
                logger.debug("Using advanced HybridChunker with custom tokenizer and serializers")
            else:
                chunker = HybridChunker()
                logger.debug("Using default HybridChunker")
            
            # Generate chunks using the hybrid chunker
            logger.debug("Generating chunks with advanced tokenization and serialization")
            chunk_iter = chunker.chunk(dl_doc=doc)
            
            # Convert to our DocumentChunk format with enhanced processing
            chunks = []
            source_name = os.path.basename(markdown_path)
            
            for i, chunk in enumerate(chunk_iter):
                # Get context-enriched text (recommended for embeddings)
                enriched_text = self._advanced_contextualize(chunker, chunk, doc)
                
                # Use enriched text as the main content
                chunk_text = enriched_text if enriched_text else chunk.text
                
                # Extract enhanced metadata from the chunk
                metadata = self._extract_enhanced_chunk_metadata(
                    chunk, i, chunker, self._tokenizer
                )
                
                # Calculate token count using custom tokenizer if available
                if self._tokenizer:
                    try:
                        token_count = len(self._tokenizer.tokenizer.encode(chunk_text))
                        metadata["precise_token_count"] = token_count
                    except Exception as e:
                        logger.debug(f"Could not calculate precise token count: {e}")
                        metadata["token_count"] = len(chunk_text.split()) * 1.3
                else:
                    metadata["token_count"] = len(chunk_text.split()) * 1.3
                
                # Analyze multimodal content in chunk
                multimodal_info = self._analyze_chunk_multimodal_content(chunk_text)
                metadata.update(multimodal_info)
                
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    chunk_id=i,
                    source=source_name,
                    metadata=metadata
                ))
            
            # Post-process chunks for enhanced relationships and quality scoring
            chunks = self._post_process_docling_chunks(chunks)
            
            logger.info(f"Created {len(chunks)} chunks using advanced Docling HybridChunker")
            logger.info(f"Tokenizer: {type(self._tokenizer).__name__ if self._tokenizer else 'default'}")
            logger.info(f"Serializer: {'custom' if self._serializer_provider else 'default'}")
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error chunking with advanced Docling: {str(e)}")
            logger.info("Falling back to heading-based chunking")
            return self._chunk_by_headings(markdown_path)
    
    def _advanced_contextualize(self, chunker, chunk, doc):
        """Enhanced contextualization with additional context enrichment."""
        try:
            # Get standard contextualized text
            enriched_text = chunker.contextualize(chunk=chunk)
            
            # Add additional context enhancement if available
            if enriched_text and hasattr(chunk, 'meta') and chunk.meta:
                # Extract heading context from chunk metadata
                heading_context = self._extract_heading_context(chunk, doc)
                if heading_context:
                    enriched_text = f"{heading_context}\n\n{enriched_text}"
            
            return enriched_text
            
        except Exception as e:
            logger.debug(f"Error in advanced contextualization: {e}")
            return chunk.text
    
    def _extract_heading_context(self, chunk, doc):
        """Extract relevant heading context for better chunk understanding."""
        try:
            # This is a simplified implementation
            # In practice, you'd traverse the document structure to find relevant headings
            if hasattr(chunk, 'meta') and chunk.meta:
                # Extract any heading information from metadata
                return None  # Placeholder for heading extraction logic
            return None
        except Exception as e:
            logger.debug(f"Error extracting heading context: {e}")
            return None
    
    def _extract_enhanced_chunk_metadata(self, chunk, chunk_index, chunker, tokenizer):
        """Extract comprehensive metadata from Docling chunk with advanced features."""
        metadata = {
            "original_text": chunk.text,
            "chunking_method": "docling_hybrid_advanced",
            "chunk_index": chunk_index,
            "relationships": {
                "previous": chunk_index - 1 if chunk_index > 0 else None,
                "next": chunk_index + 1,  # Will be updated when we know total
            },
            "advanced_features": {
                "custom_tokenizer": tokenizer is not None,
                "custom_serializer": self._serializer_provider is not None,
                "peer_merging": True,
            }
        }
        
        # Add tokenizer information
        if tokenizer:
            try:
                metadata["tokenizer_info"] = {
                    "type": type(tokenizer).__name__,
                    "max_tokens": getattr(tokenizer, 'max_tokens', None),
                    "model_name": getattr(tokenizer.tokenizer, 'name_or_path', 'unknown') if hasattr(tokenizer, 'tokenizer') else 'unknown'
                }
            except Exception as e:
                logger.debug(f"Could not extract tokenizer info: {e}")
        
        # Add any additional metadata from the chunk object
        if hasattr(chunk, 'meta') and chunk.meta:
            try:
                # Convert DocMeta to dict if possible, or extract useful info
                if hasattr(chunk.meta, '__dict__'):
                    meta_dict = chunk.meta.__dict__
                    for key, value in meta_dict.items():
                        try:
                            json.dumps(value)  # Test if it's JSON serializable
                            metadata[f"docling_{key}"] = value
                        except (TypeError, ValueError):
                            # Skip non-serializable objects, but keep a reference
                            metadata[f"docling_{key}_type"] = str(type(value).__name__)
                else:
                    metadata["docling_meta_type"] = str(type(chunk.meta).__name__)
            except Exception as e:
                logger.debug(f"Could not extract metadata from chunk.meta: {e}")
                metadata["docling_meta_available"] = True
        
        return metadata
    
    def _analyze_chunk_multimodal_content(self, chunk_text: str) -> Dict[str, Any]:
        """Analyze multimodal content within a specific chunk."""
        analysis = {
            "has_tables": False,
            "has_images": False,
            "table_count": 0,
            "image_count": 0,
            "multimodal_elements": []
        }
        
        try:
            import re
            
            # Detect tables
            markdown_tables = re.findall(r'^\|.*\|\s*$', chunk_text, re.MULTILINE)
            if markdown_tables:
                analysis["has_tables"] = True
                analysis["table_count"] = len(markdown_tables)
                analysis["multimodal_elements"].append("markdown_table")
            
            # Detect images
            image_refs = re.findall(r'!\[.*?\]\(.*?\.(png|jpg|jpeg|gif|svg)\)', chunk_text, re.IGNORECASE)
            if image_refs:
                analysis["has_images"] = True
                analysis["image_count"] = len(image_refs)
                analysis["multimodal_elements"].append("image_reference")
            
            # Detect image placeholders from custom serializer
            image_placeholders = re.findall(r'<!-- \[Image:.*?\] -->', chunk_text)
            if image_placeholders:
                analysis["has_images"] = True
                analysis["image_count"] += len(image_placeholders)
                if "image_reference" not in analysis["multimodal_elements"]:
                    analysis["multimodal_elements"].append("image_placeholder")
            
        except Exception as e:
            logger.debug(f"Error analyzing multimodal content: {e}")
        
        return analysis
    
    def _post_process_docling_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process chunks for enhanced relationships and quality scoring."""
        # Update the "next" relationship for the last chunk
        if chunks:
            chunks[-1].metadata["relationships"]["next"] = None
        
        # Calculate enhanced quality scores for advanced chunks
        for chunk in chunks:
            quality_score = self._calculate_advanced_chunk_quality(chunk)
            chunk.metadata["quality_score"] = quality_score
            
            # Add enhanced content classification
            content_type = self._classify_advanced_content_type(chunk.text)
            chunk.metadata["content_type"] = content_type
            
            # Add chunk coherence metrics
            coherence_metrics = self._calculate_chunk_coherence(chunk)
            chunk.metadata["coherence_metrics"] = coherence_metrics
        
        return chunks
    
    def _calculate_advanced_chunk_quality(self, chunk: DocumentChunk) -> float:\n        \"\"\"Calculate enhanced quality score for advanced chunks.\"\"\"\n        base_score = self._calculate_chunk_quality_score(\n            chunk.text, \n            chunk.metadata.get(\"content_type\", \"general_content\")\n        )\n        \n        # Bonus for advanced features\n        advanced_bonus = 0.0\n        \n        # Tokenizer alignment bonus\n        if chunk.metadata.get(\"advanced_features\", {}).get(\"custom_tokenizer\"):\n            advanced_bonus += 0.1\n        \n        # Multimodal content bonus\n        if chunk.metadata.get(\"has_tables\") or chunk.metadata.get(\"has_images\"):\n            advanced_bonus += 0.05\n        \n        # Context enrichment bonus\n        if \"enriched_text\" in chunk.metadata and chunk.metadata[\"enriched_text\"]:\n            advanced_bonus += 0.05\n        \n        # Precise token count bonus\n        if \"precise_token_count\" in chunk.metadata:\n            advanced_bonus += 0.05\n        \n        return min(1.0, base_score + advanced_bonus)\n    \n    def _classify_advanced_content_type(self, text: str) -> str:\n        \"\"\"Enhanced content type classification for advanced chunks.\"\"\"\n        # Use the existing classification as base\n        base_type = self._classify_section_type(text)\n        \n        # Add more sophisticated classification for technical content\n        text_lower = text.lower()\n        \n        # Enhanced technical classifications\n        if any(keyword in text_lower for keyword in ['algorithm', 'implementation', 'protocol']):\n            return 'technical_implementation'\n        elif any(keyword in text_lower for keyword in ['specification', 'requirements', 'standard']):\n            return 'technical_specification'\n        elif any(keyword in text_lower for keyword in ['troubleshooting', 'debugging', 'error']):\n            return 'troubleshooting_guide'\n        elif any(keyword in text_lower for keyword in ['installation', 'setup', 'deployment']):\n            return 'installation_guide'\n        \n        return base_type\n    \n    def _calculate_chunk_coherence(self, chunk: DocumentChunk) -> Dict[str, float]:\n        \"\"\"Calculate coherence metrics for chunk quality assessment.\"\"\"\n        text = chunk.text\n        metrics = {\n            \"sentence_coherence\": 0.0,\n            \"lexical_coherence\": 0.0,\n            \"structural_coherence\": 0.0\n        }\n        \n        try:\n            # Sentence coherence (based on sentence completeness)\n            sentences = text.split('.')\n            complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10 and not s.strip().endswith(','))\n            if sentences:\n                metrics[\"sentence_coherence\"] = complete_sentences / len(sentences)\n            \n            # Lexical coherence (based on vocabulary consistency)\n            words = text.lower().split()\n            unique_words = set(words)\n            if words:\n                metrics[\"lexical_coherence\"] = len(unique_words) / len(words)\n            \n            # Structural coherence (based on formatting consistency)\n            lines = text.split('\\n')\n            formatted_lines = sum(1 for line in lines if line.strip())\n            if lines:\n                metrics[\"structural_coherence\"] = formatted_lines / len(lines)\n            \n        except Exception as e:\n            logger.debug(f\"Error calculating coherence metrics: {e}\")\n        \n        return metrics\n    \n    def _chunk_by_headings(self, markdown_path: str) -> List[DocumentChunk]:
        """Split markdown content into chunks based on headings with enhanced semantic awareness.
        
        Args:
            markdown_path: Path to the markdown file
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Chunking {markdown_path} by headings with semantic enhancement")
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata
        doc_metadata, content = self._extract_metadata_from_markdown(content)
        
        # Clean text using improved methods
        content = self._clean_technical_text(content)
        
        # Try semantic chunking first
        try:
            chunks = self._chunk_with_semantic_awareness(content, os.path.basename(markdown_path), doc_metadata)
            if chunks:
                logger.info(f"Created {len(chunks)} chunks using semantic chunking")
                return chunks
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to heading-based: {e}")
        
        # Fallback to heading-based chunking
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(content))
        
        chunks = []
        source_name = os.path.basename(markdown_path)
        
        if not headings:
            # No headings found, use enhanced size-based chunking
            chunks = self._chunk_by_enhanced_size(content, source_name, doc_metadata)
        else:
            # Enhanced heading-based chunking with size optimization
            chunks = self._chunk_by_optimized_headings(content, headings, source_name, doc_metadata)
        
        logger.info(f"Created {len(chunks)} chunks by enhanced headings")
        return chunks
    
    def _chunk_by_size(
        self, 
        content: str, 
        source_name: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk content by size when no headings are available (legacy method).
        
        Args:
            content: Text content to chunk
            source_name: Name of the source document
            doc_metadata: Document-level metadata
            
        Returns:
            List of DocumentChunk objects
        """
        return self._chunk_by_enhanced_size(content, source_name, doc_metadata)
    
    def _chunk_by_enhanced_size(
        self, 
        content: str, 
        source_name: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Enhanced size-based chunking with sentence boundary awareness.
        
        Args:
            content: Text content to chunk
            source_name: Name of the source document
            doc_metadata: Document-level metadata
            
        Returns:
            List of DocumentChunk objects
        """
        # Split into sentences for better boundary awareness
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.semantic_max_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.semantic_min_size:
                    chunks.append(self._create_enhanced_chunk(
                        chunk_id, current_chunk, source_name, doc_metadata, "enhanced_size_based"
                    ))
                    chunk_id += 1
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.semantic_min_size:
            chunks.append(self._create_enhanced_chunk(
                chunk_id, current_chunk, source_name, doc_metadata, "enhanced_size_based"
            ))
        
        return chunks
    
    def _clean_technical_text(self, text: str) -> str:
        """Clean malformed table data and formatting issues from technical documentation."""
        # Fix broken table syntax specific to technical docs
        text = re.sub(r',\s*=\s*\.\s*,\s*LTE\s*=\s*\.', '', text)
        text = re.sub(r'Level,\s*=\s*\.\s*Level,\s*LTE\s*=\s*[A-Za-z]+\.', '', text)
        text = re.sub(r'Formula Status,\s*=\s*\.\s*Formula Status,\s*LTE\s*=\s*[A-Z]+\.', '', text)
        text = re.sub(r'Beneficial Trend,\s*=\s*\.\s*Beneficial Trend,\s*LTE\s*=\s*[A-Za-z]+', '', text)
        
        # Clean excessive whitespace
        text = re.sub(r'\\n\\s*\\n\\s*\\n', '\\n\\n', text)
        text = re.sub(r'\\t+', ' ', text)
        
        return text.strip()
    
    def _classify_section_type(self, content: str) -> str:
        """Classify the type of content section for better chunking strategy."""
        content_lower = content.lower()
        
        if 'class ' in content and ('attributes' in content_lower or 'dependencies' in content_lower):
            return 'class_definition'
        elif 'enum ' in content:
            return 'enum_definition'
        elif 'kpi' in content_lower or 'performance' in content_lower:
            return 'kpi_definition'
        elif 'formula' in content_lower and any(counter in content for counter in ['pm', 'counter']):
            return 'kpi_calculation'
        elif 'feature' in content_lower and 'identity' in content_lower:
            return 'feature_description'
        elif any(proc in content_lower for proc in ['procedure', 'process', 'step']):
            return 'procedure'
        elif any(cfg in content_lower for cfg in ['configure', 'configuration', 'parameter']):
            return 'configuration'
        else:
            return 'general_content'
    
    def _extract_semantic_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract semantic sections from technical documentation."""
        sections = []
        lines = content.split('\\n')
        current_section = []
        current_headers = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect headers (numbered sections, class definitions, etc.)
            if (re.match(r'^\\d+(\\.(\\d+))*\\s+', line) or  # "2.3.1 Section Name"
                line.startswith('class ') or 
                line.startswith('enum ') or
                line.startswith('#') or
                re.match(r'^[A-Z][A-Za-z\\s]+:', line)):  # "Configuration:"
                
                # Save previous section if it exists and is substantial
                if current_section:
                    content_text = '\\n'.join(current_section)
                    if len(content_text.strip()) > 50:
                        sections.append({
                            'headers': current_headers.copy(),
                            'content': self._clean_technical_text(content_text),
                            'type': self._classify_section_type(content_text)
                        })
                
                # Start new section
                current_headers = [line]
                current_section = [line]
                
            elif line and not line.startswith(' '):  # Potential subsection header
                current_section.append(line)
                
            else:
                current_section.append(line)
            
            i += 1
        
        # Don't forget the last section
        if current_section:
            content_text = '\\n'.join(current_section)
            if len(content_text.strip()) > 50:
                sections.append({
                    'headers': current_headers.copy(),
                    'content': self._clean_technical_text(content_text),
                    'type': self._classify_section_type(content_text)
                })
        
        return sections
    
    def _chunk_with_semantic_awareness(self, content: str, source_name: str, doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create optimally sized chunks with semantic boundaries."""
        sections = self._extract_semantic_sections(content)
        chunks = []
        current_chunk = ""
        current_headers = []
        chunk_id = 0
        
        for section in sections:
            section_content = section['content']
            section_headers = section['headers']
            section_type = section['type']
            
            # For complete definitions, try to keep them together
            if section_type in ['class_definition', 'enum_definition', 'kpi_calculation']:
                if len(section_content) <= self.semantic_max_size:
                    # Complete section fits in one chunk
                    if current_chunk and len(current_chunk + '\\n\\n' + section_content) > self.semantic_max_size:
                        # Save current chunk and start new one
                        if len(current_chunk) >= self.semantic_min_size:
                            chunks.append(self._create_enhanced_chunk(
                                chunk_id, current_chunk, source_name, doc_metadata, 
                                "semantic_aware", current_headers
                            ))
                            chunk_id += 1
                        current_chunk = section_content
                        current_headers = section_headers
                    else:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += '\\n\\n' + section_content
                        else:
                            current_chunk = section_content
                            current_headers = section_headers
                else:
                    # Section too large, split but preserve logical boundaries
                    if current_chunk and len(current_chunk) >= self.semantic_min_size:
                        chunks.append(self._create_enhanced_chunk(
                            chunk_id, current_chunk, source_name, doc_metadata, 
                            "semantic_aware", current_headers
                        ))
                        chunk_id += 1
                    
                    # Split large section
                    section_chunks = self._split_large_section(section_content)
                    for sc in section_chunks:
                        chunks.append(self._create_enhanced_chunk(
                            chunk_id, sc, source_name, doc_metadata, 
                            "semantic_aware", section_headers
                        ))
                        chunk_id += 1
                    
                    current_chunk = ""
                    current_headers = []
            else:
                # General content - can be split more freely
                if current_chunk and len(current_chunk + '\\n\\n' + section_content) > self.semantic_max_size:
                    if len(current_chunk) >= self.semantic_min_size:
                        chunks.append(self._create_enhanced_chunk(
                            chunk_id, current_chunk, source_name, doc_metadata, 
                            "semantic_aware", current_headers
                        ))
                        chunk_id += 1
                    current_chunk = section_content
                    current_headers = section_headers
                else:
                    if current_chunk:
                        current_chunk += '\\n\\n' + section_content
                    else:
                        current_chunk = section_content
                        current_headers = section_headers
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.semantic_min_size:
            chunks.append(self._create_enhanced_chunk(
                chunk_id, current_chunk, source_name, doc_metadata, 
                "semantic_aware", current_headers
            ))
        
        return chunks
    
    def _split_large_section(self, content: str) -> List[str]:
        """Split large sections while preserving semantic meaning."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\\s+', content)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + ' ' + sentence) <= self.semantic_max_size:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_by_optimized_headings(self, content: str, headings: List, source_name: str, doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Enhanced heading-based chunking with size optimization."""
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for i, heading in enumerate(headings):
            # Determine chunk start and end
            start_pos = heading.start()
            end_pos = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            # If section is too large, split it
            if len(section_content) > self.semantic_max_size:
                # Save current chunk if it exists
                if current_chunk and len(current_chunk) >= self.semantic_min_size:
                    chunks.append(self._create_enhanced_chunk(
                        chunk_id, current_chunk, source_name, doc_metadata, "optimized_heading"
                    ))
                    chunk_id += 1
                    current_chunk = ""
                
                # Split large section
                section_chunks = self._split_large_section(section_content)
                for sc in section_chunks:
                    chunks.append(self._create_enhanced_chunk(
                        chunk_id, sc, source_name, doc_metadata, "optimized_heading"
                    ))
                    chunk_id += 1
            else:
                # Try to combine with current chunk
                potential_chunk = current_chunk + ('\\n\\n' if current_chunk else '') + section_content
                
                if len(potential_chunk) <= self.semantic_max_size:
                    current_chunk = potential_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk and len(current_chunk) >= self.semantic_min_size:
                        chunks.append(self._create_enhanced_chunk(
                            chunk_id, current_chunk, source_name, doc_metadata, "optimized_heading"
                        ))
                        chunk_id += 1
                    current_chunk = section_content
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.semantic_min_size:
            chunks.append(self._create_enhanced_chunk(
                chunk_id, current_chunk, source_name, doc_metadata, "optimized_heading"
            ))
        
        return chunks
    
    def _create_enhanced_chunk(self, chunk_id: int, content: str, source_name: str, 
                             doc_metadata: Dict[str, Any], method: str, 
                             headers: Optional[List[str]] = None) -> DocumentChunk:
        """Create an enhanced chunk with improved metadata."""
        # Accurate token count using configured tokenizer
        token_count = self.count_tokens(content)
        
        # Extract content type classification
        content_type = self._classify_section_type(content)
        
        metadata = {
            "original_text": content,
            "enriched_text": content,  # Could be enhanced with context in the future
            "chunking_method": method,
            "chunk_index": chunk_id,
            "content_type": content_type,
            "tokens": token_count,
            "char_count": len(content),
            "word_count": len(content.split()),
            "quality_score": self._calculate_chunk_quality_score(content, content_type),
            "tokenizer_type": type(self._tokenizer).__name__,
            "tokenizer_model": getattr(self._tokenizer, 'model_name', 'unknown'),
            "token_limits": {
                "max_tokens": self.max_tokens,
                "min_tokens": self.min_tokens,
                "overlap_tokens": self.overlap_tokens
            },
            "relationships": {
                "previous": chunk_id - 1 if chunk_id > 0 else None,
                "next": chunk_id + 1  # Will be updated when we know total
            },
            "source_metadata": doc_metadata
        }
        
        if headers:
            metadata["headers"] = headers
            metadata["heading_context"] = headers[0] if headers else None
        
        return DocumentChunk(
            text=content,
            chunk_id=chunk_id,
            source=source_name,
            metadata=metadata
        )
    
    def _calculate_chunk_quality_score(self, content: str, content_type: str) -> float:
        """Calculate a quality score for the chunk based on various factors including token optimization."""
        score = 0.0
        
        # Token-aware size scoring (optimal token count gets higher score)
        token_count = self.count_tokens(content)
        token_ratio = token_count / self.max_tokens
        
        if 0.4 <= token_ratio <= 0.8:  # Sweet spot for token utilization
            score += 0.4
        elif 0.2 <= token_ratio <= 0.9:  # Acceptable token utilization
            score += 0.3
        elif token_ratio < 0.2:  # Too few tokens
            score += 0.1
        else:  # Over token limit
            score += 0.05
        
        # Content completeness (fewer broken sentences)
        sentences = content.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
        if complete_sentences > 0:
            completeness = min(1.0, complete_sentences / max(1, len(sentences)))
            score += 0.2 * completeness
        
        # Content type bonus (structured content gets higher score)
        if content_type in ['class_definition', 'enum_definition', 'kpi_calculation']:
            score += 0.2
        elif content_type in ['procedure', 'configuration']:
            score += 0.15
        else:
            score += 0.1
        
        # Technical content indicators
        technical_indicators = len(re.findall(r'\\b(class|enum|attribute|parameter|configure|procedure)\\b', content.lower()))
        score += min(0.2, technical_indicators * 0.04)
        
        # Token efficiency bonus
        if self.min_tokens <= token_count <= self.max_tokens:
            score += 0.1  # Bonus for being within token limits
        
        return min(1.0, score)  # Cap at 1.0
    
    def chunk_file(self, markdown_path: str, use_docling: bool = True) -> List[DocumentChunk]:
        """Chunk a single markdown file.
        
        Args:
            markdown_path: Path to the markdown file
            use_docling: Whether to try Docling first (if available)
            
        Returns:
            List of DocumentChunk objects
        """
        if use_docling and HAS_DOCLING:
            try:
                return self._chunk_with_docling(markdown_path)
            except Exception as e:
                logger.warning(f"Docling chunking failed, falling back to simple chunking: {e}")
                return self._chunk_by_headings(markdown_path)
        else:
            return self._chunk_by_headings(markdown_path)
    
    def save_chunks(self, chunks: List[DocumentChunk], output_dir: str) -> None:
        """Save chunks to individual files with metadata.
        
        Args:
            chunks: List of chunks to save
            output_dir: Directory to save chunks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for chunk in chunks:
            # Save chunk content
            chunk_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.md")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk.text)
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(chunk.to_dict(), f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_dir}")
    
    def batch_chunk(
        self, 
        markdown_files: List[str], 
        output_dir: str,
        num_workers: int = 4,
        use_docling: bool = True
    ) -> List[DocumentChunk]:
        """Chunk multiple markdown files in parallel.
        
        Args:
            markdown_files: List of markdown file paths
            output_dir: Directory to save chunks
            num_workers: Number of parallel workers
            use_docling: Whether to try Docling first
            
        Returns:
            Combined list of all chunks
        """
        logger.info(f"Starting batch chunking of {len(markdown_files)} files")
        
        all_chunks = []
        
        # For now, process sequentially since Docling may not be thread-safe
        # TODO: Investigate Docling thread safety for parallel processing
        for markdown_file in markdown_files:
            try:
                chunks = self.chunk_file(markdown_file, use_docling)
                
                # Adjust chunk IDs to be globally unique
                base_id = len(all_chunks)
                for chunk in chunks:
                    chunk.chunk_id = base_id + chunk.chunk_id
                
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking {markdown_file}: {e}")
        
        # Update relationship metadata for the last chunks
        if all_chunks:
            all_chunks[-1].metadata["relationships"]["next"] = None
        
        # Optimize chunk sizes using token-aware strategies
        if self.tokenizer_config.enable_token_aware_splitting or self.tokenizer_config.enable_merge_peers:
            logger.info("Applying token-aware optimizations...")
            all_chunks = self.optimize_chunk_sizes(all_chunks)
        
        # Save all chunks
        if output_dir:
            self.save_chunks(all_chunks, output_dir)
        
        # Calculate quality statistics
        quality_scores = [chunk.metadata.get("quality_score", 0.0) for chunk in all_chunks]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Calculate token statistics
        token_counts = [chunk.metadata.get("tokens", 0) for chunk in all_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
        oversized_chunks = sum(1 for tokens in token_counts if tokens > self.max_tokens)
        undersized_chunks = sum(1 for tokens in token_counts if tokens < self.min_tokens)
        
        logger.info(f"Batch chunking completed: {len(all_chunks)} total chunks")
        logger.info(f"Average chunk quality score: {avg_quality:.3f}")
        logger.info(f"Token statistics: avg={avg_tokens:.1f}, oversized={oversized_chunks}, undersized={undersized_chunks}")
        
        # Log method distribution
        method_counts = {}
        for chunk in all_chunks:
            method = chunk.metadata.get("chunking_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        logger.info(f"Chunking method distribution: {method_counts}")
        
        return all_chunks
    
    def create_rag_dataset(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Create a RAG-compatible dataset from chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            RAG dataset dictionary
        """
        rag_data = []
        
        for chunk in chunks:
            # Extract keywords and topics for better searchability
            keywords = self._extract_keywords(chunk.text)
            
            rag_entry = {
                "id": f"chunk_{chunk.chunk_id:04d}",
                "text": chunk.text,
                "keywords": keywords,
                "metadata": {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "char_count": len(chunk.text),
                    "word_count": len(chunk.text.split()),
                    "paragraph_count": len(chunk.text.split('\n\n')),
                    **chunk.metadata
                }
            }
            rag_data.append(rag_entry)
        
        # Calculate dataset statistics
        total_chars = sum(len(chunk.text) for chunk in chunks)
        total_words = sum(len(chunk.text.split()) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        dataset = {
            "chunks": rag_data,
            "total_chunks": len(chunks),
            "statistics": {
                "total_characters": total_chars,
                "total_words": total_words,
                "average_chunk_size": round(avg_chunk_size, 2),
                "min_chunk_size": min(len(chunk.text) for chunk in chunks) if chunks else 0,
                "max_chunk_size": max(len(chunk.text) for chunk in chunks) if chunks else 0
            },
            "chunking_config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "method": "docling_hybrid" if HAS_DOCLING else "heading_based"
            },
            "version": "1.0.0",
            "created_at": json.loads(json.dumps({"timestamp": ""}))["timestamp"] or "2025-06-21"
        }
        
        return dataset
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for better searchability.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction - remove common words and extract meaningful terms
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words that are likely to be keywords
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        keywords = []
        
        for word in words:
            if word not in common_words and len(word) > 2:
                keywords.append(word)
        
        # Remove duplicates and return top 10 most frequent
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    def create_enhanced_rag_dataset(self, chunks: List[DocumentChunk], output_dir: str, enable_deduplication: bool = True, multimodal_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Create enhanced RAG datasets with multiple formats, deduplication, and multimodal support.
        
        Args:
            chunks: List of document chunks
            output_dir: Output directory for datasets
            enable_deduplication: Whether to deduplicate datasets
            multimodal_metadata: Metadata about multimodal content in the dataset
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Initialize deduplicator if enabled
        deduplicator = DatasetDeduplicator() if enable_deduplication else None
        
        # Analyze multimodal content in chunks
        multimodal_analysis = self._analyze_multimodal_content(chunks)
        
        # 1. Standard RAG JSON format with multimodal enhancements
        rag_dataset = self.create_rag_dataset(chunks)
        
        # Add multimodal metadata to the dataset
        if multimodal_metadata:
            rag_dataset["multimodal_source_metadata"] = multimodal_metadata
        rag_dataset["multimodal_analysis"] = multimodal_analysis
        
        # Deduplicate chunks if enabled
        if enable_deduplication and deduplicator:
            logger.info("🔍 Deduplicating RAG chunks...")
            original_chunks = rag_dataset["chunks"]
            deduplicated_chunks, dedup_report = deduplicator.deduplicate_comprehensive(
                original_chunks, 
                keys=['text', 'id'], 
                strategy="progressive"
            )
            rag_dataset["chunks"] = deduplicated_chunks
            rag_dataset["deduplication_report"] = dedup_report
            logger.info(f"📉 RAG dataset: {len(original_chunks)} → {len(deduplicated_chunks)} chunks")
        
        json_path = os.path.join(output_dir, "rag_dataset.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rag_dataset, f, indent=2, ensure_ascii=False)
        results["json"] = json_path
        
        # 2. JSONL format for streaming
        jsonl_path = os.path.join(output_dir, "rag_dataset.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk_data in rag_dataset["chunks"]:
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
        results["jsonl"] = jsonl_path
        
        # 3. Question-Answer format for RAG training
        qa_dataset = self._create_qa_dataset(chunks)
        
        # Deduplicate QA pairs if enabled
        if enable_deduplication and deduplicator:
            logger.info("🔍 Deduplicating QA pairs...")
            original_qa_pairs = qa_dataset["qa_pairs"]
            deduplicated_qa_pairs, qa_dedup_report = deduplicator.deduplicate_comprehensive(
                original_qa_pairs, 
                keys=['question', 'answer'], 
                strategy="progressive"
            )
            qa_dataset["qa_pairs"] = deduplicated_qa_pairs
            qa_dataset["deduplication_report"] = qa_dedup_report
            logger.info(f"📉 QA pairs: {len(original_qa_pairs)} → {len(deduplicated_qa_pairs)} pairs")
        
        qa_path = os.path.join(output_dir, "rag_qa_dataset.json")
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, indent=2, ensure_ascii=False)
        results["qa"] = qa_path
        
        # 4. Embedding-ready format
        embedding_dataset = self._create_embedding_dataset(chunks)
        
        # Deduplicate embedding entries if enabled
        if enable_deduplication and deduplicator:
            logger.info("🔍 Deduplicating embedding entries...")
            original_entries = embedding_dataset["entries"]
            deduplicated_entries, embed_dedup_report = deduplicator.deduplicate_comprehensive(
                original_entries, 
                keys=['text', 'content'], 
                strategy="progressive"
            )
            embedding_dataset["entries"] = deduplicated_entries
            embedding_dataset["deduplication_report"] = embed_dedup_report
            logger.info(f"📉 Embedding entries: {len(original_entries)} → {len(deduplicated_entries)} entries")
        
        embedding_path = os.path.join(output_dir, "embedding_dataset.json")
        with open(embedding_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_dataset, f, indent=2, ensure_ascii=False)
        results["embedding"] = embedding_path
        
        logger.info(f"✅ Enhanced RAG datasets created with deduplication and multimodal support: {list(results.keys())}")
        logger.info(f"📊 Multimodal content: {multimodal_analysis['chunks_with_tables']} table chunks, {multimodal_analysis['chunks_with_figures']} figure chunks")
        return results
    
    def _analyze_multimodal_content(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze multimodal content across all chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Analysis results with multimodal content statistics
        """
        analysis = {
            "total_chunks": len(chunks),
            "chunks_with_tables": 0,
            "chunks_with_figures": 0,
            "chunks_with_multimodal": 0,
            "multimodal_types": {
                "markdown_tables": 0,
                "csv_references": 0,
                "image_references": 0,
                "figure_references": 0
            },
            "content_analysis": {
                "table_keywords": [],
                "figure_keywords": [],
                "multimodal_chunk_ids": []
            }
        }
        
        import re
        from collections import Counter
        
        table_keywords = []
        figure_keywords = []
        
        for chunk in chunks:
            has_multimodal = False
            chunk_multimodal_types = []
            
            # Analyze tables
            # Check for markdown tables
            if re.search(r'^\|.*\|\s*$', chunk.text, re.MULTILINE):
                analysis["multimodal_types"]["markdown_tables"] += 1
                analysis["chunks_with_tables"] += 1
                chunk_multimodal_types.append("markdown_table")
                has_multimodal = True
            
            # Check for CSV references or table mentions
            if re.search(r'table|csv|\.csv', chunk.text, re.IGNORECASE):
                analysis["multimodal_types"]["csv_references"] += 1
                if "table" not in chunk_multimodal_types:
                    analysis["chunks_with_tables"] += 1
                chunk_multimodal_types.append("table_reference")
                has_multimodal = True
                
                # Extract table-related keywords
                table_words = re.findall(r'\b\w*table\w*\b|\b\w*data\w*\b|\b\w*row\w*\b|\b\w*column\w*\b', 
                                        chunk.text, re.IGNORECASE)
                table_keywords.extend(table_words)
            
            # Analyze figures
            # Check for image markdown syntax
            if re.search(r'!\[.*?\]\(.*?\.(png|jpg|jpeg|gif|svg)\)', chunk.text, re.IGNORECASE):
                analysis["multimodal_types"]["image_references"] += 1
                analysis["chunks_with_figures"] += 1
                chunk_multimodal_types.append("image_reference")
                has_multimodal = True
            
            # Check for figure mentions
            if re.search(r'figure|diagram|chart|graph|image', chunk.text, re.IGNORECASE):
                analysis["multimodal_types"]["figure_references"] += 1
                if "figure" not in chunk_multimodal_types:
                    analysis["chunks_with_figures"] += 1
                chunk_multimodal_types.append("figure_reference")
                has_multimodal = True
                
                # Extract figure-related keywords
                figure_words = re.findall(r'\b\w*figure\w*\b|\b\w*diagram\w*\b|\b\w*chart\w*\b|\b\w*graph\w*\b', 
                                         chunk.text, re.IGNORECASE)
                figure_keywords.extend(figure_words)
            
            if has_multimodal:
                analysis["chunks_with_multimodal"] += 1
                analysis["content_analysis"]["multimodal_chunk_ids"].append(chunk.chunk_id)
                
                # Add multimodal metadata to chunk if not already present
                if not hasattr(chunk, 'multimodal_types'):
                    chunk.metadata["multimodal_types"] = chunk_multimodal_types
                    chunk.metadata["has_multimodal"] = True
        
        # Add keyword analysis
        if table_keywords:
            table_counter = Counter(table_keywords)
            analysis["content_analysis"]["table_keywords"] = [word for word, count in table_counter.most_common(10)]
        
        if figure_keywords:
            figure_counter = Counter(figure_keywords)
            analysis["content_analysis"]["figure_keywords"] = [word for word, count in figure_counter.most_common(10)]
        
        # Calculate ratios
        analysis["multimodal_ratio"] = analysis["chunks_with_multimodal"] / len(chunks) if chunks else 0
        analysis["table_ratio"] = analysis["chunks_with_tables"] / len(chunks) if chunks else 0
        analysis["figure_ratio"] = analysis["chunks_with_figures"] / len(chunks) if chunks else 0
        
        return analysis
    
    def _create_qa_dataset(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Create question-answer pairs from chunks for RAG training.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            QA dataset dictionary
        """
        qa_pairs = []
        
        for chunk in chunks:
            # Generate potential questions based on content
            questions = self._generate_questions_from_chunk(chunk)
            
            for question in questions:
                qa_pairs.append({
                    "id": f"qa_{chunk.chunk_id}_{len(qa_pairs)}",
                    "question": question,
                    "answer": chunk.text,
                    "context": chunk.text,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "metadata": chunk.metadata
                })
        
        return {
            "qa_pairs": qa_pairs,
            "total_pairs": len(qa_pairs),
            "description": "Question-Answer pairs generated from document chunks for RAG training",
            "version": "1.0.0"
        }
    
    def _generate_questions_from_chunk(self, chunk: DocumentChunk) -> List[str]:
        """Generate potential questions from a chunk.
        
        Args:
            chunk: Document chunk
            
        Returns:
            List of generated questions
        """
        text = chunk.text
        questions = []
        
        # Simple question generation based on content patterns
        # This is a basic implementation - in production, use a proper NLG model
        
        # If chunk contains "what is" or definitions
        if any(phrase in text.lower() for phrase in ["what is", "definition", "means", "refers to"]):
            # Extract the term being defined
            lines = text.split('\n')
            for line in lines:
                if any(phrase in line.lower() for phrase in ["what is", "definition"]):
                    questions.append(f"What is described in this section?")
                    break
        
        # If chunk contains procedures or steps
        if any(phrase in text.lower() for phrase in ["step", "procedure", "process", "how to"]):
            questions.append("How is this process performed?")
            questions.append("What are the steps involved?")
        
        # If chunk contains features or capabilities
        if any(phrase in text.lower() for phrase in ["feature", "capability", "function", "enables"]):
            questions.append("What features are described?")
            questions.append("What capabilities are mentioned?")
        
        # If chunk contains configuration or parameters
        if any(phrase in text.lower() for phrase in ["configure", "parameter", "setting", "option"]):
            questions.append("How is this configured?")
            questions.append("What parameters are available?")
        
        # Generic fallback questions
        if not questions:
            questions.append("What information is contained in this section?")
            questions.append(f"What is discussed about {chunk.source.replace('.md', '').replace('_', ' ')}?")
        
        return questions[:3]  # Limit to 3 questions per chunk
    
    def _create_embedding_dataset(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Create embedding-ready dataset format.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Embedding dataset dictionary
        """
        embedding_data = []
        
        for chunk in chunks:
            # Create embedding-ready entries with different text representations
            embedding_data.append({
                "id": f"chunk_{chunk.chunk_id:04d}",
                "text": chunk.text,
                "title": self._extract_title(chunk.text),
                "summary": self._create_summary(chunk.text),
                "keywords": self._extract_keywords(chunk.text),
                "source": chunk.source,
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "char_count": len(chunk.text),
                    "word_count": len(chunk.text.split()),
                    **chunk.metadata
                },
                "embedding_text": self._prepare_embedding_text(chunk.text),
                "search_text": self._prepare_search_text(chunk.text)
            })
        
        return {
            "entries": embedding_data,  # Changed from "documents" to "entries" to match deduplication code
            "documents": embedding_data,  # Keep both for compatibility
            "total_documents": len(embedding_data),
            "description": "Documents prepared for embedding generation and vector search",
            "embedding_fields": ["text", "embedding_text", "search_text"],
            "version": "1.0.0"
        }
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted title
        """
        lines = text.strip().split('\n')
        # Look for the first non-empty line as title
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                return line[:100]  # Limit title length
            elif line.startswith('#'):
                return line.strip('# ').strip()[:100]
        return "Document Content"
    
    def _create_summary(self, text: str, max_length: int = 200) -> str:
        """Create a summary from text.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Text summary
        """
        # Simple extractive summary - take first few sentences
        sentences = text.split('.')
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) < max_length:
                summary += sentence + "."
            else:
                break
        return summary.strip() or text[:max_length]
    
    def _prepare_embedding_text(self, text: str) -> str:
        """Prepare text for embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Processed text for embedding
        """
        # Clean and normalize text for embedding
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with embeddings
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    def _prepare_search_text(self, text: str) -> str:
        """Prepare text for search indexing.
        
        Args:
            text: Input text
            
        Returns:
            Processed text for search
        """
        # Prepare text for full-text search
        # Extract key phrases and terms
        keywords = self._extract_keywords(text)
        title = self._extract_title(text)
        
        # Combine title, keywords, and original text
        search_components = [title] + keywords + [text]
        return ' '.join(search_components)
    
    def analyze_chunk_quality(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze the quality of generated chunks and provide statistics.
        
        Args:
            chunks: List of document chunks to analyze
            
        Returns:
            Dictionary with quality analysis results
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Calculate quality metrics
        quality_scores = [chunk.metadata.get("quality_score", 0.0) for chunk in chunks]
        sizes = [len(chunk.text) for chunk in chunks]
        word_counts = [chunk.metadata.get("word_count", 0) for chunk in chunks]
        
        # Method distribution
        methods = [chunk.metadata.get("chunking_method", "unknown") for chunk in chunks]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Content type distribution
        content_types = [chunk.metadata.get("content_type", "unknown") for chunk in chunks]
        type_counts = {}
        for content_type in content_types:
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        # Quality distribution
        high_quality = sum(1 for score in quality_scores if score >= 0.8)
        medium_quality = sum(1 for score in quality_scores if 0.5 <= score < 0.8)
        low_quality = sum(1 for score in quality_scores if score < 0.5)
        
        analysis = {
            "total_chunks": len(chunks),
            "quality_metrics": {
                "average_quality_score": sum(quality_scores) / len(quality_scores),
                "min_quality_score": min(quality_scores),
                "max_quality_score": max(quality_scores),
                "quality_distribution": {
                    "high_quality": high_quality,
                    "medium_quality": medium_quality,
                    "low_quality": low_quality
                }
            },
            "size_metrics": {
                "average_size": sum(sizes) / len(sizes),
                "min_size": min(sizes),
                "max_size": max(sizes),
                "average_word_count": sum(word_counts) / len(word_counts)
            },
            "method_distribution": method_counts,
            "content_type_distribution": type_counts,
            "recommendations": self._generate_chunking_recommendations(chunks)
        }
        
        return analysis
    
    def _generate_chunking_recommendations(self, chunks: List[DocumentChunk]) -> List[str]:
        """Generate recommendations for improving chunking quality."""
        recommendations = []
        
        quality_scores = [chunk.metadata.get("quality_score", 0.0) for chunk in chunks]
        sizes = [len(chunk.text) for chunk in chunks]
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_size = sum(sizes) / len(sizes)
        
        # Quality recommendations
        if avg_quality < 0.6:
            recommendations.append("Consider adjusting chunk size parameters to improve semantic coherence")
        
        low_quality_count = sum(1 for score in quality_scores if score < 0.5)
        if low_quality_count > len(chunks) * 0.2:  # More than 20% low quality
            recommendations.append("High percentage of low-quality chunks detected - review content structure")
        
        # Size recommendations
        if avg_size < self.semantic_min_size * 0.8:
            recommendations.append("Chunks are smaller than optimal - consider increasing minimum size")
        elif avg_size > self.semantic_max_size * 0.9:
            recommendations.append("Chunks are near maximum size - consider better splitting strategies")
        
        # Content type recommendations
        content_types = [chunk.metadata.get("content_type", "unknown") for chunk in chunks]
        if content_types.count("general_content") > len(chunks) * 0.7:
            recommendations.append("High percentage of general content - improve section detection")
        
        # Method recommendations
        methods = [chunk.metadata.get("chunking_method", "unknown") for chunk in chunks]
        if "enhanced_size_based" in methods and methods.count("enhanced_size_based") > len(chunks) * 0.5:
            recommendations.append("Frequent fallback to size-based chunking - improve heading detection")
        
        if not recommendations:
            recommendations.append("Chunking quality looks good!")
        
        return recommendations
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the configured tokenizer."""
        return {
            "tokenizer_class": type(self._tokenizer).__name__,
            "model_name": getattr(self._tokenizer, 'model_name', 'unknown'),
            "tokenizer_type": self.tokenizer_config.tokenizer_type,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "overlap_tokens": self.overlap_tokens,
            "merge_threshold": self.merge_threshold,
            "supports_advanced_tokenization": HAS_TOKENIZERS,
            "cache_enabled": self.tokenizer_config.cache_tokenizers,
            "token_aware_splitting": self.tokenizer_config.enable_token_aware_splitting,
            "merge_peers_enabled": self.tokenizer_config.enable_merge_peers
        }
    
    def align_with_embedding_model(self, embedding_model: str) -> None:
        """Reconfigure tokenizer to align with a specific embedding model."""
        logger.info(f"Aligning tokenizer with embedding model: {embedding_model}")
        
        # Update tokenizer configuration
        self.tokenizer_config.embedding_model = embedding_model
        
        # Recreate tokenizer with new configuration
        old_tokenizer = type(self._tokenizer).__name__
        self._tokenizer = self._create_tokenizer()
        new_tokenizer = type(self._tokenizer).__name__
        
        logger.info(f"Tokenizer alignment: {old_tokenizer} → {new_tokenizer}")
    
    def estimate_tokens_from_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Estimate token usage across all chunks."""
        if not chunks:
            return {"total_tokens": 0, "avg_tokens": 0, "chunks": 0}
        
        token_counts = [chunk.metadata.get("tokens", self.count_tokens(chunk.text)) for chunk in chunks]
        
        return {
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chunks": len(chunks),
            "oversized_chunks": sum(1 for t in token_counts if t > self.max_tokens),
            "undersized_chunks": sum(1 for t in token_counts if t < self.min_tokens),
            "optimal_chunks": sum(1 for t in token_counts if self.min_tokens <= t <= self.max_tokens),
            "token_efficiency": sum(1 for t in token_counts if self.min_tokens <= t <= self.max_tokens) / len(chunks)
        }
    
    def enhance_chunks_with_serialization(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Enhance chunks using the advanced serialization framework.
        
        Args:
            chunks: List of document chunks to enhance
            
        Returns:
            Enhanced chunks with improved serialization
        """
        if not self._serialization_framework:
            logger.warning("Serialization framework not available, returning original chunks")
            return chunks
        
        enhanced_chunks = []
        
        for chunk in chunks:
            try:
                enhanced_chunk = self._enhance_chunk_content(chunk)
                enhanced_chunks.append(enhanced_chunk)
            except Exception as e:
                logger.warning(f"Error enhancing chunk {chunk.chunk_id}: {e}")
                enhanced_chunks.append(chunk)  # Keep original on error
        
        logger.info(f"Enhanced {len(enhanced_chunks)} chunks with advanced serialization")
        return enhanced_chunks
    
    def _enhance_chunk_content(self, chunk: DocumentChunk) -> DocumentChunk:
        """Enhance individual chunk content using serialization framework.
        
        Args:
            chunk: Original chunk
            
        Returns:
            Enhanced chunk with improved content serialization
        """
        # Extract multimodal content from chunk text
        multimodal_items = self._extract_multimodal_content(chunk.text)
        
        if not multimodal_items:
            return chunk  # No multimodal content, return original
        
        # Serialize multimodal content using the framework
        serialized_content = self._serialization_framework.serialize_multimodal_content(
            multimodal_items, 
            context=f"Source: {chunk.source}"
        )
        
        # Create enhanced chunk with improved content
        enhanced_chunk = DocumentChunk(
            text=serialized_content,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            metadata={
                **chunk.metadata,
                "enhanced_serialization": True,
                "multimodal_items": len(multimodal_items),
                "original_text": chunk.text
            }
        )
        
        return enhanced_chunk
    
    def _extract_multimodal_content(self, text: str) -> List[Dict[str, Any]]:
        """Extract multimodal content items from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of multimodal content items
        """
        items = []
        
        # Extract tables
        table_pattern = r'^\|.*\|\s*$'
        table_matches = re.finditer(table_pattern, text, re.MULTILINE)
        
        for i, match in enumerate(table_matches):
            table_text = match.group()
            items.append({
                "type": "table",
                "data": table_text,
                "metadata": {
                    "index": i,
                    "caption": f"Table {i + 1}",
                    "position": match.start()
                }
            })
        
        # Extract images/figures
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        image_matches = re.finditer(image_pattern, text)
        
        for i, match in enumerate(image_matches):
            alt_text = match.group(1)
            src = match.group(2)
            items.append({
                "type": "picture",
                "data": src,
                "metadata": {
                    "index": i,
                    "description": alt_text or f"Figure {i + 1}",
                    "src": src,
                    "position": match.start()
                }
            })
        
        # Extract lists
        list_pattern = r'^[-*+]\s+.*(?:\n^[-*+]\s+.*)*'
        list_matches = re.finditer(list_pattern, text, re.MULTILINE)
        
        for i, match in enumerate(list_matches):
            list_text = match.group()
            items.append({
                "type": "list",
                "data": list_text.split('\n'),
                "metadata": {
                    "index": i,
                    "list_type": "bullet",
                    "position": match.start()
                }
            })
        
        # Sort by position to maintain document order
        items.sort(key=lambda x: x['metadata']['position'])
        
        return items
    
    def create_multimodal_rag_dataset(
        self, 
        chunks: List[DocumentChunk], 
        output_dir: str,
        enhance_serialization: bool = True
    ) -> Dict[str, str]:
        """Create RAG dataset with enhanced multimodal content serialization.
        
        Args:
            chunks: List of document chunks
            output_dir: Output directory
            enhance_serialization: Whether to enhance chunks with advanced serialization
            
        Returns:
            Dictionary mapping format names to file paths
        """
        # Enhance chunks with serialization if requested
        if enhance_serialization and self._serialization_framework:
            logger.info("Enhancing chunks with advanced serialization framework")
            enhanced_chunks = self.enhance_chunks_with_serialization(chunks)
        else:
            enhanced_chunks = chunks
        
        # Create enhanced RAG datasets with multimodal support
        return self.create_enhanced_rag_dataset(
            enhanced_chunks, 
            output_dir, 
            enable_deduplication=True,
            multimodal_metadata={
                "serialization_framework": "Flow4 Advanced Serialization",
                "enhanced_content": enhance_serialization,
                "supported_formats": self._get_supported_serialization_formats()
            }
        )
    
    def _get_supported_serialization_formats(self) -> Dict[str, List[str]]:
        """Get supported serialization formats from the framework."""
        if self._serialization_framework:
            return self._serialization_framework.get_supported_formats()
        else:
            return {
                "tables": ["markdown"],
                "pictures": ["placeholder"],
                "lists": ["bullet"],
                "text": ["plain"]
            }
    
    # ===== SOPHISTICATED CONTEXT ENRICHMENT METHODS =====
    
    def _extract_docling_document_metadata(self, doc) -> Dict[str, Any]:
        """Extract rich metadata from DoclingDocument.
        
        Args:
            doc: DoclingDocument object
            
        Returns:
            Dictionary with document-level metadata
        """
        if not HAS_DOCLING or not doc:
            return {}
        
        metadata = {
            "document_type": "docling_document",
            "has_structure": False,
            "elements_count": 0,
            "page_count": 0,
            "tables_count": 0,
            "figures_count": 0,
            "headings_count": 0,
            "text_elements_count": 0,
            "document_hierarchy": {}
        }
        
        try:
            # Extract document structure information
            if hasattr(doc, 'body') and doc.body:
                metadata["has_structure"] = True
                metadata["elements_count"] = len(doc.body.elements) if hasattr(doc.body, 'elements') else 0
                
                # Count different element types and build hierarchy
                hierarchy_builder = []
                if hasattr(doc.body, 'elements'):
                    for i, element in enumerate(doc.body.elements):
                        element_info = {
                            "index": i,
                            "type": str(element.label) if hasattr(element, 'label') else "unknown",
                            "text_preview": (element.text[:100] + "...") if hasattr(element, 'text') and len(element.text) > 100 else (element.text if hasattr(element, 'text') else "")
                        }
                        hierarchy_builder.append(element_info)
                        
                        if hasattr(element, 'label'):
                            element_type = str(element.label).lower()
                            if 'table' in element_type:
                                metadata["tables_count"] += 1
                            elif 'figure' in element_type or 'image' in element_type:
                                metadata["figures_count"] += 1
                            elif 'title' in element_type or 'heading' in element_type:
                                metadata["headings_count"] += 1
                            elif 'text' in element_type or 'paragraph' in element_type:
                                metadata["text_elements_count"] += 1
                
                metadata["document_hierarchy"]["elements"] = hierarchy_builder
            
            # Extract page information
            if hasattr(doc, 'pages') and doc.pages:
                metadata["page_count"] = len(doc.pages)
                metadata["document_hierarchy"]["pages"] = [
                    {"page_index": i, "element_count": len(page.elements) if hasattr(page, 'elements') else 0}
                    for i, page in enumerate(doc.pages)
                ]
            
            # Extract document properties
            if hasattr(doc, 'props') and doc.props:
                if hasattr(doc.props, '__dict__'):
                    for key, value in doc.props.__dict__.items():
                        try:
                            json.dumps(value)  # Test serializability
                            metadata[f"prop_{key}"] = value
                        except (TypeError, ValueError):
                            metadata[f"prop_{key}_type"] = str(type(value).__name__)
            
        except Exception as e:
            logger.debug(f"Error extracting document metadata: {e}")
            metadata["extraction_error"] = str(e)
        
        return metadata
    
    def _build_document_hierarchy(self, doc) -> Dict[str, Any]:
        """Build hierarchical structure map of the document.
        
        Args:
            doc: DoclingDocument object
            
        Returns:
            Dictionary representing document hierarchy
        """
        if not HAS_DOCLING or not doc:
            return {}
        
        hierarchy = {
            "headings": [],
            "sections": {},
            "element_hierarchy": {},
            "parent_child_map": {},
            "heading_levels": {},
            "semantic_groups": {},
            "cross_references": []
        }
        
        try:
            if hasattr(doc, 'body') and doc.body and hasattr(doc.body, 'elements'):
                current_heading = None
                current_level = 0
                current_section = None
                
                for i, element in enumerate(doc.body.elements):
                    element_id = f"element_{i}"
                    element_text = element.text if hasattr(element, 'text') else ""
                    element_type = str(element.label) if hasattr(element, 'label') else "unknown"
                    
                    # Store comprehensive element information
                    hierarchy["element_hierarchy"][element_id] = {
                        "index": i,
                        "type": element_type,
                        "text": element_text,
                        "text_preview": element_text[:150] + "..." if len(element_text) > 150 else element_text,
                        "parent": current_heading,
                        "section": current_section,
                        "level": current_level,
                        "word_count": len(element_text.split()) if element_text else 0,
                        "has_references": self._detect_element_references(element_text),
                        "content_type": self._classify_element_content(element_text, element_type)
                    }
                    
                    # Track headings and build hierarchy
                    if 'title' in element_type.lower() or 'heading' in element_type.lower():
                        level = self._extract_heading_level_enhanced(element_type, element_text)
                        
                        heading_info = {
                            "id": element_id,
                            "text": element_text,
                            "level": level,
                            "index": i,
                            "children": [],
                            "section_elements": [],
                            "content_summary": self._summarize_heading_content(element_text)
                        }
                        
                        hierarchy["headings"].append(heading_info)
                        hierarchy["heading_levels"][element_id] = level
                        
                        # Update current context
                        current_heading = element_id
                        current_level = level
                        current_section = element_text[:50]  # First 50 chars as section identifier
                        
                        # Build parent-child relationships
                        if len(hierarchy["headings"]) > 1:
                            parent_heading = self._find_parent_heading_enhanced(
                                hierarchy["headings"], level
                            )
                            if parent_heading:
                                hierarchy["parent_child_map"][element_id] = parent_heading["id"]
                                parent_heading["children"].append(element_id)
                    
                    # Group semantic content
                    semantic_type = self._classify_element_content(element_text, element_type)
                    if semantic_type not in hierarchy["semantic_groups"]:
                        hierarchy["semantic_groups"][semantic_type] = []
                    hierarchy["semantic_groups"][semantic_type].append(element_id)
                    
                    # Detect cross-references
                    refs = self._extract_element_cross_references(element_text, i)
                    hierarchy["cross_references"].extend(refs)
                
                # Build section maps
                for heading in hierarchy["headings"]:
                    section_elements = self._collect_section_elements(
                        heading, hierarchy["element_hierarchy"]
                    )
                    hierarchy["sections"][heading["id"]] = {
                        "heading": heading,
                        "elements": section_elements,
                        "element_count": len(section_elements),
                        "total_words": sum(
                            hierarchy["element_hierarchy"][elem_id]["word_count"]
                            for elem_id in section_elements
                            if elem_id in hierarchy["element_hierarchy"]
                        )
                    }
        
        except Exception as e:
            logger.debug(f"Error building document hierarchy: {e}")
            hierarchy["build_error"] = str(e)
        
        return hierarchy
    
    def _extract_heading_level_enhanced(self, element_type: str, text: str) -> int:
        """Extract heading level with enhanced detection."""
        # Try to extract level from element type
        level_match = re.search(r'(\d+)', element_type)
        if level_match:
            return int(level_match.group(1))
        
        # Try to extract from text (markdown-style)
        if text.startswith('#'):
            return min(6, len(text) - len(text.lstrip('#')))
        
        # Analyze text patterns for implicit levels
        if text.isupper() and len(text) < 50:
            return 1  # All caps short text = main title
        elif text.istitle() and len(text) < 100:
            return 2  # Title case medium text = subtitle
        
        # Default levels based on element type
        type_lower = element_type.lower()
        if 'title' in type_lower and 'sub' not in type_lower:
            return 1
        elif 'subtitle' in type_lower or 'sub' in type_lower:
            return 2
        elif 'heading' in type_lower:
            return 3
        
        return 4  # Default level
    
    def _find_parent_heading_enhanced(self, headings: List[Dict], current_level: int) -> Optional[Dict]:
        """Find the parent heading with enhanced logic."""
        for heading in reversed(headings[:-1]):  # Exclude the current heading
            if heading["level"] < current_level:
                return heading
        return None
    
    def _classify_element_content(self, text: str, element_type: str) -> str:
        """Classify element content type with enhanced detection."""
        if not text:
            return "empty"
        
        text_lower = text.lower()
        type_lower = element_type.lower()
        
        # Type-based classification
        if 'table' in type_lower:
            return 'table'
        elif 'figure' in type_lower or 'image' in type_lower:
            return 'figure'
        elif 'list' in type_lower:
            return 'list'
        elif 'title' in type_lower or 'heading' in type_lower:
            return 'heading'
        
        # Content-based classification
        if re.search(r'\|.*\|', text):  # Markdown table
            return 'table'
        elif 'procedure' in text_lower or 'step' in text_lower:
            return 'procedure'
        elif 'configure' in text_lower or 'setting' in text_lower:
            return 'configuration'
        elif 'class ' in text and ('attribute' in text_lower or 'method' in text_lower):
            return 'class_definition'
        elif 'enum ' in text:
            return 'enum_definition'
        elif len(text.split('.')) > 3:  # Multiple sentences
            return 'paragraph'
        elif len(text.split()) < 10:  # Short text
            return 'caption'
        else:
            return 'general_content'
    
    def _detect_element_references(self, text: str) -> List[str]:
        """Detect references in element text."""
        if not text:
            return []
        
        references = []
        ref_patterns = [
            r'see\s+(?:also\s+)?(?:section|chapter|figure|table)\s+([\\w\\d\\.]+)',
            r'refer\s+to\s+(?:section|chapter|figure|table)\s+([\\w\\d\\.]+)',
            r'(?:fig|figure)\s*\\.?\\s*(\\d+)',
            r'(?:table)\s*\\.?\\s*(\\d+)',
            r'(?:section|chapter)\s*\\.?\\s*(\\d+(?:\\.\\d+)*)',
        ]
        
        for pattern in ref_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append(match.group(0))
        
        return references
    
    def _summarize_heading_content(self, text: str) -> str:
        """Create a summary of heading content."""
        if len(text) <= 100:
            return text
        
        # Extract key terms
        words = text.split()
        important_words = [
            word for word in words
            if len(word) > 3 and word.lower() not in {
                'the', 'and', 'for', 'with', 'that', 'this', 'from', 'they', 'have', 'been'
            }
        ]
        
        return ' '.join(important_words[:10])
    
    def _extract_element_cross_references(self, text: str, element_index: int) -> List[Dict[str, Any]]:
        """Extract cross-references from element text."""
        if not text:
            return []
        
        cross_refs = []
        ref_patterns = [
            (r'see\s+(?:also\s+)?(?:section|chapter|figure|table)\s+([\\w\\d\\.]+)', 'explicit_reference'),
            (r'refer\s+to\s+(?:section|chapter|figure|table)\s+([\\w\\d\\.]+)', 'explicit_reference'),
            (r'(?:fig|figure)\s*\\.?\\s*(\\d+)', 'figure_reference'),
            (r'(?:table)\s*\\.?\\s*(\\d+)', 'table_reference'),
            (r'(?:section|chapter)\s*\\.?\\s*(\\d+(?:\\.\\d+)*)', 'section_reference'),
        ]
        
        for pattern, ref_type in ref_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cross_refs.append({
                    "source_element": element_index,
                    "reference_text": match.group(0),
                    "reference_id": match.group(1) if match.groups() else match.group(0),
                    "reference_type": ref_type,
                    "position": match.start()
                })
        
        return cross_refs
    
    def _collect_section_elements(self, heading: Dict[str, Any], 
                                 element_hierarchy: Dict[str, Any]) -> List[str]:
        """Collect all elements belonging to a section."""
        section_elements = []
        
        heading_index = heading["index"]
        heading_level = heading["level"]
        
        # Find all elements after this heading until next same-or-higher level heading
        for element_id, element_info in element_hierarchy.items():
            element_index = element_info["index"]
            
            if element_index > heading_index:
                # Check if this element belongs to this section
                element_type = element_info["type"].lower()
                
                # Stop if we hit a heading of same or higher level
                if ('title' in element_type or 'heading' in element_type):
                    element_level = self._extract_heading_level_enhanced(
                        element_info["type"], element_info["text"]
                    )
                    if element_level <= heading_level:
                        break
                
                section_elements.append(element_id)
        
        return section_elements
    
    def _enhanced_contextualize_with_hierarchy(self, chunker, chunk, doc, 
                                              hierarchy_map: Dict[str, Any]) -> str:
        """Enhanced contextualize method with hierarchical context enrichment.
        
        Args:
            chunker: HybridChunker instance
            chunk: Document chunk
            doc: DoclingDocument
            hierarchy_map: Document hierarchy information
            
        Returns:
            Enhanced context-enriched text
        """
        if not HAS_DOCLING:
            return chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        try:
            # Get standard Docling contextualization
            base_context = chunker.contextualize(chunk=chunk)
            
            # Add hierarchical context enhancement
            hierarchical_context = self._build_hierarchical_context(
                chunk, doc, hierarchy_map
            )
            
            # Add semantic context
            semantic_context = self._build_semantic_context(
                chunk, doc, hierarchy_map
            )
            
            # Combine all contexts intelligently
            context_parts = []
            
            if hierarchical_context:
                context_parts.append(f"Document Context: {hierarchical_context}")
            
            if semantic_context:
                context_parts.append(f"Content Type: {semantic_context}")
            
            if context_parts:
                enhanced_prefix = " | ".join(context_parts)
                enhanced_context = f"{enhanced_prefix}\n\n{base_context}"
            else:
                enhanced_context = base_context
            
            # Add relationship context if available
            relationship_context = self._build_relationship_context(chunk, hierarchy_map)
            if relationship_context:
                enhanced_context = f"{enhanced_context}\n\nRelated: {relationship_context}"
            
            return enhanced_context
        
        except Exception as e:
            logger.debug(f"Error in enhanced contextualization: {e}")
            return chunk.text if hasattr(chunk, 'text') else str(chunk)
    
    def _build_hierarchical_context(self, chunk, doc, 
                                   hierarchy_map: Dict[str, Any]) -> str:
        """Build hierarchical context information for chunk."""
        context_parts = []
        
        try:
            # Find the chunk's position in document hierarchy
            chunk_position = self._find_chunk_position_enhanced(chunk, doc, hierarchy_map)
            
            if chunk_position:
                # Build parent chain
                parent_chain = self._build_parent_chain_enhanced(
                    chunk_position, hierarchy_map
                )
                
                if parent_chain:
                    context_parts.extend(parent_chain)
                
                # Add section context
                section_context = self._find_current_section_enhanced(
                    chunk_position, hierarchy_map
                )
                if section_context:
                    context_parts.append(f"Section: {section_context}")
        
        except Exception as e:
            logger.debug(f"Error building hierarchical context: {e}")
        
        return " > ".join(context_parts) if context_parts else ""
    
    def _build_semantic_context(self, chunk, doc, 
                               hierarchy_map: Dict[str, Any]) -> str:
        """Build semantic context for the chunk."""
        try:
            if hasattr(chunk, 'text') and chunk.text:
                chunk_type = self._classify_element_content(
                    chunk.text, 
                    str(chunk.label) if hasattr(chunk, 'label') else ""
                )
                
                # Add content complexity assessment
                complexity = self._assess_content_complexity(chunk.text)
                
                return f"{chunk_type} ({complexity} complexity)"
        
        except Exception as e:
            logger.debug(f"Error building semantic context: {e}")
        
        return ""
    
    def _build_relationship_context(self, chunk, hierarchy_map: Dict[str, Any]) -> str:
        """Build relationship context for the chunk."""
        try:
            relationships = []
            
            # Check for cross-references
            if hasattr(chunk, 'text') and chunk.text:
                refs = self._detect_element_references(chunk.text)
                if refs:
                    relationships.append(f"References: {', '.join(refs[:3])}")
            
            # Check semantic group membership
            chunk_position = self._find_chunk_position_enhanced(chunk, None, hierarchy_map)
            if chunk_position:
                element_info = hierarchy_map.get("element_hierarchy", {}).get(chunk_position, {})
                content_type = element_info.get("content_type", "")
                
                # Find related elements of same type
                semantic_groups = hierarchy_map.get("semantic_groups", {})
                if content_type in semantic_groups:
                    related_count = len(semantic_groups[content_type])
                    if related_count > 1:
                        relationships.append(f"{related_count} related {content_type} items")
            
            return ", ".join(relationships) if relationships else ""
        
        except Exception as e:
            logger.debug(f"Error building relationship context: {e}")
        
        return ""
    
    def _find_chunk_position_enhanced(self, chunk, doc, 
                                     hierarchy_map: Dict[str, Any]) -> Optional[str]:
        """Find the chunk's position in the document structure with enhanced matching."""
        try:
            if hasattr(chunk, 'text') and chunk.text:
                chunk_text_start = chunk.text[:200].strip()  # First 200 chars for matching
                
                # Try exact matching first
                for element_id, element_info in hierarchy_map.get("element_hierarchy", {}).items():
                    element_text = element_info.get("text", "").strip()
                    if element_text and chunk_text_start.startswith(element_text[:200]):
                        return element_id
                
                # Try fuzzy matching
                best_match = None
                best_score = 0
                
                for element_id, element_info in hierarchy_map.get("element_hierarchy", {}).items():
                    element_text = element_info.get("text", "").strip()
                    if element_text:
                        # Calculate simple similarity score
                        similarity = self._calculate_text_similarity(
                            chunk_text_start, element_text[:200]
                        )
                        if similarity > best_score and similarity > 0.7:
                            best_score = similarity
                            best_match = element_id
                
                return best_match
        
        except Exception as e:
            logger.debug(f"Error finding chunk position: {e}")
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union) if union else 0.0
        
        except:
            return 0.0
    
    def _build_parent_chain_enhanced(self, element_id: str, 
                                    hierarchy_map: Dict[str, Any]) -> List[str]:
        """Build enhanced chain of parent elements for context."""
        chain = []
        
        try:
            current_id = element_id
            visited = set()
            max_depth = 10  # Prevent infinite loops
            
            while current_id and current_id not in visited and len(chain) < max_depth:
                visited.add(current_id)
                
                element_info = hierarchy_map.get("element_hierarchy", {}).get(current_id, {})
                if not element_info:
                    break
                
                parent_id = element_info.get("parent")
                if parent_id:
                    parent_info = hierarchy_map.get("element_hierarchy", {}).get(parent_id, {})
                    if parent_info and parent_info.get("text"):
                        parent_text = parent_info["text"]
                        # Truncate long parent text
                        if len(parent_text) > 50:
                            parent_text = parent_text[:47] + "..."
                        chain.insert(0, parent_text)
                    current_id = parent_id
                else:
                    break
        
        except Exception as e:
            logger.debug(f"Error building parent chain: {e}")
        
        return chain
    
    def _find_current_section_enhanced(self, element_id: str, 
                                      hierarchy_map: Dict[str, Any]) -> Optional[str]:
        """Find the current section for the element with enhanced detection."""
        try:
            sections = hierarchy_map.get("sections", {})
            
            # Find which section this element belongs to
            for section_id, section_info in sections.items():
                if element_id in section_info.get("elements", []):
                    heading = section_info.get("heading", {})
                    section_title = heading.get("text", "")
                    if len(section_title) > 60:
                        section_title = section_title[:57] + "..."
                    return section_title
            
            # Fallback: find nearest heading
            element_info = hierarchy_map.get("element_hierarchy", {}).get(element_id, {})
            element_index = element_info.get("index", -1)
            
            if element_index >= 0:
                # Find the most recent heading before this element
                best_heading = None
                best_distance = float('inf')
                
                for heading in hierarchy_map.get("headings", []):
                    heading_index = heading.get("index", -1)
                    if 0 <= heading_index < element_index:
                        distance = element_index - heading_index
                        if distance < best_distance:
                            best_distance = distance
                            best_heading = heading
                
                if best_heading:
                    heading_text = best_heading.get("text", "")
                    if len(heading_text) > 60:
                        heading_text = heading_text[:57] + "..."
                    return heading_text
        
        except Exception as e:
            logger.debug(f"Error finding current section: {e}")
        
        return None
    
    def _assess_content_complexity(self, text: str) -> str:
        """Assess content complexity with enhanced metrics."""
        if not text:
            return "empty"
        
        words = text.split()
        sentences = text.split('.')
        
        if not words:
            return "empty"
        
        # Calculate metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        technical_terms = len(re.findall(r'\\b[A-Z]{2,}\\b|\\b\\w+[A-Z]\\w*\\b', text))
        
        # Calculate complexity score
        complexity_score = 0
        
        if avg_word_length > 6:
            complexity_score += 2
        elif avg_word_length > 4:
            complexity_score += 1
        
        if avg_sentence_length > 25:
            complexity_score += 2
        elif avg_sentence_length > 15:
            complexity_score += 1
        
        if technical_terms > len(words) * 0.1:  # >10% technical terms
            complexity_score += 2
        elif technical_terms > 0:
            complexity_score += 1
        
        # Classify complexity
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"