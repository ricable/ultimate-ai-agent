"""Optimized high-performance document chunking using Docling and fallback methods."""

import os
import json
import re
import threading
import gc
import weakref
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock, RLock
from functools import lru_cache, partial
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from collections import defaultdict, Counter
import hashlib

from ..utils.config import DoclingConfig
from ..utils.logging import get_logger

from ..utils.deduplication import DatasetDeduplicator

logger = get_logger(__name__)

# Try to import docling for advanced chunking
try:
    from docling.document_converter import DocumentConverter as DoclingConverter
    from docling.chunking import HybridChunker
    from docling.datamodel.settings import settings as docling_settings
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
    logger.info("Docling available for advanced hybrid chunking with custom tokenizers")
except ImportError as e:
    logger.warning(f"Docling advanced features not available: {e}. Falling back to simple chunking.")
    HAS_DOCLING = False
    HAS_ADVANCED_TOKENIZERS = False
    DoclingConverter = None
    HybridChunker = None
    docling_settings = None
    HuggingFaceTokenizer = None
    ChunkingDocSerializer = None
    ChunkingSerializerProvider = None
    MarkdownTableSerializer = None
    MarkdownParams = None
    AutoTokenizer = None
    tiktoken = None


def make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON serializable format."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        # For complex objects, extract basic info
        try:
            result = {
                "_type": type(obj).__name__,
                "_module": getattr(type(obj), '__module__', 'unknown')
            }
            # Add simple attributes
            for key, value in obj.__dict__.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    result[key] = value
                elif isinstance(value, (list, tuple)) and len(value) < 10:  # Avoid huge lists
                    result[key] = [make_json_serializable(item) for item in value[:5]]  # Limit to first 5
                elif hasattr(value, 'id') and hasattr(value, '__str__'):
                    result[key] = {"id": str(getattr(value, 'id', '')), "str": str(value)[:100]}
            return result
        except Exception:
            return {"_type": type(obj).__name__, "_serialization_error": True}
    else:
        # For other types, try to get a string representation
        try:
            return str(obj)[:200]  # Limit string length
        except Exception:
            return f"<{type(obj).__name__} object>"


@dataclass
class PerformanceMetrics:
    """Track chunker performance metrics."""
    
    total_files: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    memory_peak: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    
    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def chunks_per_second(self) -> float:
        return self.total_chunks / self.processing_time if self.processing_time > 0 else 0.0


class ResourceManager:
    """Manages resource allocation and cleanup for the chunker."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self._resources = weakref.WeakSet()
        self._lock = Lock()
        self._memory_usage = 0
    
    def register_resource(self, resource):
        """Register a resource for tracking."""
        with self._lock:
            self._resources.add(resource)
    
    def cleanup_resources(self):
        """Force cleanup of all tracked resources."""
        with self._lock:
            for resource in list(self._resources):
                try:
                    if hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    elif hasattr(resource, 'close'):
                        resource.close()
                except Exception as e:
                    logger.debug(f"Error cleaning up resource: {e}")
            gc.collect()
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < self.max_memory_mb
        except ImportError:
            return True  # Can't check without psutil
    
    @contextmanager
    def memory_guard(self):
        """Context manager for memory-aware operations."""
        try:
            yield
        finally:
            if not self.check_memory_usage():
                logger.warning("High memory usage detected, triggering cleanup")
                self.cleanup_resources()
                gc.collect()


class AdvancedTokenizerCache:
    """Thread-safe cache for advanced tokenizers with intelligent management."""
    
    def __init__(self, max_size: int = 5, ttl_seconds: int = 1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = RLock()
        self._thread_local = threading.local()
    
    @lru_cache(maxsize=32)
    def get_tiktoken_tokenizer(self, encoding_name: str = "cl100k_base"):
        """Get cached tiktoken tokenizer."""
        if not HAS_ADVANCED_TOKENIZERS or not tiktoken:
            return None
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.debug(f"Failed to load tiktoken tokenizer {encoding_name}: {e}")
            return None
    
    def get_huggingface_tokenizer(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Get cached HuggingFace tokenizer with thread safety."""
        if not HAS_ADVANCED_TOKENIZERS or not AutoTokenizer:
            return None
        
        # Check thread-local cache first
        cache_key = f"hf_{model_name}"
        if hasattr(self._thread_local, cache_key):
            return getattr(self._thread_local, cache_key)
        
        with self._lock:
            current_time = time.time()
            
            # Check global cache
            if cache_key in self._cache:
                tokenizer, created_time = self._cache[cache_key]
                if current_time - created_time < self.ttl_seconds:
                    self._access_times[cache_key] = current_time
                    setattr(self._thread_local, cache_key, tokenizer)
                    return tokenizer
                else:
                    del self._cache[cache_key]
                    del self._access_times[cache_key]
            
            # Create new tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Manage cache size
                if len(self._cache) >= self.max_size:
                    self._evict_oldest()
                
                self._cache[cache_key] = (tokenizer, current_time)
                self._access_times[cache_key] = current_time
                setattr(self._thread_local, cache_key, tokenizer)
                
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load HuggingFace tokenizer {model_name}: {e}")
                return None
    
    def _evict_oldest(self):
        """Remove the least recently used tokenizer."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times, key=self._access_times.get)
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            # Clear lru_cache
            self.get_tiktoken_tokenizer.cache_clear()


class DoclingConverterCache:
    """Thread-safe cache for Docling converters with resource management."""
    
    def __init__(self, max_size: int = 3, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = RLock()
        self._thread_local = threading.local()
    
    def get_converter(self, config_hash: str, config: DoclingConfig) -> Optional['DoclingConverter']:
        """Get a converter from cache or create new one."""
        if not HAS_DOCLING:
            return None
        
        # Check thread-local cache first
        if hasattr(self._thread_local, 'converter'):
            return self._thread_local.converter
        
        with self._lock:
            current_time = time.time()
            
            # Check if cached converter exists and is still valid
            if config_hash in self._cache:
                converter, created_time = self._cache[config_hash]
                if current_time - created_time < self.ttl_seconds:
                    self._access_times[config_hash] = current_time
                    # Store in thread-local for fast access
                    self._thread_local.converter = converter
                    return converter
                else:
                    # Expired, remove from cache
                    del self._cache[config_hash]
                    del self._access_times[config_hash]
            
            # Create new converter
            try:
                converter = self._create_converter(config)
                
                # Manage cache size
                if len(self._cache) >= self.max_size:
                    self._evict_oldest()
                
                self._cache[config_hash] = (converter, current_time)
                self._access_times[config_hash] = current_time
                
                # Store in thread-local
                self._thread_local.converter = converter
                
                return converter
            except Exception as e:
                logger.error(f"Failed to create Docling converter: {e}")
                return None
    
    def _create_converter(self, config: DoclingConfig) -> 'DoclingConverter':
        """Create a new Docling converter with optimized settings."""
        try:
            # Create converter with basic settings to avoid API issues
            converter = DoclingConverter()
            
            # Try to configure basic settings if available
            try:
                if hasattr(converter, 'config') and hasattr(config, 'with_accelerator'):
                    converter.config.with_accelerator = config.with_accelerator
                    
                # Try to set table mode if supported
                if hasattr(converter, 'config') and hasattr(config, 'table_mode'):
                    converter.config.table_mode = getattr(config, 'table_mode', 'fast')
                    
            except Exception as e:
                logger.debug(f"Could not configure converter optimizations: {e}")
            
            return converter
            
        except Exception as e:
            logger.error(f"Failed to create basic Docling converter: {e}")
            raise
    
    def _evict_oldest(self):
        """Remove the least recently used converter."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times, key=self._access_times.get)
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self):
        """Clear all cached converters."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            # Clear thread-local storage
            if hasattr(self._thread_local, 'converter'):
                del self._thread_local.converter


class ChunkerCache:
    """Thread-safe cache for chunkers and serializers."""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._chunker_cache = {}
        self._serializer_cache = {}
        self._lock = RLock()
    
    def get_chunker(self, chunker_type: str, tokenizer=None) -> Optional['HybridChunker']:
        """Get cached chunker instance with optional custom tokenizer."""
        if not HAS_DOCLING or chunker_type != 'hybrid':
            return None
        
        cache_key = f"{chunker_type}_{id(tokenizer) if tokenizer else 'default'}"
        
        with self._lock:
            if cache_key in self._chunker_cache:
                return self._chunker_cache[cache_key]
            
            try:
                # Create chunker with optional custom tokenizer
                if tokenizer and HAS_ADVANCED_TOKENIZERS:
                    chunker = HybridChunker(tokenizer=tokenizer)
                else:
                    chunker = HybridChunker()
                
                if len(self._chunker_cache) < self.max_size:
                    self._chunker_cache[cache_key] = chunker
                return chunker
            except Exception as e:
                logger.error(f"Failed to create chunker: {e}")
                return None
    
    def get_serializer(self, serializer_type: str = "markdown") -> Optional:
        """Get cached serializer instance."""
        if not HAS_DOCLING:
            return None
        
        with self._lock:
            if serializer_type in self._serializer_cache:
                return self._serializer_cache[serializer_type]
            
            try:
                if serializer_type == "markdown" and MarkdownTableSerializer:
                    # Use simplified serializer creation to avoid API issues
                    try:
                        serializer = ChunkingSerializerProvider.get_serializer(
                            MarkdownTableSerializer
                        )
                    except TypeError:
                        # Fallback if API has changed
                        serializer = MarkdownTableSerializer()
                else:
                    serializer = None
                
                if serializer and len(self._serializer_cache) < self.max_size:
                    self._serializer_cache[serializer_type] = serializer
                return serializer
            except Exception as e:
                logger.debug(f"Serializer creation failed, using fallback: {e}")
                # Return None to use fallback chunking methods
                return None
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._chunker_cache.clear()
            self._serializer_cache.clear()


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
        """Convert chunk to dictionary representation with JSON serializable metadata."""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": make_json_serializable(self.metadata)
        }


class OptimizedDocumentChunker:
    """High-performance unified document chunker with advanced optimization features."""
    
    def __init__(self, config: Optional[DoclingConfig] = None, max_workers: int = 4, 
                 enable_caching: bool = True, memory_limit_mb: int = 2048,
                 enable_advanced_tokenizers: bool = True):
        """Initialize the optimized document chunker.
        
        Args:
            config: Chunking configuration
            max_workers: Maximum number of worker threads
            enable_caching: Enable intelligent caching
            memory_limit_mb: Memory limit in MB for resource management
            enable_advanced_tokenizers: Enable advanced tokenizer support
        """
        self.config = config or DoclingConfig()
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.enable_advanced_tokenizers = enable_advanced_tokenizers and HAS_ADVANCED_TOKENIZERS
        
        # Enhanced chunking parameters with adaptive sizing
        self.semantic_min_size = max(self.config.min_chunk_size, 600)
        self.semantic_max_size = min(self.config.max_chunk_size, 2000)
        self.semantic_overlap = min(self.config.chunk_overlap, 100)
        
        # Adaptive sizing parameters
        self.adaptive_sizing = True
        self.content_type_sizes = {
            'class_definition': {'min': 800, 'target': 1200, 'max': 1800},
            'enum_definition': {'min': 400, 'target': 800, 'max': 1200}, 
            'technical_procedure': {'min': 600, 'target': 1000, 'max': 1500},
            'table_content': {'min': 300, 'target': 600, 'max': 900},
            'list_content': {'min': 400, 'target': 700, 'max': 1000},
            'general_content': {'min': 600, 'target': 900, 'max': 1300}
        }
        
        # Performance optimization components
        self.resource_manager = ResourceManager(memory_limit_mb)
        self.metrics = PerformanceMetrics()
        
        # Caching systems (only if enabled)
        if self.enable_caching:
            self.converter_cache = DoclingConverterCache()
            self.chunker_cache = ChunkerCache()
            self.tokenizer_cache = AdvancedTokenizerCache()
            self._config_hash = self._compute_config_hash()
        else:
            self.converter_cache = None
            self.chunker_cache = None
            self.tokenizer_cache = None
            self._config_hash = None
        
        # Thread safety
        self._processing_lock = RLock()
        self._stats_lock = Lock()
        
        # Legacy converter reference (for backwards compatibility)
        self._docling_converter = None
        
        # Register with resource manager
        self.resource_manager.register_resource(self)
    
    def _compute_config_hash(self) -> str:
        """Compute a hash of the current configuration for caching."""
        config_str = f"{self.config.chunk_size}_{self.config.chunk_overlap}_{self.config.extract_tables}_{self.config.extract_figures}_{self.config.with_accelerator}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_docling_converter(self):
        """Get or create optimized Docling DocumentConverter instance."""
        if not HAS_DOCLING:
            return None
        
        # Use cached converter if available
        if self.enable_caching and self.converter_cache:
            converter = self.converter_cache.get_converter(self._config_hash, self.config)
            if converter:
                with self._stats_lock:
                    self.metrics.cache_hits += 1
                return converter
            else:
                with self._stats_lock:
                    self.metrics.cache_misses += 1
        
        # Fallback to legacy single instance
        if self._docling_converter is None:
            try:
                self._docling_converter = DoclingConverter()
                # Configure for performance
                if hasattr(self._docling_converter, 'config'):
                    self._docling_converter.config.with_accelerator = self.config.with_accelerator
            except Exception as e:
                logger.error(f"Failed to create Docling converter: {e}")
                return None
        
        return self._docling_converter
    
    def _get_optimized_tokenizer(self):
        """Get optimized tokenizer based on configuration."""
        if not self.enable_advanced_tokenizers or not self.tokenizer_cache:
            return None
        
        # Try different tokenizers in order of preference
        tokenizer = None
        
        # 1. Try tiktoken for fast performance
        tokenizer = self.tokenizer_cache.get_tiktoken_tokenizer(self.config.tokenizer)
        if tokenizer:
            return tokenizer
        
        # 2. Try HuggingFace tokenizer for better accuracy
        if hasattr(self.config, 'hf_tokenizer_model') and HuggingFaceTokenizer is not None:
            tokenizer = self.tokenizer_cache.get_huggingface_tokenizer(self.config.hf_tokenizer_model)
            if tokenizer:
                return HuggingFaceTokenizer(tokenizer)
        
        return None
    
    def _get_hybrid_chunker(self) -> Optional['HybridChunker']:
        """Get optimized HybridChunker instance with custom tokenizer."""
        if not HAS_DOCLING:
            return None
        
        tokenizer = self._get_optimized_tokenizer() if self.enable_advanced_tokenizers else None
        
        if self.enable_caching and self.chunker_cache:
            chunker = self.chunker_cache.get_chunker('hybrid', tokenizer)
            if chunker:
                with self._stats_lock:
                    self.metrics.cache_hits += 1
                return chunker
            else:
                with self._stats_lock:
                    self.metrics.cache_misses += 1
        
        # Fallback to direct creation
        try:
            if tokenizer:
                return HybridChunker(tokenizer=tokenizer)
            else:
                return HybridChunker()
        except Exception as e:
            logger.error(f"Failed to create HybridChunker: {e}")
            return None
    
    def _get_optimized_serializer(self):
        """Get optimized serializer for chunk processing."""
        if not self.enable_caching or not self.chunker_cache:
            return None
        
        return self.chunker_cache.get_serializer("markdown")
    
    def cleanup(self):
        """Clean up resources and caches."""
        try:
            if self.enable_caching:
                if self.converter_cache:
                    self.converter_cache.clear()
                if self.chunker_cache:
                    self.chunker_cache.clear()
                if self.tokenizer_cache:
                    self.tokenizer_cache.clear()
            
            self.resource_manager.cleanup_resources()
            
            # Clear legacy converter
            self._docling_converter = None
            
            logger.debug("Optimized chunker cleanup completed")
        except Exception as e:
            logger.error(f"Error during chunker cleanup: {e}")
    
    def _chunk_with_optimized_docling(self, markdown_path: str) -> List[DocumentChunk]:
        """Optimized Docling chunking with advanced features and caching."""
        if not HAS_DOCLING:
            raise ImportError("Docling is required for advanced chunking.")
        
        start_time = time.time()
        logger.info(f"Chunking {markdown_path} with optimized Docling hybrid chunker")
        
        with self.resource_manager.memory_guard():
            try:
                # Get optimized converter and chunker
                converter = self._get_docling_converter()
                if not converter:
                    raise Exception("Failed to get Docling converter")
                
                chunker = self._get_hybrid_chunker()
                if not chunker:
                    raise Exception("Failed to get HybridChunker")
                
                # Convert the markdown file to a Docling document
                logger.info(f"Converting {markdown_path} to Docling document format")
                result = converter.convert(source=markdown_path)
                logger.info(f"Docling conversion completed for {markdown_path}")
                doc = result.document
                
                if not doc:
                    raise Exception("Document conversion returned empty result")
                
                # Get optimized serializer if available
                serializer = self._get_optimized_serializer()
                
                # Generate chunks using the optimized hybrid chunker
                logger.info("Generating chunks with optimized HybridChunker")
                chunk_iter = chunker.chunk(dl_doc=doc)
                logger.info("Chunk iterator created, starting chunk processing")
                
                # Batch process chunks for better memory efficiency
                chunks = []
                source_name = os.path.basename(markdown_path)
                chunk_batch = []
                batch_size = 50  # Process chunks in batches
                
                for i, chunk in enumerate(chunk_iter):
                    try:
                        if i % 10 == 0:  # Log every 10 chunks to avoid spam
                            logger.info(f"Processing chunk {i}")
                        
                        # Get context-enriched text with error handling
                        enriched_text = None
                        try:
                            enriched_text = chunker.contextualize(chunk=chunk)
                        except Exception as ctx_error:
                            logger.debug(f"Contextualization failed for chunk {i}: {ctx_error}")
                        
                        # Use enriched text if available, otherwise fall back to original
                        chunk_text = enriched_text if enriched_text else chunk.text
                        
                        if not chunk_text or len(chunk_text.strip()) < 10:
                            logger.debug(f"Skipping empty/short chunk {i}")
                            continue
                        
                        # Extract metadata with improved error handling
                        metadata = self._extract_optimized_docling_metadata(
                            chunk, i, enriched_text, serializer
                        )
                        
                        chunk_batch.append(DocumentChunk(
                            text=chunk_text,
                            chunk_id=i,
                            source=source_name,
                            metadata=metadata
                        ))
                        
                        # Process batch when it reaches the limit
                        if len(chunk_batch) >= batch_size:
                            chunks.extend(chunk_batch)
                            chunk_batch = []
                            
                            # Check memory usage periodically
                            if not self.resource_manager.check_memory_usage():
                                logger.warning("High memory usage during chunking, forcing cleanup")
                                gc.collect()
                    
                    except Exception as chunk_error:
                        logger.warning(f"Error processing chunk {i}: {chunk_error}")
                        with self._stats_lock:
                            self.metrics.errors += 1
                        continue
                
                # Add remaining chunks from the last batch
                if chunk_batch:
                    chunks.extend(chunk_batch)
                
                # Update relationships efficiently
                self._update_chunk_relationships(chunks)
                
                processing_time = time.time() - start_time
                logger.info(f"Created {len(chunks)} chunks using optimized Docling HybridChunker in {processing_time:.2f}s")
                
                # Update metrics
                with self._stats_lock:
                    self.metrics.total_chunks += len(chunks)
                    self.metrics.processing_time += processing_time
                
                return chunks
            
            except Exception as e:
                logger.error(f"Error in optimized Docling chunking: {str(e)}")
                with self._stats_lock:
                    self.metrics.errors += 1
                raise
    
    def _extract_optimized_docling_metadata(self, chunk, chunk_index: int, 
                                          enriched_text: Optional[str], 
                                          serializer=None) -> Dict[str, Any]:
        """Extract metadata from Docling chunk with optimized error handling."""
        metadata = {
            "original_text": chunk.text,
            "enriched_text": enriched_text,
            "chunking_method": "docling_hybrid_optimized",
            "chunk_index": chunk_index,
            "relationships": {
                "previous": chunk_index - 1 if chunk_index > 0 else None,
                "next": chunk_index + 1,  # Will be updated later
            }
        }
        
        # Add advanced tokenizer information if available
        if self.enable_advanced_tokenizers:
            tokenizer = self._get_optimized_tokenizer()
            if tokenizer:
                try:
                    if hasattr(tokenizer, 'encode'):
                        # Handle long sequences by truncating for token counting
                        text_to_encode = chunk.text[:2000] if len(chunk.text) > 2000 else chunk.text
                        tokens = tokenizer.encode(text_to_encode)
                        # Estimate full token count if text was truncated
                        if len(chunk.text) > 2000:
                            ratio = len(chunk.text) / len(text_to_encode)
                            estimated_tokens = int(len(tokens) * ratio)
                            metadata["token_count"] = estimated_tokens
                            metadata["token_count_estimated"] = True
                        else:
                            metadata["token_count"] = len(tokens)
                            metadata["token_count_estimated"] = False
                        metadata["tokenizer_type"] = type(tokenizer).__name__
                except Exception as e:
                    logger.debug(f"Token counting failed: {e}")
        
        # Add serializer information if available
        if serializer:
            try:
                metadata["serializer_type"] = type(serializer).__name__
            except Exception:
                pass
        
        # Optimized metadata extraction
        if hasattr(chunk, 'meta') and chunk.meta:
            try:
                # Use a more efficient approach for metadata extraction
                if hasattr(chunk.meta, '__dict__'):
                    meta_dict = chunk.meta.__dict__
                    # Only extract serializable metadata to avoid overhead
                    for key, value in meta_dict.items():
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            try:
                                metadata[f"docling_{key}"] = value
                            except Exception:
                                continue
                        elif value is not None:
                            metadata[f"docling_{key}_type"] = str(type(value).__name__)
                else:
                    metadata["docling_meta_type"] = str(type(chunk.meta).__name__)
            except Exception as e:
                logger.debug(f"Optimized metadata extraction failed: {e}")
                metadata["docling_meta_available"] = True
        
        return metadata
    
    def _update_chunk_relationships(self, chunks: List[DocumentChunk]):
        """Efficiently update chunk relationships."""
        if not chunks:
            return
        
        # Update the "next" relationship for the last chunk
        if len(chunks) > 0:
            chunks[-1].metadata["relationships"]["next"] = None
        
        # Optionally update all relationships if needed
        for i, chunk in enumerate(chunks):
            chunk.metadata["relationships"] = {
                "previous": i - 1 if i > 0 else None,
                "next": i + 1 if i < len(chunks) - 1 else None
            }
    
    def _extract_metadata_from_markdown(self, markdown_content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter metadata from markdown."""
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
    
    def _chunk_by_headings(self, markdown_path: str) -> List[DocumentChunk]:
        """Split markdown content into chunks based on headings with enhanced semantic awareness."""
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
    
    def _clean_technical_text(self, text: str) -> str:
        """Clean malformed table data and formatting issues from technical documentation."""
        # Fix broken table syntax specific to technical docs
        text = re.sub(r',\s*=\s*\.\s*,\s*LTE\s*=\s*\.', '', text)
        text = re.sub(r'Level,\s*=\s*\.\s*Level,\s*LTE\s*=\s*[A-Za-z]+\.', '', text)
        text = re.sub(r'Formula Status,\s*=\s*\.\s*Formula Status,\s*LTE\s*=\s*[A-Z]+\.', '', text)
        text = re.sub(r'Beneficial Trend,\s*=\s*\.\s*Beneficial Trend,\s*LTE\s*=\s*[A-Za-z]+', '', text)
        
        # Remove repetitive metadata patterns that cause duplication
        text = re.sub(r'(Feature Identity[^.=]*=\s*[^.=]+\.?\s*){2,}', lambda m: m.group(0).split('.')[0] + '.', text)
        text = re.sub(r'(Value Package Name[^.=]*=\s*[^.=]+\.?\s*){2,}', lambda m: m.group(0).split('.')[0] + '.', text)
        text = re.sub(r'(Access Type[^.=]*=\s*[^.=]+\.?\s*){2,}', lambda m: m.group(0).split('.')[0] + '.', text)
        text = re.sub(r'(Licensing[^.=]*=\s*[^.=]+\.?\s*){2,}', lambda m: m.group(0).split('.')[0] + '.', text)
        text = re.sub(r'(Node Type[^.=]*=\s*[^.=]+\.?\s*){2,}', lambda m: m.group(0).split('.')[0] + '.', text)
        
        # Clean broken enum patterns like "1, OFF   PDCCH beamforming = SLOW"
        text = re.sub(r'(\d+),?\s+(OFF|ON)\s+([^=]+=\s*[A-Z]+)', r'\1. \2: \3', text)
        
        # Fix malformed attribute patterns
        text = re.sub(r'([A-Za-z]+)\s*\[\d+\.\.\d+\]\s+(noNotification|readOnly|restricted)\s*,\s*([A-Za-z\s]+)\s*=\s*([^\n]+)', 
                     r'\1 \3 = \4', text)
        
        # Clean excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r' {3,}', '  ', text)  # Reduce excessive spaces to max 2
        
        # Remove trailing periods in lists and improve formatting
        text = re.sub(r'\.(\s*\n\s*[0-9]+[\.\)])', r'\1', text)
        
        return text.strip()
    
    def _classify_section_type(self, content: str) -> str:
        """Classify the type of content section for better chunking strategy."""
        content_lower = content.lower()
        
        # Check for specific technical patterns first
        if 'class ' in content and ('attributes' in content_lower or 'dependencies' in content_lower):
            return 'class_definition'
        elif 'enum ' in content and any(word in content for word in ['OFF', 'ON', 'SLOW', 'FAST', 'NORMAL']):
            return 'enum_definition'
        elif re.search(r'pm[A-Z][a-zA-Z]+', content):  # PM counter patterns
            return 'pm_counter_list'
        elif 'kpi' in content_lower or 'performance' in content_lower:
            return 'kpi_definition'
        elif 'formula' in content_lower and any(counter in content for counter in ['pm', 'counter']):
            return 'kpi_calculation'
        elif 'feature' in content_lower and 'identity' in content_lower:
            return 'feature_description'
        elif 'value package' in content_lower and 'access type' in content_lower:
            return 'feature_metadata'
        elif re.search(r'\d+\s+[A-Z][A-Za-z\s]+Overview', content):  # "3.27 O&M Security Overview"
            return 'feature_overview'
        elif any(proc in content_lower for proc in ['procedure', 'process', 'step']):
            return 'procedure'
        elif any(cfg in content_lower for cfg in ['configure', 'configuration', 'parameter']):
            return 'configuration'
        elif re.search(r'Appendix [A-Z]', content):
            return 'appendix'
        elif 'summary' in content_lower and len(content) < 1000:
            return 'summary'
        elif re.search(r'^\d+(\.\d+)*\s+[A-Z]', content):  # Numbered section
            return 'numbered_section'
        else:
            return 'general_content'
    
    def _chunk_with_semantic_awareness(self, content: str, source_name: str, doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create optimally sized chunks with adaptive semantic boundaries."""
        sections = self._extract_semantic_sections(content)
        
        # Use adaptive combining for optimal chunk distribution
        combined_chunks = self._combine_sections_optimally(sections)
        
        # Convert to DocumentChunk objects with enhanced metadata
        chunks = []
        for chunk_id, chunk_content in enumerate(combined_chunks):
            # Determine the predominant content type for this chunk
            chunk_type = self._classify_section_type(chunk_content)
            
            # Create enhanced chunk with all the improved metadata
            chunk = self._create_enhanced_chunk(
                chunk_id, chunk_content, source_name, doc_metadata,
                "adaptive_semantic_aware"
            )
            
            # Add adaptive sizing information to metadata
            adaptive_sizes = self._get_adaptive_chunk_sizes(chunk_type, chunk_content)
            chunk.metadata["adaptive_sizing"] = {
                "content_type": chunk_type,
                "target_size": adaptive_sizes['target_size'],
                "size_efficiency": len(chunk_content) / adaptive_sizes['target_size'],
                "within_optimal_range": adaptive_sizes['min_size'] <= len(chunk_content) <= adaptive_sizes['max_size']
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _extract_semantic_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract semantic sections with enhanced boundary detection and context preservation."""
        sections = []
        lines = content.split('\n')
        current_section = []
        current_headers = []
        section_type = 'general'
        
        # Enhanced boundary detection patterns
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        definition_pattern = re.compile(r'^\s*(class|enum|interface|struct|typedef)\s+(\w+)', re.IGNORECASE)
        procedure_pattern = re.compile(r'^\s*(\d+\.\s+|Step\s+\d+|Procedure)', re.IGNORECASE)
        table_pattern = re.compile(r'^\s*\|.*\|\s*$')
        code_block_pattern = re.compile(r'^\s*```')
        
        in_code_block = False
        code_block_lang = None
        table_start = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track code blocks to avoid breaking them
            if code_block_pattern.match(line):
                if not in_code_block:
                    in_code_block = True
                    code_block_lang = stripped[3:].strip() if len(stripped) > 3 else 'unknown'
                else:
                    in_code_block = False
                    code_block_lang = None
            
            # Enhanced semantic boundary detection
            should_break = False
            new_section_type = section_type
            new_headers = current_headers.copy()
            
            if not in_code_block:  # Don't break inside code blocks
                # Heading boundaries (stronger signal)
                heading_match = heading_pattern.match(line)
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2)
                    should_break = True
                    new_headers = [f"H{level}: {title}"]
                    new_section_type = self._classify_section_type(title)
                
                # Technical definition boundaries
                elif definition_pattern.match(line):
                    def_match = definition_pattern.match(line)
                    def_type = def_match.group(1).lower()
                    def_name = def_match.group(2)
                    should_break = True
                    new_headers.append(f"{def_type.title()}: {def_name}")
                    new_section_type = f"{def_type}_definition"
                
                # Procedure/step boundaries
                elif procedure_pattern.match(line) and len(current_section) > 10:
                    should_break = True
                    new_headers.append(f"Procedure: {stripped[:50]}...")
                    new_section_type = 'procedure'
                
                # Table boundaries
                elif table_pattern.match(line) and not table_start:
                    table_start = True
                    if len(current_section) > 5:  # Don't break for small sections
                        should_break = True
                        new_headers.append("Table Content")
                        new_section_type = 'table_content'
                elif not table_pattern.match(line) and table_start:
                    table_start = False
            
            # Execute section break if needed
            if should_break and current_section:
                # Calculate semantic cohesion score for the section
                section_content = '\n'.join(current_section)
                cohesion_score = self._calculate_semantic_cohesion(section_content)
                
                sections.append({
                    'content': section_content,
                    'headers': current_headers.copy(),
                    'type': section_type,
                    'start_line': i - len(current_section),
                    'end_line': i - 1,
                    'cohesion_score': cohesion_score,
                    'line_count': len(current_section),
                    'char_count': len(section_content)
                })
                
                # Start new section
                current_section = [line]
                current_headers = new_headers
                section_type = new_section_type
            else:
                current_section.append(line)
        
        # Add final section with enhanced metadata
        if current_section:
            section_content = '\n'.join(current_section)
            cohesion_score = self._calculate_semantic_cohesion(section_content)
            
            sections.append({
                'content': section_content,
                'headers': current_headers.copy(),
                'type': section_type,
                'start_line': len(lines) - len(current_section),
                'end_line': len(lines) - 1,
                'cohesion_score': cohesion_score,
                'line_count': len(current_section),
                'char_count': len(section_content)
            })
        
        return sections
    
    def _split_large_section(self, content: str) -> List[str]:
        """Split large sections while preserving semantic meaning and technical context."""
        chunks = []
        
        # First try to split by logical boundaries (double newlines, major sections)
        logical_parts = re.split(r'\n\s*\n\s*', content)
        current_chunk = ""
        
        for part in logical_parts:
            part = part.strip()
            if not part:
                continue
                
            # If adding this part would exceed size, finalize current chunk
            if current_chunk and len(current_chunk + '\n\n' + part) > self.semantic_max_size:
                if len(current_chunk) >= self.semantic_min_size:
                    chunks.append(current_chunk)
                    current_chunk = part
                else:
                    # Current chunk too small, try sentence-level splitting
                    combined = current_chunk + '\n\n' + part
                    sentence_chunks = self._split_by_sentences(combined)
                    chunks.extend(sentence_chunks[:-1])  # Add all but last
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + part
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, content: str) -> List[str]:
        """Split content by sentences with technical content awareness."""
        # Improved sentence splitting that handles technical abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z]|[0-9]+[\.\)])'
        sentences = re.split(sentence_pattern, content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence exceeds size
            potential_chunk = current_chunk + (' ' if current_chunk else '') + sentence
            
            if len(potential_chunk) <= self.semantic_max_size:
                current_chunk = potential_chunk
            else:
                # Sentence would make chunk too large
                if current_chunk and len(current_chunk) >= self.semantic_min_size:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Force include even if over size to maintain meaning
                    current_chunk = potential_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_by_enhanced_size(self, content: str, source_name: str, doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Enhanced size-based chunking with sentence boundary awareness."""
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
                potential_chunk = current_chunk + ('\n\n' if current_chunk else '') + section_content
                
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
        # Estimate token count more accurately using advanced tokenizers if available
        token_count = len(content.split()) * 1.3  # Rough fallback estimate
        
        if self.enable_advanced_tokenizers and self.tokenizer_cache:
            tokenizer = self._get_optimized_tokenizer()
            if tokenizer:
                try:
                    if hasattr(tokenizer, 'encode'):
                        # Handle long sequences by truncating for token counting
                        text_to_encode = content[:2000] if len(content) > 2000 else content
                        tokens = tokenizer.encode(text_to_encode)
                        # Estimate full token count if text was truncated
                        if len(content) > 2000:
                            ratio = len(content) / len(text_to_encode)
                            token_count = int(len(tokens) * ratio)
                        else:
                            token_count = len(tokens)
                    elif hasattr(tokenizer, 'tokenize'):
                        # Handle long sequences for tokenize method too
                        text_to_tokenize = content[:2000] if len(content) > 2000 else content
                        tokens = tokenizer.tokenize(text_to_tokenize)
                        # Estimate full token count if text was truncated
                        if len(content) > 2000:
                            ratio = len(content) / len(text_to_tokenize)
                            token_count = int(len(tokens) * ratio)
                        else:
                            token_count = len(tokens)
                except Exception as e:
                    logger.debug(f"Token counting failed, using fallback: {e}")
        
        # Enhanced content type classification and analysis
        content_type = self._classify_section_type(content)
        content_analysis = self._analyze_content_characteristics(content)
        semantic_features = self._extract_semantic_features(content)
        
        metadata = {
            "original_text": content,
            "enriched_text": content,  # Could be enhanced with context in the future
            "chunking_method": method,
            "chunk_index": chunk_id,
            "content_type": content_type,
            "content_analysis": content_analysis,
            "semantic_features": semantic_features,
            "tokens": token_count,
            "char_count": len(content),
            "word_count": len(content.split()),
            "quality_score": self._calculate_chunk_quality_score(content, content_type),
            "cohesion_score": self._calculate_semantic_cohesion(content),
            "completeness_score": self._calculate_completeness_score(content),
            "technical_density": self._calculate_technical_density(content),
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
        """Calculate an enhanced quality score for the chunk based on various factors."""
        score = 0.0
        
        # Enhanced size scoring with optimal ranges based on validation findings
        content_length = len(content)
        if 600 <= content_length <= 1200:  # Optimal range from validation
            score += 0.4  # Higher weight for optimal size
        elif 400 <= content_length < 600 or 1200 < content_length <= 1800:
            score += 0.3  # Good size
        elif content_length < 400:
            score += 0.1  # Too small penalty
        else:
            score += 0.2  # Too large but acceptable
        
        # Enhanced boundary detection scoring
        semantic_boundaries = 0
        # Check for natural section breaks
        if re.search(r'^#+\s+', content, re.MULTILINE):  # Headings
            semantic_boundaries += 1
        if re.search(r'^\s*(class|enum|interface|function)\s+\w+', content, re.MULTILINE):  # Definitions
            semantic_boundaries += 1
        if re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):  # Numbered lists
            semantic_boundaries += 1
        
        boundary_score = min(0.25, semantic_boundaries * 0.1)
        score += boundary_score
        
        # Enhanced completeness scoring
        completeness = 0
        if content.strip().endswith(('.', '!', '?', ':', ';')):
            completeness += 0.1
        # Check for complete code blocks
        if content.count('```') % 2 == 0 and content.count('```') > 0:
            completeness += 0.1
        # Check for complete table structures
        if content.count('|') >= 4 and '\n' in content:  # Table indicators
            completeness += 0.05
        score += completeness
        
        # Enhanced content type scoring with technical document focus
        if content_type in ['class_definition', 'enum_definition', 'technical_procedure']:
            score += 0.15
        elif content_type in ['table_content', 'list_content']:
            score += 0.1
        
        # Enhanced technical content scoring
        technical_patterns = {
            'definitions': r'\b(class|enum|interface|struct|typedef)\b',
            'procedures': r'\b(configure|procedure|step|install|setup)\b',
            'parameters': r'\b(parameter|attribute|property|setting|option)\b',
            'technical_terms': r'\b(LTE|eNodeB|MME|HSS|PCRF|SGW|PGW)\b'  # Domain-specific
        }
        
        technical_score = 0
        for pattern_type, pattern in technical_patterns.items():
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            technical_score += min(0.05, matches * 0.02)
        
        score += min(0.2, technical_score)
        
        return min(1.0, score)  # Cap at 1.0
    
    def _calculate_semantic_cohesion(self, content: str) -> float:
        """Calculate semantic cohesion score for a content section."""
        if not content.strip():
            return 0.0
        
        score = 0.0
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return 0.5  # Minimal content
        
        # Keyword consistency scoring
        all_words = []
        for line in lines:
            words = re.findall(r'\b\w{3,}\b', line.lower())
            all_words.extend(words)
        
        if len(all_words) > 0:
            word_counts = Counter(all_words)
            # Calculate keyword density
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            keyword_density = repeated_words / len(word_counts) if word_counts else 0
            score += min(0.4, keyword_density)
        
        # Structural consistency
        structural_patterns = 0
        if len([line for line in lines if re.match(r'^\s*[-*+]', line)]) > 1:  # List items
            structural_patterns += 1
        if len([line for line in lines if re.match(r'^\s*\d+\.', line)]) > 1:  # Numbered lists
            structural_patterns += 1
        if content.count('```') >= 2:  # Code blocks
            structural_patterns += 1
        
        score += min(0.3, structural_patterns * 0.1)
        
        # Length consistency (avoiding too short or too long outliers)
        line_lengths = [len(line) for line in lines]
        if line_lengths:
            avg_length = sum(line_lengths) / len(line_lengths)
            consistency = 1 - (max(line_lengths) - min(line_lengths)) / (avg_length + 1)
            score += min(0.3, consistency * 0.3)
        
        return min(1.0, score)
    
    def _analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics for enhanced metadata."""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        analysis = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "average_line_length": sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0,
            "has_code_blocks": '```' in content,
            "has_tables": '|' in content and content.count('|') >= 4,
            "has_lists": bool(re.search(r'^\s*[-*+]\s|^\s*\d+\.\s', content, re.MULTILINE)),
            "has_headings": bool(re.search(r'^#+\s', content, re.MULTILINE)),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "sentence_count": len([s for s in re.split(r'[.!?]+', content) if s.strip()]),
        }
        
        # Structural analysis
        analysis["structural_complexity"] = sum([
            analysis["has_code_blocks"],
            analysis["has_tables"], 
            analysis["has_lists"],
            analysis["has_headings"]
        ]) / 4.0
        
        return analysis
    
    def _extract_semantic_features(self, content: str) -> Dict[str, Any]:
        """Extract semantic features for better content understanding."""
        features = {
            "contains_definitions": bool(re.search(r'\b(class|enum|interface|struct|typedef)\s+\w+', content, re.IGNORECASE)),
            "contains_procedures": bool(re.search(r'\b(step|procedure|configure|install|setup)\b', content, re.IGNORECASE)),
            "contains_technical_terms": bool(re.search(r'\b(LTE|eNodeB|MME|HSS|PCRF|SGW|PGW|parameter|attribute)\b', content, re.IGNORECASE)),
            "contains_numbers": bool(re.search(r'\b\d+\b', content)),
            "contains_references": bool(re.search(r'\b(see|refer|section|chapter|figure|table)\s+\d+', content, re.IGNORECASE)),
            "contains_examples": bool(re.search(r'\b(example|for instance|e\.g\.|such as)\b', content, re.IGNORECASE)),
        }
        
        # Calculate semantic density
        semantic_indicators = sum(features.values())
        features["semantic_density"] = semantic_indicators / len(features)
        
        return features
    
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate how complete/self-contained the content appears."""
        score = 0.0
        
        # Check for complete sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if sentences:
            complete_sentences = sum(1 for s in sentences if len(s) > 10 and not s.endswith(('and', 'or', 'but', 'because')))
            sentence_completeness = complete_sentences / len(sentences)
            score += 0.4 * sentence_completeness
        
        # Check for complete structures
        if content.count('```') % 2 == 0:  # Complete code blocks
            score += 0.2
        
        # Check for balanced markers
        balanced_markers = 0
        for open_char, close_char in [('(', ')'), ('[', ']'), ('{', '}')]:
            if content.count(open_char) == content.count(close_char):
                balanced_markers += 1
        score += 0.2 * (balanced_markers / 3.0)
        
        # Check for proper endings
        content_stripped = content.strip()
        if content_stripped.endswith(('.', '!', '?', ':', ';')):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_technical_density(self, content: str) -> float:
        """Calculate the density of technical content."""
        words = re.findall(r'\b\w+\b', content.lower())
        if not words:
            return 0.0
        
        technical_terms = {
            'lte', 'enodeb', 'mme', 'hss', 'pcrf', 'sgw', 'pgw',
            'class', 'enum', 'interface', 'struct', 'typedef',
            'parameter', 'attribute', 'configure', 'procedure',
            'algorithm', 'protocol', 'function', 'method',
            'database', 'server', 'client', 'api', 'sdk'
        }
        
        technical_count = sum(1 for word in words if word in technical_terms)
        return technical_count / len(words)
    
    def _get_adaptive_chunk_sizes(self, content_type: str, content: str) -> Dict[str, int]:
        """Get adaptive chunk sizes based on content type and characteristics."""
        if not self.adaptive_sizing or content_type not in self.content_type_sizes:
            return {
                'min_size': self.semantic_min_size,
                'target_size': (self.semantic_min_size + self.semantic_max_size) // 2,
                'max_size': self.semantic_max_size
            }
        
        base_sizes = self.content_type_sizes[content_type]
        
        # Adjust based on content characteristics
        content_analysis = self._analyze_content_characteristics(content)
        adjustment_factor = 1.0
        
        # Adjust for structural complexity
        if content_analysis.get('structural_complexity', 0) > 0.5:
            adjustment_factor *= 1.2  # Allow larger chunks for complex content
        
        # Adjust for technical density
        technical_density = self._calculate_technical_density(content)
        if technical_density > 0.3:
            adjustment_factor *= 1.1  # Technical content gets slightly larger chunks
        
        # Apply adjustments
        return {
            'min_size': int(base_sizes['min'] * adjustment_factor),
            'target_size': int(base_sizes['target'] * adjustment_factor),
            'max_size': int(base_sizes['max'] * adjustment_factor)
        }
    
    def _should_split_section(self, section_content: str, content_type: str) -> bool:
        """Determine if a section should be split based on adaptive sizing."""
        sizes = self._get_adaptive_chunk_sizes(content_type, section_content)
        content_length = len(section_content)
        
        # Use content-specific max size instead of global max
        return content_length > sizes['max_size']
    
    def _combine_sections_optimally(self, sections: List[Dict[str, Any]]) -> List[str]:
        """Combine sections optimally based on adaptive sizing and semantic coherence."""
        combined_chunks = []
        current_chunk = ""
        current_type = "general_content"
        
        for section in sections:
            section_content = section['content']
            section_type = section.get('type', 'general_content')
            cohesion_score = section.get('cohesion_score', 0.5)
            
            # Get adaptive sizes for current content type
            sizes = self._get_adaptive_chunk_sizes(section_type, section_content)
            
            # Decide whether to combine or split
            if not current_chunk:
                # Start new chunk
                current_chunk = section_content
                current_type = section_type
            elif len(current_chunk + '\n\n' + section_content) <= sizes['max_size']:
                # Can combine - check if it makes sense semantically
                if section_type == current_type or cohesion_score > 0.7:
                    current_chunk += '\n\n' + section_content
                else:
                    # Different content types, finalize current chunk
                    if len(current_chunk) >= self._get_adaptive_chunk_sizes(current_type, current_chunk)['min_size']:
                        combined_chunks.append(current_chunk)
                    current_chunk = section_content
                    current_type = section_type
            else:
                # Cannot combine, finalize current chunk
                if len(current_chunk) >= self._get_adaptive_chunk_sizes(current_type, current_chunk)['min_size']:
                    combined_chunks.append(current_chunk)
                
                # Check if new section needs splitting
                if self._should_split_section(section_content, section_type):
                    split_chunks = self._adaptive_split_section(section_content, section_type)
                    combined_chunks.extend(split_chunks)
                    current_chunk = ""
                else:
                    current_chunk = section_content
                    current_type = section_type
        
        # Don't forget the last chunk
        if current_chunk:
            min_size = self._get_adaptive_chunk_sizes(current_type, current_chunk)['min_size']
            if len(current_chunk) >= min_size:
                combined_chunks.append(current_chunk)
            elif combined_chunks:
                # Combine with previous chunk if too small
                combined_chunks[-1] += '\n\n' + current_chunk
            else:
                # Even if small, keep it if it's the only content
                combined_chunks.append(current_chunk)
        
        return combined_chunks
    
    def _adaptive_split_section(self, content: str, content_type: str) -> List[str]:
        """Split a section adaptively based on content type and characteristics."""
        sizes = self._get_adaptive_chunk_sizes(content_type, content)
        target_size = sizes['target_size']
        max_size = sizes['max_size']
        
        # Try different splitting strategies based on content type
        if content_type in ['class_definition', 'enum_definition']:
            return self._split_by_definitions(content, target_size, max_size)
        elif content_type == 'technical_procedure':
            return self._split_by_procedure_steps(content, target_size, max_size)
        elif content_type == 'table_content':
            return self._split_by_table_sections(content, target_size, max_size)
        else:
            return self._split_by_semantic_boundaries(content, target_size, max_size)
    
    def _split_by_definitions(self, content: str, target_size: int, max_size: int) -> List[str]:
        """Split content by definition boundaries."""
        # Split by class/enum definitions
        definition_pattern = r'(?=^\s*(class|enum|interface|struct|typedef)\s+\w+)'
        parts = re.split(definition_pattern, content, flags=re.MULTILINE)
        return self._combine_parts_to_target_size(parts, target_size, max_size)
    
    def _split_by_procedure_steps(self, content: str, target_size: int, max_size: int) -> List[str]:
        """Split content by procedure steps."""
        # Split by numbered steps or procedure markers
        step_pattern = r'(?=^\s*(\d+\.\s+|Step\s+\d+|Procedure))'
        parts = re.split(step_pattern, content, flags=re.MULTILINE)
        return self._combine_parts_to_target_size(parts, target_size, max_size)
    
    def _split_by_table_sections(self, content: str, target_size: int, max_size: int) -> List[str]:
        """Split content by table sections."""
        # Split by table boundaries (empty lines between tables)
        parts = [part.strip() for part in content.split('\n\n') if part.strip()]
        return self._combine_parts_to_target_size(parts, target_size, max_size)
    
    def _split_by_semantic_boundaries(self, content: str, target_size: int, max_size: int) -> List[str]:
        """Split content by semantic boundaries like paragraphs and sections."""
        # Split by double newlines (paragraph boundaries)
        parts = [part.strip() for part in content.split('\n\n') if part.strip()]
        return self._combine_parts_to_target_size(parts, target_size, max_size)
    
    def _combine_parts_to_target_size(self, parts: List[str], target_size: int, max_size: int) -> List[str]:
        """Combine parts to achieve target size without exceeding max size."""
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if not part.strip():
                continue
            
            if not current_chunk:
                current_chunk = part
            elif len(current_chunk + '\n\n' + part) <= max_size:
                current_chunk += '\n\n' + part
            else:
                # Current chunk would be too large, finalize it
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_file(self, markdown_path: str, use_docling: bool = True) -> List[DocumentChunk]:
        """Chunk a single markdown file with optimized processing."""
        if use_docling and HAS_DOCLING:
            return self._chunk_with_optimized_docling(markdown_path)
        else:
            return self._chunk_by_headings(markdown_path)
    
    def batch_chunk(
        self, 
        markdown_files: List[str], 
        output_dir: str,
        num_workers: Optional[int] = None,
        use_docling: bool = True,
        batch_size: int = 10
    ) -> List[DocumentChunk]:
        """Optimized parallel chunking of multiple markdown files."""
        if num_workers is None:
            num_workers = self.max_workers
        
        start_time = time.time()
        logger.info(f"Starting optimized batch chunking of {len(markdown_files)} files with {num_workers} workers")
        
        all_chunks = []
        total_errors = 0
        
        with self.resource_manager.memory_guard():
            if num_workers == 1 or not use_docling:
                # Sequential processing for better Docling compatibility
                logger.info("Using sequential processing for better Docling compatibility")
                all_chunks, total_errors = self._process_files_sequential(
                    markdown_files, use_docling, batch_size
                )
            else:
                # Parallel processing with careful resource management
                logger.info(f"Using parallel processing with {num_workers} workers")
                all_chunks, total_errors = self._process_files_parallel(
                    markdown_files, num_workers, use_docling, batch_size
                )
        
        # Post-process all chunks efficiently
        if all_chunks:
            self._post_process_batch_chunks(all_chunks)
        
        # Save all chunks efficiently
        if output_dir and all_chunks:
            self._save_chunks_batch(all_chunks, output_dir)
        
        # Calculate and log comprehensive statistics
        processing_time = time.time() - start_time
        self._log_batch_statistics(all_chunks, processing_time, total_errors)
        
        # Update global metrics
        with self._stats_lock:
            self.metrics.total_files += len(markdown_files)
            self.metrics.total_chunks += len(all_chunks)
            self.metrics.processing_time += processing_time
            self.metrics.errors += total_errors
        
        return all_chunks
    
    def _process_files_sequential(self, markdown_files: List[str], use_docling: bool, 
                                 batch_size: int) -> Tuple[List[DocumentChunk], int]:
        """Process files sequentially with batching for memory efficiency."""
        all_chunks = []
        total_errors = 0
        
        # Process files in batches to manage memory
        for i in range(0, len(markdown_files), batch_size):
            batch_files = markdown_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(markdown_files) + batch_size - 1)//batch_size}")
            
            batch_chunks = []
            for j, markdown_file in enumerate(batch_files):
                try:
                    logger.info(f"  Processing file {j+1}/{len(batch_files)}: {os.path.basename(markdown_file)}")
                    chunks = self.chunk_file(markdown_file, use_docling)
                    logger.info(f"  Created {len(chunks)} chunks from {os.path.basename(markdown_file)}")
                    
                    # Adjust chunk IDs to be globally unique
                    base_id = len(all_chunks) + len(batch_chunks)
                    for chunk in chunks:
                        chunk.chunk_id = base_id + chunk.chunk_id
                    
                    batch_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error chunking {markdown_file}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    total_errors += 1
            
            all_chunks.extend(batch_chunks)
            
            # Periodic memory cleanup
            if not self.resource_manager.check_memory_usage():
                logger.info("Performing memory cleanup between batches")
                gc.collect()
        
        return all_chunks, total_errors
    
    def _process_files_parallel(self, markdown_files: List[str], num_workers: int, 
                               use_docling: bool, batch_size: int) -> Tuple[List[DocumentChunk], int]:
        """Process files in parallel with optimized resource management."""
        all_chunks = []
        total_errors = 0
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=num_workers, 
                               thread_name_prefix="chunker") as executor:
            
            # Submit work in batches to manage resource usage
            for i in range(0, len(markdown_files), batch_size):
                batch_files = markdown_files[i:i + batch_size]
                logger.info(f"Submitting batch {i//batch_size + 1} for parallel processing")
                
                # Submit batch jobs
                future_to_file = {
                    executor.submit(self._chunk_file_safe, file, use_docling): file 
                    for file in batch_files
                }
                
                # Collect results as they complete
                batch_chunks = []
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        chunks = future.result(timeout=300)  # 5 minute timeout per file
                        if chunks:
                            # Adjust chunk IDs to be globally unique
                            base_id = len(all_chunks) + len(batch_chunks)
                            for chunk in chunks:
                                chunk.chunk_id = base_id + chunk.chunk_id
                            batch_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        total_errors += 1
                
                all_chunks.extend(batch_chunks)
                
                # Memory management between batches
                if not self.resource_manager.check_memory_usage():
                    logger.info("High memory usage detected, cleaning up")
                    gc.collect()
        
        return all_chunks, total_errors
    
    def _chunk_file_safe(self, markdown_path: str, use_docling: bool) -> List[DocumentChunk]:
        """Thread-safe wrapper for chunk_file with enhanced error handling."""
        with self._processing_lock:
            return self.chunk_file(markdown_path, use_docling)
    
    def _post_process_batch_chunks(self, all_chunks: List[DocumentChunk]):
        """Efficiently post-process all chunks after batch processing."""
        if not all_chunks:
            return
        
        # Update the final chunk's next relationship
        all_chunks[-1].metadata["relationships"]["next"] = None
        
        # Optionally add batch-level metadata
        for chunk in all_chunks:
            if "batch_processing" not in chunk.metadata:
                chunk.metadata["batch_processing"] = True
    
    def _save_chunks_batch(self, all_chunks: List[DocumentChunk], output_dir: str):
        """Efficiently save chunks using batch operations."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Use batch writing for better performance
            chunk_batch = []
            batch_size = 100  # Write in batches of 100 chunks
            
            for chunk in all_chunks:
                chunk_batch.append(chunk)
                
                if len(chunk_batch) >= batch_size:
                    self._write_chunk_batch(chunk_batch, output_dir)
                    chunk_batch = []
            
            # Write remaining chunks
            if chunk_batch:
                self._write_chunk_batch(chunk_batch, output_dir)
            
            logger.info(f"Batch saved {len(all_chunks)} chunks to {output_dir}")
        except Exception as e:
            logger.error(f"Error in batch saving: {e}")
    
    def _write_chunk_batch(self, chunk_batch: List[DocumentChunk], output_dir: str):
        """Write a batch of chunks efficiently."""
        for chunk in chunk_batch:
            try:
                # Save chunk content
                chunk_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.md")
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk.text)
                
                # Save metadata
                metadata_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Error writing chunk {chunk.chunk_id}: {e}")
    
    def _log_batch_statistics(self, all_chunks: List[DocumentChunk], processing_time: float, total_errors: int):
        """Log comprehensive batch processing statistics."""
        if not all_chunks:
            logger.warning("No chunks were created during batch processing")
            return
        
        # Calculate quality statistics
        quality_scores = [chunk.metadata.get("quality_score", 0.0) for chunk in all_chunks]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Calculate size statistics
        chunk_sizes = [len(chunk.text) for chunk in all_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        # Method distribution
        method_counts = defaultdict(int)
        for chunk in all_chunks:
            method = chunk.metadata.get("chunking_method", "unknown")
            method_counts[method] += 1
        
        # Performance metrics
        chunks_per_second = len(all_chunks) / processing_time if processing_time > 0 else 0
        
        logger.info(f"\n=== Optimized Batch Chunking Statistics ===")
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Performance: {chunks_per_second:.2f} chunks/second")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Average chunk quality: {avg_quality:.3f}")
        logger.info(f"Average chunk size: {avg_size:.0f} characters")
        logger.info(f"Method distribution: {dict(method_counts)}")
        
        # Log tokenizer information
        if self.enable_advanced_tokenizers and hasattr(self, 'tokenizer_cache'):
            tokenizer = self._get_optimized_tokenizer()
            if tokenizer:
                logger.info(f"Tokenizer: {type(tokenizer).__name__} ({getattr(tokenizer, 'model_name', 'default')})")
            else:
                logger.info("Tokenizer: Basic word-based tokenizer (fallback)")
        else:
            logger.info("Tokenizer: Basic word-based tokenizer")
        
        if self.enable_caching:
            logger.info(f"Cache hit ratio: {self.metrics.cache_hit_ratio:.2f}")
        
        if self.enable_advanced_tokenizers:
            logger.info("Advanced tokenizers enabled for precise token counting")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._stats_lock:
            return PerformanceMetrics(
                total_files=self.metrics.total_files,
                total_chunks=self.metrics.total_chunks,
                processing_time=self.metrics.processing_time,
                memory_peak=self.metrics.memory_peak,
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses,
                errors=self.metrics.errors
            )


    def save_chunks(self, chunks: List[DocumentChunk], output_dir: str) -> None:
        """Save chunks to individual files with metadata."""
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
    
    def create_rag_dataset(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Create a RAG-compatible dataset from chunks."""
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
                    **make_json_serializable(chunk.metadata)
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
                "method": "docling_hybrid_optimized" if HAS_DOCLING else "heading_based"
            },
            "version": "1.0.0",
            "created_at": "2025-06-28"
        }
        
        return dataset
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for better searchability."""
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
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    def create_enhanced_rag_dataset(self, chunks: List[DocumentChunk], output_dir: str, 
                                   enable_deduplication: bool = True, 
                                   multimodal_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Create enhanced RAG datasets with multiple formats, deduplication, and multimodal support."""
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Initialize deduplicator if enabled and available
        deduplicator = None
        if enable_deduplication:
            try:
                deduplicator = DatasetDeduplicator()
            except Exception as e:
                logger.warning(f"Deduplication not available: {e}")
        
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
            logger.info("Deduplicating RAG chunks...")
            original_chunks = rag_dataset["chunks"]
            deduplicated_chunks, dedup_report = deduplicator.deduplicate_comprehensive(
                original_chunks, 
                keys=['text', 'id'], 
                strategy="progressive"
            )
            rag_dataset["chunks"] = deduplicated_chunks
            rag_dataset["deduplication_report"] = dedup_report
            logger.info(f"RAG dataset: {len(original_chunks)} → {len(deduplicated_chunks)} chunks")
        
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
            logger.info("Deduplicating QA pairs...")
            original_qa_pairs = qa_dataset["qa_pairs"]
            deduplicated_qa_pairs, qa_dedup_report = deduplicator.deduplicate_comprehensive(
                original_qa_pairs, 
                keys=['question', 'answer'], 
                strategy="progressive"
            )
            qa_dataset["qa_pairs"] = deduplicated_qa_pairs
            qa_dataset["deduplication_report"] = qa_dedup_report
            logger.info(f"QA pairs: {len(original_qa_pairs)} → {len(deduplicated_qa_pairs)} pairs")
        
        qa_path = os.path.join(output_dir, "rag_qa_dataset.json")
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, indent=2, ensure_ascii=False)
        results["qa"] = qa_path
        
        # 4. Embedding-ready format
        embedding_dataset = self._create_embedding_dataset(chunks)
        
        # Deduplicate embedding entries if enabled
        if enable_deduplication and deduplicator:
            logger.info("Deduplicating embedding entries...")
            original_entries = embedding_dataset["entries"]
            deduplicated_entries, embed_dedup_report = deduplicator.deduplicate_comprehensive(
                original_entries, 
                keys=['text', 'content'], 
                strategy="progressive"
            )
            embedding_dataset["entries"] = deduplicated_entries
            embedding_dataset["deduplication_report"] = embed_dedup_report
            logger.info(f"Embedding entries: {len(original_entries)} → {len(deduplicated_entries)} entries")
        
        embedding_path = os.path.join(output_dir, "embedding_dataset.json")
        with open(embedding_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_dataset, f, indent=2, ensure_ascii=False)
        results["embedding"] = embedding_path
        
        logger.info(f"Enhanced RAG datasets created with deduplication and multimodal support: {list(results.keys())}")
        logger.info(f"Multimodal content: {multimodal_analysis['chunks_with_tables']} table chunks, {multimodal_analysis['chunks_with_figures']} figure chunks")
        return results
    
    def _analyze_multimodal_content(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze multimodal content across all chunks."""
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
        """Create question-answer pairs from chunks for RAG training."""
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
        """Generate potential questions from a chunk."""
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
        """Create embedding-ready dataset format."""
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
                    **make_json_serializable(chunk.metadata)
                },
                "embedding_text": self._prepare_embedding_text(chunk.text),
                "search_text": self._prepare_search_text(chunk.text)
            })
        
        return {
            "entries": embedding_data,  # For deduplication compatibility
            "documents": embedding_data,  # Keep both for compatibility
            "total_documents": len(embedding_data),
            "description": "Documents prepared for embedding generation and vector search",
            "embedding_fields": ["text", "embedding_text", "search_text"],
            "version": "1.0.0"
        }
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from text."""
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
        """Create a summary from text."""
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
        """Prepare text for embedding generation."""
        # Clean and normalize text for embedding
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with embeddings
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    def _prepare_search_text(self, text: str) -> str:
        """Prepare text for search indexing."""
        # Prepare text for full-text search
        # Extract key phrases and terms
        keywords = self._extract_keywords(text)
        title = self._extract_title(text)
        
        # Combine title, keywords, and original text
        search_components = [title] + keywords + [text]
        return ' '.join(search_components)
    
    def analyze_chunk_quality(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze the quality of generated chunks and provide statistics."""
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
    
    def _extract_comprehensive_chunk_metadata(self, chunk, chunk_index: int, chunker, 
                                            tokenizer, doc, hierarchy_map: Dict[str, Any], 
                                            doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata from chunk with hierarchy and context enrichment.
        
        Args:
            chunk: Document chunk
            chunk_index: Index of the chunk
            chunker: HybridChunker instance
            tokenizer: Tokenizer instance
            doc: DoclingDocument
            hierarchy_map: Document hierarchy
            doc_metadata: Document-level metadata
            
        Returns:
            Dictionary with comprehensive chunk metadata
        """
        metadata = {
            "chunk_type": "docling_enhanced_chunk",
            "has_docling_metadata": False,
            "element_type": "unknown",
            "hierarchical_context": {},
            "structural_metadata": {},
            "content_analysis": {},
            "relationship_metadata": {},
            "quality_metrics": {}
        }
        
        try:
            # Extract basic chunk information
            if hasattr(chunk, 'meta') and chunk.meta:
                metadata["has_docling_metadata"] = True
                
                # Extract metadata attributes safely
                if hasattr(chunk.meta, '__dict__'):
                    for key, value in chunk.meta.__dict__.items():
                        try:
                            json.dumps(value)
                            metadata[f"docling_meta_{key}"] = value
                        except (TypeError, ValueError):
                            metadata[f"docling_meta_{key}_type"] = str(type(value).__name__)
            
            # Extract element type information
            if hasattr(chunk, 'label'):
                metadata["element_type"] = str(chunk.label)
            
            # Add hierarchical context
            chunk_position = self._find_chunk_position_enhanced(chunk, doc, hierarchy_map)
            if chunk_position:
                element_info = hierarchy_map.get("element_hierarchy", {}).get(chunk_position, {})
                metadata["hierarchical_context"] = {
                    "position_id": chunk_position,
                    "parent": element_info.get("parent"),
                    "section": element_info.get("section"),
                    "level": element_info.get("level", 0),
                    "element_index": element_info.get("index", -1),
                    "content_type": element_info.get("content_type", "unknown")
                }
                
                # Add parent chain
                parent_chain = self._build_parent_chain_enhanced(chunk_position, hierarchy_map)
                metadata["hierarchical_context"]["parent_chain"] = parent_chain
                
                # Add section context
                section_context = self._find_current_section_enhanced(chunk_position, hierarchy_map)
                metadata["hierarchical_context"]["current_section"] = section_context
            
            # Structural metadata
            metadata["structural_metadata"] = {
                "in_table": self._is_chunk_in_table_enhanced(chunk),
                "in_figure": self._is_chunk_in_figure_enhanced(chunk),
                "is_heading": self._is_chunk_heading_enhanced(chunk),
                "has_references": self._has_references_enhanced(chunk),
                "multimodal_content": self._detect_multimodal_content(chunk)
            }
            
            # Content analysis
            if hasattr(chunk, 'text'):
                text = chunk.text
                metadata["content_analysis"] = {
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "sentence_count": len(text.split('.')),
                    "paragraph_count": len(text.split('\n\n')),
                    "has_technical_terms": self._has_technical_terms(text),
                    "language_complexity": self._assess_content_complexity(text),
                    "content_classification": self._classify_element_content(
                        text, str(chunk.label) if hasattr(chunk, 'label') else ""
                    )
                }
            
            # Relationship metadata
            metadata["relationship_metadata"] = {
                "cross_references": self._detect_element_references(
                    chunk.text if hasattr(chunk, 'text') else ""
                ),
                "semantic_group": self._find_semantic_group(chunk_position, hierarchy_map),
                "contextual_links": self._find_contextual_links(chunk, hierarchy_map)
            }
            
            # Document context
            metadata["document_context"] = {
                "total_elements": doc_metadata.get("elements_count", 0),
                "document_hierarchy_depth": len(hierarchy_map.get("headings", [])),
                "semantic_groups": list(hierarchy_map.get("semantic_groups", {}).keys()),
                "cross_reference_count": len(hierarchy_map.get("cross_references", []))
            }
        
        except Exception as e:
            logger.debug(f"Error extracting comprehensive chunk metadata: {e}")
            metadata["extraction_error"] = str(e)
        
        return metadata
    
    def _assess_enhanced_chunk_quality(self, chunk_text: str, chunk, doc, 
                                     hierarchy_map: Dict[str, Any]) -> Dict[str, Any]:
        """Assess comprehensive chunk quality metrics with enhanced analysis.
        
        Args:
            chunk_text: The chunk text content
            chunk: Original chunk object
            doc: DoclingDocument
            hierarchy_map: Document hierarchy
            
        Returns:
            Dictionary with enhanced quality metrics
        """
        quality_metrics = {
            "context_enrichment_quality": 0.0,
            "metadata_completeness": 0.0,
            "hierarchical_integration": 0.0,
            "relationship_strength": 0.0,
            "content_coherence": 0.0,
            "overall_enhanced_quality": 0.0,
            "quality_breakdown": {}
        }
        
        try:
            # Context enrichment quality assessment
            context_score = 0.0
            context_factors = {}
            
            # Check for hierarchical context
            chunk_position = self._find_chunk_position_enhanced(chunk, doc, hierarchy_map)
            if chunk_position:
                context_score += 0.3
                context_factors["has_position"] = True
                
                # Check parent chain
                parent_chain = self._build_parent_chain_enhanced(chunk_position, hierarchy_map)
                if parent_chain:
                    context_score += 0.2
                    context_factors["has_parent_chain"] = True
                    context_factors["parent_chain_depth"] = len(parent_chain)
            
            # Check for section context
            if hierarchy_map.get("sections"):
                context_score += 0.2
                context_factors["has_sections"] = True
            
            # Check for semantic classification
            if hasattr(chunk, 'text') and chunk.text:
                content_type = self._classify_element_content(
                    chunk.text, str(chunk.label) if hasattr(chunk, 'label') else ""
                )
                if content_type != "general_content":
                    context_score += 0.3
                    context_factors["semantic_classification"] = content_type
            
            quality_metrics["context_enrichment_quality"] = min(1.0, context_score)
            quality_metrics["quality_breakdown"]["context_factors"] = context_factors
            
            # Metadata completeness assessment
            metadata_score = 0.0
            metadata_factors = {}
            
            if hasattr(chunk, 'meta') and chunk.meta:
                metadata_score += 0.4
                metadata_factors["has_docling_meta"] = True
            
            if hierarchy_map.get("element_hierarchy"):
                metadata_score += 0.3
                metadata_factors["has_hierarchy"] = True
            
            if hasattr(chunk, 'label'):
                metadata_score += 0.3
                metadata_factors["has_element_label"] = True
            
            quality_metrics["metadata_completeness"] = min(1.0, metadata_score)
            quality_metrics["quality_breakdown"]["metadata_factors"] = metadata_factors
            
            # Hierarchical integration assessment
            hierarchical_score = 0.0
            hierarchical_factors = {}
            
            if chunk_position:
                element_info = hierarchy_map.get("element_hierarchy", {}).get(chunk_position, {})
                
                if element_info.get("parent"):
                    hierarchical_score += 0.4
                    hierarchical_factors["has_parent"] = True
                
                if element_info.get("section"):
                    hierarchical_score += 0.3
                    hierarchical_factors["has_section"] = True
                
                if element_info.get("level", 0) > 0:
                    hierarchical_score += 0.3
                    hierarchical_factors["has_level"] = element_info["level"]
            
            quality_metrics["hierarchical_integration"] = min(1.0, hierarchical_score)
            quality_metrics["quality_breakdown"]["hierarchical_factors"] = hierarchical_factors
            
            # Relationship strength assessment
            relationship_score = 0.0
            relationship_factors = {}
            
            # Check for cross-references
            if hasattr(chunk, 'text') and chunk.text:
                refs = self._detect_element_references(chunk.text)
                if refs:
                    relationship_score += 0.4
                    relationship_factors["cross_references"] = len(refs)
            
            # Check semantic group membership
            semantic_group = self._find_semantic_group(chunk_position, hierarchy_map)
            if semantic_group:
                relationship_score += 0.3
                relationship_factors["semantic_group"] = semantic_group
            
            # Check for multimodal relationships
            if self._detect_multimodal_content(chunk):
                relationship_score += 0.3
                relationship_factors["has_multimodal"] = True
            
            quality_metrics["relationship_strength"] = min(1.0, relationship_score)
            quality_metrics["quality_breakdown"]["relationship_factors"] = relationship_factors
            
            # Content coherence assessment
            coherence_score = 0.0
            coherence_factors = {}
            
            if chunk_text and len(chunk_text.strip()) > 50:
                coherence_score += 0.3
                coherence_factors["adequate_length"] = True
            
            # Check for complete sentences
            if chunk_text:
                sentences = chunk_text.split('.')
                complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
                if complete_sentences > 0:
                    completeness_ratio = complete_sentences / max(len(sentences), 1)
                    coherence_score += 0.4 * completeness_ratio
                    coherence_factors["sentence_completeness"] = completeness_ratio
            
            # Check for technical coherence
            if self._has_technical_terms(chunk_text):
                coherence_score += 0.3
                coherence_factors["has_technical_content"] = True
            
            quality_metrics["content_coherence"] = min(1.0, coherence_score)
            quality_metrics["quality_breakdown"]["coherence_factors"] = coherence_factors
            
            # Calculate overall enhanced quality
            overall = (
                quality_metrics["context_enrichment_quality"] * 0.25 +
                quality_metrics["metadata_completeness"] * 0.20 +
                quality_metrics["hierarchical_integration"] * 0.25 +
                quality_metrics["relationship_strength"] * 0.15 +
                quality_metrics["content_coherence"] * 0.15
            )
            quality_metrics["overall_enhanced_quality"] = overall
            
            # Add quality classification
            if overall >= 0.8:
                quality_metrics["quality_class"] = "excellent"
            elif overall >= 0.6:
                quality_metrics["quality_class"] = "good"
            elif overall >= 0.4:
                quality_metrics["quality_class"] = "fair"
            else:
                quality_metrics["quality_class"] = "poor"
        
        except Exception as e:
            logger.debug(f"Error assessing enhanced chunk quality: {e}")
            quality_metrics["assessment_error"] = str(e)
        
        return quality_metrics
    
    def _is_chunk_in_table_enhanced(self, chunk) -> bool:
        """Enhanced check if chunk is part of a table."""
        try:
            if hasattr(chunk, 'label'):
                label_str = str(chunk.label).lower()
                if 'table' in label_str:
                    return True
            
            if hasattr(chunk, 'text') and chunk.text:
                # Check for table markers
                if re.search(r'\|.*\|', chunk.text) or 'table' in chunk.text.lower():
                    return True
        except:
            pass
        return False
    
    def _is_chunk_in_figure_enhanced(self, chunk) -> bool:
        """Enhanced check if chunk is part of a figure."""
        try:
            if hasattr(chunk, 'label'):
                label_str = str(chunk.label).lower()
                if any(term in label_str for term in ['figure', 'image', 'diagram', 'chart', 'picture']):
                    return True
        except:
            pass
        return False
    
    def _is_chunk_heading_enhanced(self, chunk) -> bool:
        """Enhanced check if chunk is a heading."""
        try:
            if hasattr(chunk, 'label'):
                label_str = str(chunk.label).lower()
                if any(term in label_str for term in ['title', 'heading', 'subtitle', 'header']):
                    return True
        except:
            pass
        return False
    
    def _has_references_enhanced(self, chunk) -> bool:
        """Enhanced check if chunk contains references."""
        try:
            if hasattr(chunk, 'text') and chunk.text:
                text = chunk.text.lower()
                reference_patterns = [
                    'see also', 'refer to', 'reference', 'fig.', 'figure', 'table', 'section',
                    'chapter', 'appendix', 'page', 'above', 'below', 'mentioned', 'described'
                ]
                return any(pattern in text for pattern in reference_patterns)
        except:
            pass
        return False
    
    def _detect_multimodal_content(self, chunk) -> Dict[str, Any]:
        """Detect multimodal content in chunk."""
        multimodal = {
            "has_multimodal": False,
            "types": [],
            "indicators": []
        }
        
        try:
            # Check label-based multimodal content
            if hasattr(chunk, 'label'):
                label_str = str(chunk.label).lower()
                if 'table' in label_str:
                    multimodal["types"].append("table")
                    multimodal["indicators"].append("label:table")
                if any(term in label_str for term in ['figure', 'image', 'picture']):
                    multimodal["types"].append("figure")
                    multimodal["indicators"].append("label:figure")
            
            # Check text-based multimodal content
            if hasattr(chunk, 'text') and chunk.text:
                text = chunk.text
                if re.search(r'\|.*\|', text):
                    multimodal["types"].append("markdown_table")
                    multimodal["indicators"].append("text:table_syntax")
                if re.search(r'!\[.*?\]\(.*?\)', text):
                    multimodal["types"].append("image_markdown")
                    multimodal["indicators"].append("text:image_syntax")
            
            multimodal["has_multimodal"] = len(multimodal["types"]) > 0
        
        except Exception as e:
            logger.debug(f"Error detecting multimodal content: {e}")
        
        return multimodal
    
    def _find_semantic_group(self, chunk_position: Optional[str], 
                           hierarchy_map: Dict[str, Any]) -> Optional[str]:
        """Find the semantic group for a chunk."""
        if not chunk_position:
            return None
        
        try:
            semantic_groups = hierarchy_map.get("semantic_groups", {})
            for group_name, element_ids in semantic_groups.items():
                if chunk_position in element_ids:
                    return group_name
        except Exception as e:
            logger.debug(f"Error finding semantic group: {e}")
        
        return None
    
    def _find_contextual_links(self, chunk, hierarchy_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find contextual links for the chunk."""
        links = []
        
        try:
            if hasattr(chunk, 'text') and chunk.text:
                # Find cross-references in text
                refs = self._detect_element_references(chunk.text)
                for ref in refs:
                    links.append({
                        "type": "cross_reference",
                        "target": ref,
                        "strength": 0.8
                    })
                
                # Find semantic similarities
                chunk_position = self._find_chunk_position_enhanced(chunk, None, hierarchy_map)
                if chunk_position:
                    semantic_group = self._find_semantic_group(chunk_position, hierarchy_map)
                    if semantic_group:
                        semantic_groups = hierarchy_map.get("semantic_groups", {})
                        related_elements = semantic_groups.get(semantic_group, [])
                        if len(related_elements) > 1:
                            links.append({
                                "type": "semantic_group",
                                "group": semantic_group,
                                "related_count": len(related_elements) - 1,
                                "strength": 0.6
                            })
        
        except Exception as e:
            logger.debug(f"Error finding contextual links: {e}")
        
        return links


# Backwards compatibility alias
DocumentChunker = OptimizedDocumentChunker