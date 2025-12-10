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
                # Simplified serializer creation without complex parameters
                if serializer_type == "markdown":
                    # Just return None for now - will use fallback chunking
                    serializer = None
                else:
                    serializer = None
                
                if serializer and len(self._serializer_cache) < self.max_size:
                    self._serializer_cache[serializer_type] = serializer
                return serializer
            except Exception as e:
                logger.debug(f"Serializer creation failed: {e}")
                return None
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._chunker_cache.clear()
            self._serializer_cache.clear()


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
        
        # Enhanced chunking parameters
        self.semantic_min_size = max(self.config.min_chunk_size, 600)
        self.semantic_max_size = min(self.config.max_chunk_size, 2000)
        self.semantic_overlap = min(self.config.chunk_overlap, 100)
        
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
        if hasattr(self.config, 'hf_tokenizer_model'):
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
                logger.debug(f"Converting {markdown_path} to Docling document format")
                result = converter.convert(source=markdown_path)
                doc = result.document
                
                if not doc:
                    raise Exception("Document conversion returned empty result")
                
                # Get optimized serializer if available (simplified for now)
                serializer = None
                try:
                    serializer = self._get_optimized_serializer()
                except Exception as e:
                    logger.debug(f"Serializer not available: {e}")
                    serializer = None
                
                # Generate chunks using the optimized hybrid chunker
                logger.debug("Generating chunks with optimized HybridChunker")
                chunk_iter = chunker.chunk(dl_doc=doc)
                
                # Batch process chunks for better memory efficiency
                chunks = []
                source_name = os.path.basename(markdown_path)
                chunk_batch = []
                batch_size = 50  # Process chunks in batches
                
                for i, chunk in enumerate(chunk_iter):
                    try:
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
                logger.info("Falling back to heading-based chunking")
                with self._stats_lock:
                    self.metrics.errors += 1
                return self._chunk_by_headings(markdown_path)
    
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
                        tokens = tokenizer.encode(chunk.text)
                        metadata["token_count"] = len(tokens)
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
                    # Extract metadata with JSON serialization safety
                    for key, value in meta_dict.items():
                        try:
                            # Test if value is directly JSON serializable
                            json.dumps(value)
                            metadata[f"docling_{key}"] = value
                        except (TypeError, ValueError):
                            # For non-serializable objects, extract basic info
                            if value is not None:
                                if hasattr(value, '__dict__') and hasattr(value, '__class__'):
                                    # Extract basic object info
                                    obj_info = {
                                        "_type": type(value).__name__,
                                        "_module": getattr(type(value), '__module__', 'unknown')
                                    }
                                    # Add simple attributes if they exist
                                    if hasattr(value, 'id'):
                                        obj_info["id"] = str(getattr(value, 'id', ''))
                                    if hasattr(value, 'label'):
                                        obj_info["label"] = str(getattr(value, 'label', ''))
                                    if hasattr(value, 'text') and len(str(getattr(value, 'text', ''))) < 200:
                                        obj_info["text"] = str(getattr(value, 'text', ''))
                                    metadata[f"docling_{key}"] = obj_info
                                else:
                                    metadata[f"docling_{key}_type"] = str(type(value).__name__)
                            else:
                                metadata[f"docling_{key}"] = None
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
        
        # Clean excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'\t+', ' ', text)
        
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
                    if current_chunk and len(current_chunk + '\n\n' + section_content) > self.semantic_max_size:
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
                            current_chunk += '\n\n' + section_content
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
                if current_chunk and len(current_chunk + '\n\n' + section_content) > self.semantic_max_size:
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
                        current_chunk += '\n\n' + section_content
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
    
    def _extract_semantic_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract semantic sections from technical documentation."""
        sections = []
        lines = content.split('\n')
        current_section = []
        current_headers = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect headers (numbered sections, class definitions, etc.)
            if (re.match(r'^\d+(\.\d+)*\s+', line) or  # "2.3.1 Section Name"
                line.startswith('class ') or 
                line.startswith('enum ') or
                line.startswith('#') or
                re.match(r'^[A-Z][A-Za-z\s]+:', line)):  # "Configuration:"
                
                # Save previous section if it exists and is substantial
                if current_section:
                    content_text = '\n'.join(current_section)
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
            content_text = '\n'.join(current_section)
            if len(content_text.strip()) > 50:
                sections.append({
                    'headers': current_headers.copy(),
                    'content': self._clean_technical_text(content_text),
                    'type': self._classify_section_type(content_text)
                })
        
        return sections
    
    def _split_large_section(self, content: str) -> List[str]:
        """Split large sections while preserving semantic meaning."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
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
                        tokens = tokenizer.encode(content)
                        token_count = len(tokens)
                    elif hasattr(tokenizer, 'tokenize'):
                        tokens = tokenizer.tokenize(content)
                        token_count = len(tokens)
                except Exception as e:
                    logger.debug(f"Token counting failed, using fallback: {e}")
        
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
        """Calculate a quality score for the chunk based on various factors."""
        score = 0.0
        
        # Size scoring (optimal size gets higher score)
        size_ratio = len(content) / self.semantic_max_size
        if 0.3 <= size_ratio <= 0.8:  # Sweet spot
            score += 0.3
        elif 0.1 <= size_ratio <= 0.9:  # Acceptable
            score += 0.2
        else:  # Too small or too large
            score += 0.1
        
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
        technical_indicators = len(re.findall(r'\b(class|enum|attribute|parameter|configure|procedure)\b', content.lower()))
        score += min(0.3, technical_indicators * 0.05)
        
        return min(1.0, score)  # Cap at 1.0
    
    def chunk_file(self, markdown_path: str, use_docling: bool = True) -> List[DocumentChunk]:
        """Chunk a single markdown file with optimized processing."""
        if use_docling and HAS_DOCLING:
            try:
                return self._chunk_with_optimized_docling(markdown_path)
            except Exception as e:
                logger.warning(f"Optimized Docling chunking failed, falling back to simple chunking: {e}")
                return self._chunk_by_headings(markdown_path)
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
                all_chunks, total_errors = self._process_files_sequential(
                    markdown_files, use_docling, batch_size
                )
            else:
                # Parallel processing with careful resource management
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
            for markdown_file in batch_files:
                try:
                    chunks = self.chunk_file(markdown_file, use_docling)
                    
                    # Adjust chunk IDs to be globally unique
                    base_id = len(all_chunks) + len(batch_chunks)
                    for chunk in chunks:
                        chunk.chunk_id = base_id + chunk.chunk_id
                    
                    batch_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error chunking {markdown_file}: {e}")
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
        try:
            with self._processing_lock:
                return self.chunk_file(markdown_path, use_docling)
        except Exception as e:
            logger.error(f"Safe chunking failed for {markdown_path}: {e}")
            return []
    
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
                
                # Save metadata with error handling for JSON serialization
                metadata_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    try:
                        chunk_dict = chunk.to_dict()
                        json.dump(chunk_dict, f, indent=2, ensure_ascii=False)
                    except (TypeError, ValueError) as e:
                        # Fallback: save basic info if full serialization fails
                        logger.warning(f"JSON serialization failed for chunk {chunk.chunk_id}, using fallback: {e}")
                        fallback_dict = {
                            "id": chunk.chunk_id,
                            "text": chunk.text,
                            "source": chunk.source,
                            "metadata": {
                                "chunking_method": chunk.metadata.get("chunking_method", "unknown"),
                                "chunk_index": chunk.metadata.get("chunk_index", chunk.chunk_id),
                                "char_count": len(chunk.text),
                                "word_count": len(chunk.text.split()),
                                "serialization_error": str(e),
                                "fallback_used": True
                            }
                        }
                        json.dump(fallback_dict, f, indent=2, ensure_ascii=False)
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
    
    def create_rag_dataset(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Create a RAG-compatible dataset from chunks with JSON-safe serialization.
        
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
            "version": "2.0.0",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
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
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    def save_chunks(self, chunks: List[DocumentChunk], output_dir: str) -> None:
        """Save chunks to individual files with metadata.
        
        Args:
            chunks: List of chunks to save
            output_dir: Directory to save chunks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for chunk in chunks:
            try:
                # Save chunk content
                chunk_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.md")
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk.text)
                
                # Save metadata with JSON-safe serialization
                metadata_path = os.path.join(output_dir, f"chunk_{chunk.chunk_id:04d}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    try:
                        chunk_dict = chunk.to_dict()
                        json.dump(chunk_dict, f, indent=2, ensure_ascii=False)
                    except (TypeError, ValueError) as e:
                        # Fallback: save basic info if full serialization fails
                        logger.warning(f"JSON serialization failed for chunk {chunk.chunk_id}, using fallback: {e}")
                        fallback_dict = {
                            "id": chunk.chunk_id,
                            "text": chunk.text,
                            "source": chunk.source,
                            "metadata": {
                                "chunking_method": chunk.metadata.get("chunking_method", "unknown"),
                                "chunk_index": chunk.metadata.get("chunk_index", chunk.chunk_id),
                                "char_count": len(chunk.text),
                                "word_count": len(chunk.text.split()),
                                "serialization_error": str(e),
                                "fallback_used": True
                            }
                        }
                        json.dump(fallback_dict, f, indent=2, ensure_ascii=False)
                        
            except Exception as e:
                logger.error(f"Error saving chunk {chunk.chunk_id}: {e}")
        
        logger.info(f"Saved {len(chunks)} chunks to {output_dir}")

    def create_enhanced_rag_dataset(self, chunks: List[DocumentChunk], output_dir: str, 
                                   enable_deduplication: bool = True, 
                                   multimodal_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
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
        
        try:
            # Initialize deduplicator if enabled and available
            deduplicator = None
            if enable_deduplication:
                try:
                    deduplicator = DatasetDeduplicator()
                except Exception as e:
                    logger.warning(f"Deduplication not available: {e}")
            
            # 1. Standard RAG JSON format with multimodal enhancements
            rag_dataset = self.create_rag_dataset(chunks)
            
            # Add multimodal metadata to the dataset
            if multimodal_metadata:
                rag_dataset["multimodal_source_metadata"] = make_json_serializable(multimodal_metadata)
            
            # Add performance metrics
            if hasattr(self, 'metrics'):
                rag_dataset["performance_metrics"] = {
                    "cache_hit_ratio": self.metrics.cache_hit_ratio,
                    "chunks_per_second": self.metrics.chunks_per_second,
                    "total_processing_time": self.metrics.processing_time
                }
            
            # Deduplicate chunks if enabled
            if enable_deduplication and deduplicator:
                logger.info("🔍 Deduplicating RAG chunks...")
                try:
                    original_chunks = rag_dataset["chunks"]
                    deduplicated_chunks, dedup_report = deduplicator.deduplicate_comprehensive(
                        original_chunks, 
                        keys=['text', 'id'], 
                        strategy="progressive"
                    )
                    rag_dataset["chunks"] = deduplicated_chunks
                    rag_dataset["deduplication_report"] = make_json_serializable(dedup_report)
                    logger.info(f"📉 RAG dataset: {len(original_chunks)} → {len(deduplicated_chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Deduplication failed: {e}")
            
            # Save enhanced RAG dataset
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
            
            logger.info(f"✅ Enhanced RAG datasets created: {list(results.keys())}")
            return results
            
        except Exception as e:
            logger.error(f"Error creating enhanced RAG dataset: {e}")
            # Fallback to basic dataset
            basic_dataset = self.create_rag_dataset(chunks)
            basic_path = os.path.join(output_dir, "rag_dataset_basic.json")
            with open(basic_path, 'w', encoding='utf-8') as f:
                json.dump(basic_dataset, f, indent=2, ensure_ascii=False)
            return {"basic": basic_path}


# Backwards compatibility alias
DocumentChunker = OptimizedDocumentChunker