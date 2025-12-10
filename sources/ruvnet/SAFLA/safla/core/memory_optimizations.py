"""
Performance optimizations and external library integrations for Hybrid Memory Architecture.

This module provides enhanced implementations with:
- FAISS integration for high-performance vector search
- Chroma integration for LLM-optimized operations
- GPU acceleration support
- Persistent storage backends
- Advanced consolidation algorithms
"""

import numpy as np
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

# Optional imports for performance libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using fallback vector search.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available. Using fallback vector storage.")

from .hybrid_memory import (
    VectorMemoryManager, SimilarityMetric, MemoryItem, SimilarityResult
)

logger = logging.getLogger(__name__)


class VectorSearchBackend(ABC):
    """Abstract base class for vector search backends."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> bool:
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors. Returns (distances, indices)."""
        pass
    
    @abstractmethod
    def remove_vectors(self, ids: List[str]) -> bool:
        """Remove vectors from the index."""
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get number of vectors in the index."""
        pass


class FAISSVectorBackend(VectorSearchBackend):
    """FAISS-based high-performance vector search backend."""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "IVFFlat",
        metric: str = "L2",
        use_gpu: bool = False,
        nlist: int = 100
    ):
        """Initialize FAISS backend."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for FAISSVectorBackend")
        
        self.dimension = dimension
        self.metric = metric
        self.use_gpu = use_gpu
        
        # Create FAISS index
        if index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension) if metric == "L2" else faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension) if metric == "L2" else faiss.IndexFlatIP(dimension)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # GPU support
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info("Using GPU acceleration for FAISS")
        
        # ID mapping
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx = 0
        
        # Training flag for IVF indices
        self._is_trained = False
        
        logger.info(f"Initialized FAISS backend: {index_type}, dimension={dimension}, metric={metric}")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> bool:
        """Add vectors to FAISS index."""
        try:
            if len(vectors) != len(ids):
                raise ValueError("Number of vectors must match number of IDs")
            
            # Ensure vectors are float32
            vectors = vectors.astype(np.float32)
            
            # Train index if necessary
            if not self._is_trained and hasattr(self.index, 'is_trained'):
                if not self.index.is_trained:
                    self.index.train(vectors)
                self._is_trained = True
            
            # Add vectors
            start_idx = self._next_idx
            self.index.add(vectors)
            
            # Update ID mappings
            for i, vector_id in enumerate(ids):
                idx = start_idx + i
                self._id_to_idx[vector_id] = idx
                self._idx_to_id[idx] = vector_id
            
            self._next_idx += len(vectors)
            
            logger.debug(f"Added {len(vectors)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors in FAISS index."""
        try:
            # Ensure query is float32 and 2D
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_vector, k)
            
            return distances[0], indices[0]
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return np.array([]), np.array([])
    
    def remove_vectors(self, ids: List[str]) -> bool:
        """Remove vectors from FAISS index."""
        # FAISS doesn't support efficient removal, so we mark as removed
        # In production, this would require index rebuilding
        try:
            for vector_id in ids:
                if vector_id in self._id_to_idx:
                    idx = self._id_to_idx[vector_id]
                    del self._id_to_idx[vector_id]
                    del self._idx_to_id[idx]
            
            logger.debug(f"Marked {len(ids)} vectors as removed from FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing vectors from FAISS: {e}")
            return False
    
    def get_vector_count(self) -> int:
        """Get number of vectors in FAISS index."""
        return self.index.ntotal


class ChromaVectorBackend(VectorSearchBackend):
    """ChromaDB-based vector search backend optimized for LLM operations."""
    
    def __init__(
        self,
        collection_name: str = "safla_vectors",
        persist_directory: Optional[str] = None
    ):
        """Initialize ChromaDB backend."""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required for ChromaVectorBackend")
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
        
        logger.info(f"Initialized ChromaDB backend: collection={collection_name}")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> bool:
        """Add vectors to ChromaDB collection."""
        try:
            # Convert to list format for ChromaDB
            embeddings = vectors.tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                ids=ids
            )
            
            logger.debug(f"Added {len(vectors)} vectors to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors to ChromaDB: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors in ChromaDB."""
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=k
            )
            
            # Extract distances and convert IDs to indices
            distances = np.array(results['distances'][0])
            ids = results['ids'][0]
            
            # Convert IDs to indices (ChromaDB returns IDs, not indices)
            indices = np.arange(len(ids))
            
            return distances, indices
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return np.array([]), np.array([])
    
    def remove_vectors(self, ids: List[str]) -> bool:
        """Remove vectors from ChromaDB collection."""
        try:
            self.collection.delete(ids=ids)
            logger.debug(f"Removed {len(ids)} vectors from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error removing vectors from ChromaDB: {e}")
            return False
    
    def get_vector_count(self) -> int:
        """Get number of vectors in ChromaDB collection."""
        return self.collection.count()


class OptimizedVectorMemoryManager(VectorMemoryManager):
    """Enhanced vector memory manager with pluggable backends."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        max_capacity: int = 10000,
        backend: str = "faiss",
        backend_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize optimized vector memory manager."""
        super().__init__(embedding_dim, similarity_metric, max_capacity, **kwargs)
        
        # Initialize backend
        backend_config = backend_config or {}
        
        if backend == "faiss" and FAISS_AVAILABLE:
            metric = "L2" if similarity_metric == SimilarityMetric.EUCLIDEAN else "IP"
            self.backend = FAISSVectorBackend(
                dimension=embedding_dim,
                metric=metric,
                **backend_config
            )
            self._use_backend = True
        elif backend == "chroma" and CHROMA_AVAILABLE:
            self.backend = ChromaVectorBackend(**backend_config)
            self._use_backend = True
        else:
            logger.warning(f"Backend {backend} not available, using fallback")
            self.backend = None
            self._use_backend = False
        
        logger.info(f"Initialized OptimizedVectorMemoryManager with backend: {backend}")
    
    def store(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Store vector using optimized backend."""
        # Use parent implementation for validation and storage
        item_id = super().store(embedding, metadata)
        
        # Add to backend if available
        if self._use_backend and self.backend:
            vectors = embedding.reshape(1, -1)
            self.backend.add_vectors(vectors, [item_id])
        
        return item_id
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """Perform optimized similarity search."""
        if self.is_empty():
            return []
        
        # Use backend if available
        if self._use_backend and self.backend:
            try:
                distances, indices = self.backend.search(query_embedding, k)
                
                results = []
                for distance, idx in zip(distances, indices):
                    if idx < len(self._item_ids):
                        item_id = self._item_ids[idx]
                        if item_id in self._items:
                            # Convert distance to similarity based on metric
                            if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
                                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                            else:
                                similarity = 1.0 - distance  # Assume normalized distance
                            
                            if similarity >= similarity_threshold:
                                item = self._items[item_id]
                                item.access_count += 1
                                self._access_order[item_id] = datetime.now()
                                
                                results.append(SimilarityResult(
                                    item_id=item_id,
                                    similarity_score=similarity,
                                    item=item
                                ))
                
                return results
                
            except Exception as e:
                logger.error(f"Backend search failed, falling back to default: {e}")
        
        # Fallback to parent implementation
        return super().similarity_search(query_embedding, k, similarity_threshold)
    
    def delete(self, item_id: str) -> bool:
        """Delete item from both storage and backend."""
        success = super().delete(item_id)
        
        # Remove from backend if available
        if success and self._use_backend and self.backend:
            self.backend.remove_vectors([item_id])
        
        return success
    
    def batch_store(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict[str, Any]]
    ) -> List[str]:
        """Optimized batch storage."""
        # Use parent implementation for validation and storage
        item_ids = super().batch_store(embeddings, metadata_list)
        
        # Add to backend in batch if available
        if self._use_backend and self.backend and item_ids:
            vectors = np.array(embeddings)
            self.backend.add_vectors(vectors, item_ids)
        
        return item_ids


@dataclass
class MemoryPerformanceMetrics:
    """Performance metrics for memory operations."""
    search_latency_ms: float
    storage_latency_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    consolidation_rate: float
    throughput_ops_per_sec: float


class MemoryPerformanceMonitor:
    """Monitor and track memory system performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_history: List[MemoryPerformanceMetrics] = []
        self.operation_times: Dict[str, List[float]] = {
            'search': [],
            'store': [],
            'retrieve': [],
            'consolidate': []
        }
        
        logger.info("Initialized MemoryPerformanceMonitor")
    
    def record_operation_time(self, operation: str, duration_ms: float):
        """Record operation timing."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        self.operation_times[operation].append(duration_ms)
        
        # Keep only recent measurements
        if len(self.operation_times[operation]) > 500:
            self.operation_times[operation] = self.operation_times[operation][-500:]
    
    def get_performance_metrics(self) -> MemoryPerformanceMetrics:
        """Calculate current performance metrics."""
        search_times = self.operation_times.get('search', [0])
        store_times = self.operation_times.get('store', [0])
        
        return MemoryPerformanceMetrics(
            search_latency_ms=np.mean(search_times) if search_times else 0.0,
            storage_latency_ms=np.mean(store_times) if store_times else 0.0,
            memory_usage_mb=self._get_memory_usage_mb(),
            cache_hit_rate=self._calculate_cache_hit_rate(),
            consolidation_rate=self._calculate_consolidation_rate(),
            throughput_ops_per_sec=self._calculate_throughput()
        )
    
    def _get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Placeholder implementation
        return 0.85
    
    def _calculate_consolidation_rate(self) -> float:
        """Calculate consolidation rate."""
        # Placeholder implementation
        return 0.1
    
    def _calculate_throughput(self) -> float:
        """Calculate operations per second."""
        total_ops = sum(len(times) for times in self.operation_times.values())
        if total_ops == 0:
            return 0.0
        
        # Estimate based on recent operations
        return min(total_ops / 60.0, 1000.0)  # Ops per minute, capped at 1000


class AdvancedConsolidationAlgorithms:
    """Advanced algorithms for memory consolidation."""
    
    @staticmethod
    async def importance_based_consolidation(
        contexts: List[Any],
        importance_threshold: float = 0.7,
        decay_factor: float = 0.9
    ) -> List[Any]:
        """Advanced importance-based consolidation with temporal decay."""
        current_time = datetime.now()
        candidates = []
        
        for context in contexts:
            # Calculate time-decayed importance
            time_diff = current_time - context.timestamp
            hours_passed = time_diff.total_seconds() / 3600
            
            decayed_importance = context.attention_weight * (decay_factor ** hours_passed)
            
            # Add access frequency boost
            access_boost = min(context.access_count * 0.05, 0.2)
            final_importance = decayed_importance + access_boost
            
            if final_importance >= importance_threshold:
                candidates.append(context)
        
        return candidates
    
    @staticmethod
    async def semantic_clustering_consolidation(
        events: List[Any],
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 3
    ) -> List[List[Any]]:
        """Advanced semantic clustering for episodic to semantic consolidation."""
        if not events:
            return []
        
        # Use hierarchical clustering for better results
        clusters = []
        used_events = set()
        
        for i, event in enumerate(events):
            if i in used_events:
                continue
            
            cluster = [event]
            used_events.add(i)
            
            # Find semantically similar events
            for j, other_event in enumerate(events):
                if j in used_events or i == j:
                    continue
                
                # Calculate semantic similarity
                similarity = await AdvancedConsolidationAlgorithms._calculate_semantic_similarity(
                    event, other_event
                )
                
                if similarity >= similarity_threshold:
                    cluster.append(other_event)
                    used_events.add(j)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    async def _calculate_semantic_similarity(event1: Any, event2: Any) -> float:
        """Calculate semantic similarity between events."""
        # Embedding similarity
        norm1 = np.linalg.norm(event1.embedding)
        norm2 = np.linalg.norm(event2.embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding_sim = np.dot(event1.embedding, event2.embedding) / (norm1 * norm2)
        
        # Context similarity (if available)
        context_sim = 0.0
        if hasattr(event1, 'context') and hasattr(event2, 'context'):
            # Simple context similarity based on common keys
            keys1 = set(event1.context.keys())
            keys2 = set(event2.context.keys())
            if keys1 or keys2:
                context_sim = len(keys1 & keys2) / len(keys1 | keys2)
        
        # Type similarity
        type_sim = 1.0 if event1.event_type == event2.event_type else 0.0
        
        # Weighted combination
        result = 0.6 * embedding_sim + 0.2 * context_sim + 0.2 * type_sim
        return float(result)  # Ensure Python float type


# Factory function for creating optimized memory managers
def create_optimized_memory_manager(
    config: Dict[str, Any]
) -> OptimizedVectorMemoryManager:
    """Factory function to create optimized memory manager based on configuration."""
    backend = config.get('backend', 'faiss')
    embedding_dim = config.get('embedding_dim', 512)
    
    # Determine best backend based on availability and requirements
    if backend == 'auto':
        if FAISS_AVAILABLE:
            backend = 'faiss'
        elif CHROMA_AVAILABLE:
            backend = 'chroma'
        else:
            backend = 'default'
    
    backend_config = config.get('backend_config', {})
    
    # Auto-configure backend based on dataset size
    dataset_size = config.get('expected_dataset_size', 10000)
    if backend == 'faiss' and dataset_size > 100000:
        # Use IVF for large datasets
        backend_config.setdefault('index_type', 'IVFFlat')
        backend_config.setdefault('nlist', min(dataset_size // 100, 4096))
    elif backend == 'faiss':
        # Use flat index for smaller datasets
        backend_config.setdefault('index_type', 'Flat')
    
    return OptimizedVectorMemoryManager(
        embedding_dim=embedding_dim,
        similarity_metric=SimilarityMetric(config.get('similarity_metric', 'cosine')),
        max_capacity=config.get('max_capacity', 10000),
        backend=backend,
        backend_config=backend_config
    )