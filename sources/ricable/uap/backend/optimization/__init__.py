# File: backend/optimization/__init__.py
"""
UAP Performance Optimization Module

Provides comprehensive performance optimization including load balancing, 
database optimization, response optimization, memory management, request batching,
and WebSocket optimization for improved system performance and scalability.
"""

from .load_balancer import (
    LoadBalancer, HealthChecker, ServerInstance, ServerStatus,
    RoundRobinStrategy, WeightedRoundRobinStrategy, 
    LeastConnectionsStrategy, HealthBasedStrategy
)
from .database_optimizer import (
    DatabaseOptimizer, ConnectionPool, QueryOptimizer,
    PostgreSQLAdapter, MySQLAdapter, ConnectionPoolConfig,
    database_optimizer
)
from .response_optimizer import (
    ResponseOptimizer, ResponseCompressor, FieldSelector,
    ResponsePaginator, RequestBatcher as ResponseBatcher,
    CompressionConfig, PaginationConfig, response_optimizer
)
from .memory_manager import (
    MemoryManager, MemoryThresholds, MemorySnapshot,
    ObjectTracker, GarbageCollectionOptimizer, MemoryProfiler,
    ResourceCleaner, memory_manager, track_memory_usage, monitor_memory_growth
)
from .request_batcher import (
    RequestBatcher, BatchConfig, BatchStrategy, RequestPriority,
    BatchRequest, RequestBatch, LoadMonitor, RequestDeduplicator,
    SimilarityGrouper, AdaptiveBatcher, request_batcher, batch_context
)

__all__ = [
    # Load balancing
    'LoadBalancer',
    'HealthChecker', 
    'ServerInstance',
    'ServerStatus',
    'RoundRobinStrategy',
    'WeightedRoundRobinStrategy',
    'LeastConnectionsStrategy',
    'HealthBasedStrategy',
    
    # Database optimization
    'DatabaseOptimizer',
    'ConnectionPool',
    'QueryOptimizer',
    'PostgreSQLAdapter',
    'MySQLAdapter',
    'ConnectionPoolConfig',
    'database_optimizer',
    
    # Response optimization
    'ResponseOptimizer',
    'ResponseCompressor',
    'FieldSelector',
    'ResponsePaginator',
    'ResponseBatcher',
    'CompressionConfig',
    'PaginationConfig',
    'response_optimizer',
    
    # Memory management
    'MemoryManager',
    'MemoryThresholds',
    'MemorySnapshot',
    'ObjectTracker',
    'GarbageCollectionOptimizer',
    'MemoryProfiler',
    'ResourceCleaner',
    'memory_manager',
    'track_memory_usage',
    'monitor_memory_growth',
    
    # Request batching
    'RequestBatcher',
    'BatchConfig',
    'BatchStrategy',
    'RequestPriority',
    'BatchRequest',
    'RequestBatch',
    'LoadMonitor',
    'RequestDeduplicator',
    'SimilarityGrouper',
    'AdaptiveBatcher',
    'request_batcher',
    'batch_context'
]