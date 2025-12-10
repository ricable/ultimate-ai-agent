# File: backend/cache/__init__.py
"""
UAP Caching System

Provides Redis-based caching for agent interactions, document processing,
and other frequently accessed data to improve performance.
"""

from .redis_cache import RedisCache, CacheManager
from .cache_strategies import (
    AgentResponseCache, DocumentCache, 
    LRUCacheStrategy, TTLCacheStrategy
)
from .decorators import cache_result, cache_async_result, invalidate_cache

__all__ = [
    'RedisCache',
    'CacheManager', 
    'AgentResponseCache',
    'DocumentCache',
    'LRUCacheStrategy',
    'TTLCacheStrategy',
    'cache_result',
    'cache_async_result',
    'invalidate_cache'
]