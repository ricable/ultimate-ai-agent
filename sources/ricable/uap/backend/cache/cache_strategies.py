# File: backend/cache/cache_strategies.py
"""
Caching strategies for different types of data in UAP platform.
Implements LRU, TTL, and specialized caching strategies for agents and documents.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def touch(self):
        """Update access time and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

class CacheStrategy(ABC):
    """Abstract base class for cache strategies"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

class LRUCacheStrategy(CacheStrategy):
    """Least Recently Used cache strategy"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from LRU cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.touch()
                
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in LRU cache"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict oldest items if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
            
            # Add new entry
            entry = CacheEntry(value=value, ttl_seconds=ttl)
            self.cache[key] = entry
            
            self.stats['sets'] += 1
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from LRU cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_ops = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_ops * 100) if total_ops > 0 else 0
            
            return {
                'strategy': 'LRU',
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate_percent': round(hit_rate, 2),
                **self.stats
            }

class TTLCacheStrategy(CacheStrategy):
    """Time-To-Live cache strategy"""
    
    def __init__(self, default_ttl: int = 3600, cleanup_interval: int = 300):
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'expirations': 0,
            'sets': 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup()
    
    def _start_cleanup(self):
        """Start background cleanup task"""
        async def cleanup_expired():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"TTL cache cleanup error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired())
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.stats['expirations'] += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from TTL cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['misses'] += 1
                    self.stats['expirations'] += 1
                    return None
                
                entry.touch()
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in TTL cache"""
        with self.lock:
            cache_ttl = ttl or self.default_ttl
            entry = CacheEntry(value=value, ttl_seconds=cache_ttl)
            self.cache[key] = entry
            
            self.stats['sets'] += 1
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from TTL cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_ops = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_ops * 100) if total_ops > 0 else 0
            
            return {
                'strategy': 'TTL',
                'size': len(self.cache),
                'default_ttl': self.default_ttl,
                'hit_rate_percent': round(hit_rate, 2),
                **self.stats
            }
    
    def stop_cleanup(self):
        """Stop cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()

class AgentResponseCache:
    """Specialized cache for agent responses with intelligent invalidation"""
    
    def __init__(self, redis_cache, max_local_size: int = 500):
        self.redis = redis_cache
        self.local_cache = LRUCacheStrategy(max_local_size)
        self.cache_rules = {
            'copilot': {'ttl': 1800, 'local': True},   # 30 min, use local cache
            'agno': {'ttl': 7200, 'local': True},      # 2 hours, use local cache  
            'mastra': {'ttl': 3600, 'local': False}    # 1 hour, Redis only
        }
    
    async def get_response(self, agent_id: str, framework: str, message_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached agent response with multi-level caching"""
        cache_key = f"agent:{framework}:{agent_id}:{message_hash}"
        
        # Check local cache first if enabled for this framework
        if self.cache_rules.get(framework, {}).get('local', False):
            local_result = await self.local_cache.get(cache_key)
            if local_result is not None:
                return local_result
        
        # Check Redis cache
        redis_result = await self.redis.get(cache_key)
        
        # Update local cache if we got a Redis hit
        if redis_result and self.cache_rules.get(framework, {}).get('local', False):
            await self.local_cache.set(cache_key, redis_result, 
                                     self.cache_rules[framework]['ttl'])
        
        return redis_result
    
    async def cache_response(self, agent_id: str, framework: str, message_hash: str, 
                           response: Dict[str, Any]) -> bool:
        """Cache agent response with framework-specific rules"""
        cache_key = f"agent:{framework}:{agent_id}:{message_hash}"
        ttl = self.cache_rules.get(framework, {}).get('ttl', 3600)
        
        # Cache in Redis
        redis_success = await self.redis.set(cache_key, response, ttl)
        
        # Cache locally if enabled
        if self.cache_rules.get(framework, {}).get('local', False):
            await self.local_cache.set(cache_key, response, ttl)
        
        return redis_success
    
    async def invalidate_agent(self, agent_id: str, framework: str = None):
        """Invalidate all cached responses for an agent"""
        if framework:
            pattern = f"agent:{framework}:{agent_id}:*"
        else:
            pattern = f"agent:*:{agent_id}:*"
        
        # Clear from Redis
        await self.redis.delete_pattern(pattern)
        
        # Clear from local cache
        await self.local_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        redis_stats = self.redis.get_stats()
        local_stats = self.local_cache.get_stats()
        
        return {
            'agent_cache': {
                'redis': redis_stats,
                'local': local_stats,
                'cache_rules': self.cache_rules
            }
        }

class DocumentCache:
    """Specialized cache for document processing results"""
    
    def __init__(self, redis_cache):
        self.redis = redis_cache
        self.cache_policies = {
            'document_content': {'ttl': 86400, 'compress': True},      # 24 hours
            'document_analysis': {'ttl': 7200, 'compress': False},     # 2 hours
            'document_metadata': {'ttl': 43200, 'compress': False},    # 12 hours
            'document_thumbnails': {'ttl': 604800, 'compress': True}   # 1 week
        }
    
    async def cache_document_content(self, doc_id: str, content: Any) -> bool:
        """Cache document content with long TTL"""
        cache_key = f"doc:content:{doc_id}"
        policy = self.cache_policies['document_content']
        return await self.redis.set(cache_key, content, policy['ttl'])
    
    async def get_document_content(self, doc_id: str) -> Optional[Any]:
        """Get cached document content"""
        cache_key = f"doc:content:{doc_id}"
        return await self.redis.get(cache_key)
    
    async def cache_analysis_result(self, doc_id: str, analysis_type: str, result: Dict[str, Any]) -> bool:
        """Cache document analysis result"""
        cache_key = f"doc:analysis:{doc_id}:{analysis_type}"
        policy = self.cache_policies['document_analysis']
        return await self.redis.set(cache_key, result, policy['ttl'])
    
    async def get_analysis_result(self, doc_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        cache_key = f"doc:analysis:{doc_id}:{analysis_type}"
        return await self.redis.get(cache_key)
    
    async def cache_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Cache document metadata"""
        cache_key = f"doc:metadata:{doc_id}"
        policy = self.cache_policies['document_metadata']
        return await self.redis.set(cache_key, metadata, policy['ttl'])
    
    async def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document metadata"""
        cache_key = f"doc:metadata:{doc_id}"
        return await self.redis.get(cache_key)
    
    async def invalidate_document(self, doc_id: str):
        """Invalidate all cached data for a document"""
        patterns = [
            f"doc:content:{doc_id}",
            f"doc:analysis:{doc_id}:*",
            f"doc:metadata:{doc_id}",
            f"doc:thumbnails:{doc_id}:*"
        ]
        
        for pattern in patterns:
            await self.redis.delete_pattern(pattern)
    
    async def get_cache_size_by_type(self) -> Dict[str, int]:
        """Get cache size breakdown by document type"""
        # This would require scanning Redis keys, which can be expensive
        # In production, consider using Redis modules or separate counters
        return {
            'content_items': 0,  # Placeholder
            'analysis_items': 0,
            'metadata_items': 0,
            'thumbnail_items': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document cache statistics"""
        return {
            'document_cache': {
                'policies': self.cache_policies,
                'redis_stats': self.redis.get_stats()
            }
        }

class AdaptiveCacheStrategy:
    """Adaptive cache strategy that adjusts based on usage patterns"""
    
    def __init__(self, redis_cache, analysis_window: int = 3600):
        self.redis = redis_cache
        self.analysis_window = analysis_window  # seconds
        self.access_patterns = {}  # key -> [access_times]
        self.dynamic_ttl = {}      # key -> calculated_ttl
        self.min_ttl = 300         # 5 minutes
        self.max_ttl = 86400       # 24 hours
    
    async def smart_cache(self, key: str, value: Any, base_ttl: int = 3600) -> bool:
        """Cache with adaptive TTL based on access patterns"""
        # Analyze access pattern
        adaptive_ttl = self._calculate_adaptive_ttl(key, base_ttl)
        
        # Cache with adaptive TTL
        success = await self.redis.set(key, value, adaptive_ttl)
        
        if success:
            self.dynamic_ttl[key] = adaptive_ttl
        
        return success
    
    async def smart_get(self, key: str) -> Optional[Any]:
        """Get with access pattern tracking"""
        # Record access
        now = time.time()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        self.access_patterns[key].append(now)
        
        # Clean old access records
        cutoff = now - self.analysis_window
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff]
        
        return await self.redis.get(key)
    
    def _calculate_adaptive_ttl(self, key: str, base_ttl: int) -> int:
        """Calculate adaptive TTL based on access patterns"""
        if key not in self.access_patterns:
            return base_ttl
        
        access_times = self.access_patterns[key]
        if len(access_times) < 2:
            return base_ttl
        
        # Calculate access frequency (accesses per hour)
        time_span = access_times[-1] - access_times[0]
        if time_span == 0:
            return base_ttl
        
        access_frequency = len(access_times) / (time_span / 3600)  # per hour
        
        # Adjust TTL based on frequency
        if access_frequency > 10:  # High frequency
            adaptive_ttl = min(base_ttl * 2, self.max_ttl)
        elif access_frequency > 1:  # Medium frequency
            adaptive_ttl = base_ttl
        else:  # Low frequency
            adaptive_ttl = max(base_ttl // 2, self.min_ttl)
        
        return adaptive_ttl
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptive cache statistics"""
        total_keys = len(self.access_patterns)
        avg_ttl = sum(self.dynamic_ttl.values()) / len(self.dynamic_ttl) if self.dynamic_ttl else 0
        
        frequency_distribution = {
            'high_frequency': 0,
            'medium_frequency': 0,
            'low_frequency': 0
        }
        
        for key, access_times in self.access_patterns.items():
            if len(access_times) >= 2:
                time_span = access_times[-1] - access_times[0]
                if time_span > 0:
                    frequency = len(access_times) / (time_span / 3600)
                    if frequency > 10:
                        frequency_distribution['high_frequency'] += 1
                    elif frequency > 1:
                        frequency_distribution['medium_frequency'] += 1
                    else:
                        frequency_distribution['low_frequency'] += 1
        
        return {
            'adaptive_cache': {
                'total_tracked_keys': total_keys,
                'average_ttl': round(avg_ttl, 2),
                'frequency_distribution': frequency_distribution,
                'ttl_range': {
                    'min': self.min_ttl,
                    'max': self.max_ttl
                }
            }
        }