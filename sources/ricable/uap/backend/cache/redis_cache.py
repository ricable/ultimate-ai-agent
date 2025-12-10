# File: backend/cache/redis_cache.py
"""
Redis-based caching implementation for UAP platform.
Provides high-performance caching for agent responses, document processing, and system data.
"""

import json
import asyncio
import hashlib
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import aioredis
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for Redis cache"""
    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    health_check_interval: int = 30
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes

class RedisCache:
    """High-performance Redis cache implementation"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_pool = None
        self.connected = False
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0,
            'last_error': None
        }
    
    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_keepalive=self.config.socket_keepalive,
                decode_responses=False  # We handle encoding ourselves
            )
            
            # Test connection
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                await redis.ping()
            
            self.connected = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            # Continue without cache (graceful degradation)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.connected:
            return None
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                data = await redis.get(key)
                
                if data is None:
                    self.stats['misses'] += 1
                    return None
                
                # Deserialize data
                value = self._deserialize(data)
                self.stats['hits'] += 1
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        if not self.connected:
            return False
        
        try:
            # Use default TTL if not specified
            cache_ttl = ttl or self.config.default_ttl
            
            # Serialize data
            data = self._serialize(value)
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                await redis.setex(key, cache_ttl, data)
                
            self.stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.connected:
            return False
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                result = await redis.delete(key)
                return result > 0
                
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        if not self.connected:
            return 0
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                keys = await redis.keys(pattern)
                if keys:
                    return await redis.delete(*keys)
                return 0
                
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.connected:
            return False
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                return await redis.exists(key) > 0
                
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache"""
        if not self.connected:
            return None
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                return await redis.incrby(key, amount)
                
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not self.connected or not keys:
            return {}
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                values = await redis.mget(keys)
                
                result = {}
                for key, data in zip(keys, values):
                    if data is not None:
                        result[key] = self._deserialize(data)
                        self.stats['hits'] += 1
                    else:
                        self.stats['misses'] += 1
                
                return result
                
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            self.stats['errors'] += 1
            return {}
    
    async def set_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        if not self.connected or not data:
            return False
        
        try:
            cache_ttl = ttl or self.config.default_ttl
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                # Use pipeline for efficiency
                pipe = redis.pipeline()
                
                for key, value in data.items():
                    serialized = self._serialize(value)
                    pipe.setex(key, cache_ttl, serialized)
                
                await pipe.execute()
                
            self.stats['sets'] += len(data)
            return True
            
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            self.stats['errors'] += 1
            return False
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Use JSON for simple types, pickle for complex ones
            if isinstance(value, (str, int, float, bool, list, dict)):
                data = json.dumps(value, default=str).encode('utf-8')
                prefix = b'json:'
            else:
                data = pickle.dumps(value)
                prefix = b'pickle:'
            
            # Apply compression if enabled and data is large enough
            if (self.config.compression_enabled and 
                len(data) > self.config.compression_threshold):
                import zlib
                data = zlib.compress(data)
                prefix = b'zlib:' + prefix
            
            return prefix + data
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Check for compression
            if data.startswith(b'zlib:'):
                import zlib
                data = data[5:]  # Remove prefix
                # Extract the actual prefix after decompression
                if data.startswith(b'json:') or data.startswith(b'pickle:'):
                    decompressed = zlib.decompress(data[6:])  # Remove json:/pickle: prefix
                    prefix = data[:6]
                    data = prefix + decompressed
                else:
                    data = zlib.decompress(data)
            
            # Deserialize based on type
            if data.startswith(b'json:'):
                return json.loads(data[5:].decode('utf-8'))
            elif data.startswith(b'pickle:'):
                return pickle.loads(data[7:])
            else:
                # Legacy format, try JSON first
                try:
                    return json.loads(data.decode('utf-8'))
                except:
                    return pickle.loads(data)
                    
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def flush_all(self) -> bool:
        """Clear all cache data (use with caution)"""
        if not self.connected:
            return False
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                await redis.flushdb()
                return True
                
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            self.stats['errors'] += 1
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        if not self.connected:
            return {}
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                info = await redis.info()
                return info
                
        except Exception as e:
            logger.error(f"Cache info error: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_operations = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_operations * 100) if total_operations > 0 else 0
        
        return {
            'connected': self.connected,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'errors': self.stats['errors'],
            'hit_rate_percent': round(hit_rate, 2),
            'last_error': self.stats['last_error']
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        if not self.connected:
            return {'healthy': False, 'error': 'Not connected'}
        
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                start_time = datetime.utcnow()
                await redis.ping()
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                info = await redis.info('memory')
                memory_usage = info.get('used_memory', 0)
                max_memory = info.get('maxmemory', 0)
                
                return {
                    'healthy': True,
                    'response_time_ms': response_time,
                    'memory_usage_bytes': memory_usage,
                    'max_memory_bytes': max_memory,
                    'memory_usage_percent': (memory_usage / max_memory * 100) if max_memory > 0 else 0
                }
                
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def cleanup(self):
        """Clean up cache connections"""
        if self.redis_pool:
            await self.redis_pool.disconnect()
            self.connected = False
            logger.info("Redis cache connections closed")

class CacheManager:
    """High-level cache manager with different caching strategies"""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache
    
    async def cache_agent_response(self, agent_id: str, message: str, response: Dict[str, Any], ttl: int = 1800):
        """Cache agent response with 30 min TTL"""
        cache_key = self._get_agent_cache_key(agent_id, message)
        await self.redis.set(cache_key, response, ttl)
    
    async def get_cached_agent_response(self, agent_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Get cached agent response"""
        cache_key = self._get_agent_cache_key(agent_id, message)
        return await self.redis.get(cache_key)
    
    async def cache_document_analysis(self, doc_id: str, analysis_type: str, result: Dict[str, Any], ttl: int = 7200):
        """Cache document analysis with 2 hour TTL"""
        cache_key = f"doc:analysis:{doc_id}:{analysis_type}"
        await self.redis.set(cache_key, result, ttl)
    
    async def get_cached_document_analysis(self, doc_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached document analysis"""
        cache_key = f"doc:analysis:{doc_id}:{analysis_type}"
        return await self.redis.get(cache_key)
    
    async def cache_user_session(self, user_id: str, session_data: Dict[str, Any], ttl: int = 3600):
        """Cache user session data"""
        cache_key = f"session:{user_id}"
        await self.redis.set(cache_key, session_data, ttl)
    
    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session"""
        cache_key = f"session:{user_id}"
        return await self.redis.get(cache_key)
    
    async def invalidate_agent_cache(self, agent_id: str):
        """Invalidate all cached responses for an agent"""
        pattern = f"agent:response:{agent_id}:*"
        await self.redis.delete_pattern(pattern)
    
    async def invalidate_document_cache(self, doc_id: str):
        """Invalidate all cached analyses for a document"""
        pattern = f"doc:analysis:{doc_id}:*"
        await self.redis.delete_pattern(pattern)
    
    def _get_agent_cache_key(self, agent_id: str, message: str) -> str:
        """Generate cache key for agent response"""
        # Hash message to handle long/complex messages
        message_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        return f"agent:response:{agent_id}:{message_hash}"
    
    @asynccontextmanager
    async def transaction(self):
        """Cache transaction context manager"""
        if not self.redis.connected:
            yield None
            return
        
        try:
            async with aioredis.Redis(connection_pool=self.redis.redis_pool) as redis:
                pipe = redis.pipeline()
                yield pipe
                await pipe.execute()
        except Exception as e:
            logger.error(f"Cache transaction error: {e}")
            raise

# Global cache instances
_redis_cache = None
_cache_manager = None

async def get_cache() -> RedisCache:
    """Get global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
        await _redis_cache.initialize()
    return _redis_cache

async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        redis_cache = await get_cache()
        _cache_manager = CacheManager(redis_cache)
    return _cache_manager