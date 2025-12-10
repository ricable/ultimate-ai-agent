# File: backend/cache/decorators.py
"""
Cache decorators for easy function-level caching in UAP platform.
Provides synchronous and asynchronous decorators with various caching strategies.
"""

import hashlib
import asyncio
import functools
import json
import inspect
from typing import Any, Callable, Optional, Union, Dict
from datetime import datetime
import logging

from .redis_cache import get_cache, get_cache_manager

logger = logging.getLogger(__name__)

def cache_key_generator(func_name: str, args: tuple, kwargs: dict, 
                       exclude_args: list = None, key_prefix: str = None) -> str:
    """Generate cache key from function name and arguments"""
    exclude_args = exclude_args or []
    
    # Filter out excluded arguments
    filtered_args = []
    filtered_kwargs = {}
    
    # Get function signature to map args to parameter names
    sig = inspect.signature(func_name) if hasattr(func_name, '__name__') else None
    
    if sig:
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, value in bound_args.arguments.items():
            if param_name not in exclude_args:
                filtered_kwargs[param_name] = value
    else:
        # Fallback: use positional args and kwargs as-is
        filtered_args = [arg for i, arg in enumerate(args) if i not in exclude_args]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_args}
    
    # Create cache key
    key_data = {
        'function': func_name.__name__ if hasattr(func_name, '__name__') else str(func_name),
        'args': filtered_args,
        'kwargs': filtered_kwargs
    }
    
    # Serialize and hash
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    # Add prefix if specified
    if key_prefix:
        return f"{key_prefix}:{key_hash}"
    return f"func:{key_hash}"

def cache_result(ttl: int = 3600, key_prefix: str = None, exclude_args: list = None,
                condition: Callable = None, cache_empty: bool = True):
    """
    Decorator for caching synchronous function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        exclude_args: Arguments to exclude from cache key generation
        condition: Function to determine if result should be cached
        cache_empty: Whether to cache empty/None results
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_generator(func, args, kwargs, exclude_args, key_prefix)
            
            try:
                # Try to get from cache
                cache = await get_cache()
                cached_result = await cache.get(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                    return cached_result
                
                logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Check caching condition
                should_cache = True
                if condition and not condition(result):
                    should_cache = False
                
                if not cache_empty and (result is None or result == ""):
                    should_cache = False
                
                # Cache result
                if should_cache:
                    await cache.set(cache_key, result, ttl)
                    logger.debug(f"Cached result for {func.__name__}: {cache_key}")
                
                return result
                
            except Exception as e:
                logger.error(f"Cache error in {func.__name__}: {e}")
                # Execute function without cache on error
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle async cache operations
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except RuntimeError:
                # No event loop, execute without cache
                logger.warning(f"No event loop for caching {func.__name__}, executing without cache")
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cache_async_result(ttl: int = 3600, key_prefix: str = None, exclude_args: list = None,
                      condition: Callable = None, cache_empty: bool = True):
    """
    Decorator for caching asynchronous function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        exclude_args: Arguments to exclude from cache key generation
        condition: Function to determine if result should be cached
        cache_empty: Whether to cache empty/None results
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_generator(func, args, kwargs, exclude_args, key_prefix)
            
            try:
                # Try to get from cache
                cache = await get_cache()
                cached_result = await cache.get(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                    return cached_result
                
                logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Check caching condition
                should_cache = True
                if condition and not condition(result):
                    should_cache = False
                
                if not cache_empty and (result is None or result == ""):
                    should_cache = False
                
                # Cache result
                if should_cache:
                    await cache.set(cache_key, result, ttl)
                    logger.debug(f"Cached result for {func.__name__}: {cache_key}")
                
                return result
                
            except Exception as e:
                logger.error(f"Cache error in {func.__name__}: {e}")
                # Execute function without cache on error
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def invalidate_cache(pattern: str = None, key_prefix: str = None):
    """
    Decorator to invalidate cache entries after function execution
    
    Args:
        pattern: Pattern to match for cache invalidation
        key_prefix: Prefix for cache key invalidation
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function first
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            try:
                # Invalidate cache
                cache = await get_cache()
                
                if pattern:
                    await cache.delete_pattern(pattern)
                    logger.debug(f"Invalidated cache pattern: {pattern}")
                elif key_prefix:
                    invalidation_pattern = f"{key_prefix}:*"
                    await cache.delete_pattern(invalidation_pattern)
                    logger.debug(f"Invalidated cache prefix: {key_prefix}")
                else:
                    # Generate key for this specific function call
                    cache_key = cache_key_generator(func, args, kwargs)
                    await cache.delete(cache_key)
                    logger.debug(f"Invalidated cache key: {cache_key}")
                
            except Exception as e:
                logger.error(f"Cache invalidation error in {func.__name__}: {e}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except RuntimeError:
                logger.warning(f"No event loop for cache invalidation in {func.__name__}")
                return func(*args, **kwargs)
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cached_property(ttl: int = 3600, key_prefix: str = None):
    """
    Property decorator that caches the result for a specified time
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        cache_attr = f"_cached_{func.__name__}"
        cache_time_attr = f"_cached_time_{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(self):
            now = datetime.utcnow()
            
            # Check if we have a cached value and it's still valid
            if hasattr(self, cache_attr) and hasattr(self, cache_time_attr):
                cached_time = getattr(self, cache_time_attr)
                if (now - cached_time).total_seconds() < ttl:
                    return getattr(self, cache_attr)
            
            # Compute new value
            result = func(self)
            
            # Cache the result
            setattr(self, cache_attr, result)
            setattr(self, cache_time_attr, now)
            
            return result
        
        return wrapper
    return decorator

def cache_with_lock(ttl: int = 3600, lock_timeout: int = 30, key_prefix: str = None):
    """
    Decorator that prevents cache stampede by using distributed locks
    
    Args:
        ttl: Time to live in seconds
        lock_timeout: Lock timeout in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache_key_generator(func, args, kwargs, key_prefix=key_prefix)
            lock_key = f"lock:{cache_key}"
            
            try:
                cache = await get_cache()
                
                # Try to get from cache first
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Try to acquire lock
                lock_acquired = await cache.set(lock_key, "locked", lock_timeout)
                
                if lock_acquired:
                    try:
                        # Double-check cache after acquiring lock
                        cached_result = await cache.get(cache_key)
                        if cached_result is not None:
                            return cached_result
                        
                        # Execute function
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                        
                        # Cache result
                        await cache.set(cache_key, result, ttl)
                        return result
                        
                    finally:
                        # Release lock
                        await cache.delete(lock_key)
                else:
                    # Couldn't acquire lock, wait a bit and try cache again
                    await asyncio.sleep(0.1)
                    cached_result = await cache.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                    
                    # Still no cache hit, execute without lock
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Cache with lock error in {func.__name__}: {e}")
                # Execute function without cache on error
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator

def conditional_cache(condition_func: Callable, ttl: int = 3600, key_prefix: str = None):
    """
    Decorator that caches results based on a condition function
    
    Args:
        condition_func: Function that takes (args, kwargs, result) and returns bool
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache_key_generator(func, args, kwargs, key_prefix=key_prefix)
            
            try:
                cache = await get_cache()
                
                # Try to get from cache
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Check condition for caching
                if condition_func(args, kwargs, result):
                    await cache.set(cache_key, result, ttl)
                
                return result
                
            except Exception as e:
                logger.error(f"Conditional cache error in {func.__name__}: {e}")
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Convenience decorators for common use cases
def cache_agent_response(ttl: int = 1800):
    """Cache agent responses for 30 minutes"""
    return cache_async_result(ttl=ttl, key_prefix="agent_response")

def cache_document_analysis(ttl: int = 7200):
    """Cache document analysis for 2 hours"""
    return cache_async_result(ttl=ttl, key_prefix="doc_analysis")

def cache_system_status(ttl: int = 60):
    """Cache system status for 1 minute"""
    return cache_async_result(ttl=ttl, key_prefix="system_status")

def cache_user_data(ttl: int = 3600):
    """Cache user data for 1 hour"""
    return cache_async_result(ttl=ttl, key_prefix="user_data", exclude_args=['password'])

# Example usage:
"""
@cache_async_result(ttl=1800, key_prefix="expensive_computation")
async def expensive_computation(param1, param2):
    # Expensive operation here
    await asyncio.sleep(2)
    return f"Result for {param1}, {param2}"

@cache_with_lock(ttl=3600, lock_timeout=30)
async def critical_section_operation(data):
    # Operation that should not run concurrently
    return process_data(data)

@conditional_cache(
    condition_func=lambda args, kwargs, result: len(result) > 10,
    ttl=1800
)
async def fetch_large_data(query):
    # Only cache if result is large enough
    return perform_database_query(query)
"""