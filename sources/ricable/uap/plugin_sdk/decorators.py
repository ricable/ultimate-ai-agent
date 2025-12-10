# File: plugin_sdk/decorators.py
"""
Decorators for plugin development.
"""

import asyncio
import time
from functools import wraps
from typing import List, Callable, Any, Dict
from datetime import datetime, timedelta

from .base import PluginResponse, PluginContext


def action(name: str = None, description: str = None, permissions: List[str] = None):
    """
    Decorator to mark a method as a plugin action.
    
    Args:
        name: Action name (defaults to method name)
        description: Action description
        permissions: Required permissions for this action
    """
    def decorator(func):
        # Store action metadata
        func._is_action = True
        func._action_name = name or func.__name__
        func._action_description = description or func.__doc__ or "No description"
        func._action_permissions = permissions or []
        
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Log action execution
            if hasattr(self, '_update_metrics'):
                self._update_metrics(f"action_{func._action_name}_calls", 
                                   self._metrics.get(f"action_{func._action_name}_calls", 0) + 1)
            
            return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def permission_required(*permissions: str):
    """
    Decorator to require specific permissions for a method.
    
    Args:
        *permissions: Required permission names
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Extract context from arguments
            context = None
            for arg in args:
                if isinstance(arg, PluginContext):
                    context = arg
                    break
            
            if context and hasattr(self, 'validate_permissions'):
                if not await self.validate_permissions(list(permissions), context):
                    return PluginResponse(
                        success=False,
                        error=f"Missing required permissions: {', '.join(permissions)}",
                        error_code="permission_denied"
                    )
            
            return await func(self, *args, **kwargs)
        
        wrapper._required_permissions = permissions
        return wrapper
    return decorator


def rate_limit(calls_per_minute: int = 60, per_user: bool = True):
    """
    Decorator to rate limit method calls.
    
    Args:
        calls_per_minute: Maximum calls per minute
        per_user: Whether to apply rate limiting per user
    """
    def decorator(func):
        # Store rate limit state
        if not hasattr(func, '_rate_limit_state'):
            func._rate_limit_state = {}
        
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            current_time = time.time()
            
            # Extract user ID from context
            user_id = "global"
            if per_user:
                for arg in args:
                    if isinstance(arg, PluginContext) and arg.user_id:
                        user_id = arg.user_id
                        break
            
            # Create rate limit key
            rate_key = f"{self.plugin_id}:{func.__name__}:{user_id}"
            
            # Check rate limit
            if rate_key in func._rate_limit_state:
                last_calls = func._rate_limit_state[rate_key]
                # Remove calls older than 1 minute
                recent_calls = [call_time for call_time in last_calls 
                              if current_time - call_time < 60]
                
                if len(recent_calls) >= calls_per_minute:
                    return PluginResponse(
                        success=False,
                        error=f"Rate limit exceeded: {calls_per_minute} calls per minute",
                        error_code="rate_limit_exceeded",
                        metadata={
                            "rate_limit": calls_per_minute,
                            "reset_time": min(recent_calls) + 60
                        }
                    )
                
                func._rate_limit_state[rate_key] = recent_calls
            else:
                func._rate_limit_state[rate_key] = []
            
            # Record this call
            func._rate_limit_state[rate_key].append(current_time)
            
            return await func(self, *args, **kwargs)
        
        wrapper._rate_limit = {
            "calls_per_minute": calls_per_minute,
            "per_user": per_user
        }
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a method on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    
                    # If result is a PluginResponse and successful, return it
                    if isinstance(result, PluginResponse):
                        if result.success or attempt == max_attempts - 1:
                            return result
                        # If not successful and we have more attempts, retry
                        last_exception = Exception(result.error)
                    else:
                        return result
                        
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                
                # Wait before retry
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # All attempts failed
            if hasattr(args[0], '_format_error_response'):
                return args[0]._format_error_response(
                    last_exception or Exception("All retry attempts failed"),
                    "retry_attempts_exhausted"
                )
            else:
                raise last_exception or Exception("All retry attempts failed")
        
        wrapper._retry_config = {
            "max_attempts": max_attempts,
            "delay": delay,
            "backoff": backoff
        }
        return wrapper
    return decorator


def cache_result(ttl_seconds: int = 300, per_user: bool = True):
    """
    Decorator to cache method results.
    
    Args:
        ttl_seconds: Time to live for cached results
        per_user: Whether to cache per user
    """
    def decorator(func):
        if not hasattr(func, '_cache'):
            func._cache = {}
        
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            current_time = time.time()
            
            # Extract user ID from context
            user_id = "global"
            if per_user:
                for arg in args:
                    if isinstance(arg, PluginContext) and arg.user_id:
                        user_id = arg.user_id
                        break
            
            # Create cache key
            cache_key = f"{self.plugin_id}:{func.__name__}:{user_id}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            if cache_key in func._cache:
                cached_result, cache_time = func._cache[cache_key]
                if current_time - cache_time < ttl_seconds:
                    # Update metrics
                    if hasattr(self, '_update_metrics'):
                        self._update_metrics("cache_hits", 
                                           self._metrics.get("cache_hits", 0) + 1)
                    return cached_result
            
            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            func._cache[cache_key] = (result, current_time)
            
            # Update metrics
            if hasattr(self, '_update_metrics'):
                self._update_metrics("cache_misses", 
                                   self._metrics.get("cache_misses", 0) + 1)
            
            # Clean old cache entries
            expired_keys = [key for key, (_, cache_time) in func._cache.items() 
                          if current_time - cache_time >= ttl_seconds]
            for key in expired_keys:
                del func._cache[key]
            
            return result
        
        wrapper._cache_config = {
            "ttl_seconds": ttl_seconds,
            "per_user": per_user
        }
        return wrapper
    return decorator


def validate_input(schema: Dict[str, Any]):
    """
    Decorator to validate input parameters against a JSON schema.
    
    Args:
        schema: JSON schema for validation
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # For now, just store the schema - in a real implementation,
            # you would use a JSON schema validation library
            # like jsonschema to validate the inputs
            
            # Simple validation example
            if 'params' in kwargs:
                params = kwargs['params']
                if isinstance(params, dict):
                    # Check required fields
                    required = schema.get('required', [])
                    for field in required:
                        if field not in params:
                            return PluginResponse(
                                success=False,
                                error=f"Missing required field: {field}",
                                error_code="validation_failed"
                            )
            
            return await func(self, *args, **kwargs)
        
        wrapper._input_schema = schema
        return wrapper
    return decorator


def log_execution(include_params: bool = False, include_result: bool = False):
    """
    Decorator to log method execution.
    
    Args:
        include_params: Whether to log parameters
        include_result: Whether to log results
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            # Log start
            log_data = {
                "plugin_id": self.plugin_id,
                "method": func.__name__,
                "start_time": datetime.now().isoformat()
            }
            
            if include_params:
                log_data["params"] = {"args": str(args), "kwargs": str(kwargs)}
            
            try:
                result = await func(self, *args, **kwargs)
                
                # Log success
                execution_time = time.time() - start_time
                log_data.update({
                    "status": "success",
                    "execution_time_seconds": execution_time,
                    "end_time": datetime.now().isoformat()
                })
                
                if include_result:
                    log_data["result"] = str(result)
                
                # Update metrics
                if hasattr(self, '_update_metrics'):
                    self._update_metrics(f"{func.__name__}_execution_time", execution_time)
                    self._update_metrics(f"{func.__name__}_calls", 
                                       self._metrics.get(f"{func.__name__}_calls", 0) + 1)
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                log_data.update({
                    "status": "error",
                    "error": str(e),
                    "execution_time_seconds": execution_time,
                    "end_time": datetime.now().isoformat()
                })
                
                # Update metrics
                if hasattr(self, '_update_metrics'):
                    self._update_metrics(f"{func.__name__}_errors", 
                                       self._metrics.get(f"{func.__name__}_errors", 0) + 1)
                
                raise
        
        wrapper._log_config = {
            "include_params": include_params,
            "include_result": include_result
        }
        return wrapper
    return decorator