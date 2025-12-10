"""
Rate Limiting System with Token Bucket Algorithm
Provides flexible rate limiting for API requests
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Types of rate limiting"""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    TOKENS_PER_SECOND = "tokens_per_second"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    CONCURRENT_REQUESTS = "concurrent_requests"

@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting
    """
    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from the bucket
        Returns True if successful, False if not enough tokens
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def peek(self, tokens: float = 1.0) -> bool:
        """
        Check if tokens are available without consuming them
        """
        self._refill()
        return self.tokens >= tokens
    
    def _refill(self) -> None:
        """Refill the bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_wait_time(self, tokens: float = 1.0) -> float:
        """
        Get time to wait before tokens are available
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate
    
    def get_status(self) -> Dict[str, Any]:
        """Get bucket status"""
        self._refill()
        return {
            'capacity': self.capacity,
            'tokens': self.tokens,
            'refill_rate': self.refill_rate,
            'utilization': (self.capacity - self.tokens) / self.capacity
        }

@dataclass
class RateLimit:
    """
    Rate limit configuration
    """
    limit_type: RateLimitType
    limit: float  # limit value (requests, tokens, etc.)
    window_seconds: float
    bucket: Optional[TokenBucket] = None
    
    def __post_init__(self):
        if self.bucket is None:
            # Convert window-based limits to token bucket parameters
            if self.limit_type in [RateLimitType.REQUESTS_PER_SECOND, RateLimitType.TOKENS_PER_SECOND]:
                capacity = self.limit
                refill_rate = self.limit
            elif self.limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.TOKENS_PER_MINUTE]:
                capacity = self.limit
                refill_rate = self.limit / 60.0
            elif self.limit_type == RateLimitType.REQUESTS_PER_HOUR:
                capacity = self.limit
                refill_rate = self.limit / 3600.0
            elif self.limit_type == RateLimitType.CONCURRENT_REQUESTS:
                capacity = self.limit
                refill_rate = float('inf')  # Immediate refill for concurrent limits
            else:
                capacity = self.limit
                refill_rate = self.limit / self.window_seconds
            
            self.bucket = TokenBucket(
                capacity=capacity,
                tokens=capacity,
                refill_rate=refill_rate
            )

@dataclass
class RateLimitResult:
    """
    Result of a rate limit check
    """
    allowed: bool
    limit_type: RateLimitType
    limit: float
    remaining: float
    reset_time: float
    retry_after: Optional[float] = None

class RateLimiter:
    """
    Advanced rate limiter supporting multiple strategies
    """
    
    def __init__(self):
        self.limits: Dict[str, List[RateLimit]] = {}  # key -> [limits]
        self.concurrent_counts: Dict[str, int] = {}  # For concurrent request tracking
        self.lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0
        self.allowed_requests = 0
        self.denied_requests = 0
        self.start_time = time.time()
        
        logger.info("Rate limiter initialized")
    
    def add_limit(
        self,
        key: str,
        limit_type: RateLimitType,
        limit: float,
        window_seconds: Optional[float] = None
    ) -> None:
        """
        Add a rate limit for a specific key
        """
        with self.lock:
            if key not in self.limits:
                self.limits[key] = []
            
            # Set default window based on limit type
            if window_seconds is None:
                if limit_type in [RateLimitType.REQUESTS_PER_SECOND, RateLimitType.TOKENS_PER_SECOND]:
                    window_seconds = 1.0
                elif limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.TOKENS_PER_MINUTE]:
                    window_seconds = 60.0
                elif limit_type == RateLimitType.REQUESTS_PER_HOUR:
                    window_seconds = 3600.0
                else:
                    window_seconds = 60.0  # Default to 1 minute
            
            rate_limit = RateLimit(
                limit_type=limit_type,
                limit=limit,
                window_seconds=window_seconds
            )
            
            self.limits[key].append(rate_limit)
            
            if limit_type == RateLimitType.CONCURRENT_REQUESTS:
                self.concurrent_counts[key] = 0
            
            logger.info(f"Added rate limit for {key}: {limit_type.value} = {limit}")
    
    def check_limit(
        self,
        key: str,
        tokens: float = 1.0,
        request_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limits
        """
        with self.lock:
            self.total_requests += 1
            
            if key not in self.limits:
                # No limits defined for this key
                self.allowed_requests += 1
                return RateLimitResult(
                    allowed=True,
                    limit_type=RateLimitType.REQUESTS_PER_SECOND,
                    limit=float('inf'),
                    remaining=float('inf'),
                    reset_time=time.time()
                )
            
            # Check all limits for this key
            for rate_limit in self.limits[key]:
                if rate_limit.limit_type == RateLimitType.CONCURRENT_REQUESTS:
                    # Handle concurrent requests separately
                    current_count = self.concurrent_counts.get(key, 0)
                    if current_count >= rate_limit.limit:
                        self.denied_requests += 1
                        return RateLimitResult(
                            allowed=False,
                            limit_type=rate_limit.limit_type,
                            limit=rate_limit.limit,
                            remaining=max(0, rate_limit.limit - current_count),
                            reset_time=time.time(),
                            retry_after=1.0  # Retry in 1 second
                        )
                else:
                    # Use token bucket for rate-based limits
                    if not rate_limit.bucket.consume(tokens):
                        self.denied_requests += 1
                        wait_time = rate_limit.bucket.get_wait_time(tokens)
                        
                        return RateLimitResult(
                            allowed=False,
                            limit_type=rate_limit.limit_type,
                            limit=rate_limit.limit,
                            remaining=rate_limit.bucket.tokens,
                            reset_time=time.time() + wait_time,
                            retry_after=wait_time
                        )
            
            # All limits passed
            self.allowed_requests += 1
            
            # For the response, use the most restrictive limit
            most_restrictive = min(
                self.limits[key],
                key=lambda x: x.bucket.tokens / x.bucket.capacity if x.bucket else 1.0
            )
            
            return RateLimitResult(
                allowed=True,
                limit_type=most_restrictive.limit_type,
                limit=most_restrictive.limit,
                remaining=most_restrictive.bucket.tokens if most_restrictive.bucket else float('inf'),
                reset_time=time.time() + (most_restrictive.bucket.get_wait_time(most_restrictive.bucket.capacity) if most_restrictive.bucket else 0)
            )
    
    def start_request(self, key: str, request_id: str) -> bool:
        """
        Start tracking a concurrent request
        """
        with self.lock:
            if key in self.concurrent_counts:
                # Check if under concurrent limit
                for rate_limit in self.limits.get(key, []):
                    if rate_limit.limit_type == RateLimitType.CONCURRENT_REQUESTS:
                        if self.concurrent_counts[key] >= rate_limit.limit:
                            return False
                
                self.concurrent_counts[key] += 1
            
            return True
    
    def end_request(self, key: str, request_id: str) -> None:
        """
        End tracking a concurrent request
        """
        with self.lock:
            if key in self.concurrent_counts:
                self.concurrent_counts[key] = max(0, self.concurrent_counts[key] - 1)
    
    def reset_limits(self, key: str) -> None:
        """
        Reset all limits for a key (admin function)
        """
        with self.lock:
            if key in self.limits:
                for rate_limit in self.limits[key]:
                    if rate_limit.bucket:
                        rate_limit.bucket.tokens = rate_limit.bucket.capacity
                        rate_limit.bucket.last_refill = time.time()
            
            if key in self.concurrent_counts:
                self.concurrent_counts[key] = 0
            
            logger.info(f"Reset rate limits for {key}")
    
    def remove_limits(self, key: str) -> None:
        """
        Remove all limits for a key
        """
        with self.lock:
            if key in self.limits:
                del self.limits[key]
            if key in self.concurrent_counts:
                del self.concurrent_counts[key]
            
            logger.info(f"Removed all rate limits for {key}")
    
    def get_status(self, key: str) -> Dict[str, Any]:
        """
        Get current status for a key
        """
        with self.lock:
            if key not in self.limits:
                return {"error": "No limits defined for key"}
            
            status = {
                "key": key,
                "limits": []
            }
            
            for rate_limit in self.limits[key]:
                limit_status = {
                    "type": rate_limit.limit_type.value,
                    "limit": rate_limit.limit,
                    "window_seconds": rate_limit.window_seconds
                }
                
                if rate_limit.limit_type == RateLimitType.CONCURRENT_REQUESTS:
                    limit_status["current_count"] = self.concurrent_counts.get(key, 0)
                    limit_status["remaining"] = max(0, rate_limit.limit - self.concurrent_counts.get(key, 0))
                elif rate_limit.bucket:
                    bucket_status = rate_limit.bucket.get_status()
                    limit_status.update(bucket_status)
                    limit_status["remaining"] = bucket_status["tokens"]
                
                status["limits"].append(limit_status)
            
            return status
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global rate limiter statistics
        """
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                "uptime": uptime,
                "total_requests": self.total_requests,
                "allowed_requests": self.allowed_requests,
                "denied_requests": self.denied_requests,
                "allow_rate": self.allowed_requests / max(self.total_requests, 1),
                "deny_rate": self.denied_requests / max(self.total_requests, 1),
                "requests_per_second": self.total_requests / max(uptime, 1),
                "total_keys": len(self.limits),
                "concurrent_keys": len(self.concurrent_counts)
            }
    
    def cleanup_expired(self) -> None:
        """
        Clean up unused tracking data
        """
        with self.lock:
            # Reset concurrent counts that have been zero for a while
            # This is a simple cleanup - in production you might want more sophisticated logic
            keys_to_reset = []
            for key, count in self.concurrent_counts.items():
                if count == 0:
                    # Check if any recent activity in buckets
                    has_recent_activity = False
                    for rate_limit in self.limits.get(key, []):
                        if rate_limit.bucket and rate_limit.bucket.tokens < rate_limit.bucket.capacity:
                            has_recent_activity = True
                            break
                    
                    if not has_recent_activity:
                        keys_to_reset.append(key)
            
            logger.info(f"Cleaned up {len(keys_to_reset)} inactive rate limit keys")

class AdaptiveRateLimiter(RateLimiter):
    """
    Advanced rate limiter that adapts based on system load
    """
    
    def __init__(self, base_capacity_factor: float = 1.0):
        super().__init__()
        self.base_capacity_factor = base_capacity_factor
        self.system_load_factor = 1.0
        self.adaptation_enabled = True
        
        # Load tracking
        self.recent_response_times: List[float] = []
        self.recent_error_rates: List[float] = []
        self.adaptation_window = 300  # 5 minutes
        
    def update_system_metrics(
        self,
        avg_response_time: float,
        error_rate: float,
        cpu_usage: float,
        memory_usage: float
    ) -> None:
        """
        Update system metrics for adaptive rate limiting
        """
        if not self.adaptation_enabled:
            return
        
        # Track recent metrics
        self.recent_response_times.append(avg_response_time)
        self.recent_error_rates.append(error_rate)
        
        # Keep only recent data
        cutoff = time.time() - self.adaptation_window
        # For simplicity, just keep last 10 measurements
        if len(self.recent_response_times) > 10:
            self.recent_response_times = self.recent_response_times[-10:]
            self.recent_error_rates = self.recent_error_rates[-10:]
        
        # Calculate adaptive factor
        response_factor = 1.0
        error_factor = 1.0
        resource_factor = 1.0
        
        # Response time factor (slower responses = lower capacity)
        if self.recent_response_times:
            avg_response = sum(self.recent_response_times) / len(self.recent_response_times)
            if avg_response > 2.0:  # > 2 seconds
                response_factor = 0.5
            elif avg_response > 1.0:  # > 1 second
                response_factor = 0.7
            elif avg_response > 0.5:  # > 500ms
                response_factor = 0.9
        
        # Error rate factor (high errors = lower capacity)
        if self.recent_error_rates:
            avg_error_rate = sum(self.recent_error_rates) / len(self.recent_error_rates)
            if avg_error_rate > 0.1:  # > 10% errors
                error_factor = 0.3
            elif avg_error_rate > 0.05:  # > 5% errors
                error_factor = 0.6
            elif avg_error_rate > 0.02:  # > 2% errors
                error_factor = 0.8
        
        # Resource factor (high resource usage = lower capacity)
        max_resource_usage = max(cpu_usage, memory_usage)
        if max_resource_usage > 90:
            resource_factor = 0.3
        elif max_resource_usage > 80:
            resource_factor = 0.5
        elif max_resource_usage > 70:
            resource_factor = 0.7
        elif max_resource_usage > 60:
            resource_factor = 0.9
        
        # Combine factors
        new_factor = min(response_factor, error_factor, resource_factor)
        
        # Smooth the transition
        self.system_load_factor = (self.system_load_factor * 0.8) + (new_factor * 0.2)
        
        # Apply to all buckets
        with self.lock:
            for key, rate_limits in self.limits.items():
                for rate_limit in rate_limits:
                    if rate_limit.bucket and rate_limit.limit_type != RateLimitType.CONCURRENT_REQUESTS:
                        # Adjust capacity and refill rate
                        base_capacity = rate_limit.limit * self.base_capacity_factor
                        adjusted_capacity = base_capacity * self.system_load_factor
                        
                        rate_limit.bucket.capacity = adjusted_capacity
                        rate_limit.bucket.refill_rate = rate_limit.bucket.refill_rate * self.system_load_factor
        
        logger.debug(f"Adaptive rate limiting factor: {self.system_load_factor:.3f}")

# Factory functions and utilities
def create_rate_limiter(adaptive: bool = False) -> RateLimiter:
    """Create a rate limiter"""
    if adaptive:
        return AdaptiveRateLimiter()
    return RateLimiter()

def create_common_limits(rate_limiter: RateLimiter, key: str, user_tier: str = "standard") -> None:
    """
    Add common rate limits based on user tier
    """
    if user_tier == "free":
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_MINUTE, 60)
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_HOUR, 1000)
        rate_limiter.add_limit(key, RateLimitType.CONCURRENT_REQUESTS, 2)
        rate_limiter.add_limit(key, RateLimitType.TOKENS_PER_MINUTE, 10000)
    
    elif user_tier == "standard":
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_MINUTE, 300)
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_HOUR, 10000)
        rate_limiter.add_limit(key, RateLimitType.CONCURRENT_REQUESTS, 10)
        rate_limiter.add_limit(key, RateLimitType.TOKENS_PER_MINUTE, 50000)
    
    elif user_tier == "premium":
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_MINUTE, 1000)
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_HOUR, 50000)
        rate_limiter.add_limit(key, RateLimitType.CONCURRENT_REQUESTS, 50)
        rate_limiter.add_limit(key, RateLimitType.TOKENS_PER_MINUTE, 200000)
    
    elif user_tier == "enterprise":
        rate_limiter.add_limit(key, RateLimitType.REQUESTS_PER_MINUTE, 5000)
        rate_limiter.add_limit(key, RateLimitType.CONCURRENT_REQUESTS, 200)
        rate_limiter.add_limit(key, RateLimitType.TOKENS_PER_MINUTE, 1000000)

# Example usage
if __name__ == "__main__":
    # Create rate limiter
    limiter = create_rate_limiter()
    
    # Add some limits
    create_common_limits(limiter, "user:123", "standard")
    create_common_limits(limiter, "user:456", "premium")
    
    # Test rate limiting
    for i in range(10):
        result = limiter.check_limit("user:123", tokens=100)
        print(f"Request {i+1}: allowed={result.allowed}, remaining={result.remaining:.1f}")
        
        if not result.allowed:
            print(f"Rate limited! Retry after {result.retry_after:.2f} seconds")
            break
        
        time.sleep(0.1)
    
    # Show status
    status = limiter.get_status("user:123")
    print(f"\nUser 123 status: {json.dumps(status, indent=2)}")
    
    stats = limiter.get_global_stats()
    print(f"\nGlobal stats: {json.dumps(stats, indent=2)}")