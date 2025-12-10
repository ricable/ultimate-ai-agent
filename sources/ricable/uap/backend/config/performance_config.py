# File: backend/config/performance_config.py
"""
Performance configuration for UAP platform.
Manages caching, load balancing, and database optimization settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RedisConfig:
    """Redis caching configuration"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    db: int = int(os.getenv("REDIS_DB", "0"))
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    health_check_interval: int = 30
    default_ttl: int = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))
    compression_enabled: bool = True
    compression_threshold: int = 1024
    
    @property
    def url(self) -> str:
        """Generate Redis URL"""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"

@dataclass
class DatabaseConfig:
    """Database optimization configuration"""
    enable_connection_pooling: bool = True
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "30"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    enable_query_cache: bool = True
    query_cache_size: int = int(os.getenv("DB_QUERY_CACHE_SIZE", "1000"))
    slow_query_threshold: float = float(os.getenv("DB_SLOW_QUERY_THRESHOLD", "1.0"))
    enable_read_replicas: bool = os.getenv("DB_ENABLE_READ_REPLICAS", "false").lower() == "true"

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    strategy: str = os.getenv("LB_STRATEGY", "health_based")  # round_robin, weighted, least_connections, health_based
    health_check_interval: int = int(os.getenv("LB_HEALTH_CHECK_INTERVAL", "30"))
    health_check_timeout: int = int(os.getenv("LB_HEALTH_CHECK_TIMEOUT", "5"))
    health_threshold: float = float(os.getenv("LB_HEALTH_THRESHOLD", "0.7"))
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

@dataclass
class CacheConfig:
    """Application-level caching configuration"""
    # Agent response caching
    agent_response_ttl: int = int(os.getenv("CACHE_AGENT_RESPONSE_TTL", "1800"))  # 30 minutes
    agent_response_enabled: bool = True
    
    # Document analysis caching
    document_analysis_ttl: int = int(os.getenv("CACHE_DOCUMENT_ANALYSIS_TTL", "7200"))  # 2 hours
    document_content_ttl: int = int(os.getenv("CACHE_DOCUMENT_CONTENT_TTL", "86400"))  # 24 hours
    
    # System status caching
    system_status_ttl: int = int(os.getenv("CACHE_SYSTEM_STATUS_TTL", "60"))  # 1 minute
    
    # User session caching
    user_session_ttl: int = int(os.getenv("CACHE_USER_SESSION_TTL", "3600"))  # 1 hour
    
    # Cache invalidation rules
    auto_invalidate_on_error: bool = True
    cache_stampede_protection: bool = True

@dataclass
class CDNConfig:
    """CDN configuration for static assets"""
    enabled: bool = os.getenv("CDN_ENABLED", "false").lower() == "true"
    provider: str = os.getenv("CDN_PROVIDER", "cloudfront")  # cloudfront, cloudflare, fastly
    base_url: str = os.getenv("CDN_BASE_URL", "")
    cache_control_max_age: int = int(os.getenv("CDN_CACHE_MAX_AGE", "86400"))  # 24 hours
    
    # Asset types to serve via CDN
    static_assets: list = None
    document_assets: bool = True
    image_assets: bool = True
    
    def __post_init__(self):
        if self.static_assets is None:
            self.static_assets = ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2']

@dataclass
class PerformanceConfig:
    """Main performance configuration"""
    redis: RedisConfig
    database: DatabaseConfig
    load_balancer: LoadBalancerConfig
    cache: CacheConfig
    cdn: CDNConfig
    
    # General performance settings
    enable_compression: bool = True
    compression_threshold: int = 1024
    enable_etag: bool = True
    enable_cors_caching: bool = True
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "1000"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
    
    # Connection limits
    max_connections_per_ip: int = int(os.getenv("MAX_CONNECTIONS_PER_IP", "100"))
    connection_timeout: int = int(os.getenv("CONNECTION_TIMEOUT", "60"))
    
    # Background tasks
    enable_background_optimization: bool = True
    optimization_interval: int = int(os.getenv("OPTIMIZATION_INTERVAL", "300"))  # 5 minutes

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return PerformanceConfig(
        redis=RedisConfig(),
        database=DatabaseConfig(),
        load_balancer=LoadBalancerConfig(),
        cache=CacheConfig(),
        cdn=CDNConfig()
    )

# Environment-specific configurations
def get_development_config() -> PerformanceConfig:
    """Development environment configuration"""
    config = get_performance_config()
    
    # Disable Redis in development if not available
    if os.getenv("REDIS_AVAILABLE", "true").lower() == "false":
        config.redis.host = "localhost"
        config.cache.agent_response_enabled = False
    
    # Disable CDN in development
    config.cdn.enabled = False
    
    # Reduce cache TTLs for development
    config.cache.agent_response_ttl = 300  # 5 minutes
    config.cache.system_status_ttl = 10   # 10 seconds
    
    return config

def get_production_config() -> PerformanceConfig:
    """Production environment configuration"""
    config = get_performance_config()
    
    # Production optimizations
    config.redis.max_connections = 100
    config.database.pool_size = 50
    config.database.enable_read_replicas = True
    
    # Enable all caching in production
    config.cache.agent_response_enabled = True
    config.cdn.enabled = True
    
    # Longer cache TTLs in production
    config.cache.agent_response_ttl = 3600    # 1 hour
    config.cache.document_analysis_ttl = 14400  # 4 hours
    
    return config

def get_test_config() -> PerformanceConfig:
    """Test environment configuration"""
    config = get_performance_config()
    
    # Use in-memory caching for tests
    config.redis.host = "localhost"
    config.cache.agent_response_enabled = False
    config.cdn.enabled = False
    
    # Very short TTLs for tests
    config.cache.agent_response_ttl = 1
    config.cache.system_status_ttl = 1
    
    return config

# Load configuration based on environment
ENV = os.getenv("UAP_ENV", "development").lower()

if ENV == "production":
    performance_config = get_production_config()
elif ENV == "test":
    performance_config = get_test_config()
else:
    performance_config = get_development_config()

# Configuration validation
def validate_config(config: PerformanceConfig) -> Dict[str, Any]:
    """Validate performance configuration"""
    issues = []
    
    # Redis validation
    if config.cache.agent_response_enabled and not config.redis.host:
        issues.append("Redis host not configured but caching is enabled")
    
    # Database validation
    if config.database.pool_size > config.database.max_overflow:
        issues.append("Database pool_size should not exceed max_overflow")
    
    # CDN validation
    if config.cdn.enabled and not config.cdn.base_url:
        issues.append("CDN enabled but base URL not configured")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "environment": ENV,
        "redis_enabled": config.cache.agent_response_enabled,
        "cdn_enabled": config.cdn.enabled,
        "load_balancer_strategy": config.load_balancer.strategy
    }

# Export the active configuration
__all__ = [
    'PerformanceConfig',
    'RedisConfig', 
    'DatabaseConfig',
    'LoadBalancerConfig',
    'CacheConfig',
    'CDNConfig',
    'performance_config',
    'get_performance_config',
    'validate_config'
]