# File: backend/services/performance_service.py
"""
Performance service for UAP platform.
Integrates caching, load balancing, and database optimization.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

# Import caching components
from ..cache import (
    RedisCache, CacheManager, AgentResponseCache, DocumentCache,
    get_cache, get_cache_manager
)
from ..cache.decorators import cache_async_result, cache_agent_response, cache_document_analysis

# Import optimization components
from ..optimization import (
    DatabaseOptimizer, LoadBalancer, ServerInstance, ServerStatus,
    HealthBasedStrategy, ConnectionPoolConfig
)
from ..optimization.response_optimizer import response_optimizer, ResponseOptimizer
from ..optimization.memory_manager import memory_manager, MemoryManager
from ..optimization.request_batcher import request_batcher, RequestBatcher
from ..websocket.optimized_handler import optimized_websocket_handler, OptimizedWebSocketHandler

# Import configuration
from ..config.performance_config import performance_config, validate_config
from ..config.cdn_config import cdn_manager

logger = logging.getLogger(__name__)

class PerformanceService:
    """Main performance service that coordinates caching, load balancing, and optimization"""
    
    def __init__(self):
        self.config = performance_config
        self.redis_cache: Optional[RedisCache] = None
        self.cache_manager: Optional[CacheManager] = None
        self.database_optimizer: Optional[DatabaseOptimizer] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.cdn_manager = cdn_manager
        
        # New optimization components
        self.response_optimizer = response_optimizer
        self.memory_manager = memory_manager
        self.request_batcher = request_batcher
        self.websocket_handler = optimized_websocket_handler
        
        self.initialized = False
        self.stats = {
            'service_start_time': datetime.utcnow(),
            'cache_hits': 0,
            'cache_misses': 0,
            'database_queries': 0,
            'load_balanced_requests': 0,
            'responses_optimized': 0,
            'memory_cleanups': 0,
            'requests_batched': 0,
            'websocket_connections': 0
        }
    
    async def initialize(self):
        """Initialize all performance components"""
        try:
            logger.info("Initializing performance service...")
            
            # Validate configuration
            config_validation = validate_config(self.config)
            if not config_validation['valid']:
                logger.warning(f"Performance config issues: {config_validation['issues']}")
            
            # Initialize Redis cache if enabled
            if self.config.cache.agent_response_enabled:
                await self._initialize_cache()
            
            # Initialize database optimizer
            await self._initialize_database_optimizer()
            
            # Initialize load balancer if configured
            await self._initialize_load_balancer()
            
            # Initialize new optimization components
            await self._initialize_advanced_optimizations()
            
            self.initialized = True
            logger.info("Performance service initialized successfully with advanced optimizations")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance service: {e}")
            # Continue with limited functionality
    
    async def _initialize_cache(self):
        """Initialize Redis cache and cache manager"""
        try:
            self.redis_cache = await get_cache()
            self.cache_manager = await get_cache_manager()
            
            # Test cache connection
            await self.redis_cache.set("health_check", "ok", 60)
            test_value = await self.redis_cache.get("health_check")
            
            if test_value == "ok":
                logger.info("Redis cache initialized and tested successfully")
            else:
                logger.warning("Redis cache test failed")
                
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            self.redis_cache = None
            self.cache_manager = None
    
    async def _initialize_database_optimizer(self):
        """Initialize database connection pooling and optimization"""
        try:
            self.database_optimizer = DatabaseOptimizer()
            
            # Add primary database connection pool
            if self.config.database.enable_connection_pooling:
                from ..database.connection import DATABASE_URL
                
                # Parse database URL to create config
                pool_config = ConnectionPoolConfig(
                    host="localhost",  # This should be parsed from DATABASE_URL
                    port=5432,
                    database="uap_db",
                    username="uap_user",
                    password="uap_password",
                    min_connections=5,
                    max_connections=self.config.database.pool_size,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300.0
                )
                
                await self.database_optimizer.add_connection_pool(
                    "primary", pool_config, "postgresql"
                )
                
                logger.info("Database optimizer initialized with connection pooling")
            
        except Exception as e:
            logger.error(f"Database optimizer initialization failed: {e}")
            self.database_optimizer = None
    
    async def _initialize_load_balancer(self):
        """Initialize load balancer for distributed requests"""
        try:
            from ..optimization.load_balancer import HealthBasedStrategy
            
            self.load_balancer = LoadBalancer(strategy=HealthBasedStrategy())
            
            # Add local server instance (in production, this would be discovered)
            local_server = ServerInstance(
                id="local",
                host="127.0.0.1",
                port=8000,
                weight=1
            )
            self.load_balancer.add_server(local_server)
            
            await self.load_balancer.start()
            logger.info("Load balancer initialized")
            
        except Exception as e:
            logger.error(f"Load balancer initialization failed: {e}")
            self.load_balancer = None
    
    async def _initialize_advanced_optimizations(self):
        """Initialize advanced optimization components"""
        try:
            # Initialize memory manager
            await self.memory_manager.start_monitoring(interval=60)
            logger.info("Memory manager initialized with monitoring")
            
            # Initialize request batcher
            await self.request_batcher.start()
            logger.info("Request batcher initialized")
            
            # Initialize WebSocket handler
            await self.websocket_handler.start()
            logger.info("Optimized WebSocket handler initialized")
            
            # Response optimizer is already initialized (singleton)
            logger.info("Response optimizer ready")
            
        except Exception as e:
            logger.error(f"Advanced optimizations initialization failed: {e}")
            # Continue with limited functionality
    
    # Advanced optimization methods
    async def optimize_response(self, data: Any, compress: bool = True, 
                              fields: Optional[List[str]] = None,
                              exclude_fields: Optional[List[str]] = None,
                              paginate: bool = False, page: int = 1,
                              page_size: Optional[int] = None) -> Dict[str, Any]:
        """Optimize API response"""
        try:
            result = await self.response_optimizer.optimize_response(
                data=data,
                compress=compress,
                fields=fields,
                exclude_fields=exclude_fields,
                paginate=paginate,
                page=page,
                page_size=page_size
            )
            self.stats['responses_optimized'] += 1
            return result
        except Exception as e:
            logger.error(f"Response optimization failed: {e}")
            return {'data': data, 'optimized': False, 'error': str(e)}
    
    async def batch_request(self, request_id: str, data: Dict[str, Any], 
                          priority: str = 'normal') -> Any:
        """Submit request for batching"""
        try:
            from ..optimization.request_batcher import RequestPriority
            priority_map = {
                'low': RequestPriority.LOW,
                'normal': RequestPriority.NORMAL,
                'high': RequestPriority.HIGH,
                'critical': RequestPriority.CRITICAL
            }
            req_priority = priority_map.get(priority, RequestPriority.NORMAL)
            
            result = await self.request_batcher.submit_request(request_id, data, req_priority)
            self.stats['requests_batched'] += 1
            return result
        except Exception as e:
            logger.error(f"Request batching failed: {e}")
            raise
    
    async def force_memory_cleanup(self) -> Dict[str, Any]:
        """Force memory cleanup"""
        try:
            result = await self.memory_manager.force_cleanup()
            self.stats['memory_cleanups'] += 1
            return result
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_websocket_connection(self, websocket, user_id: Optional[str] = None) -> str:
        """Handle optimized WebSocket connection"""
        try:
            connection_id = await self.websocket_handler.connect(websocket, user_id)
            self.stats['websocket_connections'] += 1
            return connection_id
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def send_websocket_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send optimized WebSocket message"""
        try:
            return await self.websocket_handler.send_message(connection_id, message)
        except Exception as e:
            logger.error(f"WebSocket message send failed: {e}")
            return False
    
    # Caching methods
    async def cache_agent_response(self, agent_id: str, framework: str, 
                                 message: str, response: Dict[str, Any]) -> bool:
        """Cache agent response"""
        if not self.cache_manager:
            return False
        
        try:
            await self.cache_manager.cache_agent_response(
                agent_id, message, response, 
                ttl=self.config.cache.agent_response_ttl
            )
            self.stats['cache_hits'] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to cache agent response: {e}")
            return False
    
    async def get_cached_agent_response(self, agent_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Get cached agent response"""
        if not self.cache_manager:
            return None
        
        try:
            result = await self.cache_manager.get_cached_agent_response(agent_id, message)
            if result:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
            return result
        except Exception as e:
            logger.error(f"Failed to get cached agent response: {e}")
            self.stats['cache_misses'] += 1
            return None
    
    async def cache_document_analysis(self, doc_id: str, analysis_type: str, 
                                    result: Dict[str, Any]) -> bool:
        """Cache document analysis result"""
        if not self.cache_manager:
            return False
        
        try:
            await self.cache_manager.cache_document_analysis(
                doc_id, analysis_type, result,
                ttl=self.config.cache.document_analysis_ttl
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache document analysis: {e}")
            return False
    
    async def get_cached_document_analysis(self, doc_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached document analysis"""
        if not self.cache_manager:
            return None
        
        try:
            return await self.cache_manager.get_cached_document_analysis(doc_id, analysis_type)
        except Exception as e:
            logger.error(f"Failed to get cached document analysis: {e}")
            return None
    
    # Database optimization methods
    async def execute_optimized_query(self, query: str, params: tuple = None, 
                                    cache_result: bool = False) -> Any:
        """Execute database query with optimization"""
        if not self.database_optimizer:
            # Fallback to regular database connection
            from ..database.connection import database
            return await database.fetch_all(query, params)
        
        try:
            result = await self.database_optimizer.execute_optimized(
                query, params, cache_result=cache_result
            )
            self.stats['database_queries'] += 1
            return result
        except Exception as e:
            logger.error(f"Optimized query execution failed: {e}")
            raise
    
    # Load balancing methods
    async def route_request(self, request_context: Dict[str, Any] = None) -> Optional[str]:
        """Route request using load balancer"""
        if not self.load_balancer:
            return None
        
        try:
            server = await self.load_balancer.route_request(request_context)
            if server:
                self.stats['load_balanced_requests'] += 1
                return server.url
            return None
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return None
    
    # CDN methods
    def get_cdn_url(self, asset_path: str, asset_type: str = "static") -> str:
        """Get CDN URL for an asset"""
        try:
            return self.cdn_manager.get_cdn_url(asset_path, asset_type)
        except Exception as e:
            logger.error(f"CDN URL generation failed: {e}")
            return asset_path
    
    def get_optimized_headers(self, file_path: str) -> Dict[str, str]:
        """Get optimized headers for file serving"""
        try:
            return self.cdn_manager.get_optimized_headers(file_path)
        except Exception as e:
            logger.error(f"Header optimization failed: {e}")
            return {}
    
    # Performance monitoring
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'service': {
                'initialized': self.initialized,
                'uptime_seconds': (datetime.utcnow() - self.stats['service_start_time']).total_seconds(),
                **self.stats
            },
            'cache': {
                'enabled': self.redis_cache is not None,
                'stats': self.redis_cache.get_stats() if self.redis_cache else {}
            },
            'database': {
                'optimizer_enabled': self.database_optimizer is not None,
                'stats': self.database_optimizer.get_database_stats() if self.database_optimizer else {}
            },
            'load_balancer': {
                'enabled': self.load_balancer is not None,
                'stats': self.load_balancer.get_stats() if self.load_balancer else {}
            },
            'cdn': {
                'enabled': self.cdn_manager is not None,
                'stats': self.cdn_manager.get_stats() if self.cdn_manager else {}
            },
            'response_optimization': {
                'enabled': True,
                'stats': self.response_optimizer.get_comprehensive_stats()
            },
            'memory_management': {
                'enabled': True,
                'stats': self.memory_manager.get_comprehensive_stats()
            },
            'request_batching': {
                'enabled': True,
                'stats': self.request_batcher.get_comprehensive_stats()
            },
            'websocket_optimization': {
                'enabled': True,
                'stats': self.websocket_handler.get_handler_stats()
            },
            'configuration': {
                'cache_ttl': {
                    'agent_response': self.config.cache.agent_response_ttl,
                    'document_analysis': self.config.cache.document_analysis_ttl,
                    'system_status': self.config.cache.system_status_ttl
                },
                'database_pool_size': self.config.database.pool_size,
                'redis_url': self.config.redis.url,
                'environment': validate_config(self.config)['environment']
            }
        }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'healthy': True,
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check cache health
        if self.redis_cache:
            cache_health = await self.redis_cache.health_check()
            health['components']['cache'] = cache_health
            if not cache_health.get('healthy', False):
                health['healthy'] = False
        else:
            health['components']['cache'] = {'healthy': False, 'error': 'Cache not initialized'}
        
        # Check database health
        if self.database_optimizer:
            db_stats = self.database_optimizer.get_database_stats()
            health['components']['database'] = {
                'healthy': len(db_stats['connection_pools']) > 0,
                'pools': len(db_stats['connection_pools'])
            }
        else:
            health['components']['database'] = {'healthy': False, 'error': 'Database optimizer not initialized'}
        
        # Check load balancer health
        if self.load_balancer:
            lb_stats = self.load_balancer.get_stats()
            health['components']['load_balancer'] = {
                'healthy': lb_stats['healthy_servers'] > 0,
                'healthy_servers': lb_stats['healthy_servers'],
                'total_servers': lb_stats['total_servers']
            }
        else:
            health['components']['load_balancer'] = {'healthy': True, 'note': 'Load balancer optional'}
        
        return health
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization routines"""
        optimizations = {
            'timestamp': datetime.utcnow().isoformat(),
            'optimizations_applied': []
        }
        
        # Cache optimization
        if self.redis_cache:
            try:
                # Clear expired entries manually if needed
                info = await self.redis_cache.get_info()
                if info.get('used_memory', 0) > 100 * 1024 * 1024:  # 100MB threshold
                    # Could implement cache cleanup here
                    optimizations['optimizations_applied'].append('cache_memory_check')
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
        
        # Database optimization
        if self.database_optimizer:
            try:
                # Get slow queries and log them
                stats = self.database_optimizer.get_database_stats()
                slow_queries = stats.get('slow_queries', [])
                if slow_queries:
                    logger.warning(f"Found {len(slow_queries)} slow queries")
                    optimizations['optimizations_applied'].append('slow_query_analysis')
            except Exception as e:
                logger.error(f"Database optimization error: {e}")
        
        return optimizations
    
    async def cleanup(self):
        """Cleanup performance service resources"""
        try:
            # Cleanup advanced optimization components
            if self.websocket_handler:
                await self.websocket_handler.stop()
                logger.info("WebSocket handler stopped")
            
            if self.request_batcher:
                await self.request_batcher.stop()
                logger.info("Request batcher stopped")
            
            if self.memory_manager:
                await self.memory_manager.stop_monitoring()
                logger.info("Memory manager stopped")
            
            # Cleanup original components
            if self.load_balancer:
                await self.load_balancer.stop()
            
            if self.database_optimizer:
                await self.database_optimizer.close_all_pools()
            
            if self.redis_cache:
                await self.redis_cache.cleanup()
            
            logger.info("Performance service cleanup completed")
        except Exception as e:
            logger.error(f"Performance service cleanup error: {e}")

# Global performance service instance
performance_service = PerformanceService()

# Convenience decorators that use the performance service
def cache_with_performance_service(ttl: int = None):
    """Decorator that uses the performance service for caching"""
    def decorator(func):
        return cache_async_result(
            ttl=ttl or performance_config.cache.agent_response_ttl
        )(func)
    return decorator

# Context manager for performance monitoring
@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for tracking operation performance"""
    start_time = datetime.utcnow()
    try:
        yield
    finally:
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.debug(f"Operation '{operation_name}' took {duration:.3f} seconds")

# Export main components
__all__ = [
    'PerformanceService',
    'performance_service',
    'cache_with_performance_service',
    'performance_context'
]