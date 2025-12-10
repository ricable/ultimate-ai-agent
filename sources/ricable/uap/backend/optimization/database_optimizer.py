# File: backend/optimization/database_optimizer.py
"""
Database optimization and connection pooling for UAP platform.
Provides query optimization, connection management, and performance monitoring.
"""

import asyncio
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncpg
import aiopg
import aiomysql
from collections import defaultdict, deque
import statistics
import json

logger = logging.getLogger(__name__)

@dataclass
class QueryStats:
    """Statistics for a database query"""
    query_hash: str
    query_template: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: Optional[datetime] = None
    error_count: int = 0
    cache_hits: int = 0
    
    def record_execution(self, execution_time: float, success: bool = True):
        """Record query execution"""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.utcnow()
        
        if not success:
            self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_hash': self.query_hash,
            'query_template': self.query_template,
            'execution_count': self.execution_count,
            'avg_time_ms': round(self.avg_time * 1000, 2),
            'min_time_ms': round(self.min_time * 1000, 2),
            'max_time_ms': round(self.max_time * 1000, 2),
            'error_count': self.error_count,
            'cache_hits': self.cache_hits,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None
        }

@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pool"""
    host: str
    port: int
    database: str
    username: str
    password: str
    min_connections: int = 5
    max_connections: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    command_timeout: float = 30.0
    ssl_enabled: bool = False
    pool_recycle: int = 3600  # 1 hour

class DatabaseType(ABC):
    """Abstract base class for database types"""
    
    @abstractmethod
    async def create_pool(self, config: ConnectionPoolConfig):
        pass
    
    @abstractmethod
    async def execute_query(self, connection, query: str, params: Tuple = None):
        pass
    
    @abstractmethod
    async def fetch_one(self, connection, query: str, params: Tuple = None):
        pass
    
    @abstractmethod
    async def fetch_all(self, connection, query: str, params: Tuple = None):
        pass

class PostgreSQLAdapter(DatabaseType):
    """PostgreSQL database adapter"""
    
    async def create_pool(self, config: ConnectionPoolConfig):
        """Create PostgreSQL connection pool"""
        return await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password,
            min_size=config.min_connections,
            max_size=config.max_connections,
            max_queries=config.max_queries,
            max_inactive_connection_lifetime=config.max_inactive_connection_lifetime,
            command_timeout=config.command_timeout
        )
    
    async def execute_query(self, connection, query: str, params: Tuple = None):
        """Execute query"""
        if params:
            return await connection.execute(query, *params)
        return await connection.execute(query)
    
    async def fetch_one(self, connection, query: str, params: Tuple = None):
        """Fetch one row"""
        if params:
            return await connection.fetchrow(query, *params)
        return await connection.fetchrow(query)
    
    async def fetch_all(self, connection, query: str, params: Tuple = None):
        """Fetch all rows"""
        if params:
            return await connection.fetch(query, *params)
        return await connection.fetch(query)

class MySQLAdapter(DatabaseType):
    """MySQL database adapter"""
    
    async def create_pool(self, config: ConnectionPoolConfig):
        """Create MySQL connection pool"""
        return await aiomysql.create_pool(
            host=config.host,
            port=config.port,
            db=config.database,
            user=config.username,
            password=config.password,
            minsize=config.min_connections,
            maxsize=config.max_connections,
            autocommit=True
        )
    
    async def execute_query(self, connection, query: str, params: Tuple = None):
        """Execute query"""
        async with connection.cursor() as cursor:
            await cursor.execute(query, params)
            return cursor.rowcount
    
    async def fetch_one(self, connection, query: str, params: Tuple = None):
        """Fetch one row"""
        async with connection.cursor() as cursor:
            await cursor.execute(query, params)
            return await cursor.fetchone()
    
    async def fetch_all(self, connection, query: str, params: Tuple = None):
        """Fetch all rows"""
        async with connection.cursor() as cursor:
            await cursor.execute(query, params)
            return await cursor.fetchall()

class ConnectionPool:
    """Database connection pool with monitoring and optimization"""
    
    def __init__(self, config: ConnectionPoolConfig, db_adapter: DatabaseType):
        self.config = config
        self.db_adapter = db_adapter
        self.pool = None
        self.connected = False
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'queries_executed': 0,
            'total_query_time': 0.0,
            'connection_errors': 0,
            'query_errors': 0
        }
        self.connection_history = deque(maxlen=1000)
        
    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await self.db_adapter.create_pool(self.config)
            self.connected = True
            logger.info(f"Database pool initialized: {self.config.database}@{self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            self.stats['connection_errors'] += 1
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.connected:
            raise RuntimeError("Database pool not initialized")
        
        connection = None
        start_time = time.time()
        
        try:
            connection = await self.pool.acquire()
            self.stats['active_connections'] += 1
            
            # Record connection acquisition
            self.connection_history.append({
                'timestamp': datetime.utcnow(),
                'action': 'acquire',
                'duration': time.time() - start_time
            })
            
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self.stats['connection_errors'] += 1
            raise
        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                    self.stats['active_connections'] -= 1
                    
                    # Record connection release
                    self.connection_history.append({
                        'timestamp': datetime.utcnow(),
                        'action': 'release',
                        'duration': time.time() - start_time
                    })
                except Exception as e:
                    logger.error(f"Error releasing connection: {e}")
    
    async def execute(self, query: str, params: Tuple = None) -> Any:
        """Execute query"""
        start_time = time.time()
        
        try:
            async with self.get_connection() as conn:
                result = await self.db_adapter.execute_query(conn, query, params)
                
                # Record successful execution
                execution_time = time.time() - start_time
                self.stats['queries_executed'] += 1
                self.stats['total_query_time'] += execution_time
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats['query_errors'] += 1
            logger.error(f"Query execution error: {e}")
            raise
    
    async def fetch_one(self, query: str, params: Tuple = None) -> Optional[Any]:
        """Fetch one row"""
        start_time = time.time()
        
        try:
            async with self.get_connection() as conn:
                result = await self.db_adapter.fetch_one(conn, query, params)
                
                execution_time = time.time() - start_time
                self.stats['queries_executed'] += 1
                self.stats['total_query_time'] += execution_time
                
                return result
                
        except Exception as e:
            self.stats['query_errors'] += 1
            logger.error(f"Fetch one error: {e}")
            raise
    
    async def fetch_all(self, query: str, params: Tuple = None) -> List[Any]:
        """Fetch all rows"""
        start_time = time.time()
        
        try:
            async with self.get_connection() as conn:
                result = await self.db_adapter.fetch_all(conn, query, params)
                
                execution_time = time.time() - start_time
                self.stats['queries_executed'] += 1
                self.stats['total_query_time'] += execution_time
                
                return result
                
        except Exception as e:
            self.stats['query_errors'] += 1
            logger.error(f"Fetch all error: {e}")
            raise
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.connected = False
            logger.info("Database pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        avg_query_time = (self.stats['total_query_time'] / self.stats['queries_executed'] 
                         if self.stats['queries_executed'] > 0 else 0)
        
        pool_stats = {}
        if hasattr(self.pool, 'get_size'):
            pool_stats.update({
                'pool_size': self.pool.get_size(),
                'pool_free_size': self.pool.get_free_size(),
                'pool_used_size': self.pool.get_size() - self.pool.get_free_size()
            })
        
        return {
            'connected': self.connected,
            'config': {
                'host': self.config.host,
                'database': self.config.database,
                'min_connections': self.config.min_connections,
                'max_connections': self.config.max_connections
            },
            'stats': {
                **self.stats,
                'avg_query_time_ms': round(avg_query_time * 1000, 2),
                **pool_stats
            }
        }

class QueryOptimizer:
    """Query optimization and caching system"""
    
    def __init__(self, enable_caching: bool = True, cache_size: int = 1000):
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.query_stats: Dict[str, QueryStats] = {}
        self.query_cache: Dict[str, Any] = {}
        self.cache_order = deque()
        self.optimization_rules = []
    
    def add_optimization_rule(self, rule_func):
        """Add query optimization rule"""
        self.optimization_rules.append(rule_func)
    
    def optimize_query(self, query: str, params: Tuple = None) -> Tuple[str, Tuple]:
        """Apply optimization rules to query"""
        optimized_query = query
        optimized_params = params
        
        for rule in self.optimization_rules:
            try:
                optimized_query, optimized_params = rule(optimized_query, optimized_params)
            except Exception as e:
                logger.warning(f"Query optimization rule failed: {e}")
        
        return optimized_query, optimized_params
    
    def get_query_hash(self, query: str, params: Tuple = None) -> str:
        """Generate hash for query and parameters"""
        query_normalized = " ".join(query.split())  # Normalize whitespace
        cache_key = f"{query_normalized}:{params}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def get_cached_result(self, query: str, params: Tuple = None) -> Optional[Any]:
        """Get cached query result"""
        if not self.enable_caching:
            return None
        
        query_hash = self.get_query_hash(query, params)
        
        if query_hash in self.query_cache:
            # Move to end (LRU)
            self.cache_order.remove(query_hash)
            self.cache_order.append(query_hash)
            
            # Update stats
            if query_hash in self.query_stats:
                self.query_stats[query_hash].cache_hits += 1
            
            return self.query_cache[query_hash]
        
        return None
    
    def cache_result(self, query: str, params: Tuple, result: Any):
        """Cache query result"""
        if not self.enable_caching:
            return
        
        query_hash = self.get_query_hash(query, params)
        
        # Evict oldest if cache is full
        while len(self.query_cache) >= self.cache_size and self.cache_order:
            oldest_hash = self.cache_order.popleft()
            self.query_cache.pop(oldest_hash, None)
        
        self.query_cache[query_hash] = result
        self.cache_order.append(query_hash)
    
    def record_query_execution(self, query: str, params: Tuple, execution_time: float, success: bool = True):
        """Record query execution statistics"""
        query_hash = self.get_query_hash(query, params)
        
        if query_hash not in self.query_stats:
            # Create template by removing parameter values
            query_template = self._create_query_template(query)
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_template=query_template
            )
        
        self.query_stats[query_hash].record_execution(execution_time, success)
    
    def _create_query_template(self, query: str) -> str:
        """Create query template by replacing values with placeholders"""
        # Simple template creation - replace quoted strings and numbers
        import re
        
        # Replace string literals
        template = re.sub(r"'[^']*'", "?", query)
        template = re.sub(r'"[^"]*"', "?", template)
        
        # Replace numbers
        template = re.sub(r'\b\d+\b', "?", template)
        
        return template
    
    def get_slow_queries(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Get queries slower than threshold"""
        slow_queries = []
        threshold_seconds = threshold_ms / 1000
        
        for stats in self.query_stats.values():
            if stats.avg_time > threshold_seconds:
                slow_queries.append(stats.to_dict())
        
        return sorted(slow_queries, key=lambda x: x['avg_time_ms'], reverse=True)
    
    def get_most_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently executed queries"""
        frequent_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x.execution_count,
            reverse=True
        )[:limit]
        
        return [q.to_dict() for q in frequent_queries]
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get query optimizer statistics"""
        total_queries = sum(s.execution_count for s in self.query_stats.values())
        total_cache_hits = sum(s.cache_hits for s in self.query_stats.values())
        cache_hit_rate = (total_cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            'total_unique_queries': len(self.query_stats),
            'total_executions': total_queries,
            'cache_enabled': self.enable_caching,
            'cache_size': len(self.query_cache),
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'optimization_rules': len(self.optimization_rules)
        }

class DatabaseOptimizer:
    """Main database optimizer with multiple connection pools"""
    
    def __init__(self):
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.query_optimizer = QueryOptimizer()
        self.read_replicas = []
        self.write_master = None
        
        # Add default optimization rules
        self._setup_default_optimization_rules()
    
    def _setup_default_optimization_rules(self):
        """Setup default query optimization rules"""
        
        def limit_protection_rule(query: str, params: Tuple) -> Tuple[str, Tuple]:
            """Add LIMIT to SELECT queries without one"""
            query_upper = query.upper().strip()
            if (query_upper.startswith('SELECT') and 
                'LIMIT' not in query_upper and 
                'COUNT(' not in query_upper):
                return f"{query} LIMIT 1000", params
            return query, params
        
        def index_hint_rule(query: str, params: Tuple) -> Tuple[str, Tuple]:
            """Add index hints for common patterns"""
            # Example: Add index hints for user_id lookups
            if 'WHERE user_id =' in query and 'USE INDEX' not in query:
                query = query.replace('WHERE user_id =', 'USE INDEX (idx_user_id) WHERE user_id =')
            return query, params
        
        self.query_optimizer.add_optimization_rule(limit_protection_rule)
        self.query_optimizer.add_optimization_rule(index_hint_rule)
    
    async def add_connection_pool(self, name: str, config: ConnectionPoolConfig, 
                                db_type: str = "postgresql"):
        """Add a connection pool"""
        if db_type.lower() == "postgresql":
            adapter = PostgreSQLAdapter()
        elif db_type.lower() == "mysql":
            adapter = MySQLAdapter()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        pool = ConnectionPool(config, adapter)
        await pool.initialize()
        
        self.connection_pools[name] = pool
        
        # Set as master/replica
        if name == "master" or not self.write_master:
            self.write_master = name
        else:
            self.read_replicas.append(name)
        
        logger.info(f"Added connection pool '{name}' ({db_type})")
    
    def get_pool(self, pool_name: str = None, read_only: bool = False) -> ConnectionPool:
        """Get connection pool for query execution"""
        if pool_name:
            return self.connection_pools.get(pool_name)
        
        # Automatic pool selection
        if read_only and self.read_replicas:
            # Load balance across read replicas
            import random
            replica_name = random.choice(self.read_replicas)
            return self.connection_pools[replica_name]
        
        # Use master for writes or when no replicas available
        if self.write_master:
            return self.connection_pools[self.write_master]
        
        # Fallback to any available pool
        if self.connection_pools:
            return next(iter(self.connection_pools.values()))
        
        raise RuntimeError("No database pools available")
    
    async def execute_optimized(self, query: str, params: Tuple = None, 
                              pool_name: str = None, cache_result: bool = False) -> Any:
        """Execute query with optimization"""
        # Check cache first
        if cache_result:
            cached_result = self.query_optimizer.get_cached_result(query, params)
            if cached_result is not None:
                return cached_result
        
        # Optimize query
        optimized_query, optimized_params = self.query_optimizer.optimize_query(query, params)
        
        # Execute query
        start_time = time.time()
        pool = self.get_pool(pool_name, read_only=self._is_read_only_query(query))
        
        try:
            result = await pool.execute(optimized_query, optimized_params)
            execution_time = time.time() - start_time
            
            # Record statistics
            self.query_optimizer.record_query_execution(
                query, params, execution_time, True
            )
            
            # Cache result if requested
            if cache_result:
                self.query_optimizer.cache_result(query, params, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.query_optimizer.record_query_execution(
                query, params, execution_time, False
            )
            raise
    
    async def fetch_one_optimized(self, query: str, params: Tuple = None, 
                                pool_name: str = None, cache_result: bool = True) -> Optional[Any]:
        """Fetch one row with optimization"""
        # Check cache first
        if cache_result:
            cached_result = self.query_optimizer.get_cached_result(query, params)
            if cached_result is not None:
                return cached_result
        
        optimized_query, optimized_params = self.query_optimizer.optimize_query(query, params)
        
        start_time = time.time()
        pool = self.get_pool(pool_name, read_only=True)  # Assume fetch is read-only
        
        try:
            result = await pool.fetch_one(optimized_query, optimized_params)
            execution_time = time.time() - start_time
            
            self.query_optimizer.record_query_execution(
                query, params, execution_time, True
            )
            
            if cache_result:
                self.query_optimizer.cache_result(query, params, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.query_optimizer.record_query_execution(
                query, params, execution_time, False
            )
            raise
    
    async def fetch_all_optimized(self, query: str, params: Tuple = None, 
                                pool_name: str = None, cache_result: bool = True) -> List[Any]:
        """Fetch all rows with optimization"""
        if cache_result:
            cached_result = self.query_optimizer.get_cached_result(query, params)
            if cached_result is not None:
                return cached_result
        
        optimized_query, optimized_params = self.query_optimizer.optimize_query(query, params)
        
        start_time = time.time()
        pool = self.get_pool(pool_name, read_only=True)
        
        try:
            result = await pool.fetch_all(optimized_query, optimized_params)
            execution_time = time.time() - start_time
            
            self.query_optimizer.record_query_execution(
                query, params, execution_time, True
            )
            
            if cache_result:
                self.query_optimizer.cache_result(query, params, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.query_optimizer.record_query_execution(
                query, params, execution_time, False
            )
            raise
    
    def _is_read_only_query(self, query: str) -> bool:
        """Check if query is read-only"""
        query_upper = query.strip().upper()
        return query_upper.startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN'))
    
    async def close_all_pools(self):
        """Close all connection pools"""
        for pool in self.connection_pools.values():
            await pool.close()
        self.connection_pools.clear()
        logger.info("All database pools closed")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        pool_stats = {}
        for name, pool in self.connection_pools.items():
            pool_stats[name] = pool.get_stats()
        
        optimizer_stats = self.query_optimizer.get_optimizer_stats()
        slow_queries = self.query_optimizer.get_slow_queries(1000)  # 1 second threshold
        frequent_queries = self.query_optimizer.get_most_frequent_queries(5)
        
        return {
            'connection_pools': pool_stats,
            'query_optimization': optimizer_stats,
            'slow_queries': slow_queries[:5],  # Top 5 slowest
            'frequent_queries': frequent_queries,
            'configuration': {
                'write_master': self.write_master,
                'read_replicas': self.read_replicas,
                'total_pools': len(self.connection_pools)
            }
        }

# Global database optimizer instance
database_optimizer = DatabaseOptimizer()

# Example usage
async def example_usage():
    """Example usage of database optimizer"""
    # Configure connection pools
    master_config = ConnectionPoolConfig(
        host="localhost",
        port=5432,
        database="uap_db",
        username="uap_user",
        password="password",
        min_connections=5,
        max_connections=20
    )
    
    replica_config = ConnectionPoolConfig(
        host="localhost",
        port=5433,
        database="uap_db",
        username="uap_reader",
        password="password",
        min_connections=3,
        max_connections=15
    )
    
    # Add pools
    await database_optimizer.add_connection_pool("master", master_config)
    await database_optimizer.add_connection_pool("replica1", replica_config)
    
    # Execute optimized queries
    users = await database_optimizer.fetch_all_optimized(
        "SELECT * FROM users WHERE active = $1",
        (True,),
        cache_result=True
    )
    
    user = await database_optimizer.fetch_one_optimized(
        "SELECT * FROM users WHERE id = $1",
        (123,),
        cache_result=True
    )
    
    # Get statistics
    stats = database_optimizer.get_database_stats()
    print(f"Database stats: {json.dumps(stats, indent=2)}")
    
    # Close pools
    await database_optimizer.close_all_pools()

if __name__ == "__main__":
    asyncio.run(example_usage())