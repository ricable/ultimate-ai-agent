# File: backend/websocket/optimized_handler.py
"""
Optimized WebSocket handler for UAP platform.
Provides connection pooling, message batching, compression, and performance monitoring.
"""

import asyncio
import json
import time
import gzip
import logging
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import uuid
import weakref
from fastapi import WebSocket, WebSocketDisconnect
import statistics

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket optimization"""
    # Connection pooling
    max_connections_per_user: int = 5
    max_total_connections: int = 10000
    connection_timeout: int = 300  # 5 minutes
    
    # Message batching
    enable_message_batching: bool = True
    batch_size: int = 10
    batch_timeout: float = 0.05  # 50ms
    
    # Compression
    enable_compression: bool = True
    compression_threshold: int = 512  # bytes
    
    # Rate limiting
    max_messages_per_second: int = 100
    max_message_size: int = 1024 * 1024  # 1MB
    
    # Health monitoring
    ping_interval: int = 30  # seconds
    ping_timeout: int = 10   # seconds
    
    # Buffer management
    max_buffer_size: int = 1000
    buffer_flush_interval: float = 0.1  # 100ms

@dataclass
class ConnectionMetrics:
    """Metrics for a WebSocket connection"""
    connection_id: str
    user_id: Optional[str]
    connected_at: datetime
    last_activity: datetime = field(default_factory=datetime.utcnow)
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    ping_latency: float = 0.0
    state: ConnectionState = ConnectionState.CONNECTING
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def record_message_sent(self, size: int):
        """Record sent message"""
        self.messages_sent += 1
        self.bytes_sent += size
        self.update_activity()
    
    def record_message_received(self, size: int):
        """Record received message"""
        self.messages_received += 1
        self.bytes_received += size
        self.update_activity()
    
    def record_error(self):
        """Record connection error"""
        self.errors += 1
        self.update_activity()
    
    @property
    def connection_duration(self) -> float:
        """Get connection duration in seconds"""
        return (datetime.utcnow() - self.connected_at).total_seconds()
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return (datetime.utcnow() - self.last_activity).total_seconds()

class MessageBatch:
    """Batch of WebSocket messages"""
    
    def __init__(self, batch_id: str, max_size: int, timeout: float):
        self.batch_id = batch_id
        self.max_size = max_size
        self.timeout = timeout
        self.messages: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.connections: Set[str] = set()
    
    def add_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """Add message to batch, return True if batch is full"""
        self.messages.append(message)
        self.connections.add(connection_id)
        return len(self.messages) >= self.max_size
    
    def is_expired(self) -> bool:
        """Check if batch has expired"""
        return time.time() - self.created_at >= self.timeout
    
    def should_flush(self) -> bool:
        """Check if batch should be flushed"""
        return len(self.messages) >= self.max_size or self.is_expired()

class ConnectionPool:
    """WebSocket connection pool manager"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.connections: Dict[str, WebSocket] = {}
        self.metrics: Dict[str, ConnectionMetrics] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.rate_limiters: Dict[str, deque] = defaultdict(deque)
        self.connection_count = 0
        
        # Message batching
        self.message_batches: Dict[str, MessageBatch] = {}
        self.batch_processors: Dict[str, Callable] = {}
        
        # Performance monitoring
        self.pool_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages': 0,
            'total_bytes': 0,
            'compression_ratio': 0.0,
            'average_latency': 0.0,
            'error_rate': 0.0
        }
    
    async def add_connection(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """Add connection to pool"""
        # Check connection limits
        if self.connection_count >= self.config.max_total_connections:
            raise Exception("Maximum connections exceeded")
        
        if user_id:
            user_conn_count = len(self.user_connections[user_id])
            if user_conn_count >= self.config.max_connections_per_user:
                raise Exception(f"Maximum connections per user exceeded: {user_conn_count}")
        
        # Generate connection ID
        connection_id = str(uuid.uuid4())
        
        # Add to pool
        self.connections[connection_id] = websocket
        self.metrics[connection_id] = ConnectionMetrics(
            connection_id=connection_id,
            user_id=user_id,
            connected_at=datetime.utcnow()
        )
        
        if user_id:
            self.user_connections[user_id].add(connection_id)
        
        self.connection_count += 1
        self.pool_stats['total_connections'] += 1
        self.pool_stats['active_connections'] = self.connection_count
        
        logger.info(f"WebSocket connection added: {connection_id} (user: {user_id})")
        return connection_id
    
    async def remove_connection(self, connection_id: str):
        """Remove connection from pool"""
        if connection_id not in self.connections:
            return
        
        metrics = self.metrics.get(connection_id)
        if metrics:
            metrics.state = ConnectionState.DISCONNECTED
            
            # Remove from user connections
            if metrics.user_id:
                self.user_connections[metrics.user_id].discard(connection_id)
                if not self.user_connections[metrics.user_id]:
                    del self.user_connections[metrics.user_id]
        
        # Remove from pool
        del self.connections[connection_id]
        if connection_id in self.rate_limiters:
            del self.rate_limiters[connection_id]
        
        self.connection_count -= 1
        self.pool_stats['active_connections'] = self.connection_count
        
        logger.info(f"WebSocket connection removed: {connection_id}")
    
    def get_connection(self, connection_id: str) -> Optional[WebSocket]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user"""
        return list(self.user_connections.get(user_id, set()))
    
    async def check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits"""
        now = time.time()
        rate_limiter = self.rate_limiters[connection_id]
        
        # Remove old timestamps
        while rate_limiter and rate_limiter[0] < now - 1.0:
            rate_limiter.popleft()
        
        # Check rate limit
        if len(rate_limiter) >= self.config.max_messages_per_second:
            return False
        
        # Add current timestamp
        rate_limiter.append(now)
        return True
    
    def update_metrics(self, connection_id: str, message_size: int, sent: bool = True):
        """Update connection metrics"""
        if connection_id in self.metrics:
            metrics = self.metrics[connection_id]
            if sent:
                metrics.record_message_sent(message_size)
            else:
                metrics.record_message_received(message_size)
            
            self.pool_stats['total_messages'] += 1
            self.pool_stats['total_bytes'] += message_size
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        # Calculate aggregated metrics
        if self.metrics:
            avg_latency = statistics.mean([m.ping_latency for m in self.metrics.values() if m.ping_latency > 0])
            error_rate = sum(m.errors for m in self.metrics.values()) / len(self.metrics) * 100
        else:
            avg_latency = 0.0
            error_rate = 0.0
        
        connection_states = defaultdict(int)
        for metrics in self.metrics.values():
            connection_states[metrics.state.value] += 1
        
        return {
            **self.pool_stats,
            'average_latency_ms': round(avg_latency * 1000, 2),
            'error_rate_percent': round(error_rate, 2),
            'connection_states': dict(connection_states),
            'user_distribution': {user: len(conns) for user, conns in self.user_connections.items()},
            'config': {
                'max_connections': self.config.max_total_connections,
                'max_per_user': self.config.max_connections_per_user,
                'batch_size': self.config.batch_size,
                'compression_enabled': self.config.enable_compression
            }
        }

class MessageCompressor:
    """WebSocket message compression"""
    
    def __init__(self, threshold: int = 512):
        self.threshold = threshold
        self.compression_stats = {
            'total_messages': 0,
            'compressed_messages': 0,
            'bytes_saved': 0
        }
    
    def compress_message(self, message: Union[str, bytes]) -> Dict[str, Any]:
        """Compress message if beneficial"""
        self.compression_stats['total_messages'] += 1
        
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        else:
            message_bytes = message
        
        original_size = len(message_bytes)
        
        # Only compress if above threshold
        if original_size < self.threshold:
            return {
                'data': message,
                'compressed': False,
                'original_size': original_size,
                'compressed_size': original_size
            }
        
        try:
            compressed_data = gzip.compress(message_bytes)
            compressed_size = len(compressed_data)
            
            # Only use compression if it saves space
            if compressed_size < original_size * 0.9:  # At least 10% savings
                self.compression_stats['compressed_messages'] += 1
                self.compression_stats['bytes_saved'] += (original_size - compressed_size)
                
                return {
                    'data': compressed_data,
                    'compressed': True,
                    'original_size': original_size,
                    'compressed_size': compressed_size
                }
            else:
                return {
                    'data': message,
                    'compressed': False,
                    'original_size': original_size,
                    'compressed_size': original_size
                }
                
        except Exception as e:
            logger.error(f"Message compression failed: {e}")
            return {
                'data': message,
                'compressed': False,
                'original_size': original_size,
                'compressed_size': original_size,
                'error': str(e)
            }
    
    def decompress_message(self, data: bytes) -> str:
        """Decompress message"""
        try:
            return gzip.decompress(data).decode('utf-8')
        except Exception as e:
            logger.error(f"Message decompression failed: {e}")
            return data.decode('utf-8') if isinstance(data, bytes) else str(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        total = self.compression_stats['total_messages']
        compressed = self.compression_stats['compressed_messages']
        compression_rate = (compressed / total * 100) if total > 0 else 0
        
        return {
            'total_messages': total,
            'compressed_messages': compressed,
            'compression_rate_percent': round(compression_rate, 2),
            'bytes_saved': self.compression_stats['bytes_saved'],
            'threshold': self.threshold
        }

class OptimizedWebSocketHandler:
    """Main optimized WebSocket handler"""
    
    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig()
        self.connection_pool = ConnectionPool(self.config)
        self.compressor = MessageCompressor(self.config.compression_threshold)
        
        # Background tasks
        self.health_check_task = None
        self.batch_processor_task = None
        self.cleanup_task = None
        
        # Handler registry
        self.message_handlers: Dict[str, Callable] = {}
        
        self.handler_stats = {
            'started_at': datetime.utcnow(),
            'total_connections': 0,
            'total_messages_processed': 0,
            'total_errors': 0,
            'batches_processed': 0
        }
    
    async def start(self):
        """Start the WebSocket handler"""
        logger.info("Starting optimized WebSocket handler")
        
        # Start background tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Optimized WebSocket handler started")
    
    async def stop(self):
        """Stop the WebSocket handler"""
        logger.info("Stopping optimized WebSocket handler")
        
        # Cancel background tasks
        for task in [self.health_check_task, self.batch_processor_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Disconnect all connections
        for connection_id in list(self.connection_pool.connections.keys()):
            await self.disconnect(connection_id)
        
        logger.info("Optimized WebSocket handler stopped")
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """Handle new WebSocket connection"""
        try:
            await websocket.accept()
            connection_id = await self.connection_pool.add_connection(websocket, user_id)
            
            # Update metrics
            self.handler_stats['total_connections'] += 1
            
            # Set connection state
            if connection_id in self.connection_pool.metrics:
                self.connection_pool.metrics[connection_id].state = ConnectionState.CONNECTED
            
            logger.info(f"WebSocket connected: {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        try:
            websocket = self.connection_pool.get_connection(connection_id)
            if websocket:
                await websocket.close()
            
            await self.connection_pool.remove_connection(connection_id)
            logger.info(f"WebSocket disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"WebSocket disconnection error: {e}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any], 
                          compress: bool = None) -> bool:
        """Send message to specific connection"""
        websocket = self.connection_pool.get_connection(connection_id)
        if not websocket:
            return False
        
        try:
            # Serialize message
            message_str = json.dumps(message)
            
            # Apply compression if enabled
            if compress is None:
                compress = self.config.enable_compression
            
            if compress:
                compression_result = self.compressor.compress_message(message_str)
                if compression_result['compressed']:
                    # Send compressed message with header
                    await websocket.send_bytes(compression_result['data'])
                else:
                    await websocket.send_text(message_str)
                
                message_size = compression_result['compressed_size']
            else:
                await websocket.send_text(message_str)
                message_size = len(message_str)
            
            # Update metrics
            self.connection_pool.update_metrics(connection_id, message_size, sent=True)
            self.handler_stats['total_messages_processed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.handler_stats['total_errors'] += 1
            
            # Mark connection as error state
            if connection_id in self.connection_pool.metrics:
                self.connection_pool.metrics[connection_id].record_error()
                self.connection_pool.metrics[connection_id].state = ConnectionState.ERROR
            
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], 
                               user_ids: Optional[List[str]] = None,
                               exclude_connection_ids: Optional[List[str]] = None) -> int:
        """Broadcast message to multiple connections"""
        exclude_connection_ids = exclude_connection_ids or []
        sent_count = 0
        
        if user_ids:
            # Send to specific users
            connection_ids = []
            for user_id in user_ids:
                connection_ids.extend(self.connection_pool.get_user_connections(user_id))
        else:
            # Send to all connections
            connection_ids = list(self.connection_pool.connections.keys())
        
        # Filter out excluded connections
        connection_ids = [cid for cid in connection_ids if cid not in exclude_connection_ids]
        
        # Send messages concurrently
        tasks = []
        for connection_id in connection_ids:
            task = asyncio.create_task(self.send_message(connection_id, message))
            tasks.append(task)
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        sent_count = sum(1 for result in results if result is True)
        
        logger.info(f"Broadcast message sent to {sent_count}/{len(connection_ids)} connections")
        return sent_count
    
    async def receive_message(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Receive message from connection"""
        websocket = self.connection_pool.get_connection(connection_id)
        if not websocket:
            return None
        
        try:
            # Check rate limit
            if not await self.connection_pool.check_rate_limit(connection_id):
                logger.warning(f"Rate limit exceeded for connection {connection_id}")
                return None
            
            # Receive message
            data = await websocket.receive()
            
            if data['type'] == 'websocket.receive':
                if 'bytes' in data:
                    # Compressed message
                    message_str = self.compressor.decompress_message(data['bytes'])
                elif 'text' in data:
                    message_str = data['text']
                else:
                    return None
                
                # Parse message
                message = json.loads(message_str)
                
                # Update metrics
                message_size = len(message_str)
                self.connection_pool.update_metrics(connection_id, message_size, sent=False)
                self.handler_stats['total_messages_processed'] += 1
                
                return message
            
            return None
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return None
        except Exception as e:
            logger.error(f"Failed to receive message from {connection_id}: {e}")
            self.handler_stats['total_errors'] += 1
            return None
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register message handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered message handler for type: {message_type}")
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Handle received message using registered handlers"""
        message_type = message.get('type')
        if not message_type:
            logger.warning(f"Message without type from {connection_id}: {message}")
            return False
        
        handler = self.message_handlers.get(message_type)
        if not handler:
            logger.warning(f"No handler for message type '{message_type}' from {connection_id}")
            return False
        
        try:
            await handler(connection_id, message)
            return True
        except Exception as e:
            logger.error(f"Message handler error for {message_type}: {e}")
            self.handler_stats['total_errors'] += 1
            return False
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.ping_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all connections"""
        ping_tasks = []
        
        for connection_id, websocket in self.connection_pool.connections.items():
            task = asyncio.create_task(self._ping_connection(connection_id, websocket))
            ping_tasks.append(task)
        
        if ping_tasks:
            await asyncio.gather(*ping_tasks, return_exceptions=True)
    
    async def _ping_connection(self, connection_id: str, websocket: WebSocket):
        """Ping individual connection"""
        try:
            start_time = time.time()
            await websocket.ping()
            latency = time.time() - start_time
            
            # Update metrics
            if connection_id in self.connection_pool.metrics:
                self.connection_pool.metrics[connection_id].ping_latency = latency
                
        except Exception as e:
            logger.warning(f"Ping failed for connection {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def _batch_processor_loop(self):
        """Background batch processor loop"""
        while True:
            try:
                await asyncio.sleep(self.config.buffer_flush_interval)
                await self._process_message_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor loop error: {e}")
    
    async def _process_message_batches(self):
        """Process pending message batches"""
        # This would be implemented based on specific batching needs
        # For now, it's a placeholder for batch processing logic
        pass
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_idle_connections(self):
        """Clean up idle connections"""
        current_time = datetime.utcnow()
        idle_connections = []
        
        for connection_id, metrics in self.connection_pool.metrics.items():
            if metrics.idle_time > self.config.connection_timeout:
                idle_connections.append(connection_id)
        
        # Disconnect idle connections
        for connection_id in idle_connections:
            logger.info(f"Disconnecting idle connection: {connection_id}")
            await self.disconnect(connection_id)
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get comprehensive handler statistics"""
        uptime = (datetime.utcnow() - self.handler_stats['started_at']).total_seconds()
        
        return {
            'websocket_handler': {
                'uptime_seconds': uptime,
                'total_connections': self.handler_stats['total_connections'],
                'active_connections': self.connection_pool.connection_count,
                'total_messages_processed': self.handler_stats['total_messages_processed'],
                'total_errors': self.handler_stats['total_errors'],
                'messages_per_second': self.handler_stats['total_messages_processed'] / max(1, uptime),
                'error_rate_percent': (self.handler_stats['total_errors'] / 
                                     max(1, self.handler_stats['total_messages_processed']) * 100)
            },
            'connection_pool': self.connection_pool.get_pool_stats(),
            'compression': self.compressor.get_stats(),
            'registered_handlers': list(self.message_handlers.keys())
        }

# Global optimized WebSocket handler
optimized_websocket_handler = OptimizedWebSocketHandler()

# Export components
__all__ = [
    'OptimizedWebSocketHandler',
    'ConnectionPool',
    'MessageCompressor',
    'WebSocketConfig',
    'ConnectionMetrics',
    'ConnectionState',
    'optimized_websocket_handler'
]
