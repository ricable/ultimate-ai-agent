import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from uuid import uuid4

import aioredis
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.service import DatabaseService
from backend.models.conversation import MobileSession, MobileSyncOperation
from backend.monitoring.logs.logger import get_logger
from backend.services.auth import AuthService
from .edge_manager import EdgeManager

logger = get_logger(__name__)

@dataclass
class MobileClient:
    """Represents a connected mobile client"""
    session_id: str
    user_id: str
    device_id: str
    platform: str  # ios, android
    app_version: str
    connected_at: datetime
    last_ping: datetime
    websocket: WebSocket
    is_online: bool = True
    sync_enabled: bool = True

@dataclass
class SyncOperation:
    """Represents a sync operation between mobile and backend"""
    id: str
    client_session_id: str
    operation_type: str  # 'upload', 'download', 'execute', 'sync'
    data: Dict[str, Any]
    priority: int = 5  # 1 (highest) to 10 (lowest)
    created_at: datetime = None
    attempts: int = 0
    max_attempts: int = 3
    status: str = 'pending'  # pending, processing, completed, failed
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class MobileBridge:
    """Bridge service for mobile-edge communication and synchronization"""
    
    def __init__(
        self,
        db_service: DatabaseService,
        edge_manager: EdgeManager,
        auth_service: AuthService,
        redis_url: str = "redis://localhost:6379"
    ):
        self.db_service = db_service
        self.edge_manager = edge_manager
        self.auth_service = auth_service
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        
        # Active connections
        self.clients: Dict[str, MobileClient] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        
        # Sync operations
        self.sync_queue: asyncio.Queue = asyncio.Queue()
        self.pending_operations: Dict[str, SyncOperation] = {}
        
        # Background tasks
        self.is_running = False
        self.sync_worker_task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'clients_connected': 0,
            'total_connections': 0,
            'sync_operations_completed': 0,
            'sync_operations_failed': 0,
            'data_synced_bytes': 0,
        }
    
    async def initialize(self) -> None:
        """Initialize the mobile bridge"""
        logger.info("Initializing MobileBridge")
        
        try:
            # Initialize Redis connection
            self.redis = await aioredis.from_url(self.redis_url)
            
            # Load pending operations from database
            await self._load_pending_operations()
            
            # Start background tasks
            self.is_running = True
            self.sync_worker_task = asyncio.create_task(self._sync_worker())
            self.ping_task = asyncio.create_task(self._ping_worker())
            
            logger.info("MobileBridge initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MobileBridge: {e}")
            raise
    
    async def connect_client(
        self,
        websocket: WebSocket,
        token: str,
        device_id: str,
        platform: str,
        app_version: str
    ) -> str:
        """Connect a mobile client"""
        try:
            # Validate token and get user
            user = await self.auth_service.validate_token(token)
            if not user:
                raise ValueError("Invalid authentication token")
            
            # Create client session
            session_id = str(uuid4())
            client = MobileClient(
                session_id=session_id,
                user_id=user.id,
                device_id=device_id,
                platform=platform,
                app_version=app_version,
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow(),
                websocket=websocket
            )
            
            # Store client
            self.clients[session_id] = client
            
            # Track user sessions
            if user.id not in self.user_sessions:
                self.user_sessions[user.id] = set()
            self.user_sessions[user.id].add(session_id)
            
            # Store session in database
            await self._store_session(client)
            
            # Cache in Redis
            await self.redis.setex(
                f"mobile:session:{session_id}",
                3600,  # 1 hour TTL
                json.dumps({
                    'user_id': user.id,
                    'device_id': device_id,
                    'platform': platform,
                    'connected_at': client.connected_at.isoformat(),
                }, default=str)
            )
            
            # Update statistics
            self.stats['clients_connected'] += 1
            self.stats['total_connections'] += 1
            
            # Send welcome message
            await self._send_to_client(session_id, {
                'type': 'connection_established',
                'session_id': session_id,
                'server_time': datetime.utcnow().isoformat(),
            })
            
            # Start sync for this client
            await self._trigger_client_sync(session_id)
            
            logger.info(f"Mobile client connected: {session_id} (user: {user.id})")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to connect mobile client: {e}")
            raise
    
    async def disconnect_client(self, session_id: str) -> None:
        """Disconnect a mobile client"""
        if session_id not in self.clients:
            return
        
        client = self.clients[session_id]
        
        try:
            # Remove from tracking
            del self.clients[session_id]
            
            # Update user sessions
            if client.user_id in self.user_sessions:
                self.user_sessions[client.user_id].discard(session_id)
                if not self.user_sessions[client.user_id]:
                    del self.user_sessions[client.user_id]
            
            # Update database
            async with self.db_service.get_session() as session:
                await session.execute(
                    update(MobileSession)
                    .where(MobileSession.session_id == session_id)
                    .values(
                        disconnected_at=datetime.utcnow(),
                        is_active=False
                    )
                )
                await session.commit()
            
            # Remove from Redis
            await self.redis.delete(f"mobile:session:{session_id}")
            
            # Update statistics
            self.stats['clients_connected'] -= 1
            
            logger.info(f"Mobile client disconnected: {session_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting client {session_id}: {e}")
    
    async def handle_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle message from mobile client"""
        if session_id not in self.clients:
            logger.warning(f"Message from unknown session: {session_id}")
            return
        
        client = self.clients[session_id]
        client.last_ping = datetime.utcnow()
        
        try:
            message_type = message.get('type')
            
            if message_type == 'ping':
                await self._handle_ping(session_id, message)
            elif message_type == 'sync_request':
                await self._handle_sync_request(session_id, message)
            elif message_type == 'agent_interaction':
                await self._handle_agent_interaction(session_id, message)
            elif message_type == 'document_upload':
                await self._handle_document_upload(session_id, message)
            elif message_type == 'offline_operation':
                await self._handle_offline_operation(session_id, message)
            else:
                logger.warning(f"Unknown message type from {session_id}: {message_type}")
                await self._send_error(session_id, f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling message from {session_id}: {e}")
            await self._send_error(session_id, "Failed to process message")
    
    async def sync_user_data(self, user_id: str, operation_type: str, data: Dict[str, Any]) -> None:
        """Sync data to all sessions for a user"""
        if user_id not in self.user_sessions:
            # User not connected, store for later sync
            await self._store_pending_sync(user_id, operation_type, data)
            return
        
        # Send to all active sessions for this user
        for session_id in self.user_sessions[user_id].copy():
            if session_id in self.clients:
                await self._send_to_client(session_id, {
                    'type': 'sync_data',
                    'operation_type': operation_type,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat(),
                })
    
    async def get_client_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a connected client"""
        if session_id not in self.clients:
            return None
        
        client = self.clients[session_id]
        return {
            'session_id': client.session_id,
            'user_id': client.user_id,
            'device_id': client.device_id,
            'platform': client.platform,
            'app_version': client.app_version,
            'connected_at': client.connected_at,
            'last_ping': client.last_ping,
            'is_online': client.is_online,
        }
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        sessions = []
        
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id]:
                if session_id in self.clients:
                    client_info = await self.get_client_info(session_id)
                    if client_info:
                        sessions.append(client_info)
        
        return sessions
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get mobile bridge statistics"""
        return {
            **self.stats,
            'active_clients': len(self.clients),
            'active_users': len(self.user_sessions),
            'pending_sync_operations': len(self.pending_operations),
            'sync_queue_size': self.sync_queue.qsize(),
        }
    
    async def shutdown(self) -> None:
        """Shutdown the mobile bridge"""
        logger.info("Shutting down MobileBridge")
        
        self.is_running = False
        
        # Cancel background tasks
        if self.sync_worker_task:
            self.sync_worker_task.cancel()
        if self.ping_task:
            self.ping_task.cancel()
        
        # Disconnect all clients
        session_ids = list(self.clients.keys())
        for session_id in session_ids:
            await self.disconnect_client(session_id)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        logger.info("MobileBridge shut down")
    
    # Private methods
    
    async def _handle_ping(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle ping from client"""
        await self._send_to_client(session_id, {
            'type': 'pong',
            'timestamp': datetime.utcnow().isoformat(),
        })
    
    async def _handle_sync_request(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle sync request from client"""
        try:
            sync_type = message.get('sync_type', 'full')
            last_sync = message.get('last_sync')
            
            # Create sync operation
            operation = SyncOperation(
                id=str(uuid4()),
                client_session_id=session_id,
                operation_type='sync',
                data={
                    'sync_type': sync_type,
                    'last_sync': last_sync,
                },
                priority=3  # High priority for sync requests
            )
            
            await self._queue_sync_operation(operation)
            
        except Exception as e:
            logger.error(f"Error handling sync request from {session_id}: {e}")
            await self._send_error(session_id, "Failed to process sync request")
    
    async def _handle_agent_interaction(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle agent interaction from mobile client"""
        try:
            client = self.clients[session_id]
            
            # Extract interaction data
            agent_id = message.get('agent_id')
            interaction_type = message.get('interaction_type')
            data = message.get('data')
            
            if not agent_id or not interaction_type:
                raise ValueError("Missing agent_id or interaction_type")
            
            # Process through edge manager if it's a computation
            if interaction_type == 'execute_function':
                result = await self.edge_manager.execute_function(
                    instance_id=data.get('instance_id'),
                    function_name=data.get('function_name'),
                    args=data.get('args', [])
                )
                
                await self._send_to_client(session_id, {
                    'type': 'agent_interaction_result',
                    'request_id': message.get('request_id'),
                    'result': result,
                })
            else:
                # Handle other interaction types
                await self._send_to_client(session_id, {
                    'type': 'agent_interaction_acknowledged',
                    'request_id': message.get('request_id'),
                })
        
        except Exception as e:
            logger.error(f"Error handling agent interaction from {session_id}: {e}")
            await self._send_error(session_id, "Failed to process agent interaction")
    
    async def _handle_document_upload(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle document upload from mobile client"""
        try:
            # Create upload operation
            operation = SyncOperation(
                id=str(uuid4()),
                client_session_id=session_id,
                operation_type='upload',
                data=message.get('data', {}),
                priority=5  # Normal priority for uploads
            )
            
            await self._queue_sync_operation(operation)
            
        except Exception as e:
            logger.error(f"Error handling document upload from {session_id}: {e}")
            await self._send_error(session_id, "Failed to process document upload")
    
    async def _handle_offline_operation(self, session_id: str, message: Dict[str, Any]) -> None:
        """Handle offline operation sync from mobile client"""
        try:
            operations = message.get('operations', [])
            
            for op_data in operations:
                operation = SyncOperation(
                    id=op_data.get('id', str(uuid4())),
                    client_session_id=session_id,
                    operation_type=op_data.get('type'),
                    data=op_data.get('data', {}),
                    priority=op_data.get('priority', 5)
                )
                
                await self._queue_sync_operation(operation)
            
            await self._send_to_client(session_id, {
                'type': 'offline_operations_queued',
                'count': len(operations),
            })
            
        except Exception as e:
            logger.error(f"Error handling offline operations from {session_id}: {e}")
            await self._send_error(session_id, "Failed to process offline operations")
    
    async def _send_to_client(self, session_id: str, message: Dict[str, Any]) -> None:
        """Send message to mobile client"""
        if session_id not in self.clients:
            return
        
        client = self.clients[session_id]
        
        try:
            await client.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message to client {session_id}: {e}")
            # Mark client as offline and clean up
            await self.disconnect_client(session_id)
    
    async def _send_error(self, session_id: str, error_message: str) -> None:
        """Send error message to client"""
        await self._send_to_client(session_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
        })
    
    async def _queue_sync_operation(self, operation: SyncOperation) -> None:
        """Queue a sync operation for processing"""
        self.pending_operations[operation.id] = operation
        await self.sync_queue.put(operation)
        
        # Store in database
        await self._store_sync_operation(operation)
    
    async def _sync_worker(self) -> None:
        """Background worker for processing sync operations"""
        logger.info("Starting sync worker")
        
        while self.is_running:
            try:
                # Get operation from queue with timeout
                operation = await asyncio.wait_for(
                    self.sync_queue.get(),
                    timeout=5.0
                )
                
                await self._process_sync_operation(operation)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                await asyncio.sleep(1)
    
    async def _process_sync_operation(self, operation: SyncOperation) -> None:
        """Process a single sync operation"""
        try:
            operation.attempts += 1
            operation.status = 'processing'
            
            # Process based on operation type
            if operation.operation_type == 'sync':
                await self._process_sync_operation_sync(operation)
            elif operation.operation_type == 'upload':
                await self._process_sync_operation_upload(operation)
            elif operation.operation_type == 'download':
                await self._process_sync_operation_download(operation)
            else:
                logger.warning(f"Unknown sync operation type: {operation.operation_type}")
            
            operation.status = 'completed'
            self.stats['sync_operations_completed'] += 1
            
            # Notify client of completion
            await self._send_to_client(operation.client_session_id, {
                'type': 'sync_operation_completed',
                'operation_id': operation.id,
                'operation_type': operation.operation_type,
            })
            
        except Exception as e:
            logger.error(f"Failed to process sync operation {operation.id}: {e}")
            operation.status = 'failed'
            self.stats['sync_operations_failed'] += 1
            
            if operation.attempts < operation.max_attempts:
                # Retry later
                operation.status = 'pending'
                await asyncio.sleep(2 ** operation.attempts)  # Exponential backoff
                await self.sync_queue.put(operation)
            else:
                # Max attempts reached, notify client
                await self._send_to_client(operation.client_session_id, {
                    'type': 'sync_operation_failed',
                    'operation_id': operation.id,
                    'error': str(e),
                })
        
        finally:
            # Clean up completed/failed operations
            if operation.status in ['completed', 'failed']:
                self.pending_operations.pop(operation.id, None)
    
    async def _process_sync_operation_sync(self, operation: SyncOperation) -> None:
        """Process a sync operation"""
        # Implement sync logic here
        pass
    
    async def _process_sync_operation_upload(self, operation: SyncOperation) -> None:
        """Process an upload operation"""
        # Implement upload logic here
        pass
    
    async def _process_sync_operation_download(self, operation: SyncOperation) -> None:
        """Process a download operation"""
        # Implement download logic here
        pass
    
    async def _ping_worker(self) -> None:
        """Background worker for checking client connections"""
        logger.info("Starting ping worker")
        
        while self.is_running:
            try:
                now = datetime.utcnow()
                stale_clients = []
                
                for session_id, client in self.clients.items():
                    # Check if client hasn't pinged in 2 minutes
                    if now - client.last_ping > timedelta(minutes=2):
                        stale_clients.append(session_id)
                
                # Disconnect stale clients
                for session_id in stale_clients:
                    logger.warning(f"Disconnecting stale client: {session_id}")
                    await self.disconnect_client(session_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Ping worker error: {e}")
                await asyncio.sleep(30)
    
    async def _store_session(self, client: MobileClient) -> None:
        """Store mobile session in database"""
        try:
            async with self.db_service.get_session() as session:
                mobile_session = MobileSession(
                    session_id=client.session_id,
                    user_id=client.user_id,
                    device_id=client.device_id,
                    platform=client.platform,
                    app_version=client.app_version,
                    connected_at=client.connected_at,
                    is_active=True
                )
                session.add(mobile_session)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store mobile session: {e}")
    
    async def _store_sync_operation(self, operation: SyncOperation) -> None:
        """Store sync operation in database"""
        try:
            async with self.db_service.get_session() as session:
                sync_op = MobileSyncOperation(
                    operation_id=operation.id,
                    session_id=operation.client_session_id,
                    operation_type=operation.operation_type,
                    data=operation.data,
                    priority=operation.priority,
                    status=operation.status,
                    created_at=operation.created_at,
                    attempts=operation.attempts
                )
                session.add(sync_op)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store sync operation: {e}")
    
    async def _load_pending_operations(self) -> None:
        """Load pending operations from database on startup"""
        try:
            async with self.db_service.get_session() as session:
                result = await session.execute(
                    select(MobileSyncOperation)
                    .where(MobileSyncOperation.status == 'pending')
                    .order_by(MobileSyncOperation.priority.asc())
                )
                operations = result.scalars().all()
                
                for op in operations:
                    sync_op = SyncOperation(
                        id=op.operation_id,
                        client_session_id=op.session_id,
                        operation_type=op.operation_type,
                        data=op.data,
                        priority=op.priority,
                        created_at=op.created_at,
                        attempts=op.attempts,
                        status=op.status
                    )
                    
                    self.pending_operations[sync_op.id] = sync_op
                    await self.sync_queue.put(sync_op)
                
                logger.info(f"Loaded {len(operations)} pending sync operations")
                
        except Exception as e:
            logger.error(f"Failed to load pending operations: {e}")
    
    async def _store_pending_sync(self, user_id: str, operation_type: str, data: Dict[str, Any]) -> None:
        """Store pending sync operation for offline user"""
        try:
            operation = SyncOperation(
                id=str(uuid4()),
                client_session_id='offline',
                operation_type=operation_type,
                data={'user_id': user_id, **data}
            )
            
            await self._store_sync_operation(operation)
            logger.info(f"Stored pending sync for offline user: {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store pending sync: {e}")
    
    async def _trigger_client_sync(self, session_id: str) -> None:
        """Trigger sync for a newly connected client"""
        try:
            client = self.clients[session_id]
            
            # Check for pending operations for this user
            async with self.db_service.get_session() as session:
                result = await session.execute(
                    select(MobileSyncOperation)
                    .where(
                        MobileSyncOperation.data.contains({'user_id': client.user_id}),
                        MobileSyncOperation.status == 'pending'
                    )
                )
                pending_ops = result.scalars().all()
                
                if pending_ops:
                    await self._send_to_client(session_id, {
                        'type': 'sync_available',
                        'pending_operations': len(pending_ops),
                    })
                    
                    logger.info(f"Triggered sync for client {session_id} with {len(pending_ops)} operations")
        
        except Exception as e:
            logger.error(f"Failed to trigger client sync: {e}")