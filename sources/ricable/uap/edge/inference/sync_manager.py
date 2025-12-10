# edge/inference/sync_manager.py
"""
Edge-Cloud Synchronization Manager
Manages synchronization of models, data, and results between edge devices and cloud.
"""

import asyncio
import time
import json
import logging
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path

try:
    import aiohttp
    import aiofiles
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)

class SyncType(Enum):
    """Types of synchronization"""
    MODEL_DOWNLOAD = "model_download"
    MODEL_UPLOAD = "model_upload"
    RESULT_UPLOAD = "result_upload"
    CONFIG_SYNC = "config_sync"
    TELEMETRY_UPLOAD = "telemetry_upload"
    HEARTBEAT = "heartbeat"
    BATCH_SYNC = "batch_sync"

class SyncStatus(Enum):
    """Synchronization status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ConnectivityMode(Enum):
    """Connectivity modes"""
    ONLINE = "online"
    OFFLINE = "offline"
    INTERMITTENT = "intermittent"
    LOW_BANDWIDTH = "low_bandwidth"

@dataclass
class SyncTask:
    """Synchronization task"""
    task_id: str
    sync_type: SyncType
    status: SyncStatus
    priority: int = 0  # Higher = more important
    data_size_bytes: int = 0
    source_path: Optional[str] = None
    destination_path: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'sync_type': self.sync_type.value,
            'status': self.status.value
        }

@dataclass
class SyncConfig:
    """Configuration for edge-cloud synchronization"""
    cloud_endpoint: str
    device_id: str
    api_key: str
    sync_interval_seconds: int = 60
    batch_size: int = 10
    max_retry_attempts: int = 3
    connection_timeout_seconds: int = 30
    chunk_size_bytes: int = 1024 * 1024  # 1MB chunks
    enable_compression: bool = True
    offline_storage_path: str = "./offline_data"
    heartbeat_interval_seconds: int = 30
    bandwidth_limit_mbps: Optional[float] = None
    enable_delta_sync: bool = True
    sync_priorities: Dict[str, int] = None
    
    def __post_init__(self):
        if self.sync_priorities is None:
            self.sync_priorities = {
                'model_download': 10,
                'result_upload': 8,
                'telemetry_upload': 5,
                'config_sync': 7,
                'heartbeat': 1
            }

class OfflineStorage:
    """Offline storage for sync tasks when cloud is unavailable"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.tasks_file = self.storage_path / "sync_tasks.json"
        self.data_dir = self.storage_path / "data"
        self.data_dir.mkdir(exist_ok=True)
    
    async def store_task(self, task: SyncTask) -> None:
        """Store sync task for offline processing"""
        try:
            # Load existing tasks
            tasks = await self._load_tasks()
            
            # Add new task
            tasks[task.task_id] = task.to_dict()
            
            # Save tasks
            await self._save_tasks(tasks)
            
            # Store task data if available
            if task.payload:
                data_file = self.data_dir / f"{task.task_id}.json"
                async with aiofiles.open(data_file, 'w') as f:
                    await f.write(json.dumps(task.payload, indent=2))
            
            logger.debug(f"Stored offline task: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to store offline task {task.task_id}: {e}")
    
    async def load_pending_tasks(self) -> List[SyncTask]:
        """Load pending sync tasks from offline storage"""
        try:
            tasks_data = await self._load_tasks()
            pending_tasks = []
            
            for task_id, task_dict in tasks_data.items():
                if task_dict['status'] in ['pending', 'failed']:
                    # Load task data if available
                    data_file = self.data_dir / f"{task_id}.json"
                    payload = None
                    
                    if data_file.exists():
                        async with aiofiles.open(data_file, 'r') as f:
                            payload = json.loads(await f.read())
                    
                    # Create task object
                    task = SyncTask(
                        task_id=task_dict['task_id'],
                        sync_type=SyncType(task_dict['sync_type']),
                        status=SyncStatus(task_dict['status']),
                        priority=task_dict.get('priority', 0),
                        data_size_bytes=task_dict.get('data_size_bytes', 0),
                        source_path=task_dict.get('source_path'),
                        destination_path=task_dict.get('destination_path'),
                        payload=payload,
                        created_at=task_dict.get('created_at', time.time()),
                        error_message=task_dict.get('error_message'),
                        retry_count=task_dict.get('retry_count', 0),
                        max_retries=task_dict.get('max_retries', 3),
                        metadata=task_dict.get('metadata', {})
                    )
                    
                    pending_tasks.append(task)
            
            return sorted(pending_tasks, key=lambda t: (t.priority, t.created_at), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to load pending tasks: {e}")
            return []
    
    async def update_task_status(self, task_id: str, status: SyncStatus, error_message: Optional[str] = None) -> None:
        """Update task status in offline storage"""
        try:
            tasks = await self._load_tasks()
            
            if task_id in tasks:
                tasks[task_id]['status'] = status.value
                if status == SyncStatus.COMPLETED:
                    tasks[task_id]['completed_at'] = time.time()
                elif status == SyncStatus.FAILED:
                    tasks[task_id]['error_message'] = error_message
                    tasks[task_id]['retry_count'] = tasks[task_id].get('retry_count', 0) + 1
                elif status == SyncStatus.IN_PROGRESS:
                    tasks[task_id]['started_at'] = time.time()
                
                await self._save_tasks(tasks)
            
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
    
    async def remove_completed_tasks(self, max_age_hours: int = 24) -> None:
        """Remove completed tasks older than specified age"""
        try:
            tasks = await self._load_tasks()
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            tasks_to_remove = []
            for task_id, task_dict in tasks.items():
                if (task_dict['status'] == 'completed' and
                    task_dict.get('completed_at', 0) < cutoff_time):
                    tasks_to_remove.append(task_id)
            
            # Remove tasks
            for task_id in tasks_to_remove:
                del tasks[task_id]
                
                # Remove associated data file
                data_file = self.data_dir / f"{task_id}.json"
                if data_file.exists():
                    data_file.unlink()
            
            if tasks_to_remove:
                await self._save_tasks(tasks)
                logger.info(f"Removed {len(tasks_to_remove)} completed tasks")
            
        except Exception as e:
            logger.error(f"Failed to cleanup completed tasks: {e}")
    
    async def _load_tasks(self) -> Dict[str, Any]:
        """Load tasks from file"""
        if not self.tasks_file.exists():
            return {}
        
        try:
            async with aiofiles.open(self.tasks_file, 'r') as f:
                return json.loads(await f.read())
        except Exception:
            return {}
    
    async def _save_tasks(self, tasks: Dict[str, Any]) -> None:
        """Save tasks to file"""
        async with aiofiles.open(self.tasks_file, 'w') as f:
            await f.write(json.dumps(tasks, indent=2))

class EdgeCloudSyncManager:
    """Manager for edge-cloud synchronization"""
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self.device_id = config.device_id
        
        # Sync state
        self.connectivity_mode = ConnectivityMode.OFFLINE
        self.last_successful_sync = None
        self.is_running = False
        
        # Task management
        self.pending_tasks: List[SyncTask] = []
        self.active_tasks: Dict[str, SyncTask] = {}
        self.sync_queue = asyncio.PriorityQueue()
        
        # Storage
        self.offline_storage = OfflineStorage(config.offline_storage_path)
        
        # Performance tracking
        self.sync_metrics = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'bytes_synced': 0,
            'avg_sync_time_ms': 0.0,
            'connectivity_uptime': 0.0,
            'last_heartbeat': None
        }
        
        # Background tasks
        self.sync_tasks = []
        
        logger.info(f"EdgeCloudSyncManager initialized for device: {self.device_id}")
    
    async def start(self) -> None:
        """Start the synchronization manager"""
        if self.is_running:
            return
        
        logger.info("Starting EdgeCloudSyncManager")
        self.is_running = True
        
        # Load pending tasks from offline storage
        offline_tasks = await self.offline_storage.load_pending_tasks()
        for task in offline_tasks:
            await self._add_task_to_queue(task)
        
        # Start background tasks
        self.sync_tasks = [
            asyncio.create_task(self._sync_worker()),
            asyncio.create_task(self._connectivity_monitor()),
            asyncio.create_task(self._heartbeat_worker()),
            asyncio.create_task(self._cleanup_worker())
        ]
        
        logger.info(f"Started sync manager with {len(offline_tasks)} pending tasks")
    
    async def add_sync_task(self, 
                          sync_type: SyncType,
                          priority: int = 0,
                          source_path: Optional[str] = None,
                          destination_path: Optional[str] = None,
                          payload: Optional[Dict[str, Any]] = None,
                          **kwargs) -> str:
        """Add a synchronization task"""
        task_id = str(uuid.uuid4())
        
        # Calculate data size
        data_size = 0
        if payload:
            data_size = len(json.dumps(payload).encode('utf-8'))
        elif source_path and Path(source_path).exists():
            data_size = Path(source_path).stat().st_size
        
        task = SyncTask(
            task_id=task_id,
            sync_type=sync_type,
            status=SyncStatus.PENDING,
            priority=priority,
            data_size_bytes=data_size,
            source_path=source_path,
            destination_path=destination_path,
            payload=payload,
            **kwargs
        )
        
        # Store offline immediately
        await self.offline_storage.store_task(task)
        
        # Add to queue if online
        if self.connectivity_mode == ConnectivityMode.ONLINE:
            await self._add_task_to_queue(task)
        
        logger.info(f"Added sync task: {task_id} ({sync_type.value})")
        return task_id
    
    async def _add_task_to_queue(self, task: SyncTask) -> None:
        """Add task to priority queue"""
        # Use negative priority for max-heap behavior
        priority = -task.priority
        await self.sync_queue.put((priority, task.created_at, task))
    
    async def _sync_worker(self) -> None:
        """Background worker for processing sync tasks"""
        logger.info("Starting sync worker")
        
        while self.is_running:
            try:
                # Wait for task from queue
                _, _, task = await asyncio.wait_for(
                    self.sync_queue.get(),
                    timeout=1.0
                )
                
                # Process task if online
                if self.connectivity_mode == ConnectivityMode.ONLINE:
                    await self._process_sync_task(task)
                else:
                    # Put task back in queue and wait
                    await self._add_task_to_queue(task)
                    await asyncio.sleep(5.0)
                
            except asyncio.TimeoutError:
                # Check for offline tasks when queue is empty
                if self.connectivity_mode == ConnectivityMode.ONLINE:
                    offline_tasks = await self.offline_storage.load_pending_tasks()
                    for task in offline_tasks[:5]:  # Process up to 5 tasks
                        await self._add_task_to_queue(task)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_sync_task(self, task: SyncTask) -> None:
        """Process a single sync task"""
        task.status = SyncStatus.IN_PROGRESS
        task.started_at = time.time()
        self.active_tasks[task.task_id] = task
        
        await self.offline_storage.update_task_status(task.task_id, SyncStatus.IN_PROGRESS)
        
        try:
            if task.sync_type == SyncType.MODEL_DOWNLOAD:
                await self._download_model(task)
            elif task.sync_type == SyncType.MODEL_UPLOAD:
                await self._upload_model(task)
            elif task.sync_type == SyncType.RESULT_UPLOAD:
                await self._upload_results(task)
            elif task.sync_type == SyncType.CONFIG_SYNC:
                await self._sync_config(task)
            elif task.sync_type == SyncType.TELEMETRY_UPLOAD:
                await self._upload_telemetry(task)
            elif task.sync_type == SyncType.HEARTBEAT:
                await self._send_heartbeat(task)
            else:
                raise ValueError(f"Unknown sync type: {task.sync_type}")
            
            # Mark as completed
            task.status = SyncStatus.COMPLETED
            task.completed_at = time.time()
            
            await self.offline_storage.update_task_status(task.task_id, SyncStatus.COMPLETED)
            
            # Update metrics
            self.sync_metrics['total_syncs'] += 1
            self.sync_metrics['successful_syncs'] += 1
            self.sync_metrics['bytes_synced'] += task.data_size_bytes
            
            sync_time_ms = (task.completed_at - task.started_at) * 1000
            self.sync_metrics['avg_sync_time_ms'] = (
                (self.sync_metrics['avg_sync_time_ms'] * (self.sync_metrics['total_syncs'] - 1) + sync_time_ms) /
                self.sync_metrics['total_syncs']
            )
            
            logger.info(f"Completed sync task: {task.task_id} in {sync_time_ms:.1f}ms")
            
        except Exception as e:
            # Mark as failed
            task.status = SyncStatus.FAILED
            task.error_message = str(e)
            
            await self.offline_storage.update_task_status(task.task_id, SyncStatus.FAILED, str(e))
            
            self.sync_metrics['total_syncs'] += 1
            self.sync_metrics['failed_syncs'] += 1
            
            logger.error(f"Failed sync task {task.task_id}: {e}")
            
            # Retry if not exceeded max attempts
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = SyncStatus.PENDING
                
                # Add back to queue with delay
                await asyncio.sleep(min(2 ** task.retry_count, 60))  # Exponential backoff
                await self._add_task_to_queue(task)
        
        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _download_model(self, task: SyncTask) -> None:
        """Download model from cloud"""
        if not HTTP_AVAILABLE:
            raise ImportError("aiohttp required for model download")
        
        url = f"{self.config.cloud_endpoint}/models/{task.payload['model_id']}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.connection_timeout_seconds)) as session:
            headers = {'Authorization': f'Bearer {self.config.api_key}'}
            
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                
                # Download in chunks
                if task.destination_path:
                    async with aiofiles.open(task.destination_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.config.chunk_size_bytes):
                            await f.write(chunk)
    
    async def _upload_model(self, task: SyncTask) -> None:
        """Upload model to cloud"""
        if not HTTP_AVAILABLE:
            raise ImportError("aiohttp required for model upload")
        
        url = f"{self.config.cloud_endpoint}/models"
        
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.config.api_key}'}
            
            # Upload file
            if task.source_path:
                data = aiohttp.FormData()
                data.add_field('file',
                              open(task.source_path, 'rb'),
                              filename=Path(task.source_path).name)
                
                async with session.post(url, headers=headers, data=data) as response:
                    response.raise_for_status()
    
    async def _upload_results(self, task: SyncTask) -> None:
        """Upload inference results to cloud"""
        if not HTTP_AVAILABLE:
            raise ImportError("aiohttp required for result upload")
        
        url = f"{self.config.cloud_endpoint}/results"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'results': task.payload
            }
            
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
    
    async def _sync_config(self, task: SyncTask) -> None:
        """Sync configuration with cloud"""
        if not HTTP_AVAILABLE:
            raise ImportError("aiohttp required for config sync")
        
        url = f"{self.config.cloud_endpoint}/devices/{self.device_id}/config"
        
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.config.api_key}'}
            
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                config_data = await response.json()
                
                # Update local configuration
                if task.destination_path:
                    async with aiofiles.open(task.destination_path, 'w') as f:
                        await f.write(json.dumps(config_data, indent=2))
    
    async def _upload_telemetry(self, task: SyncTask) -> None:
        """Upload telemetry data to cloud"""
        if not HTTP_AVAILABLE:
            raise ImportError("aiohttp required for telemetry upload")
        
        url = f"{self.config.cloud_endpoint}/telemetry"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'telemetry': task.payload
            }
            
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
    
    async def _send_heartbeat(self, task: SyncTask) -> None:
        """Send heartbeat to cloud"""
        if not HTTP_AVAILABLE:
            raise ImportError("aiohttp required for heartbeat")
        
        url = f"{self.config.cloud_endpoint}/devices/{self.device_id}/heartbeat"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'timestamp': time.time(),
                'status': 'active',
                'metrics': self.sync_metrics
            }
            
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                self.sync_metrics['last_heartbeat'] = time.time()
    
    async def _connectivity_monitor(self) -> None:
        """Monitor connectivity to cloud"""
        logger.info("Starting connectivity monitor")
        
        while self.is_running:
            try:
                # Test connectivity
                is_online = await self._test_connectivity()
                
                previous_mode = self.connectivity_mode
                
                if is_online:
                    self.connectivity_mode = ConnectivityMode.ONLINE
                    if previous_mode != ConnectivityMode.ONLINE:
                        logger.info("Connectivity restored")
                        
                        # Load and queue offline tasks
                        offline_tasks = await self.offline_storage.load_pending_tasks()
                        for task in offline_tasks:
                            await self._add_task_to_queue(task)
                else:
                    self.connectivity_mode = ConnectivityMode.OFFLINE
                    if previous_mode == ConnectivityMode.ONLINE:
                        logger.warning("Connectivity lost - entering offline mode")
                
                # Update uptime metric
                if self.connectivity_mode == ConnectivityMode.ONLINE:
                    self.sync_metrics['connectivity_uptime'] += self.config.sync_interval_seconds
                
                await asyncio.sleep(self.config.sync_interval_seconds)
                
            except Exception as e:
                logger.error(f"Connectivity monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _test_connectivity(self) -> bool:
        """Test connectivity to cloud"""
        if not HTTP_AVAILABLE:
            return False
        
        try:
            url = f"{self.config.cloud_endpoint}/health"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                headers = {'Authorization': f'Bearer {self.config.api_key}'}
                
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
                    
        except Exception:
            return False
    
    async def _heartbeat_worker(self) -> None:
        """Send periodic heartbeats"""
        while self.is_running:
            try:
                if self.connectivity_mode == ConnectivityMode.ONLINE:
                    await self.add_sync_task(
                        sync_type=SyncType.HEARTBEAT,
                        priority=self.config.sync_priorities.get('heartbeat', 1)
                    )
                
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
                
            except Exception as e:
                logger.error(f"Heartbeat worker error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
    
    async def _cleanup_worker(self) -> None:
        """Cleanup completed tasks periodically"""
        while self.is_running:
            try:
                await self.offline_storage.remove_completed_tasks()
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(3600)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        return {
            'device_id': self.device_id,
            'connectivity_mode': self.connectivity_mode.value,
            'is_running': self.is_running,
            'pending_tasks': self.sync_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'last_successful_sync': self.last_successful_sync,
            'metrics': self.sync_metrics,
            'config': {
                'cloud_endpoint': self.config.cloud_endpoint,
                'sync_interval_seconds': self.config.sync_interval_seconds,
                'offline_storage_path': self.config.offline_storage_path
            }
        }
    
    async def force_sync(self) -> None:
        """Force immediate synchronization of all pending tasks"""
        logger.info("Forcing immediate sync")
        
        if self.connectivity_mode != ConnectivityMode.ONLINE:
            logger.warning("Cannot force sync - device is offline")
            return
        
        # Load all pending tasks and queue them
        offline_tasks = await self.offline_storage.load_pending_tasks()
        for task in offline_tasks:
            task.priority = 100  # High priority for forced sync
            await self._add_task_to_queue(task)
    
    async def stop(self) -> None:
        """Stop the synchronization manager"""
        logger.info("Stopping EdgeCloudSyncManager")
        
        self.is_running = False
        
        # Wait for background tasks to complete
        if self.sync_tasks:
            await asyncio.gather(*self.sync_tasks, return_exceptions=True)
        
        # Final cleanup
        await self.offline_storage.remove_completed_tasks()
        
        logger.info("EdgeCloudSyncManager stopped")

# Global sync manager
sync_manager = None

# Convenience functions
async def initialize_edge_cloud_sync(cloud_endpoint: str,
                                    device_id: str,
                                    api_key: str,
                                    **kwargs) -> EdgeCloudSyncManager:
    """Initialize edge-cloud synchronization"""
    global sync_manager
    
    config = SyncConfig(
        cloud_endpoint=cloud_endpoint,
        device_id=device_id,
        api_key=api_key,
        **kwargs
    )
    
    sync_manager = EdgeCloudSyncManager(config)
    await sync_manager.start()
    
    return sync_manager

async def sync_model_to_cloud(model_path: str, priority: int = 5) -> str:
    """Sync model to cloud"""
    if sync_manager is None:
        raise RuntimeError("Sync manager not initialized")
    
    return await sync_manager.add_sync_task(
        sync_type=SyncType.MODEL_UPLOAD,
        priority=priority,
        source_path=model_path
    )

async def download_model_from_cloud(model_id: str, destination_path: str, priority: int = 10) -> str:
    """Download model from cloud"""
    if sync_manager is None:
        raise RuntimeError("Sync manager not initialized")
    
    return await sync_manager.add_sync_task(
        sync_type=SyncType.MODEL_DOWNLOAD,
        priority=priority,
        destination_path=destination_path,
        payload={'model_id': model_id}
    )

async def upload_inference_results(results: Dict[str, Any], priority: int = 8) -> str:
    """Upload inference results to cloud"""
    if sync_manager is None:
        raise RuntimeError("Sync manager not initialized")
    
    return await sync_manager.add_sync_task(
        sync_type=SyncType.RESULT_UPLOAD,
        priority=priority,
        payload=results
    )

async def sync_device_telemetry(telemetry_data: Dict[str, Any], priority: int = 5) -> str:
    """Sync device telemetry to cloud"""
    if sync_manager is None:
        raise RuntimeError("Sync manager not initialized")
    
    return await sync_manager.add_sync_task(
        sync_type=SyncType.TELEMETRY_UPLOAD,
        priority=priority,
        payload=telemetry_data
    )

def get_sync_status() -> Dict[str, Any]:
    """Get current sync status"""
    if sync_manager is None:
        return {'status': 'not_initialized'}
    
    return sync_manager.get_sync_status()

async def force_cloud_sync() -> None:
    """Force immediate sync with cloud"""
    if sync_manager is None:
        raise RuntimeError("Sync manager not initialized")
    
    await sync_manager.force_sync()
