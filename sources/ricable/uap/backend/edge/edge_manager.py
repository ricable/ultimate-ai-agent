import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import aioredis
from fastapi import HTTPException
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.service import DatabaseService
from backend.models.conversation import EdgeModule, EdgeInstance, EdgeExecution
from backend.monitoring.metrics.prometheus_metrics import metrics_collector
from backend.monitoring.logs.logger import get_logger
from backend.security.encryption import SecurityManager

logger = get_logger(__name__)

@dataclass
class EdgeModuleConfig:
    """Configuration for WebAssembly edge modules"""
    name: str
    wasm_path: str
    version: str
    description: str
    memory_initial: int = 256  # Pages (16MB)
    memory_maximum: int = 1024  # Pages (64MB)
    timeout_ms: int = 30000
    max_instances: int = 10
    permissions: Dict[str, bool] = None
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {
                'file_system': False,
                'network': False,
                'compute': True,
            }
        if self.env_vars is None:
            self.env_vars = {}

@dataclass
class EdgeExecutionContext:
    """Context for edge execution"""
    module_id: str
    request_id: str
    user_id: Optional[str] = None
    timeout: int = 30000
    max_memory: int = 64 * 1024 * 1024  # 64MB
    max_cpu_time: int = 10000  # 10 seconds
    priority: str = 'normal'  # low, normal, high
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EdgeExecutionResult:
    """Result of edge execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    memory_used: int = 0
    cpu_time_ms: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EdgeManager:
    """Manages WebAssembly edge runtime and execution"""
    
    def __init__(self, db_service: DatabaseService, redis_url: str = "redis://localhost:6379"):
        self.db_service = db_service
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.security_manager = SecurityManager()
        
        # Runtime state
        self.modules: Dict[str, EdgeModuleConfig] = {}
        self.instances: Dict[str, Dict[str, Any]] = {}  # instance_id -> instance_data
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        
        # Statistics
        self.stats = {
            'modules_loaded': 0,
            'instances_created': 0,
            'executions_completed': 0,
            'executions_failed': 0,
            'total_execution_time_ms': 0,
        }
    
    async def initialize(self) -> None:
        """Initialize the edge manager"""
        logger.info("Initializing EdgeManager")
        
        try:
            # Initialize Redis connection
            self.redis = await aioredis.from_url(self.redis_url)
            
            # Load existing modules from database
            await self._load_modules_from_db()
            
            # Start execution worker
            self.is_running = True
            asyncio.create_task(self._execution_worker())
            
            # Initialize metrics
            metrics_collector.edge_manager_initialized.inc()
            
            logger.info("EdgeManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EdgeManager: {e}")
            raise
    
    async def load_module(self, config: EdgeModuleConfig) -> None:
        """Load a WebAssembly module"""
        logger.info(f"Loading edge module: {config.name}")
        
        try:
            # Validate module configuration
            await self._validate_module_config(config)
            
            # Security validation
            await self.security_manager.validate_wasm_module(config.wasm_path)
            
            # Store in database
            async with self.db_service.get_session() as session:
                module = EdgeModule(
                    name=config.name,
                    wasm_path=config.wasm_path,
                    version=config.version,
                    description=config.description,
                    config=asdict(config),
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                session.add(module)
                await session.commit()
            
            # Store in memory
            self.modules[config.name] = config
            self.stats['modules_loaded'] += 1
            
            # Cache in Redis
            await self.redis.setex(
                f"edge:module:{config.name}",
                3600,  # 1 hour TTL
                json.dumps(asdict(config))
            )
            
            metrics_collector.edge_modules_loaded.inc()
            logger.info(f"Module {config.name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load module {config.name}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load module: {e}")
    
    async def create_instance(self, module_name: str, context: EdgeExecutionContext) -> str:
        """Create a WebAssembly instance"""
        logger.info(f"Creating instance for module: {module_name}")
        
        try:
            if module_name not in self.modules:
                raise ValueError(f"Module {module_name} not found")
            
            config = self.modules[module_name]
            
            # Check instance limits
            active_instances = sum(1 for inst_data in self.instances.values() 
                                 if inst_data['module_name'] == module_name)
            
            if active_instances >= config.max_instances:
                raise ValueError(f"Maximum instances ({config.max_instances}) reached for module {module_name}")
            
            # Generate instance ID
            instance_id = f"{module_name}_{context.request_id}_{int(time.time())}"
            
            # Create instance data
            instance_data = {
                'id': instance_id,
                'module_name': module_name,
                'context': asdict(context),
                'created_at': datetime.utcnow(),
                'last_used': datetime.utcnow(),
                'execution_count': 0,
                'total_execution_time_ms': 0,
            }
            
            # Store in database
            async with self.db_service.get_session() as session:
                instance = EdgeInstance(
                    instance_id=instance_id,
                    module_name=module_name,
                    user_id=context.user_id,
                    context=asdict(context),
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                session.add(instance)
                await session.commit()
            
            # Store in memory
            self.instances[instance_id] = instance_data
            self.stats['instances_created'] += 1
            
            # Cache in Redis
            await self.redis.setex(
                f"edge:instance:{instance_id}",
                1800,  # 30 minutes TTL
                json.dumps(instance_data, default=str)
            )
            
            metrics_collector.edge_instances_created.inc()
            logger.info(f"Instance created: {instance_id}")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create instance for {module_name}: {e}")
            raise
    
    async def execute_function(
        self,
        instance_id: str,
        function_name: str,
        args: List[Any] = None,
        context: Optional[EdgeExecutionContext] = None
    ) -> EdgeExecutionResult:
        """Execute a function in a WebAssembly instance"""
        if args is None:
            args = []
        
        logger.info(f"Executing function {function_name} on instance {instance_id}")
        
        try:
            if instance_id not in self.instances:
                raise ValueError(f"Instance {instance_id} not found")
            
            instance_data = self.instances[instance_id]
            module_name = instance_data['module_name']
            config = self.modules[module_name]
            
            # Create execution context if not provided
            if context is None:
                context = EdgeExecutionContext(
                    module_id=module_name,
                    request_id=f"exec_{int(time.time())}",
                    timeout=config.timeout_ms
                )
            
            # Security validation
            await self.security_manager.validate_execution(
                instance_id, function_name, context
            )
            
            # Execute via queue for resource management
            execution_task = {
                'instance_id': instance_id,
                'function_name': function_name,
                'args': args,
                'context': context,
                'timestamp': time.time(),
            }
            
            await self.execution_queue.put(execution_task)
            
            # For now, simulate execution (in production, this would call actual WASM runtime)
            result = await self._simulate_execution(execution_task)
            
            # Update statistics
            instance_data['execution_count'] += 1
            instance_data['total_execution_time_ms'] += result.execution_time_ms
            instance_data['last_used'] = datetime.utcnow()
            
            if result.success:
                self.stats['executions_completed'] += 1
            else:
                self.stats['executions_failed'] += 1
            
            self.stats['total_execution_time_ms'] += result.execution_time_ms
            
            # Record in database
            await self._record_execution(instance_id, function_name, result)
            
            # Update metrics
            metrics_collector.edge_function_executions.inc()
            metrics_collector.edge_execution_duration.observe(
                result.execution_time_ms / 1000.0
            )
            
            logger.info(f"Function {function_name} executed in {result.execution_time_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute function {function_name}: {e}")
            self.stats['executions_failed'] += 1
            
            return EdgeExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=0
            )
    
    async def destroy_instance(self, instance_id: str) -> None:
        """Destroy a WebAssembly instance"""
        logger.info(f"Destroying instance: {instance_id}")
        
        try:
            if instance_id in self.instances:
                # Remove from memory
                del self.instances[instance_id]
                
                # Remove from Redis
                await self.redis.delete(f"edge:instance:{instance_id}")
                
                # Update database
                async with self.db_service.get_session() as session:
                    await session.execute(
                        update(EdgeInstance)
                        .where(EdgeInstance.instance_id == instance_id)
                        .values(is_active=False, destroyed_at=datetime.utcnow())
                    )
                    await session.commit()
                
                metrics_collector.edge_instances_destroyed.inc()
                logger.info(f"Instance {instance_id} destroyed")
            
        except Exception as e:
            logger.error(f"Failed to destroy instance {instance_id}: {e}")
            raise
    
    async def get_instance_stats(self, instance_id: str) -> Dict[str, Any]:
        """Get statistics for an instance"""
        if instance_id not in self.instances:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance_data = self.instances[instance_id]
        
        return {
            'instance_id': instance_id,
            'module_name': instance_data['module_name'],
            'created_at': instance_data['created_at'],
            'last_used': instance_data['last_used'],
            'execution_count': instance_data['execution_count'],
            'total_execution_time_ms': instance_data['total_execution_time_ms'],
            'average_execution_time_ms': (
                instance_data['total_execution_time_ms'] / instance_data['execution_count']
                if instance_data['execution_count'] > 0 else 0
            ),
        }
    
    async def list_modules(self) -> List[Dict[str, Any]]:
        """List all loaded modules"""
        return [
            {
                'name': config.name,
                'version': config.version,
                'description': config.description,
                'active_instances': sum(
                    1 for inst_data in self.instances.values()
                    if inst_data['module_name'] == config.name
                ),
                'max_instances': config.max_instances,
            }
            for config in self.modules.values()
        ]
    
    async def list_instances(self) -> List[Dict[str, Any]]:
        """List all active instances"""
        return [
            {
                'instance_id': instance_id,
                'module_name': data['module_name'],
                'created_at': data['created_at'],
                'execution_count': data['execution_count'],
            }
            for instance_id, data in self.instances.items()
        ]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall edge manager statistics"""
        return {
            **self.stats,
            'active_modules': len(self.modules),
            'active_instances': len(self.instances),
            'queue_size': self.execution_queue.qsize(),
            'average_execution_time_ms': (
                self.stats['total_execution_time_ms'] / self.stats['executions_completed']
                if self.stats['executions_completed'] > 0 else 0
            ),
        }
    
    async def cleanup_expired_instances(self) -> None:
        """Clean up expired instances"""
        logger.info("Cleaning up expired instances")
        
        now = datetime.utcnow()
        expired_instances = []
        
        for instance_id, data in self.instances.items():
            last_used = data['last_used']
            if isinstance(last_used, str):
                last_used = datetime.fromisoformat(last_used)
            
            if now - last_used > timedelta(minutes=30):  # 30 minute timeout
                expired_instances.append(instance_id)
        
        for instance_id in expired_instances:
            await self.destroy_instance(instance_id)
        
        logger.info(f"Cleaned up {len(expired_instances)} expired instances")
    
    async def shutdown(self) -> None:
        """Shutdown the edge manager"""
        logger.info("Shutting down EdgeManager")
        
        self.is_running = False
        
        # Destroy all instances
        instance_ids = list(self.instances.keys())
        for instance_id in instance_ids:
            await self.destroy_instance(instance_id)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        logger.info("EdgeManager shut down")
    
    # Private methods
    
    async def _load_modules_from_db(self) -> None:
        """Load modules from database on startup"""
        try:
            async with self.db_service.get_session() as session:
                result = await session.execute(
                    select(EdgeModule).where(EdgeModule.is_active == True)
                )
                modules = result.scalars().all()
                
                for module in modules:
                    config = EdgeModuleConfig(**module.config)
                    self.modules[config.name] = config
                
                logger.info(f"Loaded {len(modules)} modules from database")
                
        except Exception as e:
            logger.error(f"Failed to load modules from database: {e}")
    
    async def _validate_module_config(self, config: EdgeModuleConfig) -> None:
        """Validate module configuration"""
        if not config.name or not config.wasm_path:
            raise ValueError("Module name and WASM path are required")
        
        if config.memory_initial < 1 or config.memory_maximum < config.memory_initial:
            raise ValueError("Invalid memory configuration")
        
        if config.timeout_ms < 1000 or config.timeout_ms > 300000:  # 1s to 5min
            raise ValueError("Timeout must be between 1s and 5 minutes")
        
        # Check if WASM file exists
        wasm_path = Path(config.wasm_path)
        if not wasm_path.exists():
            raise ValueError(f"WASM file not found: {config.wasm_path}")
    
    async def _execution_worker(self) -> None:
        """Background worker for processing execution queue"""
        logger.info("Starting execution worker")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.execution_queue.get(),
                    timeout=1.0
                )
                
                # Process execution task
                await self._process_execution_task(task)
                
            except asyncio.TimeoutError:
                # Periodic cleanup
                await self.cleanup_expired_instances()
                continue
            except Exception as e:
                logger.error(f"Execution worker error: {e}")
                await asyncio.sleep(1)
    
    async def _process_execution_task(self, task: Dict[str, Any]) -> None:
        """Process a single execution task"""
        # In production, this would interface with the actual WebAssembly runtime
        # For now, we simulate the execution
        pass
    
    async def _simulate_execution(self, task: Dict[str, Any]) -> EdgeExecutionResult:
        """Simulate WebAssembly execution for development"""
        start_time = time.time()
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return EdgeExecutionResult(
            success=True,
            result={"message": f"Function {task['function_name']} executed successfully"},
            execution_time_ms=execution_time,
            memory_used=1024 * 1024,  # 1MB
            cpu_time_ms=execution_time,
        )
    
    async def _record_execution(
        self,
        instance_id: str,
        function_name: str,
        result: EdgeExecutionResult
    ) -> None:
        """Record execution in database"""
        try:
            async with self.db_service.get_session() as session:
                execution = EdgeExecution(
                    instance_id=instance_id,
                    function_name=function_name,
                    success=result.success,
                    execution_time_ms=result.execution_time_ms,
                    memory_used=result.memory_used,
                    cpu_time_ms=result.cpu_time_ms,
                    error_message=result.error,
                    result_data=result.result,
                    executed_at=datetime.utcnow()
                )
                session.add(execution)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to record execution: {e}")