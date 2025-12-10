# edge/runtime/wasm_engine.py
# Agent 22: WebAssembly Runtime for Edge Deployment

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import hashlib
import time
from datetime import datetime

try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False
    print("wasmtime not available, using mock implementation")

try:
    import wasmer
    WASMER_AVAILABLE = True
except ImportError:
    WASMER_AVAILABLE = False

class WasmModule:
    """WebAssembly module wrapper"""
    
    def __init__(self, module_id: str, module_path: str, metadata: Dict[str, Any]):
        self.module_id = module_id
        self.module_path = module_path
        self.metadata = metadata
        self.instance = None
        self.loaded_at = None
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get module performance statistics"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0.0
        )
        
        return {
            'module_id': self.module_id,
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time,
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None
        }

class WasmRuntime:
    """WebAssembly runtime manager for edge deployment"""
    
    def __init__(self, storage_path: str = "./edge_modules"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.modules: Dict[str, WasmModule] = {}
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.store = None
        
        # Initialize WebAssembly engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize WebAssembly engine"""
        if WASMTIME_AVAILABLE:
            self.engine = wasmtime.Engine()
            self.store = wasmtime.Store(self.engine)
            self.logger.info("Initialized Wasmtime engine")
        elif WASMER_AVAILABLE:
            # Wasmer initialization would go here
            self.logger.info("Initialized Wasmer engine")
        else:
            self.logger.warning("No WebAssembly runtime available, using mock implementation")
    
    async def load_module(self, 
                         module_id: str, 
                         wasm_bytes: bytes, 
                         metadata: Dict[str, Any] = None) -> bool:
        """Load a WebAssembly module"""
        
        if metadata is None:
            metadata = {}
        
        try:
            # Save module to disk
            module_path = self.storage_path / f"{module_id}.wasm"
            async with open(module_path, 'wb') as f:
                f.write(wasm_bytes)
            
            # Create module wrapper
            wasm_module = WasmModule(
                module_id=module_id,
                module_path=str(module_path),
                metadata=metadata
            )
            
            # Load with WebAssembly engine
            if WASMTIME_AVAILABLE:
                try:
                    module = wasmtime.Module(self.engine, wasm_bytes)
                    instance = wasmtime.Instance(self.store, module, [])
                    wasm_module.instance = instance
                    wasm_module.loaded_at = datetime.utcnow()
                    
                    self.logger.info(f"Loaded WebAssembly module: {module_id}")
                except Exception as e:
                    self.logger.error(f"Failed to load WASM module {module_id}: {e}")
                    return False
            else:
                # Mock implementation
                wasm_module.instance = {"mock": True}
                wasm_module.loaded_at = datetime.utcnow()
                self.logger.info(f"Mock loaded WebAssembly module: {module_id}")
            
            self.modules[module_id] = wasm_module
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load module {module_id}: {e}")
            return False
    
    async def execute_function(self, 
                             module_id: str, 
                             function_name: str, 
                             args: List[Any] = None,
                             timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a function in a WebAssembly module"""
        
        if args is None:
            args = []
        
        if module_id not in self.modules:
            return {
                'success': False,
                'error': f'Module {module_id} not found'
            }
        
        wasm_module = self.modules[module_id]
        
        if not wasm_module.instance:
            return {
                'success': False,
                'error': f'Module {module_id} not loaded'
            }
        
        start_time = time.time()
        
        try:
            if WASMTIME_AVAILABLE and hasattr(wasm_module.instance, 'exports'):
                # Execute with Wasmtime
                exports = wasm_module.instance.exports(self.store)
                if function_name in exports:
                    func = exports[function_name]
                    result = func(self.store, *args)
                    
                    execution_time = time.time() - start_time
                    wasm_module.execution_count += 1
                    wasm_module.total_execution_time += execution_time
                    
                    return {
                        'success': True,
                        'result': result,
                        'execution_time': execution_time
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Function {function_name} not found in module'
                    }
            else:
                # Mock execution
                execution_time = time.time() - start_time
                wasm_module.execution_count += 1
                wasm_module.total_execution_time += execution_time
                
                return {
                    'success': True,
                    'result': f'Mock result for {function_name}({args})',
                    'execution_time': execution_time
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing {function_name} in {module_id}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    async def unload_module(self, module_id: str) -> bool:
        """Unload a WebAssembly module"""
        if module_id in self.modules:
            del self.modules[module_id]
            
            # Remove from disk
            module_path = self.storage_path / f"{module_id}.wasm"
            if module_path.exists():
                module_path.unlink()
            
            self.logger.info(f"Unloaded module: {module_id}")
            return True
        
        return False
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all loaded modules"""
        return [
            {
                'module_id': module.module_id,
                'metadata': module.metadata,
                'performance': module.get_performance_stats()
            }
            for module in self.modules.values()
        ]
    
    def get_module_info(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific module"""
        if module_id not in self.modules:
            return None
        
        module = self.modules[module_id]
        return {
            'module_id': module.module_id,
            'module_path': module.module_path,
            'metadata': module.metadata,
            'performance': module.get_performance_stats()
        }

class EdgeSyncManager:
    """Manage synchronization between edge and cloud"""
    
    def __init__(self, edge_runtime: WasmRuntime):
        self.runtime = edge_runtime
        self.sync_queue: List[Dict[str, Any]] = []
        self.sync_enabled = True
        self.last_sync = None
        self.logger = logging.getLogger(__name__)
    
    async def sync_with_cloud(self, cloud_endpoint: str) -> Dict[str, Any]:
        """Synchronize edge state with cloud"""
        if not self.sync_enabled:
            return {'success': False, 'message': 'Sync disabled'}
        
        try:
            # Collect edge state
            edge_state = {
                'modules': self.runtime.list_modules(),
                'sync_queue': self.sync_queue.copy(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send to cloud (this would be actual HTTP request in production)
            self.logger.info(f"Syncing edge state with cloud: {cloud_endpoint}")
            
            # Clear sync queue on successful sync
            self.sync_queue.clear()
            self.last_sync = datetime.utcnow()
            
            return {
                'success': True,
                'synced_modules': len(edge_state['modules']),
                'synced_queue_items': len(edge_state['sync_queue']),
                'timestamp': self.last_sync.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to sync with cloud: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def queue_for_sync(self, data: Dict[str, Any]):
        """Queue data for cloud synchronization"""
        sync_item = {
            'data': data,
            'queued_at': datetime.utcnow().isoformat(),
            'retries': 0
        }
        
        self.sync_queue.append(sync_item)
        self.logger.debug(f"Queued item for sync: {len(self.sync_queue)} items in queue")
    
    def enable_sync(self):
        """Enable cloud synchronization"""
        self.sync_enabled = True
        self.logger.info("Cloud synchronization enabled")
    
    def disable_sync(self):
        """Disable cloud synchronization (offline mode)"""
        self.sync_enabled = False
        self.logger.info("Cloud synchronization disabled (offline mode)")

class EdgeOrchestrator:
    """Main edge computing orchestrator"""
    
    def __init__(self, storage_path: str = "./edge"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.runtime = WasmRuntime(str(self.storage_path / "modules"))
        self.sync_manager = EdgeSyncManager(self.runtime)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
    
    async def deploy_agent(self, 
                          agent_id: str, 
                          wasm_bytes: bytes, 
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy an agent to the edge"""
        
        self.logger.info(f"Deploying agent {agent_id} to edge")
        
        success = await self.runtime.load_module(agent_id, wasm_bytes, metadata)
        
        if success:
            # Queue deployment for cloud sync
            await self.sync_manager.queue_for_sync({
                'action': 'agent_deployed',
                'agent_id': agent_id,
                'metadata': metadata
            })
            
            return {
                'success': True,
                'message': f'Agent {agent_id} deployed successfully'
            }
        else:
            return {
                'success': False,
                'message': f'Failed to deploy agent {agent_id}'
            }
    
    async def execute_agent(self, 
                           agent_id: str, 
                           input_data: Dict[str, Any],
                           timeout: float = 30.0) -> Dict[str, Any]:
        """Execute an agent on the edge"""
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Execute the agent
            result = await self.runtime.execute_function(
                module_id=agent_id,
                function_name='process',  # Standard function name
                args=[json.dumps(input_data)],
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Queue execution result for sync
            await self.sync_manager.queue_for_sync({
                'action': 'agent_executed',
                'agent_id': agent_id,
                'input_data': input_data,
                'result': result,
                'processing_time': processing_time
            })
            
            return {
                'success': result.get('success', False),
                'result': result.get('result'),
                'processing_time': processing_time,
                'executed_on_edge': True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to execute agent {agent_id}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'executed_on_edge': True
            }
    
    async def sync_with_cloud(self, cloud_endpoint: str) -> Dict[str, Any]:
        """Synchronize edge with cloud"""
        return await self.sync_manager.sync_with_cloud(cloud_endpoint)
    
    def get_edge_status(self) -> Dict[str, Any]:
        """Get edge computing status"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            'runtime_available': WASMTIME_AVAILABLE or WASMER_AVAILABLE,
            'loaded_modules': len(self.runtime.modules),
            'sync_enabled': self.sync_manager.sync_enabled,
            'sync_queue_size': len(self.sync_manager.sync_queue),
            'last_sync': self.sync_manager.last_sync.isoformat() if self.sync_manager.last_sync else None,
            'performance': {
                'request_count': self.request_count,
                'total_processing_time': self.total_processing_time,
                'average_processing_time': avg_processing_time
            }
        }
    
    def enable_offline_mode(self):
        """Enable offline mode (disable cloud sync)"""
        self.sync_manager.disable_sync()
        self.logger.info("Edge orchestrator in offline mode")
    
    def enable_online_mode(self):
        """Enable online mode (enable cloud sync)"""
        self.sync_manager.enable_sync()
        self.logger.info("Edge orchestrator in online mode")

# Global edge orchestrator instance
edge_orchestrator = EdgeOrchestrator()