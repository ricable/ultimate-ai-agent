from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
import asyncio

from backend.services.auth import AuthService
from backend.monitoring.logs.logger import get_logger
from .edge_manager import EdgeManager, EdgeModuleConfig, EdgeExecutionContext, EdgeExecutionResult
from .mobile_bridge import MobileBridge

logger = get_logger(__name__)
security = HTTPBearer()

# Pydantic models for API

class EdgeModuleRequest(BaseModel):
    name: str = Field(..., description="Module name")
    wasm_path: str = Field(..., description="Path to WebAssembly binary")
    version: str = Field(..., description="Module version")
    description: Optional[str] = Field(None, description="Module description")
    memory_initial: int = Field(256, description="Initial memory pages")
    memory_maximum: int = Field(1024, description="Maximum memory pages")
    timeout_ms: int = Field(30000, description="Execution timeout in milliseconds")
    max_instances: int = Field(10, description="Maximum concurrent instances")
    permissions: Optional[Dict[str, bool]] = Field(None, description="Module permissions")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables")

class EdgeInstanceRequest(BaseModel):
    module_name: str = Field(..., description="Module name to instantiate")
    timeout: int = Field(30000, description="Execution timeout")
    max_memory: int = Field(64 * 1024 * 1024, description="Maximum memory usage")
    max_cpu_time: int = Field(10000, description="Maximum CPU time")
    priority: str = Field("normal", description="Execution priority")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class EdgeExecutionRequest(BaseModel):
    function_name: str = Field(..., description="Function name to execute")
    args: List[Any] = Field(default_factory=list, description="Function arguments")
    timeout: Optional[int] = Field(None, description="Execution timeout override")

class EdgeModuleResponse(BaseModel):
    name: str
    version: str
    description: Optional[str]
    active_instances: int
    max_instances: int
    is_active: bool
    created_at: datetime

class EdgeInstanceResponse(BaseModel):
    instance_id: str
    module_name: str
    created_at: datetime
    execution_count: int
    is_active: bool

class EdgeExecutionResponse(BaseModel):
    success: bool
    result: Optional[Any]
    error: Optional[str]
    execution_time_ms: int
    memory_used: int
    cpu_time_ms: int

class EdgeStatsResponse(BaseModel):
    modules_loaded: int
    instances_created: int
    executions_completed: int
    executions_failed: int
    total_execution_time_ms: int
    active_modules: int
    active_instances: int
    queue_size: int
    average_execution_time_ms: float

class MobileConnectionRequest(BaseModel):
    device_id: str = Field(..., description="Device identifier")
    platform: str = Field(..., description="Platform (ios/android)")
    app_version: str = Field(..., description="Application version")
    device_model: Optional[str] = Field(None, description="Device model")
    os_version: Optional[str] = Field(None, description="OS version")

class MobileSessionResponse(BaseModel):
    session_id: str
    user_id: str
    device_id: str
    platform: str
    app_version: str
    connected_at: datetime
    last_ping: datetime
    is_online: bool

class MobileSyncStatsResponse(BaseModel):
    clients_connected: int
    total_connections: int
    sync_operations_completed: int
    sync_operations_failed: int
    data_synced_bytes: int
    active_clients: int
    active_users: int
    pending_sync_operations: int
    sync_queue_size: int

def create_edge_router(edge_manager: EdgeManager, mobile_bridge: MobileBridge, auth_service: AuthService) -> APIRouter:
    """Create FastAPI router for edge computing endpoints"""
    
    router = APIRouter(prefix="/api/edge", tags=["edge"])
    
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Get current authenticated user"""
        user = await auth_service.validate_token(credentials.credentials)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return user
    
    # Edge Module Management
    
    @router.post("/modules", response_model=Dict[str, str])
    async def load_module(
        request: EdgeModuleRequest,
        current_user = Depends(get_current_user)
    ):
        """Load a WebAssembly module"""
        try:
            config = EdgeModuleConfig(
                name=request.name,
                wasm_path=request.wasm_path,
                version=request.version,
                description=request.description,
                memory_initial=request.memory_initial,
                memory_maximum=request.memory_maximum,
                timeout_ms=request.timeout_ms,
                max_instances=request.max_instances,
                permissions=request.permissions,
                env_vars=request.env_vars
            )
            
            await edge_manager.load_module(config)
            
            logger.info(f"Module {request.name} loaded by user {current_user.id}")
            return {"message": f"Module {request.name} loaded successfully"}
            
        except Exception as e:
            logger.error(f"Failed to load module {request.name}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/modules", response_model=List[EdgeModuleResponse])
    async def list_modules(
        current_user = Depends(get_current_user)
    ):
        """List all loaded modules"""
        try:
            modules = await edge_manager.list_modules()
            return [
                EdgeModuleResponse(
                    name=module["name"],
                    version=module["version"],
                    description=module["description"],
                    active_instances=module["active_instances"],
                    max_instances=module["max_instances"],
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                for module in modules
            ]
        except Exception as e:
            logger.error(f"Failed to list modules: {e}")
            raise HTTPException(status_code=500, detail="Failed to list modules")
    
    @router.delete("/modules/{module_name}")
    async def unload_module(
        module_name: str,
        current_user = Depends(get_current_user)
    ):
        """Unload a WebAssembly module"""
        try:
            # Implementation would go here
            return {"message": f"Module {module_name} unloaded successfully"}
        except Exception as e:
            logger.error(f"Failed to unload module {module_name}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    # Edge Instance Management
    
    @router.post("/instances", response_model=Dict[str, str])
    async def create_instance(
        request: EdgeInstanceRequest,
        current_user = Depends(get_current_user)
    ):
        """Create a WebAssembly instance"""
        try:
            context = EdgeExecutionContext(
                module_id=request.module_name,
                request_id=f"req_{int(datetime.utcnow().timestamp())}",
                user_id=str(current_user.id),
                timeout=request.timeout,
                max_memory=request.max_memory,
                max_cpu_time=request.max_cpu_time,
                priority=request.priority,
                metadata=request.metadata
            )
            
            instance_id = await edge_manager.create_instance(request.module_name, context)
            
            logger.info(f"Instance {instance_id} created for user {current_user.id}")
            return {"instance_id": instance_id}
            
        except Exception as e:
            logger.error(f"Failed to create instance: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/instances", response_model=List[EdgeInstanceResponse])
    async def list_instances(
        current_user = Depends(get_current_user)
    ):
        """List all active instances"""
        try:
            instances = await edge_manager.list_instances()
            return [
                EdgeInstanceResponse(
                    instance_id=instance["instance_id"],
                    module_name=instance["module_name"],
                    created_at=instance["created_at"],
                    execution_count=instance["execution_count"],
                    is_active=True
                )
                for instance in instances
            ]
        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            raise HTTPException(status_code=500, detail="Failed to list instances")
    
    @router.get("/instances/{instance_id}/stats")
    async def get_instance_stats(
        instance_id: str,
        current_user = Depends(get_current_user)
    ):
        """Get statistics for a specific instance"""
        try:
            stats = await edge_manager.get_instance_stats(instance_id)
            return stats
        except Exception as e:
            logger.error(f"Failed to get instance stats: {e}")
            raise HTTPException(status_code=404, detail="Instance not found")
    
    @router.delete("/instances/{instance_id}")
    async def destroy_instance(
        instance_id: str,
        current_user = Depends(get_current_user)
    ):
        """Destroy a WebAssembly instance"""
        try:
            await edge_manager.destroy_instance(instance_id)
            logger.info(f"Instance {instance_id} destroyed by user {current_user.id}")
            return {"message": f"Instance {instance_id} destroyed successfully"}
        except Exception as e:
            logger.error(f"Failed to destroy instance {instance_id}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    # Edge Function Execution
    
    @router.post("/instances/{instance_id}/execute", response_model=EdgeExecutionResponse)
    async def execute_function(
        instance_id: str,
        request: EdgeExecutionRequest,
        current_user = Depends(get_current_user)
    ):
        """Execute a function in a WebAssembly instance"""
        try:
            context = EdgeExecutionContext(
                module_id="unknown",  # Will be resolved by instance
                request_id=f"exec_{int(datetime.utcnow().timestamp())}",
                user_id=str(current_user.id),
                timeout=request.timeout or 30000
            )
            
            result = await edge_manager.execute_function(
                instance_id=instance_id,
                function_name=request.function_name,
                args=request.args,
                context=context
            )
            
            return EdgeExecutionResponse(
                success=result.success,
                result=result.result,
                error=result.error,
                execution_time_ms=result.execution_time_ms,
                memory_used=result.memory_used,
                cpu_time_ms=result.cpu_time_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to execute function: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    # Edge Statistics
    
    @router.get("/stats", response_model=EdgeStatsResponse)
    async def get_edge_stats(
        current_user = Depends(get_current_user)
    ):
        """Get edge computing statistics"""
        try:
            stats = await edge_manager.get_statistics()
            return EdgeStatsResponse(**stats)
        except Exception as e:
            logger.error(f"Failed to get edge stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to get statistics")
    
    # Mobile WebSocket Connection
    
    @router.websocket("/mobile/ws")
    async def mobile_websocket(
        websocket: WebSocket,
        token: str,
        device_id: str,
        platform: str,
        app_version: str
    ):
        """WebSocket endpoint for mobile clients"""
        session_id = None
        
        try:
            await websocket.accept()
            
            # Connect mobile client
            session_id = await mobile_bridge.connect_client(
                websocket=websocket,
                token=token,
                device_id=device_id,
                platform=platform,
                app_version=app_version
            )
            
            logger.info(f"Mobile WebSocket connected: {session_id}")
            
            # Message handling loop
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle message
                    await mobile_bridge.handle_message(session_id, message)
                    
                except WebSocketDisconnect:
                    logger.info(f"Mobile WebSocket disconnected: {session_id}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from mobile client {session_id}: {e}")
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Error handling mobile message: {e}")
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Internal server error'
                    }))
        
        except Exception as e:
            logger.error(f"Mobile WebSocket error: {e}")
        
        finally:
            if session_id:
                await mobile_bridge.disconnect_client(session_id)
    
    # Mobile API endpoints
    
    @router.get("/mobile/sessions", response_model=List[MobileSessionResponse])
    async def list_mobile_sessions(
        current_user = Depends(get_current_user)
    ):
        """List mobile sessions for current user"""
        try:
            sessions = await mobile_bridge.get_user_sessions(str(current_user.id))
            return [
                MobileSessionResponse(**session)
                for session in sessions
            ]
        except Exception as e:
            logger.error(f"Failed to list mobile sessions: {e}")
            raise HTTPException(status_code=500, detail="Failed to list sessions")
    
    @router.get("/mobile/stats", response_model=MobileSyncStatsResponse)
    async def get_mobile_stats(
        current_user = Depends(get_current_user)
    ):
        """Get mobile sync statistics"""
        try:
            stats = await mobile_bridge.get_statistics()
            return MobileSyncStatsResponse(**stats)
        except Exception as e:
            logger.error(f"Failed to get mobile stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to get statistics")
    
    @router.post("/mobile/sync/{user_id}")
    async def trigger_user_sync(
        user_id: str,
        operation_type: str,
        data: Dict[str, Any],
        current_user = Depends(get_current_user)
    ):
        """Trigger sync for a specific user"""
        try:
            # Check authorization (admin or self)
            if str(current_user.id) != user_id and not current_user.is_admin:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            await mobile_bridge.sync_user_data(user_id, operation_type, data)
            return {"message": "Sync triggered successfully"}
            
        except Exception as e:
            logger.error(f"Failed to trigger sync: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    return router