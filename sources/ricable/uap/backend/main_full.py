# File: backend/main.py
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, WebSocketDisconnect, status, Query, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import json
import os
import asyncio
from datetime import datetime, timezone

from .services.agent_orchestrator import UAP_AgentOrchestrator
from .services.enhanced_orchestrator import EnhancedAgentOrchestrator
from .services.tenant_manager import enhanced_tenant_manager
from .middleware.tenant_middleware import EnhancedTenantMiddleware
from .services.auth import (
    auth_service, User, UserCreate, UserLogin, Token,
    UserInDB, TokenData, get_current_active_user, require_permission
)
from .integrations.manager import IntegrationManager
from .integrations.registry import IntegrationRegistry
from .integrations.oauth_provider import OAuth2Provider
from .integrations.slack_integration import SlackIntegration
from .integrations.teams_integration import TeamsIntegration
from .integrations.notion_integration import NotionIntegration
from .integrations.github_integration import GitHubIntegration
from .webhooks.receiver import WebhookReceiver
from .services.distributed_orchestrator import (
    initialize_distributed_orchestrator, submit_distributed_workload,
    get_workload_status, get_distributed_status, WorkloadType, ProcessingStrategy
)
from .services.performance_service import performance_service
from .processors.document_service import DocumentService
from .database.service import get_database_service, DatabaseService
from .database.connection import init_database, close_database
from .database.migrations.manager import get_migration_manager
from .monitoring.logs.middleware import LoggingMiddleware, SecurityLoggingMixin
from .monitoring.logs.logger import uap_logger, EventType, LogLevel, set_request_context
from .monitoring.metrics.performance import performance_monitor, start_agent_request, finish_agent_request
from .monitoring.alerting.alerts import alert_manager, start_alerting, stop_alerting
from .monitoring.dashboard.api import router as monitoring_router
from .tenancy.middleware import TenancyMiddleware, WhiteLabelMiddleware, TenantRateLimitMiddleware, TenantSecurityMiddleware
from .tenancy.admin_api import router as admin_router
from .billing.billing_api import router as billing_router
from .billing.webhook_handlers import router as webhook_router
from .config.cdn_config import CDNMiddleware, cdn_manager
from .config.performance_config import performance_config
from .api_routes.analytics import router as analytics_router
from .services.analytics_service import analytics_service
from .services.metrics_collector import metrics_collector
from .api_routes.robotics import router as robotics_router
from .api_routes.collaboration import router as collaboration_router
from .workflows.api import router as workflow_router, initialize_workflow_system
from .workflows.execution_engine import WorkflowExecutionEngine
from .workflows.scheduler import WorkflowScheduler
from .workflows.triggers import TriggerManager
from .workflows.marketplace import WorkflowMarketplace
from .plugins.plugin_manager import PluginManager
from .api_routes.marketplace import create_marketplace_routes

# JWT Security
security = HTTPBearer()

# Auth dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInDB:
    """Get current authenticated user from JWT token"""
    return auth_service.get_current_user_from_token(credentials.credentials)

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Optional auth for some endpoints
async def get_optional_user(authorization: Optional[str] = Query(None)) -> Optional[UserInDB]:
    """Get user if token provided, None otherwise"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization.replace("Bearer ", "")
        return auth_service.get_current_user_from_token(token)
    except HTTPException:
        return None

# Permission checker
def require_permission(permission: str):
    """Dependency to require specific permission"""
    def permission_checker(current_user: UserInDB = Depends(get_current_active_user)):
        auth_service.require_permission(current_user, permission)
        return current_user
    return permission_checker

class AgentRequest(BaseModel):
    message: str
    framework: Optional[str] = 'auto'
    context: Optional[Dict[str, Any]] = {}
    stream: Optional[bool] = False

class AgentResponse(BaseModel):
    message: str
    agent_id: str
    framework: str
    timestamp: datetime
    metadata: Dict[str, Any]

class DocumentAnalysisRequest(BaseModel):
    content: Optional[str] = None
    document_type: Optional[str] = "text"
    analysis_type: Optional[str] = "general"  # summary, extraction, reasoning, general
    message: Optional[str] = None

class DocumentAnalysisResponse(BaseModel):
    analysis: str
    document_type: str
    analysis_type: str
    metadata: Dict[str, Any]
    timestamp: datetime

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    filename: str
    file_size: int
    content_type: str
    upload_timestamp: str
    processing_status: str
    error_message: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    has_more: bool

class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    processing: bool
    error_message: Optional[str] = None

class DistributedWorkloadRequest(BaseModel):
    workload_type: str  # document_processing, ai_inference, batch_analysis, etc.
    input_data: Dict[str, Any]
    strategy: Optional[str] = "adaptive"  # sequential, parallel, map_reduce, pipeline, adaptive
    priority: Optional[int] = 0
    metadata: Optional[Dict[str, Any]] = {}

class DistributedWorkloadResponse(BaseModel):
    workload_id: str
    status: str
    message: str
    submitted_at: datetime

class WorkloadStatusResponse(BaseModel):
    workload_id: str
    workload_type: str
    status: str
    strategy: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    task_count: int
    metadata: Dict[str, Any]

# Global orchestrator instances
base_orchestrator = UAP_AgentOrchestrator()
orchestrator = None  # Will be initialized as EnhancedAgentOrchestrator

# Global document service instance
document_service = DocumentService()

# Global database service instance
database_service = get_database_service()

# Global distributed orchestrator instance (initialized after agent orchestrator)
distributed_orchestrator = None

# Global integration manager and components
integration_manager = None
webhook_receiver = None
oauth2_provider = None

# Global plugin manager
plugin_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, distributed_orchestrator, integration_manager, webhook_receiver, oauth2_provider, plugin_manager
    
    # Startup
    uap_logger.log_system_event("Starting UAP Backend API", "main")
    
    # Initialize database first
    try:
        await init_database()
        await database_service.initialize()
        uap_logger.log_system_event("Database service initialized", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Database initialization failed: {e}", "main", LogLevel.ERROR)
        print(f"Warning: Database initialization failed: {e}")
    
    # Initialize base agent orchestrator
    await base_orchestrator.initialize_services()
    
    # Initialize enhanced orchestrator
    orchestrator = EnhancedAgentOrchestrator(base_orchestrator)
    await orchestrator.start_background_tasks()
    uap_logger.log_system_event("Enhanced orchestrator initialized", "main")
    
    # Initialize enhanced tenant manager
    await enhanced_tenant_manager.start_background_tasks() if hasattr(enhanced_tenant_manager, 'start_background_tasks') else None
    uap_logger.log_system_event("Enhanced tenant manager initialized", "main")
    
    # Initialize document service
    await document_service.initialize()
    
    # Initialize integration system
    try:
        # Create integration manager
        integration_manager = IntegrationManager(auth_service)
        
        # Create OAuth2 provider
        oauth2_provider = OAuth2Provider(auth_service)
        
        # Create webhook receiver
        webhook_receiver = WebhookReceiver(integration_manager)
        await webhook_receiver.start_workers()
        
        # Register default integrations with their configurations
        registry = IntegrationRegistry()
        
        # Register integration implementations
        registry.register_integration_class("slack", SlackIntegration)
        registry.register_integration_class("microsoft_teams", TeamsIntegration)
        registry.register_integration_class("notion", NotionIntegration)
        registry.register_integration_class("github", GitHubIntegration)
        
        # Initialize integration manager
        await integration_manager.initialize()
        
        uap_logger.log_system_event("Integration system initialized successfully", "main")
        
    except Exception as e:
        uap_logger.log_system_event(f"Failed to initialize integration system: {str(e)}", "main")
        # Continue startup - integrations are optional
    
    # Initialize plugin system
    try:
        # Create plugin manager
        plugins_directory = os.getenv("PLUGINS_DIRECTORY", "plugins")
        plugin_manager = PluginManager(plugins_directory, auth_service)
        
        # Initialize plugin manager
        await plugin_manager.initialize()
        
        uap_logger.log_system_event("Plugin system initialized successfully", "main")
        
    except Exception as e:
        uap_logger.log_system_event(f"Failed to initialize plugin system: {str(e)}", "main")
        # Continue startup - plugins are optional
    
    # Initialize performance service (caching, optimization)
    await performance_service.initialize()
    uap_logger.log_system_event("Performance service initialized", "main")
    
    # Initialize distributed orchestrator
    distributed_orchestrator = initialize_distributed_orchestrator(orchestrator)
    
    # Link distributed orchestrator back to enhanced orchestrator
    if hasattr(orchestrator, 'base_orchestrator'):
        orchestrator.base_orchestrator.set_distributed_orchestrator(distributed_orchestrator)
    else:
        orchestrator.set_distributed_orchestrator(distributed_orchestrator)
    
    uap_logger.log_system_event("Distributed orchestrator initialized and linked", "main")
    
    # Initialize workflow automation system
    try:
        workflow_execution_engine = WorkflowExecutionEngine(orchestrator, integration_manager)
        workflow_scheduler = WorkflowScheduler(workflow_execution_engine)
        workflow_trigger_manager = TriggerManager(workflow_execution_engine)
        workflow_marketplace = WorkflowMarketplace()
        
        # Initialize workflow system components
        initialize_workflow_system(
            workflow_execution_engine,
            workflow_scheduler,
            workflow_trigger_manager,
            workflow_marketplace
        )
        
        # Start workflow scheduler
        await workflow_scheduler.start()
        
        uap_logger.log_system_event("Workflow automation system initialized", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Failed to initialize workflow system: {str(e)}", "main", LogLevel.ERROR)
        # Continue startup - workflows are optional
    
    # Start monitoring systems
    await start_alerting()
    asyncio.create_task(performance_monitor.start_background_monitoring())
    
    # Start analytics service and metrics collection
    try:
        await analytics_service.start_analytics_collection()
        await metrics_collector.start_collection()
        uap_logger.log_system_event("Analytics service and metrics collection started", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Analytics startup error: {e}", "main", LogLevel.ERROR)
    
    uap_logger.log_system_event("UAP Backend API started successfully", "main")
    print("UAP Backend API started successfully")
    
    yield
    
    # Shutdown
    uap_logger.log_system_event("Shutting down UAP Backend API", "main")
    
    # Cleanup enhanced orchestrator
    if orchestrator:
        await orchestrator.cleanup()
        uap_logger.log_system_event("Enhanced orchestrator shutdown", "main")
    
    # Cleanup enhanced tenant manager
    await enhanced_tenant_manager.cleanup()
    uap_logger.log_system_event("Enhanced tenant manager shutdown", "main")
    
    # Cleanup performance service
    await performance_service.cleanup()
    uap_logger.log_system_event("Performance service shutdown", "main")
    
    # Cleanup integration system
    if integration_manager:
        await integration_manager.cleanup()
        uap_logger.log_system_event("Integration system shutdown", "main")
    
    # Cleanup plugin system
    if plugin_manager:
        await plugin_manager.cleanup()
        uap_logger.log_system_event("Plugin system shutdown", "main")
    
    # Cleanup distributed orchestrator
    if distributed_orchestrator:
        await distributed_orchestrator.cleanup()
        uap_logger.log_system_event("Distributed orchestrator shutdown", "main")
    
    # Stop monitoring systems
    await stop_alerting()
    performance_monitor.stop_monitoring()
    
    # Stop analytics service and metrics collection
    try:
        await analytics_service.stop_collection() if hasattr(analytics_service, 'stop_collection') else None
        await metrics_collector.stop_collection()
        uap_logger.log_system_event("Analytics service and metrics collection stopped", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Analytics shutdown error: {e}", "main", LogLevel.ERROR)
    
    # Shutdown workflow system
    try:
        if 'workflow_scheduler' in locals():
            await workflow_scheduler.stop()
        uap_logger.log_system_event("Workflow system shutdown", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Workflow shutdown error: {e}", "main", LogLevel.ERROR)
    
    # Shutdown database service
    try:
        await database_service.shutdown()
        await close_database()
        uap_logger.log_system_event("Database service shutdown", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Database shutdown error: {e}", "main", LogLevel.ERROR)
    
    print("UAP Backend API shutting down")

app = FastAPI(
    title="UAP Backend API",
    version="3.0.0",
    description="Unified Agentic Platform Backend API",
    lifespan=lifespan
)

# Add performance middleware
if performance_config.cdn.enabled:
    app.add_middleware(CDNMiddleware, cdn_manager=cdn_manager)

# Add enhanced tenant middleware (order matters)
app.add_middleware(EnhancedTenantMiddleware)
app.add_middleware(TenantSecurityMiddleware)
app.add_middleware(TenantRateLimitMiddleware)
app.add_middleware(WhiteLabelMiddleware)
app.add_middleware(TenancyMiddleware, enable_tenant_routing=True)

# Add monitoring middleware
app.add_middleware(LoggingMiddleware, log_request_body=False, log_response_body=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(monitoring_router)
app.include_router(admin_router)
app.include_router(billing_router)
app.include_router(webhook_router)
app.include_router(analytics_router)
app.include_router(robotics_router)
app.include_router(collaboration_router)
app.include_router(workflow_router)

# Add marketplace router when managers are available
if plugin_manager and integration_manager:
    try:
        marketplace_router = create_marketplace_routes(plugin_manager, integration_manager)
        app.include_router(marketplace_router)
        uap_logger.log_system_event("Marketplace router added successfully", "main")
    except Exception as e:
        uap_logger.log_system_event(f"Failed to add marketplace router: {str(e)}", "main", LogLevel.ERROR)

# Marketplace router is now added above with both managers

@app.websocket("/ws/agents/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str, token: Optional[str] = Query(None)):
    """WebSocket endpoint for real-time agent communication via AG-UI protocol."""
    # Authenticate WebSocket connection
    user = None
    if token:
        try:
            user = auth_service.get_current_user_from_token(token)
            # Check websocket permission
            auth_service.require_permission(user, "websocket:connect")
        except HTTPException as e:
            await websocket.close(code=1008, reason=f"Authentication failed: {e.detail}")
            return
    
    await websocket.accept()
    connection_id = f"{agent_id}_{datetime.now(timezone.utc).timestamp()}"
    await orchestrator.register_connection(connection_id, websocket, user)
    
    try:
        while True:
            data = await websocket.receive_text()
            event = json.loads(data)
            # Add user context to event
            if user:
                event["user_context"] = {
                    "user_id": user.id,
                    "username": user.username,
                    "roles": user.roles
                }
            # Route AG-UI events to the enhanced orchestrator for processing
            await orchestrator.handle_agui_event(connection_id, event)
    except WebSocketDisconnect:
        await orchestrator.unregister_connection(connection_id)
        print(f"WebSocket {connection_id} disconnected.")
    except Exception as e:
        print(f"WebSocket error for {connection_id}: {e}")
        await orchestrator.unregister_connection(connection_id)

@app.websocket("/ws/analytics")
async def analytics_websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
    """WebSocket endpoint for real-time analytics dashboard updates."""
    # Authenticate WebSocket connection
    user = None
    if token:
        try:
            user = auth_service.get_current_user_from_token(token)
            # Check analytics permission
            auth_service.require_permission(user, "analytics:read")
        except HTTPException as e:
            await websocket.close(code=1008, reason=f"Authentication failed: {e.detail}")
            return
    
    await websocket.accept()
    connection_id = f"analytics_{datetime.now(timezone.utc).timestamp()}"
    
    # Store connection for broadcasting updates
    if not hasattr(app.state, 'analytics_connections'):
        app.state.analytics_connections = {}
    app.state.analytics_connections[connection_id] = websocket
    
    try:
        # Send initial dashboard data
        dashboard_data = await analytics_service.get_dashboard_data(user.id if user else None)
        await websocket.send_text(json.dumps({
            "type": "dashboard_update",
            "payload": dashboard_data.to_dict()
        }, default=str))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for ping or close
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif data.get("type") == "request_update":
                    # Send updated dashboard data
                    dashboard_data = await analytics_service.get_dashboard_data(user.id if user else None)
                    await websocket.send_text(json.dumps({
                        "type": "dashboard_update",
                        "payload": dashboard_data.to_dict()
                    }, default=str))
                    
            except asyncio.TimeoutError:
                # Send periodic updates every 30 seconds
                dashboard_data = await analytics_service.get_dashboard_data(user.id if user else None)
                await websocket.send_text(json.dumps({
                    "type": "dashboard_update",
                    "payload": dashboard_data.to_dict()
                }, default=str))
                
    except WebSocketDisconnect:
        if connection_id in app.state.analytics_connections:
            del app.state.analytics_connections[connection_id]
        print(f"Analytics WebSocket {connection_id} disconnected.")
    except Exception as e:
        print(f"Analytics WebSocket error for {connection_id}: {e}")
        if connection_id in app.state.analytics_connections:
            del app.state.analytics_connections[connection_id]

@app.websocket("/ws/analytics/metrics")
async def metrics_websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
    """WebSocket endpoint for real-time metrics updates."""
    # Authenticate WebSocket connection
    user = None
    if token:
        try:
            user = auth_service.get_current_user_from_token(token)
            # Check analytics permission
            auth_service.require_permission(user, "analytics:read")
        except HTTPException as e:
            await websocket.close(code=1008, reason=f"Authentication failed: {e.detail}")
            return
    
    await websocket.accept()
    connection_id = f"metrics_{datetime.now(timezone.utc).timestamp()}"
    
    # Store connection for broadcasting metric updates
    if not hasattr(app.state, 'metrics_connections'):
        app.state.metrics_connections = {}
    app.state.metrics_connections[connection_id] = websocket
    
    # Subscribe to real-time metrics
    def metric_callback(metric):
        """Callback function for metric updates"""
        try:
            # Send metric update to this WebSocket connection
            asyncio.create_task(websocket.send_text(json.dumps({
                "type": "metric_update",
                "payload": {
                    "metric_name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                    "unit": metric.unit,
                    "category": metric.category,
                    "labels": metric.labels
                }
            }, default=str)))
        except Exception as e:
            print(f"Error sending metric update: {e}")
    
    # Subscribe to all metrics (or specific ones based on user preferences)
    for metric_name in ['system_cpu_percent', 'system_memory_percent', 'agent_response_time', 'active_connections']:
        analytics_service.subscribe_to_metric(metric_name, metric_callback)
    
    try:
        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif data.get("type") == "subscribe":
                    # Subscribe to specific metrics
                    metric_names = data.get("metrics", [])
                    for metric_name in metric_names:
                        analytics_service.subscribe_to_metric(metric_name, metric_callback)
                        
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))
                
    except WebSocketDisconnect:
        if connection_id in app.state.metrics_connections:
            del app.state.metrics_connections[connection_id]
        print(f"Metrics WebSocket {connection_id} disconnected.")
    except Exception as e:
        print(f"Metrics WebSocket error for {connection_id}: {e}")
        if connection_id in app.state.metrics_connections:
            del app.state.metrics_connections[connection_id]

@app.post("/api/agents/{agent_id}/chat", response_model=AgentResponse)
async def chat_with_agent(
    agent_id: str, 
    request: AgentRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """HTTP endpoint for stateless agent interactions."""
    # Check agent access permission
    auth_service.require_permission(current_user, "agent:read")
    
    try:
        # Add user context to request
        user_context = {
            "user_id": current_user.id,
            "username": current_user.username,
            "roles": current_user.roles
        }
        context = {**request.context, "user_context": user_context}
        
        response_data = await orchestrator.handle_http_chat(
            agent_id,
            request.message,
            request.framework,
            context
        )
        return AgentResponse(
            message=response_data.get("content", ""),
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc),
            framework=response_data.get("framework", "unknown"),
            metadata=response_data.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get status of all agents and frameworks."""
    return await orchestrator.get_enhanced_system_status()

# Document processing endpoints
@app.post("/api/documents/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document_content(
    request: DocumentAnalysisRequest,
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Analyze document content using the Agno framework."""
    try:
        if not request.content and not request.message:
            raise HTTPException(
                status_code=400, 
                detail="Either 'content' or 'message' must be provided"
            )
        
        # Use content if provided, otherwise use message
        document_content = request.content or ""
        analysis_message = request.message or f"Please analyze this {request.document_type} document."
        
        # Create context for document processing
        context = {
            "document_content": document_content,
            "document_type": request.document_type,
            "analysis_type": request.analysis_type,
            "user_id": current_user.id,
            "username": current_user.username
        }
        
        # Process with Agno framework specifically
        response_data = await orchestrator.agno_manager.process_document(
            document_content=document_content,
            document_type=request.document_type,
            analysis_type=request.analysis_type
        )
        
        return DocumentAnalysisResponse(
            analysis=response_data.get("content", ""),
            document_type=request.document_type,
            analysis_type=request.analysis_type,
            metadata=response_data.get("metadata", {}),
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    process_immediately: bool = Query(True, description="Whether to start processing immediately"),
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Upload a document file for processing."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Read file content
        content = await file.read()
        
        # Upload and optionally process document
        result = await document_service.upload_document(
            file_data=content,
            filename=file.filename,
            content_type=file.content_type,
            process_immediately=process_immediately
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return DocumentUploadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/api/documents/process-sync")
async def process_document_sync(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Upload and process a document synchronously."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        content = await file.read()
        
        # Process document synchronously
        result = await document_service.process_document_sync(
            file_data=content,
            filename=file.filename,
            content_type=file.content_type
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by status: completed, processing, error"),
    current_user: UserInDB = Depends(require_permission("agent:read"))
):
    """List documents with pagination and filtering."""
    try:
        result = await document_service.list_documents(
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )
        
        return DocumentListResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/api/documents/{doc_id}")
async def get_document(
    doc_id: str,
    include_content: bool = Query(False, description="Whether to include full document content"),
    current_user: UserInDB = Depends(require_permission("agent:read"))
):
    """Retrieve a specific document by ID."""
    try:
        document = await document_service.get_document(doc_id, include_content=include_content)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")

@app.get("/api/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    current_user: UserInDB = Depends(require_permission("agent:read"))
):
    """Get the processing status of a document."""
    try:
        status_info = await document_service.get_processing_status(doc_id)
        
        return DocumentStatusResponse(
            document_id=doc_id,
            **status_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")

@app.post("/api/documents/{doc_id}/analyze")
async def analyze_document_with_agno(
    doc_id: str,
    analysis_type: str = Query("general", description="Type of analysis to perform"),
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Analyze a processed document using the Agno framework."""
    try:
        result = await document_service.analyze_document_with_agno(doc_id, analysis_type)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: UserInDB = Depends(require_permission("agent:delete"))
):
    """Delete a document and its associated data."""
    try:
        result = await document_service.delete_document(doc_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/api/documents/service/status")
async def get_document_service_status(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get the current status of the document processing service."""
    try:
        return await document_service.get_service_status()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service status: {str(e)}")

@app.post("/api/workflows/{workflow_name}/execute")
async def execute_workflow_endpoint(
    workflow_name: str,
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    # This would execute a Mastra workflow via the orchestrator
    return {"status": "ok", "message": f"Workflow '{workflow_name}' execution placeholder.", "user": current_user.username}

# Authentication endpoints
@app.post("/api/auth/register", response_model=User)
async def register(
    user_data: UserCreate
):
    """Register a new user"""
    return auth_service.create_user(user_data)

@app.post("/api/auth/login", response_model=Dict[str, Any])
async def login(
    login_data: UserLogin
):
    """Login user and return tokens"""
    try:
        tokens, user = auth_service.login(login_data)
        
        # Log successful authentication
        SecurityLoggingMixin.log_authentication_attempt(
            user_id=user.username,
            success=True,
            method="password"
        )
        
        return {
            "tokens": tokens,
            "user": user,
            "message": "Login successful"
        }
    except HTTPException as e:
        # Log failed authentication
        SecurityLoggingMixin.log_authentication_attempt(
            user_id=login_data.username,
            success=False,
            method="password"
        )
        raise

@app.post("/api/auth/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str
):
    """Refresh access token"""
    return auth_service.refresh_access_token(refresh_token)

@app.post("/api/auth/logout")
async def logout(
    refresh_token: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Logout user"""
    auth_service.logout(refresh_token)
    return {"message": "Logout successful"}

@app.get("/api/auth/me", response_model=User)
async def get_current_user_info(
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Get current user information"""
    return User(**current_user.dict(exclude={"hashed_password"}))

@app.get("/api/auth/roles")
async def get_roles(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get all available roles"""
    return auth_service.get_all_roles()

# Distributed processing endpoints
@app.post("/api/distributed/workloads", response_model=DistributedWorkloadResponse)
async def submit_workload(
    request: DistributedWorkloadRequest,
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Submit a distributed workload for processing."""
    try:
        # Validate workload type
        try:
            workload_type = WorkloadType(request.workload_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid workload type. Supported types: {[wt.value for wt in WorkloadType]}"
            )
        
        # Validate processing strategy
        try:
            strategy = ProcessingStrategy(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Supported strategies: {[ps.value for ps in ProcessingStrategy]}"
            )
        
        # Add user context to metadata
        metadata = request.metadata or {}
        metadata.update({
            "user_id": current_user.id,
            "username": current_user.username,
            "submitted_at": datetime.now(timezone.utc).isoformat()
        })
        
        # Submit workload
        workload_id = await submit_distributed_workload(
            workload_type=workload_type,
            input_data=request.input_data,
            strategy=strategy
        )
        
        return DistributedWorkloadResponse(
            workload_id=workload_id,
            status="submitted",
            message=f"Distributed workload {workload_type.value} submitted successfully",
            submitted_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit workload: {str(e)}")

@app.get("/api/distributed/workloads/{workload_id}", response_model=WorkloadStatusResponse)
async def get_workload_status_endpoint(
    workload_id: str,
    current_user: UserInDB = Depends(require_permission("agent:read"))
):
    """Get the status of a specific distributed workload."""
    try:
        workload = await get_workload_status(workload_id)
        
        if not workload:
            raise HTTPException(status_code=404, detail="Workload not found")
        
        return WorkloadStatusResponse(
            workload_id=workload.workload_id,
            workload_type=workload.workload_type.value,
            status=workload.status,
            strategy=workload.strategy.value,
            progress=workload.progress,
            created_at=workload.created_at,
            started_at=workload.started_at,
            completed_at=workload.completed_at,
            result=workload.result,
            error=workload.error,
            task_count=len(workload.tasks),
            metadata=workload.metadata or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workload status: {str(e)}")

@app.delete("/api/distributed/workloads/{workload_id}")
async def cancel_workload(
    workload_id: str,
    current_user: UserInDB = Depends(require_permission("agent:delete"))
):
    """Cancel a distributed workload."""
    try:
        if not distributed_orchestrator:
            raise HTTPException(status_code=503, detail="Distributed orchestrator not available")
        
        success = await distributed_orchestrator.cancel_workload(workload_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workload not found or cannot be cancelled")
        
        return {
            "success": True,
            "message": f"Workload {workload_id} cancelled successfully",
            "cancelled_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel workload: {str(e)}")

@app.get("/api/distributed/status")
async def get_distributed_system_status(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get comprehensive status of the distributed processing system."""
    try:
        return await get_distributed_status()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get distributed system status: {str(e)}")

@app.post("/api/distributed/workloads/document-batch")
async def submit_document_batch_processing(
    documents: List[str] = Query(..., description="List of document IDs or file paths"),
    analysis_type: str = Query("general", description="Type of analysis to perform"),
    strategy: str = Query("adaptive", description="Processing strategy"),
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Submit a batch of documents for distributed processing."""
    try:
        # Prepare document processing workload
        input_data = {
            "documents": [{"document_id": doc_id} for doc_id in documents],
            "analysis_type": analysis_type
        }
        
        workload_id = await submit_distributed_workload(
            workload_type=WorkloadType.DOCUMENT_PROCESSING,
            input_data=input_data,
            strategy=ProcessingStrategy(strategy)
        )
        
        return DistributedWorkloadResponse(
            workload_id=workload_id,
            status="submitted",
            message=f"Batch document processing workload submitted with {len(documents)} documents",
            submitted_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit document batch: {str(e)}")

@app.post("/api/distributed/workloads/ai-inference-batch")
async def submit_ai_inference_batch(
    queries: List[str] = Query(..., description="List of queries for AI inference"),
    framework: str = Query("copilot", description="AI framework to use"),
    strategy: str = Query("parallel", description="Processing strategy"),
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Submit a batch of queries for distributed AI inference."""
    try:
        # Prepare AI inference workload
        input_data = {
            "queries": queries,
            "framework": framework
        }
        
        workload_id = await submit_distributed_workload(
            workload_type=WorkloadType.AI_INFERENCE,
            input_data=input_data,
            strategy=ProcessingStrategy(strategy)
        )
        
        return DistributedWorkloadResponse(
            workload_id=workload_id,
            status="submitted",
            message=f"Batch AI inference workload submitted with {len(queries)} queries",
            submitted_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit AI inference batch: {str(e)}")

@app.post("/api/distributed/workloads/multi-agent")
async def submit_multi_agent_task(
    task_definition: Dict[str, Any],
    agents: List[str] = Query(["copilot", "agno", "mastra"], description="Agents to collaborate"),
    strategy: str = Query("parallel", description="Processing strategy"),
    current_user: UserInDB = Depends(require_permission("agent:create"))
):
    """Submit a multi-agent collaborative task."""
    try:
        # Prepare multi-agent workload
        input_data = {
            "task_definition": {
                **task_definition,
                "agents": agents
            }
        }
        
        workload_id = await submit_distributed_workload(
            workload_type=WorkloadType.MULTI_AGENT_TASK,
            input_data=input_data,
            strategy=ProcessingStrategy(strategy)
        )
        
        return DistributedWorkloadResponse(
            workload_id=workload_id,
            status="submitted",
            message=f"Multi-agent collaborative task submitted with {len(agents)} agents",
            submitted_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit multi-agent task: {str(e)}")

# Performance optimization endpoints
@app.get("/api/performance/stats")
async def get_performance_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get comprehensive performance statistics"""
    try:
        return await performance_service.get_performance_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

@app.get("/api/performance/health")
async def get_performance_health(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get performance service health status"""
    try:
        return await performance_service.health_check()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance health: {str(e)}")

@app.post("/api/performance/optimize")
async def optimize_performance(
    current_user: UserInDB = Depends(require_permission("system:admin"))
):
    """Run performance optimization routines"""
    try:
        return await performance_service.optimize_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize performance: {str(e)}")

@app.post("/api/performance/cache/invalidate")
async def invalidate_cache(
    pattern: str = Query(..., description="Cache pattern to invalidate"),
    current_user: UserInDB = Depends(require_permission("system:admin"))
):
    """Invalidate cache entries matching pattern"""
    try:
        if performance_service.redis_cache:
            count = await performance_service.redis_cache.delete_pattern(pattern)
            return {
                "success": True,
                "pattern": pattern,
                "invalidated_count": count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Cache not available",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")

@app.get("/api/performance/cache/stats")
async def get_cache_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get cache performance statistics"""
    try:
        if performance_service.redis_cache:
            return performance_service.redis_cache.get_stats()
        else:
            return {"error": "Cache not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@app.get("/api/performance/database/stats")
async def get_database_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get database optimization statistics"""
    try:
        if performance_service.database_optimizer:
            return performance_service.database_optimizer.get_database_stats()
        else:
            return {"error": "Database optimizer not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

@app.get("/api/performance/load-balancer/stats")
async def get_load_balancer_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get load balancer statistics"""
    try:
        if performance_service.load_balancer:
            return performance_service.load_balancer.get_stats()
        else:
            return {"error": "Load balancer not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get load balancer stats: {str(e)}")

@app.get("/api/performance/cdn/stats")
async def get_cdn_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get CDN statistics and configuration"""
    try:
        return cdn_manager.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get CDN stats: {str(e)}")

@app.get("/api/performance/response-optimization/stats")
async def get_response_optimization_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get response optimization statistics"""
    try:
        return performance_service.response_optimizer.get_comprehensive_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get response optimization stats: {str(e)}")

@app.get("/api/performance/memory/stats")
async def get_memory_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get memory management statistics"""
    try:
        return performance_service.memory_manager.get_comprehensive_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")

@app.post("/api/performance/memory/cleanup")
async def force_memory_cleanup(
    current_user: UserInDB = Depends(require_permission("system:admin"))
):
    """Force memory cleanup"""
    try:
        return await performance_service.force_memory_cleanup()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup memory: {str(e)}")

@app.get("/api/performance/request-batching/stats")
async def get_request_batching_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get request batching statistics"""
    try:
        return performance_service.request_batcher.get_comprehensive_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request batching stats: {str(e)}")

@app.get("/api/performance/websocket/stats")
async def get_websocket_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get WebSocket optimization statistics"""
    try:
        return performance_service.websocket_handler.get_handler_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket stats: {str(e)}")

@app.get("/api/performance/comprehensive")
async def get_comprehensive_performance_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get comprehensive performance statistics across all optimization components"""
    try:
        return await performance_service.get_performance_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get comprehensive performance stats: {str(e)}")

@app.post("/api/performance/optimize-response")
async def optimize_response_endpoint(
    data: Dict[str, Any],
    compress: bool = True,
    fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    paginate: bool = False,
    page: int = 1,
    page_size: Optional[int] = None,
    current_user: Optional[UserInDB] = Depends(get_optional_user)
):
    """Optimize API response with compression, field selection, and pagination"""
    try:
        return await performance_service.optimize_response(
            data=data,
            compress=compress,
            fields=fields,
            exclude_fields=exclude_fields,
            paginate=paginate,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize response: {str(e)}")

@app.post("/api/performance/cdn/invalidate")
async def invalidate_cdn_cache(
    paths: List[str],
    current_user: UserInDB = Depends(require_permission("system:admin"))
):
    """Invalidate CDN cache for specified paths"""
    try:
        result = await cdn_manager.invalidate_cache(paths)
        return {
            "cdn_invalidation": result,
            "paths": paths,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate CDN cache: {str(e)}")

# OAuth2 Endpoints
@app.get("/oauth2/authorize")
async def oauth2_authorize(
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    response_type: str = Query("code"),
    scope: str = Query("read"),
    state: Optional[str] = Query(None),
    code_challenge: Optional[str] = Query(None),
    code_challenge_method: Optional[str] = Query(None)
):
    """OAuth2 authorization endpoint."""
    try:
        if not oauth2_provider:
            raise HTTPException(status_code=503, detail="OAuth2 provider not available")
        
        # Validate client
        client = oauth2_provider.get_client(client_id)
        if not client:
            raise HTTPException(status_code=400, detail="Invalid client_id")
        
        if redirect_uri not in client.redirect_uris:
            raise HTTPException(status_code=400, detail="Invalid redirect_uri")
        
        # Generate authorization URL
        scopes = scope.split(" ")
        auth_url = oauth2_provider.create_authorization_url(
            client_id=client_id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method
        )
        
        # In a real implementation, this would redirect to a consent page
        # For now, return the authorization data
        return {
            "authorization_url": auth_url,
            "client_name": client.client_name,
            "requested_scopes": scopes,
            "state": state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authorization failed: {str(e)}")

@app.post("/oauth2/authorize")
async def oauth2_authorize_consent(
    client_id: str,
    redirect_uri: str,
    scopes: List[str],
    state: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Handle OAuth2 consent and generate authorization code."""
    try:
        if not oauth2_provider:
            raise HTTPException(status_code=503, detail="OAuth2 provider not available")
        
        # Generate authorization code
        auth_code = oauth2_provider.generate_authorization_code(
            client_id=client_id,
            user_id=current_user.id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method
        )
        
        # Return authorization code (normally would redirect)
        return {
            "code": auth_code.code,
            "state": state,
            "redirect_uri": redirect_uri
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authorization failed: {str(e)}")

@app.post("/oauth2/token")
async def oauth2_token(
    grant_type: str,
    client_id: str,
    client_secret: Optional[str] = None,
    code: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    refresh_token: Optional[str] = None,
    code_verifier: Optional[str] = None,
    scope: Optional[str] = None
):
    """OAuth2 token endpoint."""
    try:
        if not oauth2_provider:
            raise HTTPException(status_code=503, detail="OAuth2 provider not available")
        
        if grant_type == "authorization_code":
            if not code:
                raise HTTPException(status_code=400, detail="Missing authorization code")
            
            access_token = oauth2_provider.exchange_code_for_tokens(
                code=code,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier
            )
            
            return {
                "access_token": access_token.access_token,
                "token_type": access_token.token_type,
                "refresh_token": access_token.refresh_token,
                "expires_in": int((access_token.expires_at - datetime.now(timezone.utc)).total_seconds()),
                "scope": " ".join(access_token.scopes)
            }
            
        elif grant_type == "refresh_token":
            if not refresh_token:
                raise HTTPException(status_code=400, detail="Missing refresh token")
            
            new_scopes = scope.split(" ") if scope else None
            new_token = oauth2_provider.refresh_access_token(
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                scopes=new_scopes
            )
            
            return {
                "access_token": new_token.access_token,
                "token_type": new_token.token_type,
                "refresh_token": new_token.refresh_token,
                "expires_in": int((new_token.expires_at - datetime.now(timezone.utc)).total_seconds()),
                "scope": " ".join(new_token.scopes)
            }
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported grant type")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/oauth2/introspect")
async def oauth2_introspect(
    token: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
):
    """OAuth2 token introspection endpoint."""
    try:
        if not oauth2_provider:
            raise HTTPException(status_code=503, detail="OAuth2 provider not available")
        
        # Authenticate client if credentials provided
        if client_id and client_secret:
            oauth2_provider.authenticate_client(client_id, client_secret)
        
        introspection = oauth2_provider.introspect_token(token, client_id)
        return introspection
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/oauth2/revoke")
async def oauth2_revoke(
    token: str,
    token_type_hint: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
):
    """OAuth2 token revocation endpoint."""
    try:
        if not oauth2_provider:
            raise HTTPException(status_code=503, detail="OAuth2 provider not available")
        
        # Authenticate client if credentials provided
        if client_id and client_secret:
            oauth2_provider.authenticate_client(client_id, client_secret)
        
        oauth2_provider.revoke_token(token, token_type_hint, client_id)
        return {"revoked": True}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Webhook Endpoints
@app.post("/webhooks/{integration_id}")
async def receive_webhook(
    integration_id: str,
    request: Request
):
    """Universal webhook receiver for all integrations."""
    try:
        if not webhook_receiver:
            raise HTTPException(status_code=503, detail="Webhook receiver not available")
        
        # Process webhook
        result = await webhook_receiver.receive_webhook(request, integration_id)
        
        if not result["success"]:
            if "rate limit" in result.get("message", "").lower():
                raise HTTPException(status_code=429, detail=result["message"])
            else:
                raise HTTPException(status_code=400, detail=result["message"])
        
        # Return appropriate response based on integration
        response_data = result.get("data", {})
        
        # Handle special response types
        if response_data.get("response_type") == "url_verification":
            # Slack URL verification
            return {"challenge": response_data.get("challenge")}
        elif response_data.get("response_type") == "validation":
            # Teams validation
            return response_data.get("validationToken", "")
        else:
            # Standard response
            return {"status": "ok", "message": "Webhook processed successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Webhook processing error for {integration_id}: {str(e)}",
            EventType.ERROR,
            {"integration_id": integration_id, "error": str(e)},
            "webhook_endpoint"
        )
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.get("/webhooks/stats")
async def get_webhook_stats(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get webhook processing statistics."""
    try:
        if not webhook_receiver:
            raise HTTPException(status_code=503, detail="Webhook receiver not available")
        
        stats = webhook_receiver.get_webhook_stats()
        return {
            "webhook_stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get webhook stats: {str(e)}")

# Database Management Endpoints
@app.get("/api/database/status")
async def get_database_status(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get comprehensive database status and health information"""
    try:
        status = await database_service.get_database_health()
        return {
            "database_status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database status: {str(e)}")

@app.get("/api/database/statistics")
async def get_database_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get database usage statistics for the specified number of days"""
    try:
        stats = await database_service.get_usage_statistics(days)
        return {
            "statistics": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")

@app.post("/api/database/optimize")
async def optimize_database(
    current_user: UserInDB = Depends(require_permission("system:manage"))
):
    """Optimize database performance by running VACUUM ANALYZE on all tables"""
    try:
        result = await database_service.optimize_database()
        return {
            "optimization_result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database optimization failed: {str(e)}")

@app.post("/api/database/backup")
async def create_database_backup(
    backup_path: Optional[str] = None,
    current_user: UserInDB = Depends(require_permission("system:manage"))
):
    """Create a database backup"""
    try:
        backup_info = await database_service.backup_database(backup_path)
        return {
            "backup_info": backup_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database backup failed: {str(e)}")

@app.post("/api/database/retention/apply")
async def apply_retention_policies(
    current_user: UserInDB = Depends(require_permission("system:manage"))
):
    """Apply data retention policies to clean up old data"""
    try:
        result = await database_service.apply_retention_policies()
        return {
            "retention_result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply retention policies: {str(e)}")

@app.get("/api/database/retention/report")
async def get_data_usage_report(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get data usage report showing table sizes and row counts"""
    try:
        report = await database_service.retention_manager.get_data_usage_report()
        return {
            "data_usage_report": report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate data usage report: {str(e)}")

@app.post("/api/database/migrate")
async def run_database_migrations(
    current_user: UserInDB = Depends(require_permission("system:manage"))
):
    """Run database migrations to create/update schema"""
    try:
        migration_manager = get_migration_manager()
        await migration_manager.create_migration_table()
        result = await migration_manager.create_all_tables()
        
        return {
            "migration_result": {
                "success": result,
                "message": "Database tables created/updated successfully"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database migration failed: {str(e)}")

@app.get("/api/database/migrations/history")
async def get_migration_history(
    current_user: UserInDB = Depends(require_permission("system:read"))
):
    """Get database migration history"""
    try:
        migration_manager = get_migration_manager()
        applied_migrations = await migration_manager.get_applied_migrations()
        
        return {
            "migration_history": applied_migrations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get migration history: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}