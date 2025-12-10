"""
Enhanced API Server with Full Phase 3 Integration
Integrates load balancing, authentication, rate limiting, and cluster management
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available - Enhanced API server will not work")

# Import our components
try:
    from .distributed_inference_engine import DistributedInferenceEngine, InferenceRequest, InferenceStatus
    from .memory_manager import DistributedMemoryManager
    from .load_balancer import LoadBalancer, NodeInfo, LoadBalancingStrategy, create_load_balancer, RequestRouter
    from .auth import AuthenticationManager, FastAPIAuthenticator, Permission, UserRole, create_auth_manager
    from .rate_limiter import RateLimiter, RateLimitType, create_rate_limiter, create_common_limits
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    logging.warning("Some components not available - running in limited mode")

logger = logging.getLogger(__name__)

# Enhanced Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for inference")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter")
    stop: Optional[List[str]] = Field([], description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Stream response")
    user: Optional[str] = Field(None, description="User identifier")

class ClusterStatusResponse(BaseModel):
    status: str
    total_nodes: int
    healthy_nodes: int
    loaded_models: List[str]
    cluster_load: Dict[str, Any]
    load_balancer_stats: Dict[str, Any]

class MetricsResponse(BaseModel):
    server_metrics: Dict[str, Any]
    cluster_metrics: Dict[str, Any]
    rate_limit_stats: Dict[str, Any]
    auth_stats: Dict[str, Any]

class EnhancedDistributedAPIServer:
    """
    Enhanced API server with full Phase 3 capabilities:
    - Load balancing across cluster nodes
    - API key authentication and authorization
    - Rate limiting with token bucket algorithm
    - Integrated cluster management
    """
    
    def __init__(self, node_id: str, config: Optional[Dict[str, Any]] = None):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available - cannot create enhanced API server")
        
        self.node_id = node_id
        self.config = config or {}
        
        # Core components
        self.inference_engine: Optional[DistributedInferenceEngine] = None
        self.memory_manager: Optional[DistributedMemoryManager] = None
        
        # Phase 3 components
        self.load_balancer: Optional[LoadBalancer] = None
        self.request_router: Optional[RequestRouter] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.fastapi_auth: Optional[FastAPIAuthenticator] = None
        
        # Server state
        self.start_time = time.time()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8000)
        self.max_requests = self.config.get('max_requests', 100)
        self.request_timeout = self.config.get('request_timeout', 300)
        
        # Security configuration
        self.enable_auth = self.config.get('enable_auth', True)
        self.enable_rate_limiting = self.config.get('enable_rate_limiting', True)
        self.enable_load_balancing = self.config.get('enable_load_balancing', True)
        self.trusted_hosts = self.config.get('trusted_hosts', ["*"])
        
        # Create FastAPI app
        self.app = self._create_app()
        
        logger.info(f"Enhanced distributed API server initialized for node {node_id}")
    
    def _create_app(self) -> FastAPI:
        """Create and configure enhanced FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            # Background tasks
            self._start_background_tasks()
            yield
            # Shutdown
            await self._shutdown()
        
        app = FastAPI(
            title="Enhanced Distributed MLX-Exo API",
            description="Production-ready API for distributed inference on Apple Silicon cluster",
            version="2.0.0",
            lifespan=lifespan
        )
        
        # Add security middleware
        if self.trusted_hosts and self.trusted_hosts != ["*"]:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.trusted_hosts
            )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add custom middleware
        app.middleware("http")(self._request_middleware)
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    async def _request_middleware(self, request: Request, call_next):
        """Custom middleware for request processing"""
        start_time = time.time()
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        try:
            response = await call_next(request)
            
            # Log successful request
            duration = time.time() - start_time
            logger.info(f"Request {request_id} completed in {duration:.3f}s")
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Log failed request
            duration = time.time() - start_time
            logger.error(f"Request {request_id} failed after {duration:.3f}s: {e}")
            raise
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes with enhanced features"""
        
        # Create authentication dependency
        if self.enable_auth and COMPONENTS_AVAILABLE:
            def get_auth():
                return self.fastapi_auth.authenticate if self.fastapi_auth else None
            
            def require_permission(permission: Permission):
                return self.fastapi_auth.require_permission(permission) if self.fastapi_auth else None
        else:
            # Mock auth for development
            async def get_auth():
                return None
            
            def require_permission(permission: Permission):
                async def mock_auth():
                    return None
                return mock_auth
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with server information"""
            return {
                "message": "Enhanced Distributed MLX-Exo API Server",
                "node_id": self.node_id,
                "version": "2.0.0",
                "status": "running",
                "features": {
                    "authentication": self.enable_auth,
                    "rate_limiting": self.enable_rate_limiting,
                    "load_balancing": self.enable_load_balancing
                }
            }
        
        @app.get("/health", response_model=ClusterStatusResponse)
        async def health_check():
            """Enhanced health check with cluster status"""
            cluster_status = {"status": "unknown"}
            loaded_models = []
            cluster_load = {}
            load_balancer_stats = {}
            
            if self.inference_engine:
                engine_status = self.inference_engine.get_engine_status()
                cluster_status = engine_status
                loaded_models = engine_status.get('loaded_models', [])
            
            if self.load_balancer:
                cluster_load = self.load_balancer.get_cluster_status()
                load_balancer_stats = self.load_balancer.get_routing_stats()
            
            return ClusterStatusResponse(
                status="healthy",
                total_nodes=cluster_load.get('total_nodes', 0),
                healthy_nodes=cluster_load.get('healthy_nodes', 0),
                loaded_models=loaded_models,
                cluster_load=cluster_load,
                load_balancer_stats=load_balancer_stats
            )
        
        @app.get("/v1/models")
        async def list_models(auth = Depends(require_permission(Permission.MODEL_LIST)) if self.enable_auth else None):
            """List available models (with auth)"""
            models = []
            
            if self.inference_engine:
                loaded_models = self.inference_engine.loaded_models
                for model_name in loaded_models.keys():
                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": int(loaded_models[model_name].get('loaded_at', time.time())),
                        "owned_by": "distributed-mlx-exo"
                    })
            else:
                # Default models if engine not available
                default_models = ["llama-7b", "llama-13b", "mistral-7b"]
                for model_name in default_models:
                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "distributed-mlx-exo"
                    })
            
            return {"object": "list", "data": models}
        
        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            background_tasks: BackgroundTasks,
            http_request: Request,
            auth = Depends(require_permission(Permission.INFERENCE)) if self.enable_auth else None
        ):
            """Enhanced chat completions with full Phase 3 features"""
            request_id = http_request.state.request_id
            
            try:
                # Rate limiting
                if self.enable_rate_limiting and self.rate_limiter and auth:
                    user_key = f"user:{auth.user_id}" if auth else f"ip:{http_request.client.host}"
                    estimated_tokens = request.max_tokens or 100
                    
                    rate_result = self.rate_limiter.check_limit(user_key, tokens=estimated_tokens)
                    if not rate_result.allowed:
                        raise HTTPException(
                            status_code=429,
                            detail=f"Rate limit exceeded. Retry after {rate_result.retry_after:.1f} seconds",
                            headers={
                                "Retry-After": str(int(rate_result.retry_after or 1)),
                                "X-RateLimit-Limit": str(int(rate_result.limit)),
                                "X-RateLimit-Remaining": str(int(rate_result.remaining)),
                                "X-RateLimit-Reset": str(int(rate_result.reset_time))
                            }
                        )
                
                # Validate request
                if not request.messages:
                    raise HTTPException(status_code=400, detail="Messages cannot be empty")
                
                # Prepare request data for routing
                request_data = {
                    'request_id': request_id,
                    'model': request.model,
                    'messages': [msg.dict() for msg in request.messages],
                    'max_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                    'top_k': request.top_k,
                    'stop': request.stop,
                    'stream': request.stream,
                    'user': auth.user_id if auth else None
                }
                
                # Route and execute request
                if self.enable_load_balancing and self.request_router:
                    if request.stream:
                        return StreamingResponse(
                            self._stream_routed_chat_completion(request_data),
                            media_type="text/event-stream"
                        )
                    else:
                        response = await self._handle_routed_chat_completion(request_data)
                        background_tasks.add_task(self._cleanup_request, request_id)
                        return response
                else:
                    # Fallback to direct execution
                    if request.stream:
                        return StreamingResponse(
                            self._stream_direct_chat_completion(request_data),
                            media_type="text/event-stream"
                        )
                    else:
                        response = await self._handle_direct_chat_completion(request_data)
                        background_tasks.add_task(self._cleanup_request, request_id)
                        return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Chat completion failed for {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/v1/cluster/status")
        async def cluster_status(auth = Depends(require_permission(Permission.CLUSTER_STATUS)) if self.enable_auth else None):
            """Get detailed cluster status"""
            status = {}
            
            if self.load_balancer:
                status["load_balancer"] = self.load_balancer.get_cluster_status()
                status["routing"] = self.load_balancer.get_routing_stats()
            
            if self.inference_engine:
                status["inference_engine"] = self.inference_engine.get_engine_status()
            
            if self.memory_manager:
                status["memory"] = self.memory_manager.get_memory_stats()
            
            return status
        
        @app.get("/v1/metrics", response_model=MetricsResponse)
        async def get_metrics(auth = Depends(require_permission(Permission.METRICS_READ)) if self.enable_auth else None):
            """Get comprehensive system metrics"""
            server_metrics = {
                "uptime": time.time() - self.start_time,
                "active_requests": len(self.active_requests),
                "total_requests": len(self.request_history),
                "node_id": self.node_id
            }
            
            cluster_metrics = {}
            if self.load_balancer:
                cluster_metrics = self.load_balancer.get_cluster_status()
            
            rate_limit_stats = {}
            if self.rate_limiter:
                rate_limit_stats = self.rate_limiter.get_global_stats()
            
            auth_stats = {}
            if self.auth_manager:
                auth_stats = self.auth_manager.get_auth_stats()
            
            return MetricsResponse(
                server_metrics=server_metrics,
                cluster_metrics=cluster_metrics,
                rate_limit_stats=rate_limit_stats,
                auth_stats=auth_stats
            )
        
        @app.post("/v1/admin/keys")
        async def create_api_key(
            key_request: dict,
            auth = Depends(require_permission(Permission.ADMIN)) if self.enable_auth else None
        ):
            """Create new API key (admin only)"""
            if not self.auth_manager:
                raise HTTPException(status_code=501, detail="Authentication not enabled")
            
            user_id = key_request.get('user_id')
            role = UserRole(key_request.get('role', 'user'))
            description = key_request.get('description', '')
            expires_days = key_request.get('expires_days')
            rate_limit = key_request.get('rate_limit')
            
            raw_key, api_key_obj = self.auth_manager.generate_api_key(
                user_id=user_id,
                role=role,
                description=description,
                expires_days=expires_days,
                rate_limit=rate_limit
            )
            
            # Set up rate limits for new key
            if self.rate_limiter and rate_limit:
                user_key = f"user:{user_id}"
                create_common_limits(self.rate_limiter, user_key, role.value)
            
            return {
                "api_key": raw_key,
                "key_id": api_key_obj.key_id,
                "user_id": user_id,
                "role": role.value,
                "created_at": api_key_obj.created_at
            }
    
    async def _startup(self) -> None:
        """Enhanced startup logic"""
        logger.info("Starting enhanced distributed API server...")
        
        # Initialize authentication
        if self.enable_auth and COMPONENTS_AVAILABLE:
            self.auth_manager = create_auth_manager(
                jwt_secret=self.config.get('jwt_secret')
            )
            self.fastapi_auth = FastAPIAuthenticator(self.auth_manager)
            
            # Create default admin key if specified
            if self.config.get('create_admin_key'):
                admin_key, _ = self.auth_manager.generate_api_key(
                    user_id="admin",
                    role=UserRole.ADMIN,
                    description="Default admin key"
                )
                logger.info(f"Created admin API key: {admin_key}")
            
            logger.info("✓ Authentication system initialized")
        
        # Initialize rate limiting
        if self.enable_rate_limiting and COMPONENTS_AVAILABLE:
            self.rate_limiter = create_rate_limiter(
                adaptive=self.config.get('adaptive_rate_limiting', False)
            )
            logger.info("✓ Rate limiting system initialized")
        
        # Initialize load balancer
        if self.enable_load_balancing and COMPONENTS_AVAILABLE:
            strategy = LoadBalancingStrategy(
                self.config.get('load_balancing_strategy', 'resource_aware')
            )
            self.load_balancer = create_load_balancer(strategy)
            
            # Add configured nodes
            cluster_nodes = self.config.get('cluster_nodes', [])
            for node_config in cluster_nodes:
                node_info = NodeInfo(
                    node_id=node_config['node_id'],
                    host=node_config['host'],
                    port=node_config['port'],
                    weight=node_config.get('weight', 1.0),
                    max_connections=node_config.get('max_connections', 100),
                    active_models=node_config.get('active_models', [])
                )
                self.load_balancer.add_node(node_info)
            
            # Create request router
            from .load_balancer import create_request_router
            self.request_router = create_request_router(self.load_balancer)
            
            logger.info(f"✓ Load balancer initialized with {len(cluster_nodes)} nodes")
        
        # Initialize core components
        if COMPONENTS_AVAILABLE:
            try:
                # Initialize inference engine
                from .distributed_inference_engine import create_inference_engine
                self.inference_engine = create_inference_engine(self.node_id)
                
                if await self.inference_engine.initialize():
                    logger.info("✓ Inference engine initialized")
                else:
                    logger.warning("✗ Inference engine initialization failed")
                
                # Initialize memory manager
                from .memory_manager import create_memory_manager
                self.memory_manager = create_memory_manager(self.node_id)
                logger.info("✓ Memory manager initialized")
                
                # Load default models
                default_models = self.config.get('default_models', [])
                for model_config in default_models:
                    model_name = model_config['name']
                    if await self.inference_engine.load_model(model_name, model_config):
                        logger.info(f"✓ Loaded default model: {model_name}")
                    else:
                        logger.warning(f"✗ Failed to load default model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize core components: {e}")
        
        logger.info("Enhanced distributed API server startup complete")
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        async def maintenance_loop():
            while True:
                try:
                    # Cleanup expired data
                    if self.auth_manager:
                        self.auth_manager.cleanup_expired()
                    
                    if self.rate_limiter:
                        self.rate_limiter.cleanup_expired()
                    
                    # Update system metrics for adaptive rate limiting
                    if hasattr(self.rate_limiter, 'update_system_metrics'):
                        # Get system metrics (mock values for now)
                        avg_response_time = 0.5  # TODO: Calculate from request history
                        error_rate = 0.01  # TODO: Calculate from error tracking
                        cpu_usage = 50.0  # TODO: Get from system monitor
                        memory_usage = 60.0  # TODO: Get from system monitor
                        
                        self.rate_limiter.update_system_metrics(
                            avg_response_time, error_rate, cpu_usage, memory_usage
                        )
                    
                    await asyncio.sleep(60)  # Run every minute
                    
                except Exception as e:
                    logger.error(f"Maintenance loop error: {e}")
                    await asyncio.sleep(10)
        
        # Start maintenance task
        asyncio.create_task(maintenance_loop())
    
    async def _shutdown(self) -> None:
        """Enhanced shutdown logic"""
        logger.info("Shutting down enhanced distributed API server...")
        
        # Cancel active requests
        for request_id in list(self.active_requests.keys()):
            self.active_requests[request_id]["status"] = "cancelled"
        
        # Shutdown components
        if self.inference_engine:
            await self.inference_engine.shutdown()
        
        if self.memory_manager:
            self.memory_manager.cleanup()
        
        logger.info("Enhanced distributed API server shutdown complete")
    
    async def _handle_routed_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat completion with load balancing"""
        async def execute_on_node(route, data):
            # Mock execution - in production this would make HTTP requests to the target node
            await asyncio.sleep(0.5)  # Simulate processing
            return {
                "id": data['request_id'],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data['model'],
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Routed response from {route.target_node.node_id}"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        
        return await self.request_router.route_and_execute(request_data, execute_on_node)
    
    async def _stream_routed_chat_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion with load balancing"""
        try:
            # Mock streaming - in production this would stream from the target node
            mock_tokens = ["Routed", " streaming", " response", " from", " cluster"]
            for i, token in enumerate(mock_tokens):
                chunk = {
                    "id": request_data['request_id'],
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_data['model'],
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": "stop" if i == len(mock_tokens) - 1 else None
                    }]
                }
                
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.1)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_chunk = {
                "id": request_data['request_id'],
                "error": {"message": str(e), "type": "server_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def _handle_direct_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat completion directly (fallback)"""
        await asyncio.sleep(0.3)  # Simulate processing
        return {
            "id": request_data['request_id'],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data['model'],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Direct response from local inference"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 15,
                "total_tokens": 23
            }
        }
    
    async def _stream_direct_chat_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion directly (fallback)"""
        mock_tokens = ["Direct", " streaming", " response"]
        for i, token in enumerate(mock_tokens):
            chunk = {
                "id": request_data['request_id'],
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request_data['model'],
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": "stop" if i == len(mock_tokens) - 1 else None
                }]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.1)
        
        yield "data: [DONE]\n\n"
    
    async def _cleanup_request(self, request_id: str) -> None:
        """Clean up completed request"""
        if request_id in self.active_requests:
            request_data = self.active_requests.pop(request_id)
            request_data["completed_at"] = time.time()
            self.request_history.append(request_data)
            
            # Keep only last 1000 requests in history
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
    
    def run(self) -> None:
        """Run the enhanced API server"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available - cannot run enhanced server")
            return
        
        logger.info(f"Starting enhanced API server on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )

# Factory function
def create_enhanced_api_server(node_id: str, config: Optional[Dict[str, Any]] = None) -> EnhancedDistributedAPIServer:
    """Create an enhanced distributed API server"""
    return EnhancedDistributedAPIServer(node_id, config)

# Example usage
if __name__ == "__main__":
    import sys
    
    node_id = sys.argv[1] if len(sys.argv) > 1 else "enhanced-api-node-1"
    
    config = {
        'host': '0.0.0.0',
        'port': 8000,
        'enable_auth': True,
        'enable_rate_limiting': True,
        'enable_load_balancing': True,
        'create_admin_key': True,
        'load_balancing_strategy': 'resource_aware',
        'adaptive_rate_limiting': True,
        'cluster_nodes': [
            {
                'node_id': 'worker-1',
                'host': '10.0.1.10',
                'port': 8001,
                'weight': 1.0,
                'active_models': ['llama-7b', 'mistral-7b']
            },
            {
                'node_id': 'worker-2',
                'host': '10.0.1.11',
                'port': 8001,
                'weight': 1.5,
                'active_models': ['llama-7b', 'llama-13b']
            }
        ],
        'default_models': [
            {
                'name': 'llama-7b',
                'architecture': 'llama',
                'num_layers': 32,
                'hidden_size': 4096
            }
        ]
    }
    
    try:
        server = create_enhanced_api_server(node_id, config)
        server.run()
    except KeyboardInterrupt:
        logger.info("Enhanced server stopped by user")
    except Exception as e:
        logger.error(f"Enhanced server error: {e}")