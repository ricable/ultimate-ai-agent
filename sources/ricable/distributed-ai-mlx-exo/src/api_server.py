"""
Basic API Server for Distributed Inference
Provides OpenAI-compatible REST API endpoints for distributed MLX-Exo inference
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available - API server will not work")

# Import our distributed components
try:
    from .distributed_inference_engine import DistributedInferenceEngine, InferenceRequest, InferenceStatus
    from .memory_manager import DistributedMemoryManager
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    # Create dummy classes to avoid NameError
    class DistributedInferenceEngine:
        def __init__(self, *args, **kwargs): pass
    class InferenceRequest:
        def __init__(self, *args, **kwargs): pass
    class InferenceStatus:
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
    class DistributedMemoryManager:
        def __init__(self, *args, **kwargs): pass
    logging.warning("Distributed components not available")

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
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

class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for inference")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter")
    stop: Optional[List[str]] = Field([], description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Stream response")
    user: Optional[str] = Field(None, description="User identifier")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "distributed-mlx-exo"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class HealthResponse(BaseModel):
    status: str
    node_id: str
    cluster_status: Dict[str, Any]
    loaded_models: List[str]
    active_requests: int
    uptime: float

class DistributedAPIServer:
    """
    FastAPI-based server for distributed inference API
    Provides OpenAI-compatible endpoints
    """
    
    def __init__(self, node_id: str, config: Optional[Dict[str, Any]] = None):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available - cannot create API server")
        
        self.node_id = node_id
        self.config = config or {}
        
        # Initialize components
        self.inference_engine: Optional[DistributedInferenceEngine] = None
        self.memory_manager: Optional[DistributedMemoryManager] = None
        
        # Server state
        self.start_time = time.time()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8000)
        self.max_requests = self.config.get('max_requests', 100)
        self.request_timeout = self.config.get('request_timeout', 300)  # 5 minutes
        
        # Create FastAPI app
        self.app = self._create_app()
        
        logger.info(f"Distributed API server initialized for node {node_id}")
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        app = FastAPI(
            title="Distributed MLX-Exo API",
            description="OpenAI-compatible API for distributed inference on Apple Silicon cluster",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes to FastAPI app"""
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "Distributed MLX-Exo API Server",
                "node_id": self.node_id,
                "status": "running"
            }
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            cluster_status = {}
            loaded_models = []
            
            if self.inference_engine:
                engine_status = self.inference_engine.get_engine_status()
                cluster_status = engine_status
                loaded_models = engine_status.get('loaded_models', [])
            
            return HealthResponse(
                status="healthy",
                node_id=self.node_id,
                cluster_status=cluster_status,
                loaded_models=loaded_models,
                active_requests=len(self.active_requests),
                uptime=time.time() - self.start_time
            )
        
        @app.get("/v1/models", response_model=ModelsResponse)
        async def list_models():
            """List available models"""
            models = []
            
            if self.inference_engine:
                loaded_models = self.inference_engine.loaded_models
                for model_name in loaded_models.keys():
                    models.append(ModelInfo(
                        id=model_name,
                        created=int(loaded_models[model_name].get('loaded_at', time.time()))
                    ))
            else:
                # Default models if engine not available
                default_models = ["llama-7b", "llama-13b", "mistral-7b"]
                for model_name in default_models:
                    models.append(ModelInfo(
                        id=model_name,
                        created=int(time.time())
                    ))
            
            return ModelsResponse(data=models)
        
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
            """Chat completions endpoint (OpenAI compatible)"""
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            
            try:
                # Validate request
                if not request.messages:
                    raise HTTPException(status_code=400, detail="Messages cannot be empty")
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(request.messages)
                
                # Create inference request
                inference_request = self._create_inference_request(
                    request_id=request_id,
                    model_name=request.model,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop_sequences=request.stop or [],
                    stream=request.stream,
                    user=request.user
                )
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_chat_completion(inference_request),
                        media_type="text/event-stream"
                    )
                else:
                    response = await self._handle_chat_completion(inference_request)
                    
                    # Add to background cleanup
                    background_tasks.add_task(self._cleanup_request, request_id)
                    
                    return response
                    
            except Exception as e:
                logger.error(f"Chat completion failed for {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/completions")
        async def completions(request: CompletionRequest, background_tasks: BackgroundTasks):
            """Text completions endpoint (OpenAI compatible)"""
            request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            
            try:
                # Create inference request
                inference_request = self._create_inference_request(
                    request_id=request_id,
                    model_name=request.model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop_sequences=request.stop or [],
                    stream=request.stream,
                    user=request.user
                )
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_completion(inference_request),
                        media_type="text/event-stream"
                    )
                else:
                    response = await self._handle_completion(inference_request)
                    
                    # Add to background cleanup
                    background_tasks.add_task(self._cleanup_request, request_id)
                    
                    return response
                    
            except Exception as e:
                logger.error(f"Completion failed for {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/v1/status/{request_id}")
        async def get_request_status(request_id: str):
            """Get status of a specific request"""
            if request_id in self.active_requests:
                return self.active_requests[request_id]
            else:
                raise HTTPException(status_code=404, detail="Request not found")
        
        @app.delete("/v1/requests/{request_id}")
        async def cancel_request(request_id: str):
            """Cancel an active request"""
            if request_id in self.active_requests:
                self.active_requests[request_id]["status"] = "cancelled"
                return {"message": f"Request {request_id} cancelled"}
            else:
                raise HTTPException(status_code=404, detail="Request not found")
        
        @app.get("/metrics")
        async def get_metrics():
            """Get server and cluster metrics"""
            metrics = {
                "server": {
                    "uptime": time.time() - self.start_time,
                    "active_requests": len(self.active_requests),
                    "total_requests": len(self.request_history),
                    "node_id": self.node_id
                }
            }
            
            if self.inference_engine:
                engine_status = self.inference_engine.get_engine_status()
                metrics["inference_engine"] = engine_status.get("metrics", {})
            
            if self.memory_manager:
                memory_stats = self.memory_manager.get_memory_stats()
                metrics["memory"] = memory_stats
            
            return metrics
    
    async def _startup(self) -> None:
        """Server startup logic"""
        logger.info("Starting distributed API server...")
        
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
                
                # Load default models if configured
                default_models = self.config.get('default_models', [])
                for model_config in default_models:
                    model_name = model_config['name']
                    if await self.inference_engine.load_model(model_name, model_config):
                        logger.info(f"✓ Loaded default model: {model_name}")
                    else:
                        logger.warning(f"✗ Failed to load default model: {model_name}")
                        
            except Exception as e:
                logger.error(f"Failed to initialize components: {e}")
        else:
            logger.warning("Running in mock mode - distributed components not available")
        
        logger.info("Distributed API server startup complete")
    
    async def _shutdown(self) -> None:
        """Server shutdown logic"""
        logger.info("Shutting down distributed API server...")
        
        # Cancel active requests
        for request_id in list(self.active_requests.keys()):
            self.active_requests[request_id]["status"] = "cancelled"
        
        # Shutdown components
        if self.inference_engine:
            await self.inference_engine.shutdown()
        
        if self.memory_manager:
            self.memory_manager.cleanup()
        
        logger.info("Distributed API server shutdown complete")
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt"""
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    def _create_inference_request(self, **kwargs) -> InferenceRequest:
        """Create an inference request object"""
        request_id = kwargs['request_id']
        
        # Track request
        self.active_requests[request_id] = {
            "request_id": request_id,
            "status": "pending",
            "created_at": time.time(),
            "model": kwargs['model_name'],
            "prompt_length": len(kwargs['prompt'])
        }
        
        if COMPONENTS_AVAILABLE and self.inference_engine:
            return self.inference_engine.create_inference_request(
                model_name=kwargs['model_name'],
                prompt=kwargs['prompt'],
                max_tokens=kwargs.get('max_tokens', 100),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 40),
                stop_sequences=kwargs.get('stop_sequences', []),
                stream=kwargs.get('stream', False),
                metadata={'user': kwargs.get('user')}
            )
        else:
            # Mock request for testing
            return InferenceRequest(
                request_id=request_id,
                model_name=kwargs['model_name'],
                prompt=kwargs['prompt'],
                max_tokens=kwargs.get('max_tokens', 100),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 40),
                stop_sequences=kwargs.get('stop_sequences', []),
                stream=kwargs.get('stream', False),
                metadata={'user': kwargs.get('user')},
                created_at=time.time()
            )
    
    async def _handle_chat_completion(self, request: InferenceRequest) -> ChatCompletionResponse:
        """Handle non-streaming chat completion"""
        start_time = time.time()
        
        try:
            self.active_requests[request.request_id]["status"] = "processing"
            
            if self.inference_engine:
                response = await self.inference_engine.distributed_inference(request)
                generated_text = response.generated_text
                tokens_generated = response.tokens_generated
            else:
                # Mock response
                await asyncio.sleep(0.5)  # Simulate processing time
                generated_text = f"Mock response to: {request.prompt[:50]}..."
                tokens_generated = min(request.max_tokens, 50)
            
            # Calculate usage
            prompt_tokens = len(request.prompt.split())  # Rough estimate
            
            completion_response = ChatCompletionResponse(
                id=request.request_id,
                created=int(start_time),
                model=request.model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=tokens_generated,
                    total_tokens=prompt_tokens + tokens_generated
                )
            )
            
            self.active_requests[request.request_id]["status"] = "completed"
            return completion_response
            
        except Exception as e:
            self.active_requests[request.request_id]["status"] = "failed"
            self.active_requests[request.request_id]["error"] = str(e)
            raise e
    
    async def _handle_completion(self, request: InferenceRequest) -> CompletionResponse:
        """Handle non-streaming text completion"""
        start_time = time.time()
        
        try:
            self.active_requests[request.request_id]["status"] = "processing"
            
            if self.inference_engine:
                response = await self.inference_engine.distributed_inference(request)
                generated_text = response.generated_text
                tokens_generated = response.tokens_generated
            else:
                # Mock response
                await asyncio.sleep(0.5)  # Simulate processing time
                generated_text = f"Mock completion for: {request.prompt[:50]}..."
                tokens_generated = min(request.max_tokens, 50)
            
            # Calculate usage
            prompt_tokens = len(request.prompt.split())  # Rough estimate
            
            completion_response = CompletionResponse(
                id=request.request_id,
                created=int(start_time),
                model=request.model_name,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=generated_text,
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=tokens_generated,
                    total_tokens=prompt_tokens + tokens_generated
                )
            )
            
            self.active_requests[request.request_id]["status"] = "completed"
            return completion_response
            
        except Exception as e:
            self.active_requests[request.request_id]["status"] = "failed"
            self.active_requests[request.request_id]["error"] = str(e)
            raise e
    
    async def _stream_chat_completion(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion"""
        try:
            self.active_requests[request.request_id]["status"] = "streaming"
            
            if self.inference_engine:
                async for token in self.inference_engine.streaming_inference(request):
                    chunk = {
                        "id": request.request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token.token},
                            "finish_reason": "stop" if token.is_final else None
                        }]
                    }
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                    if token.is_final:
                        break
            else:
                # Mock streaming
                mock_tokens = ["Mock", " streaming", " response", " for", " distributed", " inference"]
                for i, token in enumerate(mock_tokens):
                    chunk = {
                        "id": request.request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": "stop" if i == len(mock_tokens) - 1 else None
                        }]
                    }
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.1)  # Simulate delay
            
            # Send final chunk
            yield "data: [DONE]\n\n"
            self.active_requests[request.request_id]["status"] = "completed"
            
        except Exception as e:
            self.active_requests[request.request_id]["status"] = "failed"
            error_chunk = {
                "id": request.request_id,
                "error": {"message": str(e), "type": "server_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def _stream_completion(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Handle streaming text completion"""
        try:
            self.active_requests[request.request_id]["status"] = "streaming"
            
            if self.inference_engine:
                async for token in self.inference_engine.streaming_inference(request):
                    chunk = {
                        "id": request.request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model_name,
                        "choices": [{
                            "index": 0,
                            "text": token.token,
                            "finish_reason": "stop" if token.is_final else None
                        }]
                    }
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                    if token.is_final:
                        break
            else:
                # Mock streaming
                mock_tokens = ["Mock", " streaming", " completion", " response"]
                for i, token in enumerate(mock_tokens):
                    chunk = {
                        "id": request.request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model_name,
                        "choices": [{
                            "index": 0,
                            "text": token,
                            "finish_reason": "stop" if i == len(mock_tokens) - 1 else None
                        }]
                    }
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.1)  # Simulate delay
            
            # Send final chunk
            yield "data: [DONE]\n\n"
            self.active_requests[request.request_id]["status"] = "completed"
            
        except Exception as e:
            self.active_requests[request.request_id]["status"] = "failed"
            error_chunk = {
                "id": request.request_id,
                "error": {"message": str(e), "type": "server_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def _cleanup_request(self, request_id: str) -> None:
        """Clean up completed request"""
        if request_id in self.active_requests:
            # Move to history
            request_data = self.active_requests.pop(request_id)
            request_data["completed_at"] = time.time()
            self.request_history.append(request_data)
            
            # Keep only last 1000 requests in history
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
    
    def run(self) -> None:
        """Run the API server"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available - cannot run server")
            return
        
        logger.info(f"Starting API server on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )

# Factory function
def create_api_server(node_id: str, config: Optional[Dict[str, Any]] = None) -> DistributedAPIServer:
    """Create a distributed API server"""
    return DistributedAPIServer(node_id, config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    node_id = sys.argv[1] if len(sys.argv) > 1 else "api-node-1"
    
    config = {
        'host': '0.0.0.0',
        'port': 8000,
        'default_models': [
            {
                'name': 'llama-7b',
                'architecture': 'llama',
                'num_layers': 32,
                'hidden_size': 4096,
                'num_attention_heads': 32,
                'vocab_size': 32000,
                'max_sequence_length': 2048
            }
        ]
    }
    
    try:
        server = create_api_server(node_id, config)
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")