"""
Modular SAFLA MCP Server implementation.

This module provides the main MCP server class that handles protocol
communication and delegates to specialized handlers for different domains.
"""

import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import logging
from aiohttp import web, ClientSession
from aiohttp.web import Response, Request

from safla.utils.config import SAFLAConfig, get_config
from safla.utils.logging import setup_logging, get_logger
from safla.utils.validation import validate_config
from safla.exceptions import SAFLAError

from .state.manager import StateManager
from .handlers.base import BaseHandler
from .resources.base import BaseResource

# Import specialized handlers
from .handlers.system import SystemHandler
from .handlers.optimization import OptimizationHandler
from .handlers.benchmarking import BenchmarkingHandler
from .handlers.deployment import DeploymentHandler
from .handlers.admin import AdminHandler
from .handlers.testing import TestingHandler
from .handlers.agent import AgentHandler
from .handlers.metacognitive import MetaCognitiveHandler

# Import enhanced endpoints
try:
    from safla.api.enhanced_endpoints import EnhancedSAFLAEndpoints
except ImportError:
    EnhancedSAFLAEndpoints = None

logger = get_logger(__name__)


class ModularMCPServer:
    """
    Modular SAFLA MCP Server implementation.
    
    This server follows the Model Context Protocol (MCP) for stdio communication
    and delegates functionality to specialized handlers for better maintainability
    and scalability.
    """
    
    def __init__(self, config: Optional[SAFLAConfig] = None):
        """
        Initialize the modular MCP server.
        
        Args:
            config: Optional SAFLA configuration
        """
        self.config = config or get_config()
        self.request_id = 0
        self.start_time = time.time()
        
        # Initialize state manager
        self.state_manager = StateManager(
            enable_persistence=True,
            persistence_path="safla_mcp_state.json"
        )
        
        # Handler and resource registries
        self._handlers: Dict[str, BaseHandler] = {}
        self._resources: Dict[str, BaseResource] = {}
        
        # Initialize components
        self._initialize_handlers()
        self._initialize_resources()
        
        logger.info("Modular MCP Server initialized")
    
    def _initialize_handlers(self) -> None:
        """Initialize and register all handlers."""
        handler_classes = [
            SystemHandler,
            OptimizationHandler,
            BenchmarkingHandler,
            DeploymentHandler,
            AdminHandler,
            TestingHandler,
            AgentHandler,
            MetaCognitiveHandler,
        ]
        
        for handler_class in handler_classes:
            try:
                handler = handler_class(self.config, self.state_manager)
                self._register_handler(handler)
                logger.debug(f"Registered handler: {handler_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize handler {handler_class.__name__}: {str(e)}")
    
    def _initialize_resources(self) -> None:
        """Initialize and register all resources."""
        # Resource initialization will be similar to handlers
        # For now, we'll leave this as a placeholder
        pass
    
    def _register_handler(self, handler: BaseHandler) -> None:
        """Register a handler and its tools."""
        for tool_name in handler._tools:
            if tool_name in self._handlers:
                logger.warning(f"Tool {tool_name} already registered, overwriting")
            self._handlers[tool_name] = handler
    
    async def run(self) -> None:
        """Run the MCP server, handling stdio communication."""
        logger.info("Starting MCP server stdio communication")
        
        try:
            while True:
                # Read request from stdin
                line = await self._read_line()
                if not line:
                    break
                
                try:
                    request = json.loads(line)
                    response = await self.handle_request(request)
                    
                    # Write response to stdout
                    await self._write_response(response)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON request: {str(e)}")
                    error_response = self._create_error_response(
                        -32700, "Parse error", str(e)
                    )
                    await self._write_response(error_response)
                except Exception as e:
                    logger.error(f"Request handling error: {str(e)}")
                    error_response = self._create_error_response(
                        -32603, "Internal error", str(e)
                    )
                    await self._write_response(error_response)
        
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            await self._cleanup()
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.
        
        Args:
            request: The JSON-RPC request
            
        Returns:
            JSON-RPC response
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.debug(f"Handling request: {method}")
        
        try:
            # Route to appropriate handler
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "list_tools":
                result = await self._handle_list_tools()
            elif method == "call_tool":
                result = await self._handle_call_tool(params)
            elif method == "list_resources":
                result = await self._handle_list_resources()
            elif method == "read_resource":
                result = await self._handle_read_resource(params)
            else:
                raise SAFLAError(f"Unknown method: {method}")
            
            return self._create_response(request_id, result)
            
        except Exception as e:
            logger.error(f"Error handling request {method}: {str(e)}")
            return self._create_error_response(
                -32603, "Internal error", str(e), request_id
            )
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        client_info = params.get("client_info", {})
        
        logger.info(f"Client connected: {client_info.get('name', 'Unknown')}")
        
        # Store client info in state
        self.state_manager.set(
            "client_info", client_info, namespace="session"
        )
        
        return {
            "server_info": {
                "name": "SAFLA MCP Server",
                "version": "2.0.0",
                "description": "Modular Self-Aware Future Learning Architecture MCP Server"
            },
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": False,
                "modular_architecture": True
            }
        }
    
    async def _handle_list_tools(self) -> Dict[str, Any]:
        """Handle list tools request."""
        tools = []
        
        # Collect tools from all handlers
        seen_tools = set()
        for tool_name, handler in self._handlers.items():
            if tool_name not in seen_tools:
                tools.extend(handler.get_tools())
                seen_tools.update(handler._tools.keys())
        
        return {"tools": tools}
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            raise SAFLAError("Tool name is required")
        
        # Find handler for this tool
        handler = self._handlers.get(tool_name)
        if not handler:
            raise SAFLAError(f"Tool not found: {tool_name}")
        
        # Delegate to handler
        result = await handler.handle_tool_call(tool_name, arguments)
        
        return result
    
    async def _handle_list_resources(self) -> Dict[str, Any]:
        """Handle list resources request."""
        resources = []
        
        # Collect resources from all resource handlers
        for resource_handler in self._resources.values():
            resources.extend(resource_handler.get_resources())
        
        return {"resources": resources}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read resource request."""
        uri = params.get("uri")
        
        if not uri:
            raise SAFLAError("Resource URI is required")
        
        # Find resource handler
        for resource_handler in self._resources.values():
            if resource_handler.has_resource(uri):
                return await resource_handler.read_resource(uri)
        
        raise SAFLAError(f"Resource not found: {uri}")
    
    async def _read_line(self) -> Optional[str]:
        """Read a line from stdin asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sys.stdin.readline)
    
    async def _write_response(self, response: Dict[str, Any]) -> None:
        """Write response to stdout."""
        response_str = json.dumps(response) + "\n"
        sys.stdout.write(response_str)
        sys.stdout.flush()
    
    def _create_response(self, request_id: Any, result: Any) -> Dict[str, Any]:
        """Create a JSON-RPC response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    def _create_error_response(self, code: int, message: str, 
                             data: Optional[str] = None,
                             request_id: Optional[Any] = None) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        error = {
            "code": code,
            "message": message
        }
        if data:
            error["data"] = data
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error
        }
    
    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up server resources")
        
        # Save state if persistence is enabled
        stats = self.state_manager.get_stats()
        logger.info(f"State manager stats: {stats}")
        
        # Cleanup handlers
        # (handlers can implement cleanup methods if needed)
        
        logger.info("Server shutdown complete")


async def health_check(request: Request) -> Response:
    """Health check endpoint."""
    try:
        import torch
        health_data = {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "version": "0.1.3",
            "timestamp": time.time()
        }
        
        if torch.cuda.is_available():
            health_data["gpu_name"] = torch.cuda.get_device_name()
            health_data["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            health_data["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            
    except ImportError:
        health_data = {
            "status": "healthy",
            "gpu_available": False,
            "version": "0.1.3",
            "timestamp": time.time()
        }
    
    return web.json_response(health_data)


async def safla_api(request: Request) -> Response:
    """SAFLA API endpoint for optimization requests."""
    try:
        data = await request.json()
        
        # Get the method from the request
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id", "unknown")
        
        # Initialize enhanced endpoints if available
        enhanced = None
        if EnhancedSAFLAEndpoints:
            if hasattr(request.app, 'safla_enhanced'):
                enhanced = request.app['safla_enhanced']
            else:
                # Create a mock SAFLA instance for enhanced endpoints
                from types import SimpleNamespace
                import asyncio
                
                async def mock_generate_embeddings(texts):
                    return [[0.1] * 384 for _ in texts]
                
                safla_mock = SimpleNamespace(
                    neural_engine=SimpleNamespace(
                        generate_embeddings=mock_generate_embeddings
                    ),
                    memory=SimpleNamespace(
                        store=lambda k, v: None,
                        retrieve=lambda k: None
                    )
                )
                enhanced = EnhancedSAFLAEndpoints(safla_mock)
                # Cache it for future requests
                request.app['safla_enhanced'] = enhanced
        
        # Add debug endpoints
        if method == "debug_info":
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "enhanced_available": enhanced is not None,
                    "enhanced_class": str(type(enhanced)) if enhanced else None,
                    "method_requested": method,
                    "params_received": params,
                    "enhanced_methods": [attr for attr in dir(enhanced) if not attr.startswith('_')] if enhanced else []
                }
            }
        elif method == "enhanced_status":
            response_data = {
                "jsonrpc": "2.0", 
                "id": request_id,
                "result": {
                    "EnhancedSAFLAEndpoints_imported": EnhancedSAFLAEndpoints is not None,
                    "enhanced_instance": enhanced is not None,
                    "available_methods": [
                        "analyze_text", "detect_patterns", "build_knowledge_graph",
                        "batch_process", "consolidate_memories", "optimize_parameters", 
                        "create_session", "export_memory_snapshot", "run_benchmark",
                        "monitor_health"
                    ]
                }
            }
        # Route to appropriate handler
        elif enhanced and method in [
            "analyze_text", "detect_patterns", "build_knowledge_graph",
            "batch_process", "consolidate_memories", "optimize_parameters",
            "create_session", "export_memory_snapshot", "run_benchmark",
            "monitor_health"
        ]:
            # Handle enhanced methods
            handler_method = getattr(enhanced, method, None)
            if handler_method:
                result = await handler_method(**params)
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            else:
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        else:
            # Default response for basic methods
            response_data = {
                "status": "success",
                "message": "SAFLA optimization request processed",
                "timestamp": time.time(),
                "request_id": request_id,
                "method": method,
                "params": params
            }
        
        return web.json_response(response_data)
        
    except Exception as e:
        return web.json_response({
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }, status=500)


def start_server(host: str = "0.0.0.0", port: int = 8080, gpu_enabled: bool = False, config: Optional[SAFLAConfig] = None):
    """Start SAFLA HTTP server."""
    app = web.Application()
    
    # Add routes
    app.router.add_get('/health', health_check)
    app.router.add_post('/api/safla', safla_api)
    app.router.add_get('/', lambda r: web.Response(text="SAFLA Server Running"))
    
    logger.info(f"Starting SAFLA server on {host}:{port}")
    if gpu_enabled:
        logger.info("GPU optimization enabled")
    
    web.run_app(app, host=host, port=port)


def main():
    """Main entry point for the MCP server."""
    # Setup logging
    setup_logging()
    
    # Get configuration
    config = get_config()
    
    # Validate configuration
    try:
        validate_config(config)
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        sys.exit(1)
    
    # Create and run server
    server = ModularMCPServer(config)
    
    # Run the server
    asyncio.run(server.run())


if __name__ == "__main__":
    main()