# File: backend/services/agent_orchestrator.py
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from fastapi import WebSocket
from datetime import datetime

# Import framework managers
from ..frameworks.copilot.agent import CopilotKitManager
from ..frameworks.agno.agent import AgnoAgentManager
from ..frameworks.mastra.agent import MastraAgentManager
from .local_inference import LocalInferenceService

# Import metacognition agent
from ..agents.metacognition import MetacognitionAgent

# Import distributed processing
from ..distributed.ray_manager import ray_cluster_manager, submit_distributed_task
from .distributed_orchestrator import DistributedOrchestrator, WorkloadType, ProcessingStrategy

# Import monitoring components
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.logs.middleware import WebSocketLoggingMixin
from ..monitoring.metrics.performance import (
    performance_monitor, start_agent_request, finish_agent_request,
    track_websocket_connection, remove_websocket_connection
)
from ..monitoring.metrics.prometheus_metrics import (
    record_agent_request, record_websocket_connection_opened,
    record_websocket_connection_closed, record_websocket_message
)

# Import caching components
from ..cache.decorators import cache_agent_response, cache_document_analysis
from ..services.performance_service import performance_service

# --- Mock Framework Managers for initial implementation ---
# Replace these with the real implementations later.
class MockAgentManager:
    def __init__(self, framework_name):
        self.framework_name = framework_name
        print(f"{framework_name} manager initialized.")
    
    async def process_message(self, message, context):
        return {
            "content": f"Response from {self.framework_name}: You said '{message}'",
            "metadata": {"source": self.framework_name, "context_received": bool(context)}
        }
    
    def get_status(self):
        return {"status": "active", "agents": 2}

class UAP_AgentOrchestrator:
    def __init__(self):
        # Initialize framework managers
        self.copilot_manager = CopilotKitManager()  # Real CopilotKit implementation
        self.agno_manager = AgnoAgentManager()  # Real Agno implementation
        self.mastra_manager = MastraAgentManager()  # Real Mastra implementation
        self.mlx_manager = LocalInferenceService()  # MLX Apple Silicon local inference
        self.metacognition_manager = MetacognitionAgent()  # Metacognition agent
        
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Distributed processing (initialized later)
        self.distributed_orchestrator: Optional[DistributedOrchestrator] = None
        self.distributed_threshold = {
            'batch_size': 10,  # Submit to distributed system if processing >10 items
            'content_size': 50000,  # Large content threshold (chars)
            'parallel_requests': 5  # Multiple simultaneous requests
        }

    async def initialize_services(self):
        """Initialize all agent framework services."""
        try:
            # Initialize Agno framework (real implementation)
            await self.agno_manager.initialize()
            print("Agno framework initialized successfully.")
            
            # Initialize Mastra framework (real implementation)
            await self.mastra_manager.initialize()
            print("Mastra framework initialized successfully.")
            
            # Initialize CopilotKit framework (real implementation)
            await self.copilot_manager.initialize()
            print("CopilotKit framework initialized successfully.")
            
            # Initialize MLX local inference (Apple Silicon)
            await self.mlx_manager.initialize()
            print("MLX local inference initialized successfully.")
            
            # Initialize Metacognition agent
            await self.metacognition_manager.initialize()
            print("Metacognition agent initialized successfully.")
            
            print("All agent frameworks initialized successfully.")
        except Exception as e:
            print(f"Error initializing agent frameworks: {e}")
            # Continue with available frameworks

    def register_connection(self, conn_id: str, websocket: WebSocket, user=None):
        self.active_connections[conn_id] = {
            "websocket": websocket,
            "user": user,
            "connected_at": datetime.utcnow()
        }
        
        # Extract agent ID from connection ID
        agent_id = conn_id.split('_')[0] if '_' in conn_id else conn_id
        
        # Track connection in monitoring systems
        track_websocket_connection(conn_id, agent_id, {
            "user_id": user.id if user else None,
            "username": user.username if user else None
        })
        
        # Record Prometheus metrics
        record_websocket_connection_opened(agent_id, conn_id)
        
        # Log connection
        WebSocketLoggingMixin.log_connection_opened(
            conn_id, agent_id, 
            {"user_id": user.id if user else None}
        )
        
        print(f"Connection {conn_id} registered.")

    def unregister_connection(self, conn_id: str, reason: str = "normal"):
        if conn_id in self.active_connections:
            connection_info = self.active_connections[conn_id]
            
            # Calculate connection duration
            connected_at = connection_info.get("connected_at", datetime.utcnow())
            duration_seconds = (datetime.utcnow() - connected_at).total_seconds()
            
            # Extract agent ID
            agent_id = conn_id.split('_')[0] if '_' in conn_id else conn_id
            
            # Remove from monitoring systems
            remove_websocket_connection(conn_id, reason)
            
            # Record Prometheus metrics
            record_websocket_connection_closed(agent_id, conn_id, duration_seconds)
            
            # Log disconnection
            WebSocketLoggingMixin.log_connection_closed(conn_id, agent_id, reason)
            
            del self.active_connections[conn_id]
            print(f"Connection {conn_id} unregistered.")

    async def handle_agui_event(self, conn_id: str, event: Dict[str, Any]):
        """Handles incoming events from the AG-UI client."""
        event_type = event.get("type")
        
        if event_type == "user_message":
            content = event.get("content", "")
            metadata = event.get("metadata", {})
            framework = metadata.get("framework", "auto")
            
            # Route to the correct framework
            response_data = await self._route_and_process(content, framework, metadata)
            
            # Create a response event and send it back
            response_event = {
                "type": "text_message_content",
                "content": response_data.get("content", "No response."),
                "metadata": response_data.get("metadata", {})
            }
            await self._send_to_connection(conn_id, response_event)

    async def handle_http_chat(self, agent_id: str, message: str, framework: str, context: Dict) -> Dict[str, Any]:
        """Handles stateless HTTP chat requests."""
        # Check if this should be routed to distributed processing
        if self._should_use_distributed_processing(message, framework, context):
            return await self._handle_distributed_request(agent_id, message, framework, context)
        
        response = await self._route_and_process(message, framework, context)
        response["framework"] = framework if framework != 'auto' else 'copilot' # default
        return response

    async def _route_and_process(self, message: str, framework: str, context: Dict) -> Dict[str, Any]:
        """Intelligently routes a message to the correct agent framework."""
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        
        if framework == 'auto':
            # Enhanced content-based routing logic for document processing
            framework = self._determine_best_framework(message, context)
        
        # Extract agent ID (use framework as agent ID if not provided)
        agent_id = context.get('agent_id', framework)
        
        # Start performance monitoring
        request_context = start_agent_request(agent_id, framework, request_id)
        
        # Check cache first (exclude sensitive context)
        cached_response = await performance_service.get_cached_agent_response(agent_id, message)
        if cached_response:
            # Cache hit - return cached response
            response_time_ms = (time.time() - request_context['start_time']) * 1000
            finish_agent_request(request_context, success=True)
            record_agent_request(agent_id, framework, response_time_ms / 1000, success=True, 
                               response_size_bytes=len(str(cached_response)))
            
            uap_logger.log_agent_interaction(
                agent_id=agent_id, framework=framework, message=message,
                response_time_ms=response_time_ms, success=True,
                metadata={"request_id": request_id, "cached": True}
            )
            return cached_response
        
        # Log the request (cache miss)
        uap_logger.log_event(
            LogLevel.INFO,
            f"Processing agent request (cache miss): {message[:100]}...",
            EventType.AGENT,
            {
                "agent_id": agent_id,
                "framework": framework,
                "request_id": request_id,
                "message_length": len(message),
                "cache_miss": True
            },
            "agent"
        )
        
        try:
            # Process with the selected framework
            if framework == 'copilot':
                response = await self.copilot_manager.process_message(message, context)
            elif framework == 'agno':
                response = await self.agno_manager.process_message(message, context)
            elif framework == 'mastra':
                response = await self.mastra_manager.process_message(message, context)
            elif framework == 'mlx':
                response = await self.mlx_manager.process_message(message, context)
            elif framework == 'metacognition':
                response = await self.metacognition_manager.process_message(message, context)
            else:
                response = {"content": f"Error: Unknown framework '{framework}'", "metadata": {"error": "unknown_framework"}}
            
            # Calculate response time
            response_time_ms = (time.time() - request_context['start_time']) * 1000
            
            # Cache successful response (async)
            if response and not response.get('metadata', {}).get('error'):
                asyncio.create_task(
                    performance_service.cache_agent_response(agent_id, framework, message, response)
                )
            
            # Record successful request
            finish_agent_request(request_context, success=True)
            record_agent_request(agent_id, framework, response_time_ms / 1000, success=True, 
                               response_size_bytes=len(str(response)))
            
            # Log successful processing
            uap_logger.log_agent_interaction(
                agent_id=agent_id,
                framework=framework,
                message=message,
                response_time_ms=response_time_ms,
                success=True,
                metadata={
                    "request_id": request_id,
                    "response_size": len(str(response)),
                    "cached": False
                }
            )
            
            return response
            
        except Exception as e:
            # Calculate response time for failed request
            response_time_ms = (time.time() - request_context['start_time']) * 1000
            
            # Record failed request
            finish_agent_request(request_context, success=False, error_details=str(e))
            record_agent_request(agent_id, framework, response_time_ms / 1000, success=False)
            
            # Log failed processing
            uap_logger.log_agent_interaction(
                agent_id=agent_id,
                framework=framework,
                message=message,
                response_time_ms=response_time_ms,
                success=False,
                metadata={
                    "request_id": request_id,
                    "error": str(e)
                }
            )
            
            # Return error response
            return {"content": f"Error processing request: {str(e)}", "metadata": {"error": str(e), "request_id": request_id}}
    
    def _determine_best_framework(self, message: str, context: Dict) -> str:
        """Determine the best framework for processing based on message and context."""
        message_lower = message.lower()
        
        # Check for explicit metacognitive queries first
        metacognitive_keywords = [
            'metacognition', 'self-aware', 'self-improvement', 'introspection',
            'how do you think', 'your thinking process', 'self-reflection',
            'how you learn', 'your reasoning', 'your capabilities',
            'self-analysis', 'performance monitoring', 'optimization',
            'safety constraints', 'self-modification', 'metacognitive'
        ]
        
        # Check if user wants metacognitive analysis
        wants_metacognitive = any(keyword in message_lower for keyword in metacognitive_keywords)
        
        # Check context for metacognitive preference
        metacognitive_preference = (
            context.get('prefer_metacognitive', False) or
            context.get('introspection_mode', False) or
            context.get('self_analysis_mode', False)
        )
        
        # Route to metacognition for metacognitive queries
        if wants_metacognitive or metacognitive_preference:
            return 'metacognition'
        
        # Check for explicit local/offline/private inference requests
        local_keywords = [
            'local', 'offline', 'private', 'mlx', 'apple silicon', 
            'on-device', 'no internet', 'secure', 'confidential'
        ]
        
        # Check if user explicitly wants local inference
        wants_local = any(keyword in message_lower for keyword in local_keywords)
        
        # Check context for local inference preference
        local_preference = (
            context.get('prefer_local', False) or
            context.get('offline_mode', False) or
            context.get('privacy_mode', False)
        )
        
        # Route to MLX for local inference requests
        if wants_local or local_preference:
            return 'mlx'
        
        # Enhanced document processing detection
        document_keywords = [
            'document', 'analyze', 'pdf', 'text', 'file', 'extract', 
            'summarize', 'parse', 'read', 'review', 'content', 'paper',
            'report', 'article', 'manuscript', 'data', 'table', 'chart',
            'upload', 'scan', 'ocr', 'structure', 'format'
        ]
        
        # Context-based routing (file uploads, document metadata)
        has_document_context = (
            context.get('file_path') or 
            context.get('document_type') or 
            context.get('document_content') or
            context.get('file_upload')
        )
        
        # Check for document-related keywords
        has_document_keywords = any(keyword in message_lower for keyword in document_keywords)
        
        # Route to Agno for document processing
        if has_document_context or has_document_keywords:
            return 'agno'
        
        # Support and workflow-related queries go to Mastra
        support_keywords = ['support', 'help', 'workflow', 'task', 'process', 'steps']
        if any(keyword in message_lower for keyword in support_keywords):
            return 'mastra'
        
        # Default to CopilotKit for general AI conversations
        return 'copilot'
    
    def _should_use_distributed_processing(self, message: str, framework: str, context: Dict) -> bool:
        """Determine if request should use distributed processing."""
        if not self.distributed_orchestrator:
            return False
        
        # Check for batch processing indicators
        batch_keywords = ['batch', 'multiple', 'bulk', 'process all', 'analyze all']
        has_batch_keywords = any(keyword in message.lower() for keyword in batch_keywords)
        
        # Check context for batch data
        has_batch_data = (
            context.get('documents') and len(context.get('documents', [])) >= self.distributed_threshold['batch_size'] or
            context.get('files') and len(context.get('files', [])) >= self.distributed_threshold['batch_size'] or
            context.get('queries') and len(context.get('queries', [])) >= self.distributed_threshold['batch_size']
        )
        
        # Check for large content
        has_large_content = (
            len(message) > self.distributed_threshold['content_size'] or
            context.get('document_content') and len(str(context.get('document_content', ''))) > self.distributed_threshold['content_size']
        )
        
        # Check for explicit distributed processing request
        distributed_keywords = ['distributed', 'parallel', 'cluster', 'scale']
        wants_distributed = any(keyword in message.lower() for keyword in distributed_keywords)
        
        return has_batch_keywords or has_batch_data or has_large_content or wants_distributed
    
    async def _handle_distributed_request(self, agent_id: str, message: str, framework: str, context: Dict) -> Dict[str, Any]:
        """Handle request using distributed processing."""
        request_id = str(uuid.uuid4())
        
        try:
            # Determine workload type based on framework and context
            workload_type = self._determine_workload_type(framework, context)
            
            # Prepare input data for distributed processing
            input_data = self._prepare_distributed_input(message, framework, context)
            
            # Choose processing strategy
            strategy = self._choose_processing_strategy(workload_type, input_data)
            
            # Submit to distributed orchestrator
            workload_id = await self.distributed_orchestrator.submit_workload(
                workload_type=workload_type,
                input_data=input_data,
                strategy=strategy,
                metadata={
                    'agent_id': agent_id,
                    'framework': framework,
                    'request_id': request_id,
                    'original_message': message[:200] + '...' if len(message) > 200 else message
                }
            )
            
            # Log distributed request
            uap_logger.log_event(
                LogLevel.INFO,
                f"Distributed workload submitted: {workload_type.value}",
                EventType.AGENT,
                {
                    "workload_id": workload_id,
                    "agent_id": agent_id,
                    "framework": framework,
                    "request_id": request_id,
                    "strategy": strategy.value
                },
                "distributed_agent"
            )
            
            return {
                "content": f"Processing your request using distributed system. Workload ID: {workload_id}. This may take a few moments for large-scale processing.",
                "metadata": {
                    "distributed": True,
                    "workload_id": workload_id,
                    "workload_type": workload_type.value,
                    "strategy": strategy.value,
                    "request_id": request_id,
                    "status": "submitted"
                },
                "framework": framework
            }
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Distributed processing failed: {str(e)}",
                EventType.AGENT,
                {
                    "agent_id": agent_id,
                    "framework": framework,
                    "request_id": request_id,
                    "error": str(e)
                },
                "distributed_agent"
            )
            
            # Fallback to regular processing
            return await self._route_and_process(message, framework, context)
    
    def _determine_workload_type(self, framework: str, context: Dict) -> WorkloadType:
        """Determine the type of distributed workload."""
        # Check context for specific workload indicators
        if context.get('documents') or context.get('files') or framework == 'agno':
            return WorkloadType.DOCUMENT_PROCESSING
        elif context.get('queries') or 'inference' in str(context).lower():
            return WorkloadType.AI_INFERENCE
        elif context.get('items') or 'analyze' in str(context).lower():
            return WorkloadType.BATCH_ANALYSIS
        elif context.get('agents') and len(context.get('agents', [])) > 1:
            return WorkloadType.MULTI_AGENT_TASK
        else:
            return WorkloadType.AI_INFERENCE  # Default
    
    def _prepare_distributed_input(self, message: str, framework: str, context: Dict) -> Dict[str, Any]:
        """Prepare input data for distributed processing."""
        # Base input data
        input_data = {
            'message': message,
            'framework': framework,
            'context': context
        }
        
        # Add specific data based on context
        if context.get('documents'):
            input_data['documents'] = context['documents']
        elif context.get('files'):
            input_data['documents'] = [{'file_path': f} for f in context['files']]
        
        if context.get('queries'):
            input_data['queries'] = context['queries']
        elif 'batch' in message.lower():
            # Try to extract multiple queries from message
            lines = [line.strip() for line in message.split('\n') if line.strip()]
            if len(lines) > 1:
                input_data['queries'] = lines
        
        if context.get('items'):
            input_data['items'] = context['items']
            input_data['analysis_type'] = context.get('analysis_type', 'general')
        
        return input_data
    
    def _choose_processing_strategy(self, workload_type: WorkloadType, input_data: Dict[str, Any]) -> ProcessingStrategy:
        """Choose optimal processing strategy based on workload characteristics."""
        # Get data size indicators
        data_size = 0
        if 'documents' in input_data:
            data_size = len(input_data['documents'])
        elif 'queries' in input_data:
            data_size = len(input_data['queries'])
        elif 'items' in input_data:
            data_size = len(input_data['items'])
        
        # Choose strategy based on workload type and size
        if workload_type == WorkloadType.DOCUMENT_PROCESSING:
            if data_size > 50:
                return ProcessingStrategy.MAP_REDUCE
            elif data_size > 10:
                return ProcessingStrategy.PARALLEL
            else:
                return ProcessingStrategy.SEQUENTIAL
        
        elif workload_type == WorkloadType.AI_INFERENCE:
            if data_size > 20:
                return ProcessingStrategy.PARALLEL
            else:
                return ProcessingStrategy.SEQUENTIAL
        
        elif workload_type == WorkloadType.MULTI_AGENT_TASK:
            return ProcessingStrategy.PARALLEL
        
        else:
            return ProcessingStrategy.ADAPTIVE  # Let the system decide
    
    def set_distributed_orchestrator(self, distributed_orchestrator: DistributedOrchestrator):
        """Set the distributed orchestrator instance."""
        self.distributed_orchestrator = distributed_orchestrator
        uap_logger.log_system_event(
            "Distributed orchestrator linked to agent orchestrator",
            "agent_orchestrator"
        )

    async def _send_to_connection(self, conn_id: str, data: Dict[str, Any]):
        """Sends a message to a specific WebSocket connection."""
        if conn_id in self.active_connections:
            connection_info = self.active_connections[conn_id]
            websocket = connection_info.get("websocket") if isinstance(connection_info, dict) else connection_info
            
            try:
                message_json = json.dumps(data)
                await websocket.send_text(message_json)
                
                # Track message in monitoring
                agent_id = conn_id.split('_')[0] if '_' in conn_id else conn_id
                message_type = data.get("type", "unknown")
                
                # Record WebSocket message metrics
                record_websocket_message(agent_id, "sent", message_type)
                
                # Log message sent
                WebSocketLoggingMixin.log_message_sent(
                    conn_id, agent_id, message_type, 
                    processing_time_ms=None  # Could be added if we track processing time
                )
                
            except Exception as e:
                print(f"Failed to send to {conn_id}: {e}")
                WebSocketLoggingMixin.log_websocket_error(
                    conn_id, 
                    conn_id.split('_')[0] if '_' in conn_id else conn_id,
                    str(e), 
                    "send_failed"
                )
                self.unregister_connection(conn_id, "send_error")

    async def get_system_status(self) -> Dict[str, Any]:
        # Get framework statuses
        framework_status = {
            "copilot": self.copilot_manager.get_status(),
            "agno": self.agno_manager.get_status(),
            "mastra": self.mastra_manager.get_status(),
            "mlx": await self.mlx_manager.get_service_status(),
            "metacognition": await self.metacognition_manager.get_status(),
        }
        
        # Get performance metrics
        system_health = performance_monitor.get_system_health()
        agent_stats = performance_monitor.get_agent_statistics()
        
        # Get distributed processing status
        distributed_status = {}
        if self.distributed_orchestrator:
            try:
                distributed_status = await self.distributed_orchestrator.get_orchestrator_status()
            except Exception as e:
                distributed_status = {"error": str(e), "available": False}
        else:
            distributed_status = {"available": False, "reason": "Not initialized"}
        
        # Get Ray cluster status
        ray_cluster_status = {}
        try:
            ray_cluster_status = await ray_cluster_manager.get_cluster_status()
        except Exception as e:
            ray_cluster_status = {"error": str(e), "available": False}
        
        # Connection analysis
        connections_by_agent = {}
        for conn_id, connection_info in self.active_connections.items():
            agent_id = conn_id.split('_')[0] if '_' in conn_id else conn_id
            if agent_id not in connections_by_agent:
                connections_by_agent[agent_id] = 0
            connections_by_agent[agent_id] += 1
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "frameworks": framework_status,
            "connections": {
                "total_active": len(self.active_connections),
                "by_agent": connections_by_agent
            },
            "system_health": system_health,
            "agent_performance": agent_stats,
            "distributed_processing": distributed_status,
            "ray_cluster": ray_cluster_status,
            "distributed_thresholds": self.distributed_threshold,
            "monitoring": {
                "alerts_active": len(performance_monitor.alert_manager.active_alerts) if hasattr(performance_monitor, 'alert_manager') else 0,
                "metrics_collected": len(performance_monitor.metrics_history)
            }
        }

    async def cleanup(self) -> None:
        """Clean up all framework managers and resources."""
        try:
            # Clean up Mastra framework
            if hasattr(self.mastra_manager, 'cleanup'):
                await self.mastra_manager.cleanup()
                print("Mastra framework cleanup complete.")
            
            # Clean up Agno framework  
            if hasattr(self.agno_manager, 'cleanup'):
                await self.agno_manager.cleanup()
                print("Agno framework cleanup complete.")
            
            # Clean up CopilotKit framework when implemented
            if hasattr(self.copilot_manager, 'cleanup'):
                await self.copilot_manager.cleanup()
                print("CopilotKit framework cleanup complete.")
            
            # Clean up MLX local inference
            if hasattr(self.mlx_manager, 'cleanup'):
                await self.mlx_manager.cleanup()
                print("MLX local inference cleanup complete.")
            
            # Clean up Metacognition agent
            if hasattr(self.metacognition_manager, 'cleanup'):
                await self.metacognition_manager.cleanup()
                print("Metacognition agent cleanup complete.")
            
            # Clean up distributed orchestrator
            if self.distributed_orchestrator:
                await self.distributed_orchestrator.cleanup()
                print("Distributed orchestrator cleanup complete.")
            
            # Close all WebSocket connections
            for conn_id in list(self.active_connections.keys()):
                self.unregister_connection(conn_id)
            
            print("Agent orchestrator cleanup complete.")
            
        except Exception as e:
            print(f"Error during orchestrator cleanup: {e}")