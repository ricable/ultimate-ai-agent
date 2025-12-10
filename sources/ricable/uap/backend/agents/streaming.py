# backend/agents/streaming.py
"""
Agent 37: Real-Time Stream Processing & Edge AI
Ultra-low latency streaming agent with edge-cloud optimization.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Import streaming components
from ..streaming.stream_processor import (
    stream_processor, StreamEvent, EventType, ProcessingPriority,
    initialize_streaming, process_stream_event, get_streaming_metrics
)
from ..streaming.event_pipeline import (
    pipeline_manager, create_event_pipeline, process_event_through_pipeline
)
from ..streaming.anomaly_detector import (
    anomaly_manager, create_anomaly_detector, detect_stream_anomalies
)
from ..streaming.edge_optimizer import (
    edge_optimizer, optimize_stream_placement, get_optimization_status
)
from ..streaming.low_latency_core import (
    low_latency_manager, create_ultra_low_latency_processor, 
    process_ultra_low_latency, get_all_low_latency_metrics
)

# Import edge inference components
from ...edge.inference.edge_runtime import (
    edge_runtime, load_edge_model, run_edge_inference, get_edge_runtime_metrics
)
from ...edge.inference.model_compressor import (
    model_compressor, compress_model_for_edge, get_compression_statistics
)
from ...edge.inference.inference_optimizer import (
    initialize_inference_optimizer, optimize_inference_request, 
    get_inference_optimization_stats
)
from ...edge.inference.sync_manager import (
    initialize_edge_cloud_sync, sync_model_to_cloud, get_sync_status
)

# Import other UAP components
from ..distributed.ray_manager import ray_cluster_manager
from ..edge.edge_manager import EdgeManager
from ..monitoring.logs.logger import get_logger
from ..monitoring.metrics.prometheus_metrics import metrics_collector

logger = get_logger(__name__)

@dataclass
class StreamingAgentConfig:
    """Configuration for streaming agent"""
    enable_ultra_low_latency: bool = True
    enable_edge_optimization: bool = True
    enable_anomaly_detection: bool = True
    enable_distributed_processing: bool = True
    enable_edge_inference: bool = True
    enable_cloud_sync: bool = False
    
    # Stream processing settings
    stream_buffer_size: int = 10000
    worker_threads: int = 4
    batch_size: int = 1  # For ultra-low latency
    
    # Edge optimization settings
    edge_optimization_objective: str = "latency"
    max_edge_models: int = 5
    edge_cache_size_mb: int = 100
    
    # Anomaly detection settings
    anomaly_sensitivity: float = 0.95
    anomaly_window_size: int = 100
    
    # Cloud sync settings (if enabled)
    cloud_endpoint: str = ""
    device_id: str = "edge_device_001"
    api_key: str = ""

class StreamingAgent:
    """Agent 37: Real-Time Stream Processing & Edge AI"""
    
    def __init__(self, config: StreamingAgentConfig):
        self.config = config
        self.agent_id = "agent-37-streaming"
        self.is_initialized = False
        self.is_running = False
        
        # Component references
        self.stream_processor = None
        self.edge_optimizer = None
        self.anomaly_manager = None
        self.low_latency_processor = None
        self.edge_runtime = None
        self.sync_manager = None
        
        # Performance tracking
        self.metrics = {
            'events_processed': 0,
            'anomalies_detected': 0,
            'edge_optimizations': 0,
            'models_compressed': 0,
            'sync_operations': 0,
            'avg_processing_latency_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'initialization_time_ms': 0.0
        }
        
        # Event handlers
        self.event_handlers = {
            'stream_event': [],
            'anomaly_detected': [],
            'optimization_complete': [],
            'sync_complete': []
        }
        
        logger.info(f"StreamingAgent initialized with config: {asdict(config)}")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the streaming agent and all components"""
        if self.is_initialized:
            return {'status': 'already_initialized'}
        
        start_time = time.perf_counter()
        
        try:
            logger.info("Initializing StreamingAgent components...")
            
            # Initialize stream processor
            await self._initialize_stream_processor()
            
            # Initialize ultra-low latency processor
            if self.config.enable_ultra_low_latency:
                await self._initialize_low_latency_processor()
            
            # Initialize edge optimization
            if self.config.enable_edge_optimization:
                await self._initialize_edge_optimizer()
            
            # Initialize anomaly detection
            if self.config.enable_anomaly_detection:
                await self._initialize_anomaly_detection()
            
            # Initialize edge inference
            if self.config.enable_edge_inference:
                await self._initialize_edge_inference()
            
            # Initialize distributed processing
            if self.config.enable_distributed_processing:
                await self._initialize_distributed_processing()
            
            # Initialize cloud sync
            if self.config.enable_cloud_sync and self.config.cloud_endpoint:
                await self._initialize_cloud_sync()
            
            # Set up event pipelines
            await self._setup_event_pipelines()
            
            self.is_initialized = True
            initialization_time = (time.perf_counter() - start_time) * 1000
            self.metrics['initialization_time_ms'] = initialization_time
            
            logger.info(f"StreamingAgent initialized successfully in {initialization_time:.2f}ms")
            
            return {
                'status': 'initialized',
                'initialization_time_ms': initialization_time,
                'components': {
                    'stream_processor': self.stream_processor is not None,
                    'low_latency_processor': self.low_latency_processor is not None,
                    'edge_optimizer': self.edge_optimizer is not None,
                    'anomaly_detection': self.anomaly_manager is not None,
                    'edge_inference': self.edge_runtime is not None,
                    'cloud_sync': self.sync_manager is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize StreamingAgent: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'initialization_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    async def _initialize_stream_processor(self) -> None:
        """Initialize the main stream processor"""
        await initialize_streaming()
        self.stream_processor = stream_processor
        logger.info("Stream processor initialized")
    
    async def _initialize_low_latency_processor(self) -> None:
        """Initialize ultra-low latency processor"""
        self.low_latency_processor = await create_ultra_low_latency_processor(
            processor_id="streaming-agent",
            mode="ultra_low"
        )
        
        # Register handlers for ultra-low latency processing
        self.low_latency_processor.register_default_handler(self._handle_ultra_low_latency_event)
        
        logger.info("Ultra-low latency processor initialized")
    
    async def _initialize_edge_optimizer(self) -> None:
        """Initialize edge-cloud optimization"""
        self.edge_optimizer = edge_optimizer
        
        # Configure optimization objective
        from ..streaming.edge_optimizer import OptimizationObjective
        objective = OptimizationObjective(self.config.edge_optimization_objective)
        self.edge_optimizer.optimization_objective = objective
        
        logger.info(f"Edge optimizer initialized with objective: {self.config.edge_optimization_objective}")
    
    async def _initialize_anomaly_detection(self) -> None:
        """Initialize anomaly detection"""
        self.anomaly_manager = anomaly_manager
        
        # Create anomaly detectors
        await create_anomaly_detector(
            detector_id="statistical",
            method="z_score",
            sensitivity=self.config.anomaly_sensitivity,
            window_size=self.config.anomaly_window_size
        )
        
        await create_anomaly_detector(
            detector_id="moving_average",
            method="moving_average",
            sensitivity=self.config.anomaly_sensitivity
        )
        
        # Register anomaly alert handler
        self.anomaly_manager.register_alert_handler(self._handle_anomaly_alert)
        
        logger.info("Anomaly detection initialized")
    
    async def _initialize_edge_inference(self) -> None:
        """Initialize edge inference runtime"""
        self.edge_runtime = edge_runtime
        await self.edge_runtime.start()
        
        # Initialize inference optimizer
        await initialize_inference_optimizer(
            techniques=["dynamic_batching", "result_caching"],
            cache_size_mb=self.config.edge_cache_size_mb
        )
        
        logger.info("Edge inference runtime initialized")
    
    async def _initialize_distributed_processing(self) -> None:
        """Initialize distributed processing with Ray"""
        # Ray cluster manager is already initialized globally
        if ray_cluster_manager.cluster_status.value == "ready":
            logger.info("Distributed processing (Ray) available")
        else:
            logger.warning("Ray cluster not available for distributed processing")
    
    async def _initialize_cloud_sync(self) -> None:
        """Initialize cloud synchronization"""
        try:
            self.sync_manager = await initialize_edge_cloud_sync(
                cloud_endpoint=self.config.cloud_endpoint,
                device_id=self.config.device_id,
                api_key=self.config.api_key
            )
            logger.info("Cloud synchronization initialized")
        except Exception as e:
            logger.warning(f"Cloud sync initialization failed: {e}")
    
    async def _setup_event_pipelines(self) -> None:
        """Set up event processing pipelines"""
        # Create main streaming pipeline
        main_pipeline = await create_event_pipeline(
            pipeline_id="main-streaming",
            stages=[
                {
                    'stage_id': 'validate',
                    'stage_type': 'validate',
                    'name': 'Input Validation',
                    'config': {
                        'required_fields': ['event_type', 'data', 'source'],
                        'type_validations': {
                            'timestamp': 'float'
                        }
                    }
                },
                {
                    'stage_id': 'enrich',
                    'stage_type': 'enrich',
                    'name': 'Event Enrichment',
                    'config': {
                        'enrichments': {
                            'processed_by': 'streaming_agent',
                            'processing_timestamp': time.time()
                        }
                    }
                },
                {
                    'stage_id': 'optimize',
                    'stage_type': 'custom',
                    'name': 'Edge Optimization',
                    'config': {}
                }
            ]
        )
        
        # Register custom optimization stage handler
        main_pipeline.register_stage_handler('optimize', self._handle_optimization_stage)
        
        logger.info("Event processing pipelines configured")
    
    async def process_stream_event(self, 
                                 event_type: str,
                                 data: Any,
                                 source: str = "external",
                                 priority: str = "normal",
                                 enable_ultra_low_latency: bool = None) -> Dict[str, Any]:
        """Process a streaming event with full optimization pipeline"""
        start_time = time.perf_counter()
        
        try:
            # Create stream event
            from ..streaming.stream_processor import StreamEvent, EventType, ProcessingPriority
            
            stream_event = StreamEvent(
                event_id=f"evt_{int(time.time() * 1000000)}",
                event_type=EventType(event_type),
                timestamp=time.time(),
                data=data,
                source=source,
                priority=ProcessingPriority[priority.upper()]
            )
            
            # Determine processing path
            use_ultra_low_latency = (
                enable_ultra_low_latency if enable_ultra_low_latency is not None 
                else self.config.enable_ultra_low_latency and 
                     stream_event.priority in [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH]
            )
            
            # Process with ultra-low latency if enabled
            if use_ultra_low_latency and self.low_latency_processor:
                success = await process_ultra_low_latency(stream_event, "streaming-agent")
                if success:
                    processing_time_ms = (time.perf_counter() - start_time) * 1000
                    self._update_metrics(processing_time_ms)
                    
                    return {
                        'status': 'processed',
                        'processing_path': 'ultra_low_latency',
                        'processing_time_ms': processing_time_ms,
                        'event_id': stream_event.event_id
                    }
            
            # Standard processing pipeline
            results = await self._process_standard_pipeline(stream_event)
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(processing_time_ms)
            
            return {
                'status': 'processed',
                'processing_path': 'standard_pipeline',
                'processing_time_ms': processing_time_ms,
                'event_id': stream_event.event_id,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error processing stream event: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'processing_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    async def _process_standard_pipeline(self, event: StreamEvent) -> Dict[str, Any]:
        """Process event through standard pipeline with all optimizations"""
        results = {
            'anomaly_detection': None,
            'edge_optimization': None,
            'pipeline_processing': None,
            'inference_results': None
        }
        
        # 1. Anomaly detection
        if self.config.enable_anomaly_detection:
            anomaly_results = await detect_stream_anomalies(event)
            results['anomaly_detection'] = {
                'anomalies_found': len(anomaly_results),
                'results': [result.to_dict() for result in anomaly_results]
            }
            
            if anomaly_results:
                self.metrics['anomalies_detected'] += len(anomaly_results)
        
        # 2. Edge optimization
        if self.config.enable_edge_optimization:
            optimization_result = await optimize_stream_placement(event)
            results['edge_optimization'] = optimization_result.to_dict()
            self.metrics['edge_optimizations'] += 1
        
        # 3. Pipeline processing
        pipeline_result = await process_event_through_pipeline("main-streaming", event)
        results['pipeline_processing'] = {
            'success': pipeline_result.success,
            'stages_processed': pipeline_result.processed_stages,
            'processing_time_ms': pipeline_result.processing_time_ms
        }
        
        # 4. Edge inference (if applicable)
        if (self.config.enable_edge_inference and 
            hasattr(event.data, 'inference_request') and 
            event.data.inference_request):
            
            inference_result = await self._run_edge_inference(event)
            results['inference_results'] = inference_result
        
        return results
    
    async def _run_edge_inference(self, event: StreamEvent) -> Dict[str, Any]:
        """Run edge inference for an event"""
        try:
            # Extract inference request from event data
            inference_request = event.data.get('inference_request', {})
            model_id = inference_request.get('model_id', 'default')
            input_data = inference_request.get('input_data')
            
            if not input_data:
                return {'status': 'failed', 'error': 'No input data provided'}
            
            # Run optimized inference
            result = await optimize_inference_request(
                model_id=model_id,
                input_data=input_data,
                inference_func=lambda data: run_edge_inference(model_id, data)
            )
            
            return {
                'status': 'completed',
                'model_id': model_id,
                'success': result.success,
                'output_data': result.output_data,
                'inference_time_ms': result.inference_time_ms,
                'optimizations_applied': result.optimization_applied
            }
            
        except Exception as e:
            logger.error(f"Edge inference error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _handle_optimization_stage(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Custom optimization stage handler"""
        optimized_events = []
        
        for event in events:
            try:
                # Apply edge optimization
                if self.config.enable_edge_optimization:
                    optimization_result = await optimize_stream_placement(event)
                    
                    # Add optimization metadata to event
                    event.metadata['edge_optimization'] = {
                        'recommended_location': optimization_result.recommended_location.value,
                        'confidence': optimization_result.confidence,
                        'expected_latency_ms': optimization_result.expected_latency_ms
                    }
                
                optimized_events.append(event)
                
            except Exception as e:
                logger.error(f"Optimization stage error for event {event.event_id}: {e}")
                optimized_events.append(event)  # Pass through on error
        
        return optimized_events
    
    async def _handle_ultra_low_latency_event(self, event: StreamEvent) -> None:
        """Handler for ultra-low latency events"""
        try:
            # Minimal processing for ultra-low latency
            logger.debug(f"Ultra-low latency processing: {event.event_id}")
            
            # Call registered event handlers
            for handler in self.event_handlers['stream_event']:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
            
        except Exception as e:
            logger.error(f"Ultra-low latency handler error: {e}")
    
    async def _handle_anomaly_alert(self, anomaly_result) -> None:
        """Handle anomaly detection alerts"""
        try:
            logger.warning(
                f"ANOMALY DETECTED: {anomaly_result.anomaly_type.value} "
                f"(severity: {anomaly_result.severity.value}, "
                f"confidence: {anomaly_result.confidence:.3f})"
            )
            
            # Call registered anomaly handlers
            for handler in self.event_handlers['anomaly_detected']:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(anomaly_result)
                    else:
                        handler(anomaly_result)
                except Exception as e:
                    logger.error(f"Anomaly handler error: {e}")
            
            # Update metrics
            metrics_collector.streaming_anomalies_detected.labels(
                agent_id=self.agent_id,
                anomaly_type=anomaly_result.anomaly_type.value,
                severity=anomaly_result.severity.value
            ).inc()
            
        except Exception as e:
            logger.error(f"Anomaly alert handler error: {e}")
    
    def _update_metrics(self, processing_time_ms: float) -> None:
        """Update agent metrics"""
        self.metrics['events_processed'] += 1
        self.metrics['total_processing_time_ms'] += processing_time_ms
        self.metrics['avg_processing_latency_ms'] = (
            self.metrics['total_processing_time_ms'] / self.metrics['events_processed']
        )
        
        # Update Prometheus metrics
        metrics_collector.streaming_events_processed.labels(
            agent_id=self.agent_id
        ).inc()
        
        metrics_collector.streaming_processing_latency.labels(
            agent_id=self.agent_id
        ).observe(processing_time_ms / 1000.0)
    
    async def load_edge_model(self, 
                            model_id: str,
                            model_path: str,
                            model_format: str = "onnx",
                            compress_model: bool = True) -> Dict[str, Any]:
        """Load and optionally compress a model for edge inference"""
        try:
            # Compress model if requested
            if compress_model:
                compression_result = await compress_model_for_edge(
                    model_path=model_path,
                    technique="quantization",
                    target_format=model_format
                )
                
                if compression_result.success:
                    model_path = compression_result.output_path
                    self.metrics['models_compressed'] += 1
                    logger.info(f"Model compressed: {compression_result.compression_ratio:.3f} ratio")
            
            # Load model into edge runtime
            success = await load_edge_model(
                model_id=model_id,
                model_path=model_path,
                format=model_format
            )
            
            if success:
                logger.info(f"Loaded edge model: {model_id}")
                return {
                    'status': 'loaded',
                    'model_id': model_id,
                    'compressed': compress_model,
                    'format': model_format
                }
            else:
                return {'status': 'failed', 'error': 'Failed to load model'}
                
        except Exception as e:
            logger.error(f"Error loading edge model {model_id}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Registered {event_type} handler")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the streaming agent"""
        status = {
            'agent_id': self.agent_id,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'config': asdict(self.config),
            'metrics': self.metrics
        }
        
        # Add component statuses
        if self.is_initialized:
            status['components'] = {
                'stream_processor': await get_streaming_metrics(),
                'low_latency_processor': get_all_low_latency_metrics(),
                'edge_optimizer': get_optimization_status(),
                'anomaly_detection': self.anomaly_manager.get_all_metrics() if self.anomaly_manager else {},
                'edge_inference': get_edge_runtime_metrics(),
                'inference_optimizer': get_inference_optimization_stats(),
                'model_compression': get_compression_statistics(),
                'cloud_sync': get_sync_status()
            }
        
        return status
    
    async def start(self) -> Dict[str, Any]:
        """Start the streaming agent"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            return {'status': 'already_running'}
        
        try:
            # Start all components
            if self.stream_processor:
                # Stream processor is started during initialization
                pass
            
            if self.low_latency_processor:
                await self.low_latency_processor.start()
            
            if self.anomaly_manager:
                await self.anomaly_manager.start()
            
            # Start pipeline manager
            await pipeline_manager.start_all()
            
            self.is_running = True
            
            logger.info("StreamingAgent started successfully")
            return {
                'status': 'started',
                'agent_id': self.agent_id,
                'components_running': {
                    'stream_processor': True,
                    'low_latency_processor': self.low_latency_processor is not None,
                    'anomaly_detection': self.anomaly_manager is not None,
                    'edge_inference': self.edge_runtime is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start StreamingAgent: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def stop(self) -> Dict[str, Any]:
        """Stop the streaming agent"""
        if not self.is_running:
            return {'status': 'not_running'}
        
        try:
            logger.info("Stopping StreamingAgent")
            
            # Stop all components
            if self.low_latency_processor:
                await self.low_latency_processor.stop()
            
            if self.anomaly_manager:
                await self.anomaly_manager.stop()
            
            if self.edge_runtime:
                await self.edge_runtime.stop()
            
            if self.sync_manager:
                await self.sync_manager.stop()
            
            # Stop pipeline manager
            await pipeline_manager.stop_all()
            
            self.is_running = False
            
            logger.info("StreamingAgent stopped successfully")
            return {'status': 'stopped'}
            
        except Exception as e:
            logger.error(f"Error stopping StreamingAgent: {e}")
            return {'status': 'error', 'error': str(e)}

# Global streaming agent instance
streaming_agent = None

# Convenience functions
async def initialize_streaming_agent(config: Optional[Dict[str, Any]] = None) -> StreamingAgent:
    """Initialize global streaming agent"""
    global streaming_agent
    
    if config is None:
        config = {}
    
    agent_config = StreamingAgentConfig(**config)
    streaming_agent = StreamingAgent(agent_config)
    
    await streaming_agent.initialize()
    await streaming_agent.start()
    
    return streaming_agent

async def process_streaming_event(event_type: str,
                                data: Any,
                                source: str = "external",
                                priority: str = "normal") -> Dict[str, Any]:
    """Process a streaming event through the agent"""
    if streaming_agent is None:
        await initialize_streaming_agent()
    
    return await streaming_agent.process_stream_event(event_type, data, source, priority)

def get_streaming_agent_status() -> Dict[str, Any]:
    """Get streaming agent status"""
    if streaming_agent is None:
        return {'status': 'not_initialized'}
    
    return asyncio.create_task(streaming_agent.get_comprehensive_status())

async def load_streaming_model(model_id: str, model_path: str, **kwargs) -> Dict[str, Any]:
    """Load model for streaming inference"""
    if streaming_agent is None:
        await initialize_streaming_agent()
    
    return await streaming_agent.load_edge_model(model_id, model_path, **kwargs)
