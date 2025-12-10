# backend/streaming/event_pipeline.py
"""
Event Processing Pipeline
Complex event processing with configurable pipeline stages and transformations.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, AsyncIterator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import uuid
import inspect

# Import stream processor components
from .stream_processor import StreamEvent, EventType, ProcessingPriority, StreamMetrics
from ..monitoring.logs.logger import get_logger
from ..monitoring.metrics.prometheus_metrics import metrics_collector

logger = get_logger(__name__)

class PipelineStageType(Enum):
    """Types of pipeline stages"""
    FILTER = "filter"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    VALIDATE = "validate"
    ROUTE = "route"
    STORE = "store"
    ALERT = "alert"
    CUSTOM = "custom"

class AggregationFunction(Enum):
    """Aggregation functions for event data"""
    COUNT = "count"
    SUM = "sum"
    AVG = "average"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"
    DISTINCT_COUNT = "distinct_count"
    PERCENTILE = "percentile"
    STDDEV = "stddev"

@dataclass
class PipelineStage:
    """Configuration for a pipeline stage"""
    stage_id: str
    stage_type: PipelineStageType
    name: str
    description: str = ""
    enabled: bool = True
    config: Dict[str, Any] = None
    timeout_ms: int = 1000
    retry_count: int = 3
    fallback_action: str = "skip"  # skip, stop, retry
    metrics_enabled: bool = True
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

@dataclass
class PipelineConfig:
    """Configuration for event processing pipeline"""
    pipeline_id: str
    name: str
    description: str = ""
    stages: List[PipelineStage] = None
    parallel_execution: bool = False
    max_concurrent_events: int = 100
    buffer_size: int = 1000
    enable_checkpointing: bool = True
    checkpoint_interval_ms: int = 5000
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = []

@dataclass
class EventProcessingResult:
    """Result of event processing through pipeline"""
    event_id: str
    pipeline_id: str
    success: bool
    processed_stages: List[str]
    failed_stage: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    output_events: List[StreamEvent] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.output_events is None:
            self.output_events = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AggregationWindow:
    """Configuration for time-based aggregation windows"""
    window_id: str
    window_type: str  # tumbling, sliding, session
    size_ms: int
    slide_ms: Optional[int] = None  # For sliding windows
    session_timeout_ms: Optional[int] = None  # For session windows
    aggregation_functions: List[AggregationFunction] = None
    group_by_fields: List[str] = None
    
    def __post_init__(self):
        if self.aggregation_functions is None:
            self.aggregation_functions = [AggregationFunction.COUNT]
        if self.group_by_fields is None:
            self.group_by_fields = []

class EventPipeline:
    """Complex event processing pipeline with configurable stages"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline_id = config.pipeline_id
        self.stages = {stage.stage_id: stage for stage in config.stages}
        self.stage_handlers: Dict[str, Callable] = {}
        self.is_running = False
        
        # Processing state
        self.input_buffer = deque(maxlen=config.buffer_size)
        self.processing_queue = asyncio.Queue(maxsize=config.max_concurrent_events)
        self.checkpoint_data = {}
        self.last_checkpoint = time.time()
        
        # Metrics
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'total_processing_time_ms': 0.0,
            'avg_processing_time_ms': 0.0,
            'stages_executed': defaultdict(int),
            'stage_processing_times': defaultdict(list),
            'pipeline_throughput': 0.0
        }
        
        # Aggregation windows
        self.aggregation_windows: Dict[str, AggregationWindow] = {}
        self.window_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Processing tasks
        self.processing_tasks = []
        
        logger.info(f"EventPipeline {self.pipeline_id} initialized with {len(self.stages)} stages")
    
    async def start(self) -> None:
        """Start the event pipeline"""
        if self.is_running:
            return
        
        logger.info(f"Starting EventPipeline {self.pipeline_id}")
        self.is_running = True
        
        # Start processing workers
        for i in range(min(4, self.config.max_concurrent_events // 10)):
            task = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.processing_tasks.append(task)
        
        # Start checkpoint task if enabled
        if self.config.enable_checkpointing:
            checkpoint_task = asyncio.create_task(self._checkpoint_worker())
            self.processing_tasks.append(checkpoint_task)
        
        # Start aggregation window maintenance
        aggregation_task = asyncio.create_task(self._aggregation_maintenance())
        self.processing_tasks.append(aggregation_task)
        
        logger.info(f"EventPipeline {self.pipeline_id} started with {len(self.processing_tasks)} workers")
    
    async def process_event(self, event: StreamEvent) -> EventProcessingResult:
        """Process a single event through the pipeline"""
        start_time = time.perf_counter()
        
        result = EventProcessingResult(
            event_id=event.event_id,
            pipeline_id=self.pipeline_id,
            success=False,
            processed_stages=[]
        )
        
        try:
            # Add to processing queue
            await self.processing_queue.put((event, result))
            
            # The actual processing happens in background workers
            # For immediate response, we return a pending result
            # In production, this might use a callback mechanism
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            result.processing_time_ms = processing_time_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            result.error_message = str(e)
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result
    
    async def _processing_worker(self, worker_id: str) -> None:
        """Background worker for processing events through pipeline stages"""
        logger.info(f"Starting pipeline worker: {worker_id}")
        
        while self.is_running:
            try:
                # Get next event from queue
                event, result = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process through pipeline stages
                await self._execute_pipeline(event, result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pipeline worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_pipeline(self, event: StreamEvent, result: EventProcessingResult) -> None:
        """Execute event through all pipeline stages"""
        current_events = [event]
        stage_start_time = time.perf_counter()
        
        try:
            # Process through each stage in order
            for stage in self.config.stages:
                if not stage.enabled:
                    continue
                
                stage_start = time.perf_counter()
                
                try:
                    # Process events through this stage
                    stage_output = await self._execute_stage(stage, current_events)
                    
                    # Update current events for next stage
                    current_events = stage_output if stage_output else []
                    
                    # Record stage completion
                    result.processed_stages.append(stage.stage_id)
                    
                    # Update stage metrics
                    stage_time_ms = (time.perf_counter() - stage_start) * 1000
                    self.metrics['stages_executed'][stage.stage_id] += 1
                    self.metrics['stage_processing_times'][stage.stage_id].append(stage_time_ms)
                    
                    # If no events remain, pipeline is complete
                    if not current_events:
                        break
                        
                except Exception as e:
                    logger.error(f"Stage {stage.stage_id} failed: {e}")
                    result.failed_stage = stage.stage_id
                    result.error_message = str(e)
                    
                    # Handle stage failure based on fallback action
                    if stage.fallback_action == "stop":
                        break
                    elif stage.fallback_action == "skip":
                        continue
                    elif stage.fallback_action == "retry":
                        # Implement retry logic
                        for retry in range(stage.retry_count):
                            try:
                                stage_output = await self._execute_stage(stage, current_events)
                                current_events = stage_output if stage_output else []
                                result.processed_stages.append(f"{stage.stage_id}-retry-{retry}")
                                break
                            except Exception as retry_e:
                                if retry == stage.retry_count - 1:
                                    logger.error(f"Stage {stage.stage_id} failed after {stage.retry_count} retries: {retry_e}")
                                    break
                                await asyncio.sleep(0.01 * (retry + 1))  # Exponential backoff
            
            # Set output events
            result.output_events = current_events
            result.success = len(result.processed_stages) > 0 and result.failed_stage is None
            
            # Update pipeline metrics
            total_time_ms = (time.perf_counter() - stage_start_time) * 1000
            result.processing_time_ms = total_time_ms
            
            self.metrics['events_processed'] += 1
            self.metrics['total_processing_time_ms'] += total_time_ms
            self.metrics['avg_processing_time_ms'] = (
                self.metrics['total_processing_time_ms'] / self.metrics['events_processed']
            )
            
            if not result.success:
                self.metrics['events_failed'] += 1
            
            # Update Prometheus metrics
            metrics_collector.pipeline_events_processed.labels(
                pipeline_id=self.pipeline_id
            ).inc()
            
            metrics_collector.pipeline_processing_latency.labels(
                pipeline_id=self.pipeline_id
            ).observe(total_time_ms / 1000.0)
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            result.success = False
            result.error_message = str(e)
    
    async def _execute_stage(self, stage: PipelineStage, events: List[StreamEvent]) -> List[StreamEvent]:
        """Execute a single pipeline stage"""
        if not events:
            return []
        
        # Get stage handler
        handler = self.stage_handlers.get(stage.stage_id)
        if not handler:
            handler = self._get_default_stage_handler(stage)
        
        # Execute stage with timeout
        try:
            result = await asyncio.wait_for(
                handler(events, stage.config),
                timeout=stage.timeout_ms / 1000.0
            )
            
            return result if isinstance(result, list) else [result] if result else []
            
        except asyncio.TimeoutError:
            raise Exception(f"Stage {stage.stage_id} timed out after {stage.timeout_ms}ms")
    
    def _get_default_stage_handler(self, stage: PipelineStage) -> Callable:
        """Get default handler for stage type"""
        if stage.stage_type == PipelineStageType.FILTER:
            return self._default_filter_handler
        elif stage.stage_type == PipelineStageType.TRANSFORM:
            return self._default_transform_handler
        elif stage.stage_type == PipelineStageType.AGGREGATE:
            return self._default_aggregate_handler
        elif stage.stage_type == PipelineStageType.ENRICH:
            return self._default_enrich_handler
        elif stage.stage_type == PipelineStageType.VALIDATE:
            return self._default_validate_handler
        elif stage.stage_type == PipelineStageType.ROUTE:
            return self._default_route_handler
        else:
            return self._default_passthrough_handler
    
    async def _default_filter_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default filter stage handler"""
        filtered_events = []
        
        for event in events:
            # Apply filter conditions
            include_event = True
            
            # Event type filter
            if 'event_types' in config:
                allowed_types = [EventType(et) for et in config['event_types']]
                if event.event_type not in allowed_types:
                    include_event = False
            
            # Data field filters
            if 'field_filters' in config and include_event:
                for field_filter in config['field_filters']:
                    field_name = field_filter.get('field')
                    operator = field_filter.get('operator', '==')
                    value = field_filter.get('value')
                    
                    if field_name and hasattr(event.data, field_name):
                        field_value = getattr(event.data, field_name)
                        
                        if operator == '==' and field_value != value:
                            include_event = False
                        elif operator == '!=' and field_value == value:
                            include_event = False
                        elif operator == '>' and field_value <= value:
                            include_event = False
                        elif operator == '<' and field_value >= value:
                            include_event = False
            
            if include_event:
                filtered_events.append(event)
        
        return filtered_events
    
    async def _default_transform_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default transform stage handler"""
        transformed_events = []
        
        for event in events:
            # Create a copy of the event for transformation
            transformed_event = StreamEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                data=event.data,
                source=event.source,
                priority=event.priority,
                metadata=event.metadata.copy() if event.metadata else {},
                correlation_id=event.correlation_id,
                ttl_ms=event.ttl_ms
            )
            
            # Apply transformations
            if 'field_mappings' in config:
                for mapping in config['field_mappings']:
                    from_field = mapping.get('from')
                    to_field = mapping.get('to')
                    transform_func = mapping.get('function')
                    
                    if from_field and to_field:
                        # Simple field mapping
                        if hasattr(transformed_event.data, from_field):
                            value = getattr(transformed_event.data, from_field)
                            
                            # Apply transformation function if specified
                            if transform_func == 'upper':
                                value = str(value).upper()
                            elif transform_func == 'lower':
                                value = str(value).lower()
                            elif transform_func == 'round':
                                value = round(float(value), 2)
                            
                            setattr(transformed_event.data, to_field, value)
            
            # Add processing metadata
            transformed_event.metadata['transformed_at'] = time.time()
            transformed_event.metadata['pipeline_id'] = self.pipeline_id
            
            transformed_events.append(transformed_event)
        
        return transformed_events
    
    async def _default_aggregate_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default aggregation stage handler"""
        if not events:
            return []
        
        # Group events by aggregation key
        groups = defaultdict(list)
        group_by_field = config.get('group_by', 'source')
        
        for event in events:
            if hasattr(event, group_by_field):
                key = getattr(event, group_by_field)
            else:
                key = 'default'
            groups[key].append(event)
        
        # Aggregate each group
        aggregated_events = []
        aggregation_func = config.get('function', 'count')
        
        for group_key, group_events in groups.items():
            # Perform aggregation
            if aggregation_func == 'count':
                result_data = {'count': len(group_events), 'group': group_key}
            elif aggregation_func == 'sum':
                value_field = config.get('value_field', 'value')
                total = sum(getattr(event.data, value_field, 0) for event in group_events)
                result_data = {'sum': total, 'group': group_key, 'count': len(group_events)}
            else:
                result_data = {'count': len(group_events), 'group': group_key}
            
            # Create aggregated event
            aggregated_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.AGGREGATION,
                timestamp=time.time(),
                data=result_data,
                source=f"pipeline-{self.pipeline_id}",
                priority=ProcessingPriority.NORMAL,
                metadata={
                    'aggregation_function': aggregation_func,
                    'group_key': group_key,
                    'original_event_count': len(group_events),
                    'pipeline_id': self.pipeline_id
                }
            )
            
            aggregated_events.append(aggregated_event)
        
        return aggregated_events
    
    async def _default_enrich_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default enrichment stage handler"""
        enriched_events = []
        
        for event in events:
            # Create enriched copy
            enriched_event = StreamEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                data=event.data,
                source=event.source,
                priority=event.priority,
                metadata=event.metadata.copy() if event.metadata else {},
                correlation_id=event.correlation_id,
                ttl_ms=event.ttl_ms
            )
            
            # Add enrichment data
            enrichments = config.get('enrichments', {})
            for key, value in enrichments.items():
                enriched_event.metadata[key] = value
            
            # Add timestamp enrichments
            enriched_event.metadata['enriched_at'] = time.time()
            enriched_event.metadata['processing_pipeline'] = self.pipeline_id
            
            enriched_events.append(enriched_event)
        
        return enriched_events
    
    async def _default_validate_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default validation stage handler"""
        valid_events = []
        
        for event in events:
            is_valid = True
            validation_errors = []
            
            # Validate required fields
            required_fields = config.get('required_fields', [])
            for field in required_fields:
                if not hasattr(event.data, field) or getattr(event.data, field) is None:
                    is_valid = False
                    validation_errors.append(f"Missing required field: {field}")
            
            # Validate data types
            type_validations = config.get('type_validations', {})
            for field, expected_type in type_validations.items():
                if hasattr(event.data, field):
                    value = getattr(event.data, field)
                    if not isinstance(value, eval(expected_type)):
                        is_valid = False
                        validation_errors.append(f"Field {field} should be {expected_type}")
            
            if is_valid:
                valid_events.append(event)
            else:
                logger.warning(f"Event {event.event_id} validation failed: {validation_errors}")
        
        return valid_events
    
    async def _default_route_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default routing stage handler"""
        # Simple routing based on event type or data conditions
        routed_events = []
        
        for event in events:
            # Add routing metadata
            event.metadata['routed_by'] = self.pipeline_id
            event.metadata['route_timestamp'] = time.time()
            
            # Route based on conditions
            routing_rules = config.get('rules', [])
            for rule in routing_rules:
                condition = rule.get('condition', {})
                target = rule.get('target', 'default')
                
                # Simple condition checking
                if self._evaluate_routing_condition(event, condition):
                    event.metadata['route_target'] = target
                    break
            
            routed_events.append(event)
        
        return routed_events
    
    def _evaluate_routing_condition(self, event: StreamEvent, condition: Dict[str, Any]) -> bool:
        """Evaluate routing condition for an event"""
        if 'event_type' in condition:
            if event.event_type.value != condition['event_type']:
                return False
        
        if 'source' in condition:
            if event.source != condition['source']:
                return False
        
        return True
    
    async def _default_passthrough_handler(self, events: List[StreamEvent], config: Dict[str, Any]) -> List[StreamEvent]:
        """Default passthrough handler - no processing"""
        return events
    
    def register_stage_handler(self, stage_id: str, handler: Callable) -> None:
        """Register custom handler for a pipeline stage"""
        self.stage_handlers[stage_id] = handler
        logger.info(f"Registered custom handler for stage: {stage_id}")
    
    def add_aggregation_window(self, window: AggregationWindow) -> None:
        """Add time-based aggregation window"""
        self.aggregation_windows[window.window_id] = window
        logger.info(f"Added aggregation window: {window.window_id}")
    
    async def _checkpoint_worker(self) -> None:
        """Background worker for creating processing checkpoints"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_ms / 1000.0)
                
                # Create checkpoint
                checkpoint = {
                    'timestamp': time.time(),
                    'pipeline_id': self.pipeline_id,
                    'metrics': self.metrics.copy(),
                    'queue_size': self.processing_queue.qsize(),
                    'buffer_size': len(self.input_buffer)
                }
                
                self.checkpoint_data[int(time.time())] = checkpoint
                self.last_checkpoint = time.time()
                
                # Clean old checkpoints (keep last 10)
                if len(self.checkpoint_data) > 10:
                    oldest_key = min(self.checkpoint_data.keys())
                    del self.checkpoint_data[oldest_key]
                
            except Exception as e:
                logger.error(f"Checkpoint worker error: {e}")
                await asyncio.sleep(5.0)
    
    async def _aggregation_maintenance(self) -> None:
        """Maintain aggregation windows and process expired windows"""
        while self.is_running:
            try:
                current_time = time.time() * 1000  # Convert to milliseconds
                
                for window_id, window in self.aggregation_windows.items():
                    # Process tumbling windows
                    if window.window_type == 'tumbling':
                        await self._process_tumbling_window(window, current_time)
                    # Add other window types as needed
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Aggregation maintenance error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_tumbling_window(self, window: AggregationWindow, current_time: float) -> None:
        """Process tumbling aggregation window"""
        # Implementation would aggregate events within time windows
        # This is a simplified version
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline processing metrics"""
        return {
            'pipeline_id': self.pipeline_id,
            'is_running': self.is_running,
            'stage_count': len(self.stages),
            'queue_size': self.processing_queue.qsize(),
            'buffer_size': len(self.input_buffer),
            'last_checkpoint': self.last_checkpoint,
            **self.metrics
        }
    
    async def stop(self) -> None:
        """Stop the event pipeline"""
        logger.info(f"Stopping EventPipeline {self.pipeline_id}")
        
        self.is_running = False
        
        # Wait for processing tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info(f"EventPipeline {self.pipeline_id} stopped")

class PipelineManager:
    """Manager for multiple event processing pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, EventPipeline] = {}
        self.is_running = False
    
    async def create_pipeline(self, config: PipelineConfig) -> EventPipeline:
        """Create and register a new pipeline"""
        pipeline = EventPipeline(config)
        self.pipelines[config.pipeline_id] = pipeline
        
        if self.is_running:
            await pipeline.start()
        
        logger.info(f"Created pipeline: {config.pipeline_id}")
        return pipeline
    
    async def start_all(self) -> None:
        """Start all registered pipelines"""
        self.is_running = True
        
        for pipeline in self.pipelines.values():
            await pipeline.start()
        
        logger.info(f"Started {len(self.pipelines)} pipelines")
    
    async def stop_all(self) -> None:
        """Stop all pipelines"""
        self.is_running = False
        
        for pipeline in self.pipelines.values():
            await pipeline.stop()
        
        logger.info("All pipelines stopped")
    
    def get_pipeline(self, pipeline_id: str) -> Optional[EventPipeline]:
        """Get pipeline by ID"""
        return self.pipelines.get(pipeline_id)
    
    def list_pipelines(self) -> List[str]:
        """List all pipeline IDs"""
        return list(self.pipelines.keys())
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all pipelines"""
        return {
            pipeline_id: pipeline.get_metrics()
            for pipeline_id, pipeline in self.pipelines.items()
        }

# Global pipeline manager instance
pipeline_manager = PipelineManager()

# Convenience functions
async def create_event_pipeline(pipeline_id: str, stages: List[Dict[str, Any]]) -> EventPipeline:
    """Create an event processing pipeline"""
    stage_objects = []
    for stage_config in stages:
        stage = PipelineStage(
            stage_id=stage_config['stage_id'],
            stage_type=PipelineStageType(stage_config['stage_type']),
            name=stage_config['name'],
            description=stage_config.get('description', ''),
            config=stage_config.get('config', {})
        )
        stage_objects.append(stage)
    
    config = PipelineConfig(
        pipeline_id=pipeline_id,
        name=f"Pipeline {pipeline_id}",
        stages=stage_objects
    )
    
    return await pipeline_manager.create_pipeline(config)

async def process_event_through_pipeline(pipeline_id: str, event: StreamEvent) -> EventProcessingResult:
    """Process event through specified pipeline"""
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise ValueError(f"Pipeline {pipeline_id} not found")
    
    return await pipeline.process_event(event)
