# File: backend/monitoring/tracing/distributed_tracing.py
"""
Distributed tracing implementation for UAP platform using OpenTelemetry.
Provides request tracing across microservices, frameworks, and external dependencies.
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from contextvars import ContextVar
from functools import wraps
import uuid
import json
from datetime import datetime

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader

from ..logs.logger import uap_logger, EventType, LogLevel

# Context variables for trace correlation
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class TracingConfig:
    """Configuration for distributed tracing"""
    
    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "uap-backend")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "3.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Jaeger configuration
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        self.jaeger_agent_host = os.getenv("JAEGER_AGENT_HOST", "localhost")
        self.jaeger_agent_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
        
        # OTLP configuration
        self.otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        
        # Sampling configuration
        self.sampling_rate = float(os.getenv("OTEL_SAMPLING_RATE", "0.1"))  # 10% sampling
        self.force_trace_components = os.getenv("FORCE_TRACE_COMPONENTS", "agent,framework").split(",")
        
        # Enable/disable specific instrumentations
        self.enable_fastapi = os.getenv("OTEL_ENABLE_FASTAPI", "true").lower() == "true"
        self.enable_httpx = os.getenv("OTEL_ENABLE_HTTPX", "true").lower() == "true"
        self.enable_psycopg2 = os.getenv("OTEL_ENABLE_PSYCOPG2", "true").lower() == "true"
        self.enable_redis = os.getenv("OTEL_ENABLE_REDIS", "true").lower() == "true"

class DistributedTracer:
    """Main distributed tracing manager"""
    
    def __init__(self, config: TracingConfig = None):
        self.config = config or TracingConfig()
        self.tracer_provider = None
        self.tracer = None
        self.meter_provider = None
        self.meter = None
        self.initialized = False
        
        # Custom metrics for tracing performance
        self.span_counter = None
        self.span_duration_histogram = None
        self.trace_error_counter = None
    
    def initialize(self):
        """Initialize distributed tracing"""
        if self.initialized:
            return
        
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
                "host.name": os.getenv("HOSTNAME", "localhost"),
                "service.instance.id": str(uuid.uuid4())
            })
            
            # Configure tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Add span processors
            self._add_span_processors()
            
            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(
                self.config.service_name,
                self.config.service_version
            )
            
            # Configure metrics
            self._configure_metrics(resource)
            
            # Set propagator
            set_global_textmap(B3MultiFormat())
            
            # Initialize instrumentations
            self._initialize_instrumentations()
            
            self.initialized = True
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Distributed tracing initialized",
                EventType.SYSTEM,
                {
                    "service_name": self.config.service_name,
                    "jaeger_endpoint": self.config.jaeger_endpoint,
                    "sampling_rate": self.config.sampling_rate
                },
                "tracing"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize distributed tracing: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "tracing"
            )
            raise
    
    def _add_span_processors(self):
        """Add span processors for different exporters"""
        # Jaeger exporter
        if self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
                collector_endpoint=self.config.jaeger_endpoint
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        
        # OTLP exporter
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True  # Use secure=False for development
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Console exporter for development
        if self.config.environment == "development":
            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
    
    def _configure_metrics(self, resource: Resource):
        """Configure OpenTelemetry metrics"""
        # Create metric reader for Prometheus
        prometheus_reader = PrometheusMetricReader()
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(
            self.config.service_name,
            self.config.service_version
        )
        
        # Create custom metrics
        self.span_counter = self.meter.create_counter(
            "uap_spans_total",
            description="Total number of spans created",
            unit="1"
        )
        
        self.span_duration_histogram = self.meter.create_histogram(
            "uap_span_duration_seconds",
            description="Duration of spans in seconds",
            unit="s"
        )
        
        self.trace_error_counter = self.meter.create_counter(
            "uap_trace_errors_total",
            description="Total number of trace errors",
            unit="1"
        )
    
    def _initialize_instrumentations(self):
        """Initialize automatic instrumentations"""
        try:
            # FastAPI instrumentation
            if self.config.enable_fastapi:
                FastAPIInstrumentor().instrument()
            
            # HTTP client instrumentation
            if self.config.enable_httpx:
                HTTPXClientInstrumentor().instrument()
            
            # Database instrumentation
            if self.config.enable_psycopg2:
                Psycopg2Instrumentor().instrument()
            
            # Redis instrumentation
            if self.config.enable_redis:
                RedisInstrumentor().instrument()
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Auto-instrumentation initialized",
                EventType.SYSTEM,
                {
                    "fastapi": self.config.enable_fastapi,
                    "httpx": self.config.enable_httpx,
                    "psycopg2": self.config.enable_psycopg2,
                    "redis": self.config.enable_redis
                },
                "tracing"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Some instrumentations failed to initialize: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "tracing"
            )
    
    def start_span(self, name: str, kind: trace.SpanKind = trace.SpanKind.INTERNAL,
                   attributes: Dict[str, Any] = None, parent_context=None) -> trace.Span:
        """Start a new span"""
        if not self.initialized:
            self.initialize()
        
        # Determine if we should force trace this span
        force_trace = any(component in name.lower() for component in self.config.force_trace_components)
        
        # Create span with attributes
        span_attributes = {
            "service.name": self.config.service_name,
            "service.version": self.config.service_version
        }
        
        if attributes:
            span_attributes.update(attributes)
        
        # Start span
        span = self.tracer.start_span(
            name=name,
            kind=kind,
            attributes=span_attributes,
            context=parent_context
        )
        
        # Update context variables
        trace_id = f"{span.get_span_context().trace_id:032x}"
        span_id = f"{span.get_span_context().span_id:016x}"
        
        trace_id_var.set(trace_id)
        span_id_var.set(span_id)
        
        # Record metrics
        if self.span_counter:
            self.span_counter.add(1, {"span_name": name, "service": self.config.service_name})
        
        return span
    
    def trace_function(self, name: str = None, attributes: Dict[str, Any] = None,
                      record_exception: bool = True):
        """Decorator to trace function execution"""
        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.start_span(span_name, attributes=attributes) as span:
                    start_time = time.time()
                    
                    try:
                        # Add function-specific attributes
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        if attributes:
                            for key, value in attributes.items():
                                span.set_attribute(key, value)
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                    
                    except Exception as e:
                        # Record exception
                        if record_exception:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Record error metric
                        if self.trace_error_counter:
                            self.trace_error_counter.add(1, {
                                "span_name": span_name,
                                "error_type": type(e).__name__
                            })
                        
                        raise
                    
                    finally:
                        # Record span duration
                        duration = time.time() - start_time
                        if self.span_duration_histogram:
                            self.span_duration_histogram.record(duration, {"span_name": span_name})
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.start_span(span_name, attributes=attributes) as span:
                    start_time = time.time()
                    
                    try:
                        # Add function-specific attributes
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        if attributes:
                            for key, value in attributes.items():
                                span.set_attribute(key, value)
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                    
                    except Exception as e:
                        # Record exception
                        if record_exception:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Record error metric
                        if self.trace_error_counter:
                            self.trace_error_counter.add(1, {
                                "span_name": span_name,
                                "error_type": type(e).__name__
                            })
                        
                        raise
                    
                    finally:
                        # Record span duration
                        duration = time.time() - start_time
                        if self.span_duration_histogram:
                            self.span_duration_histogram.record(duration, {"span_name": span_name})
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def trace_agent_request(self, agent_id: str, framework: str, message_type: str = "user_message"):
        """Decorator specifically for tracing agent requests"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                span_name = f"agent.{framework}.{func.__name__}"
                
                attributes = {
                    "agent.id": agent_id,
                    "agent.framework": framework,
                    "agent.message_type": message_type,
                    "component": "agent"
                }
                
                with self.start_span(span_name, trace.SpanKind.SERVER, attributes) as span:
                    start_time = time.time()
                    
                    try:
                        # Execute agent request
                        result = await func(*args, **kwargs)
                        
                        # Add result attributes
                        if isinstance(result, dict):
                            if "response_time_ms" in result:
                                span.set_attribute("agent.response_time_ms", result["response_time_ms"])
                            if "message_count" in result:
                                span.set_attribute("agent.message_count", result["message_count"])
                            if "token_count" in result:
                                span.set_attribute("agent.token_count", result["token_count"])
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                    
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Add error attributes
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        
                        raise
                    
                    finally:
                        # Record duration
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", duration_ms)
            
            return wrapper
        return decorator
    
    def trace_distributed_task(self, task_type: str, workload_id: str = None):
        """Decorator for tracing distributed tasks (Ray, etc.)"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                span_name = f"distributed.{task_type}.{func.__name__}"
                
                attributes = {
                    "task.type": task_type,
                    "task.function": func.__name__,
                    "component": "distributed"
                }
                
                if workload_id:
                    attributes["workload.id"] = workload_id
                
                with self.start_span(span_name, trace.SpanKind.CONSUMER, attributes) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Add task-specific attributes
                        if isinstance(result, dict):
                            for key in ["task_count", "processed_items", "batch_size"]:
                                if key in result:
                                    span.set_attribute(f"task.{key}", result[key])
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                    
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", duration_ms)
            
            return wrapper
        return decorator
    
    def add_span_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add an event to the current span"""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes or {})
    
    def set_span_attribute(self, key: str, value: Any):
        """Set an attribute on the current span"""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, value)
    
    def set_baggage(self, key: str, value: str):
        """Set baggage item for trace context propagation"""
        baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item from trace context"""
        return baggage.get_baggage(key)
    
    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation"""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            return {
                "trace_id": f"{span_context.trace_id:032x}",
                "span_id": f"{span_context.span_id:016x}",
                "trace_flags": f"{span_context.trace_flags:02x}"
            }
        return {}
    
    def inject_trace_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into HTTP headers"""
        from opentelemetry.propagate import inject
        inject(headers)
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]):
        """Extract trace context from HTTP headers"""
        from opentelemetry.propagate import extract
        return extract(headers)
    
    def create_child_span(self, name: str, parent_span: trace.Span,
                         attributes: Dict[str, Any] = None) -> trace.Span:
        """Create a child span from a parent span"""
        context = trace.set_span_in_context(parent_span)
        return self.start_span(name, attributes=attributes, parent_context=context)
    
    def shutdown(self):
        """Shutdown tracing system"""
        if self.tracer_provider:
            # Force flush all pending spans
            self.tracer_provider.force_flush()
            
            # Shutdown span processors
            self.tracer_provider.shutdown()
        
        if self.meter_provider:
            # Shutdown metric providers
            self.meter_provider.shutdown()
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Distributed tracing shutdown completed",
            EventType.SYSTEM,
            {},
            "tracing"
        )

# Global tracer instance
distributed_tracer = DistributedTracer()

# Convenience functions and decorators
def initialize_tracing(config: TracingConfig = None):
    """Initialize distributed tracing"""
    global distributed_tracer
    if config:
        distributed_tracer = DistributedTracer(config)
    distributed_tracer.initialize()

def trace(name: str = None, attributes: Dict[str, Any] = None, record_exception: bool = True):
    """Decorator to trace function execution"""
    return distributed_tracer.trace_function(name, attributes, record_exception)

def trace_agent(agent_id: str, framework: str, message_type: str = "user_message"):
    """Decorator to trace agent requests"""
    return distributed_tracer.trace_agent_request(agent_id, framework, message_type)

def trace_distributed(task_type: str, workload_id: str = None):
    """Decorator to trace distributed tasks"""
    return distributed_tracer.trace_distributed_task(task_type, workload_id)

def start_span(name: str, kind: trace.SpanKind = trace.SpanKind.INTERNAL,
              attributes: Dict[str, Any] = None, parent_context=None) -> trace.Span:
    """Start a new span"""
    return distributed_tracer.start_span(name, kind, attributes, parent_context)

def add_span_event(name: str, attributes: Dict[str, Any] = None):
    """Add an event to the current span"""
    distributed_tracer.add_span_event(name, attributes)

def set_span_attribute(key: str, value: Any):
    """Set an attribute on the current span"""
    distributed_tracer.set_span_attribute(key, value)

def get_trace_context() -> Dict[str, str]:
    """Get current trace context"""
    return distributed_tracer.get_trace_context()

def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject trace context into headers"""
    return distributed_tracer.inject_trace_context(headers)

def extract_trace_context(headers: Dict[str, str]):
    """Extract trace context from headers"""
    return distributed_tracer.extract_trace_context(headers)

def shutdown_tracing():
    """Shutdown tracing system"""
    distributed_tracer.shutdown()

# Middleware for FastAPI to add trace context to logs
class TracingLogMiddleware:
    """Middleware to add trace context to log records"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract trace context from headers
            headers = dict(scope.get("headers", []))
            header_dict = {k.decode(): v.decode() for k, v in headers.items()}
            
            # Extract and set trace context
            trace_context = extract_trace_context(header_dict)
            
            # Set context variables for logging
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                trace_id_var.set(f"{span_context.trace_id:032x}")
                span_id_var.set(f"{span_context.span_id:016x}")
                
                # Generate request ID if not present
                request_id = header_dict.get("x-request-id", str(uuid.uuid4()))
                request_id_var.set(request_id)
        
        await self.app(scope, receive, send)

# Context manager for manual span management
class TracingContext:
    """Context manager for manual span management"""
    
    def __init__(self, name: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span = None
    
    def __enter__(self):
        self.span = start_span(self.name, attributes=self.attributes)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.record_exception(exc_val)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()

# Helper function to create tracing context
def tracing_context(name: str, attributes: Dict[str, Any] = None) -> TracingContext:
    """Create a tracing context manager"""
    return TracingContext(name, attributes)