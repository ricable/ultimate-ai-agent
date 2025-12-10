# File: backend/monitoring/monitoring_api.py
"""
Comprehensive monitoring API endpoints for UAP platform.
Integrates metrics, alerting, health checks, incident management, and tracing.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json

from .metrics.prometheus_metrics import prometheus_metrics
from .logs.logger import uap_logger, EventType, LogLevel
from .alerting.alerts import alert_manager, get_active_alerts, get_alert_history, acknowledge_alert
from .health_checker import health_monitor, get_system_health, get_health_summary
from .incident_manager import (
    incident_manager, get_active_incidents, get_incident_history, 
    resolve_incident, generate_post_mortem
)
from .tracing.distributed_tracing import (
    distributed_tracer, get_trace_context, TracingConfig
)

# Pydantic models for API requests/responses
class AlertRuleRequest(BaseModel):
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str
    time_window_minutes: int = 5
    severity: str = "warning"
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=lambda: ["log"])

class IncidentResolveRequest(BaseModel):
    resolution: str
    resolved_by: str = "admin"

class HealthCheckRequest(BaseModel):
    name: str
    component_type: str
    check_function: str
    interval_seconds: int = 30
    timeout_seconds: int = 10
    critical: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TracingConfigRequest(BaseModel):
    sampling_rate: float = 0.1
    jaeger_endpoint: Optional[str] = None
    enable_console_export: bool = False
    force_trace_components: List[str] = Field(default_factory=lambda: ["agent", "framework"])

class MonitoringOverviewResponse(BaseModel):
    timestamp: str
    system_health: Dict[str, Any]
    active_alerts: int
    critical_alerts: int
    active_incidents: int
    total_spans_last_hour: int
    average_response_time_ms: float
    error_rate_percent: float
    websocket_connections: int
    distributed_tasks_active: int

class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    timestamp: str
    time_window_minutes: int

class TraceSearchRequest(BaseModel):
    service_name: Optional[str] = None
    operation_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_duration_ms: Optional[int] = None
    max_duration_ms: Optional[int] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    limit: int = 100

# Create router
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# ============================================================================
# Overview and Dashboard Endpoints
# ============================================================================

@router.get("/overview", response_model=MonitoringOverviewResponse)
async def get_monitoring_overview():
    """Get comprehensive monitoring overview"""
    try:
        # Get system health
        health_data = get_system_health()
        
        # Get alerts
        active_alerts_data = get_active_alerts()
        critical_alerts = len([a for a in active_alerts_data if a.get("severity") == "critical"])
        
        # Get incidents
        active_incidents_data = get_active_incidents()
        
        # Get performance metrics (simplified)
        avg_response_time = 0.0
        error_rate = 0.0
        websocket_connections = 0
        distributed_tasks = 0
        
        # Calculate metrics from performance monitor if available
        # This would integrate with your existing performance monitoring
        
        return MonitoringOverviewResponse(
            timestamp=datetime.utcnow().isoformat(),
            system_health=health_data,
            active_alerts=len(active_alerts_data),
            critical_alerts=critical_alerts,
            active_incidents=len(active_incidents_data),
            total_spans_last_hour=0,  # Would come from tracing system
            average_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            websocket_connections=websocket_connections,
            distributed_tasks_active=distributed_tasks
        )
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get monitoring overview: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring overview")

@router.get("/status")
async def get_monitoring_status():
    """Get status of all monitoring components"""
    try:
        status = {
            "monitoring_system": "operational",
            "components": {
                "prometheus_metrics": "active" if prometheus_metrics else "inactive",
                "alerting_system": "active" if alert_manager.running else "inactive",
                "health_monitoring": "active" if health_monitor.running else "inactive",
                "incident_management": "active" if incident_manager.running else "inactive",
                "distributed_tracing": "active" if distributed_tracer.initialized else "inactive"
            },
            "statistics": {
                "total_alert_rules": len(alert_manager.rules),
                "active_alerts": len(alert_manager.active_alerts),
                "health_checks": len(health_monitor.health_checks),
                "active_incidents": len([i for i in incident_manager.incidents.values() 
                                       if i.status.value not in ["resolved", "closed"]]),
                "total_incidents": len(incident_manager.incidents)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return status
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get monitoring status: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring status")

# ============================================================================
# Metrics Endpoints
# ============================================================================

@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format"""
    try:
        metrics_data = prometheus_metrics.get_metrics()
        return Response(
            content=metrics_data,
            media_type=prometheus_metrics.get_content_type()
        )
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get Prometheus metrics: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve Prometheus metrics")

@router.get("/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary(
    time_window_minutes: int = Query(60, description="Time window in minutes", ge=1, le=1440)
):
    """Get summarized metrics for the specified time window"""
    try:
        # This would integrate with your metrics collection system
        # For now, return basic structure
        metrics_summary = {
            "system_metrics": {
                "cpu_usage_percent": 0.0,
                "memory_usage_percent": 0.0,
                "disk_usage_percent": 0.0
            },
            "application_metrics": {
                "agent_requests_per_minute": 0.0,
                "average_response_time_ms": 0.0,
                "error_rate_percent": 0.0
            },
            "infrastructure_metrics": {
                "websocket_connections": 0,
                "database_connections": 0,
                "cache_hit_rate_percent": 0.0
            }
        }
        
        return MetricsResponse(
            metrics=metrics_summary,
            timestamp=datetime.utcnow().isoformat(),
            time_window_minutes=time_window_minutes
        )
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get metrics summary: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "time_window": time_window_minutes},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics summary")

# ============================================================================
# Alerting Endpoints
# ============================================================================

@router.get("/alerts")
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    limit: int = Query(100, description="Maximum number of alerts", ge=1, le=1000)
):
    """Get alerts with optional filtering"""
    try:
        alerts = get_active_alerts()
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.get("status") == status]
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
        
        return {
            "alerts": alerts[:limit],
            "total_count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get alerts: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert_endpoint(alert_id: str, acknowledged_by: str = "admin"):
    """Acknowledge an alert"""
    try:
        acknowledge_alert(alert_id, acknowledged_by)
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Alert acknowledged: {alert_id}",
            EventType.SYSTEM,
            {"alert_id": alert_id, "acknowledged_by": acknowledged_by},
            "monitoring_api"
        )
        
        return {"status": "acknowledged", "alert_id": alert_id, "acknowledged_by": acknowledged_by}
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to acknowledge alert {alert_id}: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "alert_id": alert_id},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@router.get("/alerts/history")
async def get_alert_history_endpoint(
    limit: int = Query(100, description="Maximum number of alerts", ge=1, le=1000)
):
    """Get alert history"""
    try:
        history = get_alert_history(limit)
        
        return {
            "alerts": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get alert history: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve alert history")

# ============================================================================
# Health Monitoring Endpoints
# ============================================================================

@router.get("/health")
async def get_health():
    """Get comprehensive system health status"""
    try:
        health_summary = get_health_summary()
        return health_summary
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get health status: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve health status")

@router.get("/health/checks")
async def get_health_checks():
    """Get all health check configurations and results"""
    try:
        checks_info = []
        for name, check in health_monitor.health_checks.items():
            result = health_monitor.health_results.get(name)
            
            check_info = {
                "name": check.name,
                "component_type": check.component_type.value,
                "interval_seconds": check.interval_seconds,
                "critical": check.critical,
                "enabled": check.enabled,
                "last_result": result.to_dict() if result else None
            }
            checks_info.append(check_info)
        
        return {
            "health_checks": checks_info,
            "total_checks": len(checks_info),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get health checks: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve health checks")

# ============================================================================
# Incident Management Endpoints
# ============================================================================

@router.get("/incidents")
async def get_incidents(
    status: Optional[str] = Query(None, description="Filter by incident status"),
    severity: Optional[str] = Query(None, description="Filter by incident severity"),
    limit: int = Query(50, description="Maximum number of incidents", ge=1, le=500)
):
    """Get incidents with optional filtering"""
    try:
        incidents = get_active_incidents()
        
        # Apply filters
        if status:
            incidents = [i for i in incidents if i.get("status") == status]
        if severity:
            incidents = [i for i in incidents if i.get("severity") == severity]
        
        return {
            "incidents": incidents[:limit],
            "total_count": len(incidents),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get incidents: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve incidents")

@router.post("/incidents/{incident_id}/resolve")
async def resolve_incident_endpoint(
    incident_id: str, 
    request: IncidentResolveRequest
):
    """Resolve an incident"""
    try:
        resolve_incident(incident_id, request.resolution, request.resolved_by)
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Incident resolved: {incident_id}",
            EventType.SYSTEM,
            {
                "incident_id": incident_id, 
                "resolution": request.resolution,
                "resolved_by": request.resolved_by
            },
            "monitoring_api"
        )
        
        return {
            "status": "resolved", 
            "incident_id": incident_id,
            "resolution": request.resolution,
            "resolved_by": request.resolved_by,
            "resolved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to resolve incident {incident_id}: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "incident_id": incident_id},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to resolve incident")

@router.get("/incidents/{incident_id}/post-mortem")
async def get_post_mortem(incident_id: str):
    """Generate and retrieve post-mortem for an incident"""
    try:
        post_mortem = await generate_post_mortem(incident_id)
        return post_mortem
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to generate post-mortem for {incident_id}: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "incident_id": incident_id},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to generate post-mortem")

@router.get("/incidents/history")
async def get_incident_history_endpoint(
    limit: int = Query(100, description="Maximum number of incidents", ge=1, le=500)
):
    """Get incident history"""
    try:
        history = get_incident_history(limit)
        
        return {
            "incidents": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get incident history: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve incident history")

# ============================================================================
# Distributed Tracing Endpoints
# ============================================================================

@router.get("/tracing/status")
async def get_tracing_status():
    """Get distributed tracing status"""
    try:
        status = {
            "initialized": distributed_tracer.initialized,
            "service_name": distributed_tracer.config.service_name if distributed_tracer.config else None,
            "jaeger_endpoint": distributed_tracer.config.jaeger_endpoint if distributed_tracer.config else None,
            "sampling_rate": distributed_tracer.config.sampling_rate if distributed_tracer.config else None,
            "current_trace_context": get_trace_context(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return status
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get tracing status: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve tracing status")

@router.post("/tracing/configure")
async def configure_tracing(config: TracingConfigRequest):
    """Configure distributed tracing settings"""
    try:
        # Update tracing configuration
        if distributed_tracer.config:
            distributed_tracer.config.sampling_rate = config.sampling_rate
            if config.jaeger_endpoint:
                distributed_tracer.config.jaeger_endpoint = config.jaeger_endpoint
            distributed_tracer.config.force_trace_components = config.force_trace_components
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Tracing configuration updated",
            EventType.SYSTEM,
            {
                "sampling_rate": config.sampling_rate,
                "jaeger_endpoint": config.jaeger_endpoint,
                "force_trace_components": config.force_trace_components
            },
            "monitoring_api"
        )
        
        return {
            "status": "configured",
            "sampling_rate": config.sampling_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to configure tracing: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to configure tracing")

# ============================================================================
# System Operations Endpoints
# ============================================================================

@router.post("/system/start")
async def start_monitoring_systems(background_tasks: BackgroundTasks):
    """Start all monitoring systems"""
    try:
        async def start_systems():
            # Start alerting system
            if not alert_manager.running:
                await alert_manager.start()
            
            # Start health monitoring
            if not health_monitor.running:
                await health_monitor.start_monitoring()
            
            # Start incident management
            if not incident_manager.running:
                await incident_manager.start_incident_monitoring()
            
            # Initialize tracing if not already done
            if not distributed_tracer.initialized:
                distributed_tracer.initialize()
        
        background_tasks.add_task(start_systems)
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Starting all monitoring systems",
            EventType.SYSTEM,
            {},
            "monitoring_api"
        )
        
        return {"status": "starting", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to start monitoring systems: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to start monitoring systems")

@router.post("/system/stop")
async def stop_monitoring_systems(background_tasks: BackgroundTasks):
    """Stop all monitoring systems"""
    try:
        async def stop_systems():
            # Stop alerting system
            if alert_manager.running:
                await alert_manager.stop()
            
            # Stop health monitoring
            if health_monitor.running:
                await health_monitor.stop_monitoring()
            
            # Stop incident management
            if incident_manager.running:
                await incident_manager.stop_incident_monitoring()
            
            # Shutdown tracing
            if distributed_tracer.initialized:
                distributed_tracer.shutdown()
        
        background_tasks.add_task(stop_systems)
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Stopping all monitoring systems",
            EventType.SYSTEM,
            {},
            "monitoring_api"
        )
        
        return {"status": "stopping", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to stop monitoring systems: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to stop monitoring systems")

# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/ping")
async def monitoring_ping():
    """Simple ping endpoint for monitoring system health"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "uap-monitoring",
        "version": "3.0.0"
    }

@router.get("/config")
async def get_monitoring_config():
    """Get monitoring system configuration"""
    try:
        config = {
            "alerting": {
                "total_rules": len(alert_manager.rules),
                "evaluation_interval": "30s",
                "notification_channels": list(alert_manager.notification_handlers.keys())
            },
            "health_monitoring": {
                "total_checks": len(health_monitor.health_checks),
                "sla_metrics": list(health_monitor.sla_metrics.keys())
            },
            "incident_management": {
                "response_types": list(incident_manager.incident_responses.keys()),
                "runbooks": list(incident_manager.runbooks.keys())
            },
            "tracing": {
                "initialized": distributed_tracer.initialized,
                "service_name": distributed_tracer.config.service_name if distributed_tracer.config else None,
                "sampling_rate": distributed_tracer.config.sampling_rate if distributed_tracer.config else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return config
        
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get monitoring config: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "monitoring_api"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring configuration")

# Add router to main application
def include_monitoring_router(app):
    """Include monitoring router in FastAPI app"""
    app.include_router(router)