# File: backend/monitoring/dashboard/api.py
"""
Real-time system health dashboard API endpoints.
Provides comprehensive monitoring data for visualization and alerting.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json

from ..metrics.performance import performance_monitor, get_system_health, get_performance_summary
from ..metrics.prometheus_metrics import prometheus_metrics
from ..logs.logger import uap_logger, EventType, LogLevel

# Pydantic models for API responses
class SystemHealthResponse(BaseModel):
    overall_healthy: bool
    timestamp: str
    system_health: Dict[str, bool]
    agent_health: Dict[str, Dict[str, Any]]
    current_stats: Dict[str, Any]
    thresholds: Dict[str, Any]

class PerformanceSummaryResponse(BaseModel):
    time_window_minutes: int
    metrics_summary: Dict[str, Dict[str, Any]]
    agent_stats: Dict[str, Dict[str, Any]]
    websocket_summary: Dict[str, Any]

class AgentStatsResponse(BaseModel):
    agent_id: str
    framework: str
    total_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    success_rate: float
    last_request_time: Optional[str]

class MetricDataPoint(BaseModel):
    timestamp: str
    value: Union[float, int]
    tags: Dict[str, str] = Field(default_factory=dict)

class MetricTimeSeriesResponse(BaseModel):
    metric_name: str
    unit: str
    data_points: List[MetricDataPoint]
    time_range: str

class AlertResponse(BaseModel):
    alert_id: str
    severity: str
    message: str
    component: str
    timestamp: str
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DashboardOverviewResponse(BaseModel):
    system_health: SystemHealthResponse
    active_agents: int
    active_connections: int
    total_requests_last_hour: int
    avg_response_time_ms: float
    error_rate_percent: float
    alerts_count: Dict[str, int]

# Create router
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health_endpoint():
    """Get current system health status"""
    try:
        health_data = get_system_health()
        return SystemHealthResponse(**health_data)
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get system health: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")

@router.get("/performance", response_model=PerformanceSummaryResponse)
async def get_performance_summary_endpoint(
    time_window: int = Query(60, description="Time window in minutes", ge=1, le=1440)
):
    """Get performance summary for the specified time window"""
    try:
        summary_data = get_performance_summary(time_window)
        return PerformanceSummaryResponse(**summary_data)
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get performance summary: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "time_window": time_window},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve performance summary")

@router.get("/agents", response_model=Dict[str, AgentStatsResponse])
async def get_all_agent_stats():
    """Get performance statistics for all agents"""
    try:
        agent_stats = performance_monitor.get_agent_statistics()
        
        response_data = {}
        for agent_id, stats in agent_stats.items():
            response_data[agent_id] = AgentStatsResponse(
                agent_id=stats['agent_id'],
                framework=stats['framework'],
                total_requests=stats['total_requests'],
                avg_response_time_ms=stats['avg_response_time_ms'],
                p95_response_time_ms=stats['p95_response_time_ms'],
                p99_response_time_ms=stats['p99_response_time_ms'],
                success_rate=stats['success_rate'],
                last_request_time=stats['last_request_time']
            )
        
        return response_data
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get agent stats: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve agent statistics")

@router.get("/agents/{agent_id}", response_model=AgentStatsResponse)
async def get_agent_stats(agent_id: str):
    """Get performance statistics for a specific agent"""
    try:
        stats = performance_monitor.get_agent_statistics(agent_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return AgentStatsResponse(
            agent_id=stats.agent_id,
            framework=stats.framework,
            total_requests=stats.total_requests,
            avg_response_time_ms=stats.avg_response_time_ms,
            p95_response_time_ms=stats.p95_response_time_ms,
            p99_response_time_ms=stats.p99_response_time_ms,
            success_rate=stats.success_rate,
            last_request_time=stats.last_request_time.isoformat() if stats.last_request_time else None
        )
    except HTTPException:
        raise
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get agent stats for {agent_id}: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "agent_id": agent_id},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve agent statistics")

@router.get("/metrics/{metric_name}", response_model=MetricTimeSeriesResponse)
async def get_metric_timeseries(
    metric_name: str,
    time_window: int = Query(60, description="Time window in minutes", ge=1, le=1440),
    tags: Optional[str] = Query(None, description="Filter by tags (JSON format)")
):
    """Get time series data for a specific metric"""
    try:
        # Parse tags filter
        tag_filter = {}
        if tags:
            try:
                tag_filter = json.loads(tags)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid tags format. Use JSON.")
        
        # Get metrics from history
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window)
        
        # Filter metrics by name and time
        matching_metrics = []
        for metric in performance_monitor.metrics_history:
            if (metric.metric_name == metric_name and 
                metric.timestamp >= cutoff_time):
                
                # Apply tag filter if specified
                if tag_filter:
                    metric_tags = metric.tags or {}
                    if all(metric_tags.get(k) == v for k, v in tag_filter.items()):
                        matching_metrics.append(metric)
                else:
                    matching_metrics.append(metric)
        
        # Convert to response format
        data_points = [
            MetricDataPoint(
                timestamp=metric.timestamp.isoformat(),
                value=metric.value,
                tags=metric.tags or {}
            )
            for metric in matching_metrics
        ]
        
        # Determine unit
        unit = "unknown"
        if matching_metrics:
            unit = matching_metrics[0].unit
        
        return MetricTimeSeriesResponse(
            metric_name=metric_name,
            unit=unit,
            data_points=data_points,
            time_range=f"{time_window}m"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get metric timeseries for {metric_name}: {str(e)}",
            EventType.ERROR,
            {"error": str(e), "metric_name": metric_name},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve metric data")

@router.get("/metrics", response_model=List[str])
async def get_available_metrics():
    """Get list of available metrics"""
    try:
        # Get unique metric names from history
        metric_names = set()
        for metric in performance_monitor.metrics_history:
            metric_names.add(metric.metric_name)
        
        return sorted(list(metric_names))
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get available metrics: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve available metrics")

@router.get("/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus metrics endpoint"""
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
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve Prometheus metrics")

@router.get("/websockets", response_model=Dict[str, Any])
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    try:
        connections = performance_monitor.websocket_connections
        
        stats = {
            "total_active_connections": len(connections),
            "connections_by_agent": {},
            "connection_details": []
        }
        
        # Group by agent
        for conn_id, conn_info in connections.items():
            agent_id = conn_info['agent_id']
            if agent_id not in stats["connections_by_agent"]:
                stats["connections_by_agent"][agent_id] = 0
            stats["connections_by_agent"][agent_id] += 1
            
            # Add connection details
            duration = (datetime.utcnow() - conn_info['connected_at']).total_seconds()
            stats["connection_details"].append({
                "connection_id": conn_id,
                "agent_id": agent_id,
                "connected_at": conn_info['connected_at'].isoformat(),
                "duration_seconds": duration,
                "messages_sent": conn_info['messages_sent'],
                "messages_received": conn_info['messages_received'],
                "bytes_sent": conn_info['bytes_sent'],
                "bytes_received": conn_info['bytes_received']
            })
        
        return stats
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get WebSocket stats: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve WebSocket statistics")

@router.get("/overview", response_model=DashboardOverviewResponse)
async def get_dashboard_overview():
    """Get dashboard overview with key metrics"""
    try:
        # Get system health
        health_data = get_system_health()
        system_health = SystemHealthResponse(**health_data)
        
        # Get performance summary for last hour
        performance_data = get_performance_summary(60)
        
        # Calculate overview metrics
        active_agents = len(performance_monitor.agent_stats)
        active_connections = len(performance_monitor.websocket_connections)
        
        # Calculate total requests in last hour
        total_requests = 0
        avg_response_time = 0.0
        error_count = 0
        success_count = 0
        
        for agent_stats in performance_monitor.agent_stats.values():
            total_requests += agent_stats.total_requests
            error_count += agent_stats.error_count
            success_count += agent_stats.success_count
        
        # Calculate overall average response time
        if active_agents > 0:
            avg_response_time = sum(
                stats.avg_response_time_ms 
                for stats in performance_monitor.agent_stats.values()
            ) / active_agents
        
        # Calculate error rate
        error_rate = 0.0
        if total_requests > 0:
            error_rate = (error_count / total_requests) * 100
        
        # Mock alerts count (would be from alerting system)
        alerts_count = {
            "critical": 0,
            "warning": 0,
            "info": 0
        }
        
        return DashboardOverviewResponse(
            system_health=system_health,
            active_agents=active_agents,
            active_connections=active_connections,
            total_requests_last_hour=total_requests,
            avg_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            alerts_count=alerts_count
        )
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get dashboard overview: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard overview")

@router.get("/logs", response_model=List[Dict[str, Any]])
async def get_recent_logs(
    level: Optional[str] = Query(None, description="Filter by log level"),
    component: Optional[str] = Query(None, description="Filter by component"),
    limit: int = Query(100, description="Maximum number of logs", ge=1, le=1000)
):
    """Get recent log entries (would read from log files in production)"""
    try:
        # This is a simplified implementation
        # In production, this would read from log files or a log aggregation system
        
        sample_logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "component": "agent.copilot",
                "message": "Agent request processed successfully",
                "metadata": {"response_time_ms": 150, "agent_id": "copilot-1"}
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=1)).isoformat(),
                "level": "WARNING",
                "component": "websocket",
                "message": "Connection timeout detected",
                "metadata": {"connection_id": "conn-123", "duration_ms": 30000}
            }
        ]
        
        # Apply filters
        filtered_logs = sample_logs
        if level:
            filtered_logs = [log for log in filtered_logs if log["level"] == level.upper()]
        if component:
            filtered_logs = [log for log in filtered_logs if component in log["component"]]
        
        return filtered_logs[:limit]
    except Exception as e:
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Failed to get recent logs: {str(e)}",
            EventType.ERROR,
            {"error": str(e)},
            "dashboard"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")

# Health check endpoint specifically for monitoring
@router.get("/ping")
async def monitoring_ping():
    """Simple ping endpoint for monitoring systems"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "uap-monitoring"
    }