# File: backend/services/analytics_service.py
"""
Advanced Analytics Service for UAP Platform

Provides real-time analytics data processing, metrics aggregation,
and business intelligence for the analytics dashboard.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import logging

from ..operations.monitoring import operations_monitor, AnomalyEvent
from ..analytics.usage_analytics import usage_analytics, get_real_time_metrics
from ..analytics.predictive_analytics import predictive_analytics
from ..analytics.ab_testing import ab_testing
from ..cache.redis_cache import get_redis_client
from ..database.service import get_database_service
from ..models.analytics import SystemMetrics, AgentUsage, UserSession

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RealTimeMetric:
    """Real-time metric data point"""
    name: str
    value: Union[float, int, str]
    timestamp: float
    labels: Dict[str, str]
    unit: str = ""
    category: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DashboardData:
    """Complete dashboard data structure"""
    timestamp: float
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    real_time_activity: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    predictions: Dict[str, Any]
    experiments: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AnalyticsService:
    """
    Advanced analytics service that provides real-time data processing,
    metrics aggregation, and dashboard data for the UAP platform.
    """
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.db_service = get_database_service()
        
        # Real-time metrics storage
        self.real_time_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_subscribers: Dict[str, List] = defaultdict(list)
        
        # Dashboard data cache
        self.dashboard_cache = {}
        self.cache_ttl = 30  # seconds
        self.last_cache_update = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.agent_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert management
        self.active_alerts = []
        self.alert_history = deque(maxlen=500)
        
    async def start_analytics_collection(self):
        """Start background analytics collection tasks"""
        try:
            # Start collection tasks
            tasks = [
                asyncio.create_task(self._collect_system_metrics()),
                asyncio.create_task(self._collect_agent_metrics()),
                asyncio.create_task(self._collect_user_activity()),
                asyncio.create_task(self._process_alerts()),
                asyncio.create_task(self._update_predictions())
            ]
            
            logger.info("Started analytics collection tasks")
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in analytics collection: {e}")
            
    async def get_dashboard_data(self, user_id: Optional[str] = None) -> DashboardData:
        """Get comprehensive dashboard data"""
        try:
            # Check cache first
            current_time = time.time()
            if (current_time - self.last_cache_update) < self.cache_ttl and self.dashboard_cache:
                return DashboardData(**self.dashboard_cache)
            
            # Collect all dashboard data
            dashboard_data = await self._collect_dashboard_data(user_id)
            
            # Update cache
            self.dashboard_cache = dashboard_data.to_dict()
            self.last_cache_update = current_time
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            # Return basic data on error
            return DashboardData(
                timestamp=time.time(),
                system_health={"status": "error", "message": str(e)},
                performance_metrics={},
                business_metrics={},
                real_time_activity=[],
                alerts=[],
                predictions={},
                experiments={}
            )
    
    async def _collect_dashboard_data(self, user_id: Optional[str] = None) -> DashboardData:
        """Collect comprehensive dashboard data"""
        current_time = time.time()
        
        # System health data
        system_health = await self._get_system_health()
        
        # Performance metrics
        performance_metrics = await self._get_performance_metrics()
        
        # Business metrics
        business_metrics = await self._get_business_metrics()
        
        # Real-time activity
        real_time_activity = await self._get_real_time_activity()
        
        # Active alerts
        alerts = await self._get_active_alerts()
        
        # Predictions
        predictions = await self._get_predictions()
        
        # A/B testing experiments
        experiments = await self._get_experiments()
        
        return DashboardData(
            timestamp=current_time,
            system_health=system_health,
            performance_metrics=performance_metrics,
            business_metrics=business_metrics,
            real_time_activity=real_time_activity,
            alerts=alerts,
            predictions=predictions,
            experiments=experiments
        )
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            # Get from operations monitor
            system_status = await operations_monitor.get_system_status()
            
            # Add service health checks
            services_health = {
                "database": await self._check_database_health(),
                "redis": await self._check_redis_health(),
                "analytics": True,  # We're running so analytics is healthy
                "agent_orchestrator": True  # TODO: Add actual check
            }
            
            return {
                "overall_status": "healthy" if all(services_health.values()) else "degraded",
                "system_metrics": system_status.get("system", {}),
                "services": services_health,
                "monitoring": system_status.get("monitoring", {}),
                "anomalies": system_status.get("anomalies", {}),
                "last_updated": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "last_updated": time.time()
            }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # Get real-time metrics
            real_time = get_real_time_metrics()
            
            # Calculate performance trends
            if len(self.performance_history) > 10:
                recent_metrics = list(self.performance_history)[-10:]
                avg_response_time = statistics.mean([m.get("response_time", 0) for m in recent_metrics])
                avg_throughput = statistics.mean([m.get("throughput", 0) for m in recent_metrics])
                error_rate = statistics.mean([m.get("error_rate", 0) for m in recent_metrics])
            else:
                avg_response_time = real_time.get("avg_response_time", 0)
                avg_throughput = real_time.get("requests_per_second", 0)
                error_rate = real_time.get("error_rate", 0)
            
            return {
                "response_time": {
                    "current": avg_response_time,
                    "target": 2000,  # 2s target
                    "status": "excellent" if avg_response_time < 100 else "good" if avg_response_time < 1000 else "warning"
                },
                "throughput": {
                    "current": avg_throughput,
                    "requests_per_second": real_time.get("requests_per_second", 0),
                    "messages_per_second": real_time.get("websocket_messages_per_sec", 0)
                },
                "error_rate": {
                    "current": error_rate,
                    "target": 0.01,  # 1% target
                    "status": "excellent" if error_rate < 0.001 else "good" if error_rate < 0.01 else "warning"
                },
                "resource_usage": real_time.get("system_metrics", {}),
                "last_updated": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e), "last_updated": time.time()}
    
    async def _get_business_metrics(self) -> Dict[str, Any]:
        """Get business intelligence metrics"""
        try:
            # Get usage summary
            usage_summary = usage_analytics.get_usage_summary(24)  # Last 24 hours
            
            # Calculate key business metrics
            total_users = usage_summary.get("unique_users", 0)
            total_sessions = usage_summary.get("total_sessions", 0)
            total_requests = usage_summary.get("total_requests", 0)
            avg_session_duration = usage_summary.get("avg_session_duration", 0)
            
            # Growth metrics (compare to previous period)
            growth_metrics = await self._calculate_growth_metrics()
            
            return {
                "user_engagement": {
                    "total_users": total_users,
                    "active_users": len(usage_analytics.active_sessions),
                    "total_sessions": total_sessions,
                    "avg_session_duration": avg_session_duration,
                    "growth": growth_metrics.get("user_growth", 0)
                },
                "platform_usage": {
                    "total_requests": total_requests,
                    "requests_per_user": total_requests / max(total_users, 1),
                    "framework_distribution": usage_summary.get("framework_distribution", {}),
                    "growth": growth_metrics.get("usage_growth", 0)
                },
                "cost_metrics": {
                    "daily_cost": usage_summary.get("estimated_cost", 0),
                    "cost_per_request": usage_summary.get("cost_per_request", 0),
                    "cost_trend": growth_metrics.get("cost_trend", "stable")
                },
                "last_updated": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting business metrics: {e}")
            return {"error": str(e), "last_updated": time.time()}
    
    async def _get_real_time_activity(self) -> List[Dict[str, Any]]:
        """Get recent real-time activity feed"""
        try:
            activities = []
            
            # Get recent agent requests
            recent_requests = list(usage_analytics.events_history)[-50:]
            for event in recent_requests:
                activities.append({
                    "type": "agent_request",
                    "timestamp": event.get("timestamp", time.time()),
                    "description": f"Agent request to {event.get('framework', 'unknown')} framework",
                    "metadata": {
                        "framework": event.get("framework"),
                        "response_time": event.get("response_time"),
                        "user_id": event.get("user_id")
                    }
                })
            
            # Get recent anomalies
            recent_anomalies = await operations_monitor.anomaly_detector.get_recent_anomalies(10)
            for anomaly in recent_anomalies:
                activities.append({
                    "type": "anomaly",
                    "timestamp": anomaly.timestamp,
                    "description": anomaly.description,
                    "metadata": {
                        "severity": anomaly.severity.value,
                        "metric_name": anomaly.metric_name,
                        "actual_value": anomaly.actual_value
                    }
                })
            
            # Sort by timestamp (most recent first)
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return activities[:50]  # Return last 50 activities
            
        except Exception as e:
            logger.error(f"Error getting real-time activity: {e}")
            return []
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts and warnings"""
        try:
            alerts = []
            
            # Get system alerts from monitoring
            recent_anomalies = await operations_monitor.anomaly_detector.get_recent_anomalies(20)
            
            for anomaly in recent_anomalies:
                if anomaly.severity.value in ["critical", "warning"]:
                    alerts.append({
                        "id": anomaly.id,
                        "type": "system_anomaly",
                        "severity": anomaly.severity.value,
                        "title": f"{anomaly.anomaly_type.value.title()} Anomaly",
                        "message": anomaly.description,
                        "recommendation": anomaly.recommendation,
                        "timestamp": anomaly.timestamp,
                        "metric_name": anomaly.metric_name,
                        "actual_value": anomaly.actual_value
                    })
            
            # Add performance alerts
            performance_metrics = await self._get_performance_metrics()
            
            if performance_metrics.get("response_time", {}).get("status") == "warning":
                alerts.append({
                    "id": f"perf_response_time_{int(time.time())}",
                    "type": "performance",
                    "severity": "warning",
                    "title": "High Response Time",
                    "message": "Average response time is above normal thresholds",
                    "recommendation": "Check system resources and optimize slow operations",
                    "timestamp": time.time()
                })
            
            if performance_metrics.get("error_rate", {}).get("status") == "warning":
                alerts.append({
                    "id": f"perf_error_rate_{int(time.time())}",
                    "type": "performance",
                    "severity": "warning",
                    "title": "High Error Rate",
                    "message": "Error rate is above acceptable thresholds",
                    "recommendation": "Review application logs and fix error-causing issues",
                    "timestamp": time.time()
                })
            
            # Sort by severity and timestamp
            severity_order = {"critical": 0, "warning": 1, "info": 2}
            alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x["timestamp"]))
            
            return alerts[:20]  # Return top 20 alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def _get_predictions(self) -> Dict[str, Any]:
        """Get predictive analytics insights"""
        try:
            predictions = {}
            
            # Get recent predictions
            recent_predictions = list(predictive_analytics.prediction_history)[-10:]
            
            for prediction in recent_predictions:
                pred_type = prediction.prediction_type.value
                predictions[pred_type] = {
                    "predicted_value": prediction.predicted_value,
                    "confidence": prediction.confidence_score,
                    "created_at": prediction.created_at,
                    "features": prediction.features
                }
            
            # Get model performance
            model_performance = {}
            for pred_type, model in predictive_analytics.models.items():
                model_performance[pred_type.value] = {
                    "is_trained": model.is_trained,
                    "accuracy": model.performance_metrics.r2 * 100 if model.performance_metrics else 0,
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None
                }
            
            return {
                "recent_predictions": predictions,
                "model_performance": model_performance,
                "last_updated": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {"error": str(e), "last_updated": time.time()}
    
    async def _get_experiments(self) -> Dict[str, Any]:
        """Get A/B testing experiments data"""
        try:
            experiments_data = {
                "active_experiments": [],
                "completed_experiments": [],
                "total_participants": 0
            }
            
            for exp_id, experiment in ab_testing.experiments.items():
                exp_data = {
                    "id": exp_id,
                    "name": experiment.name,
                    "status": experiment.status.value,
                    "participants": len(ab_testing.participants.get(exp_id, [])),
                    "created_at": experiment.created_at,
                    "experiment_type": experiment.experiment_type.value
                }
                
                if experiment.status.value == "active":
                    experiments_data["active_experiments"].append(exp_data)
                else:
                    experiments_data["completed_experiments"].append(exp_data)
                
                experiments_data["total_participants"] += exp_data["participants"]
            
            return experiments_data
            
        except Exception as e:
            logger.error(f"Error getting experiments: {e}")
            return {"error": str(e)}
    
    async def _calculate_growth_metrics(self) -> Dict[str, float]:
        """Calculate growth metrics compared to previous period"""
        try:
            # Get current and previous period data
            current_usage = usage_analytics.get_usage_summary(24)  # Last 24 hours
            previous_usage = usage_analytics.get_usage_summary(24, offset_hours=24)  # Previous 24 hours
            
            current_users = current_usage.get("unique_users", 0)
            previous_users = previous_usage.get("unique_users", 0)
            
            current_requests = current_usage.get("total_requests", 0)
            previous_requests = previous_usage.get("total_requests", 0)
            
            # Calculate growth percentages
            user_growth = ((current_users - previous_users) / max(previous_users, 1)) * 100
            usage_growth = ((current_requests - previous_requests) / max(previous_requests, 1)) * 100
            
            return {
                "user_growth": round(user_growth, 2),
                "usage_growth": round(usage_growth, 2),
                "cost_trend": "increasing" if usage_growth > 5 else "stable" if usage_growth > -5 else "decreasing"
            }
            
        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            return {"user_growth": 0, "usage_growth": 0, "cost_trend": "stable"}
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics"""
        while True:
            try:
                # Collect system health data
                system_health = await self._get_system_health()
                
                # Store in performance history
                metric_data = {
                    "timestamp": time.time(),
                    "system_metrics": system_health.get("system_metrics", {}),
                    "response_time": system_health.get("response_time", 0),
                    "throughput": system_health.get("throughput", 0),
                    "error_rate": system_health.get("error_rate", 0)
                }
                
                self.performance_history.append(metric_data)
                
                # Store in Redis for real-time access
                await self.redis_client.setex(
                    "analytics:system_metrics",
                    300,  # 5 minutes TTL
                    json.dumps(metric_data, default=str)
                )
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def _collect_agent_metrics(self):
        """Background task to collect agent-specific metrics"""
        while True:
            try:
                # Get agent usage data
                for framework in ["copilot", "agno", "mastra"]:
                    framework_metrics = usage_analytics.get_framework_metrics(framework)
                    
                    self.agent_metrics[framework].append({
                        "timestamp": time.time(),
                        "requests": framework_metrics.get("requests", 0),
                        "avg_response_time": framework_metrics.get("avg_response_time", 0),
                        "error_rate": framework_metrics.get("error_rate", 0),
                        "cost": framework_metrics.get("cost", 0)
                    })
                
            except Exception as e:
                logger.error(f"Error collecting agent metrics: {e}")
            
            await asyncio.sleep(60)  # Collect every minute
    
    async def _collect_user_activity(self):
        """Background task to collect user activity metrics"""
        while True:
            try:
                # Update active user count
                active_users = len(usage_analytics.active_sessions)
                
                # Store user activity metric
                await self.redis_client.setex(
                    "analytics:active_users",
                    60,
                    str(active_users)
                )
                
            except Exception as e:
                logger.error(f"Error collecting user activity: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def _process_alerts(self):
        """Background task to process and manage alerts"""
        while True:
            try:
                # Get current alerts
                current_alerts = await self._get_active_alerts()
                
                # Update active alerts
                self.active_alerts = current_alerts
                
                # Store in Redis for real-time access
                await self.redis_client.setex(
                    "analytics:active_alerts",
                    60,
                    json.dumps([alert for alert in current_alerts], default=str)
                )
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
            
            await asyncio.sleep(60)  # Process every minute
    
    async def _update_predictions(self):
        """Background task to update predictions"""
        while True:
            try:
                # Trigger prediction updates
                await predictive_analytics.collect_training_data()
                
                # Get updated predictions
                predictions = await self._get_predictions()
                
                # Store in Redis
                await self.redis_client.setex(
                    "analytics:predictions",
                    300,  # 5 minutes TTL
                    json.dumps(predictions, default=str)
                )
                
            except Exception as e:
                logger.error(f"Error updating predictions: {e}")
            
            await asyncio.sleep(300)  # Update every 5 minutes
    
    async def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            await self.db_service.execute_query("SELECT 1")
            return True
        except Exception:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity"""
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def add_real_time_metric(self, metric: RealTimeMetric):
        """Add a real-time metric for dashboard updates"""
        self.real_time_metrics[metric.name].append(metric)
        
        # Notify subscribers
        if metric.name in self.metric_subscribers:
            for callback in self.metric_subscribers[metric.name]:
                try:
                    callback(metric)
                except Exception as e:
                    logger.error(f"Error notifying metric subscriber: {e}")
    
    def subscribe_to_metric(self, metric_name: str, callback):
        """Subscribe to real-time metric updates"""
        self.metric_subscribers[metric_name].append(callback)
    
    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[RealTimeMetric]:
        """Get historical data for a specific metric"""
        if metric_name in self.real_time_metrics:
            return list(self.real_time_metrics[metric_name])[-limit:]
        return []

# Global analytics service instance
analytics_service = AnalyticsService()
