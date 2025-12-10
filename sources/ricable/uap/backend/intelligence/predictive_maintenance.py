# File: backend/intelligence/predictive_maintenance.py
"""
Predictive Maintenance and Failure Prevention System

AI-powered predictive maintenance with failure prediction, health monitoring,
and proactive system maintenance for the UAP platform.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
import uuid

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..monitoring.metrics.performance import performance_monitor
from ..services.performance_service import performance_service
from .performance_tuning import performance_tuner, PerformanceMetrics

logger = logging.getLogger(__name__)

class SystemComponent(Enum):
    """System components that can be monitored"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"
    AGENT_FRAMEWORK = "agent_framework"
    WEBSOCKET_CONNECTIONS = "websocket_connections"
    API_ENDPOINTS = "api_endpoints"

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class MaintenanceType(Enum):
    """Types of maintenance actions"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

class FailureRisk(Enum):
    """Failure risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMINENT = "imminent"

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component: SystemComponent
    timestamp: datetime
    health_status: HealthStatus
    health_score: float  # 0-100
    metrics: Dict[str, float]
    anomalies: List[str]
    trend: str  # improving, stable, degrading
    predicted_failure_time: Optional[datetime]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['component'] = self.component.value
        data['health_status'] = self.health_status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.predicted_failure_time:
            data['predicted_failure_time'] = self.predicted_failure_time.isoformat()
        return data

@dataclass
class MaintenanceAlert:
    """Maintenance alert for proactive action"""
    alert_id: str
    timestamp: datetime
    component: SystemComponent
    alert_type: MaintenanceType
    severity: str
    message: str
    failure_risk: FailureRisk
    time_to_failure: Optional[timedelta]
    recommended_actions: List[str]
    automated_actions: List[str]
    escalation_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['component'] = self.component.value
        data['alert_type'] = self.alert_type.value
        data['failure_risk'] = self.failure_risk.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.time_to_failure:
            data['time_to_failure_hours'] = self.time_to_failure.total_seconds() / 3600
        return data

@dataclass
class MaintenanceAction:
    """Maintenance action to be performed"""
    action_id: str
    timestamp: datetime
    component: SystemComponent
    action_type: MaintenanceType
    description: str
    procedure: List[str]
    estimated_duration: timedelta
    impact_assessment: Dict[str, Any]
    prerequisites: List[str]
    success_criteria: List[str]
    rollback_plan: List[str]
    automation_level: str  # manual, semi_automated, fully_automated
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['component'] = self.component.value
        data['action_type'] = self.action_type.value
        data['timestamp'] = self.timestamp.isoformat()
        data['estimated_duration_minutes'] = self.estimated_duration.total_seconds() / 60
        return data

@dataclass
class MaintenancePlan:
    """Comprehensive maintenance plan"""
    plan_id: str
    timestamp: datetime
    planning_horizon: timedelta
    component_health: List[ComponentHealth]
    alerts: List[MaintenanceAlert]
    maintenance_actions: List[MaintenanceAction]
    maintenance_schedule: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    cost_estimation: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['planning_horizon_hours'] = self.planning_horizon.total_seconds() / 3600
        return data

class PredictiveMaintenanceSystem:
    """AI-powered predictive maintenance system"""
    
    def __init__(self, monitoring_interval: int = 300):  # 5 minutes
        self.monitoring_interval = monitoring_interval
        self.component_health_history = defaultdict(lambda: deque(maxlen=1000))
        self.maintenance_history = deque(maxlen=500)
        self.alert_history = deque(maxlen=1000)
        
        # ML models for failure prediction
        self.failure_models = {}
        self.health_models = {}
        self.scalers = {}
        
        # Component configurations
        self.component_configs = {
            SystemComponent.CPU: {
                'health_thresholds': {'warning': 70, 'degraded': 80, 'critical': 90},
                'failure_indicators': ['sustained_high_usage', 'temperature_spike', 'frequency_throttling'],
                'maintenance_interval_hours': 168  # Weekly
            },
            SystemComponent.MEMORY: {
                'health_thresholds': {'warning': 75, 'degraded': 85, 'critical': 95},
                'failure_indicators': ['memory_leaks', 'fragmentation', 'swap_usage'],
                'maintenance_interval_hours': 72  # 3 days
            },
            SystemComponent.DATABASE: {
                'health_thresholds': {'warning': 70, 'degraded': 80, 'critical': 90},
                'failure_indicators': ['slow_queries', 'connection_pool_exhaustion', 'disk_space'],
                'maintenance_interval_hours': 24  # Daily
            },
            SystemComponent.CACHE: {
                'health_thresholds': {'warning': 60, 'degraded': 50, 'critical': 40},
                'failure_indicators': ['low_hit_rate', 'memory_pressure', 'connection_timeouts'],
                'maintenance_interval_hours': 12  # Twice daily
            }
        }
        
        # Initialize models
        self._initialize_models()
        
        # Start background monitoring
        self.monitoring_active = True
        
        logger.info("Predictive Maintenance System initialized")
    
    def _initialize_models(self):
        """Initialize ML models for predictive maintenance"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, using rule-based maintenance")
            return
        
        try:
            for component in SystemComponent:
                # Failure prediction model
                self.failure_models[component] = {
                    'model': GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    ),
                    'scaler': StandardScaler(),
                    'trained': False
                }
                
                # Health scoring model
                self.health_models[component] = {
                    'model': RandomForestClassifier(
                        n_estimators=50,
                        max_depth=8,
                        random_state=42
                    ),
                    'scaler': StandardScaler(),
                    'trained': False
                }
            
            logger.info("Predictive maintenance models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def assess_component_health(self, component: SystemComponent) -> ComponentHealth:
        """Assess health of a specific system component"""
        try:
            # Collect component-specific metrics
            metrics = await self._collect_component_metrics(component)
            
            # Calculate health score
            health_score = self._calculate_health_score(component, metrics)
            
            # Determine health status
            health_status = self._determine_health_status(component, health_score, metrics)
            
            # Detect anomalies
            anomalies = self._detect_component_anomalies(component, metrics)
            
            # Analyze trend
            trend = self._analyze_health_trend(component)
            
            # Predict failure time
            predicted_failure_time, confidence = await self._predict_failure_time(component, metrics)
            
            health = ComponentHealth(
                component=component,
                timestamp=datetime.utcnow(),
                health_status=health_status,
                health_score=health_score,
                metrics=metrics,
                anomalies=anomalies,
                trend=trend,
                predicted_failure_time=predicted_failure_time,
                confidence=confidence
            )
            
            # Store in history
            self.component_health_history[component].append(health)
            
            return health
            
        except Exception as e:
            logger.error(f"Error assessing {component.value} health: {e}")
            return ComponentHealth(
                component=component,
                timestamp=datetime.utcnow(),
                health_status=HealthStatus.WARNING,
                health_score=50.0,
                metrics={},
                anomalies=[f"Error assessing health: {str(e)}"],
                trend="unknown",
                predicted_failure_time=None,
                confidence=0.0
            )
    
    async def _collect_component_metrics(self, component: SystemComponent) -> Dict[str, float]:
        """Collect metrics specific to a component"""
        metrics = {}
        
        try:
            # Get system health and performance data
            system_health = performance_monitor.get_system_health()
            current_stats = system_health.get('current_stats', {})
            perf_service_stats = await performance_service.get_performance_stats()
            
            if component == SystemComponent.CPU:
                metrics = {
                    'usage_percent': current_stats.get('cpu_percent', 0),
                    'load_average': 0,  # Would get from system
                    'temperature': 0,   # Would get from sensors
                    'frequency': 0      # Would get from system
                }
            
            elif component == SystemComponent.MEMORY:
                metrics = {
                    'usage_percent': current_stats.get('memory_percent', 0),
                    'available_mb': current_stats.get('memory_available_mb', 0),
                    'used_mb': current_stats.get('memory_used_mb', 0),
                    'swap_usage_percent': 0,  # Would get from system
                    'fragmentation_ratio': 0  # Would calculate
                }
            
            elif component == SystemComponent.DATABASE:
                db_stats = perf_service_stats.get('database', {}).get('stats', {})
                metrics = {
                    'connection_pool_usage': 50,  # Would get from actual pool
                    'active_connections': current_stats.get('active_connections', 0),
                    'slow_query_count': 0,        # Would get from DB
                    'disk_usage_percent': current_stats.get('disk_usage_percent', 0),
                    'avg_query_time_ms': 0        # Would get from DB
                }
            
            elif component == SystemComponent.CACHE:
                cache_stats = perf_service_stats.get('cache', {}).get('stats', {})
                metrics = {
                    'hit_rate_percent': cache_stats.get('hit_rate', 0.5) * 100,
                    'memory_usage_percent': 70,   # Would get from cache system
                    'connection_count': 0,        # Would get from cache system
                    'eviction_rate': 0,          # Would get from cache system
                    'latency_ms': 0              # Would get from cache system
                }
            
            elif component == SystemComponent.WEBSOCKET_CONNECTIONS:
                metrics = {
                    'active_connections': current_stats.get('active_connections', 0),
                    'connection_errors': 0,       # Would track connection failures
                    'message_rate': 0,           # Would track message throughput
                    'bandwidth_usage_mbps': 0,   # Would monitor bandwidth
                    'avg_connection_duration': 0 # Would calculate from connection history
                }
            
            elif component == SystemComponent.API_ENDPOINTS:
                agent_stats = performance_monitor.get_agent_statistics()
                if isinstance(agent_stats, dict):
                    response_times = [
                        stats.get('avg_response_time_ms', 0)
                        for stats in agent_stats.values()
                        if isinstance(stats, dict)
                    ]
                    error_rates = [
                        (stats.get('error_count', 0) / max(1, stats.get('total_requests', 1))) * 100
                        for stats in agent_stats.values()
                        if isinstance(stats, dict)
                    ]
                    
                    metrics = {
                        'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
                        'max_response_time_ms': max(response_times) if response_times else 0,
                        'error_rate_percent': statistics.mean(error_rates) if error_rates else 0,
                        'throughput_rps': 0,      # Would calculate requests per second
                        'active_endpoints': len(agent_stats)
                    }
                else:
                    metrics = {
                        'avg_response_time_ms': 0,
                        'max_response_time_ms': 0,
                        'error_rate_percent': 0,
                        'throughput_rps': 0,
                        'active_endpoints': 0
                    }
            
            else:
                # Default metrics for other components
                metrics = {
                    'health_indicator': 50,
                    'availability_percent': 99,
                    'error_count': 0,
                    'performance_index': 80
                }
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {component.value}: {e}")
            metrics = {'error': 1, 'health_indicator': 0}
        
        return metrics
    
    def _calculate_health_score(self, component: SystemComponent, metrics: Dict[str, float]) -> float:
        """Calculate health score (0-100) for a component"""
        try:
            if 'error' in metrics:
                return 0.0
            
            config = self.component_configs.get(component, {})
            thresholds = config.get('health_thresholds', {'warning': 70, 'degraded': 80, 'critical': 90})
            
            if component == SystemComponent.CPU:
                usage = metrics.get('usage_percent', 0)
                if usage < thresholds['warning']:
                    return 100 - (usage / thresholds['warning'] * 30)  # 70-100 score
                elif usage < thresholds['degraded']:
                    return 70 - ((usage - thresholds['warning']) / (thresholds['degraded'] - thresholds['warning']) * 30)  # 40-70 score
                elif usage < thresholds['critical']:
                    return 40 - ((usage - thresholds['degraded']) / (thresholds['critical'] - thresholds['degraded']) * 30)  # 10-40 score
                else:
                    return max(0, 10 - (usage - thresholds['critical']))  # 0-10 score
            
            elif component == SystemComponent.MEMORY:
                usage = metrics.get('usage_percent', 0)
                # Similar calculation as CPU
                if usage < thresholds['warning']:
                    return 100 - (usage / thresholds['warning'] * 25)
                elif usage < thresholds['degraded']:
                    return 75 - ((usage - thresholds['warning']) / (thresholds['degraded'] - thresholds['warning']) * 35)
                elif usage < thresholds['critical']:
                    return 40 - ((usage - thresholds['degraded']) / (thresholds['critical'] - thresholds['degraded']) * 35)
                else:
                    return max(0, 5 - (usage - thresholds['critical']))
            
            elif component == SystemComponent.CACHE:
                hit_rate = metrics.get('hit_rate_percent', 50)
                # Higher hit rate = better health
                if hit_rate >= 80:
                    return 100
                elif hit_rate >= 60:
                    return 70 + ((hit_rate - 60) / 20 * 30)
                elif hit_rate >= 40:
                    return 40 + ((hit_rate - 40) / 20 * 30)
                else:
                    return max(0, hit_rate)
            
            elif component == SystemComponent.API_ENDPOINTS:
                response_time = metrics.get('avg_response_time_ms', 0)
                error_rate = metrics.get('error_rate_percent', 0)
                
                # Calculate score based on response time and error rate
                time_score = max(0, 100 - (response_time / 20))  # 2000ms = 0 score
                error_score = max(0, 100 - (error_rate * 10))    # 10% error = 0 score
                
                return (time_score + error_score) / 2
            
            else:
                # Generic health calculation
                health_indicator = metrics.get('health_indicator', 50)
                availability = metrics.get('availability_percent', 99)
                performance = metrics.get('performance_index', 80)
                
                return (health_indicator + availability + performance) / 3
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0  # Default moderate health
    
    def _determine_health_status(self, component: SystemComponent, health_score: float,
                                metrics: Dict[str, float]) -> HealthStatus:
        """Determine health status based on score and metrics"""
        try:
            if health_score >= 80:
                return HealthStatus.HEALTHY
            elif health_score >= 60:
                return HealthStatus.WARNING
            elif health_score >= 30:
                return HealthStatus.DEGRADED
            elif health_score >= 10:
                return HealthStatus.CRITICAL
            else:
                return HealthStatus.FAILED
                
        except Exception as e:
            logger.error(f"Error determining health status: {e}")
            return HealthStatus.WARNING
    
    def _detect_component_anomalies(self, component: SystemComponent,
                                   metrics: Dict[str, float]) -> List[str]:
        """Detect anomalies in component metrics"""
        anomalies = []
        
        try:
            config = self.component_configs.get(component, {})
            failure_indicators = config.get('failure_indicators', [])
            
            if component == SystemComponent.CPU:
                usage = metrics.get('usage_percent', 0)
                if usage > 95:
                    anomalies.append("Extremely high CPU usage detected")
                if usage > 80:
                    # Check if sustained high usage
                    recent_health = list(self.component_health_history[component])[-10:]
                    if len(recent_health) >= 5:
                        high_usage_count = sum(1 for h in recent_health if h.metrics.get('usage_percent', 0) > 80)
                        if high_usage_count >= 4:
                            anomalies.append("Sustained high CPU usage")
            
            elif component == SystemComponent.MEMORY:
                usage = metrics.get('usage_percent', 0)
                if usage > 95:
                    anomalies.append("Critically high memory usage")
                
                # Check for memory leak pattern
                recent_health = list(self.component_health_history[component])[-20:]
                if len(recent_health) >= 10:
                    memory_trend = [h.metrics.get('usage_percent', 0) for h in recent_health]
                    # Simple trend detection
                    if len(memory_trend) >= 5:
                        slope = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
                        if slope > 2:  # Increasing by more than 2% per measurement
                            anomalies.append("Potential memory leak detected")
            
            elif component == SystemComponent.CACHE:
                hit_rate = metrics.get('hit_rate_percent', 50)
                if hit_rate < 30:
                    anomalies.append("Critically low cache hit rate")
                
                # Check for degrading cache performance
                recent_health = list(self.component_health_history[component])[-10:]
                if len(recent_health) >= 5:
                    hit_rates = [h.metrics.get('hit_rate_percent', 50) for h in recent_health]
                    if all(rate < 50 for rate in hit_rates[-3:]):
                        anomalies.append("Sustained cache performance degradation")
            
            elif component == SystemComponent.API_ENDPOINTS:
                response_time = metrics.get('avg_response_time_ms', 0)
                error_rate = metrics.get('error_rate_percent', 0)
                
                if response_time > 5000:
                    anomalies.append("Extremely high API response times")
                if error_rate > 10:
                    anomalies.append("High API error rate")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            anomalies.append(f"Error in anomaly detection: {str(e)}")
        
        return anomalies
    
    def _analyze_health_trend(self, component: SystemComponent) -> str:
        """Analyze health trend for a component"""
        try:
            recent_health = list(self.component_health_history[component])[-10:]
            
            if len(recent_health) < 3:
                return "insufficient_data"
            
            scores = [h.health_score for h in recent_health]
            
            # Calculate trend
            if len(scores) >= 5:
                x = np.arange(len(scores))
                slope = np.polyfit(x, scores, 1)[0]
                
                if slope > 1:
                    return "improving"
                elif slope < -1:
                    return "degrading"
                else:
                    return "stable"
            else:
                # Simple comparison
                if scores[-1] > scores[0] + 5:
                    return "improving"
                elif scores[-1] < scores[0] - 5:
                    return "degrading"
                else:
                    return "stable"
                    
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return "unknown"
    
    async def _predict_failure_time(self, component: SystemComponent,
                                   metrics: Dict[str, float]) -> Tuple[Optional[datetime], float]:
        """Predict when component might fail"""
        try:
            # Simple rule-based prediction for now
            # In production, would use trained ML models
            
            current_health_score = self._calculate_health_score(component, metrics)
            
            if current_health_score > 70:
                return None, 0.8  # Healthy, no predicted failure
            
            # Estimate time to failure based on health trend
            recent_health = list(self.component_health_history[component])[-20:]
            
            if len(recent_health) >= 5:
                scores = [h.health_score for h in recent_health]
                x = np.arange(len(scores))
                
                if len(scores) > 1:
                    slope = np.polyfit(x, scores, 1)[0]
                    
                    if slope < -0.5:  # Degrading
                        # Estimate when health will reach critical level (10)
                        current_score = scores[-1]
                        if slope < 0:
                            time_to_critical = (current_score - 10) / abs(slope)
                            # Convert to actual time (assuming measurements every 5 minutes)
                            failure_time = datetime.utcnow() + timedelta(minutes=time_to_critical * 5)
                            confidence = min(0.9, abs(slope) / 2)
                            return failure_time, confidence
            
            # Default prediction for unhealthy components
            if current_health_score <= 30:
                failure_time = datetime.utcnow() + timedelta(hours=24)  # 24 hours estimate
                return failure_time, 0.6
            elif current_health_score <= 50:
                failure_time = datetime.utcnow() + timedelta(days=3)  # 3 days estimate
                return failure_time, 0.4
            
            return None, 0.3
            
        except Exception as e:
            logger.error(f"Error predicting failure time: {e}")
            return None, 0.0
    
    async def generate_maintenance_alerts(self, component_health: List[ComponentHealth]) -> List[MaintenanceAlert]:
        """Generate maintenance alerts based on component health"""
        alerts = []
        
        try:
            for health in component_health:
                alert = await self._create_maintenance_alert(health)
                if alert:
                    alerts.append(alert)
            
            # Store alerts in history
            for alert in alerts:
                self.alert_history.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating maintenance alerts: {e}")
            return []
    
    async def _create_maintenance_alert(self, health: ComponentHealth) -> Optional[MaintenanceAlert]:
        """Create maintenance alert for component health status"""
        try:
            # Only create alerts for components that need attention
            if health.health_status == HealthStatus.HEALTHY:
                return None
            
            alert_id = str(uuid.uuid4())
            
            # Determine alert type and severity
            if health.health_status == HealthStatus.CRITICAL or health.health_status == HealthStatus.FAILED:
                alert_type = MaintenanceType.EMERGENCY
                severity = "critical"
                failure_risk = FailureRisk.IMMINENT
            elif health.health_status == HealthStatus.DEGRADED:
                alert_type = MaintenanceType.CORRECTIVE
                severity = "high"
                failure_risk = FailureRisk.HIGH
            else:  # WARNING
                alert_type = MaintenanceType.PREVENTIVE
                severity = "medium"
                failure_risk = FailureRisk.MEDIUM
            
            # Generate message
            message = self._generate_alert_message(health)
            
            # Calculate time to failure
            time_to_failure = None
            if health.predicted_failure_time:
                time_to_failure = health.predicted_failure_time - datetime.utcnow()
            
            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(health)
            
            # Generate automated actions
            automated_actions = self._generate_automated_actions(health)
            
            # Determine if escalation is required
            escalation_required = (
                health.health_status in [HealthStatus.CRITICAL, HealthStatus.FAILED] or
                len(health.anomalies) > 2 or
                (time_to_failure and time_to_failure < timedelta(hours=1))
            )
            
            return MaintenanceAlert(
                alert_id=alert_id,
                timestamp=datetime.utcnow(),
                component=health.component,
                alert_type=alert_type,
                severity=severity,
                message=message,
                failure_risk=failure_risk,
                time_to_failure=time_to_failure,
                recommended_actions=recommended_actions,
                automated_actions=automated_actions,
                escalation_required=escalation_required
            )
            
        except Exception as e:
            logger.error(f"Error creating maintenance alert: {e}")
            return None
    
    def _generate_alert_message(self, health: ComponentHealth) -> str:
        """Generate human-readable alert message"""
        component_name = health.component.value.replace('_', ' ').title()
        
        if health.health_status == HealthStatus.FAILED:
            return f"{component_name} has failed. Immediate attention required."
        elif health.health_status == HealthStatus.CRITICAL:
            return f"{component_name} is in critical state (Health: {health.health_score:.1f}%). Urgent maintenance needed."
        elif health.health_status == HealthStatus.DEGRADED:
            return f"{component_name} performance is degraded (Health: {health.health_score:.1f}%). Maintenance recommended."
        else:
            return f"{component_name} showing warning signs (Health: {health.health_score:.1f}%). Preventive maintenance suggested."
    
    def _generate_recommended_actions(self, health: ComponentHealth) -> List[str]:
        """Generate recommended maintenance actions"""
        actions = []
        
        try:
            component = health.component
            
            if component == SystemComponent.CPU:
                if health.health_score < 30:
                    actions.extend([
                        "Scale up CPU resources immediately",
                        "Identify and optimize CPU-intensive processes",
                        "Implement CPU usage monitoring and alerting"
                    ])
                else:
                    actions.extend([
                        "Monitor CPU usage trends",
                        "Review process optimization opportunities",
                        "Consider load balancing improvements"
                    ])
            
            elif component == SystemComponent.MEMORY:
                if health.health_score < 30:
                    actions.extend([
                        "Force garbage collection",
                        "Identify memory leaks and fix immediately",
                        "Scale up memory resources",
                        "Restart services if necessary"
                    ])
                else:
                    actions.extend([
                        "Monitor memory usage patterns",
                        "Optimize memory allocation strategies",
                        "Review caching strategies"
                    ])
            
            elif component == SystemComponent.CACHE:
                if health.health_score < 30:
                    actions.extend([
                        "Restart cache service",
                        "Clear and rebuild cache",
                        "Increase cache memory allocation",
                        "Optimize cache key strategies"
                    ])
                else:
                    actions.extend([
                        "Analyze cache hit rate patterns",
                        "Optimize TTL policies",
                        "Implement cache warming strategies"
                    ])
            
            elif component == SystemComponent.DATABASE:
                if health.health_score < 30:
                    actions.extend([
                        "Optimize slow queries immediately",
                        "Increase connection pool size",
                        "Monitor disk space and I/O",
                        "Consider database scaling"
                    ])
                else:
                    actions.extend([
                        "Review query performance",
                        "Optimize database indexes",
                        "Monitor connection pool usage"
                    ])
            
            # Add general actions based on anomalies
            for anomaly in health.anomalies:
                if "high" in anomaly.lower() or "critical" in anomaly.lower():
                    actions.append(f"Address detected issue: {anomaly}")
            
            # Add default action if none specified
            if not actions:
                actions.append(f"Monitor {component.value} closely and investigate performance issues")
            
        except Exception as e:
            logger.error(f"Error generating recommended actions: {e}")
            actions = ["Manual investigation required due to error in action generation"]
        
        return actions
    
    def _generate_automated_actions(self, health: ComponentHealth) -> List[str]:
        """Generate automated actions that can be performed"""
        actions = []
        
        try:
            component = health.component
            
            # Only suggest automated actions for safe operations
            if component == SystemComponent.MEMORY and health.health_score < 50:
                actions.append("Trigger garbage collection")
            
            if component == SystemComponent.CACHE and health.health_score < 40:
                actions.append("Clear expired cache entries")
            
            # Log analysis and monitoring are always safe
            actions.extend([
                "Collect detailed performance metrics",
                "Generate diagnostic report",
                "Update monitoring alerts"
            ])
            
        except Exception as e:
            logger.error(f"Error generating automated actions: {e}")
        
        return actions
    
    async def create_maintenance_plan(self, planning_horizon: timedelta = timedelta(days=7)) -> MaintenancePlan:
        """Create comprehensive predictive maintenance plan"""
        plan_id = f"maintenance_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Assess health of all components
            component_health = []
            for component in SystemComponent:
                health = await self.assess_component_health(component)
                component_health.append(health)
            
            # Generate maintenance alerts
            alerts = await self.generate_maintenance_alerts(component_health)
            
            # Create maintenance actions
            maintenance_actions = await self._create_maintenance_actions(component_health, alerts)
            
            # Create maintenance schedule
            maintenance_schedule = self._create_maintenance_schedule(maintenance_actions, planning_horizon)
            
            # Assess risks
            risk_assessment = self._assess_maintenance_risks(maintenance_actions)
            
            # Estimate costs
            cost_estimation = self._estimate_maintenance_costs(maintenance_actions)
            
            plan = MaintenancePlan(
                plan_id=plan_id,
                timestamp=datetime.utcnow(),
                planning_horizon=planning_horizon,
                component_health=component_health,
                alerts=alerts,
                maintenance_actions=maintenance_actions,
                maintenance_schedule=maintenance_schedule,
                risk_assessment=risk_assessment,
                cost_estimation=cost_estimation
            )
            
            # Store plan in history
            self.maintenance_history.append(plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating maintenance plan: {e}")
            return MaintenancePlan(
                plan_id=plan_id,
                timestamp=datetime.utcnow(),
                planning_horizon=planning_horizon,
                component_health=[],
                alerts=[],
                maintenance_actions=[],
                maintenance_schedule=[],
                risk_assessment={'error': str(e)},
                cost_estimation={}
            )
    
    async def _create_maintenance_actions(self, component_health: List[ComponentHealth],
                                        alerts: List[MaintenanceAlert]) -> List[MaintenanceAction]:
        """Create maintenance actions based on health and alerts"""
        actions = []
        
        try:
            # Create actions for each alert
            for alert in alerts:
                action = self._create_action_from_alert(alert)
                if action:
                    actions.append(action)
            
            # Create scheduled preventive maintenance actions
            for health in component_health:
                if health.health_status == HealthStatus.HEALTHY:
                    # Check if scheduled maintenance is due
                    config = self.component_configs.get(health.component, {})
                    maintenance_interval = config.get('maintenance_interval_hours', 168)
                    
                    # Check if maintenance is due (simplified check)
                    if self._is_maintenance_due(health.component, maintenance_interval):
                        action = self._create_preventive_action(health)
                        if action:
                            actions.append(action)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error creating maintenance actions: {e}")
            return []
    
    def _create_action_from_alert(self, alert: MaintenanceAlert) -> Optional[MaintenanceAction]:
        """Create maintenance action from alert"""
        try:
            action_id = str(uuid.uuid4())
            
            # Determine automation level
            if alert.alert_type == MaintenanceType.EMERGENCY:
                automation_level = "manual"  # Emergency requires manual intervention
            elif alert.alert_type == MaintenanceType.PREVENTIVE:
                automation_level = "semi_automated"
            else:
                automation_level = "manual"
            
            # Estimate duration based on component and action type
            if alert.alert_type == MaintenanceType.EMERGENCY:
                duration = timedelta(hours=4)
            elif alert.alert_type == MaintenanceType.CORRECTIVE:
                duration = timedelta(hours=2)
            else:
                duration = timedelta(hours=1)
            
            return MaintenanceAction(
                action_id=action_id,
                timestamp=datetime.utcnow(),
                component=alert.component,
                action_type=alert.alert_type,
                description=f"Address {alert.severity} issue: {alert.message}",
                procedure=alert.recommended_actions,
                estimated_duration=duration,
                impact_assessment={
                    'service_disruption': 'high' if alert.alert_type == MaintenanceType.EMERGENCY else 'medium',
                    'user_impact': 'high' if alert.severity == 'critical' else 'medium',
                    'business_impact': alert.severity
                },
                prerequisites=["System backup", "Notification to stakeholders"],
                success_criteria=[
                    f"Component health score > 70%",
                    "No critical alerts remaining",
                    "System performance restored"
                ],
                rollback_plan=[
                    "Restore from backup if available",
                    "Revert configuration changes",
                    "Escalate to senior team if issues persist"
                ],
                automation_level=automation_level
            )
            
        except Exception as e:
            logger.error(f"Error creating action from alert: {e}")
            return None
    
    def _create_preventive_action(self, health: ComponentHealth) -> Optional[MaintenanceAction]:
        """Create preventive maintenance action"""
        try:
            action_id = str(uuid.uuid4())
            component_name = health.component.value.replace('_', ' ').title()
            
            return MaintenanceAction(
                action_id=action_id,
                timestamp=datetime.utcnow(),
                component=health.component,
                action_type=MaintenanceType.PREVENTIVE,
                description=f"Scheduled preventive maintenance for {component_name}",
                procedure=[
                    f"Review {component_name} performance metrics",
                    "Perform routine optimization",
                    "Update monitoring configurations",
                    "Verify component health"
                ],
                estimated_duration=timedelta(minutes=30),
                impact_assessment={
                    'service_disruption': 'low',
                    'user_impact': 'minimal',
                    'business_impact': 'low'
                },
                prerequisites=["System health check", "Performance baseline"],
                success_criteria=[
                    "Component health maintained above 80%",
                    "No new performance issues",
                    "Monitoring alerts updated"
                ],
                rollback_plan=[
                    "Restore previous configuration",
                    "Monitor for 30 minutes",
                    "Document any issues"
                ],
                automation_level="fully_automated"
            )
            
        except Exception as e:
            logger.error(f"Error creating preventive action: {e}")
            return None
    
    def _is_maintenance_due(self, component: SystemComponent, interval_hours: int) -> bool:
        """Check if scheduled maintenance is due for component"""
        try:
            # Simplified check - in production would track actual maintenance history
            current_hour = datetime.utcnow().hour
            return current_hour % interval_hours == 0
        except:
            return False
    
    def _create_maintenance_schedule(self, actions: List[MaintenanceAction],
                                   planning_horizon: timedelta) -> List[Dict[str, Any]]:
        """Create maintenance schedule for actions"""
        schedule = []
        
        try:
            current_time = datetime.utcnow()
            
            # Sort actions by priority (emergency first)
            priority_order = {
                MaintenanceType.EMERGENCY: 1,
                MaintenanceType.CORRECTIVE: 2,
                MaintenanceType.PREDICTIVE: 3,
                MaintenanceType.PREVENTIVE: 4
            }
            
            sorted_actions = sorted(actions, key=lambda a: priority_order.get(a.action_type, 5))
            
            # Schedule actions with appropriate spacing
            scheduled_time = current_time
            
            for action in sorted_actions:
                # Emergency actions scheduled immediately
                if action.action_type == MaintenanceType.EMERGENCY:
                    scheduled_time = current_time + timedelta(minutes=15)
                # Other actions spaced out
                else:
                    scheduled_time += timedelta(hours=2)
                
                # Don't schedule beyond planning horizon
                if scheduled_time > current_time + planning_horizon:
                    break
                
                schedule.append({
                    'action_id': action.action_id,
                    'component': action.component.value,
                    'action_type': action.action_type.value,
                    'scheduled_time': scheduled_time.isoformat(),
                    'estimated_duration_minutes': action.estimated_duration.total_seconds() / 60,
                    'automation_level': action.automation_level,
                    'priority': priority_order.get(action.action_type, 5)
                })
            
        except Exception as e:
            logger.error(f"Error creating maintenance schedule: {e}")
        
        return schedule
    
    def _assess_maintenance_risks(self, actions: List[MaintenanceAction]) -> Dict[str, Any]:
        """Assess risks associated with maintenance plan"""
        try:
            emergency_count = sum(1 for a in actions if a.action_type == MaintenanceType.EMERGENCY)
            manual_count = sum(1 for a in actions if a.automation_level == "manual")
            
            risk_level = "low"
            if emergency_count > 2:
                risk_level = "high"
            elif emergency_count > 0 or manual_count > 3:
                risk_level = "medium"
            
            return {
                'overall_risk': risk_level,
                'emergency_actions': emergency_count,
                'manual_actions': manual_count,
                'service_disruption_risk': 'high' if emergency_count > 1 else 'medium',
                'coordination_complexity': 'high' if len(actions) > 5 else 'medium',
                'recommendations': [
                    'Ensure adequate staffing for manual actions',
                    'Prepare rollback procedures',
                    'Coordinate with stakeholders',
                    'Monitor system closely during maintenance'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error assessing maintenance risks: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}
    
    def _estimate_maintenance_costs(self, actions: List[MaintenanceAction]) -> Dict[str, float]:
        """Estimate costs for maintenance actions"""
        try:
            # Simplified cost estimation
            cost_per_hour = {
                'manual': 100.0,
                'semi_automated': 50.0,
                'fully_automated': 10.0
            }
            
            total_cost = 0
            breakdown = {}
            
            for action in actions:
                duration_hours = action.estimated_duration.total_seconds() / 3600
                rate = cost_per_hour.get(action.automation_level, 75.0)
                action_cost = duration_hours * rate
                
                total_cost += action_cost
                breakdown[action.action_id] = action_cost
            
            return {
                'total_estimated_cost': total_cost,
                'cost_breakdown': breakdown,
                'manual_action_costs': sum(cost for action, cost in breakdown.items() 
                                         if any(a for a in actions if a.action_id == action and a.automation_level == 'manual')),
                'automated_action_costs': sum(cost for action, cost in breakdown.items() 
                                            if any(a for a in actions if a.action_id == action and a.automation_level == 'fully_automated'))
            }
            
        except Exception as e:
            logger.error(f"Error estimating costs: {e}")
            return {'total_estimated_cost': 0, 'error': str(e)}
    
    async def get_maintenance_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get insights from maintenance history and predictions"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            recent_plans = [p for p in self.maintenance_history if p.timestamp >= cutoff_time]
            recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
            
            # Analyze component health trends
            component_trends = {}
            for component in SystemComponent:
                recent_health = [h for h in self.component_health_history[component] 
                               if h.timestamp >= cutoff_time]
                if recent_health:
                    avg_health = statistics.mean(h.health_score for h in recent_health)
                    trend = self._analyze_health_trend(component)
                    component_trends[component.value] = {
                        'average_health_score': avg_health,
                        'trend': trend,
                        'health_measurements': len(recent_health)
                    }
            
            # Analyze alert patterns
            alert_stats = defaultdict(int)
            for alert in recent_alerts:
                alert_stats[f"{alert.component.value}_{alert.alert_type.value}"] += 1
            
            insights = {
                'analysis_period_days': days,
                'component_health_trends': component_trends,
                'maintenance_summary': {
                    'total_plans_created': len(recent_plans),
                    'total_alerts_generated': len(recent_alerts),
                    'emergency_alerts': len([a for a in recent_alerts if a.alert_type == MaintenanceType.EMERGENCY]),
                    'preventive_maintenance_opportunities': len([a for a in recent_alerts if a.alert_type == MaintenanceType.PREVENTIVE])
                },
                'alert_patterns': dict(alert_stats),
                'reliability_metrics': self._calculate_reliability_metrics(days),
                'maintenance_effectiveness': self._analyze_maintenance_effectiveness(recent_plans),
                'recommendations': self._generate_maintenance_insights_recommendations(component_trends, recent_alerts)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting maintenance insights: {e}")
            return {'error': str(e)}
    
    def _calculate_reliability_metrics(self, days: int) -> Dict[str, float]:
        """Calculate system reliability metrics"""
        try:
            # Calculate uptime and availability metrics
            total_components = len(SystemComponent)
            
            # Simplified calculation - in production would track actual uptime
            healthy_component_hours = 0
            total_component_hours = total_components * days * 24
            
            for component in SystemComponent:
                recent_health = list(self.component_health_history[component])
                if recent_health:
                    healthy_measurements = sum(1 for h in recent_health if h.health_status == HealthStatus.HEALTHY)
                    total_measurements = len(recent_health)
                    component_availability = (healthy_measurements / total_measurements) if total_measurements > 0 else 1.0
                    healthy_component_hours += component_availability * days * 24
                else:
                    healthy_component_hours += days * 24  # Assume healthy if no data
            
            overall_availability = (healthy_component_hours / total_component_hours) * 100
            
            return {
                'overall_availability_percent': overall_availability,
                'mean_time_between_failures_hours': 72,  # Placeholder
                'mean_time_to_repair_hours': 2,          # Placeholder
                'system_reliability_score': min(100, overall_availability + 10)  # Adjusted score
            }
            
        except Exception as e:
            logger.error(f"Error calculating reliability metrics: {e}")
            return {
                'overall_availability_percent': 99.0,
                'mean_time_between_failures_hours': 168,
                'mean_time_to_repair_hours': 4,
                'system_reliability_score': 95.0
            }
    
    def _analyze_maintenance_effectiveness(self, plans: List[MaintenancePlan]) -> Dict[str, Any]:
        """Analyze effectiveness of maintenance plans"""
        try:
            if not plans:
                return {'effectiveness': 'no_data'}
            
            total_actions = sum(len(p.maintenance_actions) for p in plans)
            emergency_actions = sum(len([a for a in p.maintenance_actions if a.action_type == MaintenanceType.EMERGENCY]) for p in plans)
            preventive_actions = sum(len([a for a in p.maintenance_actions if a.action_type == MaintenanceType.PREVENTIVE]) for p in plans)
            
            preventive_ratio = (preventive_actions / total_actions) if total_actions > 0 else 0
            
            return {
                'total_maintenance_actions': total_actions,
                'emergency_action_ratio': (emergency_actions / total_actions) if total_actions > 0 else 0,
                'preventive_action_ratio': preventive_ratio,
                'avg_actions_per_plan': total_actions / len(plans),
                'maintenance_approach': 'proactive' if preventive_ratio > 0.6 else 'reactive',
                'effectiveness_score': max(0, 100 - (emergency_actions / max(1, total_actions) * 100))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing maintenance effectiveness: {e}")
            return {'effectiveness': 'error', 'error': str(e)}
    
    def _generate_maintenance_insights_recommendations(self, component_trends: Dict[str, Any],
                                                     recent_alerts: List[MaintenanceAlert]) -> List[str]:
        """Generate recommendations based on maintenance insights"""
        recommendations = []
        
        try:
            # Analyze component trends
            for component, trend_data in component_trends.items():
                avg_health = trend_data.get('average_health_score', 100)
                trend = trend_data.get('trend', 'stable')
                
                if avg_health < 60:
                    recommendations.append(f"Focus attention on {component} - consistently low health score")
                elif trend == 'degrading':
                    recommendations.append(f"Monitor {component} closely - showing degrading trend")
            
            # Analyze alert patterns
            emergency_alerts = [a for a in recent_alerts if a.alert_type == MaintenanceType.EMERGENCY]
            if len(emergency_alerts) > 3:
                recommendations.append("High number of emergency alerts - review preventive maintenance strategy")
            
            # Component-specific recommendations
            frequent_components = defaultdict(int)
            for alert in recent_alerts:
                frequent_components[alert.component.value] += 1
            
            for component, count in frequent_components.items():
                if count >= 3:
                    recommendations.append(f"Increase monitoring and preventive maintenance for {component}")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Maintenance strategy appears effective - continue current approach")
            
        except Exception as e:
            logger.error(f"Error generating insights recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review suggested")
        
        return recommendations

# Global predictive maintenance system instance
predictive_maintenance = PredictiveMaintenanceSystem()

# Convenience functions
async def assess_system_health() -> List[ComponentHealth]:
    """Assess health of all system components"""
    health_results = []
    for component in SystemComponent:
        health = await predictive_maintenance.assess_component_health(component)
        health_results.append(health)
    return health_results

async def create_maintenance_plan(planning_horizon: timedelta = timedelta(days=7)) -> MaintenancePlan:
    """Create comprehensive maintenance plan"""
    return await predictive_maintenance.create_maintenance_plan(planning_horizon)

async def get_maintenance_insights(days: int = 30) -> Dict[str, Any]:
    """Get maintenance insights and analytics"""
    return await predictive_maintenance.get_maintenance_insights(days)

async def generate_maintenance_alerts() -> List[MaintenanceAlert]:
    """Generate current maintenance alerts"""
    component_health = await assess_system_health()
    return await predictive_maintenance.generate_maintenance_alerts(component_health)

__all__ = [
    'PredictiveMaintenanceSystem',
    'predictive_maintenance',
    'SystemComponent',
    'HealthStatus',
    'MaintenanceType',
    'FailureRisk',
    'ComponentHealth',
    'MaintenanceAlert',
    'MaintenanceAction',
    'MaintenancePlan',
    'assess_system_health',
    'create_maintenance_plan',
    'get_maintenance_insights',
    'generate_maintenance_alerts'
]