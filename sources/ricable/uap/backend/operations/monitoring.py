# File: backend/operations/monitoring.py
"""
Advanced monitoring and anomaly detection for UAP platform operations.
Extends the existing Prometheus monitoring with ML-based anomaly detection,
predictive alerting, and intelligent performance analysis.
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import psutil
import aioredis
from prometheus_client import Counter, Gauge, Histogram

from ..monitoring.metrics.prometheus_metrics import prometheus_metrics
from ..cache.redis_cache import get_redis_client

# Configure logging
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR_RATE = "error_rate"
    TRAFFIC = "traffic"
    LATENCY = "latency"
    SECURITY = "security"
    COST = "cost"

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str]
    metric_name: str

@dataclass
class AnomalyEvent:
    """An detected anomaly event"""
    id: str
    timestamp: float
    anomaly_type: AnomalyType
    severity: AlertSeverity
    metric_name: str
    actual_value: float
    expected_range: Tuple[float, float]
    deviation_score: float
    labels: Dict[str, str]
    description: str
    recommendation: str

@dataclass
class PerformanceMetrics:
    """System performance metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: Tuple[float, float, float]
    
class AnomalyDetector:
    """
    ML-based anomaly detection system for UAP metrics.
    Uses Isolation Forest and statistical methods to detect anomalies.
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metric_history: Dict[str, List[MetricPoint]] = {}
        self.anomaly_threshold = 0.1  # Anomaly score threshold
        self.history_size = 1000  # Number of data points to keep
        self.training_interval = 300  # Retrain models every 5 minutes
        self.last_training: Dict[str, float] = {}
        
        # Prometheus metrics for anomaly detection
        self.anomaly_counter = Counter(
            'uap_anomalies_detected_total',
            'Total number of anomalies detected',
            ['metric_name', 'anomaly_type', 'severity']
        )
        
        self.anomaly_score_gauge = Gauge(
            'uap_anomaly_score',
            'Current anomaly score for metrics',
            ['metric_name']
        )
        
    async def add_metric_point(self, metric_name: str, value: float, 
                              labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new metric point and check for anomalies"""
        labels = labels or {}
        timestamp = time.time()
        
        # Create metric point
        point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels,
            metric_name=metric_name
        )
        
        # Add to history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append(point)
        
        # Maintain history size
        if len(self.metric_history[metric_name]) > self.history_size:
            self.metric_history[metric_name] = self.metric_history[metric_name][-self.history_size:]
        
        # Check if we need to retrain the model
        if self._should_retrain_model(metric_name):
            await self._train_model(metric_name)
        
        # Detect anomalies
        anomaly = await self._detect_anomaly(metric_name, point)
        if anomaly:
            await self._handle_anomaly(anomaly)
    
    def _should_retrain_model(self, metric_name: str) -> bool:
        """Check if model should be retrained"""
        if metric_name not in self.last_training:
            return len(self.metric_history.get(metric_name, [])) >= 50
        
        time_since_training = time.time() - self.last_training[metric_name]
        return time_since_training >= self.training_interval
    
    async def _train_model(self, metric_name: str) -> None:
        """Train anomaly detection model for a specific metric"""
        if metric_name not in self.metric_history:
            return
        
        history = self.metric_history[metric_name]
        if len(history) < 50:  # Need minimum data points
            return
        
        # Prepare training data
        values = np.array([point.value for point in history]).reshape(-1, 1)
        
        # Scale the data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        model.fit(scaled_values)
        
        # Store model and scaler
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        self.last_training[metric_name] = time.time()
        
        logger.info(f"Trained anomaly detection model for metric: {metric_name}")
    
    async def _detect_anomaly(self, metric_name: str, point: MetricPoint) -> Optional[AnomalyEvent]:
        """Detect if a metric point is anomalous"""
        if metric_name not in self.models or metric_name not in self.scalers:
            return None
        
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]
        
        # Scale the current value
        scaled_value = scaler.transform([[point.value]])
        
        # Get anomaly score
        anomaly_score = model.decision_function(scaled_value)[0]
        is_anomaly = model.predict(scaled_value)[0] == -1
        
        # Update Prometheus metric
        self.anomaly_score_gauge.labels(metric_name=metric_name).set(anomaly_score)
        
        if not is_anomaly:
            return None
        
        # Calculate expected range from recent history
        recent_values = [p.value for p in self.metric_history[metric_name][-100:]]
        expected_min = min(recent_values)
        expected_max = max(recent_values)
        expected_range = (expected_min, expected_max)
        
        # Determine anomaly type and severity
        anomaly_type = self._classify_anomaly_type(metric_name, point.value, recent_values)
        severity = self._determine_severity(anomaly_score, point.value, recent_values)
        
        # Create anomaly event
        anomaly = AnomalyEvent(
            id=f"{metric_name}_{int(point.timestamp)}_{hash(str(point.labels))}",
            timestamp=point.timestamp,
            anomaly_type=anomaly_type,
            severity=severity,
            metric_name=metric_name,
            actual_value=point.value,
            expected_range=expected_range,
            deviation_score=abs(anomaly_score),
            labels=point.labels,
            description=self._generate_description(metric_name, point.value, expected_range, anomaly_type),
            recommendation=self._generate_recommendation(metric_name, anomaly_type, point.value)
        )
        
        return anomaly
    
    def _classify_anomaly_type(self, metric_name: str, value: float, recent_values: List[float]) -> AnomalyType:
        """Classify the type of anomaly based on metric name and patterns"""
        metric_lower = metric_name.lower()
        
        if any(keyword in metric_lower for keyword in ['cpu', 'memory', 'disk']):
            return AnomalyType.RESOURCE
        elif any(keyword in metric_lower for keyword in ['response_time', 'latency', 'duration']):
            return AnomalyType.LATENCY
        elif any(keyword in metric_lower for keyword in ['error', 'failed', 'exception']):
            return AnomalyType.ERROR_RATE
        elif any(keyword in metric_lower for keyword in ['request', 'traffic', 'connection']):
            return AnomalyType.TRAFFIC
        elif any(keyword in metric_lower for keyword in ['cost', 'billing', 'spend']):
            return AnomalyType.COST
        elif any(keyword in metric_lower for keyword in ['security', 'auth', 'login']):
            return AnomalyType.SECURITY
        else:
            return AnomalyType.PERFORMANCE
    
    def _determine_severity(self, anomaly_score: float, value: float, recent_values: List[float]) -> AlertSeverity:
        """Determine the severity of an anomaly"""
        # Calculate how extreme the deviation is
        mean_val = statistics.mean(recent_values)
        std_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 1
        
        z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
        
        if anomaly_score < -0.5 or z_score > 3:
            return AlertSeverity.CRITICAL
        elif anomaly_score < -0.3 or z_score > 2:
            return AlertSeverity.WARNING
        elif anomaly_score < -0.1 or z_score > 1:
            return AlertSeverity.INFO
        else:
            return AlertSeverity.DEBUG
    
    def _generate_description(self, metric_name: str, value: float, 
                            expected_range: Tuple[float, float], anomaly_type: AnomalyType) -> str:
        """Generate a human-readable description of the anomaly"""
        expected_min, expected_max = expected_range
        
        if value > expected_max:
            direction = "higher"
            percentage = ((value - expected_max) / expected_max) * 100
        else:
            direction = "lower"
            percentage = ((expected_min - value) / expected_min) * 100
        
        return (f"{anomaly_type.value.title()} anomaly detected: {metric_name} is "
                f"{percentage:.1f}% {direction} than expected "
                f"(actual: {value:.2f}, expected range: {expected_min:.2f}-{expected_max:.2f})")
    
    def _generate_recommendation(self, metric_name: str, anomaly_type: AnomalyType, value: float) -> str:
        """Generate actionable recommendations for anomalies"""
        recommendations = {
            AnomalyType.RESOURCE: "Check system resources and consider scaling up instances or optimizing resource usage.",
            AnomalyType.LATENCY: "Investigate network latency, database performance, or application bottlenecks.",
            AnomalyType.ERROR_RATE: "Review application logs, check for deployment issues, or validate input data.",
            AnomalyType.TRAFFIC: "Verify if traffic spike is expected or consider DDoS protection measures.",
            AnomalyType.COST: "Review resource usage and billing alerts, check for unexpected resource provisioning.",
            AnomalyType.SECURITY: "Investigate potential security incidents, review access logs and authentication.",
            AnomalyType.PERFORMANCE: "Analyze application performance metrics and optimize bottlenecks."
        }
        
        base_recommendation = recommendations.get(anomaly_type, "Investigate the anomaly and take appropriate action.")
        
        # Add metric-specific recommendations
        metric_lower = metric_name.lower()
        if 'cpu' in metric_lower and value > 80:
            return f"{base_recommendation} Consider adding more CPU resources or optimizing CPU-intensive processes."
        elif 'memory' in metric_lower and value > 85:
            return f"{base_recommendation} Consider increasing memory allocation or investigating memory leaks."
        elif 'disk' in metric_lower and value > 90:
            return f"{base_recommendation} Free up disk space or expand storage capacity immediately."
        
        return base_recommendation
    
    async def _handle_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Handle detected anomaly by logging and alerting"""
        # Update Prometheus counter
        self.anomaly_counter.labels(
            metric_name=anomaly.metric_name,
            anomaly_type=anomaly.anomaly_type.value,
            severity=anomaly.severity.value
        ).inc()
        
        # Log the anomaly
        log_level = {
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.DEBUG: logging.DEBUG
        }[anomaly.severity]
        
        logger.log(log_level, f"Anomaly detected: {anomaly.description}")
        
        # Store in Redis for dashboard
        await self._store_anomaly(anomaly)
        
        # Trigger alerts for critical anomalies
        if anomaly.severity in [AlertSeverity.CRITICAL, AlertSeverity.WARNING]:
            await self._trigger_alert(anomaly)
    
    async def _store_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Store anomaly in Redis for dashboard access"""
        key = f"anomaly:{anomaly.id}"
        await self.redis_client.setex(
            key, 
            86400,  # Store for 24 hours
            json.dumps(asdict(anomaly), default=str)
        )
        
        # Add to recent anomalies list
        recent_key = "anomalies:recent"
        await self.redis_client.lpush(recent_key, anomaly.id)
        await self.redis_client.ltrim(recent_key, 0, 99)  # Keep last 100
        await self.redis_client.expire(recent_key, 86400)
    
    async def _trigger_alert(self, anomaly: AnomalyEvent) -> None:
        """Trigger external alerts for critical anomalies"""
        # This would integrate with external alerting systems
        # like PagerDuty, Slack, email, etc.
        logger.critical(f"ALERT: {anomaly.description} - {anomaly.recommendation}")
    
    async def get_recent_anomalies(self, limit: int = 50) -> List[AnomalyEvent]:
        """Get recent anomalies for dashboard display"""
        recent_key = "anomalies:recent"
        anomaly_ids = await self.redis_client.lrange(recent_key, 0, limit - 1)
        
        anomalies = []
        for anomaly_id in anomaly_ids:
            key = f"anomaly:{anomaly_id.decode()}"
            data = await self.redis_client.get(key)
            if data:
                anomaly_dict = json.loads(data)
                # Convert back to enum types
                anomaly_dict['anomaly_type'] = AnomalyType(anomaly_dict['anomaly_type'])
                anomaly_dict['severity'] = AlertSeverity(anomaly_dict['severity'])
                anomalies.append(AnomalyEvent(**anomaly_dict))
        
        return anomalies

class OperationsMonitor:
    """
    Comprehensive operations monitoring system that combines
    system metrics, application metrics, and anomaly detection.
    """
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.alert_manager = AlertManager()
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_size = 2880  # 24 hours at 30-second intervals
        
    async def start_monitoring(self) -> None:
        """Start the monitoring loop"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting operations monitoring...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._application_monitoring_loop()),
            asyncio.create_task(self._performance_analysis_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Stopping operations monitoring...")
    
    async def _system_monitoring_loop(self) -> None:
        """Monitor system-level metrics"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                process_count = len(psutil.pids())
                load_avg = psutil.getloadavg()
                
                # Create performance snapshot
                performance = PerformanceMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_usage_percent=(disk.used / disk.total) * 100,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv
                    },
                    process_count=process_count,
                    load_average=load_avg
                )
                
                # Store performance metrics
                self.performance_history.append(performance)
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history = self.performance_history[-self.max_history_size:]
                
                # Update Prometheus metrics
                prometheus_metrics.update_system_metrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_bytes=memory.used,
                    disk_usage_percent=(disk.used / disk.total) * 100,
                    process_count=process_count
                )
                
                # Check for anomalies
                await self.anomaly_detector.add_metric_point("system_cpu_percent", cpu_percent)
                await self.anomaly_detector.add_metric_point("system_memory_percent", memory.percent)
                await self.anomaly_detector.add_metric_point("system_disk_percent", (disk.used / disk.total) * 100)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _application_monitoring_loop(self) -> None:
        """Monitor application-specific metrics"""
        while self.monitoring_active:
            try:
                # This would collect application-specific metrics
                # For now, we'll simulate some metrics
                timestamp = time.time()
                
                # Simulate some application metrics
                response_times = [0.1, 0.2, 0.15, 0.3]  # Would come from actual monitoring
                error_rate = 0.02  # 2% error rate
                request_count = 100  # Requests per interval
                
                avg_response_time = statistics.mean(response_times)
                
                # Check for anomalies in application metrics
                await self.anomaly_detector.add_metric_point("app_response_time", avg_response_time)
                await self.anomaly_detector.add_metric_point("app_error_rate", error_rate)
                await self.anomaly_detector.add_metric_point("app_request_count", request_count)
                
            except Exception as e:
                logger.error(f"Application monitoring error: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _performance_analysis_loop(self) -> None:
        """Analyze performance trends and patterns"""
        while self.monitoring_active:
            try:
                if len(self.performance_history) >= 10:
                    await self.performance_analyzer.analyze_performance_trends(self.performance_history)
                
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
            
            await asyncio.sleep(300)  # Analyze every 5 minutes
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        latest = self.performance_history[-1]
        recent_anomalies = await self.anomaly_detector.get_recent_anomalies(limit=10)
        
        return {
            "timestamp": latest.timestamp,
            "system": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_usage_percent": latest.disk_usage_percent,
                "process_count": latest.process_count,
                "load_average": latest.load_average
            },
            "monitoring": {
                "active": self.monitoring_active,
                "history_size": len(self.performance_history)
            },
            "anomalies": {
                "recent_count": len(recent_anomalies),
                "critical_count": len([a for a in recent_anomalies if a.severity == AlertSeverity.CRITICAL]),
                "warning_count": len([a for a in recent_anomalies if a.severity == AlertSeverity.WARNING])
            }
        }

class PerformanceAnalyzer:
    """Analyzes performance trends and provides insights"""
    
    def __init__(self):
        self.trend_window = 100  # Number of data points for trend analysis
        
    async def analyze_performance_trends(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance trends and identify patterns"""
        if len(performance_history) < self.trend_window:
            return {"status": "insufficient_data"}
        
        recent_data = performance_history[-self.trend_window:]
        
        # Analyze CPU trends
        cpu_values = [p.cpu_percent for p in recent_data]
        cpu_trend = self._calculate_trend(cpu_values)
        
        # Analyze memory trends
        memory_values = [p.memory_percent for p in recent_data]
        memory_trend = self._calculate_trend(memory_values)
        
        # Analyze disk trends
        disk_values = [p.disk_usage_percent for p in recent_data]
        disk_trend = self._calculate_trend(disk_values)
        
        analysis = {
            "timestamp": time.time(),
            "trends": {
                "cpu": {
                    "direction": cpu_trend["direction"],
                    "slope": cpu_trend["slope"],
                    "current_avg": statistics.mean(cpu_values[-10:]),
                    "prediction": self._predict_resource_exhaustion(cpu_values, 90)
                },
                "memory": {
                    "direction": memory_trend["direction"],
                    "slope": memory_trend["slope"],
                    "current_avg": statistics.mean(memory_values[-10:]),
                    "prediction": self._predict_resource_exhaustion(memory_values, 90)
                },
                "disk": {
                    "direction": disk_trend["direction"],
                    "slope": disk_trend["slope"],
                    "current_avg": statistics.mean(disk_values[-10:]),
                    "prediction": self._predict_resource_exhaustion(disk_values, 95)
                }
            }
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and slope"""
        if len(values) < 2:
            return {"direction": "stable", "slope": 0}
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {"direction": direction, "slope": slope}
    
    def _predict_resource_exhaustion(self, values: List[float], threshold: float) -> Optional[Dict[str, Any]]:
        """Predict when a resource might reach exhaustion"""
        if len(values) < 10:
            return None
        
        # Calculate trend
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            # Fit linear regression
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            
            if slope <= 0:  # Not increasing
                return None
            
            current_value = values[-1]
            if current_value >= threshold:
                return {"status": "exceeded", "eta_minutes": 0}
            
            # Calculate time to reach threshold
            steps_to_threshold = (threshold - current_value) / slope
            minutes_to_threshold = steps_to_threshold * 0.5  # Assuming 30-second intervals
            
            if minutes_to_threshold > 0 and minutes_to_threshold < 1440:  # Within 24 hours
                return {
                    "status": "predicted",
                    "eta_minutes": minutes_to_threshold,
                    "confidence": min(0.95, abs(slope) * 10)  # Simple confidence metric
                }
        
        except Exception:
            pass
        
        return None

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alert_channels = []
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
    async def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert through configured channels"""
        self.alert_history.append({
            **alert,
            "timestamp": time.time()
        })
        
        # Maintain history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # Log the alert
        logger.warning(f"Alert: {alert.get('title', 'Unknown')} - {alert.get('message', '')}")
        
        # Send to external channels (would be implemented based on requirements)
        for channel in self.alert_channels:
            await self._send_to_channel(channel, alert)
    
    async def _send_to_channel(self, channel: str, alert: Dict[str, Any]) -> None:
        """Send alert to specific channel"""
        # This would implement actual channel integrations
        # like Slack, email, PagerDuty, etc.
        pass

# Global instances
operations_monitor = OperationsMonitor()
anomaly_detector = AnomalyDetector()