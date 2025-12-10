# File: backend/monitoring/alerting/alerts.py
"""
Alerting system for UAP platform with configurable thresholds.
Provides real-time monitoring and notifications for performance, errors, and availability.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid

from ..logs.logger import uap_logger, EventType, LogLevel
from ..metrics.performance import performance_monitor

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold: Union[float, int]
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    time_window_minutes: int = 5
    evaluation_interval_seconds: int = 30
    min_data_points: int = 3
    tags_filter: Dict[str, str] = None
    enabled: bool = True
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.tags_filter is None:
            self.tags_filter = {}
        if self.notification_channels is None:
            self.notification_channels = ["log"]

@dataclass
class Alert:
    """Active or historical alert"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    component: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    current_value: Union[float, int] = None
    threshold_value: Union[float, int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['triggered_at'] = self.triggered_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        return data

class AlertManager:
    """Main alerting system manager"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_handlers: Dict[str, Callable] = {}
        self.evaluation_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Initialize default rules
        self._create_default_rules()
        
        # Register default notification handlers
        self._register_default_handlers()
    
    def _create_default_rules(self):
        """Create default alerting rules based on UAP requirements"""
        
        # Agent response time threshold (target: <2s p95)
        self.add_rule(AlertRule(
            rule_id="agent_response_time_p95",
            name="Agent Response Time P95 High",
            description="Agent 95th percentile response time exceeds 2 seconds",
            severity=AlertSeverity.WARNING,
            metric_name="agent.response_time",
            threshold=2000,  # 2 seconds in milliseconds
            comparison="gt",
            time_window_minutes=5,
            notification_channels=["log", "console"]
        ))
        
        # Critical response time threshold
        self.add_rule(AlertRule(
            rule_id="agent_response_time_critical",
            name="Agent Response Time Critical",
            description="Agent response time exceeds 5 seconds",
            severity=AlertSeverity.CRITICAL,
            metric_name="agent.response_time",
            threshold=5000,  # 5 seconds in milliseconds
            comparison="gt",
            time_window_minutes=2,
            notification_channels=["log", "console"]
        ))
        
        # System CPU usage
        self.add_rule(AlertRule(
            rule_id="system_cpu_high",
            name="High CPU Usage",
            description="System CPU usage exceeds 80%",
            severity=AlertSeverity.WARNING,
            metric_name="system.cpu_percent",
            threshold=80,
            comparison="gt",
            time_window_minutes=3,
            notification_channels=["log", "console"]
        ))
        
        # System memory usage
        self.add_rule(AlertRule(
            rule_id="system_memory_high",
            name="High Memory Usage",
            description="System memory usage exceeds 85%",
            severity=AlertSeverity.WARNING,
            metric_name="system.memory_percent",
            threshold=85,
            comparison="gt",
            time_window_minutes=3,
            notification_channels=["log", "console"]
        ))
        
        # WebSocket connection stability
        self.add_rule(AlertRule(
            rule_id="websocket_connections_high",
            name="High WebSocket Connections",
            description="Active WebSocket connections exceed threshold",
            severity=AlertSeverity.INFO,
            metric_name="websocket.active_connections",
            threshold=1000,
            comparison="gt",
            time_window_minutes=1,
            notification_channels=["log"]
        ))
        
        # Error rate threshold
        self.add_rule(AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            description="Agent error rate exceeds 5%",
            severity=AlertSeverity.WARNING,
            metric_name="agent_error_rate",
            threshold=5,
            comparison="gt",
            time_window_minutes=10,
            notification_channels=["log", "console"]
        ))
        
        # Disk space
        self.add_rule(AlertRule(
            rule_id="disk_space_high",
            name="High Disk Usage",
            description="Disk usage exceeds 90%",
            severity=AlertSeverity.CRITICAL,
            metric_name="system.disk_usage_percent",
            threshold=90,
            comparison="gt",
            time_window_minutes=5,
            notification_channels=["log", "console"]
        ))
    
    def _register_default_handlers(self):
        """Register default notification handlers"""
        
        def log_handler(alert: Alert):
            """Log alert to structured logging system"""
            uap_logger.log_event(
                LogLevel.WARNING if alert.severity == AlertSeverity.WARNING else LogLevel.ERROR,
                f"ALERT {alert.severity.value.upper()}: {alert.title}",
                EventType.SYSTEM,
                {
                    "alert_id": alert.alert_id,
                    "rule_id": alert.rule_id,
                    "component": alert.component,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold_value,
                    "metadata": alert.metadata
                },
                "alerting"
            )
        
        def console_handler(alert: Alert):
            """Print alert to console"""
            timestamp = alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S")
            print(f"🚨 [{timestamp}] {alert.severity.value.upper()}: {alert.title}")
            print(f"   {alert.message}")
            if alert.current_value is not None:
                print(f"   Current: {alert.current_value}, Threshold: {alert.threshold_value}")
        
        self.notification_handlers["log"] = log_handler
        self.notification_handlers["console"] = console_handler
    
    def add_rule(self, rule: AlertRule):
        """Add an alerting rule"""
        self.rules[rule.rule_id] = rule
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Added alert rule: {rule.name}",
            EventType.SYSTEM,
            {"rule_id": rule.rule_id, "metric": rule.metric_name, "threshold": rule.threshold},
            "alerting"
        )
    
    def remove_rule(self, rule_id: str):
        """Remove an alerting rule"""
        if rule_id in self.rules:
            rule = self.rules.pop(rule_id)
            
            # Stop evaluation task if running
            if rule_id in self.evaluation_tasks:
                self.evaluation_tasks[rule_id].cancel()
                del self.evaluation_tasks[rule_id]
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Removed alert rule: {rule.name}",
                EventType.SYSTEM,
                {"rule_id": rule_id},
                "alerting"
            )
    
    def add_notification_handler(self, channel: str, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers[channel] = handler
    
    async def start(self):
        """Start the alerting system"""
        self.running = True
        
        # Start evaluation tasks for each rule
        for rule in self.rules.values():
            if rule.enabled:
                task = asyncio.create_task(self._evaluate_rule_loop(rule))
                self.evaluation_tasks[rule.rule_id] = task
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Alert manager started",
            EventType.SYSTEM,
            {"active_rules": len(self.rules)},
            "alerting"
        )
    
    async def stop(self):
        """Stop the alerting system"""
        self.running = False
        
        # Cancel all evaluation tasks
        for task in self.evaluation_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.evaluation_tasks:
            await asyncio.gather(*self.evaluation_tasks.values(), return_exceptions=True)
        
        self.evaluation_tasks.clear()
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Alert manager stopped",
            EventType.SYSTEM,
            {},
            "alerting"
        )
    
    async def _evaluate_rule_loop(self, rule: AlertRule):
        """Continuously evaluate a rule"""
        while self.running:
            try:
                await self._evaluate_rule(rule)
                await asyncio.sleep(rule.evaluation_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Error evaluating rule {rule.rule_id}: {str(e)}",
                    EventType.ERROR,
                    {"rule_id": rule.rule_id, "error": str(e)},
                    "alerting"
                )
                await asyncio.sleep(rule.evaluation_interval_seconds)
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alerting rule"""
        try:
            # Get metrics from the performance monitor
            cutoff_time = datetime.utcnow() - timedelta(minutes=rule.time_window_minutes)
            
            # Filter metrics by name and time window
            matching_metrics = []
            for metric in performance_monitor.metrics_history:
                if (metric.metric_name == rule.metric_name and 
                    metric.timestamp >= cutoff_time):
                    
                    # Apply tag filter if specified
                    if rule.tags_filter:
                        metric_tags = metric.tags or {}
                        if all(metric_tags.get(k) == v for k, v in rule.tags_filter.items()):
                            matching_metrics.append(metric)
                    else:
                        matching_metrics.append(metric)
            
            # Check if we have enough data points
            if len(matching_metrics) < rule.min_data_points:
                return
            
            # Calculate current value based on rule metric
            current_value = None
            if rule.metric_name.endswith("_p95"):
                # Calculate 95th percentile
                values = sorted([m.value for m in matching_metrics])
                if len(values) >= 10:
                    index = int(0.95 * len(values))
                    current_value = values[min(index, len(values) - 1)]
            elif rule.metric_name.endswith("_rate"):
                # Calculate rate (e.g., error rate)
                current_value = self._calculate_error_rate()
            else:
                # Use latest value
                if matching_metrics:
                    current_value = matching_metrics[-1].value
            
            if current_value is None:
                return
            
            # Evaluate threshold
            threshold_violated = self._evaluate_threshold(
                current_value, rule.threshold, rule.comparison
            )
            
            alert_id = f"{rule.rule_id}_{rule.metric_name}"
            
            if threshold_violated:
                # Check if alert is already active
                if alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        alert_id=alert_id,
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        title=rule.name,
                        message=rule.description,
                        component=self._extract_component(rule.metric_name),
                        triggered_at=datetime.utcnow(),
                        current_value=current_value,
                        threshold_value=rule.threshold,
                        metadata={
                            "metric_name": rule.metric_name,
                            "time_window_minutes": rule.time_window_minutes,
                            "data_points": len(matching_metrics)
                        }
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Send notifications
                    await self._send_notifications(alert, rule.notification_channels)
            else:
                # Check if we need to resolve an active alert
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()
                    
                    # Move to history and remove from active
                    self.alert_history.append(alert)
                    del self.active_alerts[alert_id]
                    
                    uap_logger.log_event(
                        LogLevel.INFO,
                        f"Alert resolved: {alert.title}",
                        EventType.SYSTEM,
                        {"alert_id": alert_id, "duration_minutes": 
                         (alert.resolved_at - alert.triggered_at).total_seconds() / 60},
                        "alerting"
                    )
        
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to evaluate rule {rule.rule_id}: {str(e)}",
                EventType.ERROR,
                {"rule_id": rule.rule_id, "error": str(e)},
                "alerting"
            )
    
    def _evaluate_threshold(self, current_value: Union[float, int], 
                          threshold: Union[float, int], comparison: str) -> bool:
        """Evaluate if threshold is violated"""
        if comparison == "gt":
            return current_value > threshold
        elif comparison == "gte":
            return current_value >= threshold
        elif comparison == "lt":
            return current_value < threshold
        elif comparison == "lte":
            return current_value <= threshold
        elif comparison == "eq":
            return current_value == threshold
        else:
            return False
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate from agent statistics"""
        total_requests = 0
        total_errors = 0
        
        for stats in performance_monitor.agent_stats.values():
            total_requests += stats.total_requests
            total_errors += stats.error_count
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100
    
    def _extract_component(self, metric_name: str) -> str:
        """Extract component name from metric name"""
        if metric_name.startswith("agent."):
            return "agent"
        elif metric_name.startswith("websocket."):
            return "websocket"
        elif metric_name.startswith("system."):
            return "system"
        else:
            return "unknown"
    
    async def _send_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications to specified channels"""
        for channel in channels:
            if channel in self.notification_handlers:
                try:
                    handler = self.notification_handlers[channel]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    uap_logger.log_event(
                        LogLevel.ERROR,
                        f"Failed to send notification via {channel}: {str(e)}",
                        EventType.ERROR,
                        {"channel": channel, "alert_id": alert.alert_id, "error": str(e)},
                        "alerting"
                    )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        recent_alerts = list(self.alert_history)[-limit:]
        return [alert.to_dict() for alert in recent_alerts]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Alert acknowledged: {alert.title}",
                EventType.SYSTEM,
                {"alert_id": alert_id, "acknowledged_by": acknowledged_by},
                "alerting"
            )
    
    def suppress_alert(self, alert_id: str):
        """Suppress an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Alert suppressed: {alert.title}",
                EventType.SYSTEM,
                {"alert_id": alert_id},
                "alerting"
            )

# Global alert manager instance
alert_manager = AlertManager()

# Convenience functions
async def start_alerting():
    """Start the alerting system"""
    await alert_manager.start()

async def stop_alerting():
    """Stop the alerting system"""
    await alert_manager.stop()

def get_active_alerts() -> List[Dict[str, Any]]:
    """Get active alerts"""
    return alert_manager.get_active_alerts()

def get_alert_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Get alert history"""
    return alert_manager.get_alert_history(limit)

def acknowledge_alert(alert_id: str, acknowledged_by: str = "system"):
    """Acknowledge an alert"""
    alert_manager.acknowledge_alert(alert_id, acknowledged_by)