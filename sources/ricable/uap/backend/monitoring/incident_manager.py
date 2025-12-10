# File: backend/monitoring/incident_manager.py
"""
Automated incident response and management system for UAP platform.
Provides incident detection, response automation, runbook execution, and post-mortem analysis.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from pathlib import Path
import subprocess
import aiofiles

from .logs.logger import uap_logger, EventType, LogLevel
from .alerting.alerts import alert_manager, Alert, AlertSeverity
from .health_checker import health_monitor, HealthStatus

class IncidentSeverity(Enum):
    """Incident severity levels"""
    P1_CRITICAL = "P1-Critical"  # System down, critical functionality impacted
    P2_HIGH = "P2-High"        # Major functionality impacted
    P3_MEDIUM = "P3-Medium"    # Minor functionality impacted
    P4_LOW = "P4-Low"          # Minimal impact

class IncidentStatus(Enum):
    """Incident status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ResponseAction(Enum):
    """Types of automated response actions"""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLEAR_CACHE = "clear_cache"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    NOTIFY_ONCALL = "notify_oncall"
    RUN_DIAGNOSTIC = "run_diagnostic"
    CUSTOM_SCRIPT = "custom_script"

@dataclass
class IncidentResponse:
    """Automated response configuration"""
    action: ResponseAction
    conditions: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    enabled: bool = True
    script_path: Optional[str] = None
    description: str = ""

@dataclass
class Incident:
    """Incident record"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    component: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    assignee: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)  # Alert IDs
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.closed_at:
            data['closed_at'] = self.closed_at.isoformat()
        return data
    
    def add_timeline_entry(self, event: str, details: Dict[str, Any] = None):
        """Add entry to incident timeline"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "details": details or {}
        }
        self.timeline.append(entry)
        self.updated_at = datetime.utcnow()

@dataclass
class RunbookStep:
    """Individual step in an incident response runbook"""
    step_id: str
    title: str
    description: str
    action_type: str  # "manual", "automated", "diagnostic"
    command: Optional[str] = None
    expected_output: Optional[str] = None
    timeout_seconds: int = 60
    critical: bool = False
    depends_on: List[str] = field(default_factory=list)

@dataclass
class Runbook:
    """Incident response runbook"""
    runbook_id: str
    name: str
    description: str
    component: str
    incident_types: List[str]
    steps: List[RunbookStep]
    owner: str
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True

class IncidentManager:
    """Main incident management system"""
    
    def __init__(self):
        self.incidents: Dict[str, Incident] = {}
        self.incident_responses: Dict[str, List[IncidentResponse]] = {}
        self.runbooks: Dict[str, Runbook] = {}
        self.active_responses: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Initialize default incident responses
        self._setup_default_responses()
        self._setup_default_runbooks()
    
    def _setup_default_responses(self):
        """Setup default automated incident responses"""
        
        # High CPU usage response
        self.add_incident_response("high_cpu_usage", [
            IncidentResponse(
                action=ResponseAction.RUN_DIAGNOSTIC,
                conditions={"metric": "cpu_usage", "threshold": 85},
                parameters={"command": "top -n 1 -b"},
                description="Capture CPU usage snapshot"
            ),
            IncidentResponse(
                action=ResponseAction.SCALE_UP,
                conditions={"metric": "cpu_usage", "threshold": 90, "duration_minutes": 10},
                parameters={"scaling_factor": 1.5},
                description="Scale up resources for high CPU usage"
            )
        ])
        
        # High memory usage response
        self.add_incident_response("high_memory_usage", [
            IncidentResponse(
                action=ResponseAction.CLEAR_CACHE,
                conditions={"metric": "memory_usage", "threshold": 85},
                parameters={"cache_type": "redis"},
                description="Clear Redis cache to free memory"
            ),
            IncidentResponse(
                action=ResponseAction.RESTART_SERVICE,
                conditions={"metric": "memory_usage", "threshold": 95, "critical": True},
                parameters={"service": "uap-backend", "graceful": True},
                description="Restart backend service due to critical memory usage"
            )
        ])
        
        # Service down response
        self.add_incident_response("service_down", [
            IncidentResponse(
                action=ResponseAction.RESTART_SERVICE,
                conditions={"service_status": "down"},
                parameters={"service": "uap-backend", "wait_time": 30},
                description="Restart down service"
            ),
            IncidentResponse(
                action=ResponseAction.NOTIFY_ONCALL,
                conditions={"service_status": "down", "duration_minutes": 5},
                parameters={"urgency": "high", "channel": "slack"},
                description="Notify on-call engineer of persistent service outage"
            )
        ])
        
        # High error rate response
        self.add_incident_response("high_error_rate", [
            IncidentResponse(
                action=ResponseAction.RUN_DIAGNOSTIC,
                conditions={"error_rate": 10},
                parameters={"command": "tail -n 100 /var/log/uap/errors.log"},
                description="Capture recent error logs"
            ),
            IncidentResponse(
                action=ResponseAction.ROLLBACK_DEPLOYMENT,
                conditions={"error_rate": 25, "deployment_age_minutes": 60},
                parameters={"rollback_strategy": "previous_version"},
                description="Rollback recent deployment due to high error rate"
            )
        ])
        
        # Database connectivity issues
        self.add_incident_response("database_connectivity", [
            IncidentResponse(
                action=ResponseAction.RUN_DIAGNOSTIC,
                conditions={"component": "database", "status": "unhealthy"},
                parameters={"command": "pg_isready -h localhost -p 5432"},
                description="Check PostgreSQL connectivity"
            ),
            IncidentResponse(
                action=ResponseAction.RESTART_SERVICE,
                conditions={"component": "database", "status": "unhealthy", "duration_minutes": 5},
                parameters={"service": "postgresql", "graceful": True},
                description="Restart PostgreSQL service"
            )
        ])
    
    def _setup_default_runbooks(self):
        """Setup default incident response runbooks"""
        
        # High CPU Usage Runbook
        high_cpu_runbook = Runbook(
            runbook_id="high_cpu_usage",
            name="High CPU Usage Response",
            description="Steps to diagnose and resolve high CPU usage incidents",
            component="system",
            incident_types=["high_cpu_usage", "performance_degradation"],
            owner="platform-team",
            steps=[
                RunbookStep(
                    step_id="cpu_01",
                    title="Check Current CPU Usage",
                    description="Verify current CPU usage and identify top processes",
                    action_type="diagnostic",
                    command="top -n 1 -b | head -20",
                    timeout_seconds=30
                ),
                RunbookStep(
                    step_id="cpu_02",
                    title="Identify Resource-Heavy Processes",
                    description="List processes consuming most CPU",
                    action_type="diagnostic",
                    command="ps aux --sort=-%cpu | head -10",
                    timeout_seconds=30,
                    depends_on=["cpu_01"]
                ),
                RunbookStep(
                    step_id="cpu_03",
                    title="Check System Load",
                    description="Check system load averages",
                    action_type="diagnostic",
                    command="uptime",
                    timeout_seconds=10,
                    depends_on=["cpu_01"]
                ),
                RunbookStep(
                    step_id="cpu_04",
                    title="Scale Resources (if applicable)",
                    description="Scale up compute resources if usage is consistently high",
                    action_type="automated",
                    command="scale_up_resources",
                    timeout_seconds=300,
                    depends_on=["cpu_02", "cpu_03"]
                )
            ]
        )
        self.add_runbook(high_cpu_runbook)
        
        # Service Down Runbook
        service_down_runbook = Runbook(
            runbook_id="service_down",
            name="Service Down Response",
            description="Steps to diagnose and restore downed services",
            component="service",
            incident_types=["service_down", "availability_issue"],
            owner="platform-team",
            steps=[
                RunbookStep(
                    step_id="svc_01",
                    title="Check Service Status",
                    description="Verify service status and recent logs",
                    action_type="diagnostic",
                    command="systemctl status uap-backend",
                    timeout_seconds=30,
                    critical=True
                ),
                RunbookStep(
                    step_id="svc_02",
                    title="Check Recent Logs",
                    description="Review recent service logs for errors",
                    action_type="diagnostic",
                    command="journalctl -u uap-backend --since='10 minutes ago'",
                    timeout_seconds=60,
                    depends_on=["svc_01"]
                ),
                RunbookStep(
                    step_id="svc_03",
                    title="Restart Service",
                    description="Attempt to restart the service",
                    action_type="automated",
                    command="systemctl restart uap-backend",
                    timeout_seconds=120,
                    critical=True,
                    depends_on=["svc_02"]
                ),
                RunbookStep(
                    step_id="svc_04",
                    title="Verify Service Recovery",
                    description="Confirm service is running and healthy",
                    action_type="diagnostic",
                    command="curl -f http://localhost:8000/health",
                    timeout_seconds=30,
                    expected_output="status: ok",
                    depends_on=["svc_03"]
                )
            ]
        )
        self.add_runbook(service_down_runbook)
        
        # Database Issues Runbook
        database_runbook = Runbook(
            runbook_id="database_issues",
            name="Database Connectivity Issues",
            description="Steps to diagnose and resolve database connectivity issues",
            component="database",
            incident_types=["database_connectivity", "data_access_issues"],
            owner="data-team",
            steps=[
                RunbookStep(
                    step_id="db_01",
                    title="Check Database Connectivity",
                    description="Test connection to PostgreSQL database",
                    action_type="diagnostic",
                    command="pg_isready -h localhost -p 5432 -U uap",
                    timeout_seconds=30,
                    critical=True
                ),
                RunbookStep(
                    step_id="db_02",
                    title="Check Database Processes",
                    description="Verify PostgreSQL processes are running",
                    action_type="diagnostic",
                    command="ps aux | grep postgres",
                    timeout_seconds=30,
                    depends_on=["db_01"]
                ),
                RunbookStep(
                    step_id="db_03",
                    title="Check Database Logs",
                    description="Review PostgreSQL logs for errors",
                    action_type="diagnostic",
                    command="tail -n 50 /var/log/postgresql/postgresql.log",
                    timeout_seconds=60,
                    depends_on=["db_01"]
                ),
                RunbookStep(
                    step_id="db_04",
                    title="Restart Database (if needed)",
                    description="Restart PostgreSQL service if connection fails",
                    action_type="automated",
                    command="systemctl restart postgresql",
                    timeout_seconds=180,
                    critical=True,
                    depends_on=["db_02", "db_03"]
                )
            ]
        )
        self.add_runbook(database_runbook)
    
    def add_incident_response(self, incident_type: str, responses: List[IncidentResponse]):
        """Add automated incident responses for a specific incident type"""
        if incident_type not in self.incident_responses:
            self.incident_responses[incident_type] = []
        self.incident_responses[incident_type].extend(responses)
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Added {len(responses)} incident responses for {incident_type}",
            EventType.SYSTEM,
            {"incident_type": incident_type, "response_count": len(responses)},
            "incident_manager"
        )
    
    def add_runbook(self, runbook: Runbook):
        """Add an incident response runbook"""
        self.runbooks[runbook.runbook_id] = runbook
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Added runbook: {runbook.name}",
            EventType.SYSTEM,
            {"runbook_id": runbook.runbook_id, "component": runbook.component},
            "incident_manager"
        )
    
    async def start_incident_monitoring(self):
        """Start incident monitoring and response system"""
        self.running = True
        
        # Start monitoring for alerts that should trigger incidents
        alert_monitor_task = asyncio.create_task(self._monitor_alerts())
        self.active_responses["alert_monitor"] = alert_monitor_task
        
        # Start monitoring for health check failures
        health_monitor_task = asyncio.create_task(self._monitor_health_checks())
        self.active_responses["health_monitor"] = health_monitor_task
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Incident monitoring started",
            EventType.SYSTEM,
            {"runbooks": len(self.runbooks), "response_types": len(self.incident_responses)},
            "incident_manager"
        )
    
    async def stop_incident_monitoring(self):
        """Stop incident monitoring"""
        self.running = False
        
        # Cancel all monitoring tasks
        for task in self.active_responses.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.active_responses:
            await asyncio.gather(*self.active_responses.values(), return_exceptions=True)
        
        self.active_responses.clear()
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Incident monitoring stopped",
            EventType.SYSTEM,
            {},
            "incident_manager"
        )
    
    async def _monitor_alerts(self):
        """Monitor alerts for incident creation"""
        while self.running:
            try:
                # Check for new critical alerts
                active_alerts = alert_manager.get_active_alerts()
                
                for alert_data in active_alerts:
                    alert_id = alert_data["alert_id"]
                    severity = alert_data["severity"]
                    
                    # Check if this alert should trigger an incident
                    if severity == "critical" and not self._incident_exists_for_alert(alert_id):
                        await self._create_incident_from_alert(alert_data)
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Error monitoring alerts: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e)},
                    "incident_manager"
                )
                await asyncio.sleep(30)
    
    async def _monitor_health_checks(self):
        """Monitor health checks for incident creation"""
        while self.running:
            try:
                # Check overall system health
                health_summary = health_monitor.get_health_summary()
                overall_health = health_summary["overall_health"]
                
                # Create incident for critical health issues
                if overall_health["critical_issues"] > 0:
                    await self._create_incident_from_health_issue(health_summary)
                
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Error monitoring health checks: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e)},
                    "incident_manager"
                )
                await asyncio.sleep(60)
    
    def _incident_exists_for_alert(self, alert_id: str) -> bool:
        """Check if an incident already exists for this alert"""
        for incident in self.incidents.values():
            if alert_id in incident.alerts and incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                return True
        return False
    
    async def _create_incident_from_alert(self, alert_data: Dict[str, Any]):
        """Create an incident from a critical alert"""
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        
        # Determine severity based on alert severity
        severity_mapping = {
            "critical": IncidentSeverity.P1_CRITICAL,
            "warning": IncidentSeverity.P2_HIGH,
            "info": IncidentSeverity.P4_LOW
        }
        
        severity = severity_mapping.get(alert_data["severity"], IncidentSeverity.P3_MEDIUM)
        
        incident = Incident(
            incident_id=incident_id,
            title=f"Alert: {alert_data['title']}",
            description=alert_data["message"],
            severity=severity,
            status=IncidentStatus.DETECTED,
            component=alert_data["component"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            alerts=[alert_data["alert_id"]],
            tags=["auto-created", "alert-triggered"],
            metadata=alert_data.get("metadata", {})
        )
        
        incident.add_timeline_entry("Incident created from critical alert", {
            "alert_id": alert_data["alert_id"],
            "alert_severity": alert_data["severity"]
        })
        
        self.incidents[incident_id] = incident
        
        # Trigger automated response
        await self._trigger_automated_response(incident)
        
        uap_logger.log_event(
            LogLevel.WARNING,
            f"Incident created from alert: {incident.title}",
            EventType.SYSTEM,
            {
                "incident_id": incident_id,
                "severity": severity.value,
                "component": incident.component,
                "alert_id": alert_data["alert_id"]
            },
            "incident_manager"
        )
    
    async def _create_incident_from_health_issue(self, health_summary: Dict[str, Any]):
        """Create an incident from critical health issues"""
        # Check if we already have an active health-related incident
        for incident in self.incidents.values():
            if ("health-check" in incident.tags and 
                incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]):
                return  # Already have an active health incident
        
        overall_health = health_summary["overall_health"]
        
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        
        incident = Incident(
            incident_id=incident_id,
            title=f"System Health Issue: {overall_health['message']}",
            description=f"System health degraded with {overall_health['critical_issues']} critical issues",
            severity=IncidentSeverity.P1_CRITICAL,
            status=IncidentStatus.DETECTED,
            component="system",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=["auto-created", "health-check"],
            metadata={"health_summary": overall_health}
        )
        
        incident.add_timeline_entry("Incident created from health check failure", overall_health)
        
        self.incidents[incident_id] = incident
        
        # Trigger automated response
        await self._trigger_automated_response(incident)
        
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Health incident created: {incident.title}",
            EventType.SYSTEM,
            {
                "incident_id": incident_id,
                "critical_issues": overall_health["critical_issues"],
                "unhealthy_checks": overall_health["unhealthy_checks"]
            },
            "incident_manager"
        )
    
    async def _trigger_automated_response(self, incident: Incident):
        """Trigger automated response for an incident"""
        incident.status = IncidentStatus.INVESTIGATING
        incident.add_timeline_entry("Started automated response", {"status": "investigating"})
        
        # Find applicable incident responses
        incident_type = self._determine_incident_type(incident)
        responses = self.incident_responses.get(incident_type, [])
        
        if responses:
            response_task = asyncio.create_task(
                self._execute_incident_responses(incident, responses)
            )
            self.active_responses[incident.incident_id] = response_task
        
        # Find and execute applicable runbook
        runbook = self._find_applicable_runbook(incident)
        if runbook:
            runbook_task = asyncio.create_task(
                self._execute_runbook(incident, runbook)
            )
            self.active_responses[f"{incident.incident_id}_runbook"] = runbook_task
    
    def _determine_incident_type(self, incident: Incident) -> str:
        """Determine incident type based on incident characteristics"""
        # Simple heuristic based on title and component
        title_lower = incident.title.lower()
        
        if "cpu" in title_lower:
            return "high_cpu_usage"
        elif "memory" in title_lower:
            return "high_memory_usage"
        elif "service" in title_lower and "down" in title_lower:
            return "service_down"
        elif "error" in title_lower and "rate" in title_lower:
            return "high_error_rate"
        elif "database" in title_lower or incident.component == "database":
            return "database_connectivity"
        else:
            return "generic"
    
    def _find_applicable_runbook(self, incident: Incident) -> Optional[Runbook]:
        """Find applicable runbook for an incident"""
        incident_type = self._determine_incident_type(incident)
        
        for runbook in self.runbooks.values():
            if (runbook.enabled and 
                (incident_type in runbook.incident_types or 
                 incident.component == runbook.component)):
                return runbook
        
        return None
    
    async def _execute_incident_responses(self, incident: Incident, responses: List[IncidentResponse]):
        """Execute automated incident responses"""
        for response in responses:
            if not response.enabled:
                continue
            
            try:
                # Check if conditions are met
                if self._check_response_conditions(incident, response):
                    await self._execute_response_action(incident, response)
                    
                    incident.response_actions.append({
                        "action": response.action.value,
                        "description": response.description,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "executed"
                    })
                    
                    incident.add_timeline_entry(
                        f"Executed automated response: {response.action.value}",
                        {"description": response.description}
                    )
            
            except Exception as e:
                incident.response_actions.append({
                    "action": response.action.value,
                    "description": response.description,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": str(e)
                })
                
                incident.add_timeline_entry(
                    f"Failed to execute response: {response.action.value}",
                    {"error": str(e)}
                )
                
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to execute incident response: {str(e)}",
                    EventType.ERROR,
                    {
                        "incident_id": incident.incident_id,
                        "action": response.action.value,
                        "error": str(e)
                    },
                    "incident_manager"
                )
    
    def _check_response_conditions(self, incident: Incident, response: IncidentResponse) -> bool:
        """Check if response conditions are met"""
        # This would implement condition checking logic
        # For now, we'll return True for enabled responses
        return True
    
    async def _execute_response_action(self, incident: Incident, response: IncidentResponse):
        """Execute a specific response action"""
        action = response.action
        params = response.parameters
        
        if action == ResponseAction.RESTART_SERVICE:
            service = params.get("service", "uap-backend")
            await self._restart_service(service, params.get("graceful", True))
        
        elif action == ResponseAction.SCALE_UP:
            factor = params.get("scaling_factor", 1.5)
            await self._scale_resources(factor)
        
        elif action == ResponseAction.CLEAR_CACHE:
            cache_type = params.get("cache_type", "redis")
            await self._clear_cache(cache_type)
        
        elif action == ResponseAction.RUN_DIAGNOSTIC:
            command = params.get("command")
            if command:
                await self._run_diagnostic_command(command)
        
        elif action == ResponseAction.NOTIFY_ONCALL:
            await self._notify_oncall(incident, params)
        
        elif action == ResponseAction.CUSTOM_SCRIPT:
            if response.script_path:
                await self._execute_custom_script(response.script_path, params)
        
        # Add more action implementations as needed
    
    async def _execute_runbook(self, incident: Incident, runbook: Runbook):
        """Execute incident response runbook"""
        incident.add_timeline_entry(
            f"Started runbook execution: {runbook.name}",
            {"runbook_id": runbook.runbook_id, "steps": len(runbook.steps)}
        )
        
        executed_steps = set()
        
        for step in runbook.steps:
            # Check dependencies
            if step.depends_on and not all(dep in executed_steps for dep in step.depends_on):
                continue
            
            try:
                await self._execute_runbook_step(incident, step)
                executed_steps.add(step.step_id)
                
                incident.add_timeline_entry(
                    f"Completed runbook step: {step.title}",
                    {"step_id": step.step_id, "action_type": step.action_type}
                )
            
            except Exception as e:
                incident.add_timeline_entry(
                    f"Failed runbook step: {step.title}",
                    {"step_id": step.step_id, "error": str(e)}
                )
                
                if step.critical:
                    uap_logger.log_event(
                        LogLevel.ERROR,
                        f"Critical runbook step failed: {step.title}",
                        EventType.ERROR,
                        {
                            "incident_id": incident.incident_id,
                            "step_id": step.step_id,
                            "error": str(e)
                        },
                        "incident_manager"
                    )
                    break  # Stop execution on critical step failure
        
        incident.add_timeline_entry("Completed runbook execution", {
            "runbook_id": runbook.runbook_id,
            "executed_steps": len(executed_steps),
            "total_steps": len(runbook.steps)
        })
    
    async def _execute_runbook_step(self, incident: Incident, step: RunbookStep):
        """Execute a single runbook step"""
        if step.action_type == "diagnostic" and step.command:
            result = await self._run_diagnostic_command(step.command)
            
            # Check expected output if specified
            if step.expected_output and step.expected_output not in result:
                raise Exception(f"Expected output '{step.expected_output}' not found")
        
        elif step.action_type == "automated" and step.command:
            await self._execute_automated_command(step.command)
        
        # Manual steps would be logged but not executed automatically
        elif step.action_type == "manual":
            incident.add_timeline_entry(
                f"Manual step pending: {step.title}",
                {"description": step.description, "step_id": step.step_id}
            )
    
    # Helper methods for response actions
    async def _restart_service(self, service: str, graceful: bool = True):
        """Restart a system service"""
        try:
            if graceful:
                cmd = f"systemctl reload-or-restart {service}"
            else:
                cmd = f"systemctl restart {service}"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Service restart failed: {stderr.decode()}")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Service {service} restarted successfully",
                EventType.SYSTEM,
                {"service": service, "graceful": graceful},
                "incident_manager"
            )
        
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to restart service {service}: {str(e)}",
                EventType.ERROR,
                {"service": service, "error": str(e)},
                "incident_manager"
            )
            raise
    
    async def _scale_resources(self, scaling_factor: float):
        """Scale system resources"""
        # This would integrate with your orchestration system (Kubernetes, etc.)
        uap_logger.log_event(
            LogLevel.INFO,
            f"Scaling resources by factor {scaling_factor}",
            EventType.SYSTEM,
            {"scaling_factor": scaling_factor},
            "incident_manager"
        )
    
    async def _clear_cache(self, cache_type: str):
        """Clear system cache"""
        try:
            if cache_type == "redis":
                # This would connect to Redis and clear cache
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Cleared {cache_type} cache",
                    EventType.SYSTEM,
                    {"cache_type": cache_type},
                    "incident_manager"
                )
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to clear {cache_type} cache: {str(e)}",
                EventType.ERROR,
                {"cache_type": cache_type, "error": str(e)},
                "incident_manager"
            )
            raise
    
    async def _run_diagnostic_command(self, command: str) -> str:
        """Run a diagnostic command and return output"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            output = stdout.decode() if stdout else stderr.decode()
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Executed diagnostic command: {command}",
                EventType.SYSTEM,
                {"command": command, "output_lines": len(output.split('\n'))},
                "incident_manager"
            )
            
            return output
        
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to execute diagnostic command: {str(e)}",
                EventType.ERROR,
                {"command": command, "error": str(e)},
                "incident_manager"
            )
            raise
    
    async def _execute_automated_command(self, command: str):
        """Execute an automated command"""
        # This would map command names to actual implementations
        if command == "scale_up_resources":
            await self._scale_resources(1.5)
        else:
            await self._run_diagnostic_command(command)
    
    async def _notify_oncall(self, incident: Incident, params: Dict[str, Any]):
        """Notify on-call engineer"""
        urgency = params.get("urgency", "medium")
        channel = params.get("channel", "email")
        
        # This would integrate with your notification system (PagerDuty, Slack, etc.)
        uap_logger.log_event(
            LogLevel.WARNING,
            f"On-call notification sent for incident {incident.incident_id}",
            EventType.SYSTEM,
            {
                "incident_id": incident.incident_id,
                "urgency": urgency,
                "channel": channel,
                "severity": incident.severity.value
            },
            "incident_manager"
        )
    
    async def _execute_custom_script(self, script_path: str, params: Dict[str, Any]):
        """Execute a custom response script"""
        try:
            if not Path(script_path).exists():
                raise Exception(f"Script not found: {script_path}")
            
            # Prepare environment variables from params
            env = {**params, "PATH": "/usr/local/bin:/usr/bin:/bin"}
            
            process = await asyncio.create_subprocess_exec(
                script_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Script execution failed: {stderr.decode()}")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Custom script executed successfully: {script_path}",
                EventType.SYSTEM,
                {"script_path": script_path, "return_code": process.returncode},
                "incident_manager"
            )
        
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to execute custom script: {str(e)}",
                EventType.ERROR,
                {"script_path": script_path, "error": str(e)},
                "incident_manager"
            )
            raise
    
    def resolve_incident(self, incident_id: str, resolution: str, resolved_by: str = "system"):
        """Resolve an incident"""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = datetime.utcnow()
            incident.resolution = resolution
            incident.updated_at = datetime.utcnow()
            
            incident.add_timeline_entry(
                "Incident resolved",
                {"resolved_by": resolved_by, "resolution": resolution}
            )
            
            # Cancel any active response tasks
            if incident_id in self.active_responses:
                self.active_responses[incident_id].cancel()
                del self.active_responses[incident_id]
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Incident resolved: {incident.title}",
                EventType.SYSTEM,
                {
                    "incident_id": incident_id,
                    "resolution": resolution,
                    "duration_minutes": (incident.resolved_at - incident.created_at).total_seconds() / 60
                },
                "incident_manager"
            )
    
    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents"""
        active_incidents = []
        for incident in self.incidents.values():
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                active_incidents.append(incident.to_dict())
        return active_incidents
    
    def get_incident_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get incident history"""
        incidents = sorted(
            self.incidents.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return [incident.to_dict() for incident in incidents[:limit]]
    
    async def generate_post_mortem(self, incident_id: str) -> Dict[str, Any]:
        """Generate post-mortem analysis for an incident"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        
        # Calculate incident metrics
        duration = None
        if incident.resolved_at:
            duration = (incident.resolved_at - incident.created_at).total_seconds()
        
        # Analyze timeline for insights
        timeline_analysis = self._analyze_incident_timeline(incident)
        
        post_mortem = {
            "incident_id": incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "duration_seconds": duration,
            "created_at": incident.created_at.isoformat(),
            "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
            "root_cause": incident.root_cause,
            "resolution": incident.resolution,
            "timeline_analysis": timeline_analysis,
            "response_actions": incident.response_actions,
            "lessons_learned": incident.lessons_learned,
            "recommendations": self._generate_recommendations(incident),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Save post-mortem to file
        await self._save_post_mortem(post_mortem)
        
        return post_mortem
    
    def _analyze_incident_timeline(self, incident: Incident) -> Dict[str, Any]:
        """Analyze incident timeline for insights"""
        timeline = incident.timeline
        
        detection_time = incident.created_at
        first_response_time = None
        mitigation_time = None
        
        for entry in timeline:
            event = entry["event"]
            timestamp = datetime.fromisoformat(entry["timestamp"])
            
            if "automated response" in event.lower() and not first_response_time:
                first_response_time = timestamp
            
            if "resolved" in event.lower() and not mitigation_time:
                mitigation_time = timestamp
        
        analysis = {
            "total_timeline_events": len(timeline),
            "detection_time": detection_time.isoformat(),
            "automated_responses": len(incident.response_actions)
        }
        
        if first_response_time:
            response_delay = (first_response_time - detection_time).total_seconds()
            analysis["time_to_first_response_seconds"] = response_delay
        
        if mitigation_time:
            mitigation_delay = (mitigation_time - detection_time).total_seconds()
            analysis["time_to_mitigation_seconds"] = mitigation_delay
        
        return analysis
    
    def _generate_recommendations(self, incident: Incident) -> List[str]:
        """Generate recommendations based on incident analysis"""
        recommendations = []
        
        # Analysis based on incident characteristics
        if incident.severity == IncidentSeverity.P1_CRITICAL:
            recommendations.append("Review alert thresholds to detect issues earlier")
            recommendations.append("Consider implementing additional monitoring for this component")
        
        if len(incident.response_actions) == 0:
            recommendations.append("Implement automated response actions for this incident type")
        
        if incident.component == "database":
            recommendations.append("Review database performance and capacity planning")
            recommendations.append("Consider implementing database connection pooling")
        
        if "cpu" in incident.title.lower():
            recommendations.append("Review resource allocation and scaling policies")
            recommendations.append("Consider implementing auto-scaling for CPU-intensive workloads")
        
        return recommendations
    
    async def _save_post_mortem(self, post_mortem: Dict[str, Any]):
        """Save post-mortem to file"""
        try:
            post_mortem_dir = Path("/var/log/uap/post_mortems")
            post_mortem_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"post_mortem_{post_mortem['incident_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = post_mortem_dir / filename
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(post_mortem, indent=2))
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Post-mortem saved: {filepath}",
                EventType.SYSTEM,
                {"incident_id": post_mortem["incident_id"], "filepath": str(filepath)},
                "incident_manager"
            )
        
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to save post-mortem: {str(e)}",
                EventType.ERROR,
                {"incident_id": post_mortem["incident_id"], "error": str(e)},
                "incident_manager"
            )

# Global incident manager instance
incident_manager = IncidentManager()

# Convenience functions
async def start_incident_management():
    """Start incident management system"""
    await incident_manager.start_incident_monitoring()

async def stop_incident_management():
    """Stop incident management system"""
    await incident_manager.stop_incident_monitoring()

def get_active_incidents() -> List[Dict[str, Any]]:
    """Get active incidents"""
    return incident_manager.get_active_incidents()

def get_incident_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Get incident history"""
    return incident_manager.get_incident_history(limit)

def resolve_incident(incident_id: str, resolution: str, resolved_by: str = "system"):
    """Resolve an incident"""
    incident_manager.resolve_incident(incident_id, resolution, resolved_by)

async def generate_post_mortem(incident_id: str) -> Dict[str, Any]:
    """Generate post-mortem for an incident"""
    return await incident_manager.generate_post_mortem(incident_id)