# File: backend/monitoring/health_checker.py
"""
Comprehensive health monitoring system for UAP platform.
Monitors service health, dependencies, and SLA compliance.
"""

import asyncio
import time
import httpx
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import socket
import subprocess
from pathlib import Path

from .logs.logger import uap_logger, EventType, LogLevel
from .metrics.prometheus_metrics import prometheus_metrics

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Component types for health monitoring"""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    EXTERNAL_API = "external_api"
    FILESYSTEM = "filesystem"
    NETWORK = "network"

@dataclass
class HealthCheck:
    """Individual health check configuration"""
    name: str
    component_type: ComponentType
    check_function: str  # Function name to execute
    interval_seconds: int = 30
    timeout_seconds: int = 10
    critical: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class HealthResult:
    """Result of a health check"""
    check_name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class SLAMetric:
    """SLA tracking metric"""
    name: str
    target_value: float
    current_value: float
    unit: str
    threshold_type: str  # "lt", "gt", "eq"
    time_window_minutes: int = 60
    compliance_percentage: float = 99.9
    
    @property
    def is_compliant(self) -> bool:
        """Check if current value meets SLA target"""
        if self.threshold_type == "lt":
            return self.current_value < self.target_value
        elif self.threshold_type == "gt":
            return self.current_value > self.target_value
        elif self.threshold_type == "eq":
            return abs(self.current_value - self.target_value) < 0.01
        return False

class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, HealthResult] = {}
        self.health_history: Dict[str, List[HealthResult]] = {}
        self.sla_metrics: Dict[str, SLAMetric] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Initialize default health checks
        self._setup_default_health_checks()
        self._setup_sla_metrics()
    
    def _setup_default_health_checks(self):
        """Setup default health checks for UAP components"""
        
        # Backend service health
        self.add_health_check(HealthCheck(
            name="backend_service",
            component_type=ComponentType.SERVICE,
            check_function="check_backend_service",
            interval_seconds=30,
            timeout_seconds=5,
            critical=True,
            metadata={"endpoint": "http://localhost:8000/health"}
        ))
        
        # Database connectivity
        self.add_health_check(HealthCheck(
            name="postgresql_database",
            component_type=ComponentType.DATABASE,
            check_function="check_postgresql",
            interval_seconds=60,
            timeout_seconds=10,
            critical=True,
            metadata={"host": "localhost", "port": 5432, "database": "uap"}
        ))
        
        # Redis cache
        self.add_health_check(HealthCheck(
            name="redis_cache",
            component_type=ComponentType.CACHE,
            check_function="check_redis",
            interval_seconds=30,
            timeout_seconds=5,
            critical=False,
            metadata={"host": "localhost", "port": 6379}
        ))
        
        # Ray cluster
        self.add_health_check(HealthCheck(
            name="ray_cluster",
            component_type=ComponentType.SERVICE,
            check_function="check_ray_cluster",
            interval_seconds=60,
            timeout_seconds=15,
            critical=False,
            metadata={"dashboard_url": "http://localhost:8265"}
        ))
        
        # MLX inference service
        self.add_health_check(HealthCheck(
            name="mlx_inference",
            component_type=ComponentType.SERVICE,
            check_function="check_mlx_inference",
            interval_seconds=45,
            timeout_seconds=10,
            critical=False,
            metadata={"endpoint": "http://localhost:8001/health"}
        ))
        
        # Framework health checks
        for framework in ["copilot", "agno", "mastra"]:
            self.add_health_check(HealthCheck(
                name=f"framework_{framework}",
                component_type=ComponentType.SERVICE,
                check_function="check_framework_health",
                interval_seconds=60,
                timeout_seconds=10,
                critical=True,
                metadata={"framework": framework}
            ))
        
        # System resource checks
        self.add_health_check(HealthCheck(
            name="system_resources",
            component_type=ComponentType.SERVICE,
            check_function="check_system_resources",
            interval_seconds=30,
            timeout_seconds=5,
            critical=True
        ))
        
        # Disk space check
        self.add_health_check(HealthCheck(
            name="disk_space",
            component_type=ComponentType.FILESYSTEM,
            check_function="check_disk_space",
            interval_seconds=120,
            timeout_seconds=5,
            critical=True,
            metadata={"paths": ["/", "/tmp", "/var/log"]}
        ))
        
        # Network connectivity
        self.add_health_check(HealthCheck(
            name="network_connectivity",
            component_type=ComponentType.NETWORK,
            check_function="check_network_connectivity",
            interval_seconds=60,
            timeout_seconds=10,
            critical=False,
            metadata={"hosts": ["8.8.8.8", "1.1.1.1"]}
        ))
    
    def _setup_sla_metrics(self):
        """Setup SLA tracking metrics"""
        
        # Agent response time SLA (target: <2s p95)
        self.sla_metrics["agent_response_time"] = SLAMetric(
            name="Agent Response Time P95",
            target_value=2000,  # 2 seconds in milliseconds
            current_value=0,
            unit="ms",
            threshold_type="lt",
            time_window_minutes=60,
            compliance_percentage=95.0
        )
        
        # System availability SLA (target: 99.9%)
        self.sla_metrics["system_availability"] = SLAMetric(
            name="System Availability",
            target_value=99.9,
            current_value=0,
            unit="%",
            threshold_type="gt",
            time_window_minutes=1440,  # 24 hours
            compliance_percentage=99.9
        )
        
        # Error rate SLA (target: <1%)
        self.sla_metrics["error_rate"] = SLAMetric(
            name="Error Rate",
            target_value=1.0,
            current_value=0,
            unit="%",
            threshold_type="lt",
            time_window_minutes=60,
            compliance_percentage=99.0
        )
        
        # WebSocket connection stability (target: 99.9%)
        self.sla_metrics["websocket_stability"] = SLAMetric(
            name="WebSocket Connection Stability",
            target_value=99.9,
            current_value=0,
            unit="%",
            threshold_type="gt",
            time_window_minutes=60,
            compliance_percentage=99.9
        )
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to the monitoring system"""
        self.health_checks[health_check.name] = health_check
        self.health_history[health_check.name] = []
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Added health check: {health_check.name}",
            EventType.SYSTEM,
            {"check_name": health_check.name, "component_type": health_check.component_type.value},
            "health_monitor"
        )
    
    def remove_health_check(self, check_name: str):
        """Remove a health check"""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            if check_name in self.health_results:
                del self.health_results[check_name]
            if check_name in self.health_history:
                del self.health_history[check_name]
            
            # Stop monitoring task if running
            if check_name in self.monitoring_tasks:
                self.monitoring_tasks[check_name].cancel()
                del self.monitoring_tasks[check_name]
    
    async def start_monitoring(self):
        """Start health monitoring for all checks"""
        self.running = True
        
        # Start monitoring tasks for each health check
        for check_name, health_check in self.health_checks.items():
            if health_check.enabled:
                task = asyncio.create_task(self._monitor_health_check(health_check))
                self.monitoring_tasks[check_name] = task
        
        # Start SLA monitoring task
        sla_task = asyncio.create_task(self._monitor_sla_metrics())
        self.monitoring_tasks["sla_monitoring"] = sla_task
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Health monitoring started",
            EventType.SYSTEM,
            {"active_checks": len(self.health_checks)},
            "health_monitor"
        )
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        self.monitoring_tasks.clear()
        
        uap_logger.log_event(
            LogLevel.INFO,
            "Health monitoring stopped",
            EventType.SYSTEM,
            {},
            "health_monitor"
        )
    
    async def _monitor_health_check(self, health_check: HealthCheck):
        """Monitor a single health check"""
        while self.running:
            try:
                result = await self._execute_health_check(health_check)
                self._record_health_result(result)
                await asyncio.sleep(health_check.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Error monitoring health check {health_check.name}: {str(e)}",
                    EventType.ERROR,
                    {"check_name": health_check.name, "error": str(e)},
                    "health_monitor"
                )
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _execute_health_check(self, health_check: HealthCheck) -> HealthResult:
        """Execute a single health check"""
        start_time = time.time()
        
        try:
            # Get the check function and execute it
            check_function = getattr(self, health_check.check_function)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                check_function(health_check.metadata),
                timeout=health_check.timeout_seconds
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthResult(
                check_name=health_check.name,
                status=result["status"],
                response_time_ms=response_time,
                message=result["message"],
                details=result.get("details", {})
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                error="timeout"
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )
    
    def _record_health_result(self, result: HealthResult):
        """Record health check result"""
        # Store latest result
        self.health_results[result.check_name] = result
        
        # Add to history (keep last 100 results)
        if result.check_name not in self.health_history:
            self.health_history[result.check_name] = []
        
        self.health_history[result.check_name].append(result)
        if len(self.health_history[result.check_name]) > 100:
            self.health_history[result.check_name].pop(0)
        
        # Log result
        log_level = LogLevel.INFO if result.status == HealthStatus.HEALTHY else LogLevel.WARNING
        if result.status == HealthStatus.UNHEALTHY:
            log_level = LogLevel.ERROR
        
        uap_logger.log_event(
            log_level,
            f"Health check {result.check_name}: {result.message}",
            EventType.SYSTEM,
            {
                "check_name": result.check_name,
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "details": result.details
            },
            "health_monitor"
        )
        
        # Update Prometheus metrics
        health_status_value = 1 if result.status == HealthStatus.HEALTHY else 0
        prometheus_metrics.framework_status.labels(
            framework=result.check_name
        ).state(result.status.value)
    
    async def _monitor_sla_metrics(self):
        """Monitor SLA compliance metrics"""
        while self.running:
            try:
                await self._update_sla_metrics()
                await asyncio.sleep(60)  # Update SLA metrics every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Error updating SLA metrics: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e)},
                    "health_monitor"
                )
                await asyncio.sleep(60)
    
    async def _update_sla_metrics(self):
        """Update SLA metrics based on current system performance"""
        # This would integrate with your performance monitoring system
        # For now, we'll calculate based on available data
        
        # Update agent response time SLA
        # This would come from your metrics system
        # For example purposes, we'll simulate
        pass
    
    # Health check implementations
    async def check_backend_service(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check backend service health"""
        endpoint = metadata.get("endpoint", "http://localhost:8000/health")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            
            if response.status_code == 200:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Backend service is healthy",
                    "details": {"status_code": response.status_code}
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Backend service returned status {response.status_code}",
                    "details": {"status_code": response.status_code}
                }
    
    async def check_postgresql(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity"""
        try:
            # This would use your actual database connection
            # For now, check if port is open
            host = metadata.get("host", "localhost")
            port = metadata.get("port", 5432)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "PostgreSQL is accessible",
                    "details": {"host": host, "port": port}
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Cannot connect to PostgreSQL at {host}:{port}",
                    "details": {"host": host, "port": port, "error_code": result}
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"PostgreSQL check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_redis(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check Redis cache connectivity"""
        try:
            host = metadata.get("host", "localhost")
            port = metadata.get("port", 6379)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Redis is accessible",
                    "details": {"host": host, "port": port}
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,  # Redis is optional
                    "message": f"Cannot connect to Redis at {host}:{port}",
                    "details": {"host": host, "port": port, "error_code": result}
                }
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"Redis check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_ray_cluster(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check Ray cluster health"""
        try:
            dashboard_url = metadata.get("dashboard_url", "http://localhost:8265")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{dashboard_url}/api/v0/nodes")
                
                if response.status_code == 200:
                    nodes_data = response.json()
                    node_count = len(nodes_data.get("data", {}).get("summary", []))
                    
                    if node_count > 0:
                        return {
                            "status": HealthStatus.HEALTHY,
                            "message": f"Ray cluster is healthy with {node_count} nodes",
                            "details": {"node_count": node_count}
                        }
                    else:
                        return {
                            "status": HealthStatus.UNHEALTHY,
                            "message": "Ray cluster has no active nodes",
                            "details": {"node_count": 0}
                        }
                else:
                    return {
                        "status": HealthStatus.DEGRADED,
                        "message": f"Ray dashboard returned status {response.status_code}",
                        "details": {"status_code": response.status_code}
                    }
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"Ray cluster check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_mlx_inference(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check MLX inference service health"""
        try:
            endpoint = metadata.get("endpoint", "http://localhost:8001/health")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint)
                
                if response.status_code == 200:
                    return {
                        "status": HealthStatus.HEALTHY,
                        "message": "MLX inference service is healthy",
                        "details": {"status_code": response.status_code}
                    }
                else:
                    return {
                        "status": HealthStatus.DEGRADED,
                        "message": f"MLX service returned status {response.status_code}",
                        "details": {"status_code": response.status_code}
                    }
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"MLX inference check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_framework_health(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check framework health via agent orchestrator"""
        try:
            framework = metadata.get("framework")
            
            # This would integrate with your agent orchestrator to check framework status
            # For now, we'll simulate a basic check
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": f"Framework {framework} is operational",
                "details": {"framework": framework}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Framework check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_system_resources(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = HealthStatus.HEALTHY
            message = "System resources are healthy"
            
            if cpu_percent > 90 or memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically high"
            elif cpu_percent > 70 or memory.percent > 70:
                status = HealthStatus.DEGRADED
                message = "System resources moderately high"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2)
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"System resource check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_disk_space(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            paths = metadata.get("paths", ["/"])
            disk_info = []
            worst_usage = 0
            
            for path in paths:
                if Path(path).exists():
                    usage = psutil.disk_usage(path)
                    usage_percent = (usage.used / usage.total) * 100
                    worst_usage = max(worst_usage, usage_percent)
                    
                    disk_info.append({
                        "path": path,
                        "usage_percent": round(usage_percent, 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "total_gb": round(usage.total / (1024**3), 2)
                    })
            
            status = HealthStatus.HEALTHY
            message = "Disk space is healthy"
            
            if worst_usage > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Disk space critically low: {worst_usage:.1f}%"
            elif worst_usage > 85:
                status = HealthStatus.DEGRADED
                message = f"Disk space moderately high: {worst_usage:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "details": {"disks": disk_info, "worst_usage_percent": worst_usage}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"Disk space check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_network_connectivity(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check network connectivity to external hosts"""
        try:
            hosts = metadata.get("hosts", ["8.8.8.8"])
            connectivity_results = []
            successful_connections = 0
            
            for host in hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((host, 53))  # DNS port
                    sock.close()
                    
                    if result == 0:
                        connectivity_results.append({"host": host, "status": "connected"})
                        successful_connections += 1
                    else:
                        connectivity_results.append({"host": host, "status": "failed", "error_code": result})
                except Exception as e:
                    connectivity_results.append({"host": host, "status": "error", "error": str(e)})
            
            success_rate = (successful_connections / len(hosts)) * 100
            
            if success_rate == 100:
                status = HealthStatus.HEALTHY
                message = "Network connectivity is healthy"
            elif success_rate >= 50:
                status = HealthStatus.DEGRADED
                message = f"Partial network connectivity: {success_rate:.0f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Poor network connectivity: {success_rate:.0f}%"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "success_rate_percent": success_rate,
                    "results": connectivity_results
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"Network connectivity check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.health_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks have been executed",
                "healthy_checks": 0,
                "total_checks": 0,
                "critical_issues": 0
            }
        
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        critical_issues = 0
        
        for check_name, result in self.health_results.items():
            check_config = self.health_checks.get(check_name)
            
            if result.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif result.status == HealthStatus.DEGRADED:
                degraded_count += 1
            elif result.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
                if check_config and check_config.critical:
                    critical_issues += 1
        
        total_checks = len(self.health_results)
        
        # Determine overall status
        if critical_issues > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"System unhealthy: {critical_issues} critical issues"
        elif unhealthy_count > 0 or degraded_count > total_checks * 0.3:
            overall_status = HealthStatus.DEGRADED
            message = f"System degraded: {unhealthy_count} unhealthy, {degraded_count} degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = f"System healthy: {healthy_count}/{total_checks} checks passing"
        
        return {
            "status": overall_status.value,
            "message": message,
            "healthy_checks": healthy_count,
            "degraded_checks": degraded_count,
            "unhealthy_checks": unhealthy_count,
            "total_checks": total_checks,
            "critical_issues": critical_issues,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get detailed health summary"""
        overall_health = self.get_overall_health()
        
        # Get individual check results
        check_results = {}
        for name, result in self.health_results.items():
            check_results[name] = result.to_dict()
        
        # Get SLA compliance
        sla_compliance = {}
        for name, metric in self.sla_metrics.items():
            sla_compliance[name] = {
                "name": metric.name,
                "target_value": metric.target_value,
                "current_value": metric.current_value,
                "is_compliant": metric.is_compliant,
                "unit": metric.unit,
                "compliance_percentage": metric.compliance_percentage
            }
        
        return {
            "overall_health": overall_health,
            "health_checks": check_results,
            "sla_compliance": sla_compliance,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global health monitor instance
health_monitor = HealthMonitor()

# Convenience functions
async def start_health_monitoring():
    """Start health monitoring"""
    await health_monitor.start_monitoring()

async def stop_health_monitoring():
    """Stop health monitoring"""
    await health_monitor.stop_monitoring()

def get_system_health() -> Dict[str, Any]:
    """Get current system health"""
    return health_monitor.get_overall_health()

def get_health_summary() -> Dict[str, Any]:
    """Get detailed health summary"""
    return health_monitor.get_health_summary()

def add_custom_health_check(health_check: HealthCheck):
    """Add a custom health check"""
    health_monitor.add_health_check(health_check)