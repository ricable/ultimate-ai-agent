"""
Comprehensive Health Monitoring System for MLX-Exo Distributed Cluster
Provides health checks, failover, node failure detection, and recovery mechanisms
"""

import asyncio
import logging
import time
import json
import psutil
import socket
import subprocess
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Types of system components to monitor"""
    API_SERVER = "api_server"
    CLUSTER_MANAGER = "cluster_manager"
    NODE = "node"
    MEMORY_MANAGER = "memory_manager"
    LOAD_BALANCER = "load_balancer"
    NETWORK = "network"
    MODEL = "model"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def status(self) -> HealthStatus:
        """Determine status based on thresholds"""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_type: ComponentType
    component_id: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    last_healthy: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_type': self.component_type.value,
            'component_id': self.component_id,
            'status': self.status.value,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'status': m.status().value,
                    'timestamp': m.timestamp.isoformat()
                } for m in self.metrics
            ],
            'last_check': self.last_check.isoformat(),
            'last_healthy': self.last_healthy.isoformat() if self.last_healthy else None,
            'error_message': self.error_message
        }

class HealthChecker:
    """Base class for component health checkers"""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        
    async def check_health(self) -> ComponentHealth:
        """Perform health check and return status"""
        raise NotImplementedError

class NodeHealthChecker(HealthChecker):
    """Health checker for individual cluster nodes"""
    
    def __init__(self, node_id: str, node_ip: str, node_port: int = 52415):
        super().__init__(node_id, ComponentType.NODE)
        self.node_ip = node_ip
        self.node_port = node_port
        
    async def check_health(self) -> ComponentHealth:
        """Check node health including connectivity, resources, and services"""
        health = ComponentHealth(
            component_type=self.component_type,
            component_id=self.component_id,
            status=HealthStatus.HEALTHY
        )
        
        try:
            # Network connectivity check
            connectivity_metric = await self._check_connectivity()
            health.metrics.append(connectivity_metric)
            
            # Resource utilization checks
            if self.node_ip in ['localhost', '127.0.0.1']:
                # Local node - can check resources directly
                cpu_metric = self._check_cpu_usage()
                memory_metric = self._check_memory_usage() 
                disk_metric = self._check_disk_usage()
                
                health.metrics.extend([cpu_metric, memory_metric, disk_metric])
                
            # Service health check
            service_metric = await self._check_service_health()
            health.metrics.append(service_metric)
            
            # Determine overall status
            critical_metrics = [m for m in health.metrics if m.status() == HealthStatus.CRITICAL]
            degraded_metrics = [m for m in health.metrics if m.status() == HealthStatus.DEGRADED]
            
            if critical_metrics:
                health.status = HealthStatus.CRITICAL
                health.error_message = f"Critical metrics: {[m.name for m in critical_metrics]}"
            elif degraded_metrics:
                health.status = HealthStatus.DEGRADED
                health.error_message = f"Degraded metrics: {[m.name for m in degraded_metrics]}"
            else:
                health.status = HealthStatus.HEALTHY
                health.last_healthy = datetime.now()
                
        except Exception as e:
            health.status = HealthStatus.FAILED
            health.error_message = f"Health check failed: {str(e)}"
            logger.error(f"Node {self.component_id} health check failed: {e}")
            
        return health
    
    async def _check_connectivity(self) -> HealthMetric:
        """Check network connectivity to node"""
        try:
            start_time = time.time()
            
            # Ping test
            process = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1000', self.node_ip,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if process.returncode == 0:
                # Extract actual ping time if available
                try:
                    ping_output = stdout.decode()
                    if 'time=' in ping_output:
                        time_part = ping_output.split('time=')[1].split()[0]
                        latency = float(time_part)
                except:
                    pass  # Use our measured time
                    
                return HealthMetric(
                    name="network_latency",
                    value=latency,
                    threshold_warning=100.0,  # 100ms
                    threshold_critical=500.0,  # 500ms
                    unit="ms"
                )
            else:
                return HealthMetric(
                    name="network_latency", 
                    value=9999.0,  # Very high value for unreachable
                    threshold_warning=100.0,
                    threshold_critical=500.0,
                    unit="ms"
                )
                
        except asyncio.TimeoutError:
            return HealthMetric(
                name="network_latency",
                value=9999.0,
                threshold_warning=100.0,
                threshold_critical=500.0,
                unit="ms"
            )
    
    def _check_cpu_usage(self) -> HealthMetric:
        """Check CPU utilization"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            unit="%"
        )
    
    def _check_memory_usage(self) -> HealthMetric:
        """Check memory utilization"""
        memory = psutil.virtual_memory()
        return HealthMetric(
            name="memory_usage",
            value=memory.percent,
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="%"
        )
    
    def _check_disk_usage(self) -> HealthMetric:
        """Check disk utilization"""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        return HealthMetric(
            name="disk_usage",
            value=usage_percent,
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="%"
        )
    
    async def _check_service_health(self) -> HealthMetric:
        """Check if services are responding on the node"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.node_ip, self.node_port),
                timeout=5.0
            )
            writer.close()
            await writer.wait_closed()
            
            return HealthMetric(
                name="service_status",
                value=1.0,  # 1 = healthy, 0 = unhealthy
                threshold_warning=0.5,
                threshold_critical=0.5,
                unit="bool"
            )
        except:
            return HealthMetric(
                name="service_status",
                value=0.0,
                threshold_warning=0.5,
                threshold_critical=0.5,
                unit="bool"
            )

class ClusterHealthMonitor:
    """Main health monitoring coordinator for the entire cluster"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.checkers: Dict[str, HealthChecker] = {}
        self.health_history: Dict[str, List[ComponentHealth]] = {}
        self.monitoring_active = False
        self.check_interval = 10  # seconds
        self.history_retention = 100  # Keep last 100 health checks per component
        self.failover_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Load configuration
        if config_path:
            self._load_config(config_path)
    
    def add_node_checker(self, node_id: str, node_ip: str, node_port: int = 52415):
        """Add a node health checker"""
        checker = NodeHealthChecker(node_id, node_ip, node_port)
        self.checkers[node_id] = checker
        self.health_history[node_id] = []
        logger.info(f"Added health checker for node {node_id} at {node_ip}:{node_port}")
    
    def add_failover_callback(self, callback: Callable[[str, ComponentHealth], Any]):
        """Add callback to be called when failover is needed"""
        self.failover_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, ComponentHealth], Any]):
        """Add callback to be called when alerts should be sent"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start the health monitoring loop"""
        self.monitoring_active = True
        logger.info("Starting health monitoring")
        
        while self.monitoring_active:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop the health monitoring"""
        self.monitoring_active = False
        logger.info("Stopping health monitoring")
    
    async def _run_health_checks(self):
        """Run health checks for all components"""
        check_tasks = []
        
        for checker_id, checker in self.checkers.items():
            task = asyncio.create_task(self._check_component(checker_id, checker))
            check_tasks.append(task)
        
        if check_tasks:
            await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def _check_component(self, checker_id: str, checker: HealthChecker):
        """Check health of a single component"""
        try:
            health = await checker.check_health()
            
            # Store in history
            self.health_history[checker_id].append(health)
            
            # Trim history if too long
            if len(self.health_history[checker_id]) > self.history_retention:
                self.health_history[checker_id] = self.health_history[checker_id][-self.history_retention:]
            
            # Handle failures and alerts
            await self._handle_health_status(checker_id, health)
            
        except Exception as e:
            logger.error(f"Failed to check health for {checker_id}: {e}")
    
    async def _handle_health_status(self, component_id: str, health: ComponentHealth):
        """Handle health status changes and trigger appropriate actions"""
        
        # Check for status changes
        history = self.health_history[component_id]
        if len(history) > 1:
            previous_status = history[-2].status
            current_status = health.status
            
            if previous_status != current_status:
                logger.info(f"Component {component_id} status changed: {previous_status.value} -> {current_status.value}")
                
                # Trigger alerts for degraded or failed states
                if current_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL, HealthStatus.FAILED]:
                    await self._trigger_alerts(component_id, health)
                
                # Trigger failover for failed states
                if current_status == HealthStatus.FAILED:
                    await self._trigger_failover(component_id, health)
    
    async def _trigger_alerts(self, component_id: str, health: ComponentHealth):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_id, health)
                else:
                    callback(component_id, health)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def _trigger_failover(self, component_id: str, health: ComponentHealth):
        """Trigger failover callbacks"""
        logger.warning(f"Triggering failover for failed component: {component_id}")
        
        for callback in self.failover_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_id, health)
                else:
                    callback(component_id, health)
            except Exception as e:
                logger.error(f"Failover callback failed: {e}")
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health status"""
        if not self.health_history:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'components': {},
                'summary': {
                    'healthy': 0,
                    'degraded': 0,
                    'critical': 0,
                    'failed': 0,
                    'total': 0
                }
            }
        
        latest_health = {}
        summary = {'healthy': 0, 'degraded': 0, 'critical': 0, 'failed': 0, 'total': 0}
        
        for component_id, history in self.health_history.items():
            if history:
                latest = history[-1]
                latest_health[component_id] = latest.to_dict()
                summary[latest.status.value] += 1
                summary['total'] += 1
        
        # Determine overall cluster status
        if summary['failed'] > 0:
            overall_status = HealthStatus.FAILED
        elif summary['critical'] > 0:
            overall_status = HealthStatus.CRITICAL  
        elif summary['degraded'] > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'status': overall_status.value,
            'components': latest_health,
            'summary': summary,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_component_health(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific component"""
        if component_id in self.health_history and self.health_history[component_id]:
            return self.health_history[component_id][-1].to_dict()
        return None
    
    def get_health_history(self, component_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health history for a component"""
        if component_id in self.health_history:
            history = self.health_history[component_id][-limit:]
            return [h.to_dict() for h in history]
        return []
    
    def _load_config(self, config_path: str):
        """Load monitoring configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Configure monitoring parameters
            self.check_interval = config.get('check_interval', 10)
            self.history_retention = config.get('history_retention', 100)
            
            # Add node checkers from config
            for node_config in config.get('nodes', []):
                self.add_node_checker(
                    node_config['id'],
                    node_config['ip'],
                    node_config.get('port', 52415)
                )
                
            logger.info(f"Loaded monitoring configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")

# Example alert and failover handlers
async def log_alert_handler(component_id: str, health: ComponentHealth):
    """Simple alert handler that logs alerts"""
    logger.warning(f"ALERT: Component {component_id} is {health.status.value}")
    if health.error_message:
        logger.warning(f"Error: {health.error_message}")

async def simple_failover_handler(component_id: str, health: ComponentHealth):
    """Simple failover handler"""
    logger.critical(f"FAILOVER: Component {component_id} has failed")
    # In a real implementation, this would:
    # 1. Remove failed node from load balancer
    # 2. Redistribute workload to healthy nodes
    # 3. Attempt recovery procedures
    # 4. Send notifications to administrators

def create_health_monitor(nodes: List[Dict[str, Any]]) -> ClusterHealthMonitor:
    """Factory function to create a health monitor with node configuration"""
    monitor = ClusterHealthMonitor()
    
    # Add default alert and failover handlers
    monitor.add_alert_callback(log_alert_handler)
    monitor.add_failover_callback(simple_failover_handler)
    
    # Add node checkers
    for node in nodes:
        monitor.add_node_checker(
            node['id'],
            node['ip'], 
            node.get('port', 52415)
        )
    
    return monitor

# Usage example
if __name__ == "__main__":
    import asyncio
    
    # Example node configuration
    nodes = [
        {'id': 'mac-studio-1', 'ip': '10.0.1.10'},
        {'id': 'mac-studio-2', 'ip': '10.0.1.11'},
        {'id': 'mac-studio-3', 'ip': '10.0.1.12'}
    ]
    
    async def main():
        monitor = create_health_monitor(nodes)
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for a bit
        await asyncio.sleep(30)
        
        # Get cluster health
        health = monitor.get_cluster_health()
        print(json.dumps(health, indent=2))
        
        # Stop monitoring
        monitor.stop_monitoring()
        await monitoring_task
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())