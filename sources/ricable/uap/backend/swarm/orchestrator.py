"""
Swarm Orchestrator for Phase 5 Agent Coordination
Manages swarm-level coordination, consensus, and resource allocation
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import uuid4
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor

from ..agents.coordination import (
    AgentCoordinator, Agent, Task, AgentRole, TaskStatus, 
    AgentCapability, ConsensusProposal, Conflict
)

logger = logging.getLogger(__name__)


@dataclass
class SwarmConfig:
    """Configuration for swarm behavior"""
    max_agents: int = 100
    consensus_threshold: float = 0.66
    task_timeout: int = 3600  # seconds
    heartbeat_interval: int = 30  # seconds
    resource_sharing_enabled: bool = True
    auto_scaling_enabled: bool = True
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, capability_based


@dataclass
class SwarmMetrics:
    """Metrics for swarm performance monitoring"""
    total_agents: int
    active_agents: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_task_completion_time: float
    system_throughput: float
    resource_utilization: float
    consensus_success_rate: float
    conflict_resolution_rate: float


class SwarmOrchestrator:
    """
    Main orchestrator for swarm-based multi-agent coordination
    Handles high-level swarm management, scaling, and optimization
    """
    
    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.coordinator = AgentCoordinator()
        self.swarm_id = str(uuid4())
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Swarm state
        self.swarm_agents: Dict[str, Agent] = {}
        self.agent_connections: Dict[str, Any] = {}  # WebSocket connections
        self.performance_history: List[SwarmMetrics] = []
        self.scaling_policies: List[Callable] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "agent_joined": [],
            "agent_left": [],
            "task_completed": [],
            "consensus_reached": [],
            "conflict_detected": [],
            "scaling_triggered": []
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
    
    async def initialize(self) -> bool:
        """Initialize the swarm orchestrator"""
        try:
            logger.info(f"Initializing Swarm Orchestrator {self.swarm_id}")
            
            # Setup default scaling policies
            self._setup_default_scaling_policies()
            
            # Start background monitoring
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("Swarm Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm orchestrator: {e}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the swarm orchestrator"""
        logger.info("Shutting down Swarm Orchestrator")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close agent connections
        for conn in self.agent_connections.values():
            if hasattr(conn, 'close'):
                await conn.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Swarm Orchestrator shutdown complete")
    
    async def register_agent(self, agent: Agent, connection: Any = None) -> bool:
        """Register a new agent with the swarm"""
        try:
            # Register with the coordinator
            success = await self.coordinator.register_agent(agent)
            
            if success:
                self.swarm_agents[agent.id] = agent
                if connection:
                    self.agent_connections[agent.id] = connection
                
                # Trigger scaling evaluation
                await self._evaluate_scaling_needs()
                
                # Emit event
                await self._emit_event("agent_joined", {"agent": agent})
                
                logger.info(f"Agent {agent.id} joined swarm {self.swarm_id}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {e}")
        
        return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the swarm"""
        try:
            # Unregister from coordinator
            success = await self.coordinator.unregister_agent(agent_id)
            
            if success and agent_id in self.swarm_agents:
                agent = self.swarm_agents[agent_id]
                del self.swarm_agents[agent_id]
                
                # Close connection if exists
                if agent_id in self.agent_connections:
                    conn = self.agent_connections[agent_id]
                    if hasattr(conn, 'close'):
                        await conn.close()
                    del self.agent_connections[agent_id]
                
                # Emit event
                await self._emit_event("agent_left", {"agent": agent})
                
                logger.info(f"Agent {agent_id} left swarm {self.swarm_id}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
        
        return False
    
    async def submit_swarm_task(self, task_description: str, requirements: Dict[str, Any]) -> str:
        """Submit a task to the swarm for distributed execution"""
        task = Task(
            id=str(uuid4()),
            description=task_description,
            required_capabilities=requirements.get("capabilities", []),
            priority=requirements.get("priority", 5),
            deadline=requirements.get("deadline"),
            status=TaskStatus.PENDING,
            assigned_agent=None,
            subtasks=[],
            progress=0.0,
            metadata=requirements.get("metadata", {}),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Submit to coordinator
        task_id = await self.coordinator.submit_task(task)
        
        # Start task monitoring
        await self._start_task_monitoring(task_id)
        
        return task_id
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        coordinator_status = await self.coordinator.monitor_system_health()
        
        # Calculate swarm-specific metrics
        swarm_metrics = await self._calculate_swarm_metrics()
        
        return {
            "swarm_id": self.swarm_id,
            "is_running": self.is_running,
            "config": asdict(self.config),
            "coordinator_status": coordinator_status,
            "swarm_metrics": asdict(swarm_metrics),
            "agents": {
                "total": len(self.swarm_agents),
                "connected": len(self.agent_connections),
                "by_role": self._count_agents_by_role()
            },
            "performance_history": [asdict(m) for m in self.performance_history[-10:]]
        }
    
    async def execute_consensus_decision(self, decision_type: str, 
                                       data: Dict[str, Any]) -> str:
        """Execute a consensus decision across the swarm"""
        # Get all online agents
        online_agents = [a for a in self.swarm_agents.values() if a.status == "online"]
        
        if not online_agents:
            raise ValueError("No online agents available for consensus")
        
        # Select a coordinator agent to propose
        coordinator_agents = [a for a in online_agents if a.role == AgentRole.COORDINATOR]
        if not coordinator_agents:
            # If no coordinators, select the most reputable agent
            coordinator_agents = [max(online_agents, key=lambda x: x.reputation)]
        
        proposer = coordinator_agents[0].id
        
        # Create consensus proposal
        proposal_id = await self.coordinator.consensus_algorithm.propose(
            proposer, decision_type, data, online_agents
        )
        
        # Start consensus monitoring
        await self._monitor_consensus(proposal_id)
        
        return proposal_id
    
    async def negotiate_resource_allocation(self, resource_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Negotiate resource allocation across the swarm"""
        allocation_results = {}
        
        for request in resource_requests:
            requesting_agent = request["agent_id"]
            resource_type = request["resource_type"]
            amount = request["amount"]
            
            success = await self.coordinator.negotiate_resource_sharing(
                requesting_agent, resource_type, amount
            )
            
            allocation_results[requesting_agent] = {
                "resource_type": resource_type,
                "amount": amount,
                "allocated": success
            }
        
        return allocation_results
    
    async def optimize_swarm_performance(self) -> Dict[str, Any]:
        """Optimize overall swarm performance"""
        optimization_results = {}
        
        # Analyze current performance
        metrics = await self._calculate_swarm_metrics()
        
        # Load balancing optimization
        if metrics.resource_utilization > 0.8:
            await self._trigger_load_balancing()
            optimization_results["load_balancing"] = "triggered"
        
        # Auto-scaling evaluation
        if self.config.auto_scaling_enabled:
            scaling_action = await self._evaluate_auto_scaling(metrics)
            if scaling_action:
                optimization_results["scaling"] = scaling_action
        
        # Task reallocation if needed
        stuck_tasks = await self._identify_stuck_tasks()
        if stuck_tasks:
            await self._reallocate_tasks(stuck_tasks)
            optimization_results["task_reallocation"] = len(stuck_tasks)
        
        return optimization_results
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Heartbeat monitoring
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self.background_tasks.add(heartbeat_task)
        
        # Performance monitoring
        metrics_task = asyncio.create_task(self._metrics_collector())
        self.background_tasks.add(metrics_task)
        
        # Health monitoring
        health_task = asyncio.create_task(self._health_monitor())
        self.background_tasks.add(health_task)
        
        # Optimization loop
        optimization_task = asyncio.create_task(self._optimization_loop())
        self.background_tasks.add(optimization_task)
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and handle disconnections"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, agent in list(self.swarm_agents.items()):
                    time_since_last_seen = current_time - agent.last_seen
                    
                    if time_since_last_seen > timedelta(seconds=self.config.heartbeat_interval * 3):
                        # Agent appears to be offline
                        if agent.status != "offline":
                            agent.status = "offline"
                            logger.warning(f"Agent {agent_id} marked as offline")
                            
                            # Optionally remove from swarm
                            await self.unregister_agent(agent_id)
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collector(self):
        """Collect and store swarm performance metrics"""
        while self.is_running:
            try:
                metrics = await self._calculate_swarm_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(30)
    
    async def _health_monitor(self):
        """Monitor overall swarm health and trigger alerts"""
        while self.is_running:
            try:
                status = await self.get_swarm_status()
                
                # Check for health issues
                if status["agents"]["total"] == 0:
                    logger.warning("No agents in swarm")
                
                coordinator_status = status["coordinator_status"]
                if coordinator_status["system_load"] > 0.9:
                    logger.warning("High system load detected")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)
    
    async def _optimization_loop(self):
        """Continuous optimization loop"""
        while self.is_running:
            try:
                await self.optimize_swarm_performance()
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_swarm_metrics(self) -> SwarmMetrics:
        """Calculate current swarm performance metrics"""
        total_agents = len(self.swarm_agents)
        active_agents = len([a for a in self.swarm_agents.values() if a.status == "online"])
        
        # Get task statistics from coordinator
        coordinator_status = await self.coordinator.monitor_system_health()
        
        return SwarmMetrics(
            total_agents=total_agents,
            active_agents=active_agents,
            total_tasks=coordinator_status["tasks"]["total"],
            completed_tasks=coordinator_status["tasks"]["completed"],
            failed_tasks=0,  # Would be calculated from task history
            average_task_completion_time=120.0,  # Would be calculated from history
            system_throughput=coordinator_status["tasks"]["completed"] / 3600,  # tasks per hour
            resource_utilization=coordinator_status["system_load"],
            consensus_success_rate=0.95,  # Would be calculated from consensus history
            conflict_resolution_rate=0.90   # Would be calculated from conflict history
        )
    
    def _count_agents_by_role(self) -> Dict[str, int]:
        """Count agents by their roles"""
        role_counts = {}
        for agent in self.swarm_agents.values():
            role = agent.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        return role_counts
    
    def _setup_default_scaling_policies(self):
        """Setup default auto-scaling policies"""
        async def scale_up_policy(metrics: SwarmMetrics):
            if (metrics.resource_utilization > 0.8 and 
                metrics.total_agents < self.config.max_agents):
                return "scale_up"
            return None
        
        async def scale_down_policy(metrics: SwarmMetrics):
            if (metrics.resource_utilization < 0.3 and 
                metrics.total_agents > 1):
                return "scale_down"
            return None
        
        self.scaling_policies = [scale_up_policy, scale_down_policy]
    
    async def _evaluate_scaling_needs(self):
        """Evaluate if scaling is needed"""
        if not self.config.auto_scaling_enabled:
            return
        
        metrics = await self._calculate_swarm_metrics()
        
        for policy in self.scaling_policies:
            action = await policy(metrics)
            if action:
                await self._emit_event("scaling_triggered", {
                    "action": action,
                    "metrics": asdict(metrics)
                })
    
    async def _evaluate_auto_scaling(self, metrics: SwarmMetrics) -> Optional[str]:
        """Evaluate auto-scaling based on current metrics"""
        for policy in self.scaling_policies:
            action = await policy(metrics)
            if action:
                return action
        return None
    
    async def _trigger_load_balancing(self):
        """Trigger load balancing across agents"""
        logger.info("Triggering load balancing")
        # Implementation would redistribute tasks across agents
    
    async def _identify_stuck_tasks(self) -> List[str]:
        """Identify tasks that appear to be stuck"""
        stuck_tasks = []
        current_time = datetime.utcnow()
        
        for task in self.coordinator.tasks.values():
            if (task.status == TaskStatus.IN_PROGRESS and 
                current_time - task.updated_at > timedelta(seconds=self.config.task_timeout)):
                stuck_tasks.append(task.id)
        
        return stuck_tasks
    
    async def _reallocate_tasks(self, task_ids: List[str]):
        """Reallocate stuck tasks to different agents"""
        for task_id in task_ids:
            if task_id in self.coordinator.tasks:
                task = self.coordinator.tasks[task_id]
                task.status = TaskStatus.PENDING
                task.assigned_agent = None
                logger.info(f"Reallocating stuck task {task_id}")
    
    async def _start_task_monitoring(self, task_id: str):
        """Start monitoring a specific task"""
        async def monitor_task():
            while task_id in self.coordinator.tasks:
                task = self.coordinator.tasks[task_id]
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    await self._emit_event("task_completed", {
                        "task_id": task_id,
                        "status": task.status.value
                    })
                    break
                await asyncio.sleep(10)
        
        task = asyncio.create_task(monitor_task())
        self.background_tasks.add(task)
    
    async def _monitor_consensus(self, proposal_id: str):
        """Monitor a consensus proposal"""
        async def monitor_proposal():
            while proposal_id in self.coordinator.consensus_algorithm.proposals:
                proposal = self.coordinator.consensus_algorithm.proposals[proposal_id]
                if proposal.status != "pending":
                    await self._emit_event("consensus_reached", {
                        "proposal_id": proposal_id,
                        "status": proposal.status
                    })
                    break
                await asyncio.sleep(5)
        
        task = asyncio.create_task(monitor_proposal())
        self.background_tasks.add(task)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove an event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass