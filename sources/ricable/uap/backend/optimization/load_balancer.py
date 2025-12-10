# File: backend/optimization/load_balancer.py
"""
Load balancing and health checking for UAP platform.
Provides intelligent request distribution and service discovery.
"""

import asyncio
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import httpx
import statistics

logger = logging.getLogger(__name__)

class ServerStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"

@dataclass
class ServerInstance:
    """Represents a server instance in the load balancer"""
    id: str
    host: str
    port: int
    weight: int = 1
    status: ServerStatus = ServerStatus.HEALTHY
    health_score: float = 1.0
    last_health_check: Optional[datetime] = None
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    max_connections: int = 100
    current_connections: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return (self.success_count / total) if total > 0 else 1.0
    
    @property
    def is_available(self) -> bool:
        return (self.status == ServerStatus.HEALTHY and 
                self.current_connections < self.max_connections)
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics"""
        self.response_times.append(response_time)
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        if success:
            self.success_count += 1
            self.error_count = max(0, self.error_count - 1)  # Gradual error recovery
        else:
            self.error_count += 1
        
        # Update health score based on performance
        self._update_health_score()
    
    def _update_health_score(self):
        """Update health score based on metrics"""
        success_factor = self.success_rate
        response_time_factor = max(0.1, 1.0 - (self.avg_response_time / 5000))  # 5s baseline
        load_factor = max(0.1, 1.0 - (self.current_connections / self.max_connections))
        
        self.health_score = (success_factor * 0.5 + 
                           response_time_factor * 0.3 + 
                           load_factor * 0.2)

class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies"""
    
    @abstractmethod
    async def select_server(self, servers: List[ServerInstance], 
                          request_context: Dict[str, Any] = None) -> Optional[ServerInstance]:
        pass

class RoundRobinStrategy(LoadBalancingStrategy):
    """Round-robin load balancing strategy"""
    
    def __init__(self):
        self.current_index = 0
    
    async def select_server(self, servers: List[ServerInstance], 
                          request_context: Dict[str, Any] = None) -> Optional[ServerInstance]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        server = available_servers[self.current_index % len(available_servers)]
        self.current_index += 1
        return server

class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin load balancing strategy"""
    
    def __init__(self):
        self.current_weights = {}
    
    async def select_server(self, servers: List[ServerInstance], 
                          request_context: Dict[str, Any] = None) -> Optional[ServerInstance]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        # Initialize weights
        for server in available_servers:
            if server.id not in self.current_weights:
                self.current_weights[server.id] = 0
        
        # Find server with highest current weight
        max_weight = -1
        selected_server = None
        
        for server in available_servers:
            self.current_weights[server.id] += server.weight
            if self.current_weights[server.id] > max_weight:
                max_weight = self.current_weights[server.id]
                selected_server = server
        
        # Reduce selected server's current weight
        if selected_server:
            total_weight = sum(s.weight for s in available_servers)
            self.current_weights[selected_server.id] -= total_weight
        
        return selected_server

class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing strategy"""
    
    async def select_server(self, servers: List[ServerInstance], 
                          request_context: Dict[str, Any] = None) -> Optional[ServerInstance]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        return min(available_servers, key=lambda s: s.current_connections)

class HealthBasedStrategy(LoadBalancingStrategy):
    """Health-based load balancing strategy"""
    
    async def select_server(self, servers: List[ServerInstance], 
                          request_context: Dict[str, Any] = None) -> Optional[ServerInstance]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        # Weight by health score and inverse of current load
        weights = []
        for server in available_servers:
            load_factor = 1.0 - (server.current_connections / server.max_connections)
            weight = server.health_score * load_factor * server.weight
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(available_servers)
        
        r = random.random() * total_weight
        cumulative = 0
        for server, weight in zip(available_servers, weights):
            cumulative += weight
            if r <= cumulative:
                return server
        
        return available_servers[-1]

class HealthChecker:
    """Health checker for server instances"""
    
    def __init__(self, check_interval: int = 30, timeout: int = 5, 
                 health_threshold: float = 0.7):
        self.check_interval = check_interval
        self.timeout = timeout
        self.health_threshold = health_threshold
        self.running = False
        self.health_check_task = None
    
    async def start(self, servers: List[ServerInstance]):
        """Start health checking"""
        self.running = True
        self.health_check_task = asyncio.create_task(
            self._health_check_loop(servers)
        )
    
    async def stop(self):
        """Stop health checking"""
        self.running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self, servers: List[ServerInstance]):
        """Main health check loop"""
        while self.running:
            try:
                await self._check_all_servers(servers)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _check_all_servers(self, servers: List[ServerInstance]):
        """Check health of all servers"""
        tasks = [self._check_server_health(server) for server in servers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_server_health(self, server: ServerInstance):
        """Check health of a single server"""
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{server.url}/health")
                
            response_time = (time.time() - start_time) * 1000
            
            # Determine health based on response
            if response.status_code == 200:
                server.record_request(response_time, True)
                
                # Update status based on health score
                if server.health_score >= self.health_threshold:
                    if server.status == ServerStatus.UNHEALTHY:
                        server.status = ServerStatus.HEALTHY
                        logger.info(f"Server {server.id} recovered")
            else:
                server.record_request(response_time, False)
                logger.warning(f"Server {server.id} health check failed: {response.status_code}")
            
        except Exception as e:
            # Health check failed
            server.record_request(5000, False)  # Assume 5s timeout
            
            if server.health_score < self.health_threshold:
                if server.status == ServerStatus.HEALTHY:
                    server.status = ServerStatus.UNHEALTHY
                    logger.warning(f"Server {server.id} marked unhealthy: {e}")
        
        server.last_health_check = datetime.utcnow()

class LoadBalancer:
    """Main load balancer class"""
    
    def __init__(self, strategy: LoadBalancingStrategy = None, 
                 health_checker: HealthChecker = None):
        self.strategy = strategy or HealthBasedStrategy()
        self.health_checker = health_checker or HealthChecker()
        self.servers: List[ServerInstance] = []
        self.request_count = 0
        self.total_response_time = 0.0
        self.active_connections = {}  # request_id -> server_id
    
    async def start(self):
        """Start the load balancer"""
        if self.servers:
            await self.health_checker.start(self.servers)
            logger.info(f"Load balancer started with {len(self.servers)} servers")
    
    async def stop(self):
        """Stop the load balancer"""
        await self.health_checker.stop()
        logger.info("Load balancer stopped")
    
    def add_server(self, server: ServerInstance):
        """Add a server to the pool"""
        self.servers.append(server)
        logger.info(f"Added server {server.id} ({server.url})")
    
    def remove_server(self, server_id: str):
        """Remove a server from the pool"""
        self.servers = [s for s in self.servers if s.id != server_id]
        logger.info(f"Removed server {server_id}")
    
    def get_server(self, server_id: str) -> Optional[ServerInstance]:
        """Get server by ID"""
        return next((s for s in self.servers if s.id == server_id), None)
    
    async def route_request(self, request_context: Dict[str, Any] = None) -> Optional[ServerInstance]:
        """Route a request to an available server"""
        selected_server = await self.strategy.select_server(self.servers, request_context)
        
        if selected_server:
            selected_server.current_connections += 1
            request_id = request_context.get('request_id') if request_context else str(time.time())
            self.active_connections[request_id] = selected_server.id
            
            logger.debug(f"Routed request to server {selected_server.id}")
        
        return selected_server
    
    async def complete_request(self, request_id: str, response_time: float, success: bool):
        """Mark a request as completed"""
        if request_id in self.active_connections:
            server_id = self.active_connections.pop(request_id)
            server = self.get_server(server_id)
            
            if server:
                server.current_connections = max(0, server.current_connections - 1)
                server.record_request(response_time, success)
            
            # Update global stats
            self.request_count += 1
            self.total_response_time += response_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_servers = len([s for s in self.servers if s.status == ServerStatus.HEALTHY])
        total_connections = sum(s.current_connections for s in self.servers)
        avg_response_time = (self.total_response_time / self.request_count 
                           if self.request_count > 0 else 0)
        
        server_stats = []
        for server in self.servers:
            server_stats.append({
                'id': server.id,
                'url': server.url,
                'status': server.status.value,
                'health_score': round(server.health_score, 3),
                'current_connections': server.current_connections,
                'avg_response_time': round(server.avg_response_time, 2),
                'success_rate': round(server.success_rate * 100, 2),
                'weight': server.weight
            })
        
        return {
            'total_servers': len(self.servers),
            'healthy_servers': healthy_servers,
            'total_connections': total_connections,
            'total_requests': self.request_count,
            'avg_response_time': round(avg_response_time, 2),
            'strategy': self.strategy.__class__.__name__,
            'servers': server_stats
        }
    
    def set_server_weight(self, server_id: str, weight: int):
        """Update server weight"""
        server = self.get_server(server_id)
        if server:
            server.weight = weight
            logger.info(f"Updated server {server_id} weight to {weight}")
    
    def set_server_status(self, server_id: str, status: ServerStatus):
        """Update server status"""
        server = self.get_server(server_id)
        if server:
            old_status = server.status
            server.status = status
            logger.info(f"Updated server {server_id} status from {old_status.value} to {status.value}")
    
    async def drain_server(self, server_id: str, timeout: int = 300):
        """Gracefully drain a server"""
        server = self.get_server(server_id)
        if not server:
            return False
        
        server.status = ServerStatus.DRAINING
        logger.info(f"Draining server {server_id}")
        
        # Wait for connections to finish
        start_time = time.time()
        while server.current_connections > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        if server.current_connections == 0:
            server.status = ServerStatus.MAINTENANCE
            logger.info(f"Server {server_id} successfully drained")
            return True
        else:
            logger.warning(f"Server {server_id} drain timeout, {server.current_connections} connections remaining")
            return False

# Example usage and testing
async def example_usage():
    """Example usage of the load balancer"""
    # Create load balancer with health-based strategy
    lb = LoadBalancer(strategy=HealthBasedStrategy())
    
    # Add servers
    servers = [
        ServerInstance("server1", "127.0.0.1", 8001, weight=2),
        ServerInstance("server2", "127.0.0.1", 8002, weight=1),
        ServerInstance("server3", "127.0.0.1", 8003, weight=3),
    ]
    
    for server in servers:
        lb.add_server(server)
    
    # Start load balancer
    await lb.start()
    
    # Route some requests
    for i in range(10):
        server = await lb.route_request({'request_id': f'req_{i}'})
        if server:
            print(f"Request {i} routed to {server.id}")
            
            # Simulate request completion
            await asyncio.sleep(0.1)
            await lb.complete_request(f'req_{i}', 150, True)
    
    # Print stats
    stats = lb.get_stats()
    print(f"Load balancer stats: {stats}")
    
    # Stop load balancer
    await lb.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())