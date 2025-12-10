"""
Load Balancer and Request Router for Distributed API Gateway
Handles intelligent routing of requests across cluster nodes
"""

import asyncio
import logging
import time
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Available load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    CONSISTENT_HASHING = "consistent_hashing"

class NodeStatus(Enum):
    """Node status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class NodeInfo:
    """Information about a cluster node"""
    node_id: str
    host: str
    port: int
    weight: float = 1.0
    status: NodeStatus = NodeStatus.HEALTHY
    current_connections: int = 0
    max_connections: int = 100
    response_time: float = 0.0
    last_health_check: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    
    # Resource information
    memory_usage: float = 0.0  # Percentage
    cpu_usage: float = 0.0     # Percentage
    gpu_usage: float = 0.0     # Percentage
    active_models: List[str] = field(default_factory=list)
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for requests"""
        return (
            self.status == NodeStatus.HEALTHY and
            self.current_connections < self.max_connections
        )
    
    @property
    def load_score(self) -> float:
        """Calculate load score for resource-aware balancing"""
        if not self.is_available:
            return float('inf')
        
        # Combine multiple factors
        connection_load = self.current_connections / self.max_connections
        resource_load = (self.memory_usage + self.cpu_usage + self.gpu_usage) / 300
        response_factor = min(self.response_time / 1000, 1.0)  # Cap at 1 second
        
        return connection_load * 0.4 + resource_load * 0.4 + response_factor * 0.2

@dataclass
class RequestRoute:
    """Information about how a request should be routed"""
    target_node: NodeInfo
    request_id: str
    model_name: str
    estimated_tokens: int
    routing_reason: str
    created_at: float = field(default_factory=time.time)

class LoadBalancer:
    """
    Intelligent load balancer for distributed inference requests
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE):
        self.strategy = strategy
        self.nodes: Dict[str, NodeInfo] = {}
        self.round_robin_index = 0
        self.hash_ring: Dict[str, str] = {}  # For consistent hashing
        self.request_history: List[RequestRoute] = []
        
        # Configuration
        self.health_check_interval = 30  # seconds
        self.request_timeout = 300  # seconds
        self.max_history = 1000
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        logger.info(f"Load balancer initialized with strategy: {strategy.value}")
    
    def add_node(self, node_info: NodeInfo) -> None:
        """Add a node to the load balancer"""
        self.nodes[node_info.node_id] = node_info
        self._rebuild_hash_ring()
        logger.info(f"Added node {node_info.node_id} at {node_info.host}:{node_info.port}")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the load balancer"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._rebuild_hash_ring()
            logger.info(f"Removed node {node_id}")
    
    def update_node_status(self, node_id: str, status_update: Dict[str, Any]) -> None:
        """Update node status and metrics"""
        if node_id not in self.nodes:
            logger.warning(f"Attempted to update unknown node: {node_id}")
            return
        
        node = self.nodes[node_id]
        
        # Update status
        if 'status' in status_update:
            node.status = NodeStatus(status_update['status'])
        
        # Update resource metrics
        if 'memory_usage' in status_update:
            node.memory_usage = status_update['memory_usage']
        if 'cpu_usage' in status_update:
            node.cpu_usage = status_update['cpu_usage']
        if 'gpu_usage' in status_update:
            node.gpu_usage = status_update['gpu_usage']
        if 'response_time' in status_update:
            node.response_time = status_update['response_time']
        if 'active_models' in status_update:
            node.active_models = status_update['active_models']
        
        node.last_health_check = time.time()
        
        logger.debug(f"Updated node {node_id} status: {status_update}")
    
    async def route_request(
        self, 
        request_id: str, 
        model_name: str, 
        estimated_tokens: int = 100,
        user_id: Optional[str] = None,
        sticky_session: Optional[str] = None
    ) -> Optional[RequestRoute]:
        """
        Route a request to the best available node
        """
        self.total_requests += 1
        
        # Filter available nodes
        available_nodes = [
            node for node in self.nodes.values()
            if node.is_available and model_name in node.active_models
        ]
        
        if not available_nodes:
            logger.warning(f"No available nodes for model {model_name}")
            self.failed_requests += 1
            return None
        
        # Select node based on strategy
        selected_node = None
        routing_reason = ""
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_node, routing_reason = self._round_robin_select(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_node, routing_reason = self._least_connections_select(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_node, routing_reason = self._weighted_round_robin_select(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            selected_node, routing_reason = self._resource_aware_select(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASHING:
            key = sticky_session or user_id or request_id
            selected_node, routing_reason = self._consistent_hash_select(available_nodes, key)
        
        if selected_node:
            # Update connection count
            selected_node.current_connections += 1
            
            # Create route
            route = RequestRoute(
                target_node=selected_node,
                request_id=request_id,
                model_name=model_name,
                estimated_tokens=estimated_tokens,
                routing_reason=routing_reason
            )
            
            # Track request
            self.request_history.append(route)
            if len(self.request_history) > self.max_history:
                self.request_history = self.request_history[-self.max_history:]
            
            logger.info(f"Routed request {request_id} to {selected_node.node_id}: {routing_reason}")
            return route
        
        self.failed_requests += 1
        return None
    
    def release_request(self, request_id: str) -> None:
        """Release a request and update node connection count"""
        for route in reversed(self.request_history):
            if route.request_id == request_id:
                route.target_node.current_connections = max(0, route.target_node.current_connections - 1)
                logger.debug(f"Released request {request_id} from {route.target_node.node_id}")
                break
    
    def _round_robin_select(self, nodes: List[NodeInfo]) -> Tuple[NodeInfo, str]:
        """Round-robin selection"""
        selected = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected, f"round_robin (index {self.round_robin_index - 1})"
    
    def _least_connections_select(self, nodes: List[NodeInfo]) -> Tuple[NodeInfo, str]:
        """Select node with least connections"""
        selected = min(nodes, key=lambda n: n.current_connections)
        return selected, f"least_connections ({selected.current_connections} active)"
    
    def _weighted_round_robin_select(self, nodes: List[NodeInfo]) -> Tuple[NodeInfo, str]:
        """Weighted round-robin selection"""
        # Calculate weights
        total_weight = sum(n.weight for n in nodes)
        if total_weight == 0:
            return self._round_robin_select(nodes)
        
        # Use weighted random selection
        r = random.uniform(0, total_weight)
        current_weight = 0
        
        for node in nodes:
            current_weight += node.weight
            if r <= current_weight:
                return node, f"weighted_round_robin (weight {node.weight})"
        
        # Fallback
        return nodes[-1], f"weighted_round_robin_fallback"
    
    def _resource_aware_select(self, nodes: List[NodeInfo]) -> Tuple[NodeInfo, str]:
        """Select node based on resource availability"""
        selected = min(nodes, key=lambda n: n.load_score)
        return selected, f"resource_aware (load_score {selected.load_score:.3f})"
    
    def _consistent_hash_select(self, nodes: List[NodeInfo], key: str) -> Tuple[NodeInfo, str]:
        """Consistent hashing selection"""
        if not key:
            return self._round_robin_select(nodes)
        
        # Hash the key
        hash_value = hashlib.md5(key.encode()).hexdigest()
        
        # Find the node in the hash ring
        if hash_value in self.hash_ring:
            node_id = self.hash_ring[hash_value]
            for node in nodes:
                if node.node_id == node_id:
                    return node, f"consistent_hash (key {key[:8]}...)"
        
        # Fallback to ring lookup
        ring_keys = sorted(self.hash_ring.keys())
        for ring_key in ring_keys:
            if hash_value <= ring_key:
                node_id = self.hash_ring[ring_key]
                for node in nodes:
                    if node.node_id == node_id:
                        return node, f"consistent_hash_ring (key {key[:8]}...)"
        
        # Fallback
        return self._round_robin_select(nodes)
    
    def _rebuild_hash_ring(self) -> None:
        """Rebuild consistent hash ring"""
        self.hash_ring.clear()
        
        # Create virtual nodes for better distribution
        virtual_nodes_per_node = 150
        
        for node in self.nodes.values():
            for i in range(virtual_nodes_per_node):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = hashlib.md5(virtual_key.encode()).hexdigest()
                self.hash_ring[hash_value] = node.node_id
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        healthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
        total_connections = sum(n.current_connections for n in self.nodes.values())
        avg_response_time = sum(n.response_time for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0
        
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "degraded_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.DEGRADED),
            "failed_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED),
            "total_connections": total_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            "avg_response_time": avg_response_time,
            "uptime": time.time() - self.start_time,
            "strategy": self.strategy.value,
            "nodes": {
                node_id: {
                    "status": node.status.value,
                    "connections": node.current_connections,
                    "load_score": node.load_score,
                    "memory_usage": node.memory_usage,
                    "cpu_usage": node.cpu_usage,
                    "gpu_usage": node.gpu_usage,
                    "response_time": node.response_time,
                    "active_models": node.active_models
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.request_history:
            return {"total_routes": 0}
        
        # Analyze recent routing decisions
        recent_routes = [r for r in self.request_history if time.time() - r.created_at < 3600]  # Last hour
        
        node_distribution = {}
        model_distribution = {}
        
        for route in recent_routes:
            node_id = route.target_node.node_id
            model = route.model_name
            
            node_distribution[node_id] = node_distribution.get(node_id, 0) + 1
            model_distribution[model] = model_distribution.get(model, 0) + 1
        
        return {
            "total_routes": len(self.request_history),
            "recent_routes": len(recent_routes),
            "node_distribution": node_distribution,
            "model_distribution": model_distribution,
            "strategy": self.strategy.value
        }

class RequestRouter:
    """
    High-level request router that integrates with the load balancer
    """
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.active_routes: Dict[str, RequestRoute] = {}
        self.route_callbacks: Dict[str, List[callable]] = {}
    
    async def route_and_execute(
        self, 
        request_data: Dict[str, Any],
        execution_callback: callable
    ) -> Any:
        """
        Route a request and execute it on the selected node
        """
        request_id = request_data.get('request_id', f"req-{int(time.time())}")
        model_name = request_data.get('model', 'default')
        estimated_tokens = request_data.get('max_tokens', 100)
        user_id = request_data.get('user')
        
        # Route the request
        route = await self.load_balancer.route_request(
            request_id=request_id,
            model_name=model_name,
            estimated_tokens=estimated_tokens,
            user_id=user_id
        )
        
        if not route:
            raise RuntimeError(f"No available nodes for model {model_name}")
        
        try:
            # Store active route
            self.active_routes[request_id] = route
            
            # Execute on target node
            result = await execution_callback(route, request_data)
            
            # Record successful completion
            route.target_node.response_time = time.time() - route.created_at
            
            return result
            
        except Exception as e:
            logger.error(f"Request {request_id} failed on {route.target_node.node_id}: {e}")
            raise
        
        finally:
            # Cleanup
            self.load_balancer.release_request(request_id)
            if request_id in self.active_routes:
                del self.active_routes[request_id]
    
    def get_active_routes(self) -> Dict[str, RequestRoute]:
        """Get currently active routes"""
        return self.active_routes.copy()
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request"""
        if request_id in self.active_routes:
            self.load_balancer.release_request(request_id)
            del self.active_routes[request_id]
            return True
        return False

# Factory functions
def create_load_balancer(
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
    nodes: Optional[List[Dict[str, Any]]] = None
) -> LoadBalancer:
    """Create and configure a load balancer"""
    lb = LoadBalancer(strategy)
    
    if nodes:
        for node_config in nodes:
            node_info = NodeInfo(
                node_id=node_config['node_id'],
                host=node_config['host'],
                port=node_config['port'],
                weight=node_config.get('weight', 1.0),
                max_connections=node_config.get('max_connections', 100),
                capabilities=node_config.get('capabilities', []),
                active_models=node_config.get('active_models', [])
            )
            lb.add_node(node_info)
    
    return lb

def create_request_router(load_balancer: LoadBalancer) -> RequestRouter:
    """Create a request router with the given load balancer"""
    return RequestRouter(load_balancer)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        # Create load balancer with sample nodes
        nodes = [
            {
                'node_id': 'node-1',
                'host': '10.0.1.10',
                'port': 8000,
                'weight': 1.0,
                'active_models': ['llama-7b', 'mistral-7b']
            },
            {
                'node_id': 'node-2',
                'host': '10.0.1.11',
                'port': 8000,
                'weight': 1.5,
                'active_models': ['llama-7b', 'llama-13b']
            }
        ]
        
        lb = create_load_balancer(LoadBalancingStrategy.RESOURCE_AWARE, nodes)
        router = create_request_router(lb)
        
        # Simulate some requests
        async def mock_execution(route, request_data):
            await asyncio.sleep(0.1)  # Simulate processing
            return f"Processed on {route.target_node.node_id}"
        
        for i in range(10):
            request_data = {
                'request_id': f'test-{i}',
                'model': 'llama-7b',
                'max_tokens': 100
            }
            
            result = await router.route_and_execute(request_data, mock_execution)
            print(f"Request {i}: {result}")
        
        # Print cluster status
        status = lb.get_cluster_status()
        print(f"\nCluster Status: {json.dumps(status, indent=2)}")
    
    asyncio.run(example_usage())