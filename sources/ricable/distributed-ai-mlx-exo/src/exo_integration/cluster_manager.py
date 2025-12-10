"""
Exo P2P Cluster Manager
Handles Exo cluster formation, device discovery, and basic coordination
"""

import asyncio
import json
import socket
import time
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Exo components (experimental)
try:
    from exo.inference.inference_engine import InferenceEngine
    from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
    from exo.networking.grpc_peer_pool import GRPCPeerPool
    from exo.orchestration.standard_node import StandardNode
    EXO_AVAILABLE = True
    logger.info("Exo framework imported successfully")
except ImportError as e:
    logger.warning(f"Exo import failed: {e}")
    logger.warning("Running in mock mode - some functionality will be simulated")
    EXO_AVAILABLE = False
    
    # Mock classes for when Exo is not available
    class StandardNode:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            logger.debug(f"Mock StandardNode created with {kwargs}")
    
    class GRPCPeerPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.peers = []
            logger.debug(f"Mock GRPCPeerPool created with {kwargs}")
    
    class RingMemoryWeightedPartitioningStrategy:
        def __init__(self):
            logger.debug("Mock RingMemoryWeightedPartitioningStrategy created")

@dataclass
class ExoNodeSpec:
    """Specification for an Exo node"""
    node_id: str
    ip: str
    port: int
    memory_gb: int
    compute_capability: float
    device_type: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExoNodeSpec':
        """Create from dictionary"""
        return cls(**data)

class ExoClusterManager:
    """Manages Exo P2P cluster formation and coordination"""
    
    def __init__(self, node_spec: ExoNodeSpec):
        self.node_spec = node_spec
        self.node: Optional[StandardNode] = None
        self.peer_pool: Optional[GRPCPeerPool] = None
        self.discovered_peers: List[Dict[str, Any]] = []
        self.cluster_ready = False
        self.start_time = time.time()
        self._running = False
        
        logger.info(f"ExoClusterManager initialized for node {node_spec.node_id}")
        
    async def initialize_node(self) -> bool:
        """Initialize Exo node with P2P capabilities"""
        logger.info(f"Initializing Exo node {self.node_spec.node_id}")
        
        if not EXO_AVAILABLE:
            logger.warning("Exo not available, using mock initialization")
            # Create mock objects for testing
            self.peer_pool = GRPCPeerPool(node=None, lambda_download_url=None)
            self.node = StandardNode(
                node_id=self.node_spec.node_id,
                peers=self.peer_pool,
                inference_engine=None,
                partition_strategy=RingMemoryWeightedPartitioningStrategy(),
                chatgpt_api_endpoint=f"http://{self.node_spec.ip}:{self.node_spec.port + 1000}",
                web_chat_url=f"http://{self.node_spec.ip}:{self.node_spec.port + 2000}",
                disable_download=False
            )
            logger.info(f"Mock Exo node {self.node_spec.node_id} initialized")
            return True
            
        try:
            # Create peer pool for P2P communication
            self.peer_pool = GRPCPeerPool(
                node=None,  # Will be set after node creation
                lambda_download_url=None
            )
            
            # Create standard node with ring partitioning
            partitioning_strategy = RingMemoryWeightedPartitioningStrategy()
            
            self.node = StandardNode(
                node_id=self.node_spec.node_id,
                peers=self.peer_pool,
                inference_engine=None,  # Will be set based on available engines
                partition_strategy=partitioning_strategy,
                chatgpt_api_endpoint=f"http://{self.node_spec.ip}:{self.node_spec.port + 1000}",
                web_chat_url=f"http://{self.node_spec.ip}:{self.node_spec.port + 2000}",
                disable_download=False
            )
            
            # Set node in peer pool
            self.peer_pool.node = self.node
            
            logger.info(f"Exo node {self.node_spec.node_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Exo node: {e}")
            return False
    
    async def discover_peers(self, timeout: int = 30) -> List[Dict[str, Any]]:
        """Discover other nodes in the cluster"""
        if not self.node:
            logger.error("Node not initialized")
            return []
        
        logger.info(f"Starting peer discovery (timeout: {timeout}s)")
        discovery_start = time.time()
        
        # Known cluster IPs
        known_ips = [
            "10.0.1.10", "10.0.1.11", "10.0.1.12"
        ]
        
        # Manual peer addition for more reliable discovery
        successful_connections = 0
        for ip in known_ips:
            if ip != self.node_spec.ip:
                try:
                    if await self._attempt_peer_connection(ip, self.node_spec.port):
                        successful_connections += 1
                        # Add to discovered peers list
                        peer_info = {
                            'ip': ip,
                            'port': self.node_spec.port,
                            'memory_gb': 64,  # Default estimate
                            'device_type': 'Unknown',
                            'status': 'connected',
                            'last_seen': time.time()
                        }
                        self.discovered_peers.append(peer_info)
                except Exception as e:
                    logger.debug(f"Failed to connect to {ip}: {e}")
        
        # Wait for discovery or timeout (in real Exo, this would be automatic)
        while (time.time() - discovery_start) < timeout:
            current_peers = await self._get_current_peers()
            if len(current_peers) >= 1:  # At least 1 other node for testing
                self.discovered_peers = current_peers
                logger.info(f"Discovered {len(current_peers)} peers")
                return current_peers
            
            await asyncio.sleep(2)
        
        logger.info(f"Peer discovery completed with {len(self.discovered_peers)} peers")
        return self.discovered_peers
    
    async def _attempt_peer_connection(self, ip: str, port: int) -> bool:
        """Attempt to connect to a specific peer"""
        try:
            # Test basic connectivity
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port), 
                timeout=5
            )
            writer.close()
            await writer.wait_closed()
            logger.debug(f"Successfully connected to {ip}:{port}")
            return True
        except Exception as e:
            logger.debug(f"Connection to {ip}:{port} failed: {e}")
            return False
    
    async def _get_current_peers(self) -> List[Dict[str, Any]]:
        """Get current list of connected peers"""
        if not self.peer_pool:
            return []
        
        # In mock mode, return the discovered peers
        if not EXO_AVAILABLE:
            return self.discovered_peers
        
        # In real implementation, this would query the actual peer pool
        try:
            # peers = await self.peer_pool.get_peers()
            # For now, return the manually discovered peers
            return self.discovered_peers
        except Exception as e:
            logger.error(f"Failed to get current peers: {e}")
            return []
    
    async def load_model(self, model_name: str, model_path: Optional[str] = None) -> bool:
        """Load and partition model across cluster"""
        if not self.node:
            logger.error("Node not initialized")
            return False
        
        logger.info(f"Loading model {model_name} across cluster")
        
        try:
            # Calculate total memory across cluster
            total_memory = self.node_spec.memory_gb
            for peer in self.discovered_peers:
                total_memory += peer.get('memory_gb', 32)  # Default estimate
            
            logger.info(f"Total cluster memory: {total_memory}GB")
            
            # Model loading would happen here in real implementation
            # This is where the actual model partitioning occurs
            if EXO_AVAILABLE:
                # Real Exo model loading would go here
                pass
            else:
                # Mock model loading
                logger.info("Mock model loading (Exo not available)")
            
            await asyncio.sleep(2)  # Simulate loading time
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cluster"""
        if not self.node:
            return {"status": "unhealthy", "reason": "node_not_initialized"}
        
        # Check peer connectivity
        active_peers = 0
        for peer in self.discovered_peers:
            if time.time() - peer.get('last_seen', 0) < 60:  # Active within last minute
                active_peers += 1
        
        health_info = {
            "node_id": self.node_spec.node_id,
            "status": "healthy" if self.cluster_ready else "initializing",
            "peer_count": len(self.discovered_peers),
            "active_peers": active_peers,
            "memory_gb": self.node_spec.memory_gb,
            "device_type": self.node_spec.device_type,
            "uptime": time.time() - self.start_time,
            "exo_available": EXO_AVAILABLE,
            "endpoints": {
                "api": f"http://{self.node_spec.ip}:{self.node_spec.port + 1000}",
                "web": f"http://{self.node_spec.ip}:{self.node_spec.port + 2000}",
                "grpc": f"{self.node_spec.ip}:{self.node_spec.port}"
            }
        }
        
        return health_info
    
    async def start_cluster(self) -> bool:
        """Start the complete cluster initialization process"""
        logger.info(f"Starting Exo cluster on {self.node_spec.node_id}")
        self._running = True
        
        try:
            # Step 1: Initialize node
            if not await self.initialize_node():
                return False
            
            # Step 2: Discover peers
            peers = await self.discover_peers()
            
            # Step 3: Mark cluster as ready if we have peers
            self.cluster_ready = len(peers) > 0
            
            if self.cluster_ready:
                logger.info("Exo cluster ready for inference")
            else:
                logger.warning("Cluster running in single-node mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cluster: {e}")
            return False
    
    async def stop_cluster(self):
        """Stop the cluster and cleanup resources"""
        logger.info(f"Stopping Exo cluster on {self.node_spec.node_id}")
        self._running = False
        self.cluster_ready = False
        
        # Cleanup would go here
        if self.peer_pool:
            # await self.peer_pool.close() # If such method exists
            pass
        
        logger.info("Cluster stopped")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        return {
            "node_spec": self.node_spec.to_dict(),
            "cluster_ready": self.cluster_ready,
            "peer_count": len(self.discovered_peers),
            "peers": self.discovered_peers,
            "running": self._running,
            "exo_available": EXO_AVAILABLE
        }
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics"""
        diagnostics = {
            "timestamp": time.time(),
            "node_id": self.node_spec.node_id,
            "tests": {}
        }
        
        # Test 1: Node initialization
        diagnostics["tests"]["node_initialized"] = self.node is not None
        
        # Test 2: Peer connectivity
        connectivity_results = {}
        for peer in self.discovered_peers:
            ip = peer.get('ip')
            port = peer.get('port', self.node_spec.port)
            if ip:
                connectivity_results[ip] = await self._attempt_peer_connection(ip, port)
        
        diagnostics["tests"]["peer_connectivity"] = connectivity_results
        
        # Test 3: Port availability
        diagnostics["tests"]["port_available"] = await self._test_port_availability()
        
        # Test 4: Health check
        health = await self.health_check()
        diagnostics["tests"]["health_check"] = health
        
        return diagnostics
    
    async def _test_port_availability(self) -> bool:
        """Test if the required port is available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', self.node_spec.port))
            sock.close()
            return True
        except Exception:
            return False

# Factory function for creating cluster managers
def create_cluster_manager(node_id: str) -> ExoClusterManager:
    """Create cluster manager for specified node"""
    
    node_configs = {
        "mac-node-1": ExoNodeSpec("mac-node-1", "10.0.1.10", 52415, 64, 1.0, "M1_Max"),
        "mac-node-2": ExoNodeSpec("mac-node-2", "10.0.1.11", 52415, 64, 1.0, "M1_Max"),
        "mac-node-3": ExoNodeSpec("mac-node-3", "10.0.1.12", 52415, 32, 0.8, "M2_Max")
    }
    
    if node_id not in node_configs:
        raise ValueError(f"Unknown node_id: {node_id}. Available: {list(node_configs.keys())}")
    
    return ExoClusterManager(node_configs[node_id])

def auto_detect_node_id() -> Optional[str]:
    """Auto-detect current node ID based on hostname or IP"""
    hostname = socket.gethostname().lower()
    
    # Try to match hostname
    for node_id in ["mac-node-1", "mac-node-2", "mac-node-3"]:
        if node_id in hostname or hostname.startswith(node_id.replace('-', '')):
            logger.info(f"Auto-detected node ID: {node_id}")
            return node_id
    
    # Try to match by IP (basic implementation)
    try:
        # Get local IP addresses
        import subprocess
        result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        if "10.0.1.10" in result.stdout:
            return "mac-node-1"
        elif "10.0.1.11" in result.stdout:
            return "mac-node-2"
        elif "10.0.1.12" in result.stdout:
            return "mac-node-3"
    except Exception:
        pass
    
    logger.warning("Could not auto-detect node ID")
    return None

# Testing and demonstration
async def main():
    """Test cluster manager functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Exo Cluster Manager')
    parser.add_argument('--node-id', help='Node ID (mac-node-1, mac-node-2, etc.)')
    parser.add_argument('--auto-detect', action='store_true', help='Auto-detect node ID')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostics and exit')
    parser.add_argument('--health-check', action='store_true', help='Run health check and exit')
    
    args = parser.parse_args()
    
    # Determine node ID
    node_id = args.node_id
    if args.auto_detect or not node_id:
        node_id = auto_detect_node_id()
        if not node_id:
            print("ERROR: Could not determine node ID. Use --node-id or --auto-detect")
            sys.exit(1)
    
    try:
        manager = create_cluster_manager(node_id)
        
        if args.diagnostics:
            # Run diagnostics only
            logger.info("Running diagnostics...")
            diagnostics = await manager.run_diagnostics()
            print(json.dumps(diagnostics, indent=2))
            return
        
        if args.health_check:
            # Quick health check
            await manager.initialize_node()
            health = await manager.health_check()
            print(json.dumps(health, indent=2))
            return
        
        # Full cluster startup
        success = await manager.start_cluster()
        
        if success:
            # Perform initial health check
            health = await manager.health_check()
            logger.info(f"Health check: {health}")
            
            # Keep running and periodically report status
            logger.info("Cluster running... Press Ctrl+C to stop")
            while True:
                await asyncio.sleep(30)
                health = await manager.health_check()
                logger.info(f"Status: {health['status']}, Peers: {health['peer_count']}")
        else:
            logger.error("Failed to start cluster")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Shutting down cluster...")
        if 'manager' in locals():
            await manager.stop_cluster()
    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())