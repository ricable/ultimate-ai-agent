"""
Network Performance Optimizer for MLX Distributed System
Implements Thunderbolt ring optimization, TCP tuning, compression, and async I/O
"""

import asyncio
import socket
import struct
import time
import zlib
import lz4.frame
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import logging
import subprocess
import psutil
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    latency_ms: float
    bandwidth_mbps: float
    packet_loss_rate: float
    compression_ratio: float
    throughput_ops_per_sec: float
    buffer_utilization: float

@dataclass
class NodeConnection:
    """Connection information for a cluster node"""
    node_id: str
    ip: str
    port: int
    socket: Optional[socket.socket] = None
    is_thunderbolt: bool = False
    metrics: Optional[NetworkMetrics] = None

class CompressionEngine:
    """High-performance compression for tensor transfers"""
    
    def __init__(self, method: str = "lz4"):
        self.method = method
        self.compression_stats = {"total_compressed": 0, "total_uncompressed": 0}
    
    def compress(self, data: bytes) -> bytes:
        """Compress data with specified method"""
        start_size = len(data)
        
        if self.method == "lz4":
            compressed = lz4.frame.compress(data, compression_level=1)  # Fast compression
        elif self.method == "zlib":
            compressed = zlib.compress(data, level=1)  # Fast compression
        else:
            compressed = data  # No compression
        
        # Update stats
        self.compression_stats["total_uncompressed"] += start_size
        self.compression_stats["total_compressed"] += len(compressed)
        
        return compressed
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        if self.method == "lz4":
            return lz4.frame.decompress(data)
        elif self.method == "zlib":
            return zlib.decompress(data)
        else:
            return data
    
    def get_compression_ratio(self) -> float:
        """Get overall compression ratio"""
        if self.compression_stats["total_uncompressed"] == 0:
            return 1.0
        return self.compression_stats["total_compressed"] / self.compression_stats["total_uncompressed"]

class ThunderboltOptimizer:
    """Thunderbolt-specific network optimizations"""
    
    def __init__(self):
        self.thunderbolt_interfaces = self._detect_thunderbolt_interfaces()
        self.ring_topology = self._setup_ring_topology()
    
    def _detect_thunderbolt_interfaces(self) -> List[str]:
        """Detect available Thunderbolt network interfaces"""
        try:
            # macOS-specific: get Thunderbolt interfaces
            result = subprocess.run(['networksetup', '-listallhardwareports'], 
                                  capture_output=True, text=True)
            
            interfaces = []
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'Thunderbolt' in line and i + 1 < len(lines):
                    device_line = lines[i + 1]
                    if 'Device:' in device_line:
                        interface = device_line.split()[-1]
                        interfaces.append(interface)
            
            return interfaces
        except Exception as e:
            logger.warning(f"Could not detect Thunderbolt interfaces: {e}")
            return []
    
    def _setup_ring_topology(self) -> Dict[str, str]:
        """Setup ring topology mapping for Thunderbolt"""
        if len(self.thunderbolt_interfaces) < 2:
            logger.warning("Insufficient Thunderbolt interfaces for ring topology")
            return {}
        
        # Create ring mapping: each node connects to next in ring
        ring_map = {}
        node_ids = ["mac-node-1", "mac-node-2", "mac-node-3"]
        
        for i, node_id in enumerate(node_ids):
            next_node = node_ids[(i + 1) % len(node_ids)]
            if i < len(self.thunderbolt_interfaces):
                ring_map[node_id] = {
                    "interface": self.thunderbolt_interfaces[i],
                    "next_node": next_node
                }
        
        return ring_map
    
    def optimize_interface(self, interface: str) -> bool:
        """Apply Thunderbolt-specific optimizations"""
        try:
            # Set maximum MTU for Thunderbolt (typically 1500 or higher)
            subprocess.run(['sudo', 'ifconfig', interface, 'mtu', '1500'], check=True)
            
            # Configure for low latency
            subprocess.run(['sudo', 'sysctl', '-w', f'net.inet.tcp.delayed_ack=0'], check=True)
            subprocess.run(['sudo', 'sysctl', '-w', f'net.inet.tcp.nodelay=1'], check=True)
            
            logger.info(f"Optimized Thunderbolt interface {interface}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to optimize interface {interface}: {e}")
            return False
    
    def measure_ring_latency(self) -> Dict[str, float]:
        """Measure latency around the Thunderbolt ring"""
        latencies = {}
        
        for node_id, config in self.ring_topology.items():
            try:
                # Ping next node in ring through specific interface
                interface = config["interface"]
                next_node = config["next_node"]
                
                # This would need to be implemented with actual node IPs
                # For now, return simulated latencies
                latencies[f"{node_id}->{next_node}"] = 2.5  # Target < 10ms
                
            except Exception as e:
                logger.error(f"Failed to measure latency for {node_id}: {e}")
                latencies[f"{node_id}->error"] = float('inf')
        
        return latencies

class TCPOptimizer:
    """TCP socket optimization for high-performance networking"""
    
    def __init__(self):
        self.optimal_settings = self._calculate_optimal_settings()
    
    def _calculate_optimal_settings(self) -> Dict[str, int]:
        """Calculate optimal TCP settings based on hardware"""
        # Get system memory to calculate buffer sizes
        total_memory = psutil.virtual_memory().total
        
        # Calculate buffer sizes (use 1% of total memory, capped at 256MB)
        buffer_size = min(int(total_memory * 0.01), 256 * 1024 * 1024)
        
        return {
            'SO_RCVBUF': buffer_size,
            'SO_SNDBUF': buffer_size,
            'TCP_NODELAY': 1,
            'SO_KEEPALIVE': 1,
            'SO_REUSEADDR': 1,
            'TCP_KEEPIDLE': 60,
            'TCP_KEEPINTVL': 10,
            'TCP_KEEPCNT': 6
        }
    
    def optimize_socket(self, sock: socket.socket) -> None:
        """Apply optimizations to a socket"""
        try:
            # Basic socket options
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.optimal_settings['SO_RCVBUF'])
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.optimal_settings['SO_SNDBUF'])
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, self.optimal_settings['SO_KEEPALIVE'])
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, self.optimal_settings['SO_REUSEADDR'])
            
            # TCP-specific options
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, self.optimal_settings['TCP_NODELAY'])
            
            # Keepalive settings (macOS/BSD style)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.optimal_settings['TCP_KEEPIDLE'])
            if hasattr(socket, 'TCP_KEEPINTVL'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.optimal_settings['TCP_KEEPINTVL'])
            if hasattr(socket, 'TCP_KEEPCNT'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.optimal_settings['TCP_KEEPCNT'])
            
            logger.debug(f"Applied TCP optimizations to socket")
            
        except Exception as e:
            logger.error(f"Failed to optimize socket: {e}")
    
    def create_optimized_socket(self, family: int = socket.AF_INET) -> socket.socket:
        """Create a new socket with optimizations applied"""
        sock = socket.socket(family, socket.SOCK_STREAM)
        self.optimize_socket(sock)
        return sock

class AsyncIOManager:
    """Asynchronous I/O manager for non-blocking operations"""
    
    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.send_queue = asyncio.Queue(maxsize=1000)
        self.receive_queue = asyncio.Queue(maxsize=1000)
        self.active_connections = {}
        self._running = False
    
    async def start(self):
        """Start the async I/O manager"""
        self._running = True
        asyncio.create_task(self._process_send_queue())
        asyncio.create_task(self._process_receive_queue())
        logger.info("AsyncIOManager started")
    
    async def stop(self):
        """Stop the async I/O manager"""
        self._running = False
        self.executor.shutdown(wait=True)
        logger.info("AsyncIOManager stopped")
    
    async def _process_send_queue(self):
        """Process outgoing data queue"""
        while self._running:
            try:
                data_item = await asyncio.wait_for(self.send_queue.get(), timeout=1.0)
                await self._send_data_async(data_item)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing send queue: {e}")
    
    async def _process_receive_queue(self):
        """Process incoming data queue"""
        while self._running:
            try:
                # This would be populated by receive callbacks
                await asyncio.sleep(0.1)  # Yield control
            except Exception as e:
                logger.error(f"Error processing receive queue: {e}")
    
    async def _send_data_async(self, data_item: Dict[str, Any]):
        """Send data asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._send_data_sync,
                data_item
            )
        except Exception as e:
            logger.error(f"Async send failed: {e}")
    
    def _send_data_sync(self, data_item: Dict[str, Any]):
        """Synchronous data send (runs in thread pool)"""
        try:
            node_id = data_item['node_id']
            data = data_item['data']
            
            if node_id in self.active_connections:
                connection = self.active_connections[node_id]
                connection.socket.send(data)
                
        except Exception as e:
            logger.error(f"Sync send failed: {e}")
    
    async def send_tensor_async(self, node_id: str, tensor_data: bytes) -> bool:
        """Send tensor data asynchronously"""
        try:
            data_item = {
                'node_id': node_id,
                'data': tensor_data,
                'timestamp': time.time()
            }
            await self.send_queue.put(data_item)
            return True
        except Exception as e:
            logger.error(f"Failed to queue tensor data: {e}")
            return False

class NetworkOptimizer:
    """Main network performance optimizer"""
    
    def __init__(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config
        self.nodes: Dict[str, NodeConnection] = {}
        
        # Initialize components
        self.tcp_optimizer = TCPOptimizer()
        self.thunderbolt_optimizer = ThunderboltOptimizer()
        self.compression_engine = CompressionEngine(method="lz4")
        self.async_io_manager = AsyncIOManager()
        
        # Performance metrics
        self.metrics = NetworkMetrics(0, 0, 0, 0, 0, 0)
        self._setup_nodes()
    
    def _setup_nodes(self):
        """Setup node connections"""
        node_configs = [
            {"id": "mac-node-1", "ip": "10.0.1.10", "port": 52415},
            {"id": "mac-node-2", "ip": "10.0.1.11", "port": 52415},
            {"id": "mac-node-3", "ip": "10.0.1.12", "port": 52415}
        ]
        
        for config in node_configs:
            self.nodes[config["id"]] = NodeConnection(
                node_id=config["id"],
                ip=config["ip"],
                port=config["port"],
                is_thunderbolt=(config["id"] in self.thunderbolt_optimizer.ring_topology)
            )
    
    async def initialize(self) -> bool:
        """Initialize the network optimizer"""
        try:
            # Start async I/O manager
            await self.async_io_manager.start()
            
            # Optimize Thunderbolt interfaces
            for interface in self.thunderbolt_optimizer.thunderbolt_interfaces:
                self.thunderbolt_optimizer.optimize_interface(interface)
            
            # Setup connections to all nodes
            await self._establish_connections()
            
            logger.info("Network optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize network optimizer: {e}")
            return False
    
    async def _establish_connections(self):
        """Establish optimized connections to all nodes"""
        connection_tasks = []
        
        for node_id, node in self.nodes.items():
            task = asyncio.create_task(self._connect_to_node(node))
            connection_tasks.append(task)
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = sum(1 for r in results if r is True)
        logger.info(f"Established {successful_connections}/{len(self.nodes)} connections")
    
    async def _connect_to_node(self, node: NodeConnection) -> bool:
        """Connect to a single node with optimizations"""
        try:
            # Create optimized socket
            sock = self.tcp_optimizer.create_optimized_socket()
            
            # Connect with timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, sock.connect, (node.ip, node.port)),
                timeout=10.0
            )
            
            node.socket = sock
            self.async_io_manager.active_connections[node.node_id] = node
            
            logger.info(f"Connected to {node.node_id} at {node.ip}:{node.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {node.node_id}: {e}")
            return False
    
    def send_compressed_tensor(self, node_id: str, tensor_data: np.ndarray) -> bool:
        """Send tensor data with compression"""
        try:
            # Serialize tensor
            tensor_bytes = tensor_data.tobytes()
            
            # Compress data
            compressed_data = self.compression_engine.compress(tensor_bytes)
            
            # Create header with metadata
            header = struct.pack('!III', 
                               len(compressed_data),  # Compressed size
                               len(tensor_bytes),      # Uncompressed size
                               tensor_data.dtype.itemsize)  # Data type size
            
            # Send header + compressed data
            full_message = header + compressed_data
            
            if node_id in self.nodes and self.nodes[node_id].socket:
                self.nodes[node_id].socket.send(full_message)
                return True
            else:
                logger.error(f"No connection to {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send compressed tensor to {node_id}: {e}")
            return False
    
    def receive_compressed_tensor(self, node_id: str) -> Optional[np.ndarray]:
        """Receive and decompress tensor data"""
        try:
            if node_id not in self.nodes or not self.nodes[node_id].socket:
                return None
            
            sock = self.nodes[node_id].socket
            
            # Receive header
            header_data = sock.recv(12)  # 3 * 4 bytes
            if len(header_data) != 12:
                return None
            
            compressed_size, uncompressed_size, dtype_size = struct.unpack('!III', header_data)
            
            # Receive compressed data
            compressed_data = b''
            while len(compressed_data) < compressed_size:
                chunk = sock.recv(compressed_size - len(compressed_data))
                if not chunk:
                    break
                compressed_data += chunk
            
            # Decompress
            decompressed_data = self.compression_engine.decompress(compressed_data)
            
            # Reconstruct tensor (assuming float32 for simplicity)
            tensor = np.frombuffer(decompressed_data, dtype=np.float32)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to receive compressed tensor from {node_id}: {e}")
            return None
    
    async def measure_network_performance(self) -> NetworkMetrics:
        """Measure comprehensive network performance"""
        try:
            latencies = []
            bandwidths = []
            
            # Measure latency to all nodes
            for node_id, node in self.nodes.items():
                if node.socket:
                    latency = await self._measure_latency(node)
                    latencies.append(latency)
                    
                    bandwidth = await self._measure_bandwidth(node)
                    bandwidths.append(bandwidth)
            
            # Calculate metrics
            avg_latency = sum(latencies) / len(latencies) if latencies else float('inf')
            avg_bandwidth = sum(bandwidths) / len(bandwidths) if bandwidths else 0
            compression_ratio = self.compression_engine.get_compression_ratio()
            
            self.metrics = NetworkMetrics(
                latency_ms=avg_latency,
                bandwidth_mbps=avg_bandwidth,
                packet_loss_rate=0.0,  # Would need packet-level monitoring
                compression_ratio=compression_ratio,
                throughput_ops_per_sec=0.0,  # Would be measured during actual operations
                buffer_utilization=0.0  # Would need buffer monitoring
            )
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Failed to measure network performance: {e}")
            return NetworkMetrics(float('inf'), 0, 1.0, 1.0, 0, 0)
    
    async def _measure_latency(self, node: NodeConnection) -> float:
        """Measure latency to a specific node"""
        try:
            # Send ping message
            ping_msg = b"PING:" + str(time.time()).encode()
            start_time = time.time()
            
            node.socket.send(ping_msg)
            
            # Wait for response (simplified - would need proper protocol)
            response = node.socket.recv(1024)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            return latency_ms
            
        except Exception as e:
            logger.error(f"Latency measurement failed for {node.node_id}: {e}")
            return float('inf')
    
    async def _measure_bandwidth(self, node: NodeConnection) -> float:
        """Measure bandwidth to a specific node"""
        try:
            # Send 1MB of data and measure time
            test_data = b'x' * (1024 * 1024)  # 1MB
            start_time = time.time()
            
            node.socket.send(test_data)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate bandwidth in Mbps
            bandwidth_mbps = (len(test_data) * 8) / (elapsed * 1_000_000)
            return bandwidth_mbps
            
        except Exception as e:
            logger.error(f"Bandwidth measurement failed for {node.node_id}: {e}")
            return 0.0
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "network_metrics": {
                "latency_ms": self.metrics.latency_ms,
                "bandwidth_mbps": self.metrics.bandwidth_mbps,
                "compression_ratio": self.metrics.compression_ratio
            },
            "thunderbolt_status": {
                "interfaces_detected": len(self.thunderbolt_optimizer.thunderbolt_interfaces),
                "ring_topology_active": bool(self.thunderbolt_optimizer.ring_topology)
            },
            "tcp_optimization": {
                "buffer_sizes": {
                    "receive": self.tcp_optimizer.optimal_settings['SO_RCVBUF'],
                    "send": self.tcp_optimizer.optimal_settings['SO_SNDBUF']
                },
                "nodelay_enabled": bool(self.tcp_optimizer.optimal_settings['TCP_NODELAY'])
            },
            "compression_stats": {
                "method": self.compression_engine.method,
                "total_compressed": self.compression_engine.compression_stats["total_compressed"],
                "total_uncompressed": self.compression_engine.compression_stats["total_uncompressed"],
                "ratio": self.compression_engine.get_compression_ratio()
            },
            "async_io": {
                "max_workers": self.async_io_manager.max_workers,
                "active_connections": len(self.async_io_manager.active_connections)
            }
        }
    
    async def shutdown(self):
        """Shutdown the network optimizer"""
        try:
            # Stop async I/O manager
            await self.async_io_manager.stop()
            
            # Close all connections
            for node in self.nodes.values():
                if node.socket:
                    node.socket.close()
            
            logger.info("Network optimizer shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage and testing
async def main():
    """Example usage of NetworkOptimizer"""
    cluster_config = {"nodes": 4}
    optimizer = NetworkOptimizer(cluster_config)
    
    # Initialize
    success = await optimizer.initialize()
    if not success:
        logger.error("Failed to initialize network optimizer")
        return
    
    # Measure performance
    metrics = await optimizer.measure_network_performance()
    logger.info(f"Network metrics: {metrics}")
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    logger.info(f"Optimization report: {report}")
    
    # Shutdown
    await optimizer.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())