"""
Enhanced Exo P2P Cluster Manager with MLX Integration
Provides seamless integration between Exo P2P framework and MLX distributed operations
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import socket
import hashlib

# Try to import MLX distributed components
try:
    from ..mlx_distributed.cluster import DistributedMLXCluster
    from ..mlx_distributed.config import MLXDistributedConfig, NodeConfig
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX distributed components not available")

# Import base Exo cluster manager
from .cluster_manager import ExoClusterManager, ExoNodeSpec, EXO_AVAILABLE

logger = logging.getLogger(__name__)

@dataclass
class ModelPartition:
    """Represents a model partition assignment"""
    node_id: str
    start_layer: int
    end_layer: int
    memory_gb: float
    shard_path: str
    checksum: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'start_layer': self.start_layer,
            'end_layer': self.end_layer,
            'memory_gb': self.memory_gb,
            'shard_path': self.shard_path,
            'checksum': self.checksum
        }

@dataclass
class ModelConfig:
    """Enhanced model configuration for distributed inference"""
    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    max_sequence_length: int
    model_size_gb: float
    quantization: str = "none"
    partitions: List[ModelPartition] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'architecture': self.architecture,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'model_size_gb': self.model_size_gb,
            'quantization': self.quantization,
            'partitions': [p.to_dict() for p in self.partitions]
        }

class EnhancedExoClusterManager:
    """
    Enhanced Exo cluster manager with MLX integration for distributed inference
    Combines Exo's P2P capabilities with MLX's distributed training/inference
    """
    
    def __init__(self, node_spec: ExoNodeSpec, mlx_config_file: Optional[str] = None):
        # Initialize base Exo manager
        self.exo_manager = ExoClusterManager(node_spec)
        self.node_spec = node_spec
        
        # Initialize MLX distributed cluster if available
        self.mlx_cluster = None
        if MLX_AVAILABLE:
            self.mlx_cluster = DistributedMLXCluster(mlx_config_file)
            logger.info("MLX distributed cluster initialized")
        else:
            logger.warning("MLX distributed cluster not available")
        
        # Enhanced state management
        self.loaded_models: Dict[str, ModelConfig] = {}
        self.partition_assignments: Dict[str, Dict[str, ModelPartition]] = {}
        self.peer_capabilities: Dict[str, Dict[str, Any]] = {}
        self.active_inferences: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.metrics = {
            'total_inferences': 0,
            'avg_inference_time': 0.0,
            'total_memory_usage': 0.0,
            'peer_communication_time': 0.0,
            'model_loading_time': 0.0
        }
        
        # Event callbacks
        self.on_peer_join: Optional[Callable] = None
        self.on_peer_leave: Optional[Callable] = None
        self.on_model_loaded: Optional[Callable] = None
        
        logger.info(f"Enhanced Exo cluster manager initialized for {node_spec.node_id}")
    
    async def initialize_hybrid_cluster(self) -> bool:
        """Initialize both Exo P2P and MLX distributed components"""
        logger.info("Initializing hybrid Exo-MLX cluster")
        
        try:
            # Initialize Exo components
            exo_success = await self.exo_manager.start_cluster()
            if not exo_success:
                logger.error("Failed to initialize Exo cluster")
                return False
            
            # Initialize MLX distributed if available
            mlx_success = True
            if self.mlx_cluster:
                mlx_success = await self.mlx_cluster.initialize()
                if not mlx_success:
                    logger.warning("MLX distributed initialization failed, continuing with Exo only")
            
            # Sync cluster state between Exo and MLX
            await self._sync_cluster_state()
            
            logger.info("Hybrid cluster initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Hybrid cluster initialization failed: {e}")
            return False
    
    async def _sync_cluster_state(self) -> None:
        """Synchronize cluster state between Exo and MLX components"""
        try:
            # Get Exo peer information
            exo_peers = self.exo_manager.discovered_peers
            
            # Update peer capabilities
            for peer in exo_peers:
                peer_id = f"peer-{peer.get('ip', 'unknown')}"
                self.peer_capabilities[peer_id] = {
                    'ip': peer.get('ip'),
                    'port': peer.get('port'),
                    'memory_gb': peer.get('memory_gb', 32),
                    'device_type': peer.get('device_type', 'Unknown'),
                    'status': peer.get('status', 'unknown'),
                    'last_seen': peer.get('last_seen', time.time()),
                    'exo_available': True,
                    'mlx_available': MLX_AVAILABLE
                }
            
            logger.debug(f"Synced {len(self.peer_capabilities)} peer capabilities")
            
        except Exception as e:
            logger.error(f"Failed to sync cluster state: {e}")
    
    async def discover_and_coordinate_peers(self, timeout: int = 60) -> List[Dict[str, Any]]:
        """Enhanced peer discovery with capability negotiation"""
        logger.info("Starting enhanced peer discovery with capability negotiation")
        
        # Use base Exo discovery
        exo_peers = await self.exo_manager.discover_peers(timeout)
        
        # Enhance with capability negotiation
        enhanced_peers = []
        for peer in exo_peers:
            peer_ip = peer.get('ip')
            if peer_ip:
                capabilities = await self._negotiate_peer_capabilities(peer_ip)
                enhanced_peer = {**peer, **capabilities}
                enhanced_peers.append(enhanced_peer)
                
                # Trigger peer join callback
                if self.on_peer_join:
                    await self.on_peer_join(enhanced_peer)
        
        logger.info(f"Enhanced peer discovery complete: {len(enhanced_peers)} peers")
        return enhanced_peers
    
    async def _negotiate_peer_capabilities(self, peer_ip: str) -> Dict[str, Any]:
        """Negotiate capabilities with a peer node"""
        capabilities = {
            'mlx_available': False,
            'exo_version': 'unknown',
            'supported_models': [],
            'max_memory_gb': 32,
            'device_capabilities': {}
        }
        
        try:
            # In a real implementation, this would query the peer's capabilities
            # For now, we'll simulate capability negotiation
            capabilities['mlx_available'] = MLX_AVAILABLE
            capabilities['max_memory_gb'] = 64  # Default assumption
            capabilities['supported_models'] = ['llama-7b', 'llama-13b', 'mistral-7b']
            
            logger.debug(f"Negotiated capabilities with {peer_ip}: {capabilities}")
            
        except Exception as e:
            logger.debug(f"Capability negotiation with {peer_ip} failed: {e}")
        
        return capabilities
    
    async def load_model_with_smart_partitioning(self, model_config: ModelConfig) -> bool:
        """Load model with intelligent partitioning across hybrid cluster"""
        logger.info(f"Loading model {model_config.name} with smart partitioning")
        
        try:
            # Step 1: Analyze cluster resources
            cluster_resources = await self._analyze_cluster_resources()
            
            # Step 2: Create optimal partitioning strategy
            partitions = await self._create_optimal_partitions(model_config, cluster_resources)
            
            if not partitions:
                logger.error("Failed to create partitioning plan")
                return False
            
            # Step 3: Distribute model parts to nodes
            success = await self._distribute_model_parts(model_config, partitions)
            
            if success:
                # Step 4: Store model configuration
                model_config.partitions = partitions
                self.loaded_models[model_config.name] = model_config
                self.partition_assignments[model_config.name] = {
                    p.node_id: p for p in partitions
                }
                
                # Trigger model loaded callback
                if self.on_model_loaded:
                    await self.on_model_loaded(model_config)
                
                logger.info(f"Model {model_config.name} loaded successfully")
                return True
            else:
                logger.error(f"Failed to distribute model parts for {model_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    async def _analyze_cluster_resources(self) -> Dict[str, Any]:
        """Analyze available cluster resources for optimization"""
        resources = {
            'total_memory_gb': self.node_spec.memory_gb,
            'total_compute_power': self.node_spec.compute_capability,
            'node_count': 1,
            'nodes': [
                {
                    'id': self.node_spec.node_id,
                    'memory_gb': self.node_spec.memory_gb,
                    'compute_power': self.node_spec.compute_capability,
                    'device_type': self.node_spec.device_type
                }
            ]
        }
        
        # Add peer resources
        for peer_id, capabilities in self.peer_capabilities.items():
            resources['total_memory_gb'] += capabilities.get('memory_gb', 32)
            resources['total_compute_power'] += capabilities.get('compute_power', 0.8)
            resources['node_count'] += 1
            resources['nodes'].append({
                'id': peer_id,
                'memory_gb': capabilities.get('memory_gb', 32),
                'compute_power': capabilities.get('compute_power', 0.8),
                'device_type': capabilities.get('device_type', 'Unknown')
            })
        
        logger.debug(f"Cluster resources: {resources['node_count']} nodes, "
                    f"{resources['total_memory_gb']}GB total memory")
        
        return resources
    
    async def _create_optimal_partitions(self, model_config: ModelConfig, 
                                       cluster_resources: Dict[str, Any]) -> List[ModelPartition]:
        """Create optimal model partitions based on cluster resources"""
        partitions = []
        
        try:
            nodes = cluster_resources['nodes']
            total_layers = model_config.num_layers
            
            # Sort nodes by memory capacity (descending)
            sorted_nodes = sorted(nodes, key=lambda x: x['memory_gb'], reverse=True)
            
            # Calculate layers per node based on memory
            remaining_layers = total_layers
            layer_start = 0
            
            for node in sorted_nodes:
                if remaining_layers <= 0:
                    break
                
                # Calculate how many layers this node should handle
                memory_ratio = node['memory_gb'] / cluster_resources['total_memory_gb']
                layers_for_node = max(1, int(total_layers * memory_ratio))
                layers_for_node = min(layers_for_node, remaining_layers)
                
                # Calculate memory requirement
                memory_required = (layers_for_node / total_layers) * model_config.model_size_gb
                
                partition = ModelPartition(
                    node_id=node['id'],
                    start_layer=layer_start,
                    end_layer=layer_start + layers_for_node - 1,
                    memory_gb=memory_required,
                    shard_path=f"models/{model_config.name}/shard_{node['id']}_{layer_start}_{layer_start + layers_for_node - 1}",
                    checksum=""
                )
                
                partitions.append(partition)
                layer_start += layers_for_node
                remaining_layers -= layers_for_node
            
            # Handle any remaining layers
            if remaining_layers > 0 and partitions:
                # Add remaining layers to the last partition
                partitions[-1].end_layer += remaining_layers
                partitions[-1].memory_gb += (remaining_layers / total_layers) * model_config.model_size_gb
            
            logger.info(f"Created {len(partitions)} partitions for {model_config.name}")
            for i, partition in enumerate(partitions):
                logger.debug(f"  Partition {i}: {partition.node_id} layers {partition.start_layer}-{partition.end_layer} "
                           f"({partition.memory_gb:.2f}GB)")
            
            return partitions
            
        except Exception as e:
            logger.error(f"Failed to create optimal partitions: {e}")
            return []
    
    async def _distribute_model_parts(self, model_config: ModelConfig, 
                                    partitions: List[ModelPartition]) -> bool:
        """Distribute model parts to respective nodes"""
        logger.info(f"Distributing {len(partitions)} model parts")
        
        try:
            # Find our partition
            my_partition = None
            for partition in partitions:
                if partition.node_id == self.node_spec.node_id:
                    my_partition = partition
                    break
            
            if my_partition:
                # Load our part using MLX if available
                if self.mlx_cluster:
                    mlx_model_config = {
                        'name': model_config.name,
                        'num_layers': model_config.num_layers,
                        'size_gb': model_config.model_size_gb,
                        'architecture': model_config.architecture
                    }
                    
                    success = await self.mlx_cluster.load_model_distributed(
                        my_partition.shard_path, mlx_model_config
                    )
                    
                    if not success:
                        logger.error(f"Failed to load model part via MLX")
                        return False
                else:
                    # Load via Exo
                    success = await self.exo_manager.load_model(
                        model_config.name, my_partition.shard_path
                    )
                    
                    if not success:
                        logger.error(f"Failed to load model part via Exo")
                        return False
                
                logger.info(f"Successfully loaded partition {my_partition.start_layer}-{my_partition.end_layer}")
            
            # Coordinate with peers for their parts
            # In a real implementation, this would involve network communication
            await asyncio.sleep(1)  # Simulate coordination time
            
            return True
            
        except Exception as e:
            logger.error(f"Model part distribution failed: {e}")
            return False
    
    async def distributed_inference(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform distributed inference with hybrid Exo-MLX coordination"""
        inference_id = hashlib.md5(f"{model_name}_{time.time()}".encode()).hexdigest()[:8]
        start_time = time.time()
        
        logger.info(f"Starting distributed inference {inference_id} for model {model_name}")
        
        try:
            if model_name not in self.loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            model_config = self.loaded_models[model_name]
            partitions = self.partition_assignments[model_name]
            
            # Track active inference
            self.active_inferences[inference_id] = {
                'model': model_name,
                'start_time': start_time,
                'status': 'running',
                'input_data': input_data
            }
            
            # Get our partition
            my_partition = partitions.get(self.node_spec.node_id)
            if not my_partition:
                raise ValueError(f"No partition assigned to node {self.node_spec.node_id}")
            
            # Perform inference on our layers
            if self.mlx_cluster:
                # Use MLX distributed inference
                layer_result = await self.mlx_cluster.inference_distributed(
                    model_name, input_data
                )
            else:
                # Use Exo inference
                layer_result = await self._exo_inference(model_name, input_data, my_partition)
            
            # Coordinate with other nodes for complete inference
            complete_result = await self._coordinate_inference_results(
                inference_id, model_name, layer_result, partitions
            )
            
            # Update metrics
            inference_time = time.time() - start_time
            self.metrics['total_inferences'] += 1
            self.metrics['avg_inference_time'] = (
                (self.metrics['avg_inference_time'] * (self.metrics['total_inferences'] - 1) + inference_time) 
                / self.metrics['total_inferences']
            )
            
            # Clean up active inference
            if inference_id in self.active_inferences:
                self.active_inferences[inference_id]['status'] = 'completed'
                self.active_inferences[inference_id]['end_time'] = time.time()
            
            logger.info(f"Distributed inference {inference_id} completed in {inference_time:.3f}s")
            return complete_result
            
        except Exception as e:
            logger.error(f"Distributed inference {inference_id} failed: {e}")
            if inference_id in self.active_inferences:
                self.active_inferences[inference_id]['status'] = 'failed'
                self.active_inferences[inference_id]['error'] = str(e)
            
            return {'error': str(e), 'inference_id': inference_id}
    
    async def _exo_inference(self, model_name: str, input_data: Dict[str, Any], 
                           partition: ModelPartition) -> Dict[str, Any]:
        """Perform inference using Exo backend"""
        logger.debug(f"Performing Exo inference for layers {partition.start_layer}-{partition.end_layer}")
        
        # Simulate Exo inference processing
        processing_time = (partition.end_layer - partition.start_layer + 1) * 0.015  # 15ms per layer
        await asyncio.sleep(processing_time)
        
        return {
            'node_id': partition.node_id,
            'layers': f"{partition.start_layer}-{partition.end_layer}",
            'processing_time': processing_time,
            'output_shape': [1, input_data.get('max_tokens', 100), 4096],  # Mock output
            'backend': 'exo'
        }
    
    async def _coordinate_inference_results(self, inference_id: str, model_name: str,
                                          our_result: Dict[str, Any], 
                                          partitions: Dict[str, ModelPartition]) -> Dict[str, Any]:
        """Coordinate inference results from all nodes"""
        logger.debug(f"Coordinating inference results for {inference_id}")
        
        # In a real implementation, this would involve:
        # 1. Collecting results from all peer nodes
        # 2. Aggregating intermediate activations
        # 3. Producing final output
        
        # For now, simulate coordination
        coordination_time = len(partitions) * 0.005  # 5ms per node
        await asyncio.sleep(coordination_time)
        
        # Mock final result
        final_result = {
            'inference_id': inference_id,
            'model': model_name,
            'output': f"Generated response for inference {inference_id}",
            'tokens_generated': 150,
            'total_processing_time': our_result.get('processing_time', 0) + coordination_time,
            'coordination_time': coordination_time,
            'participating_nodes': list(partitions.keys()),
            'backend': 'hybrid_exo_mlx'
        }
        
        return final_result
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status including both Exo and MLX components"""
        exo_health = await self.exo_manager.health_check()
        mlx_status = self.mlx_cluster.get_cluster_status() if self.mlx_cluster else {}
        
        return {
            'node_id': self.node_spec.node_id,
            'hybrid_cluster': {
                'exo_available': EXO_AVAILABLE,
                'mlx_available': MLX_AVAILABLE,
                'initialized': self.exo_manager.cluster_ready
            },
            'exo_status': exo_health,
            'mlx_status': mlx_status,
            'loaded_models': list(self.loaded_models.keys()),
            'active_inferences': len(self.active_inferences),
            'peer_capabilities': self.peer_capabilities,
            'performance_metrics': self.metrics,
            'partition_assignments': {
                model: {node: partition.to_dict() for node, partition in partitions.items()}
                for model, partitions in self.partition_assignments.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up resources and shutdown cluster"""
        logger.info("Cleaning up enhanced cluster manager")
        
        # Clear active inferences
        self.active_inferences.clear()
        
        # Shutdown MLX cluster if available
        if self.mlx_cluster:
            await self.mlx_cluster.shutdown()
        
        # Shutdown Exo cluster
        await self.exo_manager.stop_cluster()
        
        logger.info("Enhanced cluster manager cleanup complete")

# Factory function for creating enhanced cluster managers
def create_enhanced_cluster_manager(node_id: str, mlx_config_file: Optional[str] = None) -> EnhancedExoClusterManager:
    """Create enhanced cluster manager for specified node"""
    
    node_configs = {
        "mac-node-1": ExoNodeSpec("mac-node-1", "10.0.1.10", 52415, 64, 1.0, "M1_Max"),
        "mac-node-2": ExoNodeSpec("mac-node-2", "10.0.1.11", 52415, 64, 1.0, "M1_Max"),
        "mac-node-3": ExoNodeSpec("mac-node-3", "10.0.1.12", 52415, 32, 0.8, "M2_Max")
    }
    
    if node_id not in node_configs:
        raise ValueError(f"Unknown node_id: {node_id}. Available: {list(node_configs.keys())}")
    
    return EnhancedExoClusterManager(node_configs[node_id], mlx_config_file)

# Example usage and testing
async def main():
    """Test enhanced cluster manager functionality"""
    import sys
    
    # Auto-detect or use provided node ID
    node_id = sys.argv[1] if len(sys.argv) > 1 else "mac-node-1"
    
    try:
        manager = create_enhanced_cluster_manager(node_id)
        
        # Initialize hybrid cluster
        if await manager.initialize_hybrid_cluster():
            logger.info("✓ Hybrid cluster initialized successfully")
        else:
            logger.error("✗ Hybrid cluster initialization failed")
            return
        
        # Discover peers
        peers = await manager.discover_and_coordinate_peers()
        logger.info(f"Discovered {len(peers)} peers with capabilities")
        
        # Test model loading
        model_config = ModelConfig(
            name="llama-7b",
            architecture="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=50000,
            max_sequence_length=2048,
            model_size_gb=7.0,
            quantization="fp16"
        )
        
        if await manager.load_model_with_smart_partitioning(model_config):
            logger.info("✓ Model loaded with smart partitioning")
        else:
            logger.error("✗ Model loading failed")
            return
        
        # Test distributed inference
        input_data = {
            'prompt': 'Hello, how are you?',
            'max_tokens': 100,
            'temperature': 0.7
        }
        
        result = await manager.distributed_inference('llama-7b', input_data)
        logger.info(f"Inference result: {result}")
        
        # Show comprehensive status
        status = await manager.get_comprehensive_status()
        logger.info(f"Comprehensive status: {json.dumps(status, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'manager' in locals():
            await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())