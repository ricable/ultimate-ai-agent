"""
MLX Distributed Cluster Implementation
Main class for managing distributed MLX operations across Apple Silicon cluster
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import mlx.core as mx
    import mlx.distributed as dist
    from mlx.utils import tree_map
except ImportError as e:
    logging.error(f"MLX import failed: {e}")
    mx = None
    dist = None

from .config import MLXDistributedConfig, NodeConfig

logger = logging.getLogger(__name__)

@dataclass
class LayerAssignment:
    """Represents which layers are assigned to which node"""
    node_name: str
    start_layer: int
    end_layer: int
    memory_required: float  # in GB
    
    def to_dict(self) -> Dict:
        return {
            'node_name': self.node_name,
            'start_layer': self.start_layer,
            'end_layer': self.end_layer,
            'memory_required': self.memory_required
        }

class DistributedMLXCluster:
    """
    Main class for distributed MLX operations across Apple Silicon cluster
    Handles initialization, model partitioning, and distributed inference
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the distributed cluster"""
        self.config = MLXDistributedConfig(config_file)
        self.world_size = self.config.get_cluster_size()
        self.rank = self.config.get_node_rank()
        self.current_node = self.config.current_node
        self.peers = self.config.get_peer_nodes()
        
        # Distributed state
        self.is_initialized = False
        self.device = None
        self.process_group = None
        
        # Model state
        self.loaded_models = {}
        self.layer_assignments = {}
        self.model_metadata = {}
        
        # Performance tracking
        self.metrics = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'total_tokens_generated': 0,
            'memory_usage': 0.0
        }
        
        logger.info(f"Initialized DistributedMLXCluster - Rank: {self.rank}/{self.world_size}")
        
    async def initialize(self) -> bool:
        """Initialize the distributed cluster"""
        try:
            if not self.current_node:
                raise RuntimeError("Current node not detected")
            
            # Initialize MLX distributed
            if dist and not self.is_initialized:
                logger.info("Initializing MLX distributed backend...")
                self.rank = dist.init()
                self.device = mx.Device(mx.gpu, self.rank % mx.metal.device_count())
                logger.info(f"MLX distributed initialized - Rank: {self.rank}, Device: {self.device}")
                
                # Create process group for communication
                self.process_group = dist.new_group(ranks=list(range(self.world_size)))
                
                self.is_initialized = True
            
            # Test connectivity with peers
            connectivity_results = await self._test_peer_connectivity()
            active_peers = sum(1 for result in connectivity_results.values() if result)
            
            logger.info(f"Cluster connectivity: {active_peers}/{len(self.peers)} peers reachable")
            
            return self.is_initialized
            
        except Exception as e:
            logger.error(f"Cluster initialization failed: {e}")
            return False
    
    async def _test_peer_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to peer nodes"""
        results = {}
        
        async def test_node(node: NodeConfig) -> Tuple[str, bool]:
            try:
                # Simple network test
                proc = await asyncio.create_subprocess_exec(
                    'ping', '-c', '1', '-W', '2000', node.ip,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(proc.wait(), timeout=5.0)
                return node.name, proc.returncode == 0
            except Exception:
                return node.name, False
        
        # Test all peers concurrently
        tasks = [test_node(node) for node in self.peers]
        if tasks:
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in test_results:
                if isinstance(result, tuple):
                    node_name, is_reachable = result
                    results[node_name] = is_reachable
                else:
                    logger.debug(f"Connectivity test exception: {result}")
        
        return results
    
    def setup_model_parallel(self, model_config: Dict[str, Any]) -> Dict[str, LayerAssignment]:
        """
        Partition model across nodes based on memory and compute capacity
        
        Args:
            model_config: Model configuration including layer count, size, etc.
            
        Returns:
            Dictionary mapping node names to layer assignments
        """
        try:
            total_layers = model_config.get('num_layers', 32)
            model_size_gb = model_config.get('size_gb', 7.0)
            
            # Calculate memory per layer (rough estimate)
            memory_per_layer = model_size_gb / total_layers
            
            # Get available memory per node
            memory_per_node = {}
            for node in self.config.cluster_config.nodes:
                # Reserve 20% of memory for system and activations
                available_memory = node.memory_gb * 0.8
                memory_per_node[node.name] = available_memory
            
            # Sort nodes by memory capacity (descending)
            sorted_nodes = sorted(
                self.config.cluster_config.nodes,
                key=lambda x: x.memory_gb,
                reverse=True
            )
            
            # Assign layers to nodes
            assignments = {}
            current_layer = 0
            
            for node in sorted_nodes:
                if current_layer >= total_layers:
                    break
                
                # Calculate how many layers this node can handle
                node_memory = memory_per_node[node.name]
                layers_for_node = min(
                    int(node_memory / memory_per_layer),
                    total_layers - current_layer
                )
                
                if layers_for_node > 0:
                    assignments[node.name] = LayerAssignment(
                        node_name=node.name,
                        start_layer=current_layer,
                        end_layer=current_layer + layers_for_node - 1,
                        memory_required=layers_for_node * memory_per_layer
                    )
                    current_layer += layers_for_node
            
            # Handle remaining layers if any
            if current_layer < total_layers:
                # Assign remaining layers to the node with most memory
                largest_node = sorted_nodes[0]
                if largest_node.name in assignments:
                    assignments[largest_node.name].end_layer = total_layers - 1
                    assignments[largest_node.name].memory_required += (total_layers - current_layer) * memory_per_layer
            
            self.layer_assignments[model_config.get('name', 'default')] = assignments
            
            logger.info(f"Model partitioning complete:")
            for node_name, assignment in assignments.items():
                logger.info(f"  {node_name}: layers {assignment.start_layer}-{assignment.end_layer} "
                          f"({assignment.memory_required:.2f}GB)")
            
            return assignments
            
        except Exception as e:
            logger.error(f"Model partitioning failed: {e}")
            return {}
    
    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        """
        Perform all-reduce operation across all nodes
        
        Args:
            tensor: Input tensor
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            
        Returns:
            Reduced tensor
        """
        if not self.is_initialized or not dist:
            logger.warning("Distributed backend not initialized, returning tensor as-is")
            return tensor
        
        try:
            if op == "sum":
                return dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elif op == "mean":
                result = dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                return result / self.world_size
            elif op == "max":
                return dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            elif op == "min":
                return dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")
                
        except Exception as e:
            logger.error(f"All-reduce operation failed: {e}")
            return tensor
    
    def all_gather(self, tensor: Any) -> List[Any]:
        """
        Gather tensors from all nodes
        
        Args:
            tensor: Input tensor
            
        Returns:
            List of tensors from all nodes
        """
        if not self.is_initialized or not dist:
            logger.warning("Distributed backend not initialized, returning single tensor")
            return [tensor]
        
        try:
            return dist.all_gather(tensor)
        except Exception as e:
            logger.error(f"All-gather operation failed: {e}")
            return [tensor]
    
    def broadcast(self, tensor: Any, root: int = 0) -> Any:
        """
        Broadcast tensor from root to all nodes
        
        Args:
            tensor: Input tensor
            root: Root node rank
            
        Returns:
            Broadcasted tensor
        """
        if not self.is_initialized or not dist:
            logger.warning("Distributed backend not initialized, returning tensor as-is")
            return tensor
        
        try:
            return dist.broadcast(tensor, root)
        except Exception as e:
            logger.error(f"Broadcast operation failed: {e}")
            return tensor
    
    async def load_model_distributed(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """
        Load model with distributed partitioning
        
        Args:
            model_path: Path to model files
            model_config: Model configuration
            
        Returns:
            True if successful
        """
        try:
            model_name = model_config.get('name', os.path.basename(model_path))
            
            # Create partitioning plan
            assignments = self.setup_model_parallel(model_config)
            if not assignments:
                raise RuntimeError("Failed to create model partitioning plan")
            
            # Get this node's assignment
            my_assignment = assignments.get(self.current_node.name)
            if not my_assignment:
                logger.warning(f"No layers assigned to node {self.current_node.name}")
                return False
            
            logger.info(f"Loading layers {my_assignment.start_layer}-{my_assignment.end_layer} "
                       f"for model {model_name}")
            
            # In a real implementation, this would load the actual model layers
            # For now, we'll simulate the loading process
            model_info = {
                'name': model_name,
                'path': model_path,
                'config': model_config,
                'assignment': my_assignment,
                'loaded_at': time.time()
            }
            
            self.loaded_models[model_name] = model_info
            self.model_metadata[model_name] = model_config
            
            # Update memory usage
            self.metrics['memory_usage'] += my_assignment.memory_required
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Distributed model loading failed: {e}")
            return False
    
    async def inference_distributed(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform distributed inference
        
        Args:
            model_name: Name of the loaded model
            input_data: Input data for inference
            
        Returns:
            Inference results
        """
        start_time = time.time()
        
        try:
            if model_name not in self.loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            model_info = self.loaded_models[model_name]
            assignment = model_info['assignment']
            
            logger.debug(f"Starting distributed inference for {model_name}")
            
            # Simulate distributed inference
            # In a real implementation, this would:
            # 1. Process input through assigned layers
            # 2. Communicate intermediate results to next nodes
            # 3. Aggregate final results
            
            # For now, simulate processing time based on layer count
            layer_count = assignment.end_layer - assignment.start_layer + 1
            processing_time = layer_count * 0.01  # 10ms per layer simulation
            await asyncio.sleep(processing_time)
            
            # Mock result
            result = {
                'model': model_name,
                'node': self.current_node.name,
                'layers_processed': f"{assignment.start_layer}-{assignment.end_layer}",
                'processing_time': processing_time,
                'tokens_generated': input_data.get('max_tokens', 100),
                'timestamp': time.time()
            }
            
            # Update metrics
            self.metrics['inference_count'] += 1
            self.metrics['total_inference_time'] += processing_time
            self.metrics['total_tokens_generated'] += result['tokens_generated']
            
            logger.debug(f"Distributed inference completed in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Distributed inference failed: {e}")
            return {'error': str(e), 'model': model_name}
        finally:
            end_time = time.time()
            logger.debug(f"Total inference time: {end_time - start_time:.3f}s")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        return {
            'node_info': {
                'name': self.current_node.name if self.current_node else 'Unknown',
                'rank': self.rank,
                'role': self.current_node.role if self.current_node else 'Unknown'
            },
            'cluster_info': {
                'world_size': self.world_size,
                'backend': self.config.cluster_config.backend,
                'initialized': self.is_initialized
            },
            'loaded_models': list(self.loaded_models.keys()),
            'metrics': self.metrics.copy(),
            'memory_usage': f"{self.metrics['memory_usage']:.2f}GB"
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if self.metrics['inference_count'] > 0:
            avg_inference_time = self.metrics['total_inference_time'] / self.metrics['inference_count']
            tokens_per_second = self.metrics['total_tokens_generated'] / self.metrics['total_inference_time'] if self.metrics['total_inference_time'] > 0 else 0
        else:
            avg_inference_time = 0
            tokens_per_second = 0
        
        return {
            'total_inferences': self.metrics['inference_count'],
            'avg_inference_time': f"{avg_inference_time:.3f}s",
            'total_tokens': self.metrics['total_tokens_generated'],
            'tokens_per_second': f"{tokens_per_second:.2f}",
            'memory_usage': f"{self.metrics['memory_usage']:.2f}GB",
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
    
    async def shutdown(self) -> None:
        """Shutdown the distributed cluster"""
        try:
            logger.info("Shutting down distributed cluster...")
            
            # Clear loaded models
            self.loaded_models.clear()
            self.layer_assignments.clear()
            self.model_metadata.clear()
            
            # Finalize distributed backend
            if self.is_initialized and dist:
                dist.finalize()
                self.is_initialized = False
            
            logger.info("Cluster shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}")

# Example usage and testing
async def main():
    """Example usage of DistributedMLXCluster"""
    cluster = DistributedMLXCluster()
    
    try:
        # Initialize cluster
        if await cluster.initialize():
            print("✓ Cluster initialized successfully")
        else:
            print("✗ Cluster initialization failed")
            return
        
        # Show cluster status
        status = cluster.get_cluster_status()
        print(f"Cluster status: {json.dumps(status, indent=2)}")
        
        # Example model configuration
        model_config = {
            'name': 'llama-7b',
            'num_layers': 32,
            'size_gb': 7.0,
            'architecture': 'llama'
        }
        
        # Load model
        if await cluster.load_model_distributed("models/llama-7b", model_config):
            print("✓ Model loaded successfully")
        else:
            print("✗ Model loading failed")
            return
        
        # Perform inference
        input_data = {
            'prompt': 'Hello, how are you?',
            'max_tokens': 100,
            'temperature': 0.7
        }
        
        result = await cluster.inference_distributed('llama-7b', input_data)
        print(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Show performance metrics
        metrics = cluster.get_performance_metrics()
        print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
    finally:
        await cluster.shutdown()

if __name__ == "__main__":
    asyncio.run(main())