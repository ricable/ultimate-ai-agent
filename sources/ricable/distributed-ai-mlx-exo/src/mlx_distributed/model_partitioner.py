"""
Model Partitioning Strategy for Distributed MLX Cluster
Implements memory-weighted and compute-aware model partitioning strategies
"""

import logging
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .config import NodeConfig

logger = logging.getLogger(__name__)

class PartitioningStrategy(Enum):
    """Available partitioning strategies"""
    RING_MEMORY_WEIGHTED = "ring_memory_weighted"
    COMPUTE_BALANCED = "compute_balanced"
    HYBRID_OPTIMAL = "hybrid_optimal"
    LAYER_SEQUENTIAL = "layer_sequential"
    PIPELINE_PARALLEL = "pipeline_parallel"

@dataclass
class LayerMetadata:
    """Metadata for individual model layers"""
    layer_id: int
    layer_type: str  # attention, feed_forward, embedding, etc.
    param_count: int
    memory_mb: float
    compute_complexity: float  # FLOPS estimate
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    dependencies: List[int] = field(default_factory=list)

@dataclass
class ModelMetadata:
    """Complete model metadata for partitioning"""
    name: str
    architecture: str
    total_params: int
    total_memory_gb: float
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    max_sequence_length: int
    layers: List[LayerMetadata] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'architecture': self.architecture,
            'total_params': self.total_params,
            'total_memory_gb': self.total_memory_gb,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'layers': [
                {
                    'layer_id': l.layer_id,
                    'layer_type': l.layer_type,
                    'param_count': l.param_count,
                    'memory_mb': l.memory_mb,
                    'compute_complexity': l.compute_complexity,
                    'input_shape': l.input_shape,
                    'output_shape': l.output_shape,
                    'dependencies': l.dependencies
                } for l in self.layers
            ]
        }

@dataclass
class PartitionPlan:
    """Represents a complete partitioning plan"""
    model_name: str
    strategy: PartitioningStrategy
    partitions: Dict[str, 'Partition']
    total_memory_required: float
    estimated_inference_time: float
    load_balance_score: float
    communication_overhead: float
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'strategy': self.strategy.value,
            'partitions': {node_id: partition.to_dict() for node_id, partition in self.partitions.items()},
            'total_memory_required': self.total_memory_required,
            'estimated_inference_time': self.estimated_inference_time,
            'load_balance_score': self.load_balance_score,
            'communication_overhead': self.communication_overhead
        }

@dataclass
class Partition:
    """Represents a single partition of the model"""
    node_id: str
    layer_range: Tuple[int, int]  # (start_layer, end_layer)
    layers: List[LayerMetadata]
    memory_required_gb: float
    compute_load: float
    estimated_latency: float
    shard_path: str
    dependencies: List[str] = field(default_factory=list)  # Other partitions this depends on
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'layer_range': self.layer_range,
            'layers': [layer.layer_id for layer in self.layers],
            'memory_required_gb': self.memory_required_gb,
            'compute_load': self.compute_load,
            'estimated_latency': self.estimated_latency,
            'shard_path': self.shard_path,
            'dependencies': self.dependencies
        }

class ModelPartitioner:
    """
    Intelligent model partitioner for distributed inference
    Supports multiple partitioning strategies optimized for Apple Silicon clusters
    """
    
    def __init__(self, nodes: List[NodeConfig]):
        self.nodes = nodes
        self.node_capabilities = self._analyze_node_capabilities()
        self.total_cluster_memory = sum(node.memory_gb for node in nodes)
        self.total_cluster_compute = sum(self._estimate_compute_power(node) for node in nodes)
        
        logger.info(f"Model partitioner initialized for {len(nodes)} nodes")
        logger.info(f"Total cluster resources: {self.total_cluster_memory}GB memory, "
                   f"{self.total_cluster_compute:.2f} compute units")
    
    def _analyze_node_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Analyze and categorize node capabilities"""
        capabilities = {}
        
        for node in self.nodes:
            compute_power = self._estimate_compute_power(node)
            memory_tier = self._categorize_memory_tier(node.memory_gb)
            
            capabilities[node.name] = {
                'memory_gb': node.memory_gb,
                'gpu_cores': node.gpu_cores,
                'cpu_cores': node.cpu_cores,
                'compute_power': compute_power,
                'memory_tier': memory_tier,
                'efficiency_score': self._calculate_efficiency_score(node),
                'role_preference': self._determine_role_preference(node)
            }
        
        return capabilities
    
    def _estimate_compute_power(self, node: NodeConfig) -> float:
        """Estimate relative compute power of a node"""
        # Simplified compute power estimation based on chip type and cores
        base_power = node.gpu_cores * 0.1  # Base GPU power
        
        # Apply chip-specific multipliers
        if "M3" in node.name:
            multiplier = 1.3
        elif "M2" in node.name:
            multiplier = 1.1
        elif "M1" in node.name:
            multiplier = 1.0
        else:
            multiplier = 0.8
        
        return base_power * multiplier
    
    def _categorize_memory_tier(self, memory_gb: int) -> str:
        """Categorize node into memory tiers"""
        if memory_gb >= 128:
            return "ultra_high"
        elif memory_gb >= 64:
            return "high"
        elif memory_gb >= 32:
            return "medium"
        else:
            return "low"
    
    def _calculate_efficiency_score(self, node: NodeConfig) -> float:
        """Calculate overall efficiency score for a node"""
        memory_score = min(node.memory_gb / 128, 1.0)  # Normalize to 128GB max
        compute_score = min(self._estimate_compute_power(node) / 5.0, 1.0)  # Normalize
        
        # Weighted combination
        return 0.6 * memory_score + 0.4 * compute_score
    
    def _determine_role_preference(self, node: NodeConfig) -> str:
        """Determine preferred role for a node"""
        if node.memory_gb >= 128:
            return "memory_intensive"  # Embeddings, large layers
        elif node.memory_gb >= 64:
            return "balanced"  # Attention layers
        elif node.memory_gb >= 32:
            return "compute_intensive"  # Feed-forward layers
        else:
            return "auxiliary"  # Coordination, small tasks
    
    def create_model_metadata(self, model_config: Dict[str, Any]) -> ModelMetadata:
        """Create detailed model metadata for partitioning"""
        name = model_config.get('name', 'unknown')
        architecture = model_config.get('architecture', 'transformer')
        num_layers = model_config.get('num_layers', 32)
        hidden_size = model_config.get('hidden_size', 4096)
        num_heads = model_config.get('num_attention_heads', 32)
        vocab_size = model_config.get('vocab_size', 50000)
        max_seq_len = model_config.get('max_sequence_length', 2048)
        
        # Calculate total parameters and memory
        total_params = self._estimate_total_parameters(model_config)
        total_memory_gb = self._estimate_memory_requirements(model_config)
        
        # Create layer metadata
        layers = self._create_layer_metadata(model_config)
        
        metadata = ModelMetadata(
            name=name,
            architecture=architecture,
            total_params=total_params,
            total_memory_gb=total_memory_gb,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_sequence_length=max_seq_len,
            layers=layers
        )
        
        logger.info(f"Created model metadata for {name}: {total_params:,} params, "
                   f"{total_memory_gb:.2f}GB memory, {num_layers} layers")
        
        return metadata
    
    def _estimate_total_parameters(self, model_config: Dict[str, Any]) -> int:
        """Estimate total model parameters"""
        num_layers = model_config.get('num_layers', 32)
        hidden_size = model_config.get('hidden_size', 4096)
        num_heads = model_config.get('num_attention_heads', 32)
        vocab_size = model_config.get('vocab_size', 50000)
        
        # Transformer parameter estimation
        # Embedding layer
        embedding_params = vocab_size * hidden_size
        
        # Each transformer layer
        attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        ffn_params = 8 * hidden_size * hidden_size  # Typically 4x hidden size
        layer_params = attention_params + ffn_params
        
        # Output layer
        output_params = hidden_size * vocab_size
        
        total_params = embedding_params + (num_layers * layer_params) + output_params
        
        return total_params
    
    def _estimate_memory_requirements(self, model_config: Dict[str, Any]) -> float:
        """Estimate memory requirements in GB"""
        total_params = self._estimate_total_parameters(model_config)
        
        # Base memory for parameters (assuming fp16)
        param_memory_gb = total_params * 2 / (1024**3)
        
        # Additional memory for activations, gradients, optimizer states
        activation_memory_gb = param_memory_gb * 0.3  # Roughly 30% for activations
        
        # Add buffer for safety
        total_memory_gb = (param_memory_gb + activation_memory_gb) * 1.2
        
        return total_memory_gb
    
    def _create_layer_metadata(self, model_config: Dict[str, Any]) -> List[LayerMetadata]:
        """Create metadata for individual layers"""
        layers = []
        num_layers = model_config.get('num_layers', 32)
        hidden_size = model_config.get('hidden_size', 4096)
        num_heads = model_config.get('num_attention_heads', 32)
        max_seq_len = model_config.get('max_sequence_length', 2048)
        
        # Create embedding layer
        embedding_layer = LayerMetadata(
            layer_id=0,
            layer_type="embedding",
            param_count=model_config.get('vocab_size', 50000) * hidden_size,
            memory_mb=(model_config.get('vocab_size', 50000) * hidden_size * 2) / (1024**2),
            compute_complexity=max_seq_len * hidden_size,
            input_shape=(max_seq_len,),
            output_shape=(max_seq_len, hidden_size)
        )
        layers.append(embedding_layer)
        
        # Create transformer layers
        for i in range(num_layers):
            # Attention layer
            attention_params = 4 * hidden_size * hidden_size
            attention_layer = LayerMetadata(
                layer_id=i * 2 + 1,
                layer_type="attention",
                param_count=attention_params,
                memory_mb=(attention_params * 2) / (1024**2),
                compute_complexity=max_seq_len * max_seq_len * hidden_size,  # Attention complexity
                input_shape=(max_seq_len, hidden_size),
                output_shape=(max_seq_len, hidden_size),
                dependencies=[max(0, i * 2)] if i > 0 else [0]
            )
            layers.append(attention_layer)
            
            # Feed-forward layer
            ffn_params = 8 * hidden_size * hidden_size
            ffn_layer = LayerMetadata(
                layer_id=i * 2 + 2,
                layer_type="feed_forward",
                param_count=ffn_params,
                memory_mb=(ffn_params * 2) / (1024**2),
                compute_complexity=max_seq_len * hidden_size * hidden_size * 4,
                input_shape=(max_seq_len, hidden_size),
                output_shape=(max_seq_len, hidden_size),
                dependencies=[i * 2 + 1]
            )
            layers.append(ffn_layer)
        
        return layers
    
    def create_partition_plan(self, model_metadata: ModelMetadata, 
                            strategy: PartitioningStrategy = PartitioningStrategy.RING_MEMORY_WEIGHTED) -> Optional[PartitionPlan]:
        """Create an optimal partition plan for the model"""
        logger.info(f"Creating partition plan for {model_metadata.name} using {strategy.value} strategy")
        
        # Check if model fits in cluster
        if model_metadata.total_memory_gb > self.total_cluster_memory * 0.8:  # Leave 20% buffer
            logger.error(f"Model requires {model_metadata.total_memory_gb:.2f}GB but cluster only has "
                        f"{self.total_cluster_memory * 0.8:.2f}GB available")
            return None
        
        # Create partitions based on strategy
        if strategy == PartitioningStrategy.RING_MEMORY_WEIGHTED:
            partitions = self._create_ring_memory_weighted_partitions(model_metadata)
        elif strategy == PartitioningStrategy.COMPUTE_BALANCED:
            partitions = self._create_compute_balanced_partitions(model_metadata)
        elif strategy == PartitioningStrategy.HYBRID_OPTIMAL:
            partitions = self._create_hybrid_optimal_partitions(model_metadata)
        elif strategy == PartitioningStrategy.LAYER_SEQUENTIAL:
            partitions = self._create_layer_sequential_partitions(model_metadata)
        elif strategy == PartitioningStrategy.PIPELINE_PARALLEL:
            partitions = self._create_pipeline_parallel_partitions(model_metadata)
        else:
            logger.error(f"Unknown partitioning strategy: {strategy}")
            return None
        
        if not partitions:
            logger.error("Failed to create partitions")
            return None
        
        # Calculate plan metrics
        plan_metrics = self._calculate_plan_metrics(partitions, model_metadata)
        
        plan = PartitionPlan(
            model_name=model_metadata.name,
            strategy=strategy,
            partitions=partitions,
            **plan_metrics
        )
        
        logger.info(f"Partition plan created: {len(partitions)} partitions, "
                   f"load balance score: {plan.load_balance_score:.3f}")
        
        return plan
    
    def _create_ring_memory_weighted_partitions(self, model_metadata: ModelMetadata) -> Dict[str, Partition]:
        """Create partitions using ring memory-weighted strategy"""
        partitions = {}
        
        # Sort nodes by memory capacity (descending)
        sorted_nodes = sorted(self.nodes, key=lambda x: x.memory_gb, reverse=True)
        
        # Calculate memory allocation ratios
        total_memory = sum(node.memory_gb for node in sorted_nodes)
        memory_ratios = [node.memory_gb / total_memory for node in sorted_nodes]
        
        # Distribute layers based on memory ratios
        total_layers = len(model_metadata.layers)
        current_layer = 0
        
        for i, (node, ratio) in enumerate(zip(sorted_nodes, memory_ratios)):
            # Calculate layers for this node
            if i == len(sorted_nodes) - 1:  # Last node gets remaining layers
                layers_for_node = total_layers - current_layer
            else:
                layers_for_node = max(1, int(total_layers * ratio))
            
            if layers_for_node <= 0:
                continue
            
            # Create partition
            end_layer = min(current_layer + layers_for_node - 1, total_layers - 1)
            partition_layers = model_metadata.layers[current_layer:end_layer + 1]
            
            memory_required = sum(layer.memory_mb for layer in partition_layers) / 1024  # Convert to GB
            compute_load = sum(layer.compute_complexity for layer in partition_layers)
            
            partition = Partition(
                node_id=node.name,
                layer_range=(current_layer, end_layer),
                layers=partition_layers,
                memory_required_gb=memory_required,
                compute_load=compute_load,
                estimated_latency=self._estimate_partition_latency(partition_layers, node),
                shard_path=f"models/{model_metadata.name}/shards/{node.name}_{current_layer}_{end_layer}"
            )
            
            partitions[node.name] = partition
            current_layer = end_layer + 1
            
            if current_layer >= total_layers:
                break
        
        return partitions
    
    def _create_compute_balanced_partitions(self, model_metadata: ModelMetadata) -> Dict[str, Partition]:
        """Create partitions balanced by compute requirements"""
        partitions = {}
        
        # Calculate total compute complexity
        total_compute = sum(layer.compute_complexity for layer in model_metadata.layers)
        
        # Sort nodes by compute power
        sorted_nodes = sorted(self.nodes, key=lambda x: self._estimate_compute_power(x), reverse=True)
        
        # Distribute compute load
        current_layer = 0
        remaining_compute = total_compute
        
        for i, node in enumerate(sorted_nodes):
            node_compute_power = self._estimate_compute_power(node)
            compute_ratio = node_compute_power / self.total_cluster_compute
            
            # Find layers that match this compute allocation
            target_compute = total_compute * compute_ratio
            layer_compute = 0
            layers_for_node = 0
            
            while (current_layer + layers_for_node < len(model_metadata.layers) and 
                   layer_compute < target_compute):
                layer_compute += model_metadata.layers[current_layer + layers_for_node].compute_complexity
                layers_for_node += 1
            
            if layers_for_node == 0 and current_layer < len(model_metadata.layers):
                layers_for_node = 1  # Ensure each node gets at least one layer
            
            if layers_for_node > 0:
                end_layer = min(current_layer + layers_for_node - 1, len(model_metadata.layers) - 1)
                partition_layers = model_metadata.layers[current_layer:end_layer + 1]
                
                memory_required = sum(layer.memory_mb for layer in partition_layers) / 1024
                compute_load = sum(layer.compute_complexity for layer in partition_layers)
                
                partition = Partition(
                    node_id=node.name,
                    layer_range=(current_layer, end_layer),
                    layers=partition_layers,
                    memory_required_gb=memory_required,
                    compute_load=compute_load,
                    estimated_latency=self._estimate_partition_latency(partition_layers, node),
                    shard_path=f"models/{model_metadata.name}/shards/{node.name}_{current_layer}_{end_layer}"
                )
                
                partitions[node.name] = partition
                current_layer = end_layer + 1
        
        # Handle any remaining layers
        if current_layer < len(model_metadata.layers):
            # Add remaining layers to the most capable node
            best_node = sorted_nodes[0]
            if best_node.name in partitions:
                # Extend existing partition
                partition = partitions[best_node.name]
                remaining_layers = model_metadata.layers[current_layer:]
                partition.layers.extend(remaining_layers)
                partition.layer_range = (partition.layer_range[0], len(model_metadata.layers) - 1)
                partition.memory_required_gb += sum(layer.memory_mb for layer in remaining_layers) / 1024
                partition.compute_load += sum(layer.compute_complexity for layer in remaining_layers)
        
        return partitions
    
    def _create_hybrid_optimal_partitions(self, model_metadata: ModelMetadata) -> Dict[str, Partition]:
        """Create partitions using hybrid optimization considering both memory and compute"""
        # This is a simplified version - in practice, this would use more sophisticated optimization
        
        # Start with memory-weighted partitions
        memory_partitions = self._create_ring_memory_weighted_partitions(model_metadata)
        
        # Evaluate and adjust based on compute balance
        compute_partitions = self._create_compute_balanced_partitions(model_metadata)
        
        # Score both approaches
        memory_score = self._score_partitions(memory_partitions)
        compute_score = self._score_partitions(compute_partitions)
        
        # Choose the better approach or create a hybrid
        if memory_score > compute_score:
            logger.debug("Using memory-weighted partitions for hybrid strategy")
            return memory_partitions
        else:
            logger.debug("Using compute-balanced partitions for hybrid strategy")
            return compute_partitions
    
    def _create_layer_sequential_partitions(self, model_metadata: ModelMetadata) -> Dict[str, Partition]:
        """Create sequential layer partitions (simple round-robin)"""
        partitions = {}
        layers_per_node = math.ceil(len(model_metadata.layers) / len(self.nodes))
        
        for i, node in enumerate(self.nodes):
            start_layer = i * layers_per_node
            end_layer = min(start_layer + layers_per_node - 1, len(model_metadata.layers) - 1)
            
            if start_layer < len(model_metadata.layers):
                partition_layers = model_metadata.layers[start_layer:end_layer + 1]
                memory_required = sum(layer.memory_mb for layer in partition_layers) / 1024
                compute_load = sum(layer.compute_complexity for layer in partition_layers)
                
                partition = Partition(
                    node_id=node.name,
                    layer_range=(start_layer, end_layer),
                    layers=partition_layers,
                    memory_required_gb=memory_required,
                    compute_load=compute_load,
                    estimated_latency=self._estimate_partition_latency(partition_layers, node),
                    shard_path=f"models/{model_metadata.name}/shards/{node.name}_{start_layer}_{end_layer}"
                )
                
                partitions[node.name] = partition
        
        return partitions
    
    def _create_pipeline_parallel_partitions(self, model_metadata: ModelMetadata) -> Dict[str, Partition]:
        """Create pipeline parallel partitions with dependency management"""
        # Similar to sequential but with explicit pipeline dependencies
        partitions = self._create_layer_sequential_partitions(model_metadata)
        
        # Add pipeline dependencies
        node_names = list(partitions.keys())
        for i, node_name in enumerate(node_names):
            if i > 0:
                # This partition depends on the previous one
                prev_node = node_names[i - 1]
                partitions[node_name].dependencies = [prev_node]
        
        return partitions
    
    def _estimate_partition_latency(self, layers: List[LayerMetadata], node: NodeConfig) -> float:
        """Estimate latency for a partition on a specific node"""
        total_compute = sum(layer.compute_complexity for layer in layers)
        node_compute_power = self._estimate_compute_power(node)
        
        # Simple latency model: compute / power + memory access overhead
        compute_latency = total_compute / (node_compute_power * 1e9)  # Rough FLOPS estimation
        memory_latency = sum(layer.memory_mb for layer in layers) / 1000  # Memory access time
        
        return compute_latency + memory_latency * 0.1  # 10% memory overhead
    
    def _calculate_plan_metrics(self, partitions: Dict[str, Partition], 
                              model_metadata: ModelMetadata) -> Dict[str, float]:
        """Calculate metrics for a partition plan"""
        total_memory = sum(p.memory_required_gb for p in partitions.values())
        max_latency = max(p.estimated_latency for p in partitions.values())
        
        # Load balance score (lower variance is better)
        latencies = [p.estimated_latency for p in partitions.values()]
        load_balance_score = 1.0 / (1.0 + np.var(latencies)) if len(latencies) > 1 else 1.0
        
        # Communication overhead (simplified)
        communication_overhead = len(partitions) * 0.01  # 10ms per partition boundary
        
        return {
            'total_memory_required': total_memory,
            'estimated_inference_time': max_latency + communication_overhead,
            'load_balance_score': load_balance_score,
            'communication_overhead': communication_overhead
        }
    
    def _score_partitions(self, partitions: Dict[str, Partition]) -> float:
        """Score a set of partitions (higher is better)"""
        if not partitions:
            return 0.0
        
        # Balance score based on load distribution
        loads = [p.compute_load for p in partitions.values()]
        load_variance = np.var(loads) if len(loads) > 1 else 0
        balance_score = 1.0 / (1.0 + load_variance)
        
        # Memory efficiency score
        total_memory = sum(p.memory_required_gb for p in partitions.values())
        memory_efficiency = min(1.0, self.total_cluster_memory / total_memory)
        
        # Combined score
        return 0.6 * balance_score + 0.4 * memory_efficiency
    
    def validate_partition_plan(self, plan: PartitionPlan) -> Tuple[bool, List[str]]:
        """Validate a partition plan for feasibility"""
        errors = []
        
        # Check memory constraints
        for node_id, partition in plan.partitions.items():
            node = next((n for n in self.nodes if n.name == node_id), None)
            if not node:
                errors.append(f"Unknown node: {node_id}")
                continue
            
            if partition.memory_required_gb > node.memory_gb * 0.9:  # Leave 10% buffer
                errors.append(f"Node {node_id} memory overflow: "
                            f"{partition.memory_required_gb:.2f}GB required, "
                            f"{node.memory_gb}GB available")
        
        # Check layer coverage
        all_layers = set()
        for partition in plan.partitions.values():
            layer_range = range(partition.layer_range[0], partition.layer_range[1] + 1)
            all_layers.update(layer_range)
        
        expected_layers = set(range(len(plan.partitions)))  # Simplified check
        if len(all_layers) == 0:
            errors.append("No layers assigned to any partition")
        
        # Check dependencies
        for partition in plan.partitions.values():
            for dep in partition.dependencies:
                if dep not in plan.partitions:
                    errors.append(f"Partition {partition.node_id} depends on missing partition {dep}")
        
        return len(errors) == 0, errors
    
    def optimize_partition_plan(self, plan: PartitionPlan) -> PartitionPlan:
        """Optimize an existing partition plan"""
        logger.info(f"Optimizing partition plan for {plan.model_name}")
        
        # Simple optimization: try to balance load
        current_score = self._score_partitions(plan.partitions)
        
        # Try different strategies and see if we can improve
        model_metadata = ModelMetadata(
            name=plan.model_name,
            architecture="transformer",  # Assumption
            total_params=0,  # Will be calculated
            total_memory_gb=plan.total_memory_required,
            num_layers=max(p.layer_range[1] for p in plan.partitions.values()) + 1,
            hidden_size=4096,  # Default
            num_attention_heads=32,  # Default
            vocab_size=50000,  # Default
            max_sequence_length=2048  # Default
        )
        
        # Try compute-balanced approach
        alt_partitions = self._create_compute_balanced_partitions(model_metadata)
        alt_score = self._score_partitions(alt_partitions)
        
        if alt_score > current_score:
            logger.info(f"Improved partition plan score from {current_score:.3f} to {alt_score:.3f}")
            plan.partitions = alt_partitions
            plan.load_balance_score = alt_score
        
        return plan

# Example usage and testing
if __name__ == "__main__":
    # Test model partitioner
    nodes = [
        NodeConfig("mac-node-1", "10.0.1.10", "compute", 64, 32, 10),
        NodeConfig("mac-node-2", "10.0.1.11", "compute", 64, 32, 10),
        NodeConfig("mac-node-3", "10.0.1.12", "compute", 32, 30, 12)
    ]
    
    partitioner = ModelPartitioner(nodes)
    
    # Test with a sample model
    model_config = {
        'name': 'llama-7b',
        'architecture': 'llama',
        'num_layers': 32,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'vocab_size': 32000,
        'max_sequence_length': 2048
    }
    
    metadata = partitioner.create_model_metadata(model_config)
    print(f"Model metadata: {metadata.total_params:,} params, {metadata.total_memory_gb:.2f}GB")
    
    # Test different strategies
    for strategy in PartitioningStrategy:
        print(f"\nTesting {strategy.value} strategy:")
        plan = partitioner.create_partition_plan(metadata, strategy)
        
        if plan:
            print(f"  Partitions: {len(plan.partitions)}")
            print(f"  Load balance score: {plan.load_balance_score:.3f}")
            print(f"  Estimated inference time: {plan.estimated_inference_time:.3f}s")
            
            # Validate plan
            valid, errors = partitioner.validate_partition_plan(plan)
            print(f"  Valid: {valid}")
            if errors:
                for error in errors:
                    print(f"    Error: {error}")
        else:
            print("  Failed to create partition plan")