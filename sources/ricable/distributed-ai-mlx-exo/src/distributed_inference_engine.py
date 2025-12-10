"""
Distributed Inference Engine
Core engine that combines MLX distributed operations with Exo P2P coordination
for seamless distributed inference across Apple Silicon cluster
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import our distributed components
try:
    from .src.mlx_distributed.cluster import DistributedMLXCluster
    from .src.mlx_distributed.model_partitioner import ModelPartitioner, PartitionPlan, ModelMetadata, PartitioningStrategy
    from .src.exo_integration.enhanced_cluster_manager import EnhancedExoClusterManager, ModelConfig
    from .src.mlx_distributed.config import MLXDistributedConfig, NodeConfig
    COMPONENTS_AVAILABLE = True
except ImportError:
    # Fallback imports for development
    try:
        from src.mlx_distributed.cluster import DistributedMLXCluster
        from src.mlx_distributed.model_partitioner import ModelPartitioner, PartitionPlan, ModelMetadata, PartitioningStrategy
        from src.exo_integration.enhanced_cluster_manager import EnhancedExoClusterManager, ModelConfig
        from src.mlx_distributed.config import MLXDistributedConfig, NodeConfig
        COMPONENTS_AVAILABLE = True
    except ImportError:
        COMPONENTS_AVAILABLE = False
        # Create dummy classes to avoid NameError
        class DistributedMLXCluster:
            def __init__(self, *args, **kwargs): pass
        class ModelPartitioner:
            def __init__(self, *args, **kwargs): pass
        class PartitionPlan:
            def __init__(self, *args, **kwargs): pass
        class ModelMetadata:
            def __init__(self, *args, **kwargs): pass
        class PartitioningStrategy:
            RING_MEMORY_WEIGHTED = "ring_memory_weighted"
            COMPUTE_BALANCED = "compute_balanced"
            HYBRID_OPTIMAL = "hybrid_optimal"
            LAYER_SEQUENTIAL = "layer_sequential"
            PIPELINE_PARALLEL = "pipeline_parallel"
        class EnhancedExoClusterManager:
            def __init__(self, *args, **kwargs): pass
        class ModelConfig:
            def __init__(self, *args, **kwargs): pass
        class MLXDistributedConfig:
            def __init__(self, *args, **kwargs): pass
        class NodeConfig:
            def __init__(self, *args, **kwargs): pass
        logging.warning("Distributed components not available - running in mock mode")

logger = logging.getLogger(__name__)

class InferenceStatus(Enum):
    """Status of distributed inference requests"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    DISTRIBUTING = "distributing"
    PROCESSING = "processing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class InferenceRequest:
    """Represents a distributed inference request"""
    request_id: str
    model_name: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop_sequences: List[str]
    stream: bool
    metadata: Dict[str, Any]
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class InferenceResponse:
    """Represents a distributed inference response"""
    request_id: str
    status: InferenceStatus
    generated_text: str
    tokens_generated: int
    processing_time: float
    node_contributions: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'generated_text': self.generated_text,
            'tokens_generated': self.tokens_generated,
            'processing_time': self.processing_time,
            'node_contributions': self.node_contributions,
            'metadata': self.metadata,
            'error_message': self.error_message
        }

@dataclass
class StreamingToken:
    """Represents a streaming token response"""
    request_id: str
    token: str
    token_id: int
    is_final: bool
    node_id: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DistributedInferenceEngine:
    """
    Core distributed inference engine that orchestrates inference across the hybrid cluster
    Combines MLX distributed operations with Exo P2P coordination
    """
    
    def __init__(self, node_id: str, config_file: Optional[str] = None):
        self.node_id = node_id
        self.config_file = config_file
        
        # Initialize components
        self.mlx_cluster = None
        self.exo_cluster = None
        self.model_partitioner = None
        self.config = None
        
        # State management
        self.initialized = False
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.partition_plans: Dict[str, PartitionPlan] = {}
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.request_status: Dict[str, InferenceStatus] = {}
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'total_tokens_generated': 0,
            'avg_tokens_per_second': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Threading and async coordination
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.request_lock = threading.RLock()
        self.status_callbacks: Dict[str, Callable] = {}
        self.stream_callbacks: Dict[str, Callable] = {}
        
        logger.info(f"Distributed inference engine initialized for node {node_id}")
    
    async def initialize(self) -> bool:
        """Initialize the distributed inference engine"""
        logger.info("Initializing distributed inference engine...")
        
        try:
            if not COMPONENTS_AVAILABLE:
                logger.error("Required components not available")
                return False
            
            # Initialize MLX distributed configuration
            self.config = MLXDistributedConfig(self.config_file)
            if not self.config.current_node:
                logger.error("Current node not detected in MLX config")
                return False
            
            # Initialize model partitioner
            self.model_partitioner = ModelPartitioner(self.config.cluster_config.nodes)
            
            # Initialize MLX distributed cluster
            self.mlx_cluster = DistributedMLXCluster(self.config_file)
            mlx_success = await self.mlx_cluster.initialize()
            
            if not mlx_success:
                logger.warning("MLX cluster initialization failed, continuing with Exo only")
            
            # Initialize enhanced Exo cluster
            from src.exo_integration.cluster_manager import auto_detect_node_id, create_cluster_manager
            detected_node_id = auto_detect_node_id() or self.node_id
            
            base_exo_manager = create_cluster_manager(detected_node_id)
            self.exo_cluster = EnhancedExoClusterManager(base_exo_manager.node_spec, self.config_file)
            
            exo_success = await self.exo_cluster.initialize_hybrid_cluster()
            
            if not exo_success:
                logger.error("Exo cluster initialization failed")
                return False
            
            # Set up callbacks
            self.exo_cluster.on_model_loaded = self._on_model_loaded_callback
            
            self.initialized = True
            logger.info("Distributed inference engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed inference engine: {e}")
            return False
    
    async def load_model(self, model_name: str, model_config: Dict[str, Any], 
                        strategy: PartitioningStrategy = PartitioningStrategy.RING_MEMORY_WEIGHTED) -> bool:
        """Load and partition a model across the cluster"""
        logger.info(f"Loading model {model_name} with {strategy.value} partitioning")
        
        try:
            if not self.initialized:
                logger.error("Engine not initialized")
                return False
            
            # Create model metadata
            model_metadata = self.model_partitioner.create_model_metadata(model_config)
            
            # Create partition plan
            partition_plan = self.model_partitioner.create_partition_plan(model_metadata, strategy)
            if not partition_plan:
                logger.error(f"Failed to create partition plan for {model_name}")
                return False
            
            # Validate partition plan
            valid, errors = self.model_partitioner.validate_partition_plan(partition_plan)
            if not valid:
                logger.error(f"Invalid partition plan: {errors}")
                return False
            
            # Store partition plan
            self.partition_plans[model_name] = partition_plan
            
            # Load model via enhanced Exo cluster
            enhanced_model_config = ModelConfig(
                name=model_name,
                architecture=model_config.get('architecture', 'transformer'),
                num_layers=model_config.get('num_layers', 32),
                hidden_size=model_config.get('hidden_size', 4096),
                num_attention_heads=model_config.get('num_attention_heads', 32),
                vocab_size=model_config.get('vocab_size', 50000),
                max_sequence_length=model_config.get('max_sequence_length', 2048),
                model_size_gb=model_metadata.total_memory_gb
            )
            
            success = await self.exo_cluster.load_model_with_smart_partitioning(enhanced_model_config)
            
            if success:
                self.loaded_models[model_name] = {
                    'config': model_config,
                    'metadata': model_metadata,
                    'partition_plan': partition_plan,
                    'loaded_at': time.time()
                }
                logger.info(f"Model {model_name} loaded successfully")
                return True
            else:
                logger.error(f"Failed to load model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    async def distributed_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform distributed inference for a request"""
        start_time = time.time()
        logger.info(f"Starting distributed inference for request {request.request_id}")
        
        try:
            # Validate request
            if request.model_name not in self.loaded_models:
                raise ValueError(f"Model {request.model_name} not loaded")
            
            # Track request
            with self.request_lock:
                self.active_requests[request.request_id] = request
                self.request_status[request.request_id] = InferenceStatus.PENDING
            
            # Update status
            await self._update_request_status(request.request_id, InferenceStatus.PREPROCESSING)
            
            # Preprocess input
            processed_input = await self._preprocess_input(request)
            
            # Distribute and process
            await self._update_request_status(request.request_id, InferenceStatus.DISTRIBUTING)
            
            node_results = await self._distribute_and_process(request, processed_input)
            
            # Aggregate results
            await self._update_request_status(request.request_id, InferenceStatus.AGGREGATING)
            
            final_result = await self._aggregate_results(request, node_results)
            
            # Create response
            processing_time = time.time() - start_time
            
            response = InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.COMPLETED,
                generated_text=final_result.get('generated_text', ''),
                tokens_generated=final_result.get('tokens_generated', 0),
                processing_time=processing_time,
                node_contributions=node_results,
                metadata={
                    'model_name': request.model_name,
                    'partition_strategy': self.partition_plans[request.model_name].strategy.value,
                    'participating_nodes': list(node_results.keys()),
                    'total_processing_time': processing_time
                }
            )
            
            # Update metrics
            self._update_metrics(response)
            
            # Update status
            await self._update_request_status(request.request_id, InferenceStatus.COMPLETED)
            
            logger.info(f"Distributed inference completed for {request.request_id} in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Distributed inference failed for {request.request_id}: {e}")
            
            # Update status to failed
            await self._update_request_status(request.request_id, InferenceStatus.FAILED)
            
            # Create error response
            processing_time = time.time() - start_time
            response = InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                generated_text="",
                tokens_generated=0,
                processing_time=processing_time,
                node_contributions={},
                metadata={'error': str(e)},
                error_message=str(e)
            )
            
            self.metrics['failed_requests'] += 1
            return response
        
        finally:
            # Clean up
            with self.request_lock:
                self.active_requests.pop(request.request_id, None)
    
    async def streaming_inference(self, request: InferenceRequest) -> AsyncIterator[StreamingToken]:
        """Perform streaming distributed inference"""
        logger.info(f"Starting streaming inference for request {request.request_id}")
        
        try:
            if request.model_name not in self.loaded_models:
                raise ValueError(f"Model {request.model_name} not loaded")
            
            # Track request
            with self.request_lock:
                self.active_requests[request.request_id] = request
                self.request_status[request.request_id] = InferenceStatus.PENDING
            
            # Set up streaming coordination
            token_queue = asyncio.Queue()
            streaming_task = asyncio.create_task(
                self._coordinate_streaming_inference(request, token_queue)
            )
            
            # Yield tokens as they become available
            tokens_generated = 0
            try:
                while True:
                    token_data = await asyncio.wait_for(token_queue.get(), timeout=30.0)
                    
                    if token_data is None:  # End of stream marker
                        break
                    
                    streaming_token = StreamingToken(
                        request_id=request.request_id,
                        token=token_data['token'],
                        token_id=token_data.get('token_id', tokens_generated),
                        is_final=token_data.get('is_final', False),
                        node_id=token_data.get('node_id', self.node_id),
                        timestamp=time.time()
                    )
                    
                    tokens_generated += 1
                    yield streaming_token
                    
                    if streaming_token.is_final:
                        break
                        
            except asyncio.TimeoutError:
                logger.warning(f"Streaming inference timeout for {request.request_id}")
                
            # Wait for streaming task to complete
            await streaming_task
            
        except Exception as e:
            logger.error(f"Streaming inference failed for {request.request_id}: {e}")
            # Yield error token
            yield StreamingToken(
                request_id=request.request_id,
                token=f"[ERROR: {str(e)}]",
                token_id=-1,
                is_final=True,
                node_id=self.node_id,
                timestamp=time.time()
            )
        
        finally:
            # Clean up
            with self.request_lock:
                self.active_requests.pop(request.request_id, None)
                self.request_status.pop(request.request_id, None)
    
    async def _preprocess_input(self, request: InferenceRequest) -> Dict[str, Any]:
        """Preprocess input for distributed processing"""
        return {
            'prompt': request.prompt,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'stop_sequences': request.stop_sequences,
            'tokenized_input': request.prompt.split(),  # Simplified tokenization
            'metadata': request.metadata
        }
    
    async def _distribute_and_process(self, request: InferenceRequest, 
                                    processed_input: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Distribute processing across cluster nodes"""
        model_name = request.model_name
        partition_plan = self.partition_plans[model_name]
        
        # Get participating nodes
        participating_nodes = list(partition_plan.partitions.keys())
        
        # Coordinate distributed processing
        node_results = {}
        
        if self.exo_cluster:
            # Use enhanced Exo cluster for coordination
            result = await self.exo_cluster.distributed_inference(
                model_name, processed_input
            )
            
            # Extract node contributions (simulated for now)
            for node_id in participating_nodes:
                node_results[node_id] = {
                    'status': 'completed',
                    'processing_time': result.get('processing_time', 0) / len(participating_nodes),
                    'layers_processed': f"simulated for {node_id}",
                    'memory_usage': partition_plan.partitions[node_id].memory_required_gb,
                    'tokens_processed': len(processed_input.get('tokenized_input', [])),
                    'backend': 'exo'
                }
        
        elif self.mlx_cluster:
            # Fallback to MLX cluster
            result = await self.mlx_cluster.inference_distributed(
                model_name, processed_input
            )
            
            node_results[self.node_id] = {
                'status': 'completed',
                'processing_time': result.get('processing_time', 0),
                'layers_processed': result.get('layers_processed', 'unknown'),
                'memory_usage': result.get('memory_usage', 0),
                'tokens_processed': result.get('tokens_generated', 0),
                'backend': 'mlx'
            }
        
        else:
            # Fallback processing
            node_results[self.node_id] = {
                'status': 'completed',
                'processing_time': 0.1,
                'layers_processed': 'local_fallback',
                'memory_usage': 1.0,
                'tokens_processed': len(processed_input.get('tokenized_input', [])),
                'backend': 'fallback'
            }
        
        return node_results
    
    async def _aggregate_results(self, request: InferenceRequest, 
                               node_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all participating nodes"""
        
        # Simple aggregation for now
        total_tokens = sum(result.get('tokens_processed', 0) for result in node_results.values())
        total_processing_time = max(result.get('processing_time', 0) for result in node_results.values())
        
        # Generate mock response based on prompt
        generated_text = f"Response to: {request.prompt[:50]}..." if len(request.prompt) > 50 else f"Response to: {request.prompt}"
        generated_text += " [Generated by distributed inference cluster]"
        
        return {
            'generated_text': generated_text,
            'tokens_generated': min(request.max_tokens, 100),  # Mock token count
            'total_processing_time': total_processing_time,
            'participating_nodes': list(node_results.keys()),
            'aggregation_method': 'sequential_pipeline'
        }
    
    async def _coordinate_streaming_inference(self, request: InferenceRequest, 
                                            token_queue: asyncio.Queue) -> None:
        """Coordinate streaming inference across nodes"""
        try:
            # Simulate streaming token generation
            mock_response = f"Response to: {request.prompt}"
            tokens = mock_response.split()
            
            for i, token in enumerate(tokens):
                if i >= request.max_tokens:
                    break
                
                # Simulate processing time
                await asyncio.sleep(0.05)
                
                token_data = {
                    'token': token + (" " if i < len(tokens) - 1 else ""),
                    'token_id': i,
                    'is_final': i == len(tokens) - 1 or i == request.max_tokens - 1,
                    'node_id': self.node_id
                }
                
                await token_queue.put(token_data)
            
            # Signal end of stream
            await token_queue.put(None)
            
        except Exception as e:
            logger.error(f"Streaming coordination failed: {e}")
            await token_queue.put(None)
    
    async def _update_request_status(self, request_id: str, status: InferenceStatus) -> None:
        """Update request status and trigger callbacks"""
        with self.request_lock:
            self.request_status[request_id] = status
        
        # Trigger status callback if registered
        if request_id in self.status_callbacks:
            try:
                await self.status_callbacks[request_id](request_id, status)
            except Exception as e:
                logger.error(f"Status callback failed for {request_id}: {e}")
    
    def _update_metrics(self, response: InferenceResponse) -> None:
        """Update performance metrics"""
        self.metrics['total_requests'] += 1
        
        if response.status == InferenceStatus.COMPLETED:
            self.metrics['successful_requests'] += 1
            
            # Update average processing time
            total_successful = self.metrics['successful_requests']
            current_avg = self.metrics['avg_processing_time']
            new_avg = ((current_avg * (total_successful - 1)) + response.processing_time) / total_successful
            self.metrics['avg_processing_time'] = new_avg
            
            # Update token metrics
            self.metrics['total_tokens_generated'] += response.tokens_generated
            if response.processing_time > 0:
                tokens_per_second = response.tokens_generated / response.processing_time
                current_tps_avg = self.metrics['avg_tokens_per_second']
                new_tps_avg = ((current_tps_avg * (total_successful - 1)) + tokens_per_second) / total_successful
                self.metrics['avg_tokens_per_second'] = new_tps_avg
        else:
            self.metrics['failed_requests'] += 1
    
    async def _on_model_loaded_callback(self, model_config: Any) -> None:
        """Callback when a model is loaded"""
        logger.info(f"Model loaded callback: {model_config.name}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            'node_id': self.node_id,
            'initialized': self.initialized,
            'loaded_models': list(self.loaded_models.keys()),
            'active_requests': len(self.active_requests),
            'metrics': self.metrics.copy(),
            'components': {
                'mlx_cluster_available': self.mlx_cluster is not None,
                'exo_cluster_available': self.exo_cluster is not None,
                'model_partitioner_available': self.model_partitioner is not None
            }
        }
    
    def create_inference_request(self, model_name: str, prompt: str, **kwargs) -> InferenceRequest:
        """Create a new inference request"""
        request_id = hashlib.md5(f"{model_name}_{prompt}_{time.time()}".encode()).hexdigest()[:16]
        
        return InferenceRequest(
            request_id=request_id,
            model_name=model_name,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 100),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 40),
            stop_sequences=kwargs.get('stop_sequences', []),
            stream=kwargs.get('stream', False),
            metadata=kwargs.get('metadata', {}),
            created_at=time.time()
        )
    
    async def shutdown(self) -> None:
        """Shutdown the distributed inference engine"""
        logger.info("Shutting down distributed inference engine...")
        
        # Cancel active requests
        with self.request_lock:
            for request_id in list(self.active_requests.keys()):
                self.request_status[request_id] = InferenceStatus.CANCELLED
        
        # Shutdown components
        if self.mlx_cluster:
            await self.mlx_cluster.shutdown()
        
        if self.exo_cluster:
            await self.exo_cluster.cleanup()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.initialized = False
        logger.info("Distributed inference engine shutdown complete")

# Factory function for creating inference engines
def create_inference_engine(node_id: str, config_file: Optional[str] = None) -> DistributedInferenceEngine:
    """Create a distributed inference engine for the specified node"""
    return DistributedInferenceEngine(node_id, config_file)

# Example usage and testing
async def main():
    """Test distributed inference engine"""
    import sys
    
    node_id = sys.argv[1] if len(sys.argv) > 1 else "mac-node-1"
    
    try:
        engine = create_inference_engine(node_id)
        
        # Initialize engine
        if await engine.initialize():
            logger.info("✓ Distributed inference engine initialized")
        else:
            logger.error("✗ Engine initialization failed")
            return
        
        # Test model loading
        model_config = {
            'name': 'test-model',
            'architecture': 'llama',
            'num_layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'vocab_size': 32000,
            'max_sequence_length': 2048
        }
        
        if await engine.load_model('test-model', model_config):
            logger.info("✓ Model loaded successfully")
        else:
            logger.error("✗ Model loading failed")
            return
        
        # Test inference
        request = engine.create_inference_request(
            'test-model', 
            'Hello, how are you?',
            max_tokens=50,
            temperature=0.7
        )
        
        response = await engine.distributed_inference(request)
        logger.info(f"Inference response: {response.to_dict()}")
        
        # Test streaming inference
        logger.info("Testing streaming inference...")
        stream_request = engine.create_inference_request(
            'test-model',
            'Tell me a story',
            max_tokens=20,
            stream=True
        )
        
        async for token in engine.streaming_inference(stream_request):
            print(f"Token: {token.token}", end='', flush=True)
        print()
        
        # Show engine status
        status = engine.get_engine_status()
        logger.info(f"Engine status: {json.dumps(status, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'engine' in locals():
            await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())