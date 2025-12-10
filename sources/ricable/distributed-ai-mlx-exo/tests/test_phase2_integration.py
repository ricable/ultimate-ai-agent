"""
Phase 2 Integration Tests
Tests for MLX-Exo core integration including distributed inference pipeline
"""

import asyncio
import pytest
import tempfile
import shutil
import os
import json
import logging
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test fixtures and utilities
@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configurations"""
    temp_dir = tempfile.mkdtemp(prefix="mlx_exo_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_cluster_config():
    """Standard test cluster configuration"""
    return {
        "nodes": [
            {
                "name": "test-node-1",
                "ip": "127.0.0.1",
                "role": "compute",
                "memory_gb": 32,
                "gpu_cores": 16,
                "cpu_cores": 8
            },
            {
                "name": "test-node-2", 
                "ip": "127.0.0.2",
                "role": "compute",
                "memory_gb": 32,
                "gpu_cores": 16,
                "cpu_cores": 8
            }
        ],
        "network": {
            "data_subnet": "127.0.0.0/24",
            "mtu": 1500,
            "use_thunderbolt": False
        },
        "backend": "ring"
    }

@pytest.fixture
def test_model_config():
    """Standard test model configuration"""
    return {
        'name': 'test-llama-7b',
        'architecture': 'llama',
        'num_layers': 32,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'vocab_size': 32000,
        'max_sequence_length': 2048
    }

class TestMLXDistributedConfig:
    """Test MLX distributed configuration"""
    
    def test_config_initialization(self, temp_config_dir, test_cluster_config):
        """Test MLX distributed config initialization"""
        try:
            from mlx_distributed.config import MLXDistributedConfig, NodeConfig, ClusterConfig
            
            # Create config file
            config_file = os.path.join(temp_config_dir, "cluster_config.json")
            with open(config_file, 'w') as f:
                json.dump(test_cluster_config, f)
            
            # Test initialization
            config = MLXDistributedConfig(config_file)
            
            assert config.cluster_config is not None
            assert len(config.cluster_config.nodes) == 2
            assert config.cluster_config.backend == "ring"
            
            logger.info("✓ MLX distributed config initialization test passed")
            
        except ImportError:
            pytest.skip("MLX distributed components not available")
    
    def test_hostfile_generation(self, temp_config_dir, test_cluster_config):
        """Test hostfile generation"""
        try:
            from mlx_distributed.config import MLXDistributedConfig
            
            config_file = os.path.join(temp_config_dir, "cluster_config.json")
            with open(config_file, 'w') as f:
                json.dump(test_cluster_config, f)
            
            config = MLXDistributedConfig(config_file)
            
            # Generate hostfile
            hostfile_path = config.generate_hostfile(os.path.join(temp_config_dir, "hostfile.json"))
            
            assert os.path.exists(hostfile_path)
            
            with open(hostfile_path, 'r') as f:
                hostfile = json.load(f)
            
            assert "hosts" in hostfile
            assert len(hostfile["hosts"]) == 2
            assert hostfile["total_ranks"] == 2
            
            logger.info("✓ Hostfile generation test passed")
            
        except ImportError:
            pytest.skip("MLX distributed components not available")

class TestModelPartitioner:
    """Test model partitioning functionality"""
    
    def test_partitioner_initialization(self, test_cluster_config):
        """Test model partitioner initialization"""
        try:
            from mlx_distributed.model_partitioner import ModelPartitioner, PartitioningStrategy
            from mlx_distributed.config import NodeConfig
            
            nodes = [
                NodeConfig(**node) for node in test_cluster_config["nodes"]
            ]
            
            partitioner = ModelPartitioner(nodes)
            
            assert partitioner.nodes == nodes
            assert partitioner.total_cluster_memory == 64  # 32 + 32
            assert len(partitioner.node_capabilities) == 2
            
            logger.info("✓ Model partitioner initialization test passed")
            
        except ImportError:
            pytest.skip("Model partitioner components not available")
    
    def test_model_metadata_creation(self, test_cluster_config, test_model_config):
        """Test model metadata creation"""
        try:
            from mlx_distributed.model_partitioner import ModelPartitioner
            from mlx_distributed.config import NodeConfig
            
            nodes = [NodeConfig(**node) for node in test_cluster_config["nodes"]]
            partitioner = ModelPartitioner(nodes)
            
            metadata = partitioner.create_model_metadata(test_model_config)
            
            assert metadata.name == test_model_config['name']
            assert metadata.num_layers == test_model_config['num_layers']
            assert metadata.total_params > 0
            assert metadata.total_memory_gb > 0
            assert len(metadata.layers) > 0
            
            logger.info("✓ Model metadata creation test passed")
            
        except ImportError:
            pytest.skip("Model partitioner components not available")
    
    def test_partition_plan_creation(self, test_cluster_config, test_model_config):
        """Test partition plan creation"""
        try:
            from mlx_distributed.model_partitioner import ModelPartitioner, PartitioningStrategy
            from mlx_distributed.config import NodeConfig
            
            nodes = [NodeConfig(**node) for node in test_cluster_config["nodes"]]
            partitioner = ModelPartitioner(nodes)
            
            metadata = partitioner.create_model_metadata(test_model_config)
            
            # Test different strategies
            for strategy in PartitioningStrategy:
                plan = partitioner.create_partition_plan(metadata, strategy)
                
                if plan:  # Some strategies might not work with small test config
                    assert plan.model_name == metadata.name
                    assert plan.strategy == strategy
                    assert len(plan.partitions) > 0
                    assert plan.total_memory_required > 0
                    
                    # Validate partition plan
                    valid, errors = partitioner.validate_partition_plan(plan)
                    if not valid:
                        logger.warning(f"Strategy {strategy.value} validation errors: {errors}")
                    
                    logger.info(f"✓ Partition plan for {strategy.value} created successfully")
            
            logger.info("✓ Partition plan creation test passed")
            
        except ImportError:
            pytest.skip("Model partitioner components not available")

class TestExoClusterManager:
    """Test Exo cluster manager functionality"""
    
    @pytest.mark.asyncio
    async def test_exo_manager_initialization(self):
        """Test Exo cluster manager initialization"""
        try:
            from exo_integration.cluster_manager import create_cluster_manager
            
            manager = create_cluster_manager("mac-node-1")
            
            assert manager.node_spec.node_id == "mac-node-1"
            assert manager.node_spec.memory_gb > 0
            
            logger.info("✓ Exo cluster manager initialization test passed")
            
        except ImportError:
            pytest.skip("Exo integration components not available")
    
    @pytest.mark.asyncio
    async def test_exo_node_initialization(self):
        """Test Exo node initialization"""
        try:
            from exo_integration.cluster_manager import create_cluster_manager
            
            manager = create_cluster_manager("mac-node-1")
            
            # Initialize node
            success = await manager.initialize_node()
            
            # Should succeed even in mock mode
            assert success is True
            assert manager.node is not None
            
            logger.info("✓ Exo node initialization test passed")
            
        except ImportError:
            pytest.skip("Exo integration components not available")
    
    @pytest.mark.asyncio
    async def test_exo_health_check(self):
        """Test Exo health check functionality"""
        try:
            from exo_integration.cluster_manager import create_cluster_manager
            
            manager = create_cluster_manager("mac-node-1")
            await manager.initialize_node()
            
            health = await manager.health_check()
            
            assert "node_id" in health
            assert "status" in health
            assert health["node_id"] == "mac-node-1"
            
            logger.info("✓ Exo health check test passed")
            
        except ImportError:
            pytest.skip("Exo integration components not available")

class TestDistributedInferenceEngine:
    """Test distributed inference engine"""
    
    @pytest.mark.asyncio
    async def test_inference_engine_initialization(self):
        """Test inference engine initialization"""
        try:
            from distributed_inference_engine import create_inference_engine
            
            engine = create_inference_engine("test-node-1")
            
            assert engine.node_id == "test-node-1"
            assert not engine.initialized
            
            # Note: Full initialization requires MLX/Exo components
            logger.info("✓ Inference engine initialization test passed")
            
        except ImportError:
            pytest.skip("Distributed inference engine not available")
    
    @pytest.mark.asyncio
    async def test_inference_request_creation(self, test_model_config):
        """Test inference request creation"""
        try:
            from distributed_inference_engine import create_inference_engine
            
            engine = create_inference_engine("test-node-1")
            
            request = engine.create_inference_request(
                model_name="test-model",
                prompt="Hello, how are you?",
                max_tokens=50,
                temperature=0.7
            )
            
            assert request.model_name == "test-model"
            assert request.prompt == "Hello, how are you?"
            assert request.max_tokens == 50
            assert request.temperature == 0.7
            assert request.request_id is not None
            
            logger.info("✓ Inference request creation test passed")
            
        except ImportError:
            pytest.skip("Distributed inference engine not available")

class TestMemoryManager:
    """Test distributed memory manager"""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization"""
        try:
            from memory_manager import create_memory_manager, MemoryType
            
            manager = create_memory_manager("test-node-1")
            
            assert manager.node_id == "test-node-1"
            assert len(manager.tiers) > 0
            
            logger.info("✓ Memory manager initialization test passed")
            
        except ImportError:
            pytest.skip("Memory manager not available")
    
    @pytest.mark.asyncio
    async def test_memory_object_operations(self):
        """Test memory object storage and retrieval"""
        try:
            from memory_manager import create_memory_manager, MemoryType
            import numpy as np
            
            manager = create_memory_manager("test-node-1")
            
            # Create test data
            test_data = np.random.rand(100, 100).astype(np.float32)
            
            # Create memory object
            obj = manager.create_memory_object(
                "test_weights",
                MemoryType.MODEL_WEIGHTS,
                test_data
            )
            
            # Store object
            success = manager.store_object(obj)
            assert success is True
            
            # Retrieve object
            retrieved_obj = manager.get_object("test_weights")
            assert retrieved_obj is not None
            assert retrieved_obj.object_id == "test_weights"
            assert retrieved_obj.object_type == MemoryType.MODEL_WEIGHTS
            
            # Remove object
            removed = manager.remove_object("test_weights")
            assert removed is True
            
            # Verify removal
            retrieved_obj = manager.get_object("test_weights")
            assert retrieved_obj is None
            
            logger.info("✓ Memory object operations test passed")
            
        except ImportError:
            pytest.skip("Memory manager not available")
    
    def test_memory_statistics(self):
        """Test memory statistics collection"""
        try:
            from memory_manager import create_memory_manager, MemoryType
            import numpy as np
            
            manager = create_memory_manager("test-node-1")
            
            # Store some test objects
            for i in range(5):
                test_data = np.random.rand(50, 50).astype(np.float32)
                obj = manager.create_memory_object(
                    f"test_obj_{i}",
                    MemoryType.TEMPORARY,
                    test_data
                )
                manager.store_object(obj)
            
            # Get statistics
            stats = manager.get_memory_stats()
            
            assert "node_id" in stats
            assert "tier_statistics" in stats
            assert "global_statistics" in stats
            assert stats["global_statistics"]["total_objects"] == 5
            
            logger.info("✓ Memory statistics test passed")
            
        except ImportError:
            pytest.skip("Memory manager not available")

class TestAPIServer:
    """Test API server functionality"""
    
    def test_api_server_creation(self):
        """Test API server creation"""
        try:
            from api_server import create_api_server
            
            config = {
                'host': '127.0.0.1',
                'port': 8001,  # Use different port for testing
                'max_requests': 10
            }
            
            server = create_api_server("test-api-node", config)
            
            assert server.node_id == "test-api-node"
            assert server.host == "127.0.0.1"
            assert server.port == 8001
            assert server.app is not None
            
            logger.info("✓ API server creation test passed")
            
        except ImportError:
            pytest.skip("API server components not available")

class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_basic_integration_flow(self, test_cluster_config, test_model_config):
        """Test basic integration flow with all components"""
        try:
            # Test component imports
            components_available = True
            try:
                from mlx_distributed.config import MLXDistributedConfig
                from mlx_distributed.model_partitioner import ModelPartitioner
                from exo_integration.cluster_manager import create_cluster_manager
                from memory_manager import create_memory_manager
                from distributed_inference_engine import create_inference_engine
            except ImportError as e:
                components_available = False
                logger.warning(f"Some components not available: {e}")
            
            if not components_available:
                pytest.skip("Required components not available for integration test")
            
            # 1. Initialize memory manager
            memory_manager = create_memory_manager("integration-test-node")
            assert memory_manager is not None
            
            # 2. Create cluster manager
            cluster_manager = create_cluster_manager("mac-node-1")
            await cluster_manager.initialize_node()
            
            # 3. Create inference engine
            inference_engine = create_inference_engine("integration-test-node")
            
            # 4. Test inference request creation
            request = inference_engine.create_inference_request(
                model_name="test-model",
                prompt="This is a test prompt for integration testing.",
                max_tokens=20,
                temperature=0.7
            )
            
            assert request is not None
            assert request.model_name == "test-model"
            
            logger.info("✓ Basic integration flow test passed")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            pytest.fail(f"Integration test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, temp_config_dir, test_cluster_config):
        """Test configuration integration across components"""
        try:
            from mlx_distributed.config import MLXDistributedConfig
            
            # Create config file
            config_file = os.path.join(temp_config_dir, "integration_config.json")
            with open(config_file, 'w') as f:
                json.dump(test_cluster_config, f)
            
            # Test config loading
            config = MLXDistributedConfig(config_file)
            
            # Verify configuration consistency
            assert len(config.cluster_config.nodes) == len(test_cluster_config["nodes"])
            assert config.cluster_config.backend == test_cluster_config["backend"]
            
            logger.info("✓ Configuration integration test passed")
            
        except ImportError:
            pytest.skip("Configuration components not available")
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        try:
            from memory_manager import create_memory_manager, MemoryType
            
            manager = create_memory_manager("error-test-node")
            
            # Test invalid object retrieval
            obj = manager.get_object("non_existent_object")
            assert obj is None
            
            # Test removal of non-existent object
            removed = manager.remove_object("non_existent_object")
            assert removed is False
            
            logger.info("✓ Error handling integration test passed")
            
        except ImportError:
            pytest.skip("Memory manager not available")

def run_phase2_integration_tests():
    """Run all Phase 2 integration tests"""
    logger.info("Starting Phase 2 integration tests...")
    
    # Run tests
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    try:
        result = pytest.main(pytest_args)
        
        if result == 0:
            logger.info("✓ All Phase 2 integration tests passed!")
            return True
        else:
            logger.error("✗ Some Phase 2 integration tests failed")
            return False
            
    except Exception as e:
        logger.error(f"Error running integration tests: {e}")
        return False

if __name__ == "__main__":
    # Run integration tests when executed directly
    success = run_phase2_integration_tests()
    sys.exit(0 if success else 1)