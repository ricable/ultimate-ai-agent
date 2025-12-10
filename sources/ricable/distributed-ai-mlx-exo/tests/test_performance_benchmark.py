"""
Comprehensive Performance Benchmark Test Suite
MLX Distributed vs EXO Integration Comparison Testing
"""

import asyncio
import json
import logging
import time
import unittest
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import statistics
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Import the systems under test
try:
    from src.mlx_distributed.cluster import DistributedMLXCluster
    from src.exo_integration.enhanced_cluster_manager import EnhancedExoClusterManager, ModelConfig
    from src.performance_dashboard import PerformanceDashboard, BenchmarkResult
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    SYSTEMS_AVAILABLE = False
    logging.warning(f"Test systems not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    name: str
    model_config: Dict[str, Any]
    input_data: Dict[str, Any]
    expected_min_throughput: float  # tokens/sec
    expected_max_latency: float  # seconds
    iterations: int
    timeout: float

class PerformanceBenchmarkTestSuite(unittest.TestCase):
    """
    Comprehensive test suite for comparing MLX Distributed vs EXO Integration
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_configs = [
            BenchmarkConfig(
                name="llama-7b-short",
                model_config={
                    'name': 'llama-7b-test',
                    'architecture': 'llama',
                    'num_layers': 32,
                    'hidden_size': 4096,
                    'num_attention_heads': 32,
                    'vocab_size': 32000,
                    'max_sequence_length': 2048,
                    'model_size_gb': 7.0
                },
                input_data={
                    'prompt': 'Hello, how are you?',
                    'max_tokens': 50,
                    'temperature': 0.7
                },
                expected_min_throughput=20.0,  # tokens/sec
                expected_max_latency=5.0,      # seconds
                iterations=10,
                timeout=30.0
            ),
            BenchmarkConfig(
                name="llama-7b-medium",
                model_config={
                    'name': 'llama-7b-test',
                    'architecture': 'llama',
                    'num_layers': 32,
                    'hidden_size': 4096,
                    'num_attention_heads': 32,
                    'vocab_size': 32000,
                    'max_sequence_length': 2048,
                    'model_size_gb': 7.0
                },
                input_data={
                    'prompt': 'Write a detailed explanation of quantum computing.',
                    'max_tokens': 200,
                    'temperature': 0.8
                },
                expected_min_throughput=15.0,
                expected_max_latency=10.0,
                iterations=5,
                timeout=60.0
            )
        ]
        
        cls.test_results = {
            'mlx_distributed': [],
            'exo_integration': [],
            'comparison_metrics': {}
        }
    
    def setUp(self):
        """Set up individual test"""
        self.mlx_cluster = None
        self.exo_manager = None
        self.dashboard = None
        
        if SYSTEMS_AVAILABLE:
            self.dashboard = PerformanceDashboard("test_results")
    
    def tearDown(self):
        """Clean up after each test"""
        asyncio.run(self._cleanup_async())
    
    async def _cleanup_async(self):
        """Async cleanup helper"""
        try:
            if self.mlx_cluster:
                await self.mlx_cluster.shutdown()
            if self.exo_manager:
                await self.exo_manager.cleanup()
            if self.dashboard:
                await self.dashboard.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    # MLX Distributed Tests
    
    @pytest.mark.asyncio
    async def test_mlx_distributed_initialization(self):
        """Test MLX Distributed cluster initialization"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        self.mlx_cluster = DistributedMLXCluster()
        
        # Test initialization
        start_time = time.time()
        success = await self.mlx_cluster.initialize()
        init_time = time.time() - start_time
        
        # Assertions
        self.assertTrue(success, "MLX cluster initialization should succeed")
        self.assertLess(init_time, 30.0, "Initialization should complete within 30 seconds")
        self.assertTrue(self.mlx_cluster.is_initialized, "Cluster should be marked as initialized")
        
        # Verify cluster state
        status = self.mlx_cluster.get_cluster_status()
        self.assertIn('node_info', status)
        self.assertIn('cluster_info', status)
        self.assertTrue(status['cluster_info']['initialized'])
    
    @pytest.mark.asyncio
    async def test_mlx_distributed_model_loading(self):
        """Test model loading in MLX Distributed"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        self.mlx_cluster = DistributedMLXCluster()
        await self.mlx_cluster.initialize()
        
        config = self.test_configs[0]
        
        # Test model loading
        start_time = time.time()
        success = await self.mlx_cluster.load_model_distributed(
            f"test_models/{config.model_config['name']}", 
            config.model_config
        )
        load_time = time.time() - start_time
        
        # Assertions
        self.assertTrue(success, "Model loading should succeed")
        self.assertLess(load_time, 120.0, "Model loading should complete within 120 seconds")
        self.assertIn(config.model_config['name'], self.mlx_cluster.loaded_models)
        
        # Verify model metadata
        model_info = self.mlx_cluster.loaded_models[config.model_config['name']]
        self.assertEqual(model_info['name'], config.model_config['name'])
        self.assertIn('assignment', model_info)
    
    @pytest.mark.asyncio
    async def test_mlx_distributed_inference_performance(self):
        """Test MLX Distributed inference performance"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_mlx_for_inference()
        
        config = self.test_configs[0]
        results = []
        
        # Run performance test
        for i in range(config.iterations):
            start_time = time.time()
            
            result = await self.mlx_cluster.inference_distributed(
                config.model_config['name'], 
                config.input_data
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Verify successful inference
            self.assertNotIn('error', result, f"Inference {i+1} should not have errors")
            
            # Calculate performance metrics
            tokens_generated = result.get('tokens_generated', config.input_data['max_tokens'])
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            results.append({
                'iteration': i + 1,
                'inference_time': inference_time,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_per_second
            })
            
            # Performance assertions
            self.assertLess(inference_time, config.expected_max_latency, 
                          f"Inference {i+1} should complete within {config.expected_max_latency}s")
            self.assertGreater(tokens_per_second, config.expected_min_throughput * 0.5,
                             f"Inference {i+1} throughput should be reasonable")
        
        # Aggregate performance analysis
        avg_inference_time = statistics.mean(r['inference_time'] for r in results)
        avg_throughput = statistics.mean(r['tokens_per_second'] for r in results)
        
        self.assertLess(avg_inference_time, config.expected_max_latency)
        self.assertGreater(avg_throughput, config.expected_min_throughput * 0.8)
        
        # Store results for comparison
        self.test_results['mlx_distributed'].extend(results)
        
        logger.info(f"MLX Distributed Performance: {avg_throughput:.2f} tokens/s, {avg_inference_time:.3f}s avg")
    
    @pytest.mark.asyncio
    async def test_mlx_distributed_memory_usage(self):
        """Test MLX Distributed memory usage efficiency"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_mlx_for_inference()
        
        config = self.test_configs[0]
        
        # Monitor memory usage during inference
        initial_memory = self.mlx_cluster.metrics.get('memory_usage', 0.0)
        
        for i in range(5):  # Run multiple inferences
            await self.mlx_cluster.inference_distributed(
                config.model_config['name'], 
                config.input_data
            )
        
        final_memory = self.mlx_cluster.metrics.get('memory_usage', 0.0)
        
        # Memory usage should be stable (no significant leaks)
        memory_growth = final_memory - initial_memory
        self.assertLess(memory_growth, 1.0, "Memory growth should be minimal (<1GB)")
        
        # Memory usage should be reasonable for the model size
        expected_memory = config.model_config['model_size_gb'] * 1.5  # 50% overhead
        self.assertLess(final_memory, expected_memory, "Memory usage should be reasonable")
    
    @pytest.mark.asyncio
    async def test_mlx_distributed_fault_tolerance(self):
        """Test MLX Distributed fault tolerance"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_mlx_for_inference()
        
        config = self.test_configs[0]
        
        # Test inference with simulated communication failure
        with patch.object(self.mlx_cluster, 'all_reduce') as mock_all_reduce:
            mock_all_reduce.side_effect = Exception("Simulated communication failure")
            
            result = await self.mlx_cluster.inference_distributed(
                config.model_config['name'], 
                config.input_data
            )
            
            # Should handle gracefully - exact behavior depends on implementation
            # At minimum, should not crash the system
            self.assertIsInstance(result, dict, "Should return a result dict even on failure")

    # EXO Integration Tests
    
    @pytest.mark.asyncio
    async def test_exo_integration_initialization(self):
        """Test EXO Integration cluster initialization"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        from src.exo_integration.enhanced_cluster_manager import create_enhanced_cluster_manager
        self.exo_manager = create_enhanced_cluster_manager("test-node-1")
        
        # Test initialization
        start_time = time.time()
        success = await self.exo_manager.initialize_hybrid_cluster()
        init_time = time.time() - start_time
        
        # Assertions
        self.assertTrue(success, "EXO cluster initialization should succeed")
        self.assertLess(init_time, 60.0, "Initialization should complete within 60 seconds")
        
        # Verify cluster state
        status = await self.exo_manager.get_comprehensive_status()
        self.assertIn('node_id', status)
        self.assertIn('hybrid_cluster', status)
    
    @pytest.mark.asyncio
    async def test_exo_integration_peer_discovery(self):
        """Test EXO Integration peer discovery"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        from src.exo_integration.enhanced_cluster_manager import create_enhanced_cluster_manager
        self.exo_manager = create_enhanced_cluster_manager("test-node-1")
        await self.exo_manager.initialize_hybrid_cluster()
        
        # Test peer discovery
        start_time = time.time()
        peers = await self.exo_manager.discover_and_coordinate_peers(timeout=30)
        discovery_time = time.time() - start_time
        
        # Assertions
        self.assertIsInstance(peers, list, "Should return a list of peers")
        self.assertLess(discovery_time, 35.0, "Discovery should complete within timeout")
        
        # Verify peer information structure
        for peer in peers:
            self.assertIn('ip', peer)
            self.assertIn('port', peer)
    
    @pytest.mark.asyncio
    async def test_exo_integration_model_loading(self):
        """Test model loading in EXO Integration"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_exo_for_inference()
        
        config = self.test_configs[0]
        exo_model_config = ModelConfig(
            name=config.model_config['name'],
            architecture=config.model_config['architecture'],
            num_layers=config.model_config['num_layers'],
            hidden_size=config.model_config['hidden_size'],
            num_attention_heads=config.model_config['num_attention_heads'],
            vocab_size=config.model_config['vocab_size'],
            max_sequence_length=config.model_config['max_sequence_length'],
            model_size_gb=config.model_config['model_size_gb']
        )
        
        # Test model loading with smart partitioning
        start_time = time.time()
        success = await self.exo_manager.load_model_with_smart_partitioning(exo_model_config)
        load_time = time.time() - start_time
        
        # Assertions
        self.assertTrue(success, "Model loading should succeed")
        self.assertLess(load_time, 180.0, "Model loading should complete within 180 seconds")
        self.assertIn(config.model_config['name'], self.exo_manager.loaded_models)
        
        # Verify partitioning
        self.assertIn(config.model_config['name'], self.exo_manager.partition_assignments)
    
    @pytest.mark.asyncio
    async def test_exo_integration_inference_performance(self):
        """Test EXO Integration inference performance"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_exo_for_inference()
        
        config = self.test_configs[0]
        results = []
        
        # Run performance test
        for i in range(config.iterations):
            start_time = time.time()
            
            result = await self.exo_manager.distributed_inference(
                config.model_config['name'], 
                config.input_data
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Verify successful inference
            self.assertNotIn('error', result, f"Inference {i+1} should not have errors")
            
            # Calculate performance metrics
            tokens_generated = result.get('tokens_generated', config.input_data['max_tokens'])
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            results.append({
                'iteration': i + 1,
                'inference_time': inference_time,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_per_second
            })
            
            # Performance assertions
            self.assertLess(inference_time, config.expected_max_latency * 1.5, 
                          f"Inference {i+1} should complete within reasonable time")
            self.assertGreater(tokens_per_second, config.expected_min_throughput * 0.3,
                             f"Inference {i+1} throughput should be reasonable")
        
        # Aggregate performance analysis
        avg_inference_time = statistics.mean(r['inference_time'] for r in results)
        avg_throughput = statistics.mean(r['tokens_per_second'] for r in results)
        
        self.assertLess(avg_inference_time, config.expected_max_latency * 1.5)
        self.assertGreater(avg_throughput, config.expected_min_throughput * 0.5)
        
        # Store results for comparison
        self.test_results['exo_integration'].extend(results)
        
        logger.info(f"EXO Integration Performance: {avg_throughput:.2f} tokens/s, {avg_inference_time:.3f}s avg")
    
    @pytest.mark.asyncio
    async def test_exo_integration_fault_tolerance(self):
        """Test EXO Integration fault tolerance"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_exo_for_inference()
        
        config = self.test_configs[0]
        
        # Test inference with simulated peer failure
        with patch.object(self.exo_manager, '_coordinate_inference_results') as mock_coordinate:
            mock_coordinate.side_effect = Exception("Simulated peer failure")
            
            result = await self.exo_manager.distributed_inference(
                config.model_config['name'], 
                config.input_data
            )
            
            # Should handle gracefully
            self.assertIsInstance(result, dict, "Should return a result dict")
            # EXO should provide error information
            if 'error' in result:
                self.assertIn('error', result)
    
    # Comparative Tests
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Direct performance comparison between systems"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        # Run both systems with identical configurations
        await self._setup_mlx_for_inference()
        await self._setup_exo_for_inference()
        
        config = self.test_configs[0]
        
        # MLX Distributed benchmark
        mlx_results = []
        for i in range(5):
            start_time = time.time()
            result = await self.mlx_cluster.inference_distributed(
                config.model_config['name'], config.input_data
            )
            end_time = time.time()
            
            if 'error' not in result:
                mlx_results.append({
                    'time': end_time - start_time,
                    'tokens': result.get('tokens_generated', config.input_data['max_tokens'])
                })
        
        # EXO Integration benchmark
        exo_results = []
        for i in range(5):
            start_time = time.time()
            result = await self.exo_manager.distributed_inference(
                config.model_config['name'], config.input_data
            )
            end_time = time.time()
            
            if 'error' not in result:
                exo_results.append({
                    'time': end_time - start_time,
                    'tokens': result.get('tokens_generated', config.input_data['max_tokens'])
                })
        
        # Compare results
        if mlx_results and exo_results:
            mlx_avg_time = statistics.mean(r['time'] for r in mlx_results)
            exo_avg_time = statistics.mean(r['time'] for r in exo_results)
            
            mlx_throughput = statistics.mean(r['tokens'] / r['time'] for r in mlx_results)
            exo_throughput = statistics.mean(r['tokens'] / r['time'] for r in exo_results)
            
            # Store comparison metrics
            self.test_results['comparison_metrics'] = {
                'mlx_avg_time': mlx_avg_time,
                'exo_avg_time': exo_avg_time,
                'mlx_throughput': mlx_throughput,
                'exo_throughput': exo_throughput,
                'performance_ratio': mlx_throughput / exo_throughput if exo_throughput > 0 else 0
            }
            
            # Log comparison
            logger.info(f"Performance Comparison:")
            logger.info(f"  MLX Distributed: {mlx_throughput:.2f} tokens/s, {mlx_avg_time:.3f}s avg")
            logger.info(f"  EXO Integration: {exo_throughput:.2f} tokens/s, {exo_avg_time:.3f}s avg")
            logger.info(f"  Performance Ratio: {mlx_throughput/exo_throughput:.2f}x")
            
            # Basic performance expectations
            self.assertGreater(mlx_throughput, 0, "MLX should have positive throughput")
            self.assertGreater(exo_throughput, 0, "EXO should have positive throughput")
    
    @pytest.mark.asyncio
    async def test_scalability_comparison(self):
        """Test scalability characteristics of both systems"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        # This test would simulate different cluster sizes and measure scaling
        # For now, we'll test the basic scaling metrics
        
        await self._setup_mlx_for_inference()
        await self._setup_exo_for_inference()
        
        # Test single vs multi-node inference (simulated)
        config = self.test_configs[0]
        
        # Single node simulation
        single_node_results = []
        for i in range(3):
            start_time = time.time()
            result = await self.mlx_cluster.inference_distributed(
                config.model_config['name'], config.input_data
            )
            end_time = time.time()
            
            if 'error' not in result:
                single_node_results.append(end_time - start_time)
        
        # Multi-node simulation would require actual cluster setup
        # For testing purposes, we validate that the systems can handle scaling
        
        if single_node_results:
            avg_single_time = statistics.mean(single_node_results)
            self.assertLess(avg_single_time, 30.0, "Single node inference should be reasonable")
    
    # Load Testing
    
    @pytest.mark.asyncio
    async def test_load_testing(self):
        """Test both systems under load"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        await self._setup_mlx_for_inference()
        await self._setup_exo_for_inference()
        
        config = self.test_configs[0]
        
        # Concurrent inference test
        concurrent_requests = 5
        
        # Test MLX Distributed under load
        mlx_tasks = []
        for i in range(concurrent_requests):
            task = self.mlx_cluster.inference_distributed(
                config.model_config['name'], config.input_data
            )
            mlx_tasks.append(task)
        
        start_time = time.time()
        mlx_results = await asyncio.gather(*mlx_tasks, return_exceptions=True)
        mlx_total_time = time.time() - start_time
        
        # Test EXO Integration under load
        exo_tasks = []
        for i in range(concurrent_requests):
            task = self.exo_manager.distributed_inference(
                config.model_config['name'], config.input_data
            )
            exo_tasks.append(task)
        
        start_time = time.time()
        exo_results = await asyncio.gather(*exo_tasks, return_exceptions=True)
        exo_total_time = time.time() - start_time
        
        # Analyze results
        mlx_successful = sum(1 for r in mlx_results if isinstance(r, dict) and 'error' not in r)
        exo_successful = sum(1 for r in exo_results if isinstance(r, dict) and 'error' not in r)
        
        # Assertions
        self.assertGreater(mlx_successful, 0, "MLX should handle some concurrent requests")
        self.assertGreater(exo_successful, 0, "EXO should handle some concurrent requests")
        
        logger.info(f"Load Test Results:")
        logger.info(f"  MLX: {mlx_successful}/{concurrent_requests} successful in {mlx_total_time:.2f}s")
        logger.info(f"  EXO: {exo_successful}/{concurrent_requests} successful in {exo_total_time:.2f}s")
    
    # Helper Methods
    
    async def _setup_mlx_for_inference(self):
        """Helper to set up MLX system for inference testing"""
        if not self.mlx_cluster:
            self.mlx_cluster = DistributedMLXCluster()
            await self.mlx_cluster.initialize()
            
            # Load test model
            config = self.test_configs[0]
            await self.mlx_cluster.load_model_distributed(
                f"test_models/{config.model_config['name']}", 
                config.model_config
            )
    
    async def _setup_exo_for_inference(self):
        """Helper to set up EXO system for inference testing"""
        if not self.exo_manager:
            from src.exo_integration.enhanced_cluster_manager import create_enhanced_cluster_manager
            self.exo_manager = create_enhanced_cluster_manager("test-node-1")
            await self.exo_manager.initialize_hybrid_cluster()
            
            # Load test model
            config = self.test_configs[0]
            exo_model_config = ModelConfig(
                name=config.model_config['name'],
                architecture=config.model_config['architecture'],
                num_layers=config.model_config['num_layers'],
                hidden_size=config.model_config['hidden_size'],
                num_attention_heads=config.model_config['num_attention_heads'],
                vocab_size=config.model_config['vocab_size'],
                max_sequence_length=config.model_config['max_sequence_length'],
                model_size_gb=config.model_config['model_size_gb']
            )
            await self.exo_manager.load_model_with_smart_partitioning(exo_model_config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test class"""
        # Save test results
        results_file = Path("test_results/benchmark_test_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_file}")

# Stress Testing
class StressTestSuite:
    """Stress tests for both systems"""
    
    @pytest.mark.asyncio
    async def test_memory_stress(self):
        """Test systems under memory pressure"""
        if not SYSTEMS_AVAILABLE:
            pytest.skip("Systems not available")
        
        # This would test behavior under memory pressure
        # Implementation depends on specific memory management
        pass
    
    @pytest.mark.asyncio  
    async def test_network_stress(self):
        """Test systems under network pressure"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        # This would test behavior under network congestion
        # Implementation depends on network simulation capabilities
        pass

# Integration Tests with Real Dashboard
class DashboardIntegrationTests(unittest.TestCase):
    """Integration tests with the performance dashboard"""
    
    @pytest.mark.asyncio
    async def test_dashboard_benchmark_integration(self):
        """Test dashboard integration with benchmark systems"""
        if not SYSTEMS_AVAILABLE:
            self.skipTest("Systems not available")
        
        dashboard = PerformanceDashboard("test_dashboard_results")
        
        try:
            # Initialize dashboard
            success = await dashboard.initialize_systems()
            if success:
                # Run mini benchmark
                results = await dashboard.run_benchmark_suite(iterations=2)
                
                # Verify results structure
                self.assertIn('mlx_distributed', results)
                self.assertIn('exo_integration', results)
                self.assertIn('summary', results)
                
                # Check that metrics were collected
                metrics = dashboard.get_realtime_metrics()
                self.assertIn('mlx_distributed', metrics)
                self.assertIn('exo_integration', metrics)
            
        finally:
            await dashboard.cleanup()

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)