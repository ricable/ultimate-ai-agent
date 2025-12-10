"""
Performance Tracking Dashboard for MLX Distributed vs EXO Integration Comparison
Real-time monitoring and benchmarking dashboard for distributed AI inference systems
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from pathlib import Path
import statistics

try:
    from .mlx_distributed.cluster import DistributedMLXCluster
    from .exo_integration.enhanced_cluster_manager import EnhancedExoClusterManager, ModelConfig
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    system: str  # 'mlx_distributed' or 'exo_integration'
    model_name: str
    inference_id: str
    timestamp: float
    total_time: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage_gb: float
    node_count: int
    communication_overhead: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system': self.system,
            'model_name': self.model_name,
            'inference_id': self.inference_id,
            'timestamp': self.timestamp,
            'total_time': self.total_time,
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': self.tokens_per_second,
            'memory_usage_gb': self.memory_usage_gb,
            'node_count': self.node_count,
            'communication_overhead': self.communication_overhead,
            'error': self.error
        }

@dataclass
class SystemMetrics:
    """System performance metrics"""
    system_name: str
    total_inferences: int = 0
    avg_inference_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_memory_usage: float = 0.0
    success_rate: float = 0.0
    uptime_seconds: float = 0.0
    recent_results: List[BenchmarkResult] = field(default_factory=list)
    
    def update_with_result(self, result: BenchmarkResult):
        """Update metrics with new benchmark result"""
        self.recent_results.append(result)
        
        # Keep only last 100 results
        if len(self.recent_results) > 100:
            self.recent_results = self.recent_results[-100:]
        
        # Recalculate averages
        successful_results = [r for r in self.recent_results if r.error is None]
        if successful_results:
            self.total_inferences = len(self.recent_results)
            self.avg_inference_time = statistics.mean(r.total_time for r in successful_results)
            self.avg_tokens_per_second = statistics.mean(r.tokens_per_second for r in successful_results)
            self.avg_memory_usage = statistics.mean(r.memory_usage_gb for r in successful_results)
            self.success_rate = len(successful_results) / len(self.recent_results) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system_name': self.system_name,
            'total_inferences': self.total_inferences,
            'avg_inference_time': self.avg_inference_time,
            'avg_tokens_per_second': self.avg_tokens_per_second,
            'avg_memory_usage': self.avg_memory_usage,
            'success_rate': self.success_rate,
            'uptime_seconds': self.uptime_seconds
        }

class PerformanceDashboard:
    """
    Performance tracking dashboard for MLX Distributed vs EXO Integration comparison
    Provides real-time metrics, benchmarking, and detailed analysis
    """
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # System instances
        self.mlx_cluster: Optional[DistributedMLXCluster] = None
        self.exo_manager: Optional[EnhancedExoClusterManager] = None
        
        # Performance tracking
        self.mlx_metrics = SystemMetrics("MLX Distributed")
        self.exo_metrics = SystemMetrics("EXO Integration")
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Dashboard state
        self.is_running = False
        self.start_time = time.time()
        self.dashboard_thread: Optional[threading.Thread] = None
        
        # Benchmark configurations
        self.benchmark_configs = self._create_benchmark_configs()
        
        logger.info(f"Performance dashboard initialized - output dir: {self.output_dir}")
    
    def _create_benchmark_configs(self) -> List[Dict[str, Any]]:
        """Create benchmark test configurations"""
        return [
            {
                'name': 'llama-7b-short',
                'model_config': {
                    'name': 'llama-7b',
                    'architecture': 'llama',
                    'num_layers': 32,
                    'hidden_size': 4096,
                    'num_attention_heads': 32,
                    'vocab_size': 32000,
                    'max_sequence_length': 2048,
                    'model_size_gb': 7.0
                },
                'input_data': {
                    'prompt': 'Hello, how are you today?',
                    'max_tokens': 50,
                    'temperature': 0.7
                }
            },
            {
                'name': 'llama-7b-medium',
                'model_config': {
                    'name': 'llama-7b',
                    'architecture': 'llama',
                    'num_layers': 32,
                    'hidden_size': 4096,
                    'num_attention_heads': 32,
                    'vocab_size': 32000,
                    'max_sequence_length': 2048,
                    'model_size_gb': 7.0
                },
                'input_data': {
                    'prompt': 'Write a detailed explanation of machine learning concepts.',
                    'max_tokens': 200,
                    'temperature': 0.8
                }
            },
            {
                'name': 'llama-13b-short',
                'model_config': {
                    'name': 'llama-13b',
                    'architecture': 'llama',
                    'num_layers': 40,
                    'hidden_size': 5120,
                    'num_attention_heads': 40,
                    'vocab_size': 32000,
                    'max_sequence_length': 2048,
                    'model_size_gb': 13.0
                },
                'input_data': {
                    'prompt': 'Explain quantum computing in simple terms.',
                    'max_tokens': 100,
                    'temperature': 0.7
                }
            }
        ]
    
    async def initialize_systems(self) -> bool:
        """Initialize both MLX and EXO systems"""
        logger.info("Initializing benchmark systems...")
        
        success = True
        
        # Initialize MLX Distributed
        if MLX_AVAILABLE:
            try:
                self.mlx_cluster = DistributedMLXCluster()
                mlx_init = await self.mlx_cluster.initialize()
                if mlx_init:
                    logger.info("✓ MLX Distributed system initialized")
                else:
                    logger.warning("✗ MLX Distributed initialization failed")
                    success = False
            except Exception as e:
                logger.error(f"MLX Distributed initialization error: {e}")
                success = False
        else:
            logger.warning("MLX Distributed not available")
        
        # Initialize EXO Integration
        try:
            from .exo_integration.enhanced_cluster_manager import create_enhanced_cluster_manager
            self.exo_manager = create_enhanced_cluster_manager("mac-node-1")
            exo_init = await self.exo_manager.initialize_hybrid_cluster()
            if exo_init:
                logger.info("✓ EXO Integration system initialized")
            else:
                logger.warning("✗ EXO Integration initialization failed")
                success = False
        except Exception as e:
            logger.error(f"EXO Integration initialization error: {e}")
            success = False
        
        return success
    
    async def run_benchmark_suite(self, iterations: int = 10) -> Dict[str, Any]:
        """Run comprehensive benchmark suite comparing both systems"""
        logger.info(f"Starting benchmark suite with {iterations} iterations")
        
        results = {
            'mlx_distributed': [],
            'exo_integration': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for config in self.benchmark_configs:
            logger.info(f"Benchmarking {config['name']}...")
            
            # Run MLX Distributed benchmarks
            if self.mlx_cluster:
                mlx_results = await self._benchmark_mlx_system(config, iterations)
                results['mlx_distributed'].extend(mlx_results)
                
                # Update metrics
                for result in mlx_results:
                    self.mlx_metrics.update_with_result(result)
            
            # Run EXO Integration benchmarks
            if self.exo_manager:
                exo_results = await self._benchmark_exo_system(config, iterations)
                results['exo_integration'].extend(exo_results)
                
                # Update metrics
                for result in exo_results:
                    self.exo_metrics.update_with_result(result)
        
        # Generate summary
        results['summary'] = self._generate_benchmark_summary()
        
        # Save results
        await self._save_benchmark_results(results)
        
        return results
    
    async def _benchmark_mlx_system(self, config: Dict[str, Any], iterations: int) -> List[BenchmarkResult]:
        """Benchmark MLX Distributed system"""
        results = []
        model_config = config['model_config']
        input_data = config['input_data']
        
        try:
            # Load model
            if not await self.mlx_cluster.load_model_distributed("models/" + model_config['name'], model_config):
                logger.error(f"Failed to load model {model_config['name']} in MLX system")
                return results
            
            # Run inference iterations
            for i in range(iterations):
                start_time = time.time()
                
                try:
                    result = await self.mlx_cluster.inference_distributed(model_config['name'], input_data)
                    
                    if 'error' not in result:
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        tokens_generated = result.get('tokens_generated', input_data['max_tokens'])
                        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
                        
                        benchmark_result = BenchmarkResult(
                            system='mlx_distributed',
                            model_name=model_config['name'],
                            inference_id=f"mlx_{i}_{int(time.time())}",
                            timestamp=start_time,
                            total_time=total_time,
                            tokens_generated=tokens_generated,
                            tokens_per_second=tokens_per_second,
                            memory_usage_gb=self.mlx_cluster.metrics.get('memory_usage', 0.0),
                            node_count=self.mlx_cluster.world_size,
                            communication_overhead=result.get('communication_time', 0.0)
                        )
                        
                        results.append(benchmark_result)
                        self.benchmark_results.append(benchmark_result)
                        
                        logger.debug(f"MLX iteration {i+1}: {total_time:.3f}s, {tokens_per_second:.2f} tokens/s")
                    else:
                        # Record error
                        error_result = BenchmarkResult(
                            system='mlx_distributed',
                            model_name=model_config['name'],
                            inference_id=f"mlx_{i}_{int(time.time())}",
                            timestamp=start_time,
                            total_time=0.0,
                            tokens_generated=0,
                            tokens_per_second=0.0,
                            memory_usage_gb=0.0,
                            node_count=self.mlx_cluster.world_size,
                            communication_overhead=0.0,
                            error=result.get('error', 'Unknown error')
                        )
                        results.append(error_result)
                        self.benchmark_results.append(error_result)
                
                except Exception as e:
                    logger.error(f"MLX benchmark iteration {i+1} failed: {e}")
                    error_result = BenchmarkResult(
                        system='mlx_distributed',
                        model_name=model_config['name'],
                        inference_id=f"mlx_{i}_{int(time.time())}",
                        timestamp=start_time,
                        total_time=0.0,
                        tokens_generated=0,
                        tokens_per_second=0.0,
                        memory_usage_gb=0.0,
                        node_count=self.mlx_cluster.world_size,
                        communication_overhead=0.0,
                        error=str(e)
                    )
                    results.append(error_result)
                    self.benchmark_results.append(error_result)
                
                # Small delay between iterations
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"MLX system benchmark failed: {e}")
        
        return results
    
    async def _benchmark_exo_system(self, config: Dict[str, Any], iterations: int) -> List[BenchmarkResult]:
        """Benchmark EXO Integration system"""
        results = []
        model_config_data = config['model_config']
        input_data = config['input_data']
        
        try:
            # Create EXO model config
            exo_model_config = ModelConfig(
                name=model_config_data['name'],
                architecture=model_config_data['architecture'],
                num_layers=model_config_data['num_layers'],
                hidden_size=model_config_data['hidden_size'],
                num_attention_heads=model_config_data['num_attention_heads'],
                vocab_size=model_config_data['vocab_size'],
                max_sequence_length=model_config_data['max_sequence_length'],
                model_size_gb=model_config_data['model_size_gb']
            )
            
            # Load model
            if not await self.exo_manager.load_model_with_smart_partitioning(exo_model_config):
                logger.error(f"Failed to load model {model_config_data['name']} in EXO system")
                return results
            
            # Run inference iterations
            for i in range(iterations):
                start_time = time.time()
                
                try:
                    result = await self.exo_manager.distributed_inference(model_config_data['name'], input_data)
                    
                    if 'error' not in result:
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        tokens_generated = result.get('tokens_generated', input_data['max_tokens'])
                        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
                        
                        benchmark_result = BenchmarkResult(
                            system='exo_integration',
                            model_name=model_config_data['name'],
                            inference_id=f"exo_{i}_{int(time.time())}",
                            timestamp=start_time,
                            total_time=total_time,
                            tokens_generated=tokens_generated,
                            tokens_per_second=tokens_per_second,
                            memory_usage_gb=self.exo_manager.metrics.get('total_memory_usage', 0.0),
                            node_count=len(self.exo_manager.peer_capabilities) + 1,
                            communication_overhead=result.get('coordination_time', 0.0)
                        )
                        
                        results.append(benchmark_result)
                        self.benchmark_results.append(benchmark_result)
                        
                        logger.debug(f"EXO iteration {i+1}: {total_time:.3f}s, {tokens_per_second:.2f} tokens/s")
                    else:
                        # Record error
                        error_result = BenchmarkResult(
                            system='exo_integration',
                            model_name=model_config_data['name'],
                            inference_id=f"exo_{i}_{int(time.time())}",
                            timestamp=start_time,
                            total_time=0.0,
                            tokens_generated=0,
                            tokens_per_second=0.0,
                            memory_usage_gb=0.0,
                            node_count=len(self.exo_manager.peer_capabilities) + 1,
                            communication_overhead=0.0,
                            error=result.get('error', 'Unknown error')
                        )
                        results.append(error_result)
                        self.benchmark_results.append(error_result)
                
                except Exception as e:
                    logger.error(f"EXO benchmark iteration {i+1} failed: {e}")
                    error_result = BenchmarkResult(
                        system='exo_integration',
                        model_name=model_config_data['name'],
                        inference_id=f"exo_{i}_{int(time.time())}",
                        timestamp=start_time,
                        total_time=0.0,
                        tokens_generated=0,
                        tokens_per_second=0.0,
                        memory_usage_gb=0.0,
                        node_count=len(self.exo_manager.peer_capabilities) + 1,
                        communication_overhead=0.0,
                        error=str(e)
                    )
                    results.append(error_result)
                    self.benchmark_results.append(error_result)
                
                # Small delay between iterations
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"EXO system benchmark failed: {e}")
        
        return results
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary"""
        summary = {
            'mlx_distributed': self.mlx_metrics.to_dict(),
            'exo_integration': self.exo_metrics.to_dict(),
            'comparison': {},
            'recommendations': []
        }
        
        # Generate comparison metrics
        if self.mlx_metrics.total_inferences > 0 and self.exo_metrics.total_inferences > 0:
            comparison = {
                'inference_time_ratio': self.exo_metrics.avg_inference_time / self.mlx_metrics.avg_inference_time,
                'throughput_ratio': self.mlx_metrics.avg_tokens_per_second / self.exo_metrics.avg_tokens_per_second,
                'memory_efficiency_ratio': self.mlx_metrics.avg_memory_usage / self.exo_metrics.avg_memory_usage,
                'reliability_difference': self.mlx_metrics.success_rate - self.exo_metrics.success_rate
            }
            summary['comparison'] = comparison
            
            # Generate recommendations
            recommendations = []
            if comparison['throughput_ratio'] > 1.2:
                recommendations.append("MLX Distributed shows 20%+ better throughput - recommended for high-performance inference")
            elif comparison['throughput_ratio'] < 0.8:
                recommendations.append("EXO Integration shows 20%+ better throughput - recommended for distributed workloads")
            
            if comparison['memory_efficiency_ratio'] < 0.8:
                recommendations.append("MLX Distributed is more memory efficient - recommended for memory-constrained environments")
            elif comparison['memory_efficiency_ratio'] > 1.2:
                recommendations.append("EXO Integration is more memory efficient - suitable for larger models")
            
            if abs(comparison['reliability_difference']) > 10:
                if comparison['reliability_difference'] > 0:
                    recommendations.append("MLX Distributed shows higher reliability - recommended for production deployments")
                else:
                    recommendations.append("EXO Integration shows higher reliability - recommended for production deployments")
            
            summary['recommendations'] = recommendations
        
        return summary
    
    async def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {json_file}")
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        return {
            'mlx_distributed': self.mlx_metrics.to_dict(),
            'exo_integration': self.exo_metrics.to_dict(),
            'dashboard': {
                'uptime_seconds': time.time() - self.start_time,
                'total_benchmarks': len(self.benchmark_results),
                'is_running': self.is_running
            }
        }
    
    def start_realtime_monitoring(self, interval: int = 30) -> None:
        """Start real-time monitoring thread"""
        if self.is_running:
            logger.warning("Real-time monitoring already running")
            return
        
        self.is_running = True
        self.dashboard_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.dashboard_thread.start()
        logger.info(f"Real-time monitoring started with {interval}s interval")
    
    def _monitoring_loop(self, interval: int) -> None:
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Update system metrics
                if self.mlx_cluster:
                    self.mlx_metrics.uptime_seconds = time.time() - self.start_time
                
                if self.exo_manager:
                    self.exo_metrics.uptime_seconds = time.time() - self.start_time
                
                # Save periodic metrics
                metrics = self.get_realtime_metrics()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metrics_file = self.output_dir / f"realtime_metrics_{timestamp}.json"
                
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def stop_realtime_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5)
        logger.info("Real-time monitoring stopped")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up performance dashboard...")
        
        self.stop_realtime_monitoring()
        
        if self.mlx_cluster:
            await self.mlx_cluster.shutdown()
        
        if self.exo_manager:
            await self.exo_manager.cleanup()
        
        logger.info("Performance dashboard cleanup complete")

# CLI interface for running benchmarks
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Dashboard for MLX vs EXO Comparison")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--output-dir", default="performance_reports", help="Output directory for reports")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = PerformanceDashboard(args.output_dir)
    
    try:
        # Initialize systems
        if not await dashboard.initialize_systems():
            logger.error("Failed to initialize benchmark systems")
            return
        
        # Start real-time monitoring if requested
        if args.realtime:
            dashboard.start_realtime_monitoring(args.interval)
        
        # Run benchmark suite
        results = await dashboard.run_benchmark_suite(args.iterations)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        summary = results['summary']
        print(f"MLX Distributed: {summary['mlx_distributed']['avg_tokens_per_second']:.2f} tokens/s avg")
        print(f"EXO Integration: {summary['exo_integration']['avg_tokens_per_second']:.2f} tokens/s avg")
        
        if 'comparison' in summary:
            comp = summary['comparison']
            print(f"Throughput Ratio (MLX/EXO): {comp['throughput_ratio']:.2f}")
            print(f"Memory Efficiency Ratio (MLX/EXO): {comp['memory_efficiency_ratio']:.2f}")
        
        if summary.get('recommendations'):
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("="*60)
        
        # Keep monitoring running if requested
        if args.realtime:
            print("Real-time monitoring active. Press Ctrl+C to stop...")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        await dashboard.cleanup()

if __name__ == "__main__":
    asyncio.run(main())