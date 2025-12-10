#!/usr/bin/env python3
"""
GPU Optimization Benchmark for SAFLA on Fly.io
This script runs comprehensive benchmarks to validate GPU optimization performance.
"""

import time
import torch
import psutil
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import logging
from pathlib import Path

# SAFLA imports
from safla.core.ml_neural_embedding_engine import NeuralEmbeddingEngine, EmbeddingConfig
from safla.core.ml_rl_optimizer import ReinforcementLearningOptimizer
from safla.core.hybrid_memory import HybridMemoryArchitecture
from safla.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUOptimizationBenchmark:
    """Comprehensive GPU optimization benchmark suite."""
    
    def __init__(self, config_path: str = "/app/config/fly_gpu_config.json"):
        self.config = load_config(config_path) if Path(config_path).exists() else {}
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and specifications."""
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            "compute_capability": torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            gpu_info["memory_free"] = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() / 1e9
            gpu_info["memory_used"] = torch.cuda.memory_allocated() / 1e9
        
        logger.info(f"GPU Info: {gpu_info}")
        return gpu_info
    
    def benchmark_embedding_engine(self) -> Dict[str, Any]:
        """Benchmark neural embedding engine performance."""
        logger.info("Benchmarking Neural Embedding Engine...")
        
        # Test data
        test_texts = [
            "This is a test sentence for embedding generation.",
            "SAFLA uses advanced neural networks for embedding optimization.",
            "GPU acceleration significantly improves model performance.",
            "Fly.io provides excellent GPU infrastructure for ML workloads."
        ] * 50  # 200 texts total
        
        # GPU-optimized configuration
        config = EmbeddingConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=64,
            use_flash_attention_2=True,
            mixed_precision="fp16",
            torch_dtype="float16",
            use_sdpa=True,
            gradient_checkpointing=True,
            dataloader_num_workers=8,
            pin_memory=True
        )
        
        engine = NeuralEmbeddingEngine(config)
        
        # Warm up
        _ = asyncio.run(engine.generate_embeddings(test_texts[:10]))
        
        # Benchmark
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        result = asyncio.run(engine.generate_embeddings(test_texts))
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        return {
            "total_time": end_time - start_time,
            "texts_processed": len(test_texts),
            "texts_per_second": len(test_texts) / (end_time - start_time),
            "memory_used_mb": (end_memory - start_memory) / 1e6,
            "embedding_dim": result.embeddings.shape[1] if result.embeddings is not None else 0,
            "batch_size": config.batch_size,
            "device": str(config.device),
            "mixed_precision": config.mixed_precision,
            "flash_attention": config.use_flash_attention_2
        }
    
    def benchmark_rl_optimizer(self) -> Dict[str, Any]:
        """Benchmark reinforcement learning optimizer."""
        logger.info("Benchmarking RL Optimizer...")
        
        # Mock environment for testing
        state_dim = 128
        action_dim = 64
        num_episodes = 100
        
        optimizer = ReinforcementLearningOptimizer(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Simulate training episodes
        for episode in range(num_episodes):
            state = torch.randn(state_dim).to(self.device)
            action = optimizer.select_action(state)
            reward = torch.randn(1).to(self.device)
            next_state = torch.randn(state_dim).to(self.device)
            
            optimizer.update(state, action, reward, next_state)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        return {
            "total_time": end_time - start_time,
            "episodes_processed": num_episodes,
            "episodes_per_second": num_episodes / (end_time - start_time),
            "memory_used_mb": (end_memory - start_memory) / 1e6,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": str(self.device)
        }
    
    def benchmark_memory_system(self) -> Dict[str, Any]:
        """Benchmark hybrid memory system."""
        logger.info("Benchmarking Memory System...")
        
        memory = HybridMemoryArchitecture()
        
        # Test data
        test_data = [
            {"content": f"Test memory item {i}", "embedding": torch.randn(384).tolist()}
            for i in range(1000)
        ]
        
        start_time = time.time()
        
        # Store items
        for item in test_data:
            asyncio.run(memory.store_memory(
                content=item["content"],
                memory_type="episodic",
                embedding=item["embedding"]
            ))
        
        # Retrieve items
        retrieval_queries = test_data[:100]
        retrieved_items = []
        
        for query in retrieval_queries:
            results = asyncio.run(memory.retrieve_memories(
                query=query["content"],
                memory_type="episodic",
                limit=5
            ))
            retrieved_items.extend(results)
        
        end_time = time.time()
        
        return {
            "total_time": end_time - start_time,
            "items_stored": len(test_data),
            "items_retrieved": len(retrieved_items),
            "storage_ops_per_second": len(test_data) / (end_time - start_time),
            "retrieval_ops_per_second": len(retrieval_queries) / (end_time - start_time),
            "memory_efficiency": len(retrieved_items) / len(retrieval_queries)
        }
    
    def benchmark_system_resources(self) -> Dict[str, Any]:
        """Benchmark overall system resource usage."""
        logger.info("Benchmarking System Resources...")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9,
                "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
                "max_reserved": torch.cuda.max_memory_reserved() / 1e9
            }
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / 1e9,
            "memory_used_gb": memory.used / 1e9,
            "gpu_memory": gpu_memory,
            "timestamp": datetime.now().isoformat()
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting Full GPU Optimization Benchmark...")
        
        start_time = datetime.now()
        
        self.results = {
            "benchmark_info": {
                "start_time": start_time.isoformat(),
                "config_path": "/app/config/fly_gpu_config.json",
                "safla_version": "0.1.3"
            },
            "gpu_info": self.check_gpu_availability(),
            "system_resources_start": self.benchmark_system_resources(),
            "embedding_engine": self.benchmark_embedding_engine(),
            "rl_optimizer": self.benchmark_rl_optimizer(),
            "memory_system": self.benchmark_memory_system(),
            "system_resources_end": self.benchmark_system_resources()
        }
        
        end_time = datetime.now()
        self.results["benchmark_info"]["end_time"] = end_time.isoformat()
        self.results["benchmark_info"]["total_duration"] = (end_time - start_time).total_seconds()
        
        return self.results
    
    def save_results(self, output_path: str = "/data/benchmark_results.json"):
        """Save benchmark results to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
    
    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            logger.error("No benchmark results available")
            return
        
        print("\n" + "="*60)
        print("SAFLA GPU OPTIMIZATION BENCHMARK SUMMARY")
        print("="*60)
        
        gpu_info = self.results["gpu_info"]
        print(f"GPU Available: {gpu_info['cuda_available']}")
        if gpu_info['cuda_available']:
            print(f"GPU Device: {gpu_info['device_name']}")
            print(f"GPU Memory: {gpu_info['memory_total']:.2f} GB")
        
        print(f"\nEmbedding Engine Performance:")
        embedding = self.results["embedding_engine"]
        print(f"  Texts/Second: {embedding['texts_per_second']:.2f}")
        print(f"  Memory Usage: {embedding['memory_used_mb']:.2f} MB")
        print(f"  Flash Attention: {embedding['flash_attention']}")
        print(f"  Mixed Precision: {embedding['mixed_precision']}")
        
        print(f"\nRL Optimizer Performance:")
        rl_opt = self.results["rl_optimizer"]
        print(f"  Episodes/Second: {rl_opt['episodes_per_second']:.2f}")
        print(f"  Memory Usage: {rl_opt['memory_used_mb']:.2f} MB")
        
        print(f"\nMemory System Performance:")
        memory = self.results["memory_system"]
        print(f"  Storage Ops/Second: {memory['storage_ops_per_second']:.2f}")
        print(f"  Retrieval Ops/Second: {memory['retrieval_ops_per_second']:.2f}")
        
        print(f"\nTotal Benchmark Time: {self.results['benchmark_info']['total_duration']:.2f} seconds")
        print("="*60)

async def main():
    """Main benchmark execution."""
    benchmark = GPUOptimizationBenchmark()
    
    try:
        results = await benchmark.run_full_benchmark()
        benchmark.save_results()
        benchmark.print_summary()
        
        return results
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())