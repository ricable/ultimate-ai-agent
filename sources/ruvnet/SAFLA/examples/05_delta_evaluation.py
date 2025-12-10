#!/usr/bin/env python3
"""
SAFLA Delta Evaluation - Performance Improvement Tracking
========================================================

This example demonstrates SAFLA's sophisticated delta evaluation system for 
tracking and quantifying performance improvements across multiple dimensions.

Learning Objectives:
- Understand delta evaluation methodology
- Learn performance, efficiency, stability, and capability metrics
- Practice adaptive weighting and context-aware evaluation
- Implement improvement tracking and trend analysis

Time to Complete: 20-25 minutes
Complexity: Intermediate to Advanced
"""

import asyncio
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

from safla.core.delta_evaluation import (
    OptimizedDeltaEvaluator,
    BatchDeltaRequest,
    AdaptiveWeights
)
from safla import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    reward: float
    throughput: float
    latency: float
    memory_usage: float
    cpu_usage: float
    error_rate: float
    capabilities: int
    context: str


class DeltaEvaluationDemo:
    """Comprehensive delta evaluation demonstration."""
    
    def __init__(self):
        self.evaluator = None
        self.performance_history: List[PerformanceSnapshot] = []
        self.evaluation_results = []
    
    async def initialize(self):
        """Initialize the delta evaluation system."""
        print("📊 Initializing SAFLA Delta Evaluation System...")
        self.evaluator = OptimizedDeltaEvaluator(
            enable_batch_processing=True,
            enable_caching=True,
            pool_size=10
        )
        print("✅ Delta evaluator ready!")
    
    def generate_performance_snapshot(self, base_scenario: str, iteration: int) -> PerformanceSnapshot:
        """Generate realistic performance data for different scenarios."""
        # Base performance varies by scenario
        scenario_configs = {
            "training": {"base_reward": 0.7, "throughput": 100, "variance": 0.1},
            "inference": {"base_reward": 0.85, "throughput": 500, "variance": 0.05},
            "optimization": {"base_reward": 0.6, "throughput": 50, "variance": 0.15},
            "production": {"base_reward": 0.9, "throughput": 300, "variance": 0.03}
        }
        
        config = scenario_configs.get(base_scenario, scenario_configs["training"])
        
        # Simulate improvement over time
        improvement_factor = 1 + (iteration * 0.02)  # 2% improvement per iteration
        noise = random.gauss(0, config["variance"])
        
        return PerformanceSnapshot(
            timestamp=datetime.now() - timedelta(minutes=30-iteration),
            reward=min(1.0, config["base_reward"] * improvement_factor + noise),
            throughput=config["throughput"] * improvement_factor + random.gauss(0, 10),
            latency=max(1.0, 50 / improvement_factor + random.gauss(0, 5)),
            memory_usage=random.uniform(100, 800) + iteration * 5,  # Slight memory increase
            cpu_usage=random.uniform(0.3, 0.8),
            error_rate=max(0.0, 0.05 / improvement_factor + random.gauss(0, 0.01)),
            capabilities=10 + (iteration // 5),  # Gain capability every 5 iterations
            context=base_scenario
        )
    
    async def demonstrate_basic_delta_evaluation(self):
        """Demonstrate basic delta evaluation between two performance states."""
        print("\n📈 Basic Delta Evaluation")
        print("-" * 40)
        
        # Simulate before and after optimization
        before = self.generate_performance_snapshot("training", 0)
        after = self.generate_performance_snapshot("training", 10)
        
        print(f"Evaluating optimization impact...")
        print(f"Before: Reward={before.reward:.3f}, Throughput={before.throughput:.1f}")
        print(f"After:  Reward={after.reward:.3f}, Throughput={after.throughput:.1f}")
        
        # Evaluate the delta
        result = self.evaluator.evaluate_delta(
            performance_data={
                "current_reward": after.reward,
                "previous_reward": before.reward,
                "tokens_used": 1000
            },
            efficiency_data={
                "current_throughput": after.throughput,
                "previous_throughput": before.throughput,
                "resource_used": 0.8
            },
            stability_data={
                "divergence_score": abs(after.error_rate - before.error_rate)
            },
            capability_data={
                "new_capabilities": after.capabilities - before.capabilities,
                "total_capabilities": after.capabilities
            },
            context="training_optimization"
        )
        
        print(f"\nDelta Evaluation Results:")
        print(f"  📊 Overall Delta: {result.overall_delta:.4f}")
        print(f"  🎯 Performance Delta: {result.metrics.performance_delta:.4f}")
        print(f"  ⚡ Efficiency Delta: {result.metrics.efficiency_delta:.4f}")
        print(f"  🔒 Stability Delta: {result.metrics.stability_delta:.4f}")
        print(f"  🚀 Capability Delta: {result.metrics.capability_delta:.4f}")
        print(f"  🎯 Confidence: {result.metrics.confidence:.2f}")
        
        improvement_status = "✅ IMPROVEMENT" if result.overall_delta > 0 else "⚠️ DEGRADATION"
        print(f"  {improvement_status} detected")
        
        self.evaluation_results.append(result)
    
    async def demonstrate_adaptive_weighting(self):
        """Demonstrate adaptive weighting based on context."""
        print("\n⚖️ Adaptive Weighting Demonstration")
        print("-" * 40)
        
        # Test different contexts with same performance data
        contexts = [
            ("performance_critical", "Performance-critical application"),
            ("efficiency_focused", "Efficiency-focused deployment"),
            ("stability_priority", "Stability-priority system"),
            ("capability_expansion", "Capability expansion phase")
        ]
        
        base_data = {
            "performance_data": {"current_reward": 0.85, "previous_reward": 0.80, "tokens_used": 1000},
            "efficiency_data": {"current_throughput": 120, "previous_throughput": 100, "resource_used": 0.7},
            "stability_data": {"divergence_score": 0.02},
            "capability_data": {"new_capabilities": 2, "total_capabilities": 15}
        }
        
        print("Same performance improvement with different contexts:")
        
        for context, description in contexts:
            result = self.evaluator.evaluate_delta(
                context=context,
                **base_data
            )
            
            print(f"\n  🎯 {description}:")
            print(f"    Overall Delta: {result.overall_delta:.4f}")
            print(f"    Context weights applied based on '{context}' priority")
            
            # Show which aspect was weighted most heavily
            deltas = {
                "Performance": result.metrics.performance_delta,
                "Efficiency": result.metrics.efficiency_delta,
                "Stability": result.metrics.stability_delta,
                "Capability": result.metrics.capability_delta
            }
            primary_focus = max(deltas.items(), key=lambda x: abs(x[1]))
            print(f"    Primary impact: {primary_focus[0]} ({primary_focus[1]:.4f})")
    
    async def demonstrate_batch_evaluation(self):
        """Demonstrate batch delta evaluation for multiple scenarios."""
        print("\n⚡ Batch Delta Evaluation")
        print("-" * 40)
        
        # Create batch requests for different optimization experiments
        experiments = [
            ("memory_optimization", "Memory usage optimization"),
            ("algorithm_improvement", "Algorithm efficiency improvement"),
            ("cache_optimization", "Caching strategy optimization"),
            ("parallel_processing", "Parallel processing implementation"),
            ("model_compression", "Model compression experiment")
        ]
        
        batch_requests = []
        for i, (exp_id, description) in enumerate(experiments):
            before = self.generate_performance_snapshot("optimization", 0)
            after = self.generate_performance_snapshot("optimization", random.randint(5, 15))
            
            request = BatchDeltaRequest(
                request_id=exp_id,
                performance_data={
                    "current_reward": after.reward,
                    "previous_reward": before.reward,
                    "tokens_used": 1000 + i * 200
                },
                efficiency_data={
                    "current_throughput": after.throughput,
                    "previous_throughput": before.throughput,
                    "resource_used": 0.6 + i * 0.1
                },
                stability_data={
                    "divergence_score": abs(after.error_rate - before.error_rate)
                },
                capability_data={
                    "new_capabilities": after.capabilities - before.capabilities,
                    "total_capabilities": after.capabilities
                },
                context="optimization_experiment"
            )
            batch_requests.append(request)
        
        print(f"Processing {len(batch_requests)} optimization experiments...")
        start_time = time.time()
        
        batch_results = self.evaluator.batch_evaluate_delta(batch_requests)
        
        processing_time = time.time() - start_time
        print(f"⏱️ Batch processing completed in {processing_time:.3f}s")
        print(f"📊 Average per evaluation: {processing_time/len(batch_requests)*1000:.1f}ms")
        
        # Analyze batch results
        print("\nExperiment Results:")
        for result, (_, description) in zip(batch_results, experiments):
            delta = result.result.overall_delta
            status = "✅ SUCCESS" if delta > 0.01 else "⚠️ MINIMAL" if delta > 0 else "❌ DEGRADED"
            print(f"  {status} {description}: Δ={delta:.4f}")
        
        # Find best experiment
        best_result = max(batch_results, key=lambda x: x.result.overall_delta)
        best_exp = experiments[batch_results.index(best_result)][1]
        print(f"\n🏆 Best performing experiment: {best_exp}")
        print(f"    Delta: {best_result.result.overall_delta:.4f}")
        print(f"    Processing time: {best_result.processing_time:.3f}ms")
    
    async def demonstrate_trend_analysis(self):
        """Demonstrate trend analysis over time."""
        print("\n📈 Performance Trend Analysis")
        print("-" * 40)
        
        # Generate performance history over time
        scenarios = ["training", "inference", "production"]
        
        for scenario in scenarios:
            print(f"\nAnalyzing {scenario} performance trend...")
            
            # Generate historical data
            history = []
            deltas = []
            
            for i in range(15):
                snapshot = self.generate_performance_snapshot(scenario, i)
                history.append(snapshot)
                
                if i > 0:
                    # Calculate delta compared to previous
                    prev = history[i-1]
                    current = history[i]
                    
                    result = self.evaluator.evaluate_delta(
                        performance_data={
                            "current_reward": current.reward,
                            "previous_reward": prev.reward,
                            "tokens_used": 1000
                        },
                        efficiency_data={
                            "current_throughput": current.throughput,
                            "previous_throughput": prev.throughput,
                            "resource_used": 0.7
                        },
                        stability_data={
                            "divergence_score": abs(current.error_rate - prev.error_rate)
                        },
                        capability_data={
                            "new_capabilities": current.capabilities - prev.capabilities,
                            "total_capabilities": current.capabilities
                        },
                        context=f"{scenario}_trend"
                    )
                    deltas.append(result.overall_delta)
            
            # Analyze trend
            avg_delta = np.mean(deltas) if deltas else 0
            trend_direction = "📈 IMPROVING" if avg_delta > 0 else "📉 DECLINING"
            
            print(f"  {trend_direction} Average delta: {avg_delta:.4f}")
            print(f"  📊 Total measurements: {len(deltas)}")
            print(f"  🎯 Positive deltas: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
            
            if deltas:
                recent_trend = np.mean(deltas[-3:]) if len(deltas) >= 3 else deltas[-1]
                print(f"  📊 Recent trend: {recent_trend:.4f}")
    
    async def demonstrate_performance_monitoring(self):
        """Demonstrate real-time performance monitoring with delta evaluation."""
        print("\n🔍 Real-time Performance Monitoring")
        print("-" * 40)
        
        print("Simulating real-time system monitoring...")
        
        # Simulate system running with periodic evaluations
        baseline = self.generate_performance_snapshot("production", 0)
        print(f"📊 Baseline established: Reward={baseline.reward:.3f}")
        
        alerts = []
        monitoring_data = []
        
        for minute in range(1, 11):  # 10 minutes of monitoring
            current = self.generate_performance_snapshot("production", minute)
            
            # Evaluate against baseline
            result = self.evaluator.evaluate_delta(
                performance_data={
                    "current_reward": current.reward,
                    "previous_reward": baseline.reward,
                    "tokens_used": 1000
                },
                efficiency_data={
                    "current_throughput": current.throughput,
                    "previous_throughput": baseline.throughput,
                    "resource_used": current.memory_usage / 1000
                },
                stability_data={
                    "divergence_score": abs(current.error_rate - baseline.error_rate)
                },
                context="realtime_monitoring"
            )
            
            monitoring_data.append({
                "minute": minute,
                "delta": result.overall_delta,
                "reward": current.reward,
                "throughput": current.throughput
            })
            
            # Check for significant changes
            if abs(result.overall_delta) > 0.05:  # 5% change threshold
                alert_type = "🚨 DEGRADATION" if result.overall_delta < 0 else "📈 IMPROVEMENT"
                alerts.append(f"Minute {minute}: {alert_type} detected (Δ={result.overall_delta:.3f})")
                print(f"  ⚡ Minute {minute}: {alert_type} Δ={result.overall_delta:.3f}")
            else:
                print(f"  ✅ Minute {minute}: Stable performance Δ={result.overall_delta:.3f}")
        
        # Summary
        print(f"\n📋 Monitoring Summary:")
        print(f"  ⏰ Duration: 10 minutes")
        print(f"  📊 Measurements: {len(monitoring_data)}")
        print(f"  🚨 Alerts triggered: {len(alerts)}")
        
        if alerts:
            print("  📢 Alert details:")
            for alert in alerts:
                print(f"    {alert}")
        
        # Performance statistics
        avg_delta = np.mean([d["delta"] for d in monitoring_data])
        max_delta = max([d["delta"] for d in monitoring_data])
        min_delta = min([d["delta"] for d in monitoring_data])
        
        print(f"  📈 Average delta: {avg_delta:.4f}")
        print(f"  📊 Delta range: {min_delta:.4f} to {max_delta:.4f}")
    
    async def demonstrate_evaluation_metrics(self):
        """Show delta evaluation system metrics."""
        print("\n📊 Delta Evaluation System Metrics")
        print("-" * 40)
        
        # Get performance metrics from the evaluator
        metrics = self.evaluator.get_performance_metrics()
        
        print("System Performance:")
        print(f"  ⏱️ Average evaluation time: {metrics['avg_evaluation_time']:.2f}ms")
        print(f"  🚀 Evaluations per second: {metrics['evaluations_per_second']:.1f}")
        print(f"  📊 Total evaluations: {metrics['total_evaluations']}")
        print(f"  📦 Batch efficiency: {metrics['batch_efficiency']:.1f} evals/sec")
        print(f"  🔄 Total batches: {metrics['total_batches']}")
        
        # Analysis of our evaluation results
        if self.evaluation_results:
            overall_deltas = [r.overall_delta for r in self.evaluation_results]
            avg_improvement = np.mean(overall_deltas)
            
            print(f"\nEvaluation Analysis:")
            print(f"  📈 Average improvement: {avg_improvement:.4f}")
            print(f"  ✅ Positive evaluations: {sum(1 for d in overall_deltas if d > 0)}/{len(overall_deltas)}")
            print(f"  📊 Delta range: {min(overall_deltas):.4f} to {max(overall_deltas):.4f}")
    
    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up delta evaluation system...")
        # Delta evaluator is lightweight and doesn't require explicit cleanup
        print("✅ Cleanup complete!")


async def main():
    """Run the delta evaluation demonstration."""
    print("📊 SAFLA Delta Evaluation System Demonstration")
    print("=" * 65)
    
    demo = DeltaEvaluationDemo()
    
    try:
        # Initialize
        await demo.initialize()
        
        # Run demonstrations
        await demo.demonstrate_basic_delta_evaluation()
        await demo.demonstrate_adaptive_weighting()
        await demo.demonstrate_batch_evaluation()
        await demo.demonstrate_trend_analysis()
        await demo.demonstrate_performance_monitoring()
        await demo.demonstrate_evaluation_metrics()
        
        print("\n🎉 Delta evaluation demonstration completed successfully!")
        print("\nKey Takeaways:")
        print("  • Delta evaluation quantifies improvements across multiple dimensions")
        print("  • Adaptive weighting adjusts evaluation based on context and priorities")
        print("  • Batch processing enables efficient evaluation of multiple experiments")
        print("  • Trend analysis reveals long-term performance patterns")
        print("  • Real-time monitoring enables proactive performance management")
        
        print("\nNext Steps:")
        print("  1. Run 06_meta_cognitive.py to explore self-awareness features")
        print("  2. Try 07_mcp_integration.py for external system integration")
        
    except Exception as e:
        logger.exception("Delta evaluation demo failed")
        print(f"❌ Demo failed: {e}")
    
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())