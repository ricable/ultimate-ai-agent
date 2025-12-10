"""
Benchmark Utilities
===================

This module provides utility classes and functions for benchmark configuration,
metrics calculation, and result analysis.
"""

import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    timeout_seconds: float = 30.0
    parallel_execution: bool = False
    store_results: bool = True
    database_path: str = "benchmarks/results.db"
    export_results: bool = False
    export_path: Optional[str] = None
    cleanup_days: int = 90
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'BenchmarkConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_data = {
            'warmup_iterations': self.warmup_iterations,
            'measurement_iterations': self.measurement_iterations,
            'timeout_seconds': self.timeout_seconds,
            'parallel_execution': self.parallel_execution,
            'store_results': self.store_results,
            'database_path': self.database_path,
            'export_results': self.export_results,
            'export_path': self.export_path,
            'cleanup_days': self.cleanup_days,
            'log_level': self.log_level
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


@dataclass
class PerformanceMetrics:
    """Performance metrics calculation and analysis."""
    
    @staticmethod
    def calculate_percentiles(values: List[float], percentiles: List[int] = None) -> Dict[str, float]:
        """Calculate percentiles for a list of values."""
        if not values:
            return {}
        
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        
        sorted_values = sorted(values)
        result = {}
        
        for p in percentiles:
            if p < 0 or p > 100:
                continue
            
            if p == 0:
                result[f'p{p}'] = sorted_values[0]
            elif p == 100:
                result[f'p{p}'] = sorted_values[-1]
            else:
                index = (p / 100) * (len(sorted_values) - 1)
                lower_index = int(index)
                upper_index = min(lower_index + 1, len(sorted_values) - 1)
                
                if lower_index == upper_index:
                    result[f'p{p}'] = sorted_values[lower_index]
                else:
                    # Linear interpolation
                    weight = index - lower_index
                    result[f'p{p}'] = (
                        sorted_values[lower_index] * (1 - weight) +
                        sorted_values[upper_index] * weight
                    )
        
        return result
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values."""
        if not values:
            return {}
        
        stats = {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }
        
        if len(values) > 1:
            stats['stdev'] = statistics.stdev(values)
            stats['variance'] = statistics.variance(values)
            stats['coefficient_of_variation'] = stats['stdev'] / stats['mean'] if stats['mean'] != 0 else 0
        else:
            stats['stdev'] = 0.0
            stats['variance'] = 0.0
            stats['coefficient_of_variation'] = 0.0
        
        # Add percentiles
        percentiles = PerformanceMetrics.calculate_percentiles(values)
        stats.update(percentiles)
        
        return stats
    
    @staticmethod
    def calculate_improvement(baseline: float, current: float, 
                            metric_type: str = "latency") -> Dict[str, float]:
        """Calculate improvement metrics between baseline and current values."""
        if baseline == 0:
            return {'improvement_ratio': float('inf') if current > 0 else 0.0}
        
        if metric_type.lower() in ['latency', 'time', 'memory', 'cpu']:
            # Lower is better
            improvement_ratio = baseline / current if current > 0 else float('inf')
            improvement_percent = ((baseline - current) / baseline) * 100
        else:
            # Higher is better (throughput, etc.)
            improvement_ratio = current / baseline
            improvement_percent = ((current - baseline) / baseline) * 100
        
        return {
            'improvement_ratio': improvement_ratio,
            'improvement_percent': improvement_percent,
            'baseline_value': baseline,
            'current_value': current,
            'absolute_change': current - baseline
        }
    
    @staticmethod
    def detect_performance_regression(historical_values: List[float], 
                                    current_value: float,
                                    threshold_percent: float = 10.0) -> Dict[str, Any]:
        """Detect performance regression based on historical data."""
        if len(historical_values) < 3:
            return {
                'regression_detected': False,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate baseline from historical data
        baseline_mean = statistics.mean(historical_values)
        baseline_stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        
        # Calculate z-score
        z_score = (current_value - baseline_mean) / baseline_stdev if baseline_stdev > 0 else 0
        
        # Calculate percentage change
        percent_change = ((current_value - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        # Detect regression (assuming lower is better for most metrics)
        regression_detected = (
            percent_change > threshold_percent or
            z_score > 2.0  # More than 2 standard deviations
        )
        
        return {
            'regression_detected': regression_detected,
            'baseline_mean': baseline_mean,
            'baseline_stdev': baseline_stdev,
            'current_value': current_value,
            'z_score': z_score,
            'percent_change': percent_change,
            'threshold_percent': threshold_percent,
            'confidence': min(abs(z_score) / 2.0, 1.0)  # Confidence in detection
        }


class BenchmarkAnalyzer:
    """Analyzer for benchmark results and trends."""
    
    def __init__(self, database_path: str = "benchmarks/results.db"):
        from .database import BenchmarkDatabase
        self.db = BenchmarkDatabase(database_path)
    
    def analyze_benchmark_trends(self, benchmark_name: str, 
                                days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends for a specific benchmark."""
        # Get execution time trends
        execution_trends = self.db.get_performance_trends(
            benchmark_name, 'execution_time', days
        )
        
        # Get memory usage trends
        memory_trends = self.db.get_performance_trends(
            benchmark_name, 'memory_usage', days
        )
        
        # Get throughput trends if available
        throughput_trends = self.db.get_performance_trends(
            benchmark_name, 'throughput', days
        )
        
        analysis = {
            'benchmark_name': benchmark_name,
            'analysis_period_days': days,
            'data_points': len(execution_trends)
        }
        
        if execution_trends:
            execution_values = [value for _, value in execution_trends]
            analysis['execution_time'] = {
                'statistics': PerformanceMetrics.calculate_statistics(execution_values),
                'trend_direction': self._calculate_trend_direction(execution_trends),
                'stability': self._calculate_stability(execution_values)
            }
        
        if memory_trends:
            memory_values = [value for _, value in memory_trends]
            analysis['memory_usage'] = {
                'statistics': PerformanceMetrics.calculate_statistics(memory_values),
                'trend_direction': self._calculate_trend_direction(memory_trends),
                'stability': self._calculate_stability(memory_values)
            }
        
        if throughput_trends:
            throughput_values = [value for _, value in throughput_trends]
            analysis['throughput'] = {
                'statistics': PerformanceMetrics.calculate_statistics(throughput_values),
                'trend_direction': self._calculate_trend_direction(throughput_trends),
                'stability': self._calculate_stability(throughput_values)
            }
        
        return analysis
    
    def _calculate_trend_direction(self, time_series: List[Tuple[datetime, float]]) -> str:
        """Calculate trend direction from time series data."""
        if len(time_series) < 2:
            return "insufficient_data"
        
        # Simple linear regression to determine trend
        x_values = [(ts.timestamp() - time_series[0][0].timestamp()) / 3600 for ts, _ in time_series]  # Hours
        y_values = [value for _, value in time_series]
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Calculate slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return "stable"
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Very small slope
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _calculate_stability(self, values: List[float]) -> Dict[str, float]:
        """Calculate stability metrics for a series of values."""
        if len(values) < 2:
            return {'coefficient_of_variation': 0.0, 'stability_score': 1.0}
        
        mean_val = statistics.mean(values)
        stdev_val = statistics.stdev(values)
        
        coefficient_of_variation = stdev_val / mean_val if mean_val != 0 else 0
        
        # Stability score: 1.0 = perfectly stable, 0.0 = highly unstable
        stability_score = max(0.0, 1.0 - coefficient_of_variation)
        
        return {
            'coefficient_of_variation': coefficient_of_variation,
            'stability_score': stability_score
        }
    
    def compare_benchmarks(self, benchmark_names: List[str], 
                          days: int = 30) -> Dict[str, Any]:
        """Compare performance across multiple benchmarks."""
        comparison = {
            'benchmark_names': benchmark_names,
            'comparison_period_days': days,
            'benchmarks': {}
        }
        
        for name in benchmark_names:
            stats = self.db.get_benchmark_statistics(name, days)
            comparison['benchmarks'][name] = stats
        
        # Find best and worst performers
        if len(benchmark_names) > 1:
            execution_times = {
                name: comparison['benchmarks'][name].get('avg_execution_time', float('inf'))
                for name in benchmark_names
                if comparison['benchmarks'][name].get('avg_execution_time') is not None
            }
            
            if execution_times:
                fastest = min(execution_times, key=execution_times.get)
                slowest = max(execution_times, key=execution_times.get)
                
                comparison['performance_ranking'] = {
                    'fastest': fastest,
                    'slowest': slowest,
                    'speed_difference': execution_times[slowest] / execution_times[fastest]
                }
        
        return comparison
    
    def generate_performance_report(self, benchmark_names: Optional[List[str]] = None,
                                  days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'analysis_period_days': days,
            'summary': {},
            'benchmarks': {},
            'recommendations': []
        }
        
        # Get all benchmark names if not specified
        if benchmark_names is None:
            # This would require a method to get all benchmark names from the database
            # For now, we'll use an empty list
            benchmark_names = []
        
        # Analyze each benchmark
        total_runs = 0
        successful_runs = 0
        
        for name in benchmark_names:
            stats = self.db.get_benchmark_statistics(name, days)
            trends = self.analyze_benchmark_trends(name, days)
            
            report['benchmarks'][name] = {
                'statistics': stats,
                'trends': trends
            }
            
            total_runs += stats.get('total_runs', 0)
            successful_runs += stats.get('successful_runs', 0)
        
        # Generate summary
        report['summary'] = {
            'total_benchmarks': len(benchmark_names),
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'overall_success_rate': successful_runs / total_runs if total_runs > 0 else 0.0
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['benchmarks'])
        
        return report
    
    def _generate_recommendations(self, benchmark_data: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        for name, data in benchmark_data.items():
            stats = data.get('statistics', {})
            trends = data.get('trends', {})
            
            # Check success rate
            success_rate = stats.get('success_rate', 0.0)
            if success_rate < 0.9:
                recommendations.append(
                    f"Benchmark '{name}' has low success rate ({success_rate:.1%}). "
                    "Investigate and fix failing test cases."
                )
            
            # Check execution time trends
            exec_trend = trends.get('execution_time', {}).get('trend_direction')
            if exec_trend == 'increasing':
                recommendations.append(
                    f"Benchmark '{name}' shows increasing execution time trend. "
                    "Consider performance optimization."
                )
            
            # Check stability
            exec_stability = trends.get('execution_time', {}).get('stability', {})
            stability_score = exec_stability.get('stability_score', 1.0)
            if stability_score < 0.7:
                recommendations.append(
                    f"Benchmark '{name}' shows unstable performance (stability: {stability_score:.2f}). "
                    "Investigate environmental factors or test consistency."
                )
        
        if not recommendations:
            recommendations.append("All benchmarks are performing well. No immediate action required.")
        
        return recommendations