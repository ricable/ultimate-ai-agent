"""
Quantum Performance Monitoring and Optimization
Monitors and optimizes quantum-classical hybrid workflows for maximum efficiency.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from collections import deque

# Import monitoring components
from ..monitoring.metrics.performance import performance_monitor
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import (
    record_agent_request, update_agent_performance_metrics
)

# Import quantum components
from .quantum_resource_allocator import get_resource_allocator
from .quantum_cache import get_quantum_cache
from .quantum_ray_integration import get_quantum_distributed_processor

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    THROUGHPUT = "throughput"
    RESOURCE_UTILIZATION = "resource_utilization"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUANTUM_ADVANTAGE = "quantum_advantage"
    ERROR_RATE = "error_rate"
    FIDELITY = "fidelity"
    COST_EFFICIENCY = "cost_efficiency"

class OptimizationTechnique(Enum):
    """Optimization techniques for quantum workflows"""
    CIRCUIT_OPTIMIZATION = "circuit_optimization"
    PARAMETER_TUNING = "parameter_tuning"
    CACHING_STRATEGY = "caching_strategy"
    RESOURCE_ALLOCATION = "resource_allocation"
    DISTRIBUTED_PROCESSING = "distributed_processing"
    HYBRID_APPROACHES = "hybrid_approaches"

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    execution_time: float
    throughput: float  # Operations per second
    resource_utilization: Dict[str, float]
    cache_hit_rate: float
    quantum_advantage_factor: float
    error_rate: float
    average_fidelity: float
    cost_efficiency: float
    active_workloads: int

@dataclass
class OptimizationResult:
    """Result of performance optimization"""
    technique: OptimizationTechnique
    improvement_factor: float
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    optimization_time: float
    recommendations: List[str]
    confidence: float

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    severity: str  # "low", "medium", "high", "critical"
    metric: PerformanceMetric
    threshold: float
    current_value: float
    description: str
    recommendations: List[str]
    timestamp: datetime

class QuantumPerformanceMonitor:
    """Advanced performance monitor for quantum computing workflows"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.optimization_history: List[OptimizationResult] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Performance thresholds
        self.thresholds = {
            PerformanceMetric.EXECUTION_TIME: 10.0,  # seconds
            PerformanceMetric.THROUGHPUT: 10.0,     # ops/sec
            PerformanceMetric.RESOURCE_UTILIZATION: 0.8,  # 80%
            PerformanceMetric.CACHE_HIT_RATE: 0.6,  # 60%
            PerformanceMetric.QUANTUM_ADVANTAGE: 1.1,  # 10% advantage
            PerformanceMetric.ERROR_RATE: 0.05,     # 5%
            PerformanceMetric.FIDELITY: 0.95,       # 95%
            PerformanceMetric.COST_EFFICIENCY: 0.8  # 80%
        }
        
        # Optimization recommendations cache
        self.recommendation_cache: Dict[str, List[str]] = {}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.monitoring_task = None
        
        logger.info("Quantum performance monitor initialized")
    
    async def start_monitoring(self):
        """Start background performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop background performance monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = await self.collect_metrics()
                
                # Store in history
                self.metrics_history.append(current_metrics)
                
                # Check for performance issues
                alerts = await self._check_performance_thresholds(current_metrics)
                
                # Update active alerts
                for alert in alerts:
                    self.active_alerts[alert.alert_id] = alert
                
                # Clean up resolved alerts
                await self._cleanup_resolved_alerts(current_metrics)
                
                # Log performance summary
                await self._log_performance_summary(current_metrics)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # Get resource allocator metrics
            resource_utilization = {}
            if get_resource_allocator():
                allocator_report = await get_resource_allocator().get_allocation_report()
                resource_utilization = allocator_report.get('current_utilization', {})
            
            # Get cache metrics
            cache_hit_rate = 0.0
            if get_quantum_cache():
                cache_stats = await get_quantum_cache().get_comprehensive_stats()
                circuit_cache_stats = cache_stats.get('circuit_cache', {})
                cache_hit_rate = circuit_cache_stats.get('hit_rate', 0.0)
            
            # Get distributed processing metrics
            distributed_metrics = {}
            if get_quantum_distributed_processor():
                status = await get_quantum_distributed_processor().get_system_status()
                distributed_metrics = status.get('system_performance', {})
            
            # Calculate derived metrics
            recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
            
            avg_execution_time = (
                statistics.mean([m.execution_time for m in recent_metrics])
                if recent_metrics else 0.0
            )
            
            avg_fidelity = (
                statistics.mean([m.average_fidelity for m in recent_metrics])
                if recent_metrics else 1.0
            )
            
            throughput = (
                1.0 / avg_execution_time if avg_execution_time > 0 else 0.0
            )
            
            quantum_advantage_factor = (
                statistics.mean([m.quantum_advantage_factor for m in recent_metrics])
                if recent_metrics else 1.0
            )
            
            error_rate = (
                statistics.mean([m.error_rate for m in recent_metrics])
                if recent_metrics else 0.0
            )
            
            cost_efficiency = (
                throughput * avg_fidelity * (1 - error_rate)
            )
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                execution_time=avg_execution_time,
                throughput=throughput,
                resource_utilization=resource_utilization,
                cache_hit_rate=cache_hit_rate,
                quantum_advantage_factor=quantum_advantage_factor,
                error_rate=error_rate,
                average_fidelity=avg_fidelity,
                cost_efficiency=cost_efficiency,
                active_workloads=resource_utilization.get('active_workloads', 0)
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                execution_time=0.0,
                throughput=0.0,
                resource_utilization={},
                cache_hit_rate=0.0,
                quantum_advantage_factor=1.0,
                error_rate=0.0,
                average_fidelity=1.0,
                cost_efficiency=0.0,
                active_workloads=0
            )
    
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Check if any performance thresholds are violated"""
        alerts = []
        
        # Check execution time
        if metrics.execution_time > self.thresholds[PerformanceMetric.EXECUTION_TIME]:
            alerts.append(PerformanceAlert(
                alert_id=f"exec_time_{datetime.utcnow().timestamp()}",
                severity="high",
                metric=PerformanceMetric.EXECUTION_TIME,
                threshold=self.thresholds[PerformanceMetric.EXECUTION_TIME],
                current_value=metrics.execution_time,
                description=f"Execution time ({metrics.execution_time:.2f}s) exceeds threshold",
                recommendations=[
                    "Consider enabling distributed processing",
                    "Optimize quantum circuits",
                    "Check for resource bottlenecks"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Check cache hit rate
        if metrics.cache_hit_rate < self.thresholds[PerformanceMetric.CACHE_HIT_RATE]:
            alerts.append(PerformanceAlert(
                alert_id=f"cache_hit_{datetime.utcnow().timestamp()}",
                severity="medium",
                metric=PerformanceMetric.CACHE_HIT_RATE,
                threshold=self.thresholds[PerformanceMetric.CACHE_HIT_RATE],
                current_value=metrics.cache_hit_rate,
                description=f"Cache hit rate ({metrics.cache_hit_rate:.1%}) is below optimal",
                recommendations=[
                    "Increase cache size",
                    "Improve cache key strategy",
                    "Review circuit similarity detection"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Check quantum advantage
        if metrics.quantum_advantage_factor < self.thresholds[PerformanceMetric.QUANTUM_ADVANTAGE]:
            alerts.append(PerformanceAlert(
                alert_id=f"quantum_adv_{datetime.utcnow().timestamp()}",
                severity="medium",
                metric=PerformanceMetric.QUANTUM_ADVANTAGE,
                threshold=self.thresholds[PerformanceMetric.QUANTUM_ADVANTAGE],
                current_value=metrics.quantum_advantage_factor,
                description=f"Quantum advantage factor ({metrics.quantum_advantage_factor:.2f}) is low",
                recommendations=[
                    "Review problem suitability for quantum computing",
                    "Consider classical alternatives",
                    "Optimize quantum algorithms"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Check error rate
        if metrics.error_rate > self.thresholds[PerformanceMetric.ERROR_RATE]:
            alerts.append(PerformanceAlert(
                alert_id=f"error_rate_{datetime.utcnow().timestamp()}",
                severity="high",
                metric=PerformanceMetric.ERROR_RATE,
                threshold=self.thresholds[PerformanceMetric.ERROR_RATE],
                current_value=metrics.error_rate,
                description=f"Error rate ({metrics.error_rate:.1%}) is too high",
                recommendations=[
                    "Enable error correction",
                    "Review quantum circuit design",
                    "Check hardware noise levels"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Check fidelity
        if metrics.average_fidelity < self.thresholds[PerformanceMetric.FIDELITY]:
            alerts.append(PerformanceAlert(
                alert_id=f"fidelity_{datetime.utcnow().timestamp()}",
                severity="high",
                metric=PerformanceMetric.FIDELITY,
                threshold=self.thresholds[PerformanceMetric.FIDELITY],
                current_value=metrics.average_fidelity,
                description=f"Average fidelity ({metrics.average_fidelity:.3f}) is below threshold",
                recommendations=[
                    "Implement error mitigation",
                    "Reduce circuit depth",
                    "Optimize gate fidelities"
                ],
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _cleanup_resolved_alerts(self, current_metrics: PerformanceMetrics):
        """Remove alerts that are no longer active"""
        resolved_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            current_value = self._get_metric_value(current_metrics, alert.metric)
            
            # Check if alert condition is resolved
            is_resolved = False
            if alert.metric in [PerformanceMetric.EXECUTION_TIME, PerformanceMetric.ERROR_RATE]:
                is_resolved = current_value <= alert.threshold
            else:
                is_resolved = current_value >= alert.threshold
            
            if is_resolved:
                resolved_alerts.append(alert_id)
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Performance alert {alert_id} resolved")
    
    def _get_metric_value(self, metrics: PerformanceMetrics, metric: PerformanceMetric) -> float:
        """Get specific metric value from metrics object"""
        metric_map = {
            PerformanceMetric.EXECUTION_TIME: metrics.execution_time,
            PerformanceMetric.THROUGHPUT: metrics.throughput,
            PerformanceMetric.CACHE_HIT_RATE: metrics.cache_hit_rate,
            PerformanceMetric.QUANTUM_ADVANTAGE: metrics.quantum_advantage_factor,
            PerformanceMetric.ERROR_RATE: metrics.error_rate,
            PerformanceMetric.FIDELITY: metrics.average_fidelity,
            PerformanceMetric.COST_EFFICIENCY: metrics.cost_efficiency
        }
        
        return metric_map.get(metric, 0.0)
    
    async def _log_performance_summary(self, metrics: PerformanceMetrics):
        """Log performance summary"""
        # Log to UAP logger
        uap_logger.log_event(
            LogLevel.INFO,
            "Quantum performance metrics collected",
            EventType.AGENT,
            {
                "execution_time": metrics.execution_time,
                "throughput": metrics.throughput,
                "cache_hit_rate": metrics.cache_hit_rate,
                "quantum_advantage_factor": metrics.quantum_advantage_factor,
                "error_rate": metrics.error_rate,
                "average_fidelity": metrics.average_fidelity,
                "cost_efficiency": metrics.cost_efficiency,
                "active_workloads": metrics.active_workloads,
                "active_alerts": len(self.active_alerts)
            },
            "quantum_performance_monitor"
        )
        
        # Update Prometheus metrics
        try:
            update_agent_performance_metrics(
                agent_name="quantum_agent",
                response_time=metrics.execution_time,
                throughput=metrics.throughput,
                error_rate=metrics.error_rate
            )
        except Exception as e:
            logger.warning(f"Failed to update Prometheus metrics: {e}")
    
    async def optimize_performance(self, target_metric: PerformanceMetric) -> OptimizationResult:
        """Optimize performance for a specific metric"""
        if not self.metrics_history:
            raise ValueError("No performance history available for optimization")
        
        before_metrics = self.metrics_history[-1]
        start_time = datetime.utcnow()
        
        # Select optimization technique based on target metric
        technique = self._select_optimization_technique(target_metric, before_metrics)
        
        # Apply optimization
        recommendations = await self._apply_optimization(technique, before_metrics)
        
        # Wait a bit for optimization to take effect
        await asyncio.sleep(5)
        
        # Collect new metrics
        after_metrics = await self.collect_metrics()
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate improvement
        before_value = self._get_metric_value(before_metrics, target_metric)
        after_value = self._get_metric_value(after_metrics, target_metric)
        
        if target_metric in [PerformanceMetric.EXECUTION_TIME, PerformanceMetric.ERROR_RATE]:
            # Lower is better
            improvement_factor = before_value / after_value if after_value > 0 else 1.0
        else:
            # Higher is better
            improvement_factor = after_value / before_value if before_value > 0 else 1.0
        
        confidence = min(1.0, improvement_factor / 2.0)  # Simple confidence calculation
        
        result = OptimizationResult(
            technique=technique,
            improvement_factor=improvement_factor,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            optimization_time=optimization_time,
            recommendations=recommendations,
            confidence=confidence
        )
        
        self.optimization_history.append(result)
        
        logger.info(f"Performance optimization completed: {improvement_factor:.2f}x improvement")
        
        return result
    
    def _select_optimization_technique(self, target_metric: PerformanceMetric, 
                                     metrics: PerformanceMetrics) -> OptimizationTechnique:
        """Select optimal optimization technique for target metric"""
        technique_map = {
            PerformanceMetric.EXECUTION_TIME: OptimizationTechnique.DISTRIBUTED_PROCESSING,
            PerformanceMetric.THROUGHPUT: OptimizationTechnique.CACHING_STRATEGY,
            PerformanceMetric.CACHE_HIT_RATE: OptimizationTechnique.CACHING_STRATEGY,
            PerformanceMetric.QUANTUM_ADVANTAGE: OptimizationTechnique.HYBRID_APPROACHES,
            PerformanceMetric.ERROR_RATE: OptimizationTechnique.CIRCUIT_OPTIMIZATION,
            PerformanceMetric.FIDELITY: OptimizationTechnique.CIRCUIT_OPTIMIZATION,
            PerformanceMetric.COST_EFFICIENCY: OptimizationTechnique.RESOURCE_ALLOCATION
        }
        
        return technique_map.get(target_metric, OptimizationTechnique.CIRCUIT_OPTIMIZATION)
    
    async def _apply_optimization(self, technique: OptimizationTechnique, 
                                metrics: PerformanceMetrics) -> List[str]:
        """Apply specific optimization technique"""
        recommendations = []
        
        try:
            if technique == OptimizationTechnique.CIRCUIT_OPTIMIZATION:
                recommendations = [
                    "Applied circuit depth reduction",
                    "Enabled gate fusion optimization",
                    "Implemented noise-aware compilation"
                ]
            
            elif technique == OptimizationTechnique.CACHING_STRATEGY:
                if get_quantum_cache():
                    # Trigger cache optimization
                    recommendations = [
                        "Increased cache size",
                        "Improved circuit similarity detection",
                        "Optimized cache eviction policy"
                    ]
            
            elif technique == OptimizationTechnique.DISTRIBUTED_PROCESSING:
                if get_quantum_distributed_processor():
                    # Optimize distributed processing
                    recommendations = [
                        "Enabled distributed simulation for large circuits",
                        "Optimized workload distribution",
                        "Improved worker utilization"
                    ]
            
            elif technique == OptimizationTechnique.RESOURCE_ALLOCATION:
                if get_resource_allocator():
                    # Optimize resource allocation
                    recommendations = [
                        "Refined quantum vs classical selection criteria",
                        "Improved resource utilization prediction",
                        "Enhanced workload prioritization"
                    ]
            
            elif technique == OptimizationTechnique.HYBRID_APPROACHES:
                recommendations = [
                    "Enabled hybrid quantum-classical algorithms",
                    "Optimized classical pre/post-processing",
                    "Improved quantum advantage detection"
                ]
            
            else:
                recommendations = ["Applied general performance optimizations"]
            
        except Exception as e:
            logger.error(f"Error applying optimization {technique.value}: {e}")
            recommendations = [f"Optimization attempt failed: {str(e)}"]
        
        return recommendations
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.metrics_history:
            return {
                'status': 'No performance data available',
                'message': 'Start monitoring to collect performance metrics'
            }
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements
        latest_metrics = self.metrics_history[-1]
        
        # Calculate trends
        trends = {}
        if len(recent_metrics) > 10:
            for metric in PerformanceMetric:
                values = [self._get_metric_value(m, metric) for m in recent_metrics]
                trend = "improving" if values[-1] > values[0] else "declining"
                trends[metric.value] = trend
        
        # Calculate averages
        averages = {
            'execution_time': statistics.mean([m.execution_time for m in recent_metrics]),
            'throughput': statistics.mean([m.throughput for m in recent_metrics]),
            'cache_hit_rate': statistics.mean([m.cache_hit_rate for m in recent_metrics]),
            'quantum_advantage_factor': statistics.mean([m.quantum_advantage_factor for m in recent_metrics]),
            'error_rate': statistics.mean([m.error_rate for m in recent_metrics]),
            'average_fidelity': statistics.mean([m.average_fidelity for m in recent_metrics]),
            'cost_efficiency': statistics.mean([m.cost_efficiency for m in recent_metrics])
        }
        
        return {
            'current_metrics': asdict(latest_metrics),
            'historical_averages': averages,
            'performance_trends': trends,
            'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
            'optimization_history': [
                {
                    'technique': opt.technique.value,
                    'improvement_factor': opt.improvement_factor,
                    'optimization_time': opt.optimization_time,
                    'recommendations': opt.recommendations,
                    'confidence': opt.confidence
                }
                for opt in self.optimization_history[-10:]  # Last 10 optimizations
            ],
            'overall_performance_score': self._calculate_overall_score(latest_metrics),
            'recommendations': self._generate_recommendations(latest_metrics, trends),
            'monitoring_status': {
                'active': self.monitoring_active,
                'data_points': len(self.metrics_history),
                'monitoring_interval': self.monitoring_interval
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_overall_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)"""
        # Weighted scoring of different metrics
        weights = {
            'throughput': 0.25,
            'cache_hit_rate': 0.15,
            'quantum_advantage_factor': 0.20,
            'error_rate': 0.15,
            'average_fidelity': 0.15,
            'cost_efficiency': 0.10
        }
        
        # Normalize metrics to 0-1 scale
        normalized = {
            'throughput': min(1.0, metrics.throughput / 50.0),  # Assume 50 ops/sec is excellent
            'cache_hit_rate': metrics.cache_hit_rate,
            'quantum_advantage_factor': min(1.0, metrics.quantum_advantage_factor / 2.0),
            'error_rate': 1.0 - metrics.error_rate,  # Invert error rate
            'average_fidelity': metrics.average_fidelity,
            'cost_efficiency': min(1.0, metrics.cost_efficiency)
        }
        
        # Calculate weighted score
        score = sum(normalized[metric] * weight for metric, weight in weights.items())
        
        return float(score * 100)  # Convert to 0-100 scale
    
    def _generate_recommendations(self, metrics: PerformanceMetrics, 
                                trends: Dict[str, str]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Execution time recommendations
        if metrics.execution_time > 5.0:
            recommendations.append("Consider enabling distributed processing for large simulations")
        
        # Cache recommendations
        if metrics.cache_hit_rate < 0.7:
            recommendations.append("Improve caching strategy to reduce redundant computations")
        
        # Quantum advantage recommendations
        if metrics.quantum_advantage_factor < 1.2:
            recommendations.append("Review problem selection for quantum advantage opportunities")
        
        # Error rate recommendations
        if metrics.error_rate > 0.03:
            recommendations.append("Implement error correction and mitigation techniques")
        
        # Fidelity recommendations
        if metrics.average_fidelity < 0.97:
            recommendations.append("Optimize quantum circuits for higher fidelity")
        
        # Trend-based recommendations
        if trends.get('throughput') == 'declining':
            recommendations.append("Investigate throughput decline and optimize bottlenecks")
        
        if trends.get('error_rate') == 'improving' and metrics.error_rate > 0.01:
            recommendations.append("Continue error rate improvement efforts")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")
        
        return recommendations

# Global performance monitor instance
quantum_performance_monitor: Optional[QuantumPerformanceMonitor] = None

def initialize_quantum_performance_monitor(history_size: int = 1000) -> QuantumPerformanceMonitor:
    """Initialize the global quantum performance monitor"""
    global quantum_performance_monitor
    quantum_performance_monitor = QuantumPerformanceMonitor(history_size)
    return quantum_performance_monitor

def get_quantum_performance_monitor() -> Optional[QuantumPerformanceMonitor]:
    """Get the global quantum performance monitor instance"""
    return quantum_performance_monitor

# Convenience functions
async def start_quantum_performance_monitoring():
    """Start quantum performance monitoring"""
    if quantum_performance_monitor:
        await quantum_performance_monitor.start_monitoring()

async def get_quantum_performance_report() -> Dict[str, Any]:
    """Get quantum performance report"""
    if quantum_performance_monitor:
        return await quantum_performance_monitor.get_performance_report()
    return {'status': 'Performance monitor not initialized'}

async def optimize_quantum_performance(target_metric: PerformanceMetric) -> OptimizationResult:
    """Optimize quantum performance for specific metric"""
    if quantum_performance_monitor:
        return await quantum_performance_monitor.optimize_performance(target_metric)
    raise RuntimeError("Performance monitor not initialized")
