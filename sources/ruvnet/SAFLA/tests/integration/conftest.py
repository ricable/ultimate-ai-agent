"""
Integration test configuration and fixtures for SAFLA system testing.

This module provides comprehensive test fixtures and utilities for integration
testing of the SAFLA (Self-Aware Feedback Loop Algorithm) system.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import random
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock
import logging

# Test configuration
TEST_CONFIG = {
    'timeout': 30.0,
    'max_retries': 3,
    'performance_threshold': 1.0,
    'memory_limit': 0.9,
    'concurrent_limit': 50
}

@dataclass
class TestMetrics:
    """Container for test performance metrics."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    operations_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_latency: float = 0.0
    peak_memory_usage: float = 0.0
    
    @property
    def duration(self) -> float:
        """Calculate test duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.0

class IntegrationTestHelpers:
    """Helper utilities for integration testing."""
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 10.0,
        interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def simulate_load(
        operation_func,
        duration: float = 5.0,
        operations_per_second: int = 10
    ) -> TestMetrics:
        """Simulate load on a system operation."""
        metrics = TestMetrics()
        interval = 1.0 / operations_per_second
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                start = time.time()
                await operation_func()
                latency = time.time() - start
                
                metrics.operations_count += 1
                metrics.success_count += 1
                metrics.average_latency = (
                    (metrics.average_latency * (metrics.success_count - 1) + latency) 
                    / metrics.success_count
                )
                
            except Exception:
                metrics.failure_count += 1
            
            await asyncio.sleep(max(0, interval - (time.time() - start)))
        
        metrics.end_time = time.time()
        return metrics

# Mock Components for Testing

class MockDeltaEvaluator:
    """Mock implementation of Delta Evaluator for testing."""
    
    def __init__(self):
        self.evaluations = []
        self.performance_data = {}
        
    async def evaluate_performance_delta(self, current_state: Dict, target_state: Dict) -> Dict:
        """Mock performance delta evaluation."""
        delta = {
            'performance_improvement': random.uniform(0.1, 0.5),
            'confidence': random.uniform(0.7, 0.95),
            'recommendations': ['optimize_memory', 'improve_throughput'],
            'evaluation_id': f"eval_{len(self.evaluations)}"
        }
        self.evaluations.append(delta)
        self.evaluations.append(delta)
        return delta
    async def evaluate(self, metrics):
        """Mock evaluate method for DeltaMetrics."""
        from test_system_integration import DeltaResult
        
        total_delta = metrics.performance + metrics.efficiency + metrics.stability + metrics.capability
        should_adapt = metrics.performance > 0.2
        
        result = DeltaResult(
            success=True,
            metrics=metrics,
            total_delta=total_delta,
            confidence=metrics.confidence,
            recommendations=['optimize_performance', 'enhance_stability'],
            should_adapt=should_adapt,
            evaluation_id=f"eval_{len(self.evaluations)}"
        )
        self.evaluations.append(result)
        
        # Simulate integration: if should_adapt, trigger meta-cognitive engine
        if should_adapt and hasattr(self, '_meta_engine'):
            decision = await self._meta_engine.make_adaptation_decision(result.__dict__)
            if decision['should_adapt']:
                await self._meta_engine.apply_adaptation(decision['strategy'])
        
        return result
    
    def set_meta_engine(self, meta_engine):
        """Set reference to meta-cognitive engine for integration."""
        self._meta_engine = meta_engine
    
    async def get_evaluation_history(self) -> List[Dict]:
        """Get evaluation history."""
        return self.evaluations.copy()

class MockMetaCognitiveEngine:
    """Mock implementation of Meta-Cognitive Engine for testing."""
    
    def __init__(self):
        self.decisions = []
        self.adaptation_history = []
        self.system_state = {'status': 'operational', 'confidence': 0.8}
        self.emergency_stopped = False
        
    async def make_strategic_decision(self, context: Dict) -> Dict:
        """Mock strategic decision making."""
        decision = {
            'action': 'optimize_performance',
            'priority': random.choice(['high', 'medium', 'low']),
            'confidence': random.uniform(0.6, 0.9),
            'reasoning': 'Performance optimization needed based on delta analysis',
            'decision_id': f"decision_{len(self.decisions)}"
        }
        self.decisions.append(decision)
        return decision
    
    async def update_system_awareness(self, new_data: Dict) -> None:
        """Mock system awareness update."""
        self.system_state.update(new_data)
    
    async def get_system_state(self) -> Dict:
        """Get current system state."""
        return self.system_state.copy()
    
    async def get_adaptation_history(self) -> List[Dict]:
        """Get adaptation history."""
        return self.adaptation_history.copy()
    
    async def is_emergency_stopped(self) -> bool:
        """Check if emergency stopped."""
        return self.emergency_stopped
    
    async def observe_system_state(self) -> Dict:
        """Observe current system state."""
        return {
            'performance': random.uniform(0.7, 0.9),
            'stability': random.uniform(0.8, 0.95),
            'resource_usage': random.uniform(0.5, 0.8),
            'timestamp': time.time()
        }
    
    async def make_adaptation_decision(self, evaluation_result: Dict) -> Dict:
        """Make adaptation decision based on evaluation."""
        decision = {
            'should_adapt': evaluation_result.get('should_adapt', False),
            'strategy': 'performance_optimization',
            'confidence': evaluation_result.get('confidence', 0.8),
            'timestamp': time.time()
        }
        return decision
    
    async def apply_adaptation(self, strategy: str) -> Dict:
        """Apply adaptation strategy."""
        result = {
            'strategy': strategy,
            'success': True,
            'changes_applied': ['memory_optimization', 'task_prioritization'],
            'timestamp': time.time()
        }
        self.adaptation_history.append({
            'trigger': 'delta_evaluation',
            'strategy': strategy,
            'result': result,
            'timestamp': time.time()
        })
        return result
    
    async def make_memory_informed_decision(self, context: Dict, memories: List) -> Dict:
        """Make decision informed by memory."""
        return {
            'selected_strategy': 'A',
            'confidence': 0.85,
            'risk_assessment': 0.05,
            'reasoning': 'Based on successful past experiences'
        }
    
    async def get_oversight_reports(self) -> List[Dict]:
        """Get oversight reports."""
        return [
            {'task_id': 'task_1', 'status': 'monitored', 'timestamp': time.time()},
            {'task_id': 'task_2', 'status': 'monitored', 'timestamp': time.time()},
            {'task_id': 'task_3', 'status': 'monitored', 'timestamp': time.time()}
        ]
    
    async def analyze_task_outcomes(self, task_results: List) -> Dict:
        """Analyze task outcomes."""
        return {
            'efficiency_score': 0.85,
            'coordination_quality': 0.9,
            'total_tasks': len(task_results),
            'success_rate': 0.95
        }
    
    async def process_system_event(self, event: Dict) -> None:
        """Process system event."""
        # Mock event processing
        pass
    
    async def process_performance_data(self, data: Dict) -> None:
        """Process performance data."""
        # Mock performance data processing
        pass
    
    async def emergency_stop(self) -> None:
        """Emergency stop the meta-cognitive engine."""
        self.emergency_stopped = True
        self.system_state['status'] = 'emergency_stopped'

class MockHybridMemory:
    """Mock implementation of Hybrid Memory for testing."""
    
    def __init__(self):
        self.memory_store = {}
        self.access_patterns = []
        
    async def store_experience(self, experience_id: str, data: Dict) -> bool:
        """Mock experience storage."""
        self.memory_store[experience_id] = {
            'data': data,
            'timestamp': time.time(),
            'access_count': 0
        }
        return True
    
    async def retrieve_experience(self, experience_id: str) -> Optional[Dict]:
        """Mock experience retrieval."""
        if experience_id in self.memory_store:
            self.memory_store[experience_id]['access_count'] += 1
            self.access_patterns.append(experience_id)
            return self.memory_store[experience_id]['data']
        return None
    
    async def search_similar_experiences(self, query: Dict, limit: int = 5) -> List[Dict]:
        """Mock similarity search."""
        # Return random subset of stored experiences
        experiences = list(self.memory_store.values())
        return random.sample(experiences, min(limit, len(experiences)))
    
    async def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'total_experiences': len(self.memory_store),
            'total_accesses': len(self.access_patterns),
            'memory_usage': len(self.memory_store) * 0.1  # Mock usage
        }
    
    async def store_node(self, node) -> bool:
        """Mock node storage for memory system."""
        node_id = getattr(node, 'id', f"node_{len(self.memory_store)}")
        self.memory_store[node_id] = {
            'node': node,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        # Trigger optimization if near capacity
        if len(self.memory_store) >= 900:
            # Simulate optimization trigger
            if hasattr(self, '_memory_optimizer') and self._memory_optimizer:
                nodes_before = len(self.memory_store)
                
                # Simulate memory pruning by removing 10% of oldest nodes
                sorted_nodes = sorted(self.memory_store.items(),
                                    key=lambda x: x[1]['timestamp'])
                nodes_to_remove = int(len(sorted_nodes) * 0.1)
                
                for i in range(nodes_to_remove):
                    node_id = sorted_nodes[i][0]
                    del self.memory_store[node_id]
                
                optimization_record = {
                    'timestamp': time.time(),
                    'trigger': 'capacity_threshold',
                    'nodes_before': nodes_before,
                    'nodes_after': len(self.memory_store),
                    'nodes_pruned': nodes_to_remove,
                    'action': 'compression'
                }
                self._memory_optimizer.optimization_history.append(optimization_record)
        
        return True
    
    async def get_node_count(self) -> int:
        """Get the current number of stored nodes."""
        return len(self.memory_store)

class MockMCPOrchestrator:
    """Mock implementation of MCP Orchestrator for testing."""
    
    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = []
        self.agents = ['agent_1', 'agent_2', 'agent_3']
        self._emergency_stopped = False
        
    async def orchestrate_task(self, task: Dict) -> str:
        """Mock task orchestration."""
        task_id = f"task_{len(self.active_tasks) + len(self.completed_tasks)}"
        self.active_tasks[task_id] = {
            'task': task,
            'status': 'running',
            'assigned_agent': random.choice(self.agents),
            'start_time': time.time()
        }
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict:
        """Get task status."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in [t['task_id'] for t in self.completed_tasks]:
            return next(t for t in self.completed_tasks if t['task_id'] == task_id)
        return {'status': 'not_found'}
    
    async def complete_task(self, task_id: str, result: Dict) -> bool:
        """Mock task completion."""
        if task_id in self.active_tasks:
            task_data = self.active_tasks.pop(task_id)
            task_data.update({
                'task_id': task_id,
                'status': 'completed',
                'result': result,
                'end_time': time.time()
            })
            self.completed_tasks.append(task_data)
            return True
        return False
    
    async def are_all_tasks_halted(self):
        """Check if all MCP tasks are halted."""
        return self._emergency_stopped
    
    def emergency_stop(self):
        """Emergency stop all tasks."""
        self._emergency_stopped = True
        # Move all active tasks to completed with emergency stop status
        for task_id, task_data in list(self.active_tasks.items()):
            task_data.update({
                'task_id': task_id,
                'status': 'emergency_stopped',
                'end_time': time.time()
            })
            self.completed_tasks.append(task_data)
        self.active_tasks.clear()

class MockSafetyValidator:
    """Mock implementation of Safety Validator for testing."""
    
    def __init__(self):
        self.safety_constraints = {
            'memory_usage_limit': 0.9,
            'delta_magnitude_limit': 0.5,
            'task_execution_time_limit': 30.0
        }
        self.violations = []
        self.emergency_stop_active = False
        
    async def validate_safety_constraints(self, operation: Dict) -> Dict:
        """Mock safety constraint validation."""
        violations = []
        
        # Mock constraint checks
        if operation.get('memory_usage', 0) > self.safety_constraints['memory_usage_limit']:
            violations.append('memory_usage_exceeded')
        
        if operation.get('delta_magnitude', 0) > self.safety_constraints['delta_magnitude_limit']:
            violations.append('delta_magnitude_exceeded')
        
        result = {
            'valid': len(violations) == 0,
            'violations': violations,
            'safety_score': random.uniform(0.7, 1.0) if len(violations) == 0 else random.uniform(0.0, 0.3)
        }
        
        if violations:
            self.violations.extend(violations)
        
        return result
    
    async def trigger_emergency_stop(self, reason: str) -> Dict:
        """Mock emergency stop trigger."""
        self.emergency_stop_active = True
        return {
            'emergency_stop_triggered': True,
            'reason': reason,
            'timestamp': time.time()
        }
    
    async def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active."""
        return self.emergency_stop_active
    
    async def reset_emergency_stop(self) -> bool:
        """Reset emergency stop."""
        self.emergency_stop_active = False
        return True
    
    async def validate_system_state(self, system_data: Dict) -> Dict:
        """Validate system state and trigger emergency stop if needed."""
        violations = []
        
        # Check for critical violations
        performance_drop = system_data.get('performance_drop', 0)
        if performance_drop >= 0.5:  # Critical threshold
            violations.append('critical_performance_drop')
        
        memory_usage = system_data.get('memory_usage', 0)
        if memory_usage > self.safety_constraints['memory_usage_limit']:
            violations.append('memory_usage_exceeded')
        
        delta_magnitude = system_data.get('delta_magnitude', 0)
        if delta_magnitude > self.safety_constraints['delta_magnitude_limit']:
            violations.append('delta_magnitude_exceeded')
        
        # If critical violations detected, trigger emergency stop
        if violations:
            self.violations.extend(violations)
            
            # Log safety violation event
            if hasattr(self, '_context'):
                self._context.log_event('safety_violation', {
                    'violations': violations,
                    'system_data': system_data,
                    'timestamp': time.time()
                })
            
            # Trigger emergency stop for critical violations
            if 'critical_performance_drop' in violations:
                await self.trigger_emergency_stop(f"Critical violations detected: {violations}")
                
                # Propagate emergency stop to connected components
                if hasattr(self, '_meta_engine') and self._meta_engine:
                    await self._meta_engine.emergency_stop()
                
                if hasattr(self, '_mcp_orchestrator') and self._mcp_orchestrator:
                    self._mcp_orchestrator.emergency_stop()
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'emergency_stop_triggered': self.emergency_stop_active
        }
    

class MockMemoryOptimizer:
    """Mock implementation of Memory Optimizer for testing."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_strategy = 'balanced'
        
    async def optimize_memory_usage(self, current_usage: float) -> Dict:
        """Mock memory optimization."""
        optimization = {
            'strategy': random.choice(['aggressive', 'balanced', 'conservative']),
            'memory_freed': random.uniform(0.1, 0.3),
            'optimization_time': random.uniform(0.5, 2.0),
            'success': True
        }
        self.optimization_history.append(optimization)
        return optimization
    
    async def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        return [
            'compress_old_memories',
            'archive_unused_experiences',
            'optimize_memory_layout'
        ]
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self.optimization_history.copy()

# Performance Monitoring

class PerformanceMonitor:
    """Performance monitoring for integration tests."""
    
    def __init__(self):
        self.metrics = {}
        self.monitoring_active = False
        self.start_time = None
        
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.monitoring_active = True
        self.start_time = time.time()
        self.metrics = {
            'operations_per_second': 0,
            'average_latency': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'error_rate': 0
        }
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        if self.monitoring_active:
            # Simulate real-time metrics
            self.metrics.update({
                'operations_per_second': random.uniform(50, 150),
                'average_latency': random.uniform(100, 500),
                'memory_usage': random.uniform(0.3, 0.8),
                'cpu_usage': random.uniform(0.2, 0.7),
                'error_rate': random.uniform(0.0, 0.05)
            })
        return self.metrics.copy()
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        return {
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
            'metrics': self.get_metrics(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        metrics = self.get_metrics()
        
        if metrics['average_latency'] > 300:
            recommendations.append('Optimize response time')
        if metrics['memory_usage'] > 0.7:
            recommendations.append('Reduce memory consumption')
        if metrics['error_rate'] > 0.02:
            recommendations.append('Investigate error sources')
            
        return recommendations

# Error Injection

class ErrorInjector:
    """Error injection for resilience testing."""
    
    def __init__(self):
        self.active_failures = {}
        self.failure_history = []
        
    async def inject_component_failure(
        self, 
        component: str, 
        failure_type: str, 
        duration: float = 5.0
    ) -> Dict:
        """Inject component failure."""
        failure_id = f"failure_{len(self.failure_history)}"
        failure_data = {
            'failure_id': failure_id,
            'component': component,
            'failure_type': failure_type,
            'duration': duration,
            'start_time': time.time(),
            'active': True
        }
        
        self.active_failures[failure_id] = failure_data
        self.failure_history.append(failure_data)
        
        # Schedule failure recovery
        asyncio.create_task(self._recover_from_failure(failure_id, duration))
        
        return failure_data
    
    async def inject_network_failure(
        self, 
        failure_type: str = 'intermittent', 
        severity: float = 0.3
    ) -> Dict:
        """Inject network failure."""
        return await self.inject_component_failure(
            'network', failure_type, duration=severity * 10
        )
    
    async def inject_memory_pressure(self, pressure_level: float = 0.9) -> Dict:
        """Inject memory pressure."""
        return await self.inject_component_failure(
            'memory', 'pressure', duration=pressure_level * 5
        )
    
    async def inject_cpu_overload(self, load_level: float = 0.9) -> Dict:
        """Inject CPU overload."""
        return await self.inject_component_failure(
            'cpu', 'overload', duration=load_level * 5
        )
    
    async def _recover_from_failure(self, failure_id: str, duration: float) -> None:
        """Recover from injected failure after duration."""
        await asyncio.sleep(duration)
        if failure_id in self.active_failures:
            self.active_failures[failure_id]['active'] = False
            self.active_failures[failure_id]['end_time'] = time.time()
    
    def get_active_failures(self) -> List[Dict]:
        """Get currently active failures."""
        return [f for f in self.active_failures.values() if f['active']]

# Test Data Generation

class TestDataGenerator:
    """Generate realistic test data for integration tests."""
    
    def __init__(self):
        self.data_cache = {}
        
    def generate_performance_data(self, count: int = 100) -> List[Dict]:
        """Generate performance test data."""
        if 'performance' not in self.data_cache:
            self.data_cache['performance'] = [
                {
                    'timestamp': time.time() - random.uniform(0, 3600),
                    'operation_type': random.choice(['read', 'write', 'compute', 'network']),
                    'latency': random.uniform(10, 1000),
                    'throughput': random.uniform(10, 500),
                    'success': random.choice([True, True, True, False])  # 75% success rate
                }
                for _ in range(count)
            ]
        return self.data_cache['performance'][:count]
    
    def generate_memory_data(self, count: int = 50) -> List[Dict]:
        """Generate memory test data."""
        if 'memory' not in self.data_cache:
            self.data_cache['memory'] = [
                {
                    'experience_id': f"exp_{i}",
                    'data': {
                        'context': f"test_context_{i}",
                        'outcome': random.choice(['success', 'failure', 'partial']),
                        'confidence': random.uniform(0.5, 1.0),
                        'metadata': {'source': 'test_generator'}
                    },
                    'similarity_vector': [random.uniform(0, 1) for _ in range(10)]
                }
                for i in range(count)
            ]
        return self.data_cache['memory'][:count]
    
    def generate_task_data(self, count: int = 20) -> List[Dict]:
        """Generate task test data."""
        if 'tasks' not in self.data_cache:
            self.data_cache['tasks'] = [
                {
                    'task_id': f"task_{i}",
                    'type': random.choice(['optimization', 'analysis', 'coordination', 'validation']),
                    'priority': random.choice(['high', 'medium', 'low']),
                    'complexity': random.uniform(0.1, 1.0),
                    'estimated_duration': random.uniform(1, 30),
                    'dependencies': random.sample([f"task_{j}" for j in range(i)], min(3, i))
                }
                for i in range(count)
            ]
        return self.data_cache['tasks'][:count]
    
    def generate_safety_test_data(self, count: int = 30) -> List[Dict]:
        """Generate safety test data."""
        if 'safety' not in self.data_cache:
            self.data_cache['safety'] = [
                {
                    'operation_id': f"op_{i}",
                    'operation_type': random.choice(['memory_access', 'delta_evaluation', 'task_execution']),
                    'memory_usage': random.uniform(0.1, 1.0),
                    'delta_magnitude': random.uniform(0.0, 0.8),
                    'execution_time': random.uniform(0.5, 45.0),
                    'safety_critical': random.choice([True, False]),
                    'expected_violations': random.sample(['memory', 'delta', 'time'], random.randint(0, 2))
                }
                for i in range(count)
            ]
        return self.data_cache['safety'][:count]

# Integration Test Context

class IntegrationTestContext:
    """Main integration test context manager."""
    
    def __init__(self, environment: str = 'test'):
        self.environment = environment
        self.components = {}
        self.performance_monitor = PerformanceMonitor()
        self.error_injector = ErrorInjector()
        self.test_data_generator = TestDataGenerator()
        self.logger = logging.getLogger(f"integration_test_{environment}")
        self.config = {
            'meta_cognitive': {
                'adaptation_threshold': 0.3,
                'confidence_threshold': 0.7,
                'max_adaptations_per_cycle': 5
            },
            'safety': {
                'max_error_rate': 0.1,
                'emergency_stop_threshold': 0.05,
                'validation_timeout': 30
            },
            'performance': {
                'target_throughput': 100,
                'max_latency': 2.0,
                'resource_utilization_limit': 0.8
            }
        }
        self.events = []
        
    async def setup_components(self) -> None:
        """Set up all mock components for testing."""
        self.components = {
            'delta_evaluator': MockDeltaEvaluator(),
            'meta_cognitive_engine': MockMetaCognitiveEngine(),
            'hybrid_memory': MockHybridMemory(),
            'mcp_orchestrator': MockMCPOrchestrator(),
            'safety_validator': MockSafetyValidator(),
            'memory_optimizer': MockMemoryOptimizer()
        }
        
        # Wire up component integration
        self.components['delta_evaluator'].set_meta_engine(self.components['meta_cognitive_engine'])
        
        # Wire safety validator to meta-cognitive engine and MCP orchestrator
        self.components['safety_validator']._meta_engine = self.components['meta_cognitive_engine']
        self.components['safety_validator']._mcp_orchestrator = self.components['mcp_orchestrator']
        self.components['safety_validator']._context = self
        
        # Wire memory optimizer to hybrid memory
        self.components['hybrid_memory']._memory_optimizer = self.components['memory_optimizer']
        
        # Initialize components
        for component_name, component in self.components.items():
            self.logger.info(f"Initialized {component_name}")
    
    def log_event(self, event_type: str, data: dict) -> None:
        """Log an event for testing purposes."""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.events.append(event)
        self.logger.info(f"Event logged: {event_type}")
    
    async def cleanup_components(self) -> None:
        """Clean up all components after testing."""
        for component_name in self.components:
            self.logger.info(f"Cleaning up {component_name}")
        self.components.clear()
    
    def get_component(self, component_name: str):
        """Get a specific component by name."""
        return self.components.get(component_name)
    
    async def wait_for_system_stability(self, timeout: float = 10.0) -> bool:
        """Wait for system to reach stable state."""
        async def check_stability():
            # Mock stability check
            safety_validator = self.get_component('safety_validator')
            if safety_validator and await safety_validator.is_emergency_stop_active():
                return False
            
            # Check if all components are responsive
            for component in self.components.values():
                if hasattr(component, 'get_system_state'):
                    state = await component.get_system_state()
                    if state.get('status') != 'operational':
                        return False
            
            return True
        
        return await IntegrationTestHelpers.wait_for_condition(
            check_stability, timeout=timeout
        )

# Pytest Fixtures

@pytest_asyncio.fixture
async def integration_context() -> AsyncGenerator[IntegrationTestContext, None]:
    """Provide integration test context."""
    context = IntegrationTestContext()
    await context.setup_components()
    try:
        yield context
    finally:
        await context.cleanup_components()

@pytest_asyncio.fixture
async def integrated_system(integration_context: IntegrationTestContext) -> Dict:
    """Provide fully integrated SAFLA system for testing."""
    # Wait for system stability
    await integration_context.wait_for_system_stability()
    
    return {
        'context': integration_context,
        'components': integration_context.components,
        'performance_monitor': integration_context.performance_monitor,
        'error_injector': integration_context.error_injector,
        'test_data_generator': integration_context.test_data_generator
    }

@pytest.fixture
def mock_external_services() -> Dict:
    """Provide mock external services."""
    return {
        'database': AsyncMock(),
        'message_queue': AsyncMock(),
        'monitoring_service': AsyncMock(),
        'logging_service': AsyncMock()
    }

@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Provide performance monitor instance."""
    return PerformanceMonitor()

@pytest.fixture
def error_injector() -> ErrorInjector:
    """Provide error injector instance."""
    return ErrorInjector()

@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator instance."""
    return TestDataGenerator()

# Test Configuration

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )