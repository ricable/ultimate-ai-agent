"""
System Integration Test Suite for SAFLA
========================================

This module implements comprehensive integration testing for the Self-Aware Feedback Loop Algorithm,
validating that all core components work together seamlessly in end-to-end scenarios.

Integration Testing Scope:
1. Component Integration Tests - Verify all components work together
2. End-to-End Workflow Tests - Test complete SAFLA feedback loops
3. Cross-Component Communication - Validate event-driven communication
4. Performance Integration - Test system performance under load
5. Safety Integration - Verify safety constraints across components

Key Integration Scenarios:
- Complete self-improvement cycle: observation → evaluation → decision → modification → validation
- Memory-driven decision making with safety constraints
- MCP orchestration coordinating multiple agents with meta-cognitive oversight
- Performance monitoring triggering adaptive strategy changes
- Emergency safety stops propagating across all components
- Cross-component error handling and recovery

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import time
import threading
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Mock data classes for testing (will be replaced with real imports when implemented)
@dataclass
class DeltaMetrics:
    performance: float
    efficiency: float
    stability: float
    capability: float
    confidence: float = 0.8
    metadata: dict = None

@dataclass
class DeltaResult:
    success: bool
    metrics: DeltaMetrics
    total_delta: float
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.8
    should_adapt: bool = False
    evaluation_id: str = ""

@dataclass
class MemoryNode:
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyConstraint:
    id: str
    constraint_type: str
    threshold: float
    action: str

@dataclass
class SafetyResult:
    success: bool
    violations: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)


@dataclass
class IntegrationTestContext:
    """Context for integration tests with shared state and configuration."""
    test_id: str
    temp_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event for analysis."""
        self.events.append({
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        })


class TestSystemIntegration:
    """
    Comprehensive system integration tests for SAFLA.
    
    Tests the complete system working together including:
    - Component initialization and coordination
    - End-to-end feedback loops
    - Cross-component communication
    - Error handling and recovery
    - Performance under load
    - Safety constraint enforcement
    """
    
    @pytest.fixture
    def integration_context(self):
        """Create integration test context with temporary directory and configuration."""
        temp_dir = Path(tempfile.mkdtemp(prefix="safla_integration_"))
        context = IntegrationTestContext(
            test_id=f"integration_{int(time.time())}",
            temp_dir=temp_dir,
            config={
                'delta_evaluation': {
                    'weights': {'performance': 0.3, 'efficiency': 0.25, 'stability': 0.25, 'capability': 0.2},
                    'thresholds': {'improvement': 0.01, 'significant': 0.2}
                },
                'meta_cognitive': {
                    'monitoring_interval': 0.1,
                    'adaptation_threshold': 0.15,
                    'safety_checks_enabled': True
                },
                'memory': {
                    'max_nodes': 1000,
                    'pruning_threshold': 0.8,
                    'embedding_dim': 128
                },
                'mcp': {
                    'max_agents': 5,
                    'timeout': 30.0,
                    'retry_attempts': 3
                },
                'safety': {
                    'max_modification_rate': 0.1,
                    'critical_thresholds': {'performance_drop': 0.3, 'stability_loss': 0.4}
                }
            }
        )
        
        yield context
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def integrated_system(self, integration_context):
        """Create a fully integrated SAFLA system for testing."""
        context = integration_context
        
        # Initialize all core components
        delta_evaluator = DeltaEvaluator(config=context.config['delta_evaluation'])
        meta_engine = MetaCognitiveEngine(config=context.config['meta_cognitive'])
        memory_system = HybridMemorySystem(config=context.config['memory'])
        mcp_orchestrator = MCPOrchestrator(config=context.config['mcp'])
        safety_validator = SafetyValidator(config=context.config['safety'])
        memory_optimizer = MemoryOptimizer()
        
        # Store components in context
        context.components = {
            'delta_evaluator': delta_evaluator,
            'meta_engine': meta_engine,
            'memory_system': memory_system,
            'mcp_orchestrator': mcp_orchestrator,
            'safety_validator': safety_validator,
            'memory_optimizer': memory_optimizer
        }
        
        # Initialize the integrated system
        await meta_engine.start()
        await memory_system.initialize()
        await mcp_orchestrator.start()
        
        # Configure cross-component communication
        await self._setup_component_integration(context)
        
        yield context
        
        # Cleanup
        await meta_engine.stop()
        await memory_system.shutdown()
        await mcp_orchestrator.stop()
    
    async def _setup_component_integration(self, context: IntegrationTestContext):
        """Setup integration between components."""
        components = context.components
        
        # Connect meta-engine to other components
        meta_engine = components['meta_engine']
        meta_engine.register_component('delta_evaluator', components['delta_evaluator'])
        meta_engine.register_component('memory_system', components['memory_system'])
        meta_engine.register_component('mcp_orchestrator', components['mcp_orchestrator'])
        meta_engine.register_component('safety_validator', components['safety_validator'])
        meta_engine.register_component('memory_optimizer', components['memory_optimizer'])
        
        # Setup event handlers for cross-component communication
        await self._setup_event_handlers(context)
    
    async def _setup_event_handlers(self, context: IntegrationTestContext):
        """Setup event handlers for component communication."""
        components = context.components
        
        # Delta evaluation triggers meta-cognitive adaptation
        components['delta_evaluator'].on_significant_change = lambda delta: \
            asyncio.create_task(components['meta_engine'].trigger_adaptation(delta))
        
        # Memory system triggers optimization
        components['memory_system'].on_capacity_threshold = lambda usage: \
            asyncio.create_task(components['memory_optimizer'].optimize(usage))
        
        # Safety validator can halt operations
        components['safety_validator'].on_safety_violation = lambda violation: \
            asyncio.create_task(self._handle_safety_violation(context, violation))
        
        # MCP orchestrator reports to meta-engine
        components['mcp_orchestrator'].on_task_completion = lambda result: \
            asyncio.create_task(components['meta_engine'].process_task_result(result))
    
    async def _handle_safety_violation(self, context: IntegrationTestContext, violation):
        """Handle safety violations across all components."""
        context.log_event('safety_violation', {'violation': str(violation)})
        
        # Emergency stop all components
        components = context.components
        await components['meta_engine'].emergency_stop()
        await components['mcp_orchestrator'].halt_all_tasks()


class TestComponentIntegration:
    """Test integration between individual components."""
    
    @pytest.mark.asyncio
    async def test_delta_evaluation_meta_cognitive_integration(self, integrated_system):
        """Test integration between delta evaluation and meta-cognitive engine."""
        components = integrated_system['components']
        context = integrated_system['context']
        delta_evaluator = components['delta_evaluator']
        meta_engine = components['meta_cognitive_engine']
        
        # Create test metrics that should trigger adaptation
        metrics = DeltaMetrics(
            performance=0.25,  # Significant improvement
            efficiency=0.15,
            stability=0.05,
            capability=0.20,
            confidence=0.9
        )
        
        # Evaluate delta - should trigger meta-cognitive adaptation
        result = await delta_evaluator.evaluate(metrics)
        
        # Verify integration
        assert result.total_delta > context.config['meta_cognitive']['adaptation_threshold']
        
        # Wait for adaptation to be triggered
        await asyncio.sleep(0.2)
        
        # Verify meta-engine received and processed the delta
        adaptation_history = await meta_engine.get_adaptation_history()
        assert len(adaptation_history) > 0
        assert adaptation_history[-1]['trigger'] == 'delta_evaluation'
    
    @pytest.mark.asyncio
    async def test_memory_system_optimization_integration(self, integrated_system):
        """Test integration between memory system and memory optimizer."""
        context = integrated_system
        memory_system = context['components']['hybrid_memory']
        memory_optimizer = context['components']['memory_optimizer']
        
        # Fill memory system to trigger optimization
        for i in range(900):  # Near capacity threshold
            node = MemoryNode(
                id=f"test_node_{i}",
                content=f"Test content {i}",
                embedding=[0.1] * 128,
                metadata={'test': True}
            )
            await memory_system.store_node(node)
        
        # Verify optimization was triggered
        await asyncio.sleep(0.1)
        
        optimization_history = memory_optimizer.get_optimization_history()
        assert len(optimization_history) > 0
        assert optimization_history[-1]['trigger'] == 'capacity_threshold'
        
        # Verify memory was actually optimized
        current_count = await memory_system.get_node_count()
        assert current_count < 900  # Some nodes should have been pruned
    
    @pytest.mark.asyncio
    async def test_safety_validator_system_halt_integration(self, integrated_system):
        """Test safety validator can halt the entire system."""
        context = integrated_system
        safety_validator = context['components']['safety_validator']
        meta_engine = context['components']['meta_cognitive_engine']
        mcp_orchestrator = context['components']['mcp_orchestrator']
        
        # Create a safety violation
        violation_data = {
            'performance_drop': 0.5,  # Exceeds critical threshold
            'component': 'test_component'
        }
        
        # Trigger safety violation
        await safety_validator.validate_system_state(violation_data)
        
        # Wait for safety response
        await asyncio.sleep(0.1)
        
        # Verify system was halted
        assert await meta_engine.is_emergency_stopped()
        assert await mcp_orchestrator.are_all_tasks_halted()
        
        # Verify safety event was logged
        safety_events = [e for e in context['context'].events if e['type'] == 'safety_violation']
        assert len(safety_events) > 0


class TestEndToEndWorkflows:
    """Test complete end-to-end SAFLA workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_self_improvement_cycle(self, integrated_system):
        """Test complete self-improvement cycle: observation → evaluation → decision → modification → validation."""
        context = integrated_system['context']
        components = integrated_system['components']
        
        # Phase 1: Observation - System observes its current state
        initial_state = await components['meta_cognitive_engine'].observe_system_state()
        context.log_event('observation', {'state': initial_state})
        
        # Phase 2: Evaluation - Delta evaluation of current performance
        performance_metrics = {
            'task_completion_rate': 0.85,
            'response_time': 1.2,
            'error_rate': 0.05,
            'resource_utilization': 0.7
        }
        
        delta_metrics = DeltaMetrics(
            performance=0.1,
            efficiency=0.05,
            stability=0.02,
            capability=0.08,
            confidence=0.9,
            metadata=performance_metrics
        )
        
        evaluation_result = await components['delta_evaluator'].evaluate(delta_metrics)
        context.log_event('evaluation', {'result': evaluation_result})
        
        # Phase 3: Decision - Meta-cognitive engine decides on adaptation
        decision = await components['meta_cognitive_engine'].make_adaptation_decision(evaluation_result)
        context.log_event('decision', {'decision': decision})
        
        # Phase 4: Modification - Apply the decided modifications
        if decision['should_adapt']:
            modification_result = await components['meta_cognitive_engine'].apply_adaptation(decision['strategy'])
            context.log_event('modification', {'result': modification_result})
            
            # Phase 5: Validation - Validate the modifications are safe and effective
            validation_result = await components['safety_validator'].validate_modification(modification_result)
            context.log_event('validation', {'result': validation_result})
            
            assert validation_result.is_safe
            assert validation_result.effectiveness_score > 0.5
        
        # Verify complete cycle was logged
        event_types = [e['type'] for e in context.events]
        expected_phases = ['observation', 'evaluation', 'decision']
        for phase in expected_phases:
            assert phase in event_types
    
    @pytest.mark.asyncio
    async def test_memory_driven_decision_making_with_safety(self, integrated_system):
        """Test memory-driven decision making with safety constraints."""
        context = integrated_system['context']
        components = integrated_system['components']
        
        # Store relevant memories
        memories = [
            MemoryNode(
                id="strategy_success_1",
                content="Strategy A led to 20% performance improvement",
                embedding=[0.8, 0.2] + [0.0] * 126,
                metadata={'strategy': 'A', 'outcome': 'success', 'improvement': 0.2}
            ),
            MemoryNode(
                id="strategy_failure_1",
                content="Strategy B caused system instability",
                embedding=[0.2, 0.8] + [0.0] * 126,
                metadata={'strategy': 'B', 'outcome': 'failure', 'stability_impact': -0.3}
            )
        ]
        
        for memory in memories:
            await components['hybrid_memory'].store_node(memory)
        
        # Create decision context requiring memory lookup
        decision_context = {
            'current_performance': 0.7,
            'available_strategies': ['A', 'B', 'C'],
            'constraints': {'max_risk': 0.1}
        }
        
        # Query memory for relevant experiences
        query = MemoryQuery(
            content="strategy performance outcomes",
            embedding=[0.5, 0.5] + [0.0] * 126,
            filters={'outcome': ['success', 'failure']}
        )
        
        relevant_memories = await components['hybrid_memory'].query(query, limit=10)
        
        # Make decision based on memory and safety constraints
        decision = await components['meta_cognitive_engine'].make_memory_informed_decision(
            decision_context, relevant_memories
        )
        
        # Validate decision against safety constraints
        safety_result = await components['safety_validator'].validate_decision(decision)
        
        # Verify memory-informed decision
        assert decision['selected_strategy'] == 'A'  # Should choose successful strategy
        assert decision['confidence'] > 0.7
        assert safety_result.is_safe
        
        # Verify safety constraints were considered
        assert decision['risk_assessment'] <= decision_context['constraints']['max_risk']
    
    @pytest.mark.asyncio
    async def test_mcp_orchestration_with_meta_cognitive_oversight(self, integrated_system):
        """Test MCP orchestration coordinating multiple agents with meta-cognitive oversight."""
        context = integrated_system['context']
        components = integrated_system['components']
        
        # Create multiple agent tasks
        tasks = [
            AgentTask(
                id="task_1",
                type="analysis",
                payload={"data": "performance_metrics", "analysis_type": "trend"},
                priority=1
            ),
            AgentTask(
                id="task_2", 
                type="optimization",
                payload={"target": "memory_usage", "strategy": "pruning"},
                priority=2
            ),
            AgentTask(
                id="task_3",
                type="validation",
                payload={"component": "safety_validator", "test_suite": "integration"},
                priority=3
            )
        ]
        
        # Submit tasks to MCP orchestrator
        task_results = []
        for task in tasks:
            result = await components['mcp_orchestrator'].submit_task(task)
            task_results.append(result)
        
        # Wait for task completion
        await asyncio.sleep(0.5)
        
        # Verify meta-cognitive oversight
        oversight_reports = await components['meta_cognitive_engine'].get_oversight_reports()
        assert len(oversight_reports) >= len(tasks)
        
        # Verify task coordination
        for i, result in enumerate(task_results):
            assert result.task_id == tasks[i].id
            assert result.status in ['completed', 'in_progress']
        
        # Verify meta-cognitive analysis of task outcomes
        task_analysis = await components['meta_cognitive_engine'].analyze_task_outcomes(task_results)
        assert 'efficiency_score' in task_analysis
        assert 'coordination_quality' in task_analysis
        assert task_analysis['efficiency_score'] > 0.5


class TestCrossComponentCommunication:
    """Test event-driven communication and data flow between components."""
    
    @pytest.mark.asyncio
    async def test_event_propagation_across_components(self, integrated_system):
        """Test that events propagate correctly across all components."""
        context = integrated_system['context']
        components = integrated_system['components']
        
        # Create an event that should propagate through multiple components
        initial_event = {
            'type': 'performance_degradation',
            'severity': 'high',
            'affected_components': ['memory_system', 'mcp_orchestrator'],
            'metrics': {'performance_drop': 0.25, 'error_rate_increase': 0.15}
        }
        
        # Trigger event in meta-cognitive engine
        await components['meta_cognitive_engine'].process_system_event(initial_event)
        
        # Wait for event propagation
        await asyncio.sleep(0.2)
        
        # Verify event was received by relevant components
        memory_events = await components['hybrid_memory'].get_received_events()
        mcp_events = await components['mcp_orchestrator'].get_received_events()
        safety_events = await components['safety_validator'].get_received_events()
        
        # Verify event propagation
        assert any(e['type'] == 'performance_degradation' for e in memory_events)
        assert any(e['type'] == 'performance_degradation' for e in mcp_events)
        assert any(e['type'] == 'performance_degradation' for e in safety_events)
        
        # Verify appropriate responses were triggered
        memory_responses = await components['hybrid_memory'].get_event_responses()
        assert any(r['action'] == 'optimize_performance' for r in memory_responses)
        
        mcp_responses = await components['mcp_orchestrator'].get_event_responses()
        assert any(r['action'] == 'reduce_task_load' for r in mcp_responses)
    
    @pytest.mark.asyncio
    async def test_data_flow_consistency(self, integrated_system):
        """Test that data flows consistently between components without corruption."""
        context = integrated_system['context']
        components = integrated_system['components']
        
        # Create test data that will flow through multiple components
        test_data = {
            'session_id': 'test_session_123',
            'performance_metrics': {
                'cpu_usage': 0.65,
                'memory_usage': 0.78,
                'task_completion_rate': 0.92,
                'error_rate': 0.03
            },
            'context': {
                'user_goals': ['optimize_performance', 'maintain_stability'],
                'constraints': ['max_cpu_80', 'max_memory_85'],
                'timestamp': time.time()
            }
        }
        
        # Flow 1: Meta-engine → Delta evaluator → Memory system
        await components['meta_cognitive_engine'].process_performance_data(test_data)
        delta_result = await components['delta_evaluator'].evaluate_from_metrics(test_data['performance_metrics'])
        memory_storage_result = await components['hybrid_memory'].store_evaluation_result(delta_result)
        
        # Flow 2: Memory system → MCP orchestrator → Safety validator
        retrieved_data = await components['hybrid_memory'].retrieve_by_session(test_data['session_id'])
        mcp_task = await components['mcp_orchestrator'].create_task_from_data(retrieved_data)
        safety_validation = await components['safety_validator'].validate_task(mcp_task)
        
        # Verify data consistency throughout the flow
        assert retrieved_data['session_id'] == test_data['session_id']
        assert retrieved_data['performance_metrics']['cpu_usage'] == test_data['performance_metrics']['cpu_usage']
        assert mcp_task.metadata['session_id'] == test_data['session_id']
        assert safety_validation.validated_data['session_id'] == test_data['session_id']
        
        # Verify no data corruption occurred
        original_checksum = hash(str(sorted(test_data.items())))
        final_data = safety_validation.validated_data
        final_checksum = hash(str(sorted(final_data.items())))
        
        # Core data should remain consistent (allowing for added metadata)
        assert final_data['performance_metrics'] == test_data['performance_metrics']
        assert final_data['context']['user_goals'] == test_data['context']['user_goals']


class TestPerformanceIntegration:
    """Test system performance under integrated load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, integrated_system):
        """Test system performance with concurrent operations across all components."""
        context = integrated_system
        components = context.components
        
        # Define concurrent operations
        async def memory_operations():
            """Concurrent memory operations."""
            tasks = []
            for i in range(50):
                node = MemoryNode(
                    id=f"perf_test_{i}",
                    content=f"Performance test content {i}",
                    embedding=[0.1 * (i % 10)] * 128,
                    metadata={'test': 'performance', 'batch': i // 10}
                )
                tasks.append(components['memory_system'].store_node(node))
            
            await asyncio.gather(*tasks)
            return len(tasks)
        
        async def delta_evaluations():
            """Concurrent delta evaluations."""
            tasks = []
            for i in range(30):
                metrics = DeltaMetrics(
                    performance=0.1 + (i * 0.01),
                    efficiency=0.05 + (i * 0.005),
                    stability=0.02,
                    capability=0.08,
                    confidence=0.9
                )
                tasks.append(components['delta_evaluator'].evaluate(metrics))
            
            results = await asyncio.gather(*tasks)
            return len(results)
        
        async def mcp_tasks():
            """Concurrent MCP tasks."""
            tasks = []
            for i in range(20):
                task = AgentTask(
                    id=f"perf_task_{i}",
                    type="performance_test",
                    payload={"iteration": i, "data_size": 1000},
                    priority=1
                )
                tasks.append(components['mcp_orchestrator'].submit_task(task))
            
            results = await asyncio.gather(*tasks)
            return len(results)
        
        # Measure performance
        start_time = time.time()
        
        # Run concurrent operations
        memory_count, delta_count, mcp_count = await asyncio.gather(
            memory_operations(),
            delta_evaluations(),
            mcp_tasks()
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all operations completed
        assert memory_count == 50
        assert delta_count == 30
        assert mcp_count == 20
        
        # Verify performance benchmarks
        total_operations = memory_count + delta_count + mcp_count
        operations_per_second = total_operations / total_time
        
        # Performance assertions (adjust thresholds based on requirements)
        assert operations_per_second > 10  # Minimum 10 ops/sec
        assert total_time < 30  # Maximum 30 seconds for all operations
        
        # Verify system stability under load
        system_state = await components['meta_engine'].get_system_state()
        assert system_state['stability_score'] > 0.7
        assert system_state['error_rate'] < 0.1
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, integrated_system):
        """Test system behavior under memory pressure."""
        context = integrated_system
        components = context.components
        
        # Create memory pressure by storing large amounts of data
        large_data_nodes = []
        for i in range(1500):  # Exceed normal capacity
            node = MemoryNode(
                id=f"large_node_{i}",
                content="Large content " * 100,  # Large content
                embedding=[0.1] * 128,
                metadata={'size': 'large', 'pressure_test': True}
            )
            large_data_nodes.append(node)
        
        # Store nodes and monitor system response
        start_memory_usage = await components['memory_system'].get_memory_usage()
        
        for node in large_data_nodes:
            await components['memory_system'].store_node(node)
            
            # Check if optimization was triggered
            if await components['memory_system'].get_memory_usage() > 0.9:
                break
        
        # Wait for optimization to complete
        await asyncio.sleep(1.0)
        
        final_memory_usage = await components['memory_system'].get_memory_usage()
        optimization_events = components['memory_optimizer'].get_optimization_history()
        
        # Verify memory pressure was handled
        assert len(optimization_events) > 0
        assert final_memory_usage < 0.9  # Should be optimized below threshold
        
        # Verify system remained stable
        system_state = await components['meta_engine'].get_system_state()
        assert system_state['status'] == 'operational'
        assert system_state['memory_health'] > 0.6
    
    @pytest.mark.asyncio
    async def test_adaptive_performance_optimization(self, integrated_system):
        """Test that system adapts performance strategies based on load."""
        context = integrated_system
        components = context.components
        
        # Phase 1: Normal load
        normal_load_metrics = []
        for i in range(10):
            start_time = time.time()
            
            # Simulate normal operations
            task = AgentTask(
                id=f"normal_task_{i}",
                type="standard_operation",
                payload={"data_size": 100},
                priority=1
            )
            await components['mcp_orchestrator'].submit_task(task)
            
            end_time = time.time()
            normal_load_metrics.append(end_time - start_time)
        
        normal_avg_time = sum(normal_load_metrics) / len(normal_load_metrics)
        
        # Phase 2: High load - should trigger adaptive optimization
        high_load_metrics = []
        for i in range(50):  # Much higher load
            start_time = time.time()
            
            task = AgentTask(
                id=f"high_load_task_{i}",
                type="standard_operation",
                payload={"data_size": 100},
                priority=1
            )
            await components['mcp_orchestrator'].submit_task(task)
            
            end_time = time.time()
            high_load_metrics.append(end_time - start_time)
        
        high_load_avg_time = sum(high_load_metrics) / len(high_load_metrics)
        
        # Verify adaptive optimization occurred
        adaptation_history = await components['meta_engine'].get_adaptation_history()
        performance_adaptations = [a for a in adaptation_history if a['type'] == 'performance_optimization']
        
        assert len(performance_adaptations) > 0
        
        # Verify performance improved or remained stable under high load
        # (System should adapt to maintain reasonable performance)
        performance_degradation = (high_load_avg_time - normal_avg_time) / normal_avg_time
        assert performance_degradation < 2.0  # Less than 200% degradation


class TestSafetyIntegration:
    """Test safety constraint enforcement across all components."""
    
    @pytest.mark.asyncio
    async def test_safety_constraints_across_components(self, integrated_system):
        """Test that safety constraints are enforced across all components."""
        context = integrated_system
        components = context.components
        
        # Define safety constraints
        safety_constraints = [
            SafetyConstraint(
                name="max_modification_rate",
                threshold=0.1,
                component="meta_engine",
                action="halt_modifications"
            ),
            SafetyConstraint(
                name="memory_corruption_prevention",
                threshold=0.05,
                component="memory_system", 
                action="rollback_changes"
            ),
            SafetyConstraint(
                name="task_failure_rate",
                threshold=0.2,
                component="mcp_orchestrator",
                action="reduce_concurrency"
            )
        ]
        
        # Register constraints with safety validator
        for constraint in safety_constraints:
            await components['safety_validator'].register_constraint(constraint)
        
        # Test 1: Trigger modification rate constraint
        rapid_modifications = []
        for i in range(20):  # Rapid modifications
            modification = {
                'type': 'strategy_update',
                'component': 'meta_engine',
                'change_magnitude': 0.15
            }
            rapid_modifications.append(
                components['meta_engine'].apply_modification(modification)
            )
        
        # Should trigger safety halt
        await asyncio.sleep(0.1)
        
        safety_violations = await components['safety_validator'].get_violations()
        modification_violations = [v for v in safety_violations if v.constraint_name == "max_modification_rate"]
        assert len(modification_violations) > 0
        
        # Verify meta-engine halted modifications
        assert await components['meta_engine'].are_modifications_halted()
        
        # Test 2: Memory corruption prevention
        corrupted_data = {
            'type': 'memory_update',
            'corruption_probability': 0.08  # Above threshold
        }
        
        result = await components['memory_system'].validate_and_store(corrupted_data)
        assert not result.success
        assert result.safety_violation == "memory_corruption_prevention"
        
        # Test 3: Task failure rate constraint
        failing_tasks = []
        for i in range(10):
            task = AgentTask(
                id=f"failing_task_{i}",
                type="guaranteed_failure",  # Task type that will fail
                payload={"should_fail": True},
                priority=1
            )
            failing_tasks.append(components['mcp_orchestrator'].submit_task(task))
        
        await asyncio.sleep(0.5)
        
        # Verify concurrency was reduced
        current_concurrency = await components['mcp_orchestrator'].get_current_concurrency()
        max_concurrency = components['mcp_orchestrator'].get_max_concurrency()
        assert current_concurrency < max_concurrency
    
    @pytest.mark.asyncio
    async def test_emergency_safety_stop_propagation(self, integrated_system):
        """Test that emergency safety stops propagate across all components."""
        context = integrated_system
        components = context.components
        
        # Create critical safety violation
        critical_violation = {
            'type': 'system_instability',
            'severity': 'critical',
            'affected_components': 'all',
            'metrics': {
                'stability_score': 0.1,  # Very low stability
                'error_rate': 0.8,       # Very high error rate
                'performance_drop': 0.9   # Severe performance drop
            }
        }
        
        # Trigger emergency stop
        await components['safety_validator'].trigger_emergency_stop(critical_violation)
        
        # Wait for propagation
        await asyncio.sleep(0.2)
        
        # Verify all components received emergency stop
        assert await components['meta_engine'].is_emergency_stopped()
        assert await components['memory_system'].is_emergency_stopped()
        assert await components['mcp_orchestrator'].is_emergency_stopped()
        assert await components['memory_optimizer'].is_emergency_stopped()
        
        # Verify no new operations are accepted
        with pytest.raises(Exception, match="emergency_stop"):
            await components['meta_engine'].process_system_event({'type': 'test'})
        
        with pytest.raises(Exception, match="emergency_stop"):
            await components['memory_system'].store_node(MemoryNode(id="test", content="test", embedding=[]))
        
        with pytest.raises(Exception, match="emergency_stop"):
            await components['mcp_orchestrator'].submit_task(AgentTask(id="test", type="test", payload={}))
        
        # Test recovery process
        await components['safety_validator'].initiate_recovery()
        await asyncio.sleep(0.3)
        
        # Verify components can resume operations
        assert not await components['meta_engine'].is_emergency_stopped()
        assert not await components['memory_system'].is_emergency_stopped()
        assert not await components['mcp_orchestrator'].is_emergency_stopped()
    
    @pytest.mark.asyncio
    async def test_cross_component_error_handling_and_recovery(self, integrated_system):
        """Test error handling and recovery across components."""
        context = integrated_system
        components = context.components
        
        # Test 1: Memory system error affecting other components
        memory_error = Exception("Simulated memory corruption")
        
        # Inject error into memory system
        with patch.object(components['memory_system'], 'store_node', side_effect=memory_error):
            # Attempt operation that should trigger error handling
            try:
                node = MemoryNode(id="error_test", content="test", embedding=[0.1] * 128)
                await components['memory_system'].store_node(node)
            except Exception:
                pass  # Expected to fail
        
        # Wait for error propagation and recovery
        await asyncio.sleep(0.2)
        
        # Verify error was handled and recovery initiated
        error_reports = await components['meta_engine'].get_error_reports()
        memory_errors = [e for e in error_reports if e['component'] == 'memory_system']
        assert len(memory_errors) > 0
        
        recovery_actions = await components['meta_engine'].get_recovery_actions()
        memory_recovery = [a for a in recovery_actions if a['target_component'] == 'memory_system']
        assert len(memory_recovery) > 0
        
        # Test 2: MCP orchestrator error cascade
        mcp_error = Exception("Agent communication failure")
        
        with patch.object(components['mcp_orchestrator'], 'submit_task', side_effect=mcp_error):
            try:
                task = AgentTask(id="error_task", type="test", payload={})
                await components['mcp_orchestrator'].submit_task(task)
            except Exception:
                pass
        
        await asyncio.sleep(0.2)
        
        # Verify error isolation prevented system-wide failure
        system_state = await components['meta_engine'].get_system_state()
        assert system_state['status'] in ['degraded', 'operational']  # Not 'failed'
        
        # Verify other components remain operational
        test_node = MemoryNode(id="recovery_test", content="test", embedding=[0.1] * 128)
        storage_result = await components['memory_system'].store_node(test_node)
        assert storage_result.success
        
        # Test 3: Cascading failure prevention
        # Simulate multiple component failures
        errors = [
            ('delta_evaluator', Exception("Evaluation error")),
            ('memory_optimizer', Exception("Optimization error"))
        ]
        
        for component_name, error in errors:
            component = components[component_name]
            # Inject error into a key method
            if hasattr(component, 'process'):
                with patch.object(component, 'process', side_effect=error):
                    try:
                        await component.process({})
                    except Exception:
                        pass
        
        await asyncio.sleep(0.3)
        
        # Verify system implemented graceful degradation
        final_state = await components['meta_engine'].get_system_state()
        assert final_state['status'] != 'failed'
        assert final_state['operational_components'] >= 2  # At least some components working


class TestDeploymentReadiness:
    """Test deployment readiness scenarios and configurations."""
    
    @pytest.mark.asyncio
    async def test_different_deployment_configurations(self, integration_context):
        """Test system works with different deployment configurations."""
        context = integration_context
        
        # Test configurations for different deployment scenarios
        configurations = [
            {
                'name': 'minimal',
                'config': {
                    'delta_evaluation': {'weights': {'performance': 1.0}},
                    'meta_cognitive': {'monitoring_interval': 1.0},
                    'memory': {'max_nodes': 100},
                    'mcp': {'max_agents': 1},
                    'safety': {'max_modification_rate': 0.05}
                }
            },
            {
                'name': 'standard',
                'config': {
                    'delta_evaluation': {'weights': {'performance': 0.4, 'efficiency': 0.3, 'stability': 0.3}},
                    'meta_cognitive': {'monitoring_interval': 0.5},
                    'memory': {'max_nodes': 1000},
                    'mcp': {'max_agents': 3},
                    'safety': {'max_modification_rate': 0.1}
                }
            },
            {
                'name': 'high_performance',
                'config': {
                    'delta_evaluation': {'weights': {'performance': 0.5, 'efficiency': 0.5}},
                    'meta_cognitive': {'monitoring_interval': 0.1},
                    'memory': {'max_nodes': 5000},
                    'mcp': {'max_agents': 10},
                    'safety': {'max_modification_rate': 0.2}
                }
            }
        ]
        
        for config_spec in configurations:
            # Initialize system with specific configuration
            system_components = await self._initialize_system_with_config(config_spec['config'])
            
            # Test basic functionality
            basic_test_result = await self._run_basic_functionality_test(system_components)
            assert basic_test_result['success'], f"Basic test failed for {config_spec['name']} config"
            
            # Test performance characteristics
            performance_result = await self._run_performance_test(system_components, config_spec['name'])
            assert performance_result['meets_expectations'], f"Performance test failed for {config_spec['name']} config"
            
            # Cleanup
            await self._cleanup_system_components(system_components)
    
    async def _initialize_system_with_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize system components with specific configuration."""
        components = {
            'delta_evaluator': DeltaEvaluator(config=config.get('delta_evaluation', {})),
            'meta_engine': MetaCognitiveEngine(config=config.get('meta_cognitive', {})),
            'memory_system': HybridMemorySystem(config=config.get('memory', {})),
            'mcp_orchestrator': MCPOrchestrator(config=config.get('mcp', {})),
            'safety_validator': SafetyValidator(config=config.get('safety', {})),
            'memory_optimizer': MemoryOptimizer()
        }
        
        # Initialize components
        await components['meta_engine'].start()
        await components['memory_system'].initialize()
        await components['mcp_orchestrator'].start()
        
        return components
    
    async def _run_basic_functionality_test(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic functionality test on system components."""
        try:
            # Test delta evaluation
            metrics = DeltaMetrics(performance=0.1, efficiency=0.05, stability=0.02, capability=0.08)
            delta_result = await components['delta_evaluator'].evaluate(metrics)
            assert delta_result.total_delta > 0
            
            # Test memory operations
            node = MemoryNode(id="test", content="test content", embedding=[0.1] * 128)
            storage_result = await components['memory_system'].store_node(node)
            assert storage_result.success
            
            # Test MCP task
            task = AgentTask(id="test_task", type="test", payload={})
            task_result = await components['mcp_orchestrator'].submit_task(task)
            assert task_result.task_id == "test_task"
            
            return {'success': True, 'details': 'All basic tests passed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _run_performance_test(self, components: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Run performance test appropriate for configuration."""
        try:
            start_time = time.time()
            
            # Performance test scaled to configuration
            if config_name == 'minimal':
                operations = 10
            elif config_name == 'standard':
                operations = 50
            else:  # high_performance
                operations = 200
            
            # Run concurrent operations
            tasks = []
            for i in range(operations):
                if i % 3 == 0:
                    # Memory operation
                    node = MemoryNode(id=f"perf_{i}", content=f"content {i}", embedding=[0.1] * 128)
                    tasks.append(components['memory_system'].store_node(node))
                elif i % 3 == 1:
                    # Delta evaluation
                    metrics = DeltaMetrics(performance=0.1, efficiency=0.05, stability=0.02, capability=0.08)
                    tasks.append(components['delta_evaluator'].evaluate(metrics))
                else:
                    # MCP task
                    task = AgentTask(id=f"perf_task_{i}", type="performance_test", payload={})
                    tasks.append(components['mcp_orchestrator'].submit_task(task))
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            ops_per_second = operations / total_time
            
            # Performance expectations based on configuration
            expected_ops_per_second = {
                'minimal': 5,
                'standard': 15,
                'high_performance': 30
            }
            
            meets_expectations = ops_per_second >= expected_ops_per_second[config_name]
            
            return {
                'meets_expectations': meets_expectations,
                'ops_per_second': ops_per_second,
                'expected': expected_ops_per_second[config_name],
                'total_time': total_time
            }
            
        except Exception as e:
            return {'meets_expectations': False, 'error': str(e)}
    
    async def _cleanup_system_components(self, components: Dict[str, Any]):
        """Cleanup system components."""
        try:
            await components['meta_engine'].stop()
            await components['memory_system'].shutdown()
            await components['mcp_orchestrator'].stop()
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")
    
    @pytest.mark.asyncio
    async def test_stress_testing_with_concurrent_operations(self, integrated_system):
        """Test system resilience under stress with concurrent operations."""
        context = integrated_system
        components = context.components
        
        # Define stress test parameters
        stress_duration = 10  # seconds
        concurrent_threads = 20
        operations_per_thread = 50
        
        async def stress_worker(worker_id: int):
            """Worker function for stress testing."""
            operations_completed = 0
            errors_encountered = 0
            
            for i in range(operations_per_thread):
                try:
                    operation_type = i % 4
                    
                    if operation_type == 0:
                        # Memory stress
                        node = MemoryNode(
                            id=f"stress_{worker_id}_{i}",
                            content=f"Stress test content {worker_id}-{i}" * 10,
                            embedding=[0.1 * (i % 10)] * 128
                        )
                        await components['memory_system'].store_node(node)
                        
                    elif operation_type == 1:
                        # Delta evaluation stress
                        metrics = DeltaMetrics(
                            performance=0.1 + (i * 0.001),
                            efficiency=0.05 + (i * 0.0005),
                            stability=0.02,
                            capability=0.08
                        )
                        await components['delta_evaluator'].evaluate(metrics)
                        
                    elif operation_type == 2:
                        # MCP stress
                        task = AgentTask(
                            id=f"stress_task_{worker_id}_{i}",
                            type="stress_test",
                            payload={"worker": worker_id, "iteration": i},
                            priority=1
                        )
                        await components['mcp_orchestrator'].submit_task(task)
                        
                    else:
                        # Meta-cognitive stress
                        event = {
                            'type': 'stress_event',
                            'worker_id': worker_id,
                            'iteration': i,
                            'timestamp': time.time()
                        }
                        await components['meta_engine'].process_system_event(event)
                    
                    operations_completed += 1
                    
                except Exception as e:
                    errors_encountered += 1
                    logging.warning(f"Stress worker {worker_id} error: {e}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            return {
                'worker_id': worker_id,
                'operations_completed': operations_completed,
                'errors_encountered': errors_encountered,
                'error_rate': errors_encountered / operations_per_thread
            }
        
        # Run stress test
        start_time = time.time()
        
        # Create and run concurrent workers
        workers = [stress_worker(i) for i in range(concurrent_threads)]
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        
        end_time = time.time()
        total_stress_time = end_time - start_time
        
        # Analyze results
        successful_workers = [r for r in worker_results if isinstance(r, dict)]
        failed_workers = [r for r in worker_results if isinstance(r, Exception)]
        
        total_operations = sum(r['operations_completed'] for r in successful_workers)
        total_errors = sum(r['errors_encountered'] for r in successful_workers)
        overall_error_rate = total_errors / (total_operations + total_errors) if (total_operations + total_errors) > 0 else 0
        
        # Verify stress test results
        assert len(failed_workers) < concurrent_threads * 0.1  # Less than 10% worker failures
        assert overall_error_rate < 0.2  # Less than 20% operation error rate
        assert total_operations > concurrent_threads * operations_per_thread * 0.7  # At least 70% operations completed
        
        # Verify system remained stable under stress
        final_system_state = await components['meta_engine'].get_system_state()
        assert final_system_state['status'] in ['operational', 'degraded']  # Not failed
        assert final_system_state['stability_score'] > 0.5
        
        # Log stress test results
        context.log_event('stress_test_completed', {
            'duration': total_stress_time,
            'concurrent_threads': concurrent_threads,
            'total_operations': total_operations,
            'total_errors': total_errors,
            'error_rate': overall_error_rate,
            'operations_per_second': total_operations / total_stress_time,
            'final_system_state': final_system_state
        })
    
    @pytest.mark.asyncio
    async def test_error_injection_resilience_validation(self, integrated_system):
        """Test system resilience through error injection testing."""
        context = integrated_system
        components = context.components
        
        # Define error injection scenarios
        error_scenarios = [
            {
                'name': 'memory_corruption',
                'component': 'memory_system',
                'method': 'store_node',
                'error': Exception("Simulated memory corruption"),
                'frequency': 0.1  # 10% of operations fail
            },
            {
                'name': 'delta_calculation_error',
                'component': 'delta_evaluator',
                'method': 'evaluate',
                'error': ValueError("Invalid delta calculation"),
                'frequency': 0.05  # 5% of operations fail
            },
            {
                'name': 'mcp_communication_failure',
                'component': 'mcp_orchestrator',
                'method': 'submit_task',
                'error': ConnectionError("Agent communication failed"),
                'frequency': 0.15  # 15% of operations fail
            },
            {
                'name': 'safety_validation_timeout',
                'component': 'safety_validator',
                'method': 'validate',
                'error': TimeoutError("Safety validation timeout"),
                'frequency': 0.08  # 8% of operations fail
            }
        ]
        
        for scenario in error_scenarios:
            context.log_event('error_injection_start', {'scenario': scenario['name']})
            
            component = components[scenario['component']]
            original_method = getattr(component, scenario['method'])
            
            # Create error injection wrapper
            def create_error_injector(original_func, error, frequency):
                call_count = 0
                
                async def error_injector(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    
                    # Inject error based on frequency
                    if (call_count * frequency) >= 1 and (call_count % int(1/frequency)) == 0:
                        raise error
                    
                    return await original_func(*args, **kwargs)
                
                return error_injector
            
            # Apply error injection
            error_injector = create_error_injector(original_method, scenario['error'], scenario['frequency'])
            setattr(component, scenario['method'], error_injector)
            
            try:
                # Run operations with error injection
                operations_attempted = 0
                operations_successful = 0
                errors_handled = 0
                
                for i in range(50):  # Run 50 operations
                    operations_attempted += 1
                    
                    try:
                        if scenario['component'] == 'memory_system':
                            node = MemoryNode(id=f"error_test_{i}", content=f"test {i}", embedding=[0.1] * 128)
                            await component.store_node(node)
                            
                        elif scenario['component'] == 'delta_evaluator':
                            metrics = DeltaMetrics(performance=0.1, efficiency=0.05, stability=0.02, capability=0.08)
                            await component.evaluate(metrics)
                            
                        elif scenario['component'] == 'mcp_orchestrator':
                            task = AgentTask(id=f"error_task_{i}", type="test", payload={})
                            await component.submit_task(task)
                            
                        elif scenario['component'] == 'safety_validator':
                            await component.validate({'test': 'data'})
                        
                        operations_successful += 1
                        
                    except Exception as e:
                        # Check if error was handled gracefully
                        if str(e) == str(scenario['error']):
                            errors_handled += 1
                        else:
                            # Unexpected error
                            raise
                
                # Verify error handling
                expected_errors = int(operations_attempted * scenario['frequency'])
                assert abs(errors_handled - expected_errors) <= 2  # Allow some variance
                
                # Verify system remained operational
                system_state = await components['meta_engine'].get_system_state()
                assert system_state['status'] in ['operational', 'degraded']
                
                # Verify error recovery
                recovery_actions = await components['meta_engine'].get_recovery_actions()
                scenario_recoveries = [a for a in recovery_actions if scenario['component'] in a.get('target_component', '')]
                assert len(scenario_recoveries) > 0
                
                context.log_event('error_injection_complete', {
                    'scenario': scenario['name'],
                    'operations_attempted': operations_attempted,
                    'operations_successful': operations_successful,
                    'errors_handled': errors_handled,
                    'recovery_actions': len(scenario_recoveries)
                })
                
            finally:
                # Restore original method
                setattr(component, scenario['method'], original_method)
        
        # Verify overall system resilience
        final_state = await components['meta_engine'].get_system_state()
        assert final_state['status'] != 'failed'
        assert final_state['resilience_score'] > 0.6


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])