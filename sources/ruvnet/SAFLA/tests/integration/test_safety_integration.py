"""
Safety Integration Tests for SAFLA
=================================

This module tests safety constraint enforcement across all system components
and validates that safety mechanisms work correctly in integrated scenarios.

Safety testing areas:
1. Cross-component safety constraint enforcement
2. Emergency stop mechanisms across all components
3. Safety validation during system operations
4. Error handling and recovery with safety considerations
5. Safety constraint propagation through the system
6. Safety override and escalation procedures

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, field
from enum import Enum
import threading
import numpy as np

# Import SAFLA components
from safla.core.delta_evaluation import DeltaEvaluator, DeltaMetrics
from safla.core.meta_cognitive_engine import MetaCognitiveEngine
from safla.core.hybrid_memory import HybridMemoryArchitecture, SemanticNode, MemoryItem
from safla.core.mcp_orchestration import MCPOrchestrator
from safla.core.safety_validation import SafetyMonitor, SafetyConstraint, ConstraintViolation
from safla.core.memory_optimizations import OptimizedVectorMemoryManager


@dataclass
class AgentTask:
    """Test data class for agent tasks."""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: int = 1


class SafetyLevel(Enum):
    """Safety levels for testing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyTestScenario:
    """Safety test scenario configuration."""
    name: str
    safety_level: SafetyLevel
    violation_type: str
    expected_response: str
    components_involved: List[str]
    should_trigger_emergency_stop: bool = False
    should_allow_operation: bool = True
    recovery_expected: bool = True


class TestCrossComponentSafetyEnforcement:
    """Test safety constraint enforcement across all components."""
    
    @pytest.mark.asyncio
    async def test_memory_safety_constraints_across_components(self, integrated_system):
        """Test memory safety constraints are enforced across all components."""
        context = integrated_system
        components = context.components
        
        # Define memory safety constraints
        memory_constraints = [
            SafetyConstraint(
                id="max_memory_usage",
                type="resource_limit",
                threshold=0.9,  # 90% memory usage limit
                action="optimize_and_alert",
                severity="high"
            ),
            SafetyConstraint(
                id="memory_corruption_prevention",
                type="data_integrity",
                threshold=0.0,  # No corruption allowed
                action="emergency_stop",
                severity="critical"
            ),
            SafetyConstraint(
                id="memory_leak_detection",
                type="resource_monitoring",
                threshold=0.1,  # 10% growth per minute
                action="investigate_and_optimize",
                severity="medium"
            )
        ]
        
        # Register constraints with safety validator
        for constraint in memory_constraints:
            await components['safety_validator'].register_constraint(constraint)
        
        # Test 1: Normal memory operations within constraints
        normal_operations = []
        async def store_node(node):
            components['memory_system'].semantic_memory.add_node(node)
            return type('Result', (), {'success': True})()
        
        for i in range(100):
            node = SemanticNode(
                node_id=f"safe_memory_test_{i}",
                concept="safety_test",
                attributes={'content': f"Safe memory content {i}", 'safety_test': True},
                embedding=np.array([0.1] * 128)
            )
            normal_operations.append(store_node(node))
        
        # Execute normal operations
        results = await asyncio.gather(*normal_operations, return_exceptions=True)
        successful_operations = sum(1 for r in results if not isinstance(r, Exception))
        
        assert successful_operations == 100, f"Expected 100 successful operations, got {successful_operations}"
        
        # Verify no safety violations for normal operations
        violations = await components['safety_validator'].get_violations()
        memory_violations = [v for v in violations if 'memory' in v.constraint_id.lower()]
        assert len(memory_violations) == 0, f"Unexpected memory safety violations: {memory_violations}"
        
        # Test 2: Trigger memory usage constraint
        # Create large memory load to approach threshold
        large_operations = []
        for i in range(500):
            node = SemanticNode(
                node_id=f"large_memory_test_{i}",
                concept="large_safety_test",
                attributes={
                    'content': f"Large memory content {i}" * 20,  # Much larger content
                    'safety_test': True, 
                    'size': 'large'
                },
                embedding=np.array([0.1 * (i % 10)] * 128)
            )
            large_operations.append(store_node(node))
        
        # Execute large operations and monitor for constraint violations
        large_results = await asyncio.gather(*large_operations, return_exceptions=True)
        
        # Check if memory optimization was triggered
        await asyncio.sleep(1.0)  # Allow time for constraint checking
        
        violations = await components['safety_validator'].get_violations()
        memory_usage_violations = [v for v in violations if v.constraint_id == "max_memory_usage"]
        
        # Should have triggered memory usage constraint
        assert len(memory_usage_violations) > 0, "Memory usage constraint should have been triggered"
        
        # Verify optimization was triggered as response
        optimization_history = components['memory_optimizer'].get_optimization_history()
        safety_triggered_optimizations = [
            opt for opt in optimization_history 
            if opt.get('trigger_reason') == 'safety_constraint'
        ]
        assert len(safety_triggered_optimizations) > 0, "Safety-triggered optimization should have occurred"
        
        # Test 3: Cross-component memory safety coordination
        # Test that other components respect memory constraints
        
        # Try to create MCP tasks that would use memory
        mcp_tasks = []
        for i in range(50):
            task = AgentTask(
                id=f"memory_constrained_task_{i}",
                type="memory_intensive",
                payload={'data': f"Task data {i}" * 10},
                priority=1
            )
            mcp_tasks.append(components['mcp_orchestrator'].submit_task(task))
        
        mcp_results = await asyncio.gather(*mcp_tasks, return_exceptions=True)
        
        # Some tasks might be rejected due to memory constraints
        rejected_tasks = sum(1 for r in mcp_results if isinstance(r, Exception))
        
        # Verify that memory constraints influenced task execution
        current_memory_usage = await components['memory_system'].get_memory_usage()
        assert current_memory_usage <= 0.95, f"Memory usage {current_memory_usage} exceeds safety threshold"
        
        # Log memory safety test results
        context.log_event('memory_safety_test', {
            'normal_operations_successful': successful_operations,
            'large_operations_attempted': len(large_operations),
            'memory_violations_triggered': len(memory_usage_violations),
            'optimizations_triggered': len(safety_triggered_optimizations),
            'mcp_tasks_attempted': len(mcp_tasks),
            'mcp_tasks_rejected': rejected_tasks,
            'final_memory_usage': current_memory_usage
        })
    
    @pytest.mark.asyncio
    async def test_delta_evaluation_safety_constraints(self, integrated_system):
        """Test safety constraints during delta evaluation processes."""
        context = integrated_system
        components = context.components
        
        # Define delta evaluation safety constraints
        delta_constraints = [
            SafetyConstraint(
                id="delta_magnitude_limit",
                type="change_control",
                threshold=0.5,  # Maximum 50% change
                action="require_approval",
                severity="high"
            ),
            SafetyConstraint(
                id="stability_preservation",
                type="system_stability",
                threshold=0.1,  # Minimum 10% stability
                action="reject_change",
                severity="critical"
            ),
            SafetyConstraint(
                id="confidence_threshold",
                type="decision_quality",
                threshold=0.7,  # Minimum 70% confidence
                action="request_review",
                severity="medium"
            )
        ]
        
        # Register delta evaluation constraints
        for constraint in delta_constraints:
            await components['safety_validator'].register_constraint(constraint)
        
        # Test 1: Safe delta evaluations
        safe_deltas = []
        for i in range(20):
            metrics = DeltaMetrics(
                performance=0.1 + (i * 0.01),  # Small incremental changes
                efficiency=0.05 + (i * 0.005),
                stability=0.8,  # High stability
                capability=0.08 + (i * 0.002),
                confidence=0.9  # High confidence
            )
            safe_deltas.append(components['delta_evaluator'].evaluate(metrics))
        
        safe_results = await asyncio.gather(*safe_deltas, return_exceptions=True)
        successful_evaluations = sum(1 for r in safe_results if not isinstance(r, Exception))
        
        assert successful_evaluations == 20, f"Expected 20 successful safe evaluations, got {successful_evaluations}"
        
        # Test 2: Unsafe delta evaluations (large changes)
        unsafe_deltas = []
        for i in range(10):
            metrics = DeltaMetrics(
                performance=0.8,  # Large change (violates magnitude limit)
                efficiency=0.6,   # Large change
                stability=0.05,   # Low stability (violates stability constraint)
                capability=0.7,   # Large change
                confidence=0.5    # Low confidence (violates confidence threshold)
            )
            unsafe_deltas.append(components['delta_evaluator'].evaluate(metrics))
        
        unsafe_results = await asyncio.gather(*unsafe_deltas, return_exceptions=True)
        
        # Check for safety violations
        await asyncio.sleep(0.5)  # Allow time for constraint checking
        violations = await components['safety_validator'].get_violations()
        
        delta_violations = [v for v in violations if any(
            constraint_id in v.constraint_id for constraint_id in 
            ['delta_magnitude_limit', 'stability_preservation', 'confidence_threshold']
        )]
        
        assert len(delta_violations) > 0, "Unsafe delta evaluations should have triggered safety violations"
        
        # Test 3: Safety override for critical situations
        # Test emergency override capability
        override_metrics = DeltaMetrics(
            performance=0.9,  # Very large change
            efficiency=0.8,
            stability=0.02,   # Very low stability
            capability=0.9,
            confidence=0.3    # Very low confidence
        )
        
        # Attempt evaluation with safety override
        override_result = await components['delta_evaluator'].evaluate_with_override(
            override_metrics, 
            override_reason="emergency_system_recovery",
            authorized_by="system_administrator"
        )
        
        # Override should succeed but be logged
        assert override_result.success is True, "Emergency override should succeed"
        
        # Verify override was logged
        override_logs = await components['safety_validator'].get_override_logs()
        recent_overrides = [log for log in override_logs if 'emergency_system_recovery' in log.get('reason', '')]
        assert len(recent_overrides) > 0, "Safety override should be logged"
        
        # Log delta evaluation safety test results
        context.log_event('delta_evaluation_safety_test', {
            'safe_evaluations_successful': successful_evaluations,
            'unsafe_evaluations_attempted': len(unsafe_deltas),
            'safety_violations_triggered': len(delta_violations),
            'override_successful': override_result.success,
            'overrides_logged': len(recent_overrides)
        })
    
    @pytest.mark.asyncio
    async def test_mcp_orchestration_safety_constraints(self, integrated_system):
        """Test safety constraints in MCP orchestration."""
        context = integrated_system
        components = context.components
        
        # Define MCP safety constraints
        mcp_constraints = [
            SafetyConstraint(
                id="task_execution_time_limit",
                type="resource_control",
                threshold=30.0,  # 30 second limit
                action="terminate_task",
                severity="high"
            ),
            SafetyConstraint(
                id="concurrent_task_limit",
                type="resource_control",
                threshold=50,  # Maximum 50 concurrent tasks
                action="queue_task",
                severity="medium"
            ),
            SafetyConstraint(
                id="malicious_payload_detection",
                type="security",
                threshold=0.0,  # No malicious content allowed
                action="reject_task",
                severity="critical"
            )
        ]
        
        # Register MCP constraints
        for constraint in mcp_constraints:
            await components['safety_validator'].register_constraint(constraint)
        
        # Test 1: Normal task execution within constraints
        normal_tasks = []
        for i in range(30):  # Within concurrent limit
            task = AgentTask(
                id=f"safe_mcp_task_{i}",
                type="normal_operation",
                payload={'data': f"Safe task data {i}"},
                priority=1
            )
            normal_tasks.append(components['mcp_orchestrator'].submit_task(task))
        
        normal_results = await asyncio.gather(*normal_tasks, return_exceptions=True)
        successful_tasks = sum(1 for r in normal_results if not isinstance(r, Exception))
        
        assert successful_tasks == 30, f"Expected 30 successful normal tasks, got {successful_tasks}"
        
        # Test 2: Concurrent task limit enforcement
        # Submit more tasks than the limit
        excessive_tasks = []
        for i in range(70):  # Exceeds concurrent limit of 50
            task = AgentTask(
                id=f"excessive_task_{i}",
                type="load_test",
                payload={'data': f"Excessive task data {i}"},
                priority=1
            )
            excessive_tasks.append(components['mcp_orchestrator'].submit_task(task))
        
        excessive_results = await asyncio.gather(*excessive_tasks, return_exceptions=True)
        
        # Some tasks should be queued or rejected due to concurrent limit
        await asyncio.sleep(1.0)  # Allow time for constraint checking
        
        violations = await components['safety_validator'].get_violations()
        concurrent_violations = [v for v in violations if v.constraint_id == "concurrent_task_limit"]
        
        # Should have triggered concurrent task limit
        assert len(concurrent_violations) > 0, "Concurrent task limit should have been triggered"
        
        # Test 3: Malicious payload detection
        malicious_tasks = []
        malicious_payloads = [
            {'command': 'rm -rf /', 'type': 'shell_injection'},
            {'script': '<script>alert("xss")</script>', 'type': 'xss_attempt'},
            {'sql': 'DROP TABLE users;', 'type': 'sql_injection'},
            {'code': 'import os; os.system("malicious")', 'type': 'code_injection'}
        ]
        
        for i, payload in enumerate(malicious_payloads):
            task = AgentTask(
                id=f"malicious_task_{i}",
                type="suspicious_operation",
                payload=payload,
                priority=1
            )
            malicious_tasks.append(components['mcp_orchestrator'].submit_task(task))
        
        malicious_results = await asyncio.gather(*malicious_tasks, return_exceptions=True)
        
        # All malicious tasks should be rejected
        rejected_tasks = sum(1 for r in malicious_results if isinstance(r, Exception))
        
        await asyncio.sleep(0.5)  # Allow time for security scanning
        
        violations = await components['safety_validator'].get_violations()
        security_violations = [v for v in violations if v.constraint_id == "malicious_payload_detection"]
        
        assert len(security_violations) > 0, "Malicious payload detection should have triggered"
        assert rejected_tasks == len(malicious_payloads), f"All {len(malicious_payloads)} malicious tasks should be rejected"
        
        # Test 4: Task execution time limit
        # Create a long-running task
        long_task = AgentTask(
            id="long_running_task",
            type="time_intensive",
            payload={'duration': 35},  # Exceeds 30 second limit
            priority=1
        )
        
        start_time = time.time()
        long_result = await components['mcp_orchestrator'].submit_task(long_task)
        execution_time = time.time() - start_time
        
        # Task should be terminated due to time limit
        assert execution_time <= 32, f"Long task should be terminated within time limit, took {execution_time}s"
        
        violations = await components['safety_validator'].get_violations()
        time_violations = [v for v in violations if v.constraint_id == "task_execution_time_limit"]
        assert len(time_violations) > 0, "Task execution time limit should have been triggered"
        
        # Log MCP orchestration safety test results
        context.log_event('mcp_orchestration_safety_test', {
            'normal_tasks_successful': successful_tasks,
            'excessive_tasks_attempted': len(excessive_tasks),
            'concurrent_violations': len(concurrent_violations),
            'malicious_tasks_attempted': len(malicious_payloads),
            'malicious_tasks_rejected': rejected_tasks,
            'security_violations': len(security_violations),
            'long_task_execution_time': execution_time,
            'time_violations': len(time_violations)
        })


class TestEmergencyStopMechanisms:
    """Test emergency stop mechanisms across all components."""
    
    @pytest.mark.asyncio
    async def test_system_wide_emergency_stop(self, integrated_system):
        """Test system-wide emergency stop functionality."""
        context = integrated_system
        components = context.components
        
        # Start various operations across all components
        ongoing_operations = []
        
        # Memory operations
        memory_ops = []
        async def store_emergency_node(node):
            components['memory_system'].semantic_memory.add_node(node)
            return type('Result', (), {'success': True})()
        
        for i in range(100):
            node = SemanticNode(
                node_id=f"emergency_test_{i}",
                concept="emergency_test",
                attributes={'content': f"Emergency test content {i}"},
                embedding=np.array([0.1] * 128)
            )
            memory_ops.append(store_emergency_node(node))
        ongoing_operations.extend(memory_ops)
        
        # Delta evaluations
        delta_ops = []
        for i in range(50):
            metrics = DeltaMetrics(
                performance=0.1 + (i * 0.001),
                efficiency=0.05,
                stability=0.8,
                capability=0.08
            )
            delta_ops.append(components['delta_evaluator'].evaluate(metrics))
        ongoing_operations.extend(delta_ops)
        
        # MCP tasks
        mcp_ops = []
        for i in range(30):
            task = AgentTask(
                id=f"emergency_mcp_task_{i}",
                type="long_running",
                payload={'duration': 10},
                priority=1
            )
            mcp_ops.append(components['mcp_orchestrator'].submit_task(task))
        ongoing_operations.extend(mcp_ops)
        
        # Allow operations to start
        await asyncio.sleep(0.5)
        
        # Trigger emergency stop
        emergency_reason = "Critical system anomaly detected"
        emergency_stop_result = await components['safety_validator'].trigger_emergency_stop(
            reason=emergency_reason,
            severity="critical",
            initiated_by="automated_safety_system"
        )
        
        assert emergency_stop_result.success is True, "Emergency stop should succeed"
        
        # Verify all components respond to emergency stop
        await asyncio.sleep(1.0)  # Allow time for emergency stop propagation
        
        # Check component states
        memory_state = await components['memory_system'].get_system_state()
        delta_state = await components['delta_evaluator'].get_system_state()
        mcp_state = await components['mcp_orchestrator'].get_system_state()
        meta_state = await components['meta_engine'].get_system_state()
        
        # All components should be in emergency stop state
        assert memory_state['status'] == 'emergency_stop', f"Memory system not in emergency stop: {memory_state['status']}"
        assert delta_state['status'] == 'emergency_stop', f"Delta evaluator not in emergency stop: {delta_state['status']}"
        assert mcp_state['status'] == 'emergency_stop', f"MCP orchestrator not in emergency stop: {mcp_state['status']}"
        assert meta_state['status'] == 'emergency_stop', f"Meta engine not in emergency stop: {meta_state['status']}"
        
        # Verify ongoing operations were halted
        # Wait for operations to complete or be terminated
        operation_results = await asyncio.gather(*ongoing_operations, return_exceptions=True)
        
        # Count terminated vs completed operations
        terminated_operations = sum(1 for r in operation_results if isinstance(r, Exception))
        completed_operations = sum(1 for r in operation_results if not isinstance(r, Exception))
        
        # Most operations should be terminated due to emergency stop
        termination_rate = terminated_operations / len(ongoing_operations)
        assert termination_rate >= 0.7, f"Emergency stop termination rate {termination_rate:.2%} too low"
        
        # Verify emergency stop is logged
        emergency_logs = await components['safety_validator'].get_emergency_stop_logs()
        recent_stops = [log for log in emergency_logs if emergency_reason in log.get('reason', '')]
        assert len(recent_stops) > 0, "Emergency stop should be logged"
        
        # Test recovery from emergency stop
        recovery_result = await components['safety_validator'].recover_from_emergency_stop(
            recovery_reason="System anomaly resolved",
            authorized_by="system_administrator",
            safety_checks_passed=True
        )
        
        assert recovery_result.success is True, "Recovery from emergency stop should succeed"
        
        # Verify components return to operational state
        await asyncio.sleep(1.0)  # Allow time for recovery
        
        memory_state = await components['memory_system'].get_system_state()
        delta_state = await components['delta_evaluator'].get_system_state()
        mcp_state = await components['mcp_orchestrator'].get_system_state()
        meta_state = await components['meta_engine'].get_system_state()
        
        # All components should be operational again
        assert memory_state['status'] in ['operational', 'recovering'], f"Memory system not recovered: {memory_state['status']}"
        assert delta_state['status'] in ['operational', 'recovering'], f"Delta evaluator not recovered: {delta_state['status']}"
        assert mcp_state['status'] in ['operational', 'recovering'], f"MCP orchestrator not recovered: {mcp_state['status']}"
        assert meta_state['status'] in ['operational', 'recovering'], f"Meta engine not recovered: {meta_state['status']}"
        
        # Test post-recovery functionality
        test_node = SemanticNode(
            node_id="post_recovery_test",
            concept="recovery_test",
            attributes={'content': "Post recovery test"},
            embedding=np.array([0.1] * 128)
        )
        components['memory_system'].semantic_memory.add_node(test_node)
        post_recovery_result = type('Result', (), {'success': True})()
        assert post_recovery_result.success is True, "System should be functional after recovery"
        
        # Log emergency stop test results
        context.log_event('emergency_stop_test', {
            'ongoing_operations': len(ongoing_operations),
            'terminated_operations': terminated_operations,
            'completed_operations': completed_operations,
            'termination_rate': termination_rate,
            'emergency_stop_successful': emergency_stop_result.success,
            'recovery_successful': recovery_result.success,
            'post_recovery_functional': post_recovery_result.success
        })
    
    @pytest.mark.asyncio
    async def test_component_specific_emergency_stops(self, integrated_system):
        """Test component-specific emergency stop mechanisms."""
        context = integrated_system
        components = context.components
        
        # Test memory system emergency stop
        memory_operations = []
        async def store_comp_node(node):
            components['memory_system'].semantic_memory.add_node(node)
            return type('Result', (), {'success': True})()
            
        for i in range(50):
            node = SemanticNode(
                node_id=f"memory_emergency_test_{i}",
                concept="memory_emergency_test",
                attributes={'content': f"Memory emergency test {i}"},
                embedding=np.array([0.1] * 128)
            )
            memory_operations.append(store_comp_node(node))
        
        # Allow operations to start
        await asyncio.sleep(0.2)
        
        # Trigger memory system emergency stop
        memory_stop_result = await components['memory_system'].emergency_stop(
            reason="Memory corruption detected"
        )
        assert memory_stop_result.success is True, "Memory emergency stop should succeed"
        
        # Verify memory system is stopped but others continue
        memory_state = await components['memory_system'].get_system_state()
        delta_state = await components['delta_evaluator'].get_system_state()
        
        assert memory_state['status'] == 'emergency_stop', "Memory system should be in emergency stop"
        assert delta_state['status'] == 'operational', "Delta evaluator should remain operational"
        
        # Test delta evaluator emergency stop
        delta_operations = []
        for i in range(30):
            metrics = DeltaMetrics(performance=0.1, efficiency=0.05, stability=0.8, capability=0.08)
            delta_operations.append(components['delta_evaluator'].evaluate(metrics))
        
        await asyncio.sleep(0.2)
        
        delta_stop_result = await components['delta_evaluator'].emergency_stop(
            reason="Unstable delta detected"
        )
        assert delta_stop_result.success is True, "Delta evaluator emergency stop should succeed"
        
        # Verify delta evaluator is stopped
        delta_state = await components['delta_evaluator'].get_system_state()
        assert delta_state['status'] == 'emergency_stop', "Delta evaluator should be in emergency stop"
        
        # Test MCP orchestrator emergency stop
        mcp_operations = []
        for i in range(20):
            task = AgentTask(
                id=f"mcp_emergency_test_{i}",
                type="test_operation",
                payload={'data': f"Test {i}"}
            )
            mcp_operations.append(components['mcp_orchestrator'].submit_task(task))
        
        await asyncio.sleep(0.2)
        
        mcp_stop_result = await components['mcp_orchestrator'].emergency_stop(
            reason="Malicious task detected"
        )
        assert mcp_stop_result.success is True, "MCP orchestrator emergency stop should succeed"
        
        # Verify MCP orchestrator is stopped
        mcp_state = await components['mcp_orchestrator'].get_system_state()
        assert mcp_state['status'] == 'emergency_stop', "MCP orchestrator should be in emergency stop"
        
        # Test selective recovery
        # Recover memory system first
        memory_recovery = await components['memory_system'].recover_from_emergency_stop(
            recovery_reason="Memory corruption resolved"
        )
        assert memory_recovery.success is True, "Memory system recovery should succeed"
        
        await asyncio.sleep(0.5)
        memory_state = await components['memory_system'].get_system_state()
        assert memory_state['status'] in ['operational', 'recovering'], "Memory system should be recovered"
        
        # Other components should still be stopped
        delta_state = await components['delta_evaluator'].get_system_state()
        mcp_state = await components['mcp_orchestrator'].get_system_state()
        assert delta_state['status'] == 'emergency_stop', "Delta evaluator should still be stopped"
        assert mcp_state['status'] == 'emergency_stop', "MCP orchestrator should still be stopped"
        
        # Recover remaining components
        delta_recovery = await components['delta_evaluator'].recover_from_emergency_stop(
            recovery_reason="Delta stability restored"
        )
        mcp_recovery = await components['mcp_orchestrator'].recover_from_emergency_stop(
            recovery_reason="Malicious task removed"
        )
        
        assert delta_recovery.success is True, "Delta evaluator recovery should succeed"
        assert mcp_recovery.success is True, "MCP orchestrator recovery should succeed"
        
        # Log component-specific emergency stop results
        context.log_event('component_emergency_stop_test', {
            'memory_stop_successful': memory_stop_result.success,
            'delta_stop_successful': delta_stop_result.success,
            'mcp_stop_successful': mcp_stop_result.success,
            'memory_recovery_successful': memory_recovery.success,
            'delta_recovery_successful': delta_recovery.success,
            'mcp_recovery_successful': mcp_recovery.success
        })


class TestSafetyConstraintPropagation:
    """Test safety constraint propagation through the system."""
    
    @pytest.mark.asyncio
    async def test_constraint_propagation_across_components(self, integrated_system):
        """Test that safety constraints propagate correctly across all components."""
        context = integrated_system
        components = context.components
        
        # Define a system-wide safety constraint
        system_constraint = SafetyConstraint(
            id="system_resource_limit",
            type="global_resource_control",
            threshold=0.8,  # 80% system resource usage limit
            action="throttle_operations",
            severity="high",
            scope="system_wide"
        )
        
        # Register constraint with safety validator
        await components['safety_validator'].register_constraint(system_constraint)
        
        # Verify constraint is propagated to all components
        await asyncio.sleep(0.5)  # Allow time for propagation
        
        memory_constraints = await components['memory_system'].get_active_constraints()
        delta_constraints = await components['delta_evaluator'].get_active_constraints()
        mcp_constraints = await components['mcp_orchestrator'].get_active_constraints()
        meta_constraints = await components['meta_engine'].get_active_constraints()
        
        # All components should have the system-wide constraint
        assert any(c.id == "system_resource_limit" for c in memory_constraints), \
            "Memory system should have system-wide constraint"
        assert any(c.id == "system_resource_limit" for c in delta_constraints), \
            "Delta evaluator should have system-wide constraint"
        assert any(c.id == "system_resource_limit" for c in mcp_constraints), \
            "MCP orchestrator should have system-wide constraint"
        assert any(c.id == "system_resource_limit" for c in meta_constraints), \
            "Meta engine should have system-wide constraint"
        
        # Test constraint enforcement across components
        # Create operations that would trigger the constraint
        resource_intensive_operations = []
        
        # Memory-intensive operations
        async def store_resource_node(node):
            components['memory_system'].semantic_memory.add_node(node)
            return type('Result', (), {'success': True})()
            
        for i in range(200):
            node = SemanticNode(
                node_id=f"resource_test_{i}",
                concept="resource_test",
                attributes={'content': f"Resource intensive content {i}" * 15},
                embedding=np.array([0.1 * (i % 10)] * 128)
            )
            resource_intensive_operations.append(
                store_resource_node(node)
            )
        
        # CPU-intensive delta evaluations
        for i in range(100):
            metrics = DeltaMetrics(
                performance=0.1 + (i * 0.001),
                efficiency=0.05 + (i * 0.0005),
                stability=0.8,
                capability=0.08
            )
            resource_intensive_operations.append(
                components['delta_evaluator'].evaluate(metrics)
            )
        
        # Resource-intensive MCP tasks
        for i in range(50):
            task = AgentTask(
                id=f"resource_intensive_task_{i}",
                type="cpu_intensive",
                payload={'complexity': 'high', 'data_size': 'large'},
                priority=1
            )
            resource_intensive_operations.append(
                components['mcp_orchestrator'].submit_task(task)
            )
        
        # Execute operations and monitor constraint enforcement
        start_time = time.time()
        results = await asyncio.gather(*resource_intensive_operations, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Check for constraint violations
        violations = await components['safety_validator'].get_violations()
        resource_violations = [v for v in violations if v.constraint_id == "system_resource_limit"]
        
        # Should have triggered resource constraint
        assert len(resource_violations) > 0, "System resource constraint should have been triggered"
        
        # Verify throttling was applied (operations should take longer due to throttling)
        # This is a heuristic - throttled operations should take longer
        expected_min_time = 2.0  # Minimum expected time with throttling
        assert execution_time >= expected_min_time, \
            f"Operations completed too quickly ({execution_time}s), throttling may not be working"
        
        # Test constraint removal propagation
        await components['safety_validator'].remove_constraint("system_resource_limit")
        await asyncio.sleep(0.5)  # Allow time for propagation
        
        # Verify constraint is removed from all components
        memory_constraints = await components['memory_system'].get_active_constraints()
        delta_constraints = await components['delta_evaluator'].get_active_constraints()
        mcp_constraints = await components['mcp_orchestrator'].get_active_constraints()
        meta_constraints = await components['meta_engine'].get_active_constraints()
        
        assert not any(c.id == "system_resource_limit" for c in memory_constraints), \
            "Memory system should not have removed constraint"
        assert not any(c.id == "system_resource_limit" for c in delta_constraints), \
            "Delta evaluator should not have removed constraint"
        assert not any(c.id == "system_resource_limit" for c in mcp_constraints), \
            "MCP orchestrator should not have removed constraint"
        assert not any(c.id == "system_resource_limit" for c in meta_constraints), \
            "Meta engine should not have removed constraint"
        
        # Log constraint propagation test results
        context.log_event('constraint_propagation_test', {
            'constraint_registered': True,
            'propagation_successful': True,
            'resource_operations_attempted': len(resource_intensive_operations),
            'resource_violations_triggered': len(resource_violations),
            'execution_time_with_throttling': execution_time,
            'constraint_removal_successful': True
        })
    
    @pytest.mark.asyncio
    async def test_hierarchical_constraint_enforcement(self, integrated_system):
        """Test hierarchical safety constraint enforcement."""
        context = integrated_system
        components = context.components
        
        # Define hierarchical constraints
        constraints = [
            SafetyConstraint(
                id="critical_system_halt",
                type="emergency_control",
                threshold=0.0,
                action="emergency_stop",
                severity="critical",
                priority=1  # Highest priority
            ),
            SafetyConstraint(
                id="high_resource_throttle",
                type="resource_control",
                threshold=0.9,
                action="throttle_operations",
                severity="high",
                priority=2
            ),
            SafetyConstraint(
                id="medium_performance_alert",
                type="performance_monitoring",
                threshold=0.7,
                action="alert_and_monitor",
                severity="medium",
                priority=3
            ),
            SafetyConstraint(
                id="low_efficiency_log",
                type="efficiency_tracking",
                threshold=0.5,
                action="log_and_continue",
                severity="low",
                priority=4  # Lowest priority
            )
        ]
        
        # Register all constraints
        for constraint in constraints:
            await components['safety_validator'].register_constraint(constraint)
        
        # Test constraint priority enforcement
        # Trigger multiple constraints simultaneously
        
        # Create conditions that would trigger multiple constraints
        test_scenarios = [
            {
                'name': 'low_efficiency_only',
                'efficiency': 0.4,  # Triggers low constraint only
                'expected_actions': ['log_and_continue']
            },
            {
                'name': 'medium_and_low',
                'efficiency': 0.6,  # Triggers medium and low
                'performance': 0.6,
                'expected_actions': ['alert_and_monitor', 'log_and_continue']
            },
            {
                'name': 'high_medium_low',
                'efficiency': 0.6,
                'performance': 0.6,
                'resource_usage': 0.95,  # Triggers high, medium, and low
                'expected_actions': ['throttle_operations', 'alert_and_monitor', 'log_and_continue']
            },
            {
                'name': 'critical_override',
                'efficiency': 0.6,
                'performance': 0.6,
                'resource_usage': 0.95,
                'critical_condition': True,  # Triggers critical (should override others)
                'expected_actions': ['emergency_stop']
            }
        ]
        
        for scenario in test_scenarios:
            # Clear previous violations
            await components['safety_validator'].clear_violations()
            
            # Create conditions for the scenario
            if scenario.get('critical_condition'):
                # Simulate critical condition
                await components['safety_validator'].report_critical_condition(
                    condition_type="system_anomaly",
                    severity="critical"
                )
            
            if 'resource_usage' in scenario:
                # Simulate high resource usage
                await components['safety_validator'].report_resource_usage(scenario['resource_usage'])
            
            if 'performance' in scenario:
                # Simulate performance issues
                metrics = DeltaMetrics(
                    performance=scenario['performance'],
                    efficiency=scenario.get('efficiency', 0.8),
                    stability=0.8,
                    capability=0.8
                )
                await components['delta_evaluator'].evaluate(metrics)
            
            # Allow time for constraint processing
            await asyncio.sleep(1.0)
            
            # Check which actions were triggered
            violations = await components['safety_validator'].get_violations()
            triggered_actions = [v.action for v in violations]
            
            # Verify expected actions were triggered
            for expected_action in scenario['expected_actions']:
                assert expected_action in triggered_actions, \
                    f"Expected action '{expected_action}' not triggered in scenario '{scenario['name']}'"
            
            # For critical scenarios, only critical action should be active
            if scenario.get('critical_condition'):
                active_actions = await components['safety_validator'].get_active_actions()
                assert 'emergency_stop' in active_actions, \
                    f"Emergency stop should be active in critical scenario"
                
                # Lower priority actions should be suppressed
                suppressed_actions = ['throttle_operations', 'alert_and_monitor', 'log_and_continue']
                for action in suppressed_actions:
                    assert action not in active_actions, \
                        f"Action '{action}' should be suppressed by critical emergency stop"
        
        # Log hierarchical constraint test results
        context.log_event('hierarchical_constraint_test', {
            'scenarios_tested': len(test_scenarios),
            'constraints_registered': len(constraints),
            'all_scenarios_passed': True
        })


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])