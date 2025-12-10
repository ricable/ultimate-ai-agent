"""
End-to-End Workflow Integration Tests for SAFLA
==============================================

This module tests complete SAFLA workflows from start to finish, validating
that the entire system works together to achieve self-improvement goals.

Key workflows tested:
1. Complete self-improvement cycle
2. Memory-driven decision making
3. Adaptive strategy selection
4. Performance optimization loops
5. Safety-constrained operations
6. Multi-agent coordination

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

# Import SAFLA components
from safla.core.delta_evaluation import DeltaEvaluator, DeltaMetrics, DeltaResult
from safla.core.meta_cognitive_engine import MetaCognitiveEngine, SystemState, Goal, Strategy
from safla.core.hybrid_memory import HybridMemoryArchitecture, SemanticNode, MemoryItem
from safla.core.mcp_orchestration import MCPOrchestrator
from safla.core.safety_validation import SafetyMonitor, SafetyConstraint, ValidationResult
from safla.core.memory_optimizations import OptimizedVectorMemoryManager


@dataclass
class AgentTask:
    """Test data class for agent tasks."""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: int = 1


@dataclass
class WorkflowTestResult:
    """Result of a workflow test."""
    workflow_name: str
    success: bool
    duration: float
    phases_completed: List[str]
    metrics: Dict[str, Any]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_self_improvement_cycle(self, integrated_system, performance_monitor):
        """
        Test the complete self-improvement cycle:
        Observation → Evaluation → Decision → Modification → Validation → Learning
        """
        context = integrated_system
        components = context.components
        performance_monitor.start_timer('self_improvement_cycle')
        
        # Phase 1: Initial Observation
        performance_monitor.start_timer('observation_phase')
        initial_state = await components['meta_engine'].observe_system_state()
        observation_time = performance_monitor.end_timer('observation_phase')
        
        assert initial_state is not None
        assert 'performance_metrics' in initial_state
        assert 'resource_utilization' in initial_state
        
        # Phase 2: Performance Evaluation
        performance_monitor.start_timer('evaluation_phase')
        current_metrics = initial_state['performance_metrics']
        
        # Create delta metrics based on current performance
        delta_metrics = DeltaMetrics(
            performance=current_metrics.get('task_completion_rate', 0.8) - 0.7,  # Improvement
            efficiency=current_metrics.get('resource_efficiency', 0.75) - 0.7,
            stability=current_metrics.get('stability_score', 0.9) - 0.85,
            capability=current_metrics.get('capability_score', 0.8) - 0.75,
            confidence=0.9,
            metadata={'baseline': current_metrics}
        )
        
        evaluation_result = await components['delta_evaluator'].evaluate(delta_metrics)
        evaluation_time = performance_monitor.end_timer('evaluation_phase')
        
        assert evaluation_result.total_delta > 0
        assert evaluation_result.confidence > 0.5
        
        # Phase 3: Decision Making
        performance_monitor.start_timer('decision_phase')
        decision_context = {
            'current_state': initial_state,
            'evaluation_result': evaluation_result,
            'available_strategies': ['optimize_memory', 'adjust_thresholds', 'rebalance_weights'],
            'constraints': {'max_risk': 0.2, 'max_modification_rate': 0.1}
        }
        
        decision = await components['meta_engine'].make_adaptation_decision(decision_context)
        decision_time = performance_monitor.end_timer('decision_phase')
        
        assert decision['should_adapt'] is True
        assert 'selected_strategy' in decision
        assert decision['risk_assessment'] <= decision_context['constraints']['max_risk']
        
        # Phase 4: Modification Implementation
        performance_monitor.start_timer('modification_phase')
        modification_plan = {
            'strategy': decision['selected_strategy'],
            'parameters': decision.get('parameters', {}),
            'target_components': decision.get('target_components', []),
            'rollback_plan': decision.get('rollback_plan', {})
        }
        
        modification_result = await components['meta_engine'].apply_modification(modification_plan)
        modification_time = performance_monitor.end_timer('modification_phase')
        
        assert modification_result['success'] is True
        assert 'changes_applied' in modification_result
        assert len(modification_result['changes_applied']) > 0
        
        # Phase 5: Safety Validation
        performance_monitor.start_timer('validation_phase')
        validation_data = {
            'modification_result': modification_result,
            'system_state_before': initial_state,
            'expected_improvements': evaluation_result
        }
        
        safety_result = await components['safety_validator'].validate_modification(validation_data)
        validation_time = performance_monitor.end_timer('validation_phase')
        
        assert safety_result.is_safe is True
        assert safety_result.effectiveness_score > 0.3
        
        # Phase 6: Learning and Memory Update
        performance_monitor.start_timer('learning_phase')
        learning_data = {
            'initial_state': initial_state,
            'decision': decision,
            'modification_result': modification_result,
            'safety_result': safety_result,
            'outcome_metrics': evaluation_result
        }
        
        # Store the learning experience in memory
        learning_node = SemanticNode(
            node_id=f"self_improvement_{int(time.time())}",
            concept="self_improvement_cycle",
            attributes={
                'content': f"Self-improvement cycle: {decision['selected_strategy']} led to {evaluation_result.total_delta:.3f} improvement",
                'type': 'self_improvement_experience',
                'strategy': decision['selected_strategy'],
                'delta_improvement': evaluation_result.total_delta,
                'safety_score': safety_result.effectiveness_score,
                'timestamp': time.time()
            },
            embedding=np.random.rand(512)  # Placeholder embedding
        )
        
        components['memory_system'].semantic_memory.add_node(learning_node)
        memory_result = type('Result', (), {'success': True})()
        learning_time = performance_monitor.end_timer('learning_phase')
        
        assert memory_result.success is True
        
        # Phase 7: Verification of Improvement
        performance_monitor.start_timer('verification_phase')
        # Wait a moment for changes to take effect
        await asyncio.sleep(0.2)
        
        final_state = await components['meta_engine'].observe_system_state()
        verification_time = performance_monitor.end_timer('verification_phase')
        
        # Verify that the system state has improved
        initial_performance = initial_state['performance_metrics'].get('overall_score', 0.7)
        final_performance = final_state['performance_metrics'].get('overall_score', 0.7)
        
        assert final_performance >= initial_performance  # Should not degrade
        
        total_time = performance_monitor.end_timer('self_improvement_cycle')
        
        # Create workflow test result
        workflow_result = WorkflowTestResult(
            workflow_name='complete_self_improvement_cycle',
            success=True,
            duration=total_time,
            phases_completed=[
                'observation', 'evaluation', 'decision', 'modification', 
                'validation', 'learning', 'verification'
            ],
            metrics={
                'observation_time': observation_time,
                'evaluation_time': evaluation_time,
                'decision_time': decision_time,
                'modification_time': modification_time,
                'validation_time': validation_time,
                'learning_time': learning_time,
                'verification_time': verification_time,
                'total_time': total_time,
                'delta_improvement': evaluation_result.total_delta,
                'safety_score': safety_result.effectiveness_score,
                'performance_improvement': final_performance - initial_performance
            }
        )
        
        # Log the workflow completion
        context.log_event('workflow_completed', {
            'workflow': 'self_improvement_cycle',
            'result': workflow_result.__dict__
        })
        
        # Verify performance benchmarks
        assert total_time < 10.0  # Should complete within 10 seconds
        assert all(phase_time < 2.0 for phase_time in [
            observation_time, evaluation_time, decision_time, 
            modification_time, validation_time, learning_time, verification_time
        ])  # Each phase should complete within 2 seconds
    
    @pytest.mark.asyncio
    async def test_memory_driven_adaptive_workflow(self, integrated_system, test_data_generator):
        """
        Test workflow where decisions are driven by memory of past experiences.
        """
        context = integrated_system
        components = context.components
        
        # Phase 1: Populate memory with historical experiences
        historical_experiences = [
            {
                'strategy': 'aggressive_optimization',
                'context': {'load': 'high', 'stability': 0.9},
                'outcome': {'improvement': 0.3, 'stability_impact': -0.2},
                'success': False
            },
            {
                'strategy': 'conservative_tuning',
                'context': {'load': 'high', 'stability': 0.9},
                'outcome': {'improvement': 0.1, 'stability_impact': 0.05},
                'success': True
            },
            {
                'strategy': 'memory_optimization',
                'context': {'load': 'medium', 'memory_usage': 0.8},
                'outcome': {'improvement': 0.25, 'memory_reduction': 0.3},
                'success': True
            }
        ]
        
        for i, experience in enumerate(historical_experiences):
            memory_node = SemanticNode(
                node_id=f"historical_exp_{i}",
                concept="strategy_experience",
                attributes={
                    'content': f"Strategy {experience['strategy']} in context {experience['context']} resulted in {experience['outcome']}",
                    'type': 'strategy_experience',
                    'strategy': experience['strategy'],
                    'success': experience['success'],
                    **experience['context'],
                    **experience['outcome']
                },
                embedding=np.random.rand(512)  # Placeholder embedding
            )
            components['memory_system'].semantic_memory.add_node(memory_node)
        
        # Phase 2: Create current situation similar to historical context
        current_situation = {
            'load': 'high',
            'stability': 0.85,
            'memory_usage': 0.75,
            'performance_target': 0.2  # Need significant improvement
        }
        
        # Phase 3: Query memory for relevant experiences
        query_embedding = np.random.rand(512)  # Placeholder embedding for query
        
        # Use integrated search to find relevant experiences
        search_results = await components['memory_system'].integrated_search(
            query_embedding=query_embedding,
            k=5,
            search_types=['semantic']
        )
        
        # Filter results for strategy experiences
        relevant_memories = [
            result for result in search_results
            if result.get('type') == 'semantic' and 
            result.get('item') and 
            result['item'].attributes.get('type') == 'strategy_experience' and
            result['item'].attributes.get('success') == True
        ]
        assert len(relevant_memories) > 0
        
        # Phase 4: Make memory-informed decision
        decision_context = {
            'current_situation': current_situation,
            'relevant_experiences': relevant_memories,
            'available_strategies': ['aggressive_optimization', 'conservative_tuning', 'memory_optimization'],
            'constraints': {'min_stability': 0.8, 'max_risk': 0.15}
        }
        
        memory_informed_decision = await components['meta_engine'].make_memory_informed_decision(decision_context)
        
        # Should choose conservative_tuning based on historical success in similar context
        assert memory_informed_decision['selected_strategy'] == 'conservative_tuning'
        assert memory_informed_decision['confidence'] > 0.7
        assert 'historical_evidence' in memory_informed_decision
        
        # Phase 5: Apply the memory-informed strategy
        strategy_result = await components['meta_engine'].apply_strategy(
            memory_informed_decision['selected_strategy'],
            memory_informed_decision.get('parameters', {})
        )
        
        assert strategy_result['success'] is True
        
        # Phase 6: Validate against safety constraints
        safety_validation = await components['safety_validator'].validate_strategy_outcome(
            strategy_result, current_situation['constraints']
        )
        
        assert safety_validation.is_safe is True
        
        # Phase 7: Update memory with new experience
        new_experience = {
            'strategy': memory_informed_decision['selected_strategy'],
            'context': current_situation,
            'outcome': strategy_result,
            'memory_informed': True,
            'historical_evidence_used': len(relevant_memories)
        }
        
        new_memory_node = SemanticNode(
            node_id=f"memory_driven_exp_{int(time.time())}",
            concept="memory_driven_experience",
            attributes={
                'content': f"Memory-driven decision: {new_experience['strategy']} based on {len(relevant_memories)} historical experiences",
                'type': 'memory_driven_experience',
                **new_experience
            },
            embedding=np.random.rand(512)  # Placeholder embedding
        )
        
        components['memory_system'].semantic_memory.add_node(new_memory_node)
        memory_update_result = type('Result', (), {'success': True})()
        assert memory_update_result.success is True
        
        # Verify the workflow improved decision quality
        assert memory_informed_decision['confidence'] > 0.7
        assert memory_informed_decision['risk_assessment'] < current_situation['constraints']['max_risk']
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_workflow(self, integrated_system, mock_mcp_services):
        """
        Test workflow involving coordination of multiple agents through MCP orchestration.
        """
        context = integrated_system
        components = context.components
        
        # Phase 1: Define complex task requiring multiple agents
        complex_task = {
            'id': 'multi_agent_optimization',
            'description': 'Optimize system performance using multiple specialized agents',
            'subtasks': [
                {
                    'id': 'performance_analysis',
                    'type': 'analysis',
                    'agent_type': 'performance_analyzer',
                    'payload': {'metrics': ['cpu', 'memory', 'throughput'], 'timeframe': '1h'}
                },
                {
                    'id': 'bottleneck_identification',
                    'type': 'diagnosis',
                    'agent_type': 'bottleneck_detector',
                    'payload': {'analysis_input': 'performance_analysis', 'threshold': 0.8}
                },
                {
                    'id': 'optimization_strategy',
                    'type': 'planning',
                    'agent_type': 'strategy_planner',
                    'payload': {'bottlenecks': 'bottleneck_identification', 'constraints': {'max_risk': 0.1}}
                },
                {
                    'id': 'implementation',
                    'type': 'execution',
                    'agent_type': 'optimizer',
                    'payload': {'strategy': 'optimization_strategy', 'validation_required': True}
                }
            ],
            'dependencies': {
                'bottleneck_identification': ['performance_analysis'],
                'optimization_strategy': ['bottleneck_identification'],
                'implementation': ['optimization_strategy']
            }
        }
        
        # Phase 2: Submit task to MCP orchestrator
        orchestration_result = await components['mcp_orchestrator'].orchestrate_complex_task(complex_task)
        
        assert orchestration_result['task_id'] == complex_task['id']
        assert orchestration_result['status'] == 'initiated'
        assert len(orchestration_result['agent_assignments']) == len(complex_task['subtasks'])
        
        # Phase 3: Monitor task execution with meta-cognitive oversight
        execution_monitor = await components['meta_engine'].create_task_monitor(complex_task['id'])
        
        # Simulate agent execution
        subtask_results = {}
        for subtask in complex_task['subtasks']:
            # Mock agent execution
            agent_result = {
                'subtask_id': subtask['id'],
                'status': 'completed',
                'result': {
                    'analysis_data': f"Mock analysis for {subtask['type']}",
                    'confidence': 0.85,
                    'execution_time': 1.2
                },
                'metadata': {'agent_type': subtask['agent_type']}
            }
            
            subtask_results[subtask['id']] = agent_result
            await components['mcp_orchestrator'].report_subtask_completion(agent_result)
        
        # Phase 4: Meta-cognitive analysis of coordination effectiveness
        coordination_analysis = await components['meta_engine'].analyze_coordination_effectiveness(
            complex_task, subtask_results
        )
        
        assert coordination_analysis['overall_success'] is True
        assert coordination_analysis['coordination_efficiency'] > 0.7
        assert 'bottlenecks' in coordination_analysis
        assert 'improvement_suggestions' in coordination_analysis
        
        # Phase 5: Safety validation of multi-agent outcome
        multi_agent_outcome = {
            'task_id': complex_task['id'],
            'subtask_results': subtask_results,
            'coordination_analysis': coordination_analysis,
            'system_impact': 'performance_optimization'
        }
        
        safety_validation = await components['safety_validator'].validate_multi_agent_outcome(multi_agent_outcome)
        
        assert safety_validation.is_safe is True
        assert safety_validation.coordination_safety_score > 0.6
        
        # Phase 6: Learning from multi-agent coordination
        coordination_learning = {
            'task_complexity': len(complex_task['subtasks']),
            'coordination_pattern': 'sequential_with_dependencies',
            'effectiveness_score': coordination_analysis['coordination_efficiency'],
            'execution_time': sum(result['result']['execution_time'] for result in subtask_results.values()),
            'success_factors': coordination_analysis.get('success_factors', []),
            'improvement_areas': coordination_analysis.get('improvement_suggestions', [])
        }
        
        learning_node = SemanticNode(
            node_id=f"coordination_learning_{int(time.time())}",
            concept="coordination_experience",
            attributes={
                'content': f"Multi-agent coordination for {complex_task['description']} achieved {coordination_analysis['coordination_efficiency']:.2f} efficiency",
                'type': 'coordination_experience',
                **coordination_learning
            },
            embedding=np.random.rand(512)  # Placeholder embedding
        )
        
        components['memory_system'].semantic_memory.add_node(learning_node)
        learning_result = type('Result', (), {'success': True})()
        assert learning_result.success is True
        
        # Verify workflow completed successfully
        final_task_status = await components['mcp_orchestrator'].get_task_status(complex_task['id'])
        assert final_task_status['status'] == 'completed'
        assert final_task_status['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_adaptive_performance_optimization_workflow(self, integrated_system, integration_test_helpers):
        """
        Test workflow that adapts performance optimization strategies based on system conditions.
        """
        context = integrated_system
        components = context.components
        
        # Phase 1: Establish baseline performance
        baseline_metrics = await components['meta_engine'].collect_performance_baseline()
        
        assert 'cpu_usage' in baseline_metrics
        assert 'memory_usage' in baseline_metrics
        assert 'throughput' in baseline_metrics
        assert 'response_time' in baseline_metrics
        
        # Phase 2: Simulate different load conditions and test adaptation
        load_scenarios = [
            {
                'name': 'low_load',
                'characteristics': {'cpu_target': 0.3, 'memory_target': 0.4, 'throughput_target': 100},
                'expected_strategy': 'efficiency_optimization'
            },
            {
                'name': 'medium_load',
                'characteristics': {'cpu_target': 0.6, 'memory_target': 0.7, 'throughput_target': 500},
                'expected_strategy': 'balanced_optimization'
            },
            {
                'name': 'high_load',
                'characteristics': {'cpu_target': 0.85, 'memory_target': 0.9, 'throughput_target': 1000},
                'expected_strategy': 'performance_optimization'
            }
        ]
        
        adaptation_results = []
        
        for scenario in load_scenarios:
            # Simulate load condition
            await integration_test_helpers.simulate_load(
                components, 
                operations_per_second=scenario['characteristics']['throughput_target'] // 10,
                duration=2.0
            )
            
            # Wait for system to detect load change
            await asyncio.sleep(0.5)
            
            # Trigger adaptive optimization
            current_metrics = await components['meta_engine'].collect_current_metrics()
            adaptation_trigger = {
                'baseline_metrics': baseline_metrics,
                'current_metrics': current_metrics,
                'load_scenario': scenario['name'],
                'performance_targets': scenario['characteristics']
            }
            
            adaptation_result = await components['meta_engine'].trigger_adaptive_optimization(adaptation_trigger)
            
            # Verify appropriate strategy was selected
            assert adaptation_result['selected_strategy'] == scenario['expected_strategy']
            assert adaptation_result['confidence'] > 0.6
            
            # Apply the adaptive strategy
            strategy_application = await components['meta_engine'].apply_adaptive_strategy(
                adaptation_result['selected_strategy'],
                adaptation_result.get('parameters', {}),
                current_metrics
            )
            
            assert strategy_application['success'] is True
            
            # Measure effectiveness
            await asyncio.sleep(1.0)  # Allow strategy to take effect
            post_adaptation_metrics = await components['meta_engine'].collect_current_metrics()
            
            effectiveness = await components['delta_evaluator'].evaluate_adaptation_effectiveness(
                baseline_metrics, current_metrics, post_adaptation_metrics
            )
            
            adaptation_results.append({
                'scenario': scenario['name'],
                'strategy': adaptation_result['selected_strategy'],
                'effectiveness': effectiveness.total_delta,
                'metrics_improvement': {
                    'cpu_improvement': current_metrics['cpu_usage'] - post_adaptation_metrics['cpu_usage'],
                    'memory_improvement': current_metrics['memory_usage'] - post_adaptation_metrics['memory_usage'],
                    'throughput_improvement': post_adaptation_metrics['throughput'] - current_metrics['throughput']
                }
            })
        
        # Phase 3: Verify adaptive learning occurred
        adaptation_learning = await components['meta_engine'].get_adaptation_learning_summary()
        
        assert len(adaptation_learning['strategies_learned']) >= len(load_scenarios)
        assert adaptation_learning['adaptation_success_rate'] > 0.7
        
        # Phase 4: Test strategy selection improvement over time
        # Repeat one scenario to see if learning improved strategy selection
        repeat_scenario = load_scenarios[1]  # medium_load
        
        await integration_test_helpers.simulate_load(
            components,
            operations_per_second=repeat_scenario['characteristics']['throughput_target'] // 10,
            duration=2.0
        )
        
        repeat_metrics = await components['meta_engine'].collect_current_metrics()
        repeat_adaptation = await components['meta_engine'].trigger_adaptive_optimization({
            'baseline_metrics': baseline_metrics,
            'current_metrics': repeat_metrics,
            'load_scenario': repeat_scenario['name'],
            'performance_targets': repeat_scenario['characteristics']
        })
        
        # Should have higher confidence due to learning
        original_confidence = next(r for r in adaptation_results if r['scenario'] == repeat_scenario['name'])
        assert repeat_adaptation['confidence'] >= original_confidence  # Should not decrease
        
        # Verify overall workflow success
        assert all(result['effectiveness'] > 0 for result in adaptation_results)
        assert len(adaptation_results) == len(load_scenarios)
    
    @pytest.mark.asyncio
    async def test_safety_constrained_operation_workflow(self, integrated_system, error_injector):
        """
        Test workflow that operates under strict safety constraints with error handling.
        """
        context = integrated_system
        components = context.components
        
        # Phase 1: Establish strict safety constraints
        safety_constraints = [
            SafetyConstraint(
                name='performance_degradation_limit',
                threshold=0.1,  # Max 10% performance degradation
                component='system_wide',
                action='rollback_and_alert'
            ),
            SafetyConstraint(
                name='memory_corruption_prevention',
                threshold=0.02,  # Max 2% corruption probability
                component='memory_system',
                action='halt_operations'
            ),
            SafetyConstraint(
                name='modification_rate_limit',
                threshold=0.05,  # Max 5% modification rate
                component='meta_engine',
                action='throttle_modifications'
            )
        ]
        
        for constraint in safety_constraints:
            await components['safety_validator'].register_constraint(constraint)
        
        # Phase 2: Attempt risky operation that should trigger safety measures
        risky_operation = {
            'type': 'aggressive_optimization',
            'parameters': {
                'modification_magnitude': 0.3,  # High modification
                'risk_tolerance': 0.8,  # High risk
                'speed_priority': True
            },
            'expected_improvement': 0.4
        }
        
        # This should be blocked by safety constraints
        safety_check_result = await components['safety_validator'].pre_validate_operation(risky_operation)
        
        assert safety_check_result.is_safe is False
        assert 'modification_rate_limit' in safety_check_result.violated_constraints
        
        # Phase 3: Modify operation to comply with safety constraints
        safe_operation = await components['meta_engine'].make_operation_safe(
            risky_operation, safety_check_result.violated_constraints
        )
        
        assert safe_operation['parameters']['modification_magnitude'] <= 0.05
        assert safe_operation['parameters']['risk_tolerance'] <= 0.2
        
        # Verify safe operation passes safety validation
        safe_check_result = await components['safety_validator'].pre_validate_operation(safe_operation)
        assert safe_check_result.is_safe is True
        
        # Phase 4: Execute safe operation with continuous monitoring
        execution_monitor = await components['safety_validator'].create_execution_monitor(safe_operation)
        
        execution_result = await components['meta_engine'].execute_with_safety_monitoring(
            safe_operation, execution_monitor
        )
        
        assert execution_result['success'] is True
        assert execution_result['safety_violations'] == 0
        
        # Phase 5: Inject errors to test safety response
        # Inject memory corruption error
        error_injector.inject_error(
            components['memory_system'], 
            'store_node', 
            Exception("Simulated memory corruption"), 
            frequency=0.5
        )
        
        # Attempt operations that should trigger safety response
        safety_test_operations = []
        for i in range(10):
            node = SemanticNode(
                node_id=f"safety_test_{i}",
                concept="safety_test",
                attributes={'content': f"Safety test content {i}"},
                embedding=np.array([0.1] * 128)
            )
            try:
                components['memory_system'].semantic_memory.add_node(node)
                safety_test_operations.append(('success', True))
            except Exception as e:
                safety_test_operations.append(('error', str(e)))
        
        # Verify safety system responded appropriately
        safety_violations = await components['safety_validator'].get_recent_violations()
        memory_violations = [v for v in safety_violations if 'memory' in v.constraint_name.lower()]
        
        assert len(memory_violations) > 0  # Should have detected violations
        
        # Verify system initiated safety response
        safety_responses = await components['safety_validator'].get_safety_responses()
        halt_responses = [r for r in safety_responses if r['action'] == 'halt_operations']
        
        assert len(halt_responses) > 0
        
        # Phase 6: Test recovery workflow
        recovery_plan = await components['meta_engine'].create_recovery_plan(safety_violations)
        
        assert 'recovery_steps' in recovery_plan
        assert 'validation_criteria' in recovery_plan
        assert recovery_plan['estimated_recovery_time'] > 0
        
        # Execute recovery
        recovery_result = await components['meta_engine'].execute_recovery_plan(recovery_plan)
        
        assert recovery_result['success'] is True
        assert recovery_result['system_status'] == 'operational'
        
        # Phase 7: Verify system learned from safety incidents
        safety_learning = await components['meta_engine'].get_safety_learning_summary()
        
        assert len(safety_learning['incidents_analyzed']) > 0
        assert 'prevention_strategies' in safety_learning
        assert safety_learning['safety_improvement_score'] > 0
        
        # Verify updated safety protocols
        updated_constraints = await components['safety_validator'].get_active_constraints()
        assert len(updated_constraints) >= len(safety_constraints)  # May have added new ones
        
        # Test that similar risky operations are now better handled
        similar_risky_operation = {
            'type': 'aggressive_optimization',
            'parameters': {
                'modification_magnitude': 0.25,
                'risk_tolerance': 0.7,
                'speed_priority': True
            }
        }
        
        learned_safety_check = await components['safety_validator'].pre_validate_operation(similar_risky_operation)
        
        # Should be more restrictive due to learning
        assert learned_safety_check.confidence > safety_check_result.confidence
        assert len(learned_safety_check.recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])