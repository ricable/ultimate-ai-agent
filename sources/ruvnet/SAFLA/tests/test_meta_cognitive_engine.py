"""
Test suite for Meta-Cognitive Engine implementation.

This module tests the core meta-cognitive capabilities including:
- Self-Awareness Module: System state monitoring and self-reflection
- Goal Management: Dynamic goal setting, tracking, and adaptation
- Strategy Selection: Context-aware strategy selection and optimization
- Performance Monitoring: Real-time performance tracking and analysis
- Adaptation Engine: Continuous learning and self-modification capabilities

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

# Import the components we'll be testing (these don't exist yet - TDD approach)
from safla.core.meta_cognitive_engine import (
    MetaCognitiveEngine,
    SelfAwarenessModule,
    GoalManager,
    StrategySelector,
    PerformanceMonitor,
    AdaptationEngine,
    SystemState,
    Goal,
    Strategy,
    PerformanceMetrics,
    AdaptationResult
)


class TestSelfAwarenessModule:
    """Test suite for Self-Awareness Module - system state monitoring and self-reflection."""
    
    @pytest.fixture
    def self_awareness_module(self):
        """Create a SelfAwarenessModule instance for testing."""
        return SelfAwarenessModule()
    
    def test_initialization(self, self_awareness_module):
        """Test that SelfAwarenessModule initializes correctly."""
        assert self_awareness_module is not None
        assert hasattr(self_awareness_module, 'observation_points')
        assert hasattr(self_awareness_module, 'introspection_enabled')
        assert self_awareness_module.introspection_enabled is True
    
    def test_system_state_monitoring(self, self_awareness_module):
        """Test that the module can monitor system state."""
        # This should fail initially - we need to implement get_current_state
        state = self_awareness_module.get_current_state()
        
        assert isinstance(state, SystemState)
        assert hasattr(state, 'timestamp')
        assert hasattr(state, 'memory_usage')
        assert hasattr(state, 'cpu_usage')
        assert hasattr(state, 'active_goals')
        assert hasattr(state, 'current_strategies')
        assert hasattr(state, 'performance_metrics')
    
    def test_introspective_monitoring(self, self_awareness_module):
        """Test introspective monitoring of internal processes."""
        # Configure observation points
        observation_points = [
            'goal_processing',
            'strategy_selection',
            'performance_evaluation',
            'adaptation_decisions'
        ]
        
        self_awareness_module.configure_observation_points(observation_points)
        
        # Start monitoring
        self_awareness_module.start_introspection()
        
        # Simulate some internal activity
        self_awareness_module.observe_internal_process('goal_processing', {
            'goals_evaluated': 5,
            'processing_time': 0.1,
            'decisions_made': 3
        })
        
        # Get introspection data
        introspection_data = self_awareness_module.get_introspection_data()
        
        assert 'goal_processing' in introspection_data
        assert introspection_data['goal_processing']['goals_evaluated'] == 5
        assert introspection_data['goal_processing']['processing_time'] == 0.1
    
    def test_self_reflection_capabilities(self, self_awareness_module):
        """Test self-reflection on system behavior and performance."""
        # Provide historical state data
        historical_states = [
            SystemState(timestamp=time.time() - 300, memory_usage=0.6, cpu_usage=0.4),
            SystemState(timestamp=time.time() - 200, memory_usage=0.7, cpu_usage=0.5),
            SystemState(timestamp=time.time() - 100, memory_usage=0.8, cpu_usage=0.6),
        ]
        
        current_state = SystemState(timestamp=time.time(), memory_usage=0.9, cpu_usage=0.7)
        
        # Perform self-reflection
        reflection_result = self_awareness_module.reflect_on_performance(
            current_state, historical_states
        )
        
        assert hasattr(reflection_result, 'trends_identified')
        assert hasattr(reflection_result, 'anomalies_detected')
        assert hasattr(reflection_result, 'improvement_opportunities')
        assert hasattr(reflection_result, 'confidence_score')
        
        # Should identify increasing resource usage trend
        assert 'increasing_memory_usage' in reflection_result.trends_identified
    
    def test_configurable_observation_points(self, self_awareness_module):
        """Test that observation points can be dynamically configured."""
        initial_points = ['basic_monitoring']
        self_awareness_module.configure_observation_points(initial_points)
        assert self_awareness_module.observation_points == initial_points
        
        # Add new observation points
        additional_points = ['advanced_monitoring', 'deep_introspection']
        self_awareness_module.add_observation_points(additional_points)
        
        expected_points = initial_points + additional_points
        assert set(self_awareness_module.observation_points) == set(expected_points)
        
        # Remove observation points
        self_awareness_module.remove_observation_points(['basic_monitoring'])
        assert 'basic_monitoring' not in self_awareness_module.observation_points


class TestGoalManager:
    """Test suite for Goal Management - dynamic goal setting, tracking, and adaptation."""
    
    @pytest.fixture
    def goal_manager(self):
        """Create a GoalManager instance for testing."""
        return GoalManager()
    
    def test_initialization(self, goal_manager):
        """Test that GoalManager initializes correctly."""
        assert goal_manager is not None
        assert hasattr(goal_manager, 'active_goals')
        assert hasattr(goal_manager, 'goal_hierarchy')
        assert hasattr(goal_manager, 'priority_manager')
        assert len(goal_manager.active_goals) == 0
    
    def test_dynamic_goal_creation(self, goal_manager):
        """Test dynamic goal setting and creation."""
        # Create a new goal
        goal = Goal(
            id="test_goal_1",
            description="Improve system performance by 10%",
            priority=0.8,
            target_metrics={'performance_delta': 0.1},
            deadline=time.time() + 3600,  # 1 hour from now
            dependencies=[]
        )
        
        goal_id = goal_manager.create_goal(goal)
        
        assert goal_id == "test_goal_1"
        assert goal_id in goal_manager.active_goals
        assert goal_manager.active_goals[goal_id].priority == 0.8
    
    def test_goal_hierarchy_management(self, goal_manager):
        """Test hierarchical goal organization with dependencies."""
        # Create parent goal
        parent_goal = Goal(
            id="parent_goal",
            description="Optimize overall system",
            priority=0.9,
            target_metrics={'overall_improvement': 0.15}
        )
        
        # Create child goals
        child_goal_1 = Goal(
            id="child_goal_1",
            description="Optimize memory usage",
            priority=0.7,
            target_metrics={'memory_efficiency': 0.1},
            dependencies=["parent_goal"]
        )
        
        child_goal_2 = Goal(
            id="child_goal_2",
            description="Optimize CPU usage",
            priority=0.6,
            target_metrics={'cpu_efficiency': 0.1},
            dependencies=["parent_goal"]
        )
        
        # Add goals to manager
        goal_manager.create_goal(parent_goal)
        goal_manager.create_goal(child_goal_1)
        goal_manager.create_goal(child_goal_2)
        
        # Test hierarchy structure
        hierarchy = goal_manager.get_goal_hierarchy()
        assert "parent_goal" in hierarchy
        assert len(hierarchy["parent_goal"]["children"]) == 2
        assert "child_goal_1" in hierarchy["parent_goal"]["children"]
        assert "child_goal_2" in hierarchy["parent_goal"]["children"]
    
    def test_priority_management_and_conflict_resolution(self, goal_manager):
        """Test priority management and conflict resolution between goals."""
        # Create conflicting goals
        goal_1 = Goal(
            id="speed_goal",
            description="Maximize processing speed",
            priority=0.8,
            target_metrics={'speed': 'maximize'},
            resource_requirements={'cpu': 0.8, 'memory': 0.6}
        )
        
        goal_2 = Goal(
            id="efficiency_goal",
            description="Minimize resource usage",
            priority=0.7,
            target_metrics={'resource_usage': 'minimize'},
            resource_requirements={'cpu': 0.3, 'memory': 0.4}
        )
        
        goal_manager.create_goal(goal_1)
        goal_manager.create_goal(goal_2)
        
        # Detect conflicts
        conflicts = goal_manager.detect_goal_conflicts()
        assert len(conflicts) > 0
        assert 'resource_conflict' in conflicts[0]['type']
        
        # Resolve conflicts
        resolution = goal_manager.resolve_conflicts(conflicts)
        assert hasattr(resolution, 'resolution_strategy')
        assert hasattr(resolution, 'adjusted_priorities')
        assert hasattr(resolution, 'resource_allocation')
    
    def test_goal_tracking_and_progress_monitoring(self, goal_manager):
        """Test goal progress tracking and monitoring."""
        goal = Goal(
            id="tracking_test_goal",
            description="Test goal tracking",
            priority=0.5,
            target_metrics={'test_metric': 0.8},
            deadline=time.time() + 1800  # 30 minutes
        )
        
        goal_manager.create_goal(goal)
        
        # Update goal progress
        progress_update = {
            'test_metric': 0.4,  # 50% of target achieved
            'timestamp': time.time(),
            'additional_data': {'steps_completed': 5, 'total_steps': 10}
        }
        
        goal_manager.update_goal_progress("tracking_test_goal", progress_update)
        
        # Get goal status
        status = goal_manager.get_goal_status("tracking_test_goal")
        assert status['progress_percentage'] == 50.0
        assert status['estimated_completion'] is not None
        assert status['on_track'] is True
    
    def test_adaptive_goal_modification(self, goal_manager):
        """Test adaptive modification of goals based on context changes."""
        goal = Goal(
            id="adaptive_goal",
            description="Adaptive test goal",
            priority=0.6,
            target_metrics={'adaptive_metric': 0.7}
        )
        
        goal_manager.create_goal(goal)
        
        # Simulate context change
        context_change = {
            'resource_availability': 0.3,  # Reduced resources
            'time_pressure': 0.8,  # High time pressure
            'external_constraints': ['memory_limit', 'cpu_throttling']
        }
        
        # Adapt goal based on context
        adaptation_result = goal_manager.adapt_goal("adaptive_goal", context_change)
        
        assert adaptation_result.goal_modified is True
        assert adaptation_result.new_priority != 0.6  # Priority should change
        assert adaptation_result.new_target_metrics != {'adaptive_metric': 0.7}


class TestStrategySelector:
    """Test suite for Strategy Selection - context-aware strategy selection and optimization."""
    
    @pytest.fixture
    def strategy_selector(self):
        """Create a StrategySelector instance for testing."""
        return StrategySelector()
    
    def test_initialization(self, strategy_selector):
        """Test that StrategySelector initializes correctly."""
        assert strategy_selector is not None
        assert hasattr(strategy_selector, 'strategy_repository')
        assert hasattr(strategy_selector, 'performance_history')
        assert hasattr(strategy_selector, 'context_analyzer')
    
    def test_strategy_repository_management(self, strategy_selector):
        """Test strategy repository operations."""
        # Create test strategies
        strategy_1 = Strategy(
            id="memory_optimization_v1",
            name="Basic Memory Optimization",
            description="Simple memory cleanup strategy",
            applicable_contexts=['high_memory_usage'],
            performance_metrics={'memory_reduction': 0.15, 'execution_time': 0.5},
            implementation_complexity=0.3
        )
        
        strategy_2 = Strategy(
            id="memory_optimization_v2",
            name="Advanced Memory Optimization",
            description="Advanced memory management with compression",
            applicable_contexts=['high_memory_usage', 'memory_pressure'],
            performance_metrics={'memory_reduction': 0.25, 'execution_time': 1.2},
            implementation_complexity=0.7
        )
        
        # Add strategies to repository
        strategy_selector.add_strategy(strategy_1)
        strategy_selector.add_strategy(strategy_2)
        
        # Test repository operations
        assert len(strategy_selector.strategy_repository) == 2
        assert "memory_optimization_v1" in strategy_selector.strategy_repository
        assert "memory_optimization_v2" in strategy_selector.strategy_repository
        
        # Test strategy retrieval
        retrieved_strategy = strategy_selector.get_strategy("memory_optimization_v1")
        assert retrieved_strategy.name == "Basic Memory Optimization"
    
    def test_context_aware_strategy_selection(self, strategy_selector):
        """Test context-aware strategy selection algorithm."""
        # Setup strategies
        strategies = [
            Strategy(
                id="fast_strategy",
                applicable_contexts=['time_critical', 'low_resources'],
                performance_metrics={'speed': 0.9, 'accuracy': 0.6},
                resource_requirements={'cpu': 0.3, 'memory': 0.2}
            ),
            Strategy(
                id="accurate_strategy",
                applicable_contexts=['high_accuracy_required', 'sufficient_resources'],
                performance_metrics={'speed': 0.4, 'accuracy': 0.95},
                resource_requirements={'cpu': 0.8, 'memory': 0.7}
            ),
            Strategy(
                id="balanced_strategy",
                applicable_contexts=['general_purpose', 'moderate_resources'],
                performance_metrics={'speed': 0.7, 'accuracy': 0.8},
                resource_requirements={'cpu': 0.5, 'memory': 0.4}
            )
        ]
        
        for strategy in strategies:
            strategy_selector.add_strategy(strategy)
        
        # Test selection for time-critical context
        time_critical_context = {
            'time_pressure': 0.9,
            'available_resources': {'cpu': 0.4, 'memory': 0.3},
            'accuracy_requirements': 0.6,
            'current_situation': 'time_critical'
        }
        
        selected_strategy = strategy_selector.select_strategy(time_critical_context)
        assert selected_strategy.id == "fast_strategy"
        
        # Test selection for high-accuracy context
        accuracy_context = {
            'time_pressure': 0.2,
            'available_resources': {'cpu': 0.9, 'memory': 0.8},
            'accuracy_requirements': 0.9,
            'current_situation': 'high_accuracy_required'
        }
        
        selected_strategy = strategy_selector.select_strategy(accuracy_context)
        assert selected_strategy.id == "accurate_strategy"
    
    def test_performance_based_optimization(self, strategy_selector):
        """Test performance-based strategy optimization."""
        strategy = Strategy(
            id="optimizable_strategy",
            performance_metrics={'baseline_score': 0.5}
        )
        strategy_selector.add_strategy(strategy)
        
        # Record performance history
        performance_records = [
            {'strategy_id': 'optimizable_strategy', 'context': 'context_1', 'score': 0.6, 'timestamp': time.time() - 300},
            {'strategy_id': 'optimizable_strategy', 'context': 'context_1', 'score': 0.65, 'timestamp': time.time() - 200},
            {'strategy_id': 'optimizable_strategy', 'context': 'context_1', 'score': 0.7, 'timestamp': time.time() - 100},
        ]
        
        for record in performance_records:
            strategy_selector.record_performance(record)
        
        # Optimize strategy based on performance history
        optimization_result = strategy_selector.optimize_strategy('optimizable_strategy')
        
        assert optimization_result.performance_improved is True
        assert optimization_result.new_performance_metrics['baseline_score'] > 0.5
        assert hasattr(optimization_result, 'optimization_details')
    
    def test_strategy_learning_and_adaptation(self, strategy_selector):
        """Test strategy learning and adaptation capabilities."""
        # Create a base strategy
        base_strategy = Strategy(
            id="learning_strategy",
            name="Learning Strategy",
            performance_metrics={'initial_score': 0.4}
        )
        strategy_selector.add_strategy(base_strategy)
        
        # Simulate learning from multiple contexts
        learning_data = [
            {'context': {'situation': 'A', 'resources': 0.5}, 'outcome': 0.6},
            {'context': {'situation': 'A', 'resources': 0.7}, 'outcome': 0.8},
            {'context': {'situation': 'B', 'resources': 0.5}, 'outcome': 0.3},
            {'context': {'situation': 'B', 'resources': 0.9}, 'outcome': 0.7},
        ]
        
        # Apply learning
        learning_result = strategy_selector.learn_from_experience('learning_strategy', learning_data)
        
        assert learning_result.strategy_updated is True
        assert hasattr(learning_result, 'learned_patterns')
        assert hasattr(learning_result, 'context_preferences')
        
        # Test that learned patterns influence future selections
        test_context = {'situation': 'A', 'resources': 0.8}
        selection_confidence = strategy_selector.get_selection_confidence('learning_strategy', test_context)
        assert selection_confidence > 0.5  # Should be confident for learned context


class TestPerformanceMonitor:
    """Test suite for Performance Monitoring - real-time performance tracking and analysis."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor()
    
    def test_initialization(self, performance_monitor):
        """Test that PerformanceMonitor initializes correctly."""
        assert performance_monitor is not None
        assert hasattr(performance_monitor, 'metrics_collectors')
        assert hasattr(performance_monitor, 'real_time_dashboard')
        assert hasattr(performance_monitor, 'alert_system')
        assert hasattr(performance_monitor, 'trend_analyzer')
    
    def test_real_time_performance_tracking(self, performance_monitor):
        """Test real-time performance tracking capabilities."""
        # Start monitoring
        performance_monitor.start_monitoring()
        
        # Simulate some system activity
        time.sleep(0.1)  # Brief pause to generate metrics
        
        # Get current metrics
        current_metrics = performance_monitor.get_current_metrics()
        
        assert isinstance(current_metrics, PerformanceMetrics)
        assert hasattr(current_metrics, 'timestamp')
        assert hasattr(current_metrics, 'cpu_usage')
        assert hasattr(current_metrics, 'memory_usage')
        assert hasattr(current_metrics, 'throughput')
        assert hasattr(current_metrics, 'latency')
        assert hasattr(current_metrics, 'error_rate')
        
        # Verify metrics are reasonable
        assert 0 <= current_metrics.cpu_usage <= 1.0
        assert 0 <= current_metrics.memory_usage <= 1.0
        assert current_metrics.throughput >= 0
    
    def test_multi_dimensional_performance_tracking(self, performance_monitor):
        """Test tracking of multiple performance dimensions."""
        # Configure multiple metric dimensions
        dimensions = [
            'computational_efficiency',
            'memory_efficiency', 
            'network_efficiency',
            'goal_achievement_rate',
            'strategy_effectiveness',
            'adaptation_success_rate'
        ]
        
        performance_monitor.configure_dimensions(dimensions)
        
        # Record metrics for each dimension
        test_metrics = {
            'computational_efficiency': 0.85,
            'memory_efficiency': 0.72,
            'network_efficiency': 0.91,
            'goal_achievement_rate': 0.68,
            'strategy_effectiveness': 0.79,
            'adaptation_success_rate': 0.83
        }
        
        performance_monitor.record_metrics(test_metrics)
        
        # Retrieve dimensional analysis
        dimensional_analysis = performance_monitor.get_dimensional_analysis()
        
        assert len(dimensional_analysis) == len(dimensions)
        for dimension in dimensions:
            assert dimension in dimensional_analysis
            assert 'current_value' in dimensional_analysis[dimension]
            assert 'trend' in dimensional_analysis[dimension]
    
    def test_trend_analysis_and_prediction(self, performance_monitor):
        """Test trend analysis and performance prediction."""
        # Generate historical performance data
        historical_data = []
        base_time = time.time() - 3600  # 1 hour ago
        
        for i in range(60):  # 60 data points over 1 hour
            timestamp = base_time + (i * 60)  # Every minute
            performance_value = 0.5 + 0.3 * (i / 60) + 0.1 * (i % 10) / 10  # Upward trend with noise
            
            historical_data.append({
                'timestamp': timestamp,
                'performance_score': performance_value,
                'cpu_usage': 0.4 + 0.2 * (i / 60),
                'memory_usage': 0.3 + 0.4 * (i / 60)
            })
        
        performance_monitor.load_historical_data(historical_data)
        
        # Perform trend analysis
        trend_analysis = performance_monitor.analyze_trends()
        
        assert hasattr(trend_analysis, 'performance_trend')
        assert hasattr(trend_analysis, 'trend_strength')
        assert hasattr(trend_analysis, 'predicted_values')
        assert trend_analysis.performance_trend in ['increasing', 'decreasing', 'stable']
        
        # Test prediction
        future_prediction = performance_monitor.predict_performance(prediction_horizon=300)  # 5 minutes ahead
        assert hasattr(future_prediction, 'predicted_score')
        assert hasattr(future_prediction, 'confidence_interval')
        assert hasattr(future_prediction, 'prediction_accuracy')
    
    def test_alerting_system(self, performance_monitor):
        """Test real-time alerting system for performance issues."""
        # Configure alert thresholds
        alert_config = {
            'cpu_usage': {'warning': 0.7, 'critical': 0.9},
            'memory_usage': {'warning': 0.8, 'critical': 0.95},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'performance_degradation': {'warning': 0.15, 'critical': 0.3}
        }
        
        performance_monitor.configure_alerts(alert_config)
        
        # Simulate performance issues
        critical_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=0.95,  # Critical level
            memory_usage=0.85,  # Warning level
            error_rate=0.12,  # Critical level
            throughput=100,
            latency=0.5
        )
        
        performance_monitor.process_metrics(critical_metrics)
        
        # Check for triggered alerts
        active_alerts = performance_monitor.get_active_alerts()
        
        assert len(active_alerts) > 0
        
        # Verify critical alerts
        critical_alerts = [alert for alert in active_alerts if alert['severity'] == 'critical']
        assert len(critical_alerts) >= 2  # CPU and error rate should trigger critical alerts
        
        # Verify alert details
        cpu_alert = next((alert for alert in critical_alerts if alert['metric'] == 'cpu_usage'), None)
        assert cpu_alert is not None
        assert cpu_alert['current_value'] == 0.95
        assert cpu_alert['threshold'] == 0.9
    
    def test_performance_dashboard_integration(self, performance_monitor):
        """Test integration with real-time performance dashboard."""
        # Configure dashboard
        dashboard_config = {
            'refresh_rate': 1.0,  # 1 second
            'metrics_to_display': ['cpu_usage', 'memory_usage', 'throughput', 'goal_progress'],
            'chart_types': {'cpu_usage': 'line', 'memory_usage': 'gauge', 'throughput': 'bar'},
            'time_window': 300  # 5 minutes
        }
        
        performance_monitor.configure_dashboard(dashboard_config)
        
        # Start dashboard
        dashboard_status = performance_monitor.start_dashboard()
        assert dashboard_status['status'] == 'running'
        assert dashboard_status['url'] is not None
        
        # Get dashboard data
        dashboard_data = performance_monitor.get_dashboard_data()
        
        assert 'metrics' in dashboard_data
        assert 'charts' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'last_updated' in dashboard_data
        
        # Verify metrics are properly formatted for dashboard
        for metric in dashboard_config['metrics_to_display']:
            assert metric in dashboard_data['metrics']


class TestAdaptationEngine:
    """Test suite for Adaptation Engine - continuous learning and self-modification capabilities."""
    
    @pytest.fixture
    def adaptation_engine(self):
        """Create an AdaptationEngine instance for testing."""
        return AdaptationEngine()
    
    def test_initialization(self, adaptation_engine):
        """Test that AdaptationEngine initializes correctly."""
        assert adaptation_engine is not None
        assert hasattr(adaptation_engine, 'learning_algorithms')
        assert hasattr(adaptation_engine, 'experience_database')
        assert hasattr(adaptation_engine, 'pattern_recognizer')
        assert hasattr(adaptation_engine, 'self_modification_engine')
    
    def test_experience_based_learning(self, adaptation_engine):
        """Test learning from experience and pattern recognition."""
        # Create experience data
        experiences = [
            {
                'context': {'task_type': 'optimization', 'resource_level': 'high', 'time_pressure': 'low'},
                'action': 'thorough_analysis_strategy',
                'outcome': {'success': True, 'performance_gain': 0.25, 'time_taken': 120},
                'timestamp': time.time() - 1000
            },
            {
                'context': {'task_type': 'optimization', 'resource_level': 'low', 'time_pressure': 'high'},
                'action': 'quick_heuristic_strategy',
                'outcome': {'success': True, 'performance_gain': 0.10, 'time_taken': 30},
                'timestamp': time.time() - 800
            },
            {
                'context': {'task_type': 'optimization', 'resource_level': 'high', 'time_pressure': 'high'},
                'action': 'thorough_analysis_strategy',
                'outcome': {'success': False, 'performance_gain': 0.0, 'time_taken': 200},
                'timestamp': time.time() - 600
            }
        ]
        
        # Add experiences to the engine
        for experience in experiences:
            adaptation_engine.add_experience(experience)
        
        # Learn patterns from experiences
        learning_result = adaptation_engine.learn_from_experiences()
        
        assert learning_result.patterns_discovered > 0
        assert hasattr(learning_result, 'learned_rules')
        assert hasattr(learning_result, 'confidence_scores')
        
        # Test pattern recognition
        test_context = {'task_type': 'optimization', 'resource_level': 'low', 'time_pressure': 'high'}
        recommended_action = adaptation_engine.recommend_action(test_context)
        
        assert recommended_action.action == 'quick_heuristic_strategy'
        assert recommended_action.confidence > 0.5
    
    def test_continuous_self_modification(self, adaptation_engine):
        """Test continuous learning and self-modification capabilities."""
        # Configure self-modification parameters
        modification_config = {
            'learning_rate': 0.1,
            'adaptation_threshold': 0.15,
            'safety_constraints': ['no_core_system_changes', 'preserve_safety_mechanisms'],
            'modification_scope': ['strategy_parameters', 'goal_priorities', 'performance_thresholds']
        }
        
        adaptation_engine.configure_self_modification(modification_config)
        
        # Simulate performance feedback that triggers adaptation
        performance_feedback = {
            'current_performance': 0.6,
            'target_performance': 0.8,
            'performance_gap': 0.2,  # Above adaptation threshold
            'context': {'system_load': 'medium', 'user_satisfaction': 0.7}
        }
        
        # Trigger adaptation
        adaptation_result = adaptation_engine.adapt_system(performance_feedback)
        
        assert isinstance(adaptation_result, AdaptationResult)
        assert adaptation_result.adaptation_applied is True
        assert hasattr(adaptation_result, 'modifications_made')
        assert hasattr(adaptation_result, 'expected_improvement')
        assert hasattr(adaptation_result, 'safety_validation_passed')
        
        # Verify safety constraints were respected
        assert adaptation_result.safety_validation_passed is True
        for modification in adaptation_result.modifications_made:
            assert modification['scope'] in modification_config['modification_scope']
    
    def test_machine_learning_integration(self, adaptation_engine):
        """Test integration with machine learning for pattern recognition."""
        # Configure ML models
        ml_config = {
            'pattern_recognition_model': 'random_forest',
            'performance_prediction_model': 'gradient_boosting',
            'anomaly_detection_model': 'isolation_forest',
            'training_data_size': 1000,
            'retraining_frequency': 'daily'
        }
        
        adaptation_engine.configure_ml_models(ml_config)
        
        # Generate training data
        training_data = []
        for i in range(100):
            context_features = {
                'feature_1': i / 100.0,
                'feature_2': (i % 10) / 10.0,
                'feature_3': ((i * 7) % 13) / 13.0
            }
            performance_outcome = 0.5 + 0.3 * context_features['feature_1'] + 0.2 * context_features['feature_2']
            
            training_data.append({
                'context_features': context_features,
                'performance_outcome': performance_outcome
            })
        
        # Train ML models
        training_result = adaptation_engine.train_ml_models(training_data)
        
        assert training_result.training_successful is True
        assert hasattr(training_result, 'model_accuracies')
        assert hasattr(training_result, 'feature_importances')
        
        # Test ML-based predictions
        test_context = {'feature_1': 0.7, 'feature_2': 0.3, 'feature_3': 0.5}
        prediction = adaptation_engine.predict_performance(test_context)
        
        assert hasattr(prediction, 'predicted_performance')
        assert hasattr(prediction, 'confidence_interval')
        assert 0 <= prediction.predicted_performance <= 1.0
    
    def test_adaptive_learning_mechanisms(self, adaptation_engine):
        """Test adaptive learning mechanisms that improve over time."""
        # Initialize learning parameters
        initial_learning_rate = 0.1
        adaptation_engine.set_learning_rate(initial_learning_rate)
        
        # Simulate learning episodes with varying success rates
        learning_episodes = [
            {'success_rate': 0.8, 'context_complexity': 0.3},
            {'success_rate': 0.6, 'context_complexity': 0.7},
            {'success_rate': 0.9, 'context_complexity': 0.2},
            {'success_rate': 0.4, 'context_complexity': 0.9},
            {'success_rate': 0.7, 'context_complexity': 0.5}
        ]
        
        for episode in learning_episodes:
            adaptation_engine.process_learning_episode(episode)
        
        # Check if learning rate adapted based on performance
        current_learning_rate = adaptation_engine.get_learning_rate()
        
        # Learning rate should have adapted based on success patterns
        assert current_learning_rate != initial_learning_rate
        
        # Test meta-learning capabilities
        meta_learning_result = adaptation_engine.perform_meta_learning()
        
        assert hasattr(meta_learning_result, 'learning_strategy_updated')
        assert hasattr(meta_learning_result, 'optimal_learning_rate')
        assert hasattr(meta_learning_result, 'context_specific_adaptations')


class TestMetaCognitiveEngineIntegration:
    """Test suite for integrated Meta-Cognitive Engine functionality."""
    
    @pytest.fixture
    def meta_cognitive_engine(self):
        """Create a complete MetaCognitiveEngine instance for integration testing."""
        return MetaCognitiveEngine()
    
    def test_engine_initialization(self, meta_cognitive_engine):
        """Test that the complete engine initializes with all components."""
        assert meta_cognitive_engine is not None
        assert hasattr(meta_cognitive_engine, 'self_awareness')
        assert hasattr(meta_cognitive_engine, 'goal_manager')
        assert hasattr(meta_cognitive_engine, 'strategy_selector')
        assert hasattr(meta_cognitive_engine, 'performance_monitor')
        assert hasattr(meta_cognitive_engine, 'adaptation_engine')
        
        # Verify all components are properly initialized
        assert isinstance(meta_cognitive_engine.self_awareness, SelfAwarenessModule)
        assert isinstance(meta_cognitive_engine.goal_manager, GoalManager)
        assert isinstance(meta_cognitive_engine.strategy_selector, StrategySelector)
        assert isinstance(meta_cognitive_engine.performance_monitor, PerformanceMonitor)
        assert isinstance(meta_cognitive_engine.adaptation_engine, AdaptationEngine)
    
    @pytest.mark.asyncio
    async def test_event_driven_architecture(self, meta_cognitive_engine):
        """Test event-driven architecture with async processing."""
        # Start the engine
        await meta_cognitive_engine.start()
        
        # Verify engine is running
        assert meta_cognitive_engine.is_running() is True
        
        # Test event processing
        test_event = {
            'type': 'performance_degradation',
            'data': {
                'metric': 'throughput',
                'current_value': 0.4,
                'expected_value': 0.7,
                'degradation_percentage': 0.43
            },
            'timestamp': time.time()
        }
        
        # Process event
        event_result = await meta_cognitive_engine.process_event(test_event)
        
        assert event_result.event_processed is True
        assert hasattr(event_result, 'actions_triggered')
        assert hasattr(event_result, 'components_notified')
        
        # Stop the engine
        await meta_cognitive_engine.stop()
        assert meta_cognitive_engine.is_running() is False
    
    def test_integration_with_existing_safla_components(self, meta_cognitive_engine):
        """Test integration with existing SAFLA components."""
        # Mock existing SAFLA components
        mock_delta_evaluation = Mock()
        mock_memory_system = Mock()
        mock_mcp_orchestration = Mock()
        mock_safety_validation = Mock()
        
        # Configure integration
        integration_config = {
            'delta_evaluation': mock_delta_evaluation,
            'memory_system': mock_memory_system,
            'mcp_orchestration': mock_mcp_orchestration,
            'safety_validation': mock_safety_validation
        }
        
        meta_cognitive_engine.configure_integration(integration_config)
        
        # Test that meta-cognitive engine can interact with existing components
        # Simulate a scenario where meta-cognitive engine needs to evaluate performance
        performance_data = {'metric': 'efficiency', 'value': 0.75}
        
        # This should trigger calls to integrated components
        integration_result = meta_cognitive_engine.evaluate_with_integration(performance_data)
        
        assert integration_result.integration_successful is True
        assert hasattr(integration_result, 'component_responses')
        
        # Verify that existing components were called
        mock_delta_evaluation.evaluate.assert_called()
        mock_memory_system.store_performance_data.assert_called()
    
    def test_meta_cognitive_feedback_loop(self, meta_cognitive_engine):
        """Test the complete meta-cognitive feedback loop."""
        # Configure the engine for a complete feedback loop test
        meta_cognitive_engine.configure_feedback_loop({
            'monitoring_interval': 0.1,  # Fast monitoring for testing
            'evaluation_threshold': 0.1,
            'adaptation_threshold': 0.2,
            'safety_checks_enabled': True
        })
        
        # Start monitoring
        meta_cognitive_engine.start_feedback_loop()
        
        # Simulate system activity that should trigger the feedback loop
        system_activity = {
            'goals_processed': 5,
            'strategies_executed': 3,
            'performance_metrics': {
                'efficiency': 0.6,
                'effectiveness': 0.7,
                'resource_usage': 0.8
            },
            'adaptation_opportunities': [
                {'type': 'strategy_optimization', 'potential_gain': 0.25},
                {'type': 'goal_reprioritization', 'potential_gain': 0.15}
            ]
        }
        
        # Process the activity through the feedback loop
        loop_result = meta_cognitive_engine.process_feedback_loop(system_activity)
        
        assert loop_result.loop_completed is True
        assert hasattr(loop_result, 'self_awareness_updates')
        assert hasattr(loop_result, 'goal_adjustments')
        assert hasattr(loop_result, 'strategy_optimizations')
        assert hasattr(loop_result, 'performance_improvements')
        assert hasattr(loop_result, 'adaptations_applied')
        
        # Verify that the feedback loop resulted in system improvements
        assert len(loop_result.adaptations_applied) > 0
        
        # Stop the feedback loop
        meta_cognitive_engine.stop_feedback_loop()
    
    def test_self_model_maintenance(self, meta_cognitive_engine):
        """Test self-model maintenance and updating capabilities."""
        # Initialize self-model
        initial_self_model = {
            'capabilities': ['goal_management', 'strategy_selection', 'performance_monitoring'],
            'limitations': ['limited_context_window', 'single_domain_focus'],
            'performance_characteristics': {
                'response_time': 0.5,
                'accuracy': 0.8,
                'resource_efficiency': 0.7
            },
            'knowledge_domains': ['optimization', 'planning', 'monitoring']
        }
        
        meta_cognitive_engine.initialize_self_model(initial_self_model)
        
        # Simulate learning new capabilities
        new_capability_data = {
            'capability': 'advanced_pattern_recognition',
            'performance_evidence': [
                {'task': 'pattern_task_1', 'success_rate': 0.85},
                {'task': 'pattern_task_2', 'success_rate': 0.92},
                {'task': 'pattern_task_3', 'success_rate': 0.78}
            ],
            'resource_requirements': {'cpu': 0.3, 'memory': 0.4}
        }
        
        # Update self-model with new capability
        model_update_result = meta_cognitive_engine.update_self_model(new_capability_data)
        
        assert model_update_result.model_updated is True
        assert 'advanced_pattern_recognition' in model_update_result.updated_capabilities
        
        # Test self-model querying
        current_capabilities = meta_cognitive_engine.get_current_capabilities()
        assert 'advanced_pattern_recognition' in current_capabilities
        
        # Test limitation discovery and updating
        limitation_discovery = {
            'limitation': 'context_switching_overhead',
            'evidence': [
                {'scenario': 'rapid_context_changes', 'performance_drop': 0.25},
                {'scenario': 'complex_multitasking', 'performance_drop': 0.3}
            ]
        }
        
        limitation_update_result = meta_cognitive_engine.update_limitations(limitation_discovery)
        assert limitation_update_result.limitations_updated is True
        assert 'context_switching_overhead' in limitation_update_result.new_limitations