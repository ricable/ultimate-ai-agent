"""
Test suite for Adaptive Safety Boundaries - ML-powered safety constraint adjustment.

This module tests the adaptive safety boundary capabilities for SAFLA:
- Dynamic safety constraint adjustment
- ML-powered risk assessment
- Real-time safety monitoring
- Constraint violation detection
- Safety policy learning and adaptation

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import numpy as np
import torch
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Import the classes we'll implement
from safla.core.adaptive_safety_boundaries import (
    AdaptiveSafetyBoundaries,
    SafetyConstraint,
    SafetyLevel,
    ConstraintType,
    SafetyMetrics,
    AdaptationStrategy,
    RiskAssessmentEngine,
    ConstraintOptimizer,
    SafetyMonitoringDashboard
)


class RiskLevel(Enum):
    """Risk level enumeration."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConstraintType(Enum):
    """Safety constraint type enumeration."""
    RESOURCE_LIMIT = "resource_limit"
    PERFORMANCE_BOUND = "performance_bound"
    SECURITY_POLICY = "security_policy"
    DATA_PRIVACY = "data_privacy"
    OPERATIONAL_SAFETY = "operational_safety"
    ETHICAL_GUIDELINE = "ethical_guideline"


class SafetyAction(Enum):
    """Safety action enumeration."""
    ALLOW = "allow"
    WARN = "warn"
    THROTTLE = "throttle"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class MockSafetyConstraint:
    """Mock safety constraint for testing."""
    constraint_id: str
    constraint_type: ConstraintType
    threshold: float
    current_value: float = 0.0
    is_active: bool = True
    priority: int = 1
    adaptation_rate: float = 0.1
    violation_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MockSafetyViolation:
    """Mock safety violation for testing."""
    violation_id: str
    constraint_id: str
    severity: RiskLevel
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class TestAdaptiveSafetyManager:
    """Test suite for Adaptive Safety Manager functionality."""
    
    @pytest.fixture
    def safety_config(self):
        """Create safety configuration for testing."""
        return SafetyConfig(
            adaptation_enabled=True,
            learning_rate=0.01,
            risk_tolerance=0.1,
            constraint_update_frequency=60,  # seconds
            violation_threshold=3,
            emergency_threshold=0.95,
            monitoring_interval=5,  # seconds
            safety_buffer=0.1,
            max_adaptation_rate=0.2,
            min_constraint_value=0.0,
            max_constraint_value=1.0
        )
    
    @pytest.fixture
    def safety_manager(self, safety_config):
        """Create AdaptiveSafetyBoundaries instance for testing."""
        return AdaptiveSafetyBoundaries(config=safety_config)
    
    @pytest.fixture
    def mock_constraints(self):
        """Create mock safety constraints for testing."""
        return [
            MockSafetyConstraint(
                constraint_id="cpu_limit",
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                threshold=0.8,
                current_value=0.6,
                priority=1
            ),
            MockSafetyConstraint(
                constraint_id="memory_limit",
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                threshold=0.9,
                current_value=0.7,
                priority=1
            ),
            MockSafetyConstraint(
                constraint_id="response_time",
                constraint_type=ConstraintType.PERFORMANCE_BOUND,
                threshold=1000.0,  # milliseconds
                current_value=500.0,
                priority=2
            ),
            MockSafetyConstraint(
                constraint_id="data_access",
                constraint_type=ConstraintType.DATA_PRIVACY,
                threshold=0.95,  # privacy score
                current_value=0.98,
                priority=3
            )
        ]
    
    def test_safety_manager_initialization(self, safety_manager, safety_config):
        """Test AdaptiveSafetyBoundaries initialization."""
        assert safety_manager.config == safety_config
        assert safety_manager.risk_assessment_engine is not None
        assert safety_manager.policy_learner is not None
        assert safety_manager.violation_detector is not None
        assert safety_manager.metrics_collector is not None
        assert safety_manager.boundary_optimizer is not None
        assert safety_manager.is_monitoring is False
        assert len(safety_manager.active_constraints) == 0
    
    def test_constraint_registration_and_management(self, safety_manager, mock_constraints):
        """Test safety constraint registration and management."""
        # Register constraints
        for constraint in mock_constraints:
            success = safety_manager.register_constraint(constraint)
            assert success
        
        assert len(safety_manager.get_active_constraints()) == 4
        
        # Test constraint retrieval
        cpu_constraint = safety_manager.get_constraint("cpu_limit")
        assert cpu_constraint is not None
        assert cpu_constraint.constraint_type == ConstraintType.RESOURCE_LIMIT
        
        # Test constraint deactivation
        success = safety_manager.deactivate_constraint("memory_limit")
        assert success
        assert len(safety_manager.get_active_constraints()) == 3
        
        # Test constraint reactivation
        success = safety_manager.activate_constraint("memory_limit")
        assert success
        assert len(safety_manager.get_active_constraints()) == 4
    
    @pytest.mark.asyncio
    async def test_real_time_safety_monitoring(self, safety_manager, mock_constraints):
        """Test real-time safety monitoring and violation detection."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        # Start monitoring
        await safety_manager.start_monitoring()
        assert safety_manager.is_monitoring
        
        # Simulate constraint violation
        violation_context = {
            "cpu_usage": 0.95,  # Exceeds 0.8 threshold
            "memory_usage": 0.85,
            "response_time": 1200.0,  # Exceeds 1000ms threshold
            "timestamp": datetime.now()
        }
        
        # Process monitoring data
        violations = await safety_manager.process_monitoring_data(violation_context)
        
        assert len(violations) >= 2  # CPU and response time violations
        assert any(v.constraint_id == "cpu_limit" for v in violations)
        assert any(v.constraint_id == "response_time" for v in violations)
        
        # Stop monitoring
        await safety_manager.stop_monitoring()
        assert not safety_manager.is_monitoring
    
    def test_risk_assessment_and_classification(self, safety_manager, mock_constraints):
        """Test risk assessment and classification."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        risk_engine = safety_manager.risk_assessment_engine
        
        # Test low risk scenario
        low_risk_context = {
            "cpu_usage": 0.5,
            "memory_usage": 0.6,
            "response_time": 300.0,
            "data_privacy_score": 0.99
        }
        
        risk_assessment = risk_engine.assess_risk(low_risk_context)
        
        assert risk_assessment.overall_risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]
        assert risk_assessment.risk_score < 0.3
        
        # Test high risk scenario
        high_risk_context = {
            "cpu_usage": 0.95,
            "memory_usage": 0.98,
            "response_time": 2000.0,
            "data_privacy_score": 0.85
        }
        
        risk_assessment = risk_engine.assess_risk(high_risk_context)
        
        assert risk_assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert risk_assessment.risk_score > 0.7
    
    @pytest.mark.asyncio
    async def test_adaptive_constraint_adjustment(self, safety_manager, mock_constraints):
        """Test adaptive adjustment of safety constraints."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        boundary_optimizer = safety_manager.boundary_optimizer
        
        # Simulate historical violation data
        violation_history = [
            MockSafetyViolation(
                violation_id=f"violation_{i}",
                constraint_id="cpu_limit",
                severity=RiskLevel.MODERATE,
                timestamp=datetime.now() - timedelta(minutes=i),
                context={"cpu_usage": 0.85 + (i * 0.02)}
            )
            for i in range(5)
        ]
        
        # Learn from violations and adapt constraints
        adaptations = await boundary_optimizer.adapt_constraints(
            constraints=safety_manager.get_active_constraints(),
            violation_history=violation_history,
            performance_metrics={"system_efficiency": 0.8, "user_satisfaction": 0.9}
        )
        
        assert len(adaptations) > 0
        
        # CPU limit should be adjusted based on violations
        cpu_adaptation = next((a for a in adaptations if a.constraint_id == "cpu_limit"), None)
        assert cpu_adaptation is not None
        assert cpu_adaptation.new_threshold != cpu_adaptation.old_threshold
    
    def test_safety_policy_learning(self, safety_manager):
        """Test safety policy learning from historical data."""
        policy_learner = safety_manager.policy_learner
        
        # Create training data from historical safety events
        training_data = []
        for i in range(100):
            context = {
                "cpu_usage": np.random.uniform(0.3, 0.95),
                "memory_usage": np.random.uniform(0.4, 0.9),
                "response_time": np.random.uniform(200, 1500),
                "concurrent_users": np.random.randint(10, 200)
            }
            
            # Determine appropriate safety action based on context
            if context["cpu_usage"] > 0.9 or context["response_time"] > 1200:
                action = SafetyAction.THROTTLE
            elif context["cpu_usage"] > 0.8 or context["response_time"] > 1000:
                action = SafetyAction.WARN
            else:
                action = SafetyAction.ALLOW
            
            training_data.append((context, action))
        
        # Train safety policy
        training_result = policy_learner.train_policy(training_data)
        
        assert training_result["success"]
        assert training_result["accuracy"] > 0.7
        assert "model_metrics" in training_result
        
        # Test policy prediction
        test_context = {
            "cpu_usage": 0.95,
            "memory_usage": 0.8,
            "response_time": 1300,
            "concurrent_users": 150
        }
        
        predicted_action = policy_learner.predict_action(test_context)
        assert predicted_action in [SafetyAction.THROTTLE, SafetyAction.BLOCK]
    
    def test_constraint_violation_detection(self, safety_manager, mock_constraints):
        """Test constraint violation detection and classification."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        violation_detector = safety_manager.violation_detector
        
        # Test normal operation (no violations)
        normal_context = {
            "cpu_usage": 0.6,
            "memory_usage": 0.7,
            "response_time": 500.0,
            "data_privacy_score": 0.98
        }
        
        violations = violation_detector.detect_violations(
            context=normal_context,
            constraints=safety_manager.get_active_constraints()
        )
        
        assert len(violations) == 0
        
        # Test violation scenario
        violation_context = {
            "cpu_usage": 0.95,  # Violates cpu_limit (0.8)
            "memory_usage": 0.95,  # Violates memory_limit (0.9)
            "response_time": 1500.0,  # Violates response_time (1000.0)
            "data_privacy_score": 0.90  # Violates data_access (0.95)
        }
        
        violations = violation_detector.detect_violations(
            context=violation_context,
            constraints=safety_manager.get_active_constraints()
        )
        
        assert len(violations) == 4  # All constraints violated
        
        # Check violation severity classification
        cpu_violation = next(v for v in violations if v.constraint_id == "cpu_limit")
        assert cpu_violation.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_emergency_safety_protocols(self, safety_manager, mock_constraints):
        """Test emergency safety protocols and responses."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        # Simulate critical safety violation
        critical_context = {
            "cpu_usage": 0.99,
            "memory_usage": 0.98,
            "response_time": 5000.0,
            "error_rate": 0.8,
            "system_temperature": 95.0  # Critical temperature
        }
        
        # Trigger emergency protocols
        emergency_response = await safety_manager.handle_emergency(critical_context)
        
        assert emergency_response["emergency_triggered"]
        assert emergency_response["action"] in [SafetyAction.EMERGENCY_STOP, SafetyAction.BLOCK]
        assert "mitigation_steps" in emergency_response
        assert len(emergency_response["mitigation_steps"]) > 0
        
        # Verify emergency state
        assert safety_manager.is_emergency_state()
        
        # Test emergency recovery
        recovery_context = {
            "cpu_usage": 0.4,
            "memory_usage": 0.5,
            "response_time": 300.0,
            "error_rate": 0.01,
            "system_temperature": 65.0
        }
        
        recovery_result = await safety_manager.attempt_recovery(recovery_context)
        
        if recovery_result["recovery_successful"]:
            assert not safety_manager.is_emergency_state()
    
    def test_safety_metrics_collection_and_analysis(self, safety_manager):
        """Test safety metrics collection and analysis."""
        metrics_collector = safety_manager.metrics_collector
        
        # Collect safety metrics over time
        for i in range(24):  # 24 hours of data
            timestamp = datetime.now() - timedelta(hours=i)
            metrics = {
                "violation_count": np.random.poisson(2),
                "average_risk_score": np.random.uniform(0.1, 0.8),
                "constraint_adjustments": np.random.poisson(1),
                "emergency_triggers": np.random.poisson(0.1),
                "system_availability": np.random.uniform(0.95, 1.0),
                "timestamp": timestamp
            }
            metrics_collector.record_metrics(metrics)
        
        # Analyze safety trends
        trend_analysis = metrics_collector.analyze_safety_trends(
            time_window=timedelta(hours=24)
        )
        
        assert "violation_trend" in trend_analysis
        assert "risk_trend" in trend_analysis
        assert "availability_trend" in trend_analysis
        assert "recommendations" in trend_analysis
        
        # Generate safety report
        safety_report = metrics_collector.generate_safety_report(
            time_period=timedelta(hours=24)
        )
        
        assert "summary_statistics" in safety_report
        assert "violation_analysis" in safety_report
        assert "constraint_effectiveness" in safety_report
        assert "improvement_recommendations" in safety_report
    
    def test_multi_objective_safety_optimization(self, safety_manager, mock_constraints):
        """Test multi-objective optimization of safety constraints."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        boundary_optimizer = safety_manager.boundary_optimizer
        
        # Define optimization objectives
        objectives = {
            "safety": {"weight": 0.4, "target": "maximize"},
            "performance": {"weight": 0.3, "target": "maximize"},
            "efficiency": {"weight": 0.2, "target": "maximize"},
            "user_experience": {"weight": 0.1, "target": "maximize"}
        }
        
        # Historical performance data
        performance_history = [
            {
                "safety_score": 0.85,
                "performance_score": 0.75,
                "efficiency_score": 0.80,
                "user_experience_score": 0.90,
                "constraint_config": {"cpu_limit": 0.8, "memory_limit": 0.9}
            }
            for _ in range(50)
        ]
        
        # Optimize constraints
        optimization_result = boundary_optimizer.multi_objective_optimize(
            constraints=safety_manager.get_active_constraints(),
            objectives=objectives,
            performance_history=performance_history,
            optimization_steps=100
        )
        
        assert optimization_result["success"]
        assert "optimized_constraints" in optimization_result
        assert "pareto_front" in optimization_result
        assert "objective_scores" in optimization_result
        
        # Verify optimization improved overall score
        assert optimization_result["objective_scores"]["combined_score"] > 0.7
    
    @pytest.mark.asyncio
    async def test_contextual_safety_adaptation(self, safety_manager, mock_constraints):
        """Test contextual adaptation of safety constraints."""
        # Register constraints
        for constraint in mock_constraints:
            safety_manager.register_constraint(constraint)
        
        # Test different operational contexts
        contexts = [
            {
                "name": "peak_hours",
                "time_of_day": "peak",
                "user_load": "high",
                "system_criticality": "high",
                "expected_adjustments": {"cpu_limit": -0.1, "response_time": -100}
            },
            {
                "name": "maintenance_window",
                "time_of_day": "night",
                "user_load": "low",
                "system_criticality": "low",
                "expected_adjustments": {"cpu_limit": +0.1, "memory_limit": +0.05}
            },
            {
                "name": "emergency_mode",
                "time_of_day": "any",
                "user_load": "any",
                "system_criticality": "critical",
                "expected_adjustments": {"cpu_limit": -0.2, "response_time": -200}
            }
        ]
        
        for context in contexts:
            # Apply contextual adaptation
            adaptations = await safety_manager.adapt_to_context(context)
            
            assert len(adaptations) > 0
            
            # Verify expected adjustments
            for constraint_id, expected_change in context["expected_adjustments"].items():
                adaptation = next((a for a in adaptations if a.constraint_id == constraint_id), None)
                if adaptation:
                    actual_change = adaptation.new_threshold - adaptation.old_threshold
                    # Allow some tolerance in the adaptation
                    assert abs(actual_change - expected_change) < abs(expected_change) * 0.5
    
    def test_safety_constraint_dependencies(self, safety_manager):
        """Test handling of safety constraint dependencies."""
        # Create constraints with dependencies
        cpu_constraint = MockSafetyConstraint(
            constraint_id="cpu_limit",
            constraint_type=ConstraintType.RESOURCE_LIMIT,
            threshold=0.8
        )
        
        memory_constraint = MockSafetyConstraint(
            constraint_id="memory_limit",
            constraint_type=ConstraintType.RESOURCE_LIMIT,
            threshold=0.9
        )
        
        response_constraint = MockSafetyConstraint(
            constraint_id="response_time",
            constraint_type=ConstraintType.PERFORMANCE_BOUND,
            threshold=1000.0
        )
        
        # Define dependencies: response_time depends on cpu_limit and memory_limit
        dependencies = {
            "response_time": ["cpu_limit", "memory_limit"]
        }
        
        # Register constraints with dependencies
        safety_manager.register_constraint(cpu_constraint)
        safety_manager.register_constraint(memory_constraint)
        safety_manager.register_constraint(response_constraint)
        safety_manager.set_constraint_dependencies(dependencies)
        
        # Test dependency-aware adaptation
        # When CPU limit is tightened, response time limit should also be adjusted
        original_response_threshold = response_constraint.threshold
        
        safety_manager.update_constraint_threshold("cpu_limit", 0.7)  # Tighten CPU limit
        
        # Response time constraint should be automatically adjusted
        updated_response_constraint = safety_manager.get_constraint("response_time")
        assert updated_response_constraint.threshold != original_response_threshold
    
    def test_safety_rule_engine(self, safety_manager):
        """Test safety rule engine for complex safety logic."""
        # Define safety rules
        safety_rules = [
            SafetyRule(
                rule_id="high_load_protection",
                condition="cpu_usage > 0.8 AND memory_usage > 0.8",
                action=SafetyAction.THROTTLE,
                priority=1
            ),
            SafetyRule(
                rule_id="response_time_degradation",
                condition="response_time > 1500 OR error_rate > 0.1",
                action=SafetyAction.WARN,
                priority=2
            ),
            SafetyRule(
                rule_id="critical_system_failure",
                condition="cpu_usage > 0.95 AND (error_rate > 0.5 OR response_time > 3000)",
                action=SafetyAction.EMERGENCY_STOP,
                priority=0  # Highest priority
            )
        ]
        
        # Register safety rules
        for rule in safety_rules:
            safety_manager.register_safety_rule(rule)
        
        # Test rule evaluation
        test_context = {
            "cpu_usage": 0.85,
            "memory_usage": 0.85,
            "response_time": 800,
            "error_rate": 0.05
        }
        
        triggered_rules = safety_manager.evaluate_safety_rules(test_context)
        
        # Should trigger high_load_protection rule
        assert len(triggered_rules) == 1
        assert triggered_rules[0].rule_id == "high_load_protection"
        assert triggered_rules[0].action == SafetyAction.THROTTLE
        
        # Test critical scenario
        critical_context = {
            "cpu_usage": 0.96,
            "memory_usage": 0.9,
            "response_time": 3500,
            "error_rate": 0.6
        }
        
        triggered_rules = safety_manager.evaluate_safety_rules(critical_context)
        
        # Should trigger critical_system_failure rule (highest priority)
        assert any(rule.rule_id == "critical_system_failure" for rule in triggered_rules)
        critical_rule = next(rule for rule in triggered_rules if rule.rule_id == "critical_system_failure")
        assert critical_rule.action == SafetyAction.EMERGENCY_STOP


class TestRiskAssessmentEngine:
    """Test suite for Risk Assessment Engine functionality."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create RiskAssessmentEngine instance for testing."""
        config = {
            "risk_factors": ["cpu_usage", "memory_usage", "response_time", "error_rate"],
            "risk_weights": [0.3, 0.25, 0.25, 0.2],
            "risk_thresholds": {
                RiskLevel.MINIMAL: 0.2,
                RiskLevel.LOW: 0.4,
                RiskLevel.MODERATE: 0.6,
                RiskLevel.HIGH: 0.8,
                RiskLevel.CRITICAL: 1.0
            }
        }
        return RiskAssessmentEngine(config=config)
    
    def test_risk_score_calculation(self, risk_engine):
        """Test risk score calculation from multiple factors."""
        # Test low risk scenario
        low_risk_context = {
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "response_time": 0.2,  # Normalized
            "error_rate": 0.01
        }
        
        risk_score = risk_engine.calculate_risk_score(low_risk_context)
        assert 0.0 <= risk_score <= 1.0
        assert risk_score < 0.4  # Should be low risk
        
        # Test high risk scenario
        high_risk_context = {
            "cpu_usage": 0.9,
            "memory_usage": 0.95,
            "response_time": 0.8,  # Normalized
            "error_rate": 0.3
        }
        
        risk_score = risk_engine.calculate_risk_score(high_risk_context)
        assert risk_score > 0.7  # Should be high risk
    
    def test_risk_level_classification(self, risk_engine):
        """Test risk level classification based on scores."""
        test_cases = [
            (0.1, RiskLevel.MINIMAL),
            (0.3, RiskLevel.LOW),
            (0.5, RiskLevel.MODERATE),
            (0.7, RiskLevel.HIGH),
            (0.9, RiskLevel.CRITICAL)
        ]
        
        for risk_score, expected_level in test_cases:
            classified_level = risk_engine.classify_risk_level(risk_score)
            assert classified_level == expected_level
    
    def test_temporal_risk_analysis(self, risk_engine):
        """Test temporal risk analysis and trend detection."""
        # Generate time series risk data
        risk_history = []
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=i)
            # Simulate increasing risk over time
            base_risk = 0.3 + (i / 100) * 0.4
            noise = np.random.normal(0, 0.05)
            risk_score = max(0, min(1, base_risk + noise))
            
            risk_history.append({
                "timestamp": timestamp,
                "risk_score": risk_score,
                "risk_level": risk_engine.classify_risk_level(risk_score)
            })
        
        # Analyze risk trends
        trend_analysis = risk_engine.analyze_risk_trends(risk_history)
        
        assert "trend_direction" in trend_analysis
        assert "trend_strength" in trend_analysis
        assert "risk_velocity" in trend_analysis
        assert "predicted_risk" in trend_analysis
        
        # Should detect increasing trend
        assert trend_analysis["trend_direction"] in ["increasing", "stable"]
    
    def test_risk_factor_importance(self, risk_engine):
        """Test risk factor importance analysis."""
        # Create scenarios with different factor combinations
        scenarios = [
            {"cpu_usage": 0.9, "memory_usage": 0.3, "response_time": 0.2, "error_rate": 0.01},
            {"cpu_usage": 0.3, "memory_usage": 0.9, "response_time": 0.2, "error_rate": 0.01},
            {"cpu_usage": 0.3, "memory_usage": 0.3, "response_time": 0.9, "error_rate": 0.01},
            {"cpu_usage": 0.3, "memory_usage": 0.3, "response_time": 0.2, "error_rate": 0.8}
        ]
        
        # Analyze factor importance
        importance_analysis = risk_engine.analyze_factor_importance(scenarios)
        
        assert "factor_importance" in importance_analysis
        assert "sensitivity_analysis" in importance_analysis
        
        # All factors should have importance scores
        for factor in risk_engine.config["risk_factors"]:
            assert factor in importance_analysis["factor_importance"]
            assert 0.0 <= importance_analysis["factor_importance"][factor] <= 1.0


class TestSafetyPolicyLearner:
    """Test suite for Safety Policy Learner functionality."""
    
    @pytest.fixture
    def policy_learner(self):
        """Create SafetyPolicyLearner instance for testing."""
        config = {
            "learning_algorithm": "reinforcement_learning",
            "state_features": ["cpu_usage", "memory_usage", "response_time", "user_load"],
            "action_space": [action.value for action in SafetyAction],
            "reward_function": "safety_performance_balance",
            "exploration_rate": 0.1,
            "learning_rate": 0.01,
            "discount_factor": 0.95
        }
        return SafetyPolicyLearner(config=config)
    
    def test_policy_initialization(self, policy_learner):
        """Test safety policy learner initialization."""
        assert policy_learner.policy_network is not None
        assert policy_learner.experience_buffer is not None
        assert policy_learner.reward_function is not None
        assert len(policy_learner.action_space) == len(SafetyAction)
    
    def test_state_representation(self, policy_learner):
        """Test state representation for policy learning."""
        # Test state encoding
        context = {
            "cpu_usage": 0.7,
            "memory_usage": 0.6,
            "response_time": 800,
            "user_load": 150,
            "time_of_day": "peak",
            "system_mode": "normal"
        }
        
        state_vector = policy_learner.encode_state(context)
        
        assert isinstance(state_vector, (np.ndarray, torch.Tensor))
        assert len(state_vector) > 0
        
        # Test state normalization
        normalized_state = policy_learner.normalize_state(state_vector)
        assert torch.all(normalized_state >= -1) and torch.all(normalized_state <= 1)
    
    def test_action_selection(self, policy_learner):
        """Test action selection from policy."""
        # Create test state
        state = torch.randn(policy_learner.state_dim)
        
        # Test greedy action selection
        action = policy_learner.select_action(state, exploration=False)
        assert action in [action.value for action in SafetyAction]
        
        # Test exploration action selection
        action_exploration = policy_learner.select_action(state, exploration=True)
        assert action_exploration in [action.value for action in SafetyAction]
    
    def test_reward_function_computation(self, policy_learner):
        """Test reward function computation."""
        # Test positive reward scenario (good safety and performance)
        positive_context = {
            "safety_score": 0.9,
            "performance_score": 0.8,
            "user_satisfaction": 0.85,
            "violation_occurred": False,
            "action_taken": SafetyAction.ALLOW
        }
        
        positive_reward = policy_learner.compute_reward(positive_context)
        assert positive_reward > 0
        
        # Test negative reward scenario (safety violation)
        negative_context = {
            "safety_score": 0.3,
            "performance_score": 0.4,
            "user_satisfaction": 0.2,
            "violation_occurred": True,
            "action_taken": SafetyAction.ALLOW  # Wrong action
        }
        
        negative_reward = policy_learner.compute_reward(negative_context)
        assert negative_reward < 0
        
        # Test neutral scenario
        neutral_context = {
            "safety_score": 0.7,
            "performance_score": 0.7,
            "user_satisfaction": 0.7,
            "violation_occurred": False,
            "action_taken": SafetyAction.WARN
        }
        
        neutral_reward = policy_learner.compute_reward(neutral_context)
        assert -0.1 <= neutral_reward <= 0.1  # Should be close to neutral
    
    @pytest.mark.asyncio
    async def test_policy_training(self, policy_learner):
        """Test policy training from experience."""
        # Generate training episodes
        episodes = []
        for episode in range(10):
            episode_data = []
            state = torch.randn(policy_learner.state_dim)
            
            for step in range(20):
                action = policy_learner.select_action(state, exploration=True)
                next_state = torch.randn(policy_learner.state_dim)
                reward = np.random.uniform(-1, 1)
                done = step == 19
                
                episode_data.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                })
                
                state = next_state
            
            episodes.append(episode_data)
        
        # Train policy
        training_result = await policy_learner.train_policy(episodes)
        
        assert training_result["success"]
        assert "loss" in training_result
        assert "policy_improvement" in training_result
        assert training_result["episodes_trained"] == 10
    
    def test_policy_evaluation(self, policy_learner):
        """Test policy evaluation and performance metrics."""
        # Create evaluation scenarios
        evaluation_scenarios = []
        for i in range(50):
            scenario = {
                "state": torch.randn(policy_learner.state_dim),
                "optimal_action": np.random.choice([action.value for action in SafetyAction]),
                "context": {
                    "cpu_usage": np.random.uniform(0.3, 0.9),
                    "memory_usage": np.random.uniform(0.4, 0.8),
                    "response_time": np.random.uniform(200, 1500)
                }
            }
            evaluation_scenarios.append(scenario)
        
        # Evaluate policy
        evaluation_result = policy_learner.evaluate_policy(evaluation_scenarios)
        
        assert "accuracy" in evaluation_result
        assert "precision" in evaluation_result
        assert "recall" in evaluation_result
        assert "f1_score" in evaluation_result
        
        assert 0.0 <= evaluation_result["accuracy"] <= 1.0
        assert 0.0 <= evaluation_result["f1_score"] <= 1.0