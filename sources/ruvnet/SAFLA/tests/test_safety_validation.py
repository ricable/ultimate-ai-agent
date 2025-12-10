"""
Test suite for Safety and Validation Framework.

This module tests the comprehensive safety mechanisms for autonomous self-modification
including safety constraints, validation pipeline, risk assessment, rollback mechanisms,
and monitoring/alerts.

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Import the modules we'll implement
from safla.core.safety_validation import (
    SafetyConstraint,
    ConstraintType,
    ConstraintViolation,
    SafetyConstraintEngine,
    ValidationStage,
    ValidationResult,
    ValidationPipeline,
    RiskFactor,
    RiskAssessment,
    RiskScorer,
    SystemCheckpoint,
    RollbackManager,
    SafetyMonitor,
    AlertLevel,
    SafetyAlert,
    SafetyValidationFramework
)


class TestSafetyConstraint:
    """Test suite for SafetyConstraint class."""
    
    def test_safety_constraint_creation(self):
        """Test creating a safety constraint with all required fields."""
        constraint = SafetyConstraint(
            name="max_memory_usage",
            constraint_type=ConstraintType.HARD,
            description="Maximum memory usage limit",
            rule="memory_usage <= 1000000000",  # 1GB
            threshold=1000000000,
            violation_action="emergency_stop"
        )
        
        assert constraint.name == "max_memory_usage"
        assert constraint.constraint_type == ConstraintType.HARD
        assert constraint.description == "Maximum memory usage limit"
        assert constraint.rule == "memory_usage <= 1000000000"
        assert constraint.threshold == 1000000000
        assert constraint.violation_action == "emergency_stop"
        assert constraint.enabled is True  # Default value
    
    def test_safety_constraint_validation_success(self):
        """Test constraint validation with valid data."""
        constraint = SafetyConstraint(
            name="cpu_usage",
            constraint_type=ConstraintType.SOFT,
            description="CPU usage warning",
            rule="cpu_usage <= 80",
            threshold=80,
            violation_action="warn"
        )
        
        # Valid data should pass
        result = constraint.validate({"cpu_usage": 75})
        assert result.is_valid is True
        assert result.violation is None
    
    def test_safety_constraint_validation_failure(self):
        """Test constraint validation with invalid data."""
        constraint = SafetyConstraint(
            name="cpu_usage",
            constraint_type=ConstraintType.HARD,
            description="CPU usage limit",
            rule="cpu_usage <= 80",
            threshold=80,
            violation_action="emergency_stop"
        )
        
        # Invalid data should fail
        result = constraint.validate({"cpu_usage": 95})
        assert result.is_valid is False
        assert result.violation is not None
        assert result.violation.constraint_name == "cpu_usage"
        assert result.violation.severity == ConstraintType.HARD
    
    def test_safety_constraint_disabled(self):
        """Test that disabled constraints always pass validation."""
        constraint = SafetyConstraint(
            name="disabled_constraint",
            constraint_type=ConstraintType.HARD,
            description="Disabled constraint",
            rule="value <= 10",
            threshold=10,
            violation_action="stop",
            enabled=False
        )
        
        # Should pass even with invalid data when disabled
        result = constraint.validate({"value": 100})
        assert result.is_valid is True
        assert result.violation is None


class TestSafetyConstraintEngine:
    """Test suite for SafetyConstraintEngine class."""
    
    def test_constraint_engine_initialization(self):
        """Test constraint engine initialization."""
        engine = SafetyConstraintEngine()
        assert len(engine.constraints) == 0
        assert engine.violation_history == []
    
    def test_add_constraint(self):
        """Test adding constraints to the engine."""
        engine = SafetyConstraintEngine()
        
        constraint = SafetyConstraint(
            name="test_constraint",
            constraint_type=ConstraintType.HARD,
            description="Test constraint",
            rule="value <= 100",
            threshold=100,
            violation_action="stop"
        )
        
        engine.add_constraint(constraint)
        assert len(engine.constraints) == 1
        assert "test_constraint" in engine.constraints
    
    def test_remove_constraint(self):
        """Test removing constraints from the engine."""
        engine = SafetyConstraintEngine()
        
        constraint = SafetyConstraint(
            name="test_constraint",
            constraint_type=ConstraintType.HARD,
            description="Test constraint",
            rule="value <= 100",
            threshold=100,
            violation_action="stop"
        )
        
        engine.add_constraint(constraint)
        assert len(engine.constraints) == 1
        
        engine.remove_constraint("test_constraint")
        assert len(engine.constraints) == 0
        assert "test_constraint" not in engine.constraints
    
    def test_validate_all_constraints_success(self):
        """Test validating all constraints with valid data."""
        engine = SafetyConstraintEngine()
        
        # Add multiple constraints
        engine.add_constraint(SafetyConstraint(
            name="memory_limit",
            constraint_type=ConstraintType.HARD,
            description="Memory limit",
            rule="memory <= 1000",
            threshold=1000,
            violation_action="stop"
        ))
        
        engine.add_constraint(SafetyConstraint(
            name="cpu_limit",
            constraint_type=ConstraintType.SOFT,
            description="CPU limit",
            rule="cpu <= 80",
            threshold=80,
            violation_action="warn"
        ))
        
        # Valid data should pass all constraints
        results = engine.validate_all({"memory": 500, "cpu": 60})
        assert len(results) == 2
        assert all(result.is_valid for result in results)
    
    def test_validate_all_constraints_failure(self):
        """Test validating all constraints with some failures."""
        engine = SafetyConstraintEngine()
        
        # Add constraints
        engine.add_constraint(SafetyConstraint(
            name="memory_limit",
            constraint_type=ConstraintType.HARD,
            description="Memory limit",
            rule="memory <= 1000",
            threshold=1000,
            violation_action="stop"
        ))
        
        engine.add_constraint(SafetyConstraint(
            name="cpu_limit",
            constraint_type=ConstraintType.SOFT,
            description="CPU limit",
            rule="cpu <= 80",
            threshold=80,
            violation_action="warn"
        ))
        
        # Data that violates one constraint
        results = engine.validate_all({"memory": 1500, "cpu": 60})
        assert len(results) == 2
        
        # Memory constraint should fail, CPU should pass
        memory_result = next(r for r in results if r.constraint_name == "memory_limit")
        cpu_result = next(r for r in results if r.constraint_name == "cpu_limit")
        
        assert memory_result.is_valid is False
        assert cpu_result.is_valid is True
    
    def test_get_violations(self):
        """Test getting current violations from validation results."""
        engine = SafetyConstraintEngine()
        
        engine.add_constraint(SafetyConstraint(
            name="test_constraint",
            constraint_type=ConstraintType.HARD,
            description="Test constraint",
            rule="value <= 100",
            threshold=100,
            violation_action="stop"
        ))
        
        # Validate with violating data
        results = engine.validate_all({"value": 150})
        violations = engine.get_violations(results)
        
        assert len(violations) == 1
        assert violations[0].constraint_name == "test_constraint"
        assert violations[0].severity == ConstraintType.HARD


class TestValidationPipeline:
    """Test suite for ValidationPipeline class."""
    
    def test_validation_pipeline_initialization(self):
        """Test validation pipeline initialization."""
        pipeline = ValidationPipeline()
        assert len(pipeline.stages) == 0
        assert pipeline.stop_on_failure is True  # Default behavior
    
    def test_add_validation_stage(self):
        """Test adding validation stages to the pipeline."""
        pipeline = ValidationPipeline()
        
        # Mock validator function
        def syntax_validator(data):
            return ValidationResult(
                stage="syntax",
                is_valid=True,
                message="Syntax is valid",
                details={}
            )
        
        stage = ValidationStage(
            name="syntax",
            description="Syntax validation",
            validator=syntax_validator,
            required=True
        )
        
        pipeline.add_stage(stage)
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "syntax"
    
    @pytest.mark.asyncio
    async def test_run_pipeline_success(self):
        """Test running validation pipeline with all stages passing."""
        pipeline = ValidationPipeline()
        
        # Add multiple validation stages
        def syntax_validator(data):
            return ValidationResult(
                stage="syntax",
                is_valid=True,
                message="Syntax is valid",
                details={}
            )
        
        def semantic_validator(data):
            return ValidationResult(
                stage="semantic",
                is_valid=True,
                message="Semantics are valid",
                details={}
            )
        
        pipeline.add_stage(ValidationStage(
            name="syntax",
            description="Syntax validation",
            validator=syntax_validator,
            required=True
        ))
        
        pipeline.add_stage(ValidationStage(
            name="semantic",
            description="Semantic validation",
            validator=semantic_validator,
            required=True
        ))
        
        # Run pipeline
        results = await pipeline.run({"code": "print('hello')"})
        
        assert len(results) == 2
        assert all(result.is_valid for result in results)
        assert results[0].stage == "syntax"
        assert results[1].stage == "semantic"
    
    @pytest.mark.asyncio
    async def test_run_pipeline_failure_stop_on_failure(self):
        """Test pipeline stopping on first failure when stop_on_failure=True."""
        pipeline = ValidationPipeline(stop_on_failure=True)
        
        def failing_validator(data):
            return ValidationResult(
                stage="failing",
                is_valid=False,
                message="Validation failed",
                details={"error": "Invalid input"}
            )
        
        def passing_validator(data):
            return ValidationResult(
                stage="passing",
                is_valid=True,
                message="Validation passed",
                details={}
            )
        
        pipeline.add_stage(ValidationStage(
            name="failing",
            description="Failing validation",
            validator=failing_validator,
            required=True
        ))
        
        pipeline.add_stage(ValidationStage(
            name="passing",
            description="Passing validation",
            validator=passing_validator,
            required=True
        ))
        
        # Run pipeline - should stop after first failure
        results = await pipeline.run({"data": "test"})
        
        assert len(results) == 1  # Only first stage should run
        assert results[0].is_valid is False
        assert results[0].stage == "failing"
    
    @pytest.mark.asyncio
    async def test_run_pipeline_failure_continue_on_failure(self):
        """Test pipeline continuing on failure when stop_on_failure=False."""
        pipeline = ValidationPipeline(stop_on_failure=False)
        
        def failing_validator(data):
            return ValidationResult(
                stage="failing",
                is_valid=False,
                message="Validation failed",
                details={"error": "Invalid input"}
            )
        
        def passing_validator(data):
            return ValidationResult(
                stage="passing",
                is_valid=True,
                message="Validation passed",
                details={}
            )
        
        pipeline.add_stage(ValidationStage(
            name="failing",
            description="Failing validation",
            validator=failing_validator,
            required=True
        ))
        
        pipeline.add_stage(ValidationStage(
            name="passing",
            description="Passing validation",
            validator=passing_validator,
            required=True
        ))
        
        # Run pipeline - should continue after failure
        results = await pipeline.run({"data": "test"})
        
        assert len(results) == 2  # Both stages should run
        assert results[0].is_valid is False
        assert results[1].is_valid is True


class TestRiskScorer:
    """Test suite for RiskScorer class."""
    
    def test_risk_scorer_initialization(self):
        """Test risk scorer initialization."""
        scorer = RiskScorer()
        assert len(scorer.risk_factors) == 0
        assert scorer.weights == {}
    
    def test_add_risk_factor(self):
        """Test adding risk factors to the scorer."""
        scorer = RiskScorer()
        
        factor = RiskFactor(
            name="performance_impact",
            description="Impact on system performance",
            weight=0.3,
            calculator=lambda data: min(data.get("cpu_increase", 0) / 100, 1.0)
        )
        
        scorer.add_risk_factor(factor)
        assert len(scorer.risk_factors) == 1
        assert "performance_impact" in scorer.risk_factors
        assert scorer.weights["performance_impact"] == 0.3
    
    def test_calculate_risk_score(self):
        """Test calculating overall risk score."""
        scorer = RiskScorer()
        
        # Add multiple risk factors
        scorer.add_risk_factor(RiskFactor(
            name="performance_impact",
            description="Performance impact",
            weight=0.4,
            calculator=lambda data: data.get("cpu_increase", 0) / 100
        ))
        
        scorer.add_risk_factor(RiskFactor(
            name="stability_risk",
            description="Stability risk",
            weight=0.6,
            calculator=lambda data: data.get("error_rate", 0)
        ))
        
        # Calculate risk score
        assessment = scorer.calculate_risk({
            "cpu_increase": 20,  # 20% CPU increase
            "error_rate": 0.1    # 10% error rate
        })
        
        # Expected: (0.2 * 0.4) + (0.1 * 0.6) = 0.08 + 0.06 = 0.14
        expected_score = 0.14
        assert abs(assessment.overall_score - expected_score) < 0.001
        assert len(assessment.factor_scores) == 2
        assert "performance_impact" in assessment.factor_scores
        assert "stability_risk" in assessment.factor_scores
    
    def test_risk_assessment_threshold_checking(self):
        """Test risk assessment threshold checking."""
        scorer = RiskScorer()
        
        scorer.add_risk_factor(RiskFactor(
            name="test_factor",
            description="Test factor",
            weight=1.0,
            calculator=lambda data: data.get("risk_value", 0)
        ))
        
        # Low risk
        low_risk = scorer.calculate_risk({"risk_value": 0.2})
        assert low_risk.is_acceptable(threshold=0.5) is True
        
        # High risk
        high_risk = scorer.calculate_risk({"risk_value": 0.8})
        assert high_risk.is_acceptable(threshold=0.5) is False


class TestRollbackManager:
    """Test suite for RollbackManager class."""
    
    def test_rollback_manager_initialization(self):
        """Test rollback manager initialization."""
        manager = RollbackManager()
        assert len(manager.checkpoints) == 0
        assert manager.max_checkpoints == 10  # Default value
    
    def test_create_checkpoint(self):
        """Test creating system checkpoints."""
        manager = RollbackManager()
        
        system_state = {
            "memory_usage": 500000000,
            "active_processes": ["process1", "process2"],
            "configuration": {"setting1": "value1"}
        }
        
        checkpoint_id = manager.create_checkpoint(
            name="test_checkpoint",
            description="Test checkpoint for unit tests",
            system_state=system_state
        )
        
        assert checkpoint_id is not None
        assert len(manager.checkpoints) == 1
        assert checkpoint_id in manager.checkpoints
        
        checkpoint = manager.checkpoints[checkpoint_id]
        assert checkpoint.name == "test_checkpoint"
        assert checkpoint.description == "Test checkpoint for unit tests"
        assert checkpoint.system_state == system_state
    
    def test_rollback_to_checkpoint(self):
        """Test rolling back to a previous checkpoint."""
        manager = RollbackManager()
        
        # Create a checkpoint
        system_state = {"value": 100}
        checkpoint_id = manager.create_checkpoint(
            name="stable_state",
            description="Stable system state",
            system_state=system_state
        )
        
        # Mock the rollback function
        rollback_called = False
        restored_state = None
        
        def mock_rollback_function(state):
            nonlocal rollback_called, restored_state
            rollback_called = True
            restored_state = state
            return True
        
        # Perform rollback
        success = manager.rollback_to_checkpoint(checkpoint_id, mock_rollback_function)
        
        assert success is True
        assert rollback_called is True
        assert restored_state == system_state
    
    def test_rollback_to_nonexistent_checkpoint(self):
        """Test rollback to non-existent checkpoint fails gracefully."""
        manager = RollbackManager()
        
        def mock_rollback_function(state):
            return True
        
        success = manager.rollback_to_checkpoint("nonexistent", mock_rollback_function)
        assert success is False
    
    def test_checkpoint_limit_enforcement(self):
        """Test that checkpoint limit is enforced."""
        manager = RollbackManager(max_checkpoints=2)
        
        # Create checkpoints beyond the limit
        for i in range(3):
            manager.create_checkpoint(
                name=f"checkpoint_{i}",
                description=f"Checkpoint {i}",
                system_state={"value": i}
            )
        
        # Should only keep the most recent 2 checkpoints
        assert len(manager.checkpoints) == 2
        
        # Should have checkpoints 1 and 2, not 0
        checkpoint_names = [cp.name for cp in manager.checkpoints.values()]
        assert "checkpoint_1" in checkpoint_names
        assert "checkpoint_2" in checkpoint_names
        assert "checkpoint_0" not in checkpoint_names


class TestSafetyMonitor:
    """Test suite for SafetyMonitor class."""
    
    def test_safety_monitor_initialization(self):
        """Test safety monitor initialization."""
        monitor = SafetyMonitor()
        assert monitor.is_monitoring is False
        assert len(monitor.alert_handlers) == 0
        assert len(monitor.metrics_history) == 0
    
    def test_add_alert_handler(self):
        """Test adding alert handlers."""
        monitor = SafetyMonitor()
        
        def test_handler(alert):
            pass
        
        monitor.add_alert_handler(AlertLevel.CRITICAL, test_handler)
        assert AlertLevel.CRITICAL in monitor.alert_handlers
        assert test_handler in monitor.alert_handlers[AlertLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self):
        """Test starting the monitoring system."""
        monitor = SafetyMonitor(monitoring_interval=0.1)  # Fast interval for testing
        
        # Mock system metrics function
        call_count = 0
        def mock_get_metrics():
            nonlocal call_count
            call_count += 1
            return {
                "cpu_usage": 50 + call_count * 10,  # Increasing CPU usage
                "memory_usage": 1000000000,
                "timestamp": time.time()
            }
        
        monitor.get_system_metrics = mock_get_metrics
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Let it run for a short time
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        assert call_count > 0
        assert len(monitor.metrics_history) > 0
    
    def test_check_safety_thresholds(self):
        """Test safety threshold checking."""
        monitor = SafetyMonitor()
        
        # Configure thresholds
        monitor.safety_thresholds = {
            "cpu_usage": {"warning": 70, "critical": 90},
            "memory_usage": {"warning": 800000000, "critical": 1000000000}
        }
        
        # Test normal metrics (no alerts)
        alerts = monitor.check_safety_thresholds({
            "cpu_usage": 50,
            "memory_usage": 500000000
        })
        assert len(alerts) == 0
        
        # Test warning level metrics
        alerts = monitor.check_safety_thresholds({
            "cpu_usage": 75,
            "memory_usage": 850000000
        })
        assert len(alerts) == 2
        assert all(alert.level == AlertLevel.WARNING for alert in alerts)
        
        # Test critical level metrics
        alerts = monitor.check_safety_thresholds({
            "cpu_usage": 95,
            "memory_usage": 1100000000
        })
        assert len(alerts) == 2
        assert all(alert.level == AlertLevel.CRITICAL for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_trigger_alert(self):
        """Test alert triggering and handling."""
        monitor = SafetyMonitor()
        
        # Track alert handling
        handled_alerts = []
        
        def test_handler(alert):
            handled_alerts.append(alert)
        
        monitor.add_alert_handler(AlertLevel.CRITICAL, test_handler)
        
        # Create and trigger an alert
        alert = SafetyAlert(
            level=AlertLevel.CRITICAL,
            message="Test critical alert",
            metric_name="test_metric",
            current_value=100,
            threshold=50,
            timestamp=time.time()
        )
        
        await monitor.trigger_alert(alert)
        
        assert len(handled_alerts) == 1
        assert handled_alerts[0] == alert


class TestSafetyValidationFramework:
    """Test suite for the integrated SafetyValidationFramework."""
    
    def test_framework_initialization(self):
        """Test framework initialization with all components."""
        framework = SafetyValidationFramework()
        
        assert framework.constraint_engine is not None
        assert framework.validation_pipeline is not None
        assert framework.risk_scorer is not None
        assert framework.rollback_manager is not None
        assert framework.safety_monitor is not None
        assert framework.is_active is False
    
    @pytest.mark.asyncio
    async def test_validate_system_modification_success(self):
        """Test successful system modification validation."""
        framework = SafetyValidationFramework()
        
        # Configure framework with permissive settings
        framework.constraint_engine.add_constraint(SafetyConstraint(
            name="memory_limit",
            constraint_type=ConstraintType.HARD,
            description="Memory limit",
            rule="memory_usage <= 2000000000",  # 2GB limit
            threshold=2000000000,
            violation_action="reject"
        ))
        
        # Add simple validation stage
        def simple_validator(data):
            return ValidationResult(
                stage="simple",
                is_valid=True,
                message="Validation passed",
                details={}
            )
        
        framework.validation_pipeline.add_stage(ValidationStage(
            name="simple",
            description="Simple validation",
            validator=simple_validator,
            required=True
        ))
        
        # Add low-risk factor
        framework.risk_scorer.add_risk_factor(RiskFactor(
            name="low_risk",
            description="Low risk factor",
            weight=1.0,
            calculator=lambda data: 0.1  # Always low risk
        ))
        
        # Test modification data
        modification_data = {
            "memory_usage": 1000000000,  # 1GB - within limit
            "change_type": "optimization",
            "description": "Performance optimization"
        }
        
        # Validate modification
        result = await framework.validate_system_modification(modification_data)
        
        assert result.is_approved is True
        assert len(result.constraint_violations) == 0
        assert len(result.validation_failures) == 0
        assert result.risk_assessment.overall_score <= 0.5  # Low risk threshold
    
    @pytest.mark.asyncio
    async def test_validate_system_modification_constraint_violation(self):
        """Test system modification validation with constraint violation."""
        framework = SafetyValidationFramework()
        
        # Configure strict constraint
        framework.constraint_engine.add_constraint(SafetyConstraint(
            name="memory_limit",
            constraint_type=ConstraintType.HARD,
            description="Strict memory limit",
            rule="memory_usage <= 500000000",  # 500MB limit
            threshold=500000000,
            violation_action="reject"
        ))
        
        # Test modification data that violates constraint
        modification_data = {
            "memory_usage": 1000000000,  # 1GB - exceeds limit
            "change_type": "expansion",
            "description": "Memory expansion"
        }
        
        # Validate modification
        result = await framework.validate_system_modification(modification_data)
        
        assert result.is_approved is False
        assert len(result.constraint_violations) > 0
        assert result.constraint_violations[0].constraint_name == "memory_limit"
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Test emergency stop functionality."""
        framework = SafetyValidationFramework()
        
        # Mock emergency stop function
        emergency_stop_called = False
        
        def mock_emergency_stop():
            nonlocal emergency_stop_called
            emergency_stop_called = True
            return True
        
        framework.emergency_stop_function = mock_emergency_stop
        
        # Trigger emergency stop
        success = await framework.emergency_stop("Test emergency stop")
        
        assert success is True
        assert emergency_stop_called is True
        assert framework.is_active is False
    
    @pytest.mark.asyncio
    async def test_create_safety_checkpoint(self):
        """Test creating safety checkpoints."""
        framework = SafetyValidationFramework()
        
        # Mock system state getter
        def mock_get_system_state():
            return {
                "memory_usage": 500000000,
                "cpu_usage": 50,
                "active_processes": ["process1", "process2"]
            }
        
        framework.get_system_state = mock_get_system_state
        
        # Create checkpoint
        checkpoint_id = await framework.create_safety_checkpoint(
            "test_checkpoint",
            "Test checkpoint for safety"
        )
        
        assert checkpoint_id is not None
        assert len(framework.rollback_manager.checkpoints) == 1
    
    @pytest.mark.asyncio
    async def test_rollback_to_safe_state(self):
        """Test rolling back to a safe state."""
        framework = SafetyValidationFramework()
        
        # Create a checkpoint first
        def mock_get_system_state():
            return {"value": 100}
        
        framework.get_system_state = mock_get_system_state
        
        checkpoint_id = await framework.create_safety_checkpoint(
            "safe_state",
            "Safe system state"
        )
        
        # Mock rollback function
        rollback_called = False
        
        def mock_rollback_function(state):
            nonlocal rollback_called
            rollback_called = True
            return True
        
        framework.rollback_function = mock_rollback_function
        
        # Perform rollback
        success = await framework.rollback_to_safe_state(checkpoint_id)
        
        assert success is True
        assert rollback_called is True


# Integration tests for the complete framework
class TestSafetyValidationIntegration:
    """Integration tests for the complete Safety and Validation Framework."""
    
    @pytest.mark.asyncio
    async def test_complete_safety_workflow(self):
        """Test the complete safety workflow from monitoring to rollback."""
        framework = SafetyValidationFramework()
        
        # Configure the framework
        framework.constraint_engine.add_constraint(SafetyConstraint(
            name="cpu_limit",
            constraint_type=ConstraintType.HARD,
            description="CPU usage limit",
            rule="cpu_usage <= 80",
            threshold=80,
            violation_action="rollback"
        ))
        
        # Add validation stage
        def performance_validator(data):
            cpu_usage = data.get("cpu_usage", 0)
            return ValidationResult(
                stage="performance",
                is_valid=cpu_usage <= 90,
                message=f"CPU usage: {cpu_usage}%",
                details={"cpu_usage": cpu_usage}
            )
        
        framework.validation_pipeline.add_stage(ValidationStage(
            name="performance",
            description="Performance validation",
            validator=performance_validator,
            required=True
        ))
        
        # Add risk factor
        framework.risk_scorer.add_risk_factor(RiskFactor(
            name="cpu_risk",
            description="CPU usage risk",
            weight=1.0,
            calculator=lambda data: data.get("cpu_usage", 0) / 100
        ))
        
        # Create initial checkpoint
        def mock_get_system_state():
            return {"cpu_usage": 30, "status": "stable"}
        
        framework.get_system_state = mock_get_system_state
        
        checkpoint_id = await framework.create_safety_checkpoint(
            "initial_state",
            "Initial stable state"
        )
        
        # Test 1: Valid modification should be approved
        valid_modification = {
            "cpu_usage": 60,
            "change_type": "optimization",
            "description": "Performance optimization"
        }
        
        result = await framework.validate_system_modification(valid_modification)
        assert result.is_approved is True
        
        # Test 2: Invalid modification should be rejected
        invalid_modification = {
            "cpu_usage": 95,
            "change_type": "expansion",
            "description": "Resource-intensive expansion"
        }
        
        result = await framework.validate_system_modification(invalid_modification)
        assert result.is_approved is False
        assert len(result.constraint_violations) > 0
        
        # Test 3: Rollback should work
        rollback_called = False
        
        def mock_rollback_function(state):
            nonlocal rollback_called
            rollback_called = True
            return True
        
        framework.rollback_function = mock_rollback_function
        
        success = await framework.rollback_to_safe_state(checkpoint_id)
        assert success is True
        assert rollback_called is True


if __name__ == "__main__":
    pytest.main([__file__])