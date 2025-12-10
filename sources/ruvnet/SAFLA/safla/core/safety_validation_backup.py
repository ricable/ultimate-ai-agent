"""
Safety and Validation Framework for SAFLA.

This module implements comprehensive safety mechanisms for autonomous self-modification
including safety constraints, validation pipeline, risk assessment, rollback mechanisms,
and monitoring/alerts.

Based on research findings from research/04_synthesis/01_integrated_model.md
Priority 4 component providing:
1. Safety Constraints - Hard limits and soft boundaries
2. Validation Pipeline - Multi-stage validation
3. Risk Assessment - Quantitative risk scoring
4. Rollback Mechanisms - Safe reversion to previous states
5. Monitoring and Alerts - Real-time safety monitoring
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
import logging
from collections import defaultdict, deque


# Configure logging
logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of safety constraints."""
    HARD = "hard"      # Cannot be overridden, causes immediate action
    SOFT = "soft"      # Can be overridden with warnings


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ConstraintViolation:
    """Represents a safety constraint violation."""
    constraint_name: str
    severity: ConstraintType
    current_value: Any
    threshold: Any
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationResult:
    """Result of a validation stage."""
    stage: str
    is_valid: bool
    message: str
    details: Dict[str, Any]
    constraint_name: Optional[str] = None
    violation: Optional[ConstraintViolation] = None


@dataclass
class ValidationStage:
    """Defines a validation stage in the pipeline."""
    name: str
    description: str
    validator: Callable[[Dict[str, Any]], ValidationResult]
    required: bool = True
    timeout: Optional[float] = None


@dataclass
class RiskFactor:
    """Defines a risk factor for risk assessment."""
    name: str
    description: str
    weight: float
    calculator: Callable[[Dict[str, Any]], float]


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    overall_score: float
    factor_scores: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if risk is acceptable based on threshold."""
        return self.overall_score <= threshold


@dataclass
class SystemCheckpoint:
    """Represents a system checkpoint for rollback."""
    id: str
    name: str
    description: str
    system_state: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    compressed: bool = False


@dataclass
class SafetyAlert:
    """Represents a safety alert."""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: Any
    threshold: Any
    timestamp: float = field(default_factory=time.time)
    
    def __eq__(self, other):
        """Equality comparison for alerts."""
        if not isinstance(other, SafetyAlert):
            return False
        return (self.level == other.level and 
                self.message == other.message and
                self.metric_name == other.metric_name)


@dataclass
class SystemModificationResult:
    """Result of system modification validation."""
    is_approved: bool
    constraint_violations: List[ConstraintViolation]
    validation_failures: List[ValidationResult]
    risk_assessment: RiskAssessment
    checkpoint_id: Optional[str] = None
    message: str = ""


class SafetyConstraint:
    """Represents a safety constraint with validation logic."""
    
    def __init__(self, name: str, constraint_type: ConstraintType, 
                 description: str, rule: str, threshold: Any,
                 violation_action: str, enabled: bool = True):
        self.name = name
        self.constraint_type = constraint_type
        self.description = description
        self.rule = rule
        self.threshold = threshold
        self.violation_action = violation_action
        self.enabled = enabled
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against this constraint."""
        if not self.enabled:
            return ValidationResult(
                stage="constraint",
                is_valid=True,
                message=f"Constraint {self.name} is disabled",
                details={},
                constraint_name=self.name
            )
        
        # Simple rule evaluation - in production this would be more sophisticated
        try:
            # Extract the metric name from the rule (simple parsing)
            # For rules like "memory_usage <= 1000000000"
            if "<=" in self.rule:
                metric_name = self.rule.split("<=")[0].strip()
                current_value = data.get(metric_name, 0)
                is_valid = current_value <= self.threshold
            elif ">=" in self.rule:
                metric_name = self.rule.split(">=")[0].strip()
                current_value = data.get(metric_name, 0)
                is_valid = current_value >= self.threshold
            elif "<" in self.rule:
                metric_name = self.rule.split("<")[0].strip()
                current_value = data.get(metric_name, 0)
                is_valid = current_value < self.threshold
            elif ">" in self.rule:
                metric_name = self.rule.split(">")[0].strip()
                current_value = data.get(metric_name, 0)
                is_valid = current_value > self.threshold
            else:
                # Default to simple equality check
                metric_name = list(data.keys())[0] if data else "unknown"
                current_value = data.get(metric_name, 0)
                is_valid = current_value == self.threshold
            
            violation = None
            if not is_valid:
                violation = ConstraintViolation(
                    constraint_name=self.name,
                    severity=self.constraint_type,
                    current_value=current_value,
                    threshold=self.threshold,
                    message=f"Constraint {self.name} violated: {current_value} vs threshold {self.threshold}"
                )
            
            return ValidationResult(
                stage="constraint",
                is_valid=is_valid,
                message=f"Constraint {self.name}: {'passed' if is_valid else 'failed'}",
                details={"current_value": current_value, "threshold": self.threshold},
                constraint_name=self.name,
                violation=violation
            )
            
        except Exception as e:
            logger.error(f"Error validating constraint {self.name}: {e}")
            return ValidationResult(
                stage="constraint",
                is_valid=False,
                message=f"Error validating constraint {self.name}: {e}",
                details={"error": str(e)},
                constraint_name=self.name
            )


class SafetyConstraintEngine:
    """Engine for managing and evaluating safety constraints."""
    
    def __init__(self):
        self.constraints: Dict[str, SafetyConstraint] = {}
        self.violation_history: List[ConstraintViolation] = []
    
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint to the engine."""
        self.constraints[constraint.name] = constraint
        logger.info(f"Added safety constraint: {constraint.name}")
    
    def remove_constraint(self, name: str):
        """Remove a safety constraint from the engine."""
        if name in self.constraints:
            del self.constraints[name]
            logger.info(f"Removed safety constraint: {name}")
    
    def validate_all(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data against all constraints."""
        results = []
        for constraint in self.constraints.values():
            result = constraint.validate(data)
            results.append(result)
            
            # Track violations
            if result.violation:
                self.violation_history.append(result.violation)
        
        return results
    
    def get_violations(self, results: List[ValidationResult]) -> List[ConstraintViolation]:
        """Extract violations from validation results."""
        violations = []
        for result in results:
            if result.violation:
                violations.append(result.violation)
        return violations


class ValidationPipeline:
    """Multi-stage validation pipeline."""
    
    def __init__(self, stop_on_failure: bool = True):
        self.stages: List[ValidationStage] = []
        self.stop_on_failure = stop_on_failure
    
    def add_stage(self, stage: ValidationStage):
        """Add a validation stage to the pipeline."""
        self.stages.append(stage)
        logger.info(f"Added validation stage: {stage.name}")
    
    async def run(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Run the validation pipeline on the provided data."""
        results = []
        
        for stage in self.stages:
            try:
                # Run the validator
                if asyncio.iscoroutinefunction(stage.validator):
                    result = await stage.validator(data)
                else:
                    result = stage.validator(data)
                
                results.append(result)
                
                # Stop on failure if configured to do so
                if not result.is_valid and self.stop_on_failure:
                    logger.warning(f"Validation pipeline stopped at stage {stage.name} due to failure")
                    break
                    
            except Exception as e:
                logger.error(f"Error in validation stage {stage.name}: {e}")
                error_result = ValidationResult(
                    stage=stage.name,
                    is_valid=False,
                    message=f"Error in stage {stage.name}: {e}",
                    details={"error": str(e)}
                )
                results.append(error_result)
                
                if self.stop_on_failure:
                    break
        
        return results


class RiskScorer:
    """Risk assessment and scoring system."""
    
    def __init__(self):
        self.risk_factors: Dict[str, RiskFactor] = {}
        self.weights: Dict[str, float] = {}
    
    def add_risk_factor(self, factor: RiskFactor):
        """Add a risk factor to the scorer."""
        self.risk_factors[factor.name] = factor
        self.weights[factor.name] = factor.weight
        logger.info(f"Added risk factor: {factor.name} (weight: {factor.weight})")
    
    def calculate_risk(self, data: Dict[str, Any]) -> RiskAssessment:
        """Calculate overall risk score based on all factors."""
        factor_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, factor in self.risk_factors.items():
            try:
                score = factor.calculator(data)
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
                factor_scores[name] = score
                
                weighted_sum += score * factor.weight
                total_weight += factor.weight
                
            except Exception as e:
                logger.error(f"Error calculating risk factor {name}: {e}")
                factor_scores[name] = 1.0  # Assume maximum risk on error
                weighted_sum += 1.0 * factor.weight
                total_weight += factor.weight
        
        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return RiskAssessment(
            overall_score=overall_score,
            factor_scores=factor_scores
        )


class RollbackManager:
    """Manages system checkpoints and rollback operations."""
    
    def __init__(self, max_checkpoints: int = 10):
        self.checkpoints: Dict[str, SystemCheckpoint] = {}
        self.max_checkpoints = max_checkpoints
        self.checkpoint_order: deque = deque(maxlen=max_checkpoints)
    
    def create_checkpoint(self, name: str, description: str,
                         system_state: Dict[str, Any]) -> str:
        """Create a new system checkpoint."""
        checkpoint_id = str(uuid.uuid4())
        
        checkpoint = SystemCheckpoint(
            id=checkpoint_id,
            name=name,
            description=description,
            system_state=system_state.copy()
        )
        
        # Remove old checkpoints if we would exceed the limit
        if len(self.checkpoints) >= self.max_checkpoints:
            # Remove the oldest checkpoint
            while len(self.checkpoint_order) > 0 and len(self.checkpoints) >= self.max_checkpoints:
                oldest_id = self.checkpoint_order.popleft()
                if oldest_id in self.checkpoints:
                    del self.checkpoints[oldest_id]
        
        # Add to storage
        self.checkpoints[checkpoint_id] = checkpoint
        self.checkpoint_order.append(checkpoint_id)
        
        logger.info(f"Created checkpoint: {name} ({checkpoint_id})")
        return checkpoint_id
    
    def rollback_to_checkpoint(self, checkpoint_id: str, 
                              rollback_function: Callable[[Dict[str, Any]], bool]) -> bool:
        """Rollback system to a specific checkpoint."""
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        try:
            success = rollback_function(checkpoint.system_state)
            if success:
                logger.info(f"Successfully rolled back to checkpoint: {checkpoint.name}")
            else:
                logger.error(f"Failed to rollback to checkpoint: {checkpoint.name}")
            return success
            
        except Exception as e:
            logger.error(f"Error during rollback to checkpoint {checkpoint.name}: {e}")
            return False


class SafetyMonitor:
    """Real-time safety monitoring with alert system."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.metrics_history: deque = deque(maxlen=1000)
        self.safety_thresholds: Dict[str, Dict[str, Any]] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.get_system_metrics: Callable[[], Dict[str, Any]] = self._default_get_metrics
    
    def _default_get_metrics(self) -> Dict[str, Any]:
        """Default system metrics getter."""
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "timestamp": time.time()
        }
    
    def add_alert_handler(self, level: AlertLevel, handler: Callable[[SafetyAlert], None]):
        """Add an alert handler for a specific alert level."""
        self.alert_handlers[level].append(handler)
        logger.info(f"Added alert handler for level: {level}")
    
    async def start_monitoring(self):
        """Start the safety monitoring system."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Safety monitoring started")
    
    async def stop_monitoring(self):
        """Stop the safety monitoring system."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Safety monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get current metrics
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check safety thresholds
                alerts = self.check_safety_thresholds(metrics)
                
                # Trigger alerts
                for alert in alerts:
                    await self.trigger_alert(alert)
                
                # Wait for next iteration
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def check_safety_thresholds(self, metrics: Dict[str, Any]) -> List[SafetyAlert]:
        """Check metrics against safety thresholds."""
        alerts = []
        
        for metric_name, thresholds in self.safety_thresholds.items():
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            
            # Check critical threshold
            if "critical" in thresholds and current_value > thresholds["critical"]:
                alerts.append(SafetyAlert(
                    level=AlertLevel.CRITICAL,
                    message=f"{metric_name} exceeded critical threshold",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=thresholds["critical"]
                ))
            # Check warning threshold
            elif "warning" in thresholds and current_value > thresholds["warning"]:
                alerts.append(SafetyAlert(
                    level=AlertLevel.WARNING,
                    message=f"{metric_name} exceeded warning threshold",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=thresholds["warning"]
                ))
        
        return alerts
    
    async def trigger_alert(self, alert: SafetyAlert):
        """Trigger an alert and call appropriate handlers."""
        logger.warning(f"Safety alert: {alert.message}")
        
        # Call handlers for this alert level
        for handler in self.alert_handlers[alert.level]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")


class SafetyValidationFramework:
    """Integrated Safety and Validation Framework."""
    
    def __init__(self):
        self.constraint_engine = SafetyConstraintEngine()
        self.validation_pipeline = ValidationPipeline()
        self.risk_scorer = RiskScorer()
        self.rollback_manager = RollbackManager()
        self.safety_monitor = SafetyMonitor()
        self.is_active = False
        
        # Configurable functions for external integration
        self.get_system_state: Callable[[], Dict[str, Any]] = self._default_get_system_state
        self.rollback_function: Callable[[Dict[str, Any]], bool] = self._default_rollback_function
        self.emergency_stop_function: Callable[[], bool] = self._default_emergency_stop
    
    def _default_get_system_state(self) -> Dict[str, Any]:
        """Default system state getter."""
        return {
            "timestamp": time.time(),
            "status": "active"
        }
    
    def _default_rollback_function(self, state: Dict[str, Any]) -> bool:
        """Default rollback function."""
        logger.info(f"Default rollback to state: {state}")
        return True
    
    def _default_emergency_stop(self) -> bool:
        """Default emergency stop function."""
        logger.critical("Emergency stop triggered")
        self.is_active = False
        return True
    
    async def validate_system_modification(self, modification_data: Dict[str, Any]) -> SystemModificationResult:
        """Validate a proposed system modification."""
        constraint_violations = []
        validation_failures = []
        
        # 1. Check safety constraints
        constraint_results = self.constraint_engine.validate_all(modification_data)
        constraint_violations = self.constraint_engine.get_violations(constraint_results)
        
        # 2. Run validation pipeline
        validation_results = await self.validation_pipeline.run(modification_data)
        validation_failures = [r for r in validation_results if not r.is_valid]
        
        # 3. Calculate risk assessment
        risk_assessment = self.risk_scorer.calculate_risk(modification_data)
        
        # 4. Make approval decision
        is_approved = (
            len(constraint_violations) == 0 and
            len(validation_failures) == 0 and
            risk_assessment.is_acceptable()
        )
        
        return SystemModificationResult(
            is_approved=is_approved,
            constraint_violations=constraint_violations,
            validation_failures=validation_failures,
            risk_assessment=risk_assessment,
            message="Validation complete"
        )
    
    async def create_safety_checkpoint(self, name: str, description: str) -> str:
        """Create a safety checkpoint."""
        system_state = self.get_system_state()
        checkpoint_id = self.rollback_manager.create_checkpoint(name, description, system_state)
        return checkpoint_id
    
    async def rollback_to_safe_state(self, checkpoint_id: str) -> bool:
        """Rollback to a safe state."""
        return self.rollback_manager.rollback_to_checkpoint(checkpoint_id, self.rollback_function)
    
    async def emergency_stop(self, reason: str) -> bool:
        """Trigger emergency stop."""
        logger.critical(f"Emergency stop triggered: {reason}")
        self.is_active = False
        
        # Stop monitoring
        await self.safety_monitor.stop_monitoring()
        
        # Call emergency stop function
        return self.emergency_stop_function()
    
    async def start(self):
        """Start the safety framework."""
        self.is_active = True
        await self.safety_monitor.start_monitoring()
        logger.info("Safety and Validation Framework started")
    
    async def stop(self):
        """Stop the safety framework."""
        self.is_active = False
        await self.safety_monitor.stop_monitoring()
        logger.info("Safety and Validation Framework stopped")