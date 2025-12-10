"""
Agent 40: Self-Improving AI Metacognition System - Safety Constraints
Implements comprehensive safety bounds and constraints for self-improvement.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from uuid import uuid4
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for different operations"""
    UNRESTRICTED = "unrestricted"  # No safety restrictions
    MONITORING = "monitoring"  # Monitor but don't restrict
    GUIDED = "guided"  # Provide guidance and warnings
    RESTRICTED = "restricted"  # Enforce strict boundaries
    LOCKED = "locked"  # No modifications allowed


class ViolationSeverity(Enum):
    """Severity levels for safety violations"""
    INFO = "info"  # Informational only
    WARNING = "warning"  # Potential issue
    MINOR = "minor"  # Minor violation
    MAJOR = "major"  # Significant violation
    CRITICAL = "critical"  # Critical safety violation
    CATASTROPHIC = "catastrophic"  # System-threatening violation


class ConstraintScope(Enum):
    """Scope of safety constraints"""
    GLOBAL = "global"  # System-wide constraints
    MODULE = "module"  # Module-specific constraints
    OPERATION = "operation"  # Operation-specific constraints
    TEMPORAL = "temporal"  # Time-based constraints
    CONTEXTUAL = "contextual"  # Context-dependent constraints


@dataclass
class SafetyBoundary:
    """Represents a safety boundary"""
    boundary_id: str
    name: str
    description: str
    boundary_type: str  # "hard", "soft", "adaptive"
    min_value: Optional[float]
    max_value: Optional[float]
    target_value: Optional[float]
    tolerance: float
    enforcement_level: SafetyLevel
    violation_consequences: List[str]
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]


@dataclass
class SafetyRule:
    """Represents a safety rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # "invariant", "precondition", "postcondition", "temporal"
    condition: str  # Logical condition expression
    scope: ConstraintScope
    priority: int  # 1-10, 10 being highest
    enforcement_level: SafetyLevel
    violation_action: str  # "log", "warn", "block", "rollback", "shutdown"
    exceptions: List[str]
    created_at: datetime
    last_checked: datetime
    violation_count: int
    metadata: Dict[str, Any]


@dataclass
class SafetyViolation:
    """Represents a detected safety violation"""
    violation_id: str
    timestamp: datetime
    rule_id: Optional[str]
    boundary_id: Optional[str]
    severity: ViolationSeverity
    description: str
    context: Dict[str, Any]
    detected_values: Dict[str, Any]
    action_taken: str
    resolution_status: str  # "pending", "resolved", "acknowledged", "ignored"
    resolution_time: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class SafetyMonitoringState:
    """Represents the current state of safety monitoring"""
    monitoring_active: bool
    safety_level: SafetyLevel
    total_boundaries: int
    total_rules: int
    active_violations: int
    recent_violations: int
    last_check_time: datetime
    system_risk_score: float
    monitoring_overhead: float
    metadata: Dict[str, Any]


class SafetyRuleEngine:
    """Engine for evaluating safety rules and conditions"""
    
    def __init__(self):
        self.rules: Dict[str, SafetyRule] = {}
        self.rule_evaluation_cache: Dict[str, Dict] = {}
        self.evaluation_context: Dict[str, Any] = {}
        self.operator_map = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
            'not': lambda a: not a,
            'in': lambda a, b: a in b,
            'between': lambda a, b, c: b <= a <= c
        }
    
    async def add_rule(self, name: str, description: str, rule_type: str,
                      condition: str, scope: ConstraintScope,
                      priority: int = 5, enforcement_level: SafetyLevel = SafetyLevel.RESTRICTED,
                      violation_action: str = "warn") -> str:
        """Add a new safety rule"""
        rule = SafetyRule(
            rule_id=str(uuid4()),
            name=name,
            description=description,
            rule_type=rule_type,
            condition=condition,
            scope=scope,
            priority=priority,
            enforcement_level=enforcement_level,
            violation_action=violation_action,
            exceptions=[],
            created_at=datetime.utcnow(),
            last_checked=datetime.utcnow(),
            violation_count=0,
            metadata={}
        )
        
        self.rules[rule.rule_id] = rule
        logger.info(f"Added safety rule: {name} ({rule.rule_id})")
        return rule.rule_id
    
    async def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific safety rule"""
        if rule_id not in self.rules:
            return {'valid': False, 'error': f'Rule {rule_id} not found'}
        
        rule = self.rules[rule_id]
        
        try:
            # Update evaluation context
            self.evaluation_context.update(context)
            
            # Evaluate the rule condition
            evaluation_result = await self._evaluate_condition(rule.condition, context)
            
            rule.last_checked = datetime.utcnow()
            
            result = {
                'rule_id': rule_id,
                'rule_name': rule.name,
                'satisfied': evaluation_result['satisfied'],
                'violation_detected': not evaluation_result['satisfied'],
                'severity': self._calculate_violation_severity(rule, evaluation_result),
                'context_values': evaluation_result.get('context_values', {}),
                'evaluation_details': evaluation_result,
                'enforcement_level': rule.enforcement_level.value,
                'violation_action': rule.violation_action
            }
            
            if not evaluation_result['satisfied']:
                rule.violation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule_id}: {e}")
            return {
                'rule_id': rule_id,
                'valid': False,
                'error': str(e),
                'satisfied': False,
                'violation_detected': True,
                'severity': ViolationSeverity.WARNING.value
            }
    
    async def evaluate_all_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all safety rules"""
        results = []
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if rule.enforcement_level != SafetyLevel.UNRESTRICTED:
                result = await self.evaluate_rule(rule.rule_id, context)
                results.append(result)
        
        return results
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a condition string"""
        try:
            # Simple condition evaluation (in practice, this would be more sophisticated)
            # For safety, we'll implement a restricted evaluator
            
            # Replace context variables in condition
            evaluated_condition = condition
            context_values = {}
            
            for key, value in context.items():
                if key in condition:
                    context_values[key] = value
                    # Simple string replacement (in practice, use proper parsing)
                    evaluated_condition = evaluated_condition.replace(f"{key}", str(value))
            
            # Simple evaluation for common patterns
            satisfied = await self._safe_evaluate_expression(evaluated_condition, context)
            
            return {
                'satisfied': satisfied,
                'original_condition': condition,
                'evaluated_condition': evaluated_condition,
                'context_values': context_values
            }
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return {
                'satisfied': False,
                'error': str(e),
                'original_condition': condition
            }
    
    async def _safe_evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a simple expression"""
        # This is a simplified evaluator for demonstration
        # In practice, you'd want a more robust and secure expression evaluator
        
        try:
            # Handle simple numeric comparisons
            if '>' in expression:
                parts = expression.split('>')
                if len(parts) == 2:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    return left > right
            
            elif '<' in expression:
                parts = expression.split('<')
                if len(parts) == 2:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    return left < right
            
            elif '==' in expression:
                parts = expression.split('==')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    # Try numeric comparison first
                    try:
                        return float(left) == float(right)
                    except ValueError:
                        return left == right
            
            elif 'between' in expression.lower():
                # Handle "value between min and max" pattern
                parts = expression.lower().split('between')
                if len(parts) == 2:
                    value = float(parts[0].strip())
                    range_part = parts[1].strip().split('and')
                    if len(range_part) == 2:
                        min_val = float(range_part[0].strip())
                        max_val = float(range_part[1].strip())
                        return min_val <= value <= max_val
            
            # Default to True for unknown expressions (conservative)
            return True
            
        except Exception:
            # If evaluation fails, assume violation for safety
            return False
    
    def _calculate_violation_severity(self, rule: SafetyRule, evaluation_result: Dict[str, Any]) -> str:
        """Calculate the severity of a rule violation"""
        # Base severity on rule priority and enforcement level
        if rule.priority >= 9:
            return ViolationSeverity.CRITICAL.value
        elif rule.priority >= 7:
            return ViolationSeverity.MAJOR.value
        elif rule.priority >= 5:
            return ViolationSeverity.MINOR.value
        else:
            return ViolationSeverity.WARNING.value


class SafetyBoundaryManager:
    """Manages safety boundaries and limits"""
    
    def __init__(self):
        self.boundaries: Dict[str, SafetyBoundary] = {}
        self.boundary_violations: List[SafetyViolation] = []
        self.adaptive_boundaries: Dict[str, Dict] = {}  # For boundaries that adapt over time
    
    async def initialize_default_boundaries(self):
        """Initialize default safety boundaries"""
        # Performance boundaries
        await self.add_boundary(
            "max_response_time",
            "Maximum response time",
            "hard",
            max_value=10.0,  # 10 seconds max
            target_value=2.0,  # 2 seconds target
            tolerance=0.5,
            enforcement_level=SafetyLevel.RESTRICTED
        )
        
        await self.add_boundary(
            "min_accuracy",
            "Minimum accuracy threshold",
            "hard",
            min_value=0.5,  # 50% minimum
            target_value=0.85,  # 85% target
            tolerance=0.05,
            enforcement_level=SafetyLevel.RESTRICTED
        )
        
        # Resource boundaries
        await self.add_boundary(
            "max_cpu_usage",
            "Maximum CPU utilization",
            "soft",
            max_value=0.95,  # 95% max
            target_value=0.70,  # 70% target
            tolerance=0.1,
            enforcement_level=SafetyLevel.GUIDED
        )
        
        await self.add_boundary(
            "max_memory_usage",
            "Maximum memory utilization",
            "hard",
            max_value=0.90,  # 90% max
            target_value=0.60,  # 60% target
            tolerance=0.1,
            enforcement_level=SafetyLevel.RESTRICTED
        )
        
        # Improvement rate boundaries
        await self.add_boundary(
            "max_improvement_rate",
            "Maximum improvement rate per cycle",
            "adaptive",
            max_value=0.3,  # 30% max improvement
            target_value=0.1,  # 10% target improvement
            tolerance=0.05,
            enforcement_level=SafetyLevel.RESTRICTED
        )
        
        # Safety violation rate boundary
        await self.add_boundary(
            "max_violation_rate",
            "Maximum safety violation rate",
            "hard",
            max_value=0.1,  # 10% max violation rate
            target_value=0.02,  # 2% target
            tolerance=0.01,
            enforcement_level=SafetyLevel.RESTRICTED
        )
        
        logger.info(f"Initialized {len(self.boundaries)} default safety boundaries")
    
    async def add_boundary(self, name: str, description: str, boundary_type: str,
                          min_value: float = None, max_value: float = None,
                          target_value: float = None, tolerance: float = 0.1,
                          enforcement_level: SafetyLevel = SafetyLevel.GUIDED) -> str:
        """Add a new safety boundary"""
        boundary = SafetyBoundary(
            boundary_id=str(uuid4()),
            name=name,
            description=description,
            boundary_type=boundary_type,
            min_value=min_value,
            max_value=max_value,
            target_value=target_value,
            tolerance=tolerance,
            enforcement_level=enforcement_level,
            violation_consequences=[],
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            metadata={}
        )
        
        self.boundaries[boundary.boundary_id] = boundary
        
        # Initialize adaptive tracking if needed
        if boundary_type == "adaptive":
            self.adaptive_boundaries[boundary.boundary_id] = {
                'history': [],
                'adaptation_rate': 0.1,
                'last_adaptation': datetime.utcnow()
            }
        
        logger.info(f"Added safety boundary: {name} ({boundary.boundary_id})")
        return boundary.boundary_id
    
    async def check_boundary(self, boundary_id: str, current_value: float,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if a value violates a safety boundary"""
        if boundary_id not in self.boundaries:
            return {'valid': False, 'error': f'Boundary {boundary_id} not found'}
        
        boundary = self.boundaries[boundary_id]
        violation_detected = False
        violation_type = None
        severity = ViolationSeverity.INFO
        
        # Check minimum boundary
        if boundary.min_value is not None and current_value < boundary.min_value:
            violation_detected = True
            violation_type = "below_minimum"
            severity = ViolationSeverity.MAJOR if boundary.enforcement_level == SafetyLevel.RESTRICTED else ViolationSeverity.WARNING
        
        # Check maximum boundary
        elif boundary.max_value is not None and current_value > boundary.max_value:
            violation_detected = True
            violation_type = "above_maximum"
            severity = ViolationSeverity.MAJOR if boundary.enforcement_level == SafetyLevel.RESTRICTED else ViolationSeverity.WARNING
        
        # Check target with tolerance
        elif boundary.target_value is not None:
            deviation = abs(current_value - boundary.target_value)
            if deviation > boundary.tolerance:
                violation_detected = True
                violation_type = "outside_tolerance"
                severity = ViolationSeverity.MINOR
        
        result = {
            'boundary_id': boundary_id,
            'boundary_name': boundary.name,
            'current_value': current_value,
            'violation_detected': violation_detected,
            'violation_type': violation_type,
            'severity': severity.value if violation_detected else None,
            'enforcement_level': boundary.enforcement_level.value,
            'boundary_info': {
                'min_value': boundary.min_value,
                'max_value': boundary.max_value,
                'target_value': boundary.target_value,
                'tolerance': boundary.tolerance
            }
        }
        
        # Record violation if detected
        if violation_detected:
            await self._record_boundary_violation(boundary, current_value, violation_type, severity, context)
        
        # Update adaptive boundaries
        if boundary.boundary_type == "adaptive":
            await self._update_adaptive_boundary(boundary_id, current_value)
        
        return result
    
    async def check_all_boundaries(self, values: Dict[str, float],
                                 context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Check multiple values against their corresponding boundaries"""
        results = []
        
        for boundary_id, boundary in self.boundaries.items():
            # Match boundary name to value key
            boundary_key = boundary.name.replace("max_", "").replace("min_", "")
            
            for value_key, value in values.items():
                if boundary_key in value_key or value_key in boundary.name:
                    result = await self.check_boundary(boundary_id, value, context)
                    results.append(result)
                    break
        
        return results
    
    async def _record_boundary_violation(self, boundary: SafetyBoundary, current_value: float,
                                       violation_type: str, severity: ViolationSeverity,
                                       context: Dict[str, Any] = None):
        """Record a boundary violation"""
        violation = SafetyViolation(
            violation_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            rule_id=None,
            boundary_id=boundary.boundary_id,
            severity=severity,
            description=f"Boundary '{boundary.name}' violated: {violation_type}",
            context=context or {},
            detected_values={'current_value': current_value, 'boundary_name': boundary.name},
            action_taken="logged",
            resolution_status="pending",
            resolution_time=None,
            metadata={'violation_type': violation_type}
        )
        
        self.boundary_violations.append(violation)
        
        # Maintain violation history size
        if len(self.boundary_violations) > 1000:
            self.boundary_violations = self.boundary_violations[-500:]
        
        logger.warning(f"Boundary violation: {boundary.name} - {violation_type} (severity: {severity.value})")
    
    async def _update_adaptive_boundary(self, boundary_id: str, current_value: float):
        """Update adaptive boundary based on historical values"""
        if boundary_id not in self.adaptive_boundaries:
            return
        
        adaptive_info = self.adaptive_boundaries[boundary_id]
        boundary = self.boundaries[boundary_id]
        
        # Add to history
        adaptive_info['history'].append({
            'timestamp': datetime.utcnow(),
            'value': current_value
        })
        
        # Maintain history size
        if len(adaptive_info['history']) > 100:
            adaptive_info['history'] = adaptive_info['history'][-50:]
        
        # Check if it's time to adapt
        time_since_adaptation = datetime.utcnow() - adaptive_info['last_adaptation']
        if time_since_adaptation > timedelta(hours=1) and len(adaptive_info['history']) >= 10:
            # Calculate new boundary values based on historical data
            recent_values = [entry['value'] for entry in adaptive_info['history'][-20:]]
            
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # Adapt target value towards historical mean
            if boundary.target_value is not None:
                adaptation_rate = adaptive_info['adaptation_rate']
                new_target = boundary.target_value * (1 - adaptation_rate) + mean_value * adaptation_rate
                
                # Ensure new target is within reasonable bounds
                if boundary.min_value is not None:
                    new_target = max(new_target, boundary.min_value)
                if boundary.max_value is not None:
                    new_target = min(new_target, boundary.max_value)
                
                boundary.target_value = new_target
                boundary.last_updated = datetime.utcnow()
                
                adaptive_info['last_adaptation'] = datetime.utcnow()
                
                logger.info(f"Adapted boundary '{boundary.name}' target to {new_target:.3f}")


class SafetyConstraintSystem:
    """Main safety constraint system coordinating rules and boundaries"""
    
    def __init__(self):
        self.system_id = str(uuid4())
        self.rule_engine = SafetyRuleEngine()
        self.boundary_manager = SafetyBoundaryManager()
        
        self.monitoring_state = SafetyMonitoringState(
            monitoring_active=True,
            safety_level=SafetyLevel.RESTRICTED,
            total_boundaries=0,
            total_rules=0,
            active_violations=0,
            recent_violations=0,
            last_check_time=datetime.utcnow(),
            system_risk_score=0.0,
            monitoring_overhead=0.0,
            metadata={}
        )
        
        self.violation_history: List[SafetyViolation] = []
        self.system_shutdown_conditions: List[str] = [
            "catastrophic_violation_detected",
            "multiple_critical_violations",
            "safety_system_failure"
        ]
        
        # Risk scoring parameters
        self.risk_weights = {
            ViolationSeverity.INFO: 0.1,
            ViolationSeverity.WARNING: 0.2,
            ViolationSeverity.MINOR: 0.3,
            ViolationSeverity.MAJOR: 0.6,
            ViolationSeverity.CRITICAL: 0.9,
            ViolationSeverity.CATASTROPHIC: 1.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the safety constraint system"""
        try:
            logger.info(f"Initializing Safety Constraint System {self.system_id}")
            
            # Initialize default boundaries
            await self.boundary_manager.initialize_default_boundaries()
            
            # Initialize default rules
            await self._initialize_default_rules()
            
            # Update monitoring state
            self.monitoring_state.total_boundaries = len(self.boundary_manager.boundaries)
            self.monitoring_state.total_rules = len(self.rule_engine.rules)
            
            # Start monitoring loop
            asyncio.create_task(self._safety_monitoring_loop())
            
            logger.info("Safety Constraint System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Safety Constraint System: {e}")
            return False
    
    async def _initialize_default_rules(self):
        """Initialize default safety rules"""
        # Performance rules
        await self.rule_engine.add_rule(
            "Response Time Limit",
            "Response time must not exceed maximum threshold",
            "invariant",
            "response_time < 10.0",
            ConstraintScope.GLOBAL,
            priority=8,
            enforcement_level=SafetyLevel.RESTRICTED,
            violation_action="block"
        )
        
        await self.rule_engine.add_rule(
            "Accuracy Minimum",
            "System accuracy must remain above minimum threshold",
            "invariant",
            "accuracy > 0.5",
            ConstraintScope.GLOBAL,
            priority=9,
            enforcement_level=SafetyLevel.RESTRICTED,
            violation_action="rollback"
        )
        
        # Resource rules
        await self.rule_engine.add_rule(
            "CPU Usage Limit",
            "CPU usage should not exceed safe limits",
            "invariant",
            "cpu_utilization < 0.95",
            ConstraintScope.GLOBAL,
            priority=7,
            enforcement_level=SafetyLevel.GUIDED,
            violation_action="warn"
        )
        
        await self.rule_engine.add_rule(
            "Memory Usage Limit",
            "Memory usage must not exceed critical threshold",
            "invariant",
            "memory_utilization < 0.90",
            ConstraintScope.GLOBAL,
            priority=8,
            enforcement_level=SafetyLevel.RESTRICTED,
            violation_action="block"
        )
        
        # Improvement safety rules
        await self.rule_engine.add_rule(
            "Improvement Rate Limit",
            "Single improvement cannot exceed maximum change rate",
            "precondition",
            "improvement_magnitude < 0.3",
            ConstraintScope.OPERATION,
            priority=8,
            enforcement_level=SafetyLevel.RESTRICTED,
            violation_action="block"
        )
        
        await self.rule_engine.add_rule(
            "Violation Rate Limit",
            "Safety violation rate must remain below threshold",
            "temporal",
            "violation_rate < 0.1",
            ConstraintScope.GLOBAL,
            priority=9,
            enforcement_level=SafetyLevel.RESTRICTED,
            violation_action="rollback"
        )
        
        logger.info(f"Initialized {len(self.rule_engine.rules)} default safety rules")
    
    async def validate_operation(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an operation against all safety constraints"""
        validation_start_time = time.time()
        
        validation_result = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'enforcement_actions': [],
            'risk_score': 0.0,
            'safety_level': self.monitoring_state.safety_level.value,
            'rule_evaluations': [],
            'boundary_checks': []
        }
        
        try:
            # Evaluate safety rules
            rule_results = await self.rule_engine.evaluate_all_rules(operation_context)
            validation_result['rule_evaluations'] = rule_results
            
            for rule_result in rule_results:
                if rule_result.get('violation_detected', False):
                    violation_info = {
                        'type': 'rule_violation',
                        'rule_id': rule_result['rule_id'],
                        'rule_name': rule_result['rule_name'],
                        'severity': rule_result['severity'],
                        'action': rule_result['violation_action']
                    }
                    
                    if rule_result['violation_action'] in ['block', 'rollback', 'shutdown']:
                        validation_result['valid'] = False
                        validation_result['violations'].append(violation_info)
                        validation_result['enforcement_actions'].append(rule_result['violation_action'])
                    else:
                        validation_result['warnings'].append(violation_info)
                    
                    # Record violation
                    await self._record_rule_violation(rule_result, operation_context)
            
            # Check boundaries
            numeric_values = {k: v for k, v in operation_context.items() if isinstance(v, (int, float))}
            if numeric_values:
                boundary_results = await self.boundary_manager.check_all_boundaries(
                    numeric_values, operation_context
                )
                validation_result['boundary_checks'] = boundary_results
                
                for boundary_result in boundary_results:
                    if boundary_result.get('violation_detected', False):
                        violation_info = {
                            'type': 'boundary_violation',
                            'boundary_id': boundary_result['boundary_id'],
                            'boundary_name': boundary_result['boundary_name'],
                            'severity': boundary_result['severity'],
                            'violation_type': boundary_result['violation_type']
                        }
                        
                        if boundary_result['enforcement_level'] == SafetyLevel.RESTRICTED.value:
                            validation_result['valid'] = False
                            validation_result['violations'].append(violation_info)
                            validation_result['enforcement_actions'].append('block')
                        else:
                            validation_result['warnings'].append(violation_info)
            
            # Calculate overall risk score
            validation_result['risk_score'] = await self._calculate_operation_risk_score(
                validation_result['violations'], validation_result['warnings']
            )
            
            # Update monitoring state
            self.monitoring_state.active_violations = len(validation_result['violations'])
            self.monitoring_state.last_check_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error during safety validation: {e}")
            validation_result = {
                'valid': False,
                'error': str(e),
                'violations': [{'type': 'validation_error', 'description': str(e)}],
                'enforcement_actions': ['block']
            }
        
        finally:
            # Calculate monitoring overhead
            validation_time = time.time() - validation_start_time
            self.monitoring_state.monitoring_overhead = validation_time
        
        return validation_result
    
    async def _record_rule_violation(self, rule_result: Dict[str, Any], context: Dict[str, Any]):
        """Record a rule violation"""
        severity_map = {
            'info': ViolationSeverity.INFO,
            'warning': ViolationSeverity.WARNING,
            'minor': ViolationSeverity.MINOR,
            'major': ViolationSeverity.MAJOR,
            'critical': ViolationSeverity.CRITICAL,
            'catastrophic': ViolationSeverity.CATASTROPHIC
        }
        
        severity = severity_map.get(rule_result.get('severity', 'warning'), ViolationSeverity.WARNING)
        
        violation = SafetyViolation(
            violation_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            rule_id=rule_result['rule_id'],
            boundary_id=None,
            severity=severity,
            description=f"Rule '{rule_result['rule_name']}' violated",
            context=context,
            detected_values=rule_result.get('context_values', {}),
            action_taken=rule_result['violation_action'],
            resolution_status="pending",
            resolution_time=None,
            metadata=rule_result.get('evaluation_details', {})
        )
        
        self.violation_history.append(violation)
        
        # Maintain violation history size
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-500:]
        
        # Check for shutdown conditions
        await self._check_shutdown_conditions(violation)
    
    async def _calculate_operation_risk_score(self, violations: List[Dict[str, Any]],
                                            warnings: List[Dict[str, Any]]) -> float:
        """Calculate risk score for an operation"""
        risk_score = 0.0
        
        # Add risk from violations
        for violation in violations:
            severity_str = violation.get('severity', 'warning')
            severity_enum = getattr(ViolationSeverity, severity_str.upper(), ViolationSeverity.WARNING)
            risk_score += self.risk_weights.get(severity_enum, 0.2)
        
        # Add risk from warnings (reduced weight)
        for warning in warnings:
            severity_str = warning.get('severity', 'warning')
            severity_enum = getattr(ViolationSeverity, severity_str.upper(), ViolationSeverity.WARNING)
            risk_score += self.risk_weights.get(severity_enum, 0.2) * 0.5
        
        # Normalize to 0-1 range
        return min(1.0, risk_score)
    
    async def _check_shutdown_conditions(self, violation: SafetyViolation):
        """Check if violation triggers system shutdown conditions"""
        if violation.severity == ViolationSeverity.CATASTROPHIC:
            logger.critical(f"Catastrophic violation detected: {violation.description}")
            await self._trigger_safety_shutdown("catastrophic_violation_detected")
        
        # Check for multiple critical violations in short time
        recent_critical = [
            v for v in self.violation_history[-10:]
            if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.CATASTROPHIC]
            and v.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]
        
        if len(recent_critical) >= 3:
            logger.critical(f"Multiple critical violations detected: {len(recent_critical)}")
            await self._trigger_safety_shutdown("multiple_critical_violations")
    
    async def _trigger_safety_shutdown(self, reason: str):
        """Trigger safety shutdown"""
        logger.critical(f"SAFETY SHUTDOWN TRIGGERED: {reason}")
        
        # Set safety level to locked
        self.monitoring_state.safety_level = SafetyLevel.LOCKED
        
        # Record shutdown event
        shutdown_violation = SafetyViolation(
            violation_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            rule_id=None,
            boundary_id=None,
            severity=ViolationSeverity.CATASTROPHIC,
            description=f"Safety shutdown triggered: {reason}",
            context={'shutdown_reason': reason},
            detected_values={},
            action_taken="shutdown",
            resolution_status="pending",
            resolution_time=None,
            metadata={'automatic_shutdown': True}
        )
        
        self.violation_history.append(shutdown_violation)
        
        # In a real system, this would trigger actual shutdown procedures
        logger.critical("System entering safe mode - no further modifications allowed")
    
    async def _safety_monitoring_loop(self):
        """Main safety monitoring loop"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Update monitoring state
                self.monitoring_state.last_check_time = current_time
                
                # Calculate recent violations
                recent_cutoff = current_time - timedelta(hours=1)
                recent_violations = [
                    v for v in self.violation_history
                    if v.timestamp > recent_cutoff
                ]
                self.monitoring_state.recent_violations = len(recent_violations)
                
                # Calculate system risk score
                if recent_violations:
                    risk_scores = []
                    for violation in recent_violations:
                        risk_scores.append(self.risk_weights.get(violation.severity, 0.2))
                    
                    self.monitoring_state.system_risk_score = np.mean(risk_scores)
                else:
                    self.monitoring_state.system_risk_score = 0.0
                
                # Adaptive safety level adjustment
                await self._adjust_safety_level()
                
                # Clean up old violations
                cutoff_time = current_time - timedelta(days=7)
                self.violation_history = [
                    v for v in self.violation_history
                    if v.timestamp > cutoff_time
                ]
                
                # Sleep before next check
                await asyncio.sleep(60)  # 1-minute monitoring cycle
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                await asyncio.sleep(300)  # 5-minute sleep on error
    
    async def _adjust_safety_level(self):
        """Automatically adjust safety level based on system state"""
        risk_score = self.monitoring_state.system_risk_score
        
        # Don't adjust if in locked state
        if self.monitoring_state.safety_level == SafetyLevel.LOCKED:
            return
        
        # Adjust based on risk score
        if risk_score > 0.8:
            self.monitoring_state.safety_level = SafetyLevel.RESTRICTED
        elif risk_score > 0.5:
            self.monitoring_state.safety_level = SafetyLevel.GUIDED
        elif risk_score > 0.2:
            self.monitoring_state.safety_level = SafetyLevel.MONITORING
        else:
            # Allow more freedom when risk is low
            if self.monitoring_state.safety_level == SafetyLevel.RESTRICTED:
                self.monitoring_state.safety_level = SafetyLevel.GUIDED
    
    async def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety system status"""
        return {
            'system_id': self.system_id,
            'monitoring_state': asdict(self.monitoring_state),
            'total_boundaries': len(self.boundary_manager.boundaries),
            'total_rules': len(self.rule_engine.rules),
            'total_violations': len(self.violation_history),
            'recent_violations': [
                asdict(v) for v in self.violation_history[-10:]
            ],
            'boundary_statuses': [
                {
                    'boundary_id': bid,
                    'name': boundary.name,
                    'type': boundary.boundary_type,
                    'enforcement_level': boundary.enforcement_level.value
                }
                for bid, boundary in self.boundary_manager.boundaries.items()
            ],
            'rule_statuses': [
                {
                    'rule_id': rid,
                    'name': rule.name,
                    'priority': rule.priority,
                    'violation_count': rule.violation_count,
                    'enforcement_level': rule.enforcement_level.value
                }
                for rid, rule in self.rule_engine.rules.items()
            ]
        }
    
    async def reset_safety_level(self, new_level: SafetyLevel, authorization_code: str = None) -> bool:
        """Reset safety level (requires authorization for sensitive operations)"""
        # In practice, this would require proper authorization
        if new_level == SafetyLevel.UNRESTRICTED and authorization_code != "EMERGENCY_OVERRIDE":
            logger.warning("Attempt to set unrestricted safety level without proper authorization")
            return False
        
        old_level = self.monitoring_state.safety_level
        self.monitoring_state.safety_level = new_level
        
        logger.info(f"Safety level changed from {old_level.value} to {new_level.value}")
        return True
    
    async def cleanup(self):
        """Clean up resources"""
        # Set monitoring to inactive
        self.monitoring_state.monitoring_active = False
        
        logger.info(f"Safety Constraint System {self.system_id} cleaned up")
