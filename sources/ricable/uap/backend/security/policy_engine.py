# File: backend/security/policy_engine.py
"""
Advanced Policy Engine for Attribute-Based Access Control (ABAC)
Provides fine-grained authorization with dynamic permission evaluation,
resource-based access control, and contextual decision making.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
import logging
import re
from abc import ABC, abstractmethod

from .audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger

logger = logging.getLogger(__name__)

class PolicyEffect(Enum):
    """Policy evaluation effects"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"

class PolicyType(Enum):
    """Types of policies"""
    RBAC = "rbac"  # Role-based access control
    ABAC = "abac"  # Attribute-based access control
    RESOURCE = "resource"  # Resource-based access control
    TIME = "time"  # Time-based access control
    LOCATION = "location"  # Location-based access control
    RISK = "risk"  # Risk-based access control

class ContextType(Enum):
    """Types of evaluation context"""
    USER = "user"
    RESOURCE = "resource"
    ACTION = "action"
    ENVIRONMENT = "environment"
    REQUEST = "request"

@dataclass
class PolicyAttribute:
    """Policy attribute definition"""
    name: str
    type: str  # string, number, boolean, list, datetime
    required: bool = False
    default_value: Any = None
    description: str = ""
    validation_pattern: Optional[str] = None

@dataclass
class PolicyCondition:
    """Policy condition for evaluation"""
    attribute: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, regex, contains
    value: Any
    context_type: ContextType

@dataclass
class PolicyRule:
    """Individual policy rule"""
    rule_id: str
    name: str
    description: str
    effect: PolicyEffect
    conditions: List[PolicyCondition]
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    tags: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Policy:
    """Complete policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    target: str  # Resource pattern or scope
    rules: List[PolicyRule]
    attributes: List[PolicyAttribute]
    version: str = "1.0"
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = ""
    enabled: bool = True
    tags: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class EvaluationContext:
    """Context for policy evaluation"""
    user_attributes: Dict[str, Any]
    resource_attributes: Dict[str, Any]
    action_attributes: Dict[str, Any]
    environment_attributes: Dict[str, Any]
    request_attributes: Dict[str, Any]
    timestamp: datetime = None

@dataclass
class PolicyDecision:
    """Result of policy evaluation"""
    decision: PolicyEffect
    applicable_policies: List[str]
    matched_rules: List[str]
    evaluation_time_ms: float
    obligations: List[str] = None  # Actions that must be performed
    advice: List[str] = None  # Recommendations
    reason: str = ""
    metadata: Dict[str, Any] = None

class PolicyEvaluator(ABC):
    """Abstract base class for policy evaluators"""
    
    @abstractmethod
    async def evaluate(self, policy: Policy, context: EvaluationContext) -> PolicyDecision:
        """Evaluate policy against context"""
        pass

class ABACEvaluator(PolicyEvaluator):
    """Attribute-Based Access Control evaluator"""
    
    async def evaluate(self, policy: Policy, context: EvaluationContext) -> PolicyDecision:
        """Evaluate ABAC policy"""
        start_time = datetime.now()
        matched_rules = []
        applicable_policies = [policy.policy_id]
        
        # Sort rules by priority (lower number = higher priority)
        sorted_rules = sorted(policy.rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            if await self._evaluate_rule(rule, context):
                matched_rules.append(rule.rule_id)
                
                # Return first matching rule's effect (highest priority)
                evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PolicyDecision(
                    decision=rule.effect,
                    applicable_policies=applicable_policies,
                    matched_rules=matched_rules,
                    evaluation_time_ms=evaluation_time,
                    reason=f"Matched rule: {rule.name}",
                    metadata={"rule_id": rule.rule_id, "rule_priority": rule.priority}
                )
        
        # No rules matched - default deny
        evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PolicyDecision(
            decision=PolicyEffect.DENY,
            applicable_policies=applicable_policies,
            matched_rules=matched_rules,
            evaluation_time_ms=evaluation_time,
            reason="No rules matched - default deny"
        )
    
    async def _evaluate_rule(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate individual rule conditions"""
        if not rule.conditions:
            return True  # Rule with no conditions matches
        
        # All conditions must be true (AND logic)
        for condition in rule.conditions:
            if not await self._evaluate_condition(condition, context):
                return False
        
        return True
    
    async def _evaluate_condition(self, condition: PolicyCondition, context: EvaluationContext) -> bool:
        """Evaluate individual condition"""
        # Get the actual value from context
        actual_value = self._get_context_value(condition.attribute, condition.context_type, context)
        expected_value = condition.value
        operator = condition.operator
        
        if actual_value is None:
            return False
        
        # Evaluate based on operator
        if operator == "eq":
            return actual_value == expected_value
        elif operator == "ne":
            return actual_value != expected_value
        elif operator == "gt":
            return actual_value > expected_value
        elif operator == "lt":
            return actual_value < expected_value
        elif operator == "gte":
            return actual_value >= expected_value
        elif operator == "lte":
            return actual_value <= expected_value
        elif operator == "in":
            return actual_value in expected_value if isinstance(expected_value, (list, set)) else False
        elif operator == "not_in":
            return actual_value not in expected_value if isinstance(expected_value, (list, set)) else True
        elif operator == "regex":
            return bool(re.match(expected_value, str(actual_value)))
        elif operator == "contains":
            return expected_value in str(actual_value)
        elif operator == "exists":
            return actual_value is not None
        elif operator == "not_exists":
            return actual_value is None
        
        return False
    
    def _get_context_value(self, attribute: str, context_type: ContextType, context: EvaluationContext) -> Any:
        """Get value from evaluation context"""
        if context_type == ContextType.USER:
            return context.user_attributes.get(attribute)
        elif context_type == ContextType.RESOURCE:
            return context.resource_attributes.get(attribute)
        elif context_type == ContextType.ACTION:
            return context.action_attributes.get(attribute)
        elif context_type == ContextType.ENVIRONMENT:
            return context.environment_attributes.get(attribute)
        elif context_type == ContextType.REQUEST:
            return context.request_attributes.get(attribute)
        
        return None

class RBACEvaluator(PolicyEvaluator):
    """Role-Based Access Control evaluator"""
    
    async def evaluate(self, policy: Policy, context: EvaluationContext) -> PolicyDecision:
        """Evaluate RBAC policy"""
        start_time = datetime.now()
        matched_rules = []
        applicable_policies = [policy.policy_id]
        
        user_roles = context.user_attributes.get("roles", [])
        required_action = context.action_attributes.get("action")
        resource_type = context.resource_attributes.get("type")
        
        for rule in policy.rules:
            if not rule.enabled:
                continue
            
            # Check if user has required role
            required_roles = [c.value for c in rule.conditions if c.attribute == "role"]
            if required_roles and any(role in user_roles for role in required_roles):
                # Check action permission
                allowed_actions = [c.value for c in rule.conditions if c.attribute == "action"]
                if not allowed_actions or required_action in allowed_actions:
                    matched_rules.append(rule.rule_id)
                    
                    evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    return PolicyDecision(
                        decision=rule.effect,
                        applicable_policies=applicable_policies,
                        matched_rules=matched_rules,
                        evaluation_time_ms=evaluation_time,
                        reason=f"RBAC rule matched: {rule.name}"
                    )
        
        # No matching role/action - deny
        evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PolicyDecision(
            decision=PolicyEffect.DENY,
            applicable_policies=applicable_policies,
            matched_rules=matched_rules,
            evaluation_time_ms=evaluation_time,
            reason="No matching role or insufficient permissions"
        )

class ResourceEvaluator(PolicyEvaluator):
    """Resource-Based Access Control evaluator"""
    
    async def evaluate(self, policy: Policy, context: EvaluationContext) -> PolicyDecision:
        """Evaluate resource-based policy"""
        start_time = datetime.now()
        matched_rules = []
        applicable_policies = [policy.policy_id]
        
        resource_id = context.resource_attributes.get("id")
        resource_owner = context.resource_attributes.get("owner")
        user_id = context.user_attributes.get("user_id")
        action = context.action_attributes.get("action")
        
        for rule in policy.rules:
            if not rule.enabled:
                continue
            
            # Check ownership
            if "owner" in [c.attribute for c in rule.conditions]:
                if resource_owner == user_id:
                    matched_rules.append(rule.rule_id)
                    
                    evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    return PolicyDecision(
                        decision=rule.effect,
                        applicable_policies=applicable_policies,
                        matched_rules=matched_rules,
                        evaluation_time_ms=evaluation_time,
                        reason=f"Resource owner access: {rule.name}"
                    )
        
        # Not owner or no resource access rules - evaluate other conditions
        abac_evaluator = ABACEvaluator()
        return await abac_evaluator.evaluate(policy, context)

class PolicyEngine:
    """
    Advanced Policy Engine for fine-grained access control.
    
    Features:
    - Multiple evaluation strategies (RBAC, ABAC, Resource-based)
    - Dynamic policy management
    - Context-aware decision making
    - Policy conflict resolution
    - Performance monitoring
    - Audit logging
    """
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.evaluators: Dict[PolicyType, PolicyEvaluator] = {
            PolicyType.ABAC: ABACEvaluator(),
            PolicyType.RBAC: RBACEvaluator(),
            PolicyType.RESOURCE: ResourceEvaluator()
        }
        self.audit_logger = get_security_audit_logger()
        
        # Performance metrics
        self.evaluation_metrics = {
            "total_evaluations": 0,
            "average_evaluation_time_ms": 0.0,
            "decisions_by_effect": {effect.value: 0 for effect in PolicyEffect}
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("Policy Engine initialized")
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        
        # Admin access policy
        admin_policy = Policy(
            policy_id="ADMIN_FULL_ACCESS",
            name="Administrator Full Access",
            description="Full system access for administrators",
            policy_type=PolicyType.RBAC,
            target="*",
            rules=[
                PolicyRule(
                    rule_id="ADMIN_RULE_001",
                    name="Admin Full Access Rule",
                    description="Administrators have full access to all resources",
                    effect=PolicyEffect.ALLOW,
                    conditions=[
                        PolicyCondition(
                            attribute="roles",
                            operator="contains",
                            value="admin",
                            context_type=ContextType.USER
                        )
                    ],
                    priority=10
                )
            ],
            attributes=[],
            created_at=datetime.now(timezone.utc)
        )
        
        # User self-service policy
        user_policy = Policy(
            policy_id="USER_SELF_SERVICE",
            name="User Self-Service Access",
            description="Users can access and modify their own resources",
            policy_type=PolicyType.RESOURCE,
            target="user:*",
            rules=[
                PolicyRule(
                    rule_id="USER_RULE_001",
                    name="User Self Access Rule",
                    description="Users can read their own data",
                    effect=PolicyEffect.ALLOW,
                    conditions=[
                        PolicyCondition(
                            attribute="owner",
                            operator="eq",
                            value="${user.user_id}",
                            context_type=ContextType.RESOURCE
                        ),
                        PolicyCondition(
                            attribute="action",
                            operator="in",
                            value=["read", "update"],
                            context_type=ContextType.ACTION
                        )
                    ],
                    priority=50
                )
            ],
            attributes=[],
            created_at=datetime.now(timezone.utc)
        )
        
        # Time-based access policy
        business_hours_policy = Policy(
            policy_id="BUSINESS_HOURS_ACCESS",
            name="Business Hours Access Control",
            description="Restrict access to business hours for sensitive operations",
            policy_type=PolicyType.TIME,
            target="sensitive:*",
            rules=[
                PolicyRule(
                    rule_id="TIME_RULE_001",
                    name="Business Hours Rule",
                    description="Allow access only during business hours (9 AM - 6 PM)",
                    effect=PolicyEffect.CONDITIONAL,
                    conditions=[
                        PolicyCondition(
                            attribute="hour",
                            operator="gte",
                            value=9,
                            context_type=ContextType.ENVIRONMENT
                        ),
                        PolicyCondition(
                            attribute="hour",
                            operator="lte",
                            value=18,
                            context_type=ContextType.ENVIRONMENT
                        )
                    ],
                    priority=30
                )
            ],
            attributes=[],
            created_at=datetime.now(timezone.utc)
        )
        
        # Store default policies
        self.policies[admin_policy.policy_id] = admin_policy
        self.policies[user_policy.policy_id] = user_policy
        self.policies[business_hours_policy.policy_id] = business_hours_policy
    
    async def evaluate(self, user_id: str, resource: str, action: str,
                      context: Dict[str, Any] = None) -> PolicyDecision:
        """
        Evaluate access decision for user, resource, and action.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            context: Additional context for evaluation
            
        Returns:
            Policy decision with effect and details
        """
        evaluation_start = datetime.now()
        
        try:
            # Build evaluation context
            eval_context = await self._build_evaluation_context(
                user_id, resource, action, context or {}
            )
            
            # Find applicable policies
            applicable_policies = self._find_applicable_policies(resource, action)
            
            if not applicable_policies:
                # No applicable policies - default deny
                decision = PolicyDecision(
                    decision=PolicyEffect.DENY,
                    applicable_policies=[],
                    matched_rules=[],
                    evaluation_time_ms=0,
                    reason="No applicable policies found - default deny"
                )
            else:
                # Evaluate policies
                decision = await self._evaluate_policies(applicable_policies, eval_context)
            
            # Update metrics
            self._update_metrics(decision)
            
            # Log access decision
            await self.audit_logger.log_authorization_check(
                user_id=user_id,
                resource=resource,
                action=action,
                allowed=(decision.decision == PolicyEffect.ALLOW),
                details={
                    "decision": decision.decision.value,
                    "applicable_policies": decision.applicable_policies,
                    "matched_rules": decision.matched_rules,
                    "evaluation_time_ms": decision.evaluation_time_ms,
                    "reason": decision.reason
                }
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")
            
            # Log evaluation error
            await self.audit_logger.log_security_event(
                "Policy evaluation error",
                "high",
                user_id=user_id,
                details={
                    "resource": resource,
                    "action": action,
                    "error": str(e)
                }
            )
            
            # Default deny on error
            return PolicyDecision(
                decision=PolicyEffect.DENY,
                applicable_policies=[],
                matched_rules=[],
                evaluation_time_ms=0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    async def _build_evaluation_context(self, user_id: str, resource: str, 
                                       action: str, context: Dict[str, Any]) -> EvaluationContext:
        """Build evaluation context from user, resource, and environment"""
        current_time = datetime.now(timezone.utc)
        
        # User attributes (would be fetched from user service)
        user_attributes = {
            "user_id": user_id,
            "roles": ["user"],  # Would fetch from user service
            "department": "unknown",
            "security_clearance": "standard"
        }
        
        # Parse resource attributes
        resource_parts = resource.split(":")
        resource_attributes = {
            "id": resource,
            "type": resource_parts[0] if len(resource_parts) > 0 else "unknown",
            "category": resource_parts[1] if len(resource_parts) > 1 else "unknown",
            "owner": context.get("resource_owner"),
            "sensitivity": context.get("resource_sensitivity", "public")
        }
        
        # Action attributes
        action_attributes = {
            "action": action,
            "category": self._categorize_action(action),
            "risk_level": self._assess_action_risk(action)
        }
        
        # Environment attributes
        environment_attributes = {
            "timestamp": current_time,
            "hour": current_time.hour,
            "day_of_week": current_time.weekday(),
            "is_business_hours": 9 <= current_time.hour <= 17,
            "ip_address": context.get("ip_address"),
            "location": context.get("location", "unknown"),
            "security_level": context.get("security_level", "standard")
        }
        
        # Request attributes
        request_attributes = {
            "session_id": context.get("session_id"),
            "user_agent": context.get("user_agent"),
            "mfa_verified": context.get("mfa_verified", False),
            "risk_score": context.get("risk_score", 0)
        }
        
        return EvaluationContext(
            user_attributes=user_attributes,
            resource_attributes=resource_attributes,
            action_attributes=action_attributes,
            environment_attributes=environment_attributes,
            request_attributes=request_attributes,
            timestamp=current_time
        )
    
    def _categorize_action(self, action: str) -> str:
        """Categorize action for policy evaluation"""
        read_actions = ["read", "view", "list", "get", "search"]
        write_actions = ["create", "update", "modify", "edit", "write"]
        delete_actions = ["delete", "remove", "destroy"]
        admin_actions = ["admin", "configure", "manage", "control"]
        
        action_lower = action.lower()
        
        if action_lower in read_actions:
            return "read"
        elif action_lower in write_actions:
            return "write"
        elif action_lower in delete_actions:
            return "delete"
        elif action_lower in admin_actions:
            return "admin"
        else:
            return "other"
    
    def _assess_action_risk(self, action: str) -> str:
        """Assess risk level of action"""
        high_risk = ["delete", "admin", "configure", "destroy"]
        medium_risk = ["create", "update", "modify", "write"]
        low_risk = ["read", "view", "list", "get"]
        
        action_lower = action.lower()
        
        if action_lower in high_risk:
            return "high"
        elif action_lower in medium_risk:
            return "medium"
        elif action_lower in low_risk:
            return "low"
        else:
            return "medium"
    
    def _find_applicable_policies(self, resource: str, action: str) -> List[Policy]:
        """Find policies applicable to resource and action"""
        applicable = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            # Check if policy target matches resource
            if self._target_matches(policy.target, resource):
                applicable.append(policy)
        
        # Sort by policy priority (if defined in metadata)
        return sorted(applicable, key=lambda p: p.metadata.get("priority", 100) if p.metadata else 100)
    
    def _target_matches(self, target: str, resource: str) -> bool:
        """Check if policy target matches resource"""
        if target == "*":
            return True
        
        # Support wildcard matching
        target_pattern = target.replace("*", ".*")
        return bool(re.match(target_pattern, resource))
    
    async def _evaluate_policies(self, policies: List[Policy], 
                               context: EvaluationContext) -> PolicyDecision:
        """Evaluate multiple policies and resolve conflicts"""
        decisions = []
        
        for policy in policies:
            evaluator = self.evaluators.get(policy.policy_type)
            if evaluator:
                decision = await evaluator.evaluate(policy, context)
                decisions.append(decision)
        
        # Resolve policy conflicts
        return self._resolve_policy_conflicts(decisions)
    
    def _resolve_policy_conflicts(self, decisions: List[PolicyDecision]) -> PolicyDecision:
        """Resolve conflicts between multiple policy decisions"""
        if not decisions:
            return PolicyDecision(
                decision=PolicyEffect.DENY,
                applicable_policies=[],
                matched_rules=[],
                evaluation_time_ms=0,
                reason="No policy decisions to evaluate"
            )
        
        # Combine all metadata
        all_policies = []
        all_rules = []
        total_time = 0
        
        for decision in decisions:
            all_policies.extend(decision.applicable_policies)
            all_rules.extend(decision.matched_rules)
            total_time += decision.evaluation_time_ms
        
        # Policy conflict resolution strategy:
        # 1. Explicit DENY takes precedence
        # 2. Then ALLOW
        # 3. Then CONDITIONAL
        # 4. Default DENY
        
        explicit_deny = [d for d in decisions if d.decision == PolicyEffect.DENY]
        if explicit_deny:
            return PolicyDecision(
                decision=PolicyEffect.DENY,
                applicable_policies=list(set(all_policies)),
                matched_rules=list(set(all_rules)),
                evaluation_time_ms=total_time,
                reason="Explicit deny policy matched"
            )
        
        explicit_allow = [d for d in decisions if d.decision == PolicyEffect.ALLOW]
        if explicit_allow:
            return PolicyDecision(
                decision=PolicyEffect.ALLOW,
                applicable_policies=list(set(all_policies)),
                matched_rules=list(set(all_rules)),
                evaluation_time_ms=total_time,
                reason="Allow policy matched"
            )
        
        conditional = [d for d in decisions if d.decision == PolicyEffect.CONDITIONAL]
        if conditional:
            return PolicyDecision(
                decision=PolicyEffect.CONDITIONAL,
                applicable_policies=list(set(all_policies)),
                matched_rules=list(set(all_rules)),
                evaluation_time_ms=total_time,
                reason="Conditional access - additional verification required"
            )
        
        # Default deny
        return PolicyDecision(
            decision=PolicyEffect.DENY,
            applicable_policies=list(set(all_policies)),
            matched_rules=list(set(all_rules)),
            evaluation_time_ms=total_time,
            reason="No explicit allow - default deny"
        )
    
    def _update_metrics(self, decision: PolicyDecision):
        """Update evaluation metrics"""
        self.evaluation_metrics["total_evaluations"] += 1
        
        # Update average evaluation time
        current_avg = self.evaluation_metrics["average_evaluation_time_ms"]
        total_evals = self.evaluation_metrics["total_evaluations"]
        new_avg = ((current_avg * (total_evals - 1)) + decision.evaluation_time_ms) / total_evals
        self.evaluation_metrics["average_evaluation_time_ms"] = new_avg
        
        # Update decision counts
        self.evaluation_metrics["decisions_by_effect"][decision.decision.value] += 1
    
    # ==================== POLICY MANAGEMENT ====================
    
    async def create_policy(self, policy: Policy) -> bool:
        """Create new policy"""
        if policy.policy_id in self.policies:
            raise ValueError(f"Policy {policy.policy_id} already exists")
        
        # Validate policy
        validation_errors = self._validate_policy(policy)
        if validation_errors:
            raise ValueError(f"Policy validation failed: {'; '.join(validation_errors)}")
        
        policy.created_at = datetime.now(timezone.utc)
        policy.updated_at = policy.created_at
        
        self.policies[policy.policy_id] = policy
        
        await self.audit_logger.log_admin_action(
            admin_id="system",
            action="create_policy",
            target=policy.policy_id,
            success=True,
            details={"policy_name": policy.name, "policy_type": policy.policy_type.value}
        )
        
        return True
    
    async def update_policy(self, policy_id: str, policy: Policy) -> bool:
        """Update existing policy"""
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        
        # Validate policy
        validation_errors = self._validate_policy(policy)
        if validation_errors:
            raise ValueError(f"Policy validation failed: {'; '.join(validation_errors)}")
        
        policy.updated_at = datetime.now(timezone.utc)
        self.policies[policy_id] = policy
        
        await self.audit_logger.log_admin_action(
            admin_id="system",
            action="update_policy",
            target=policy_id,
            success=True,
            details={"policy_name": policy.name}
        )
        
        return True
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete policy"""
        if policy_id not in self.policies:
            return False
        
        del self.policies[policy_id]
        
        await self.audit_logger.log_admin_action(
            admin_id="system",
            action="delete_policy",
            target=policy_id,
            success=True
        )
        
        return True
    
    def _validate_policy(self, policy: Policy) -> List[str]:
        """Validate policy definition"""
        errors = []
        
        if not policy.policy_id:
            errors.append("Policy ID is required")
        
        if not policy.name:
            errors.append("Policy name is required")
        
        if not policy.target:
            errors.append("Policy target is required")
        
        if not policy.rules:
            errors.append("Policy must have at least one rule")
        
        # Validate rules
        for rule in policy.rules:
            if not rule.rule_id:
                errors.append(f"Rule ID is required for rule {rule.name}")
            
            if not rule.name:
                errors.append(f"Rule name is required for rule {rule.rule_id}")
            
            # Validate conditions
            for condition in rule.conditions:
                if not condition.attribute:
                    errors.append(f"Condition attribute is required in rule {rule.rule_id}")
                
                if not condition.operator:
                    errors.append(f"Condition operator is required in rule {rule.rule_id}")
        
        return errors
    
    def get_policy_metrics(self) -> Dict[str, Any]:
        """Get policy engine performance metrics"""
        return {
            "total_policies": len(self.policies),
            "policies_by_type": {
                policy_type.value: len([
                    p for p in self.policies.values() 
                    if p.policy_type == policy_type
                ]) for policy_type in PolicyType
            },
            "evaluation_metrics": self.evaluation_metrics.copy(),
            "enabled_policies": len([p for p in self.policies.values() if p.enabled])
        }
    
    def list_policies(self, policy_type: Optional[PolicyType] = None) -> List[Dict[str, Any]]:
        """List all policies or by type"""
        policies = self.policies.values()
        
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        return [
            {
                "policy_id": p.policy_id,
                "name": p.name,
                "description": p.description,
                "policy_type": p.policy_type.value,
                "target": p.target,
                "enabled": p.enabled,
                "rules_count": len(p.rules),
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None
            }
            for p in policies
        ]

# Global policy engine instance
_global_policy_engine = None

def get_policy_engine() -> PolicyEngine:
    """Get global policy engine instance"""
    global _global_policy_engine
    if _global_policy_engine is None:
        _global_policy_engine = PolicyEngine()
    return _global_policy_engine
