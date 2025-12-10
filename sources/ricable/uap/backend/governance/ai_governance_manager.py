# File: backend/governance/ai_governance_manager.py
"""
AI Governance Manager

Central coordination system for AI governance across the UAP platform.
Integrates with compliance, security, and monitoring systems to provide
comprehensive AI governance capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid

# Import existing systems
from ..compliance.compliance_manager import get_compliance_manager, ComplianceFramework
from ..security.audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger

# Import AI explainability system
from ..ai.explainability import get_ai_explainability_engine, BiasType, SeverityLevel

logger = logging.getLogger(__name__)

class GovernancePolicy(Enum):
    """AI Governance policy types"""
    MODEL_APPROVAL_REQUIRED = "model_approval_required"
    BIAS_TESTING_MANDATORY = "bias_testing_mandatory"
    EXPLAINABILITY_REQUIRED = "explainability_required"
    DATA_LINEAGE_TRACKING = "data_lineage_tracking"
    REGULAR_AUDITS_REQUIRED = "regular_audits_required"
    HUMAN_OVERSIGHT_REQUIRED = "human_oversight_required"
    PERFORMANCE_MONITORING = "performance_monitoring"
    ETHICAL_REVIEW_REQUIRED = "ethical_review_required"

class ModelRiskLevel(Enum):
    """Risk levels for AI models"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class GovernanceStatus(Enum):
    """Governance compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    REQUIRES_ACTION = "requires_action"
    EXEMPT = "exempt"

@dataclass
class AIModelRegistry:
    """Registry entry for an AI model"""
    model_id: str
    model_name: str
    framework: str
    version: str
    owner: str
    description: str
    risk_level: ModelRiskLevel
    use_cases: List[str]
    data_sources: List[str]
    governance_status: GovernanceStatus
    compliance_frameworks: List[str]
    last_audit_date: Optional[datetime]
    next_audit_due: Optional[datetime]
    approval_status: str
    approved_by: Optional[str]
    deployment_date: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class GovernancePolicyRule:
    """Governance policy rule definition"""
    rule_id: str
    policy_type: GovernancePolicy
    title: str
    description: str
    applicable_risk_levels: List[ModelRiskLevel]
    applicable_frameworks: List[str]
    requirements: List[str]
    enforcement_level: str  # mandatory, recommended, optional
    violation_consequences: List[str]
    review_frequency: timedelta
    created_by: str
    created_at: datetime
    active: bool

@dataclass
class GovernanceViolation:
    """Governance policy violation"""
    violation_id: str
    model_id: str
    rule_id: str
    violation_type: str
    severity: SeverityLevel
    description: str
    detected_at: datetime
    detected_by: str
    status: str  # open, investigating, resolved, closed
    resolution_plan: Optional[str]
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class AIGovernanceMetrics:
    """Comprehensive AI governance metrics"""
    total_models: int
    models_by_risk_level: Dict[str, int]
    models_by_status: Dict[str, int]
    compliance_rate: float
    overdue_audits: int
    open_violations: int
    critical_violations: int
    avg_bias_score: float
    models_with_explainability: int
    governance_coverage: float
    last_updated: datetime

class AIGovernanceManager:
    """
    Central AI Governance Management System
    
    Provides comprehensive governance capabilities including:
    - Model registry and lifecycle management
    - Policy enforcement and compliance tracking
    - Risk assessment and mitigation
    - Audit coordination and reporting
    - Integration with compliance frameworks
    """
    
    def __init__(self):
        self.model_registry: Dict[str, AIModelRegistry] = {}
        self.governance_policies: Dict[str, GovernancePolicyRule] = {}
        self.violations: List[GovernanceViolation] = []
        
        # Integration with existing systems
        self.compliance_manager = get_compliance_manager()
        self.explainability_engine = get_ai_explainability_engine()
        self.audit_logger = get_security_audit_logger()
        
        # Configuration
        self.governance_config = {
            "default_audit_frequency": timedelta(days=90),
            "high_risk_audit_frequency": timedelta(days=30),
            "critical_risk_audit_frequency": timedelta(days=7),
            "automatic_policy_enforcement": True,
            "require_human_approval": True,
            "enable_continuous_monitoring": True
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        # Start background monitoring
        asyncio.create_task(self._continuous_governance_monitoring())
        
        logger.info("AI Governance Manager initialized")
    
    def _initialize_default_policies(self):
        """Initialize default governance policies"""
        
        # Model Approval Policy
        self.governance_policies["model_approval"] = GovernancePolicyRule(
            rule_id="model_approval",
            policy_type=GovernancePolicy.MODEL_APPROVAL_REQUIRED,
            title="Model Approval Required",
            description="All AI models must be approved before deployment",
            applicable_risk_levels=[ModelRiskLevel.MEDIUM, ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL],
            applicable_frameworks=["copilot", "agno", "mastra", "mlx"],
            requirements=[
                "Model documentation complete",
                "Risk assessment conducted",
                "Security review passed",
                "Stakeholder approval obtained"
            ],
            enforcement_level="mandatory",
            violation_consequences=[
                "Model deployment blocked",
                "Compliance violation recorded",
                "Stakeholder notification triggered"
            ],
            review_frequency=timedelta(days=30),
            created_by="system",
            created_at=datetime.now(timezone.utc),
            active=True
        )
        
        # Bias Testing Policy
        self.governance_policies["bias_testing"] = GovernancePolicyRule(
            rule_id="bias_testing",
            policy_type=GovernancePolicy.BIAS_TESTING_MANDATORY,
            title="Bias Testing Mandatory",
            description="Regular bias testing required for all AI models",
            applicable_risk_levels=[ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL],
            applicable_frameworks=["copilot", "agno", "mastra", "mlx"],
            requirements=[
                "Demographic parity testing",
                "Equalized odds assessment",
                "Calibration bias evaluation",
                "Bias mitigation strategies documented"
            ],
            enforcement_level="mandatory",
            violation_consequences=[
                "Model flagged for review",
                "Increased monitoring required",
                "Bias mitigation plan mandatory"
            ],
            review_frequency=timedelta(days=14),
            created_by="system",
            created_at=datetime.now(timezone.utc),
            active=True
        )
        
        # Explainability Policy
        self.governance_policies["explainability"] = GovernancePolicyRule(
            rule_id="explainability",
            policy_type=GovernancePolicy.EXPLAINABILITY_REQUIRED,
            title="Explainability Required",
            description="AI decisions must be explainable and auditable",
            applicable_risk_levels=[ModelRiskLevel.MEDIUM, ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL],
            applicable_frameworks=["copilot", "agno", "mastra", "mlx"],
            requirements=[
                "Explanation capability implemented",
                "Decision transparency documented",
                "Stakeholder explanation training completed"
            ],
            enforcement_level="mandatory",
            violation_consequences=[
                "Model usage restricted",
                "Additional documentation required"
            ],
            review_frequency=timedelta(days=30),
            created_by="system",
            created_at=datetime.now(timezone.utc),
            active=True
        )
        
        # Data Lineage Policy
        self.governance_policies["data_lineage"] = GovernancePolicyRule(
            rule_id="data_lineage",
            policy_type=GovernancePolicy.DATA_LINEAGE_TRACKING,
            title="Data Lineage Tracking",
            description="Complete data lineage must be tracked and documented",
            applicable_risk_levels=[ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL],
            applicable_frameworks=["copilot", "agno", "mastra", "mlx"],
            requirements=[
                "Data sources documented",
                "Data processing pipeline tracked",
                "Data quality metrics recorded",
                "Data governance compliance verified"
            ],
            enforcement_level="mandatory",
            violation_consequences=[
                "Data usage audit triggered",
                "Compliance review required"
            ],
            review_frequency=timedelta(days=60),
            created_by="system",
            created_at=datetime.now(timezone.utc),
            active=True
        )
    
    async def register_ai_model(self, model_name: str, framework: str, version: str,
                              owner: str, description: str, use_cases: List[str],
                              data_sources: List[str], risk_level: ModelRiskLevel = None) -> str:
        """
        Register a new AI model in the governance system.
        
        Args:
            model_name: Name of the AI model
            framework: Framework used (copilot, agno, mastra, mlx)
            version: Model version
            owner: Model owner/responsible party
            description: Model description
            use_cases: List of intended use cases
            data_sources: List of data sources used
            risk_level: Risk level (auto-assessed if not provided)
            
        Returns:
            Model ID for the registered model
        """
        try:
            model_id = f"model_{framework}_{model_name}_{uuid.uuid4().hex[:8]}"
            
            # Auto-assess risk level if not provided
            if risk_level is None:
                risk_level = await self._assess_model_risk(
                    model_name, framework, use_cases, data_sources
                )
            
            # Determine applicable compliance frameworks
            compliance_frameworks = self._determine_compliance_frameworks(risk_level, use_cases)
            
            # Calculate next audit date
            audit_frequency = self._get_audit_frequency(risk_level)
            next_audit_due = datetime.now(timezone.utc) + audit_frequency
            
            # Create registry entry
            registry_entry = AIModelRegistry(
                model_id=model_id,
                model_name=model_name,
                framework=framework,
                version=version,
                owner=owner,
                description=description,
                risk_level=risk_level,
                use_cases=use_cases,
                data_sources=data_sources,
                governance_status=GovernanceStatus.UNDER_REVIEW,
                compliance_frameworks=compliance_frameworks,
                last_audit_date=None,
                next_audit_due=next_audit_due,
                approval_status="pending",
                approved_by=None,
                deployment_date=None,
                metadata={
                    "registration_source": "governance_manager",
                    "auto_risk_assessment": risk_level is None
                },
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Store in registry
            self.model_registry[model_id] = registry_entry
            
            # Log registration
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.SUCCESS,
                actor_id=owner,
                actor_type="user",
                resource=f"ai_model:{model_id}",
                action="register_model",
                description=f"AI model {model_name} registered in governance system",
                details={
                    "model_id": model_id,
                    "model_name": model_name,
                    "framework": framework,
                    "risk_level": risk_level.value,
                    "use_cases": use_cases
                },
                risk_score=4
            )
            
            # Trigger initial governance checks
            await self._trigger_initial_governance_checks(model_id)
            
            uap_logger.log_ai_event(
                f"AI model registered: {model_name}",
                model=model_name,
                framework=framework,
                metadata={
                    "model_id": model_id,
                    "risk_level": risk_level.value,
                    "owner": owner
                }
            )
            
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register AI model: {e}")
            
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.FAILURE,
                actor_id=owner,
                actor_type="user",
                resource="ai_governance_system",
                action="register_model",
                description=f"Failed to register AI model: {str(e)}",
                details={"error": str(e)},
                risk_score=6
            )
            
            raise
    
    async def assess_model_compliance(self, model_id: str) -> Dict[str, Any]:
        """
        Assess a model's compliance with governance policies.
        
        Args:
            model_id: ID of the model to assess
            
        Returns:
            Comprehensive compliance assessment
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            model = self.model_registry[model_id]
            assessment_id = f"assess_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting compliance assessment {assessment_id} for model {model.model_name}")
            
            # Check policy compliance
            policy_compliance = {}
            overall_compliant = True
            violations_found = []
            
            for policy_id, policy in self.governance_policies.items():
                if not policy.active:
                    continue
                
                # Check if policy applies to this model
                if (model.risk_level in policy.applicable_risk_levels and
                    model.framework in policy.applicable_frameworks):
                    
                    compliance_result = await self._check_policy_compliance(model, policy)
                    policy_compliance[policy_id] = compliance_result
                    
                    if not compliance_result["compliant"]:
                        overall_compliant = False
                        violations_found.extend(compliance_result.get("violations", []))
            
            # Perform bias assessment if required
            bias_assessment = None
            if model.risk_level in [ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL]:
                bias_assessment = await self._perform_bias_assessment(model)
            
            # Check explainability requirements
            explainability_assessment = await self._check_explainability_requirements(model)
            
            # Assess data lineage compliance
            data_lineage_assessment = await self._assess_data_lineage_compliance(model)
            
            # Calculate overall compliance score
            compliance_score = self._calculate_compliance_score(
                policy_compliance, bias_assessment, explainability_assessment, data_lineage_assessment
            )
            
            # Update model status
            if overall_compliant and compliance_score >= 0.8:
                model.governance_status = GovernanceStatus.COMPLIANT
            elif compliance_score >= 0.6:
                model.governance_status = GovernanceStatus.REQUIRES_ACTION
            else:
                model.governance_status = GovernanceStatus.NON_COMPLIANT
            
            model.last_audit_date = datetime.now(timezone.utc)
            model.next_audit_due = datetime.now(timezone.utc) + self._get_audit_frequency(model.risk_level)
            model.updated_at = datetime.now(timezone.utc)
            
            # Record violations
            for violation_data in violations_found:
                await self._record_governance_violation(model_id, violation_data)
            
            assessment_result = {
                "assessment_id": assessment_id,
                "model_id": model_id,
                "model_name": model.model_name,
                "assessment_date": datetime.now(timezone.utc).isoformat(),
                "overall_compliant": overall_compliant,
                "compliance_score": compliance_score,
                "governance_status": model.governance_status.value,
                "policy_compliance": policy_compliance,
                "bias_assessment": bias_assessment,
                "explainability_assessment": explainability_assessment,
                "data_lineage_assessment": data_lineage_assessment,
                "violations_found": len(violations_found),
                "recommendations": self._generate_compliance_recommendations(
                    model, policy_compliance, violations_found
                ),
                "next_audit_due": model.next_audit_due.isoformat()
            }
            
            # Log assessment completion
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.SUCCESS,
                actor_id="ai_governance_manager",
                actor_type="system",
                resource=f"ai_model:{model_id}",
                action="assess_compliance",
                description=f"Compliance assessment completed for {model.model_name}",
                details={
                    "assessment_id": assessment_id,
                    "compliance_score": compliance_score,
                    "violations_found": len(violations_found),
                    "overall_compliant": overall_compliant
                },
                risk_score=3 if overall_compliant else 7
            )
            
            return assessment_result
            
        except Exception as e:
            logger.error(f"Failed to assess model compliance: {e}")
            raise
    
    async def approve_model_deployment(self, model_id: str, approved_by: str,
                                     approval_conditions: List[str] = None) -> Dict[str, Any]:
        """
        Approve a model for deployment after governance review.
        
        Args:
            model_id: ID of the model to approve
            approved_by: Person/system approving the model
            approval_conditions: Any conditions for the approval
            
        Returns:
            Approval result
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.model_registry[model_id]
            
            # Check if model meets approval requirements
            compliance_assessment = await self.assess_model_compliance(model_id)
            
            if not compliance_assessment["overall_compliant"]:
                return {
                    "approved": False,
                    "reason": "Model does not meet compliance requirements",
                    "compliance_score": compliance_assessment["compliance_score"],
                    "violations_count": compliance_assessment["violations_found"],
                    "required_actions": compliance_assessment["recommendations"]
                }
            
            # Approve model
            model.approval_status = "approved"
            model.approved_by = approved_by
            model.deployment_date = datetime.now(timezone.utc)
            model.governance_status = GovernanceStatus.COMPLIANT
            model.updated_at = datetime.now(timezone.utc)
            
            # Add approval conditions to metadata
            if approval_conditions:
                model.metadata["approval_conditions"] = approval_conditions
            
            approval_result = {
                "approved": True,
                "model_id": model_id,
                "model_name": model.model_name,
                "approved_by": approved_by,
                "approval_date": model.deployment_date.isoformat(),
                "approval_conditions": approval_conditions or [],
                "compliance_score": compliance_assessment["compliance_score"],
                "next_review_due": model.next_audit_due.isoformat()
            }
            
            # Log approval
            await self.audit_logger.log_event(
                event_type=AuditEventType.ADMIN_ACTION,
                outcome=AuditOutcome.SUCCESS,
                actor_id=approved_by,
                actor_type="admin",
                resource=f"ai_model:{model_id}",
                action="approve_deployment",
                description=f"AI model {model.model_name} approved for deployment",
                details={
                    "model_id": model_id,
                    "compliance_score": compliance_assessment["compliance_score"],
                    "approval_conditions": approval_conditions or []
                },
                risk_score=4
            )
            
            uap_logger.log_ai_event(
                f"AI model approved: {model.model_name}",
                model=model.model_name,
                framework=model.framework,
                metadata={
                    "approved_by": approved_by,
                    "compliance_score": compliance_assessment["compliance_score"]
                }
            )
            
            return approval_result
            
        except Exception as e:
            logger.error(f"Failed to approve model deployment: {e}")
            raise
    
    async def get_governance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive governance dashboard data.
        
        Returns:
            Dashboard data with metrics and status information
        """
        try:
            # Calculate metrics
            total_models = len(self.model_registry)
            
            models_by_risk = {
                "low": len([m for m in self.model_registry.values() if m.risk_level == ModelRiskLevel.LOW]),
                "medium": len([m for m in self.model_registry.values() if m.risk_level == ModelRiskLevel.MEDIUM]),
                "high": len([m for m in self.model_registry.values() if m.risk_level == ModelRiskLevel.HIGH]),
                "critical": len([m for m in self.model_registry.values() if m.risk_level == ModelRiskLevel.CRITICAL])
            }
            
            models_by_status = {
                "compliant": len([m for m in self.model_registry.values() if m.governance_status == GovernanceStatus.COMPLIANT]),
                "non_compliant": len([m for m in self.model_registry.values() if m.governance_status == GovernanceStatus.NON_COMPLIANT]),
                "under_review": len([m for m in self.model_registry.values() if m.governance_status == GovernanceStatus.UNDER_REVIEW]),
                "requires_action": len([m for m in self.model_registry.values() if m.governance_status == GovernanceStatus.REQUIRES_ACTION])
            }
            
            # Calculate compliance rate
            compliant_models = models_by_status["compliant"]
            compliance_rate = compliant_models / total_models if total_models > 0 else 0
            
            # Count overdue audits
            overdue_audits = len([
                m for m in self.model_registry.values()
                if m.next_audit_due and m.next_audit_due < datetime.now(timezone.utc)
            ])
            
            # Count violations
            open_violations = len([v for v in self.violations if v.status == "open"])
            critical_violations = len([
                v for v in self.violations 
                if v.status == "open" and v.severity == SeverityLevel.CRITICAL
            ])
            
            # Get recent activities
            recent_registrations = [
                {
                    "model_id": m.model_id,
                    "model_name": m.model_name,
                    "framework": m.framework,
                    "risk_level": m.risk_level.value,
                    "created_at": m.created_at.isoformat()
                }
                for m in sorted(
                    self.model_registry.values(),
                    key=lambda x: x.created_at,
                    reverse=True
                )[:10]
            ]
            
            dashboard_data = {
                "dashboard_id": f"gov_dash_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "overview": {
                    "total_models": total_models,
                    "compliance_rate": compliance_rate,
                    "overdue_audits": overdue_audits,
                    "open_violations": open_violations,
                    "critical_violations": critical_violations
                },
                "models_by_risk_level": models_by_risk,
                "models_by_status": models_by_status,
                "active_policies": len([p for p in self.governance_policies.values() if p.active]),
                "frameworks_governed": len(set([m.framework for m in self.model_registry.values()])),
                "recent_activities": {
                    "recent_registrations": recent_registrations,
                    "recent_violations": [
                        {
                            "violation_id": v.violation_id,
                            "model_id": v.model_id,
                            "violation_type": v.violation_type,
                            "severity": v.severity.value,
                            "detected_at": v.detected_at.isoformat()
                        }
                        for v in sorted(self.violations, key=lambda x: x.detected_at, reverse=True)[:10]
                    ]
                },
                "alerts": self._generate_governance_alerts(),
                "recommendations": self._generate_dashboard_recommendations()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate governance dashboard: {e}")
            raise
    
    # Internal helper methods
    
    async def _assess_model_risk(self, model_name: str, framework: str,
                               use_cases: List[str], data_sources: List[str]) -> ModelRiskLevel:
        """Assess risk level for a model"""
        risk_score = 0
        
        # Risk factors based on use cases
        high_risk_use_cases = [
            "financial_decisions", "medical_diagnosis", "legal_analysis",
            "hiring_decisions", "credit_scoring", "law_enforcement"
        ]
        
        medium_risk_use_cases = [
            "content_moderation", "recommendation_systems", "customer_service",
            "data_analysis", "process_automation"
        ]
        
        for use_case in use_cases:
            if any(high_risk in use_case.lower() for high_risk in high_risk_use_cases):
                risk_score += 3
            elif any(medium_risk in use_case.lower() for medium_risk in medium_risk_use_cases):
                risk_score += 2
            else:
                risk_score += 1
        
        # Risk factors based on data sources
        sensitive_data_sources = [
            "personal_data", "medical_records", "financial_data",
            "biometric_data", "location_data"
        ]
        
        for data_source in data_sources:
            if any(sensitive in data_source.lower() for sensitive in sensitive_data_sources):
                risk_score += 2
        
        # Determine risk level
        if risk_score >= 8:
            return ModelRiskLevel.CRITICAL
        elif risk_score >= 5:
            return ModelRiskLevel.HIGH
        elif risk_score >= 3:
            return ModelRiskLevel.MEDIUM
        else:
            return ModelRiskLevel.LOW
    
    def _determine_compliance_frameworks(self, risk_level: ModelRiskLevel,
                                       use_cases: List[str]) -> List[str]:
        """Determine applicable compliance frameworks"""
        frameworks = []
        
        # SOC 2 applies to all models
        frameworks.append("SOC2")
        
        # GDPR for personal data processing
        if any("personal" in use_case.lower() or "user" in use_case.lower() 
               for use_case in use_cases):
            frameworks.append("GDPR")
        
        # HIPAA for healthcare use cases
        if any("medical" in use_case.lower() or "health" in use_case.lower()
               for use_case in use_cases):
            frameworks.append("HIPAA")
        
        # Additional frameworks for high-risk models
        if risk_level in [ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL]:
            frameworks.append("ISO27001")
        
        return frameworks
    
    def _get_audit_frequency(self, risk_level: ModelRiskLevel) -> timedelta:
        """Get audit frequency based on risk level"""
        frequency_map = {
            ModelRiskLevel.LOW: self.governance_config["default_audit_frequency"],
            ModelRiskLevel.MEDIUM: self.governance_config["default_audit_frequency"],
            ModelRiskLevel.HIGH: self.governance_config["high_risk_audit_frequency"],
            ModelRiskLevel.CRITICAL: self.governance_config["critical_risk_audit_frequency"]
        }
        return frequency_map.get(risk_level, self.governance_config["default_audit_frequency"])
    
    async def _trigger_initial_governance_checks(self, model_id: str):
        """Trigger initial governance checks for a newly registered model"""
        try:
            # Schedule compliance assessment
            asyncio.create_task(self._schedule_compliance_assessment(model_id))
            
            # Schedule bias testing if required
            model = self.model_registry[model_id]
            if model.risk_level in [ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL]:
                asyncio.create_task(self._schedule_bias_testing(model_id))
            
        except Exception as e:
            logger.error(f"Failed to trigger initial governance checks: {e}")
    
    async def _schedule_compliance_assessment(self, model_id: str):
        """Schedule compliance assessment for a model"""
        # This would be implemented as part of a job scheduler
        # For now, log the scheduling
        logger.info(f"Compliance assessment scheduled for model {model_id}")
    
    async def _schedule_bias_testing(self, model_id: str):
        """Schedule bias testing for a model"""
        logger.info(f"Bias testing scheduled for model {model_id}")
    
    async def _check_policy_compliance(self, model: AIModelRegistry,
                                     policy: GovernancePolicyRule) -> Dict[str, Any]:
        """Check compliance with a specific policy"""
        # Mock implementation - would integrate with actual policy checking
        compliant = True
        violations = []
        details = {}
        
        if policy.policy_type == GovernancePolicy.MODEL_APPROVAL_REQUIRED:
            compliant = model.approval_status == "approved"
            if not compliant:
                violations.append({
                    "type": "missing_approval",
                    "description": "Model not approved for deployment"
                })
        
        elif policy.policy_type == GovernancePolicy.BIAS_TESTING_MANDATORY:
            # Check if bias testing has been performed
            last_bias_test = model.metadata.get("last_bias_test")
            if not last_bias_test:
                compliant = False
                violations.append({
                    "type": "missing_bias_test",
                    "description": "Bias testing not performed"
                })
        
        return {
            "policy_id": policy.rule_id,
            "compliant": compliant,
            "violations": violations,
            "details": details,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _perform_bias_assessment(self, model: AIModelRegistry) -> Dict[str, Any]:
        """Perform bias assessment for a model"""
        # This would integrate with the explainability engine
        # For now, return mock assessment
        return {
            "bias_tested": True,
            "bias_score": 0.15,
            "bias_types_detected": ["demographic_parity"],
            "severity": "medium",
            "recommendations": ["Implement bias mitigation strategies"]
        }
    
    async def _check_explainability_requirements(self, model: AIModelRegistry) -> Dict[str, Any]:
        """Check explainability requirements for a model"""
        return {
            "explainability_required": model.risk_level in [ModelRiskLevel.MEDIUM, ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL],
            "explainability_implemented": True,  # Mock
            "explanation_methods_available": ["feature_importance", "attention_visualization"],
            "compliant": True
        }
    
    async def _assess_data_lineage_compliance(self, model: AIModelRegistry) -> Dict[str, Any]:
        """Assess data lineage compliance"""
        return {
            "data_lineage_required": model.risk_level in [ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL],
            "data_sources_documented": len(model.data_sources) > 0,
            "lineage_complete": True,  # Mock
            "compliant": True
        }
    
    def _calculate_compliance_score(self, policy_compliance: Dict[str, Any],
                                  bias_assessment: Dict[str, Any],
                                  explainability_assessment: Dict[str, Any],
                                  data_lineage_assessment: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        scores = []
        
        # Policy compliance score
        policy_scores = [p["compliant"] for p in policy_compliance.values()]
        if policy_scores:
            scores.append(sum(policy_scores) / len(policy_scores))
        
        # Bias assessment score
        if bias_assessment and "bias_score" in bias_assessment:
            bias_score = 1.0 - bias_assessment["bias_score"]  # Lower bias = higher score
            scores.append(max(0, bias_score))
        
        # Explainability score
        if explainability_assessment and explainability_assessment.get("compliant"):
            scores.append(1.0)
        
        # Data lineage score
        if data_lineage_assessment and data_lineage_assessment.get("compliant"):
            scores.append(1.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _record_governance_violation(self, model_id: str, violation_data: Dict[str, Any]):
        """Record a governance violation"""
        violation = GovernanceViolation(
            violation_id=f"viol_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(self.violations)}",
            model_id=model_id,
            rule_id=violation_data.get("rule_id", "unknown"),
            violation_type=violation_data.get("type", "policy_violation"),
            severity=SeverityLevel(violation_data.get("severity", "medium")),
            description=violation_data.get("description", "Governance policy violation"),
            detected_at=datetime.now(timezone.utc),
            detected_by="ai_governance_manager",
            status="open",
            resolution_plan=None,
            resolved_at=None,
            resolved_by=None,
            metadata=violation_data
        )
        
        self.violations.append(violation)
        
        # Log violation
        await self.audit_logger.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            outcome=AuditOutcome.SUCCESS,
            actor_id="ai_governance_manager",
            actor_type="system",
            resource=f"ai_model:{model_id}",
            action="record_violation",
            description=f"Governance violation recorded: {violation.violation_type}",
            details={
                "violation_id": violation.violation_id,
                "violation_type": violation.violation_type,
                "severity": violation.severity.value
            },
            risk_score=8 if violation.severity == SeverityLevel.CRITICAL else 6
        )
    
    def _generate_compliance_recommendations(self, model: AIModelRegistry,
                                           policy_compliance: Dict[str, Any],
                                           violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if violations:
            recommendations.append(f"Address {len(violations)} governance violations")
        
        non_compliant_policies = [
            p for p in policy_compliance.values() if not p["compliant"]
        ]
        
        if non_compliant_policies:
            recommendations.append("Review and address non-compliant policies")
        
        if model.risk_level in [ModelRiskLevel.HIGH, ModelRiskLevel.CRITICAL]:
            recommendations.extend([
                "Implement enhanced monitoring for high-risk model",
                "Conduct regular bias assessments",
                "Ensure explainability mechanisms are in place"
            ])
        
        recommendations.extend([
            "Maintain up-to-date documentation",
            "Schedule regular governance reviews",
            "Monitor model performance and fairness metrics"
        ])
        
        return recommendations
    
    def _generate_governance_alerts(self) -> List[Dict[str, Any]]:
        """Generate governance alerts for dashboard"""
        alerts = []
        
        # Critical violations
        critical_violations = [v for v in self.violations if v.severity == SeverityLevel.CRITICAL and v.status == "open"]
        if critical_violations:
            alerts.append({
                "type": "critical_violation",
                "severity": "critical",
                "message": f"{len(critical_violations)} critical governance violations require immediate attention",
                "count": len(critical_violations)
            })
        
        # Overdue audits
        overdue_models = [
            m for m in self.model_registry.values()
            if m.next_audit_due and m.next_audit_due < datetime.now(timezone.utc)
        ]
        if overdue_models:
            alerts.append({
                "type": "overdue_audit",
                "severity": "high",
                "message": f"{len(overdue_models)} models have overdue governance audits",
                "count": len(overdue_models)
            })
        
        # Non-compliant models
        non_compliant_models = [
            m for m in self.model_registry.values()
            if m.governance_status == GovernanceStatus.NON_COMPLIANT
        ]
        if non_compliant_models:
            alerts.append({
                "type": "non_compliant",
                "severity": "medium",
                "message": f"{len(non_compliant_models)} models are non-compliant",
                "count": len(non_compliant_models)
            })
        
        return alerts
    
    def _generate_dashboard_recommendations(self) -> List[str]:
        """Generate recommendations for governance dashboard"""
        recommendations = []
        
        total_models = len(self.model_registry)
        compliant_models = len([
            m for m in self.model_registry.values()
            if m.governance_status == GovernanceStatus.COMPLIANT
        ])
        
        compliance_rate = compliant_models / total_models if total_models > 0 else 0
        
        if compliance_rate < 0.8:
            recommendations.append("Improve overall compliance rate through targeted interventions")
        
        if any(v.severity == SeverityLevel.CRITICAL for v in self.violations if v.status == "open"):
            recommendations.append("Address critical governance violations immediately")
        
        recommendations.extend([
            "Maintain regular governance review cycles",
            "Implement continuous monitoring of AI model performance",
            "Provide governance training to model owners",
            "Update governance policies based on emerging best practices"
        ])
        
        return recommendations
    
    async def _continuous_governance_monitoring(self):
        """Continuous monitoring of AI governance"""
        while True:
            try:
                # Check for overdue audits
                await self._check_overdue_audits()
                
                # Monitor compliance status
                await self._monitor_compliance_status()
                
                # Check for policy violations
                await self._check_policy_violations()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in continuous governance monitoring: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _check_overdue_audits(self):
        """Check for overdue audits"""
        now = datetime.now(timezone.utc)
        overdue_models = [
            m for m in self.model_registry.values()
            if m.next_audit_due and m.next_audit_due < now
        ]
        
        for model in overdue_models:
            logger.warning(f"Model {model.model_name} has overdue audit")
            # Would trigger audit scheduling in production
    
    async def _monitor_compliance_status(self):
        """Monitor overall compliance status"""
        total_models = len(self.model_registry)
        if total_models == 0:
            return
        
        compliant_models = len([
            m for m in self.model_registry.values()
            if m.governance_status == GovernanceStatus.COMPLIANT
        ])
        
        compliance_rate = compliant_models / total_models
        
        if compliance_rate < 0.7:  # Below 70% compliance
            logger.warning(f"Low governance compliance rate: {compliance_rate:.2%}")
    
    async def _check_policy_violations(self):
        """Check for new policy violations"""
        open_violations = [v for v in self.violations if v.status == "open"]
        critical_violations = [v for v in open_violations if v.severity == SeverityLevel.CRITICAL]
        
        if critical_violations:
            logger.critical(f"{len(critical_violations)} critical governance violations detected")

# Global instance
_global_ai_governance_manager = None

def get_ai_governance_manager() -> AIGovernanceManager:
    """Get global AI governance manager instance"""
    global _global_ai_governance_manager
    if _global_ai_governance_manager is None:
        _global_ai_governance_manager = AIGovernanceManager()
    return _global_ai_governance_manager