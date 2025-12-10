# File: backend/governance/risk_assessment.py
"""
AI Risk Assessment System

Comprehensive risk assessment and mitigation framework for AI models.
Evaluates technical, ethical, operational, and compliance risks.
"""

import asyncio
import json
import logging
import numpy as np
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
import math

# Import existing systems
from ..security.audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger
from ..compliance.compliance_manager import get_compliance_manager

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Categories of AI risks"""
    TECHNICAL_RISK = "technical_risk"
    ETHICAL_RISK = "ethical_risk"
    OPERATIONAL_RISK = "operational_risk"
    COMPLIANCE_RISK = "compliance_risk"
    SECURITY_RISK = "security_risk"
    BUSINESS_RISK = "business_risk"
    REPUTATIONAL_RISK = "reputational_risk"
    LEGAL_RISK = "legal_risk"

class RiskLevel(Enum):
    """Risk severity levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class RiskStatus(Enum):
    """Risk mitigation status"""
    IDENTIFIED = "identified"
    UNDER_ASSESSMENT = "under_assessment"
    MITIGATION_PLANNED = "mitigation_planned"
    MITIGATION_IN_PROGRESS = "mitigation_in_progress"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    TRANSFERRED = "transferred"
    AVOIDED = "avoided"

class ImpactArea(Enum):
    """Areas that can be impacted by AI risks"""
    USER_SAFETY = "user_safety"
    DATA_PRIVACY = "data_privacy"
    BUSINESS_OPERATIONS = "business_operations"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    FINANCIAL_PERFORMANCE = "financial_performance"
    REPUTATION = "reputation"
    STAKEHOLDER_TRUST = "stakeholder_trust"
    SYSTEM_SECURITY = "system_security"

@dataclass
class RiskFactor:
    """Individual risk factor"""
    factor_id: str
    name: str
    description: str
    category: RiskCategory
    likelihood: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 10.0
    risk_score: float  # likelihood * impact_score
    impact_areas: List[ImpactArea]
    indicators: List[str]
    evidence: Dict[str, Any]
    detection_confidence: float
    metadata: Dict[str, Any]

@dataclass
class RiskMitigationStrategy:
    """Risk mitigation strategy"""
    strategy_id: str
    risk_id: str
    strategy_type: str  # preventive, detective, corrective, compensating
    title: str
    description: str
    implementation_steps: List[str]
    cost_estimate: Optional[float]
    timeline_days: int
    effectiveness_score: float  # 0.0 to 1.0
    responsible_party: str
    dependencies: List[str]
    monitoring_metrics: List[str]
    success_criteria: List[str]
    implemented: bool
    implementation_date: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class AIRiskProfile:
    """Comprehensive risk profile for an AI model"""
    profile_id: str
    model_id: str
    model_name: str
    framework: str
    assessment_date: datetime
    assessor: str
    risk_factors: List[RiskFactor]
    overall_risk_score: float
    risk_level: RiskLevel
    risk_distribution: Dict[str, float]  # by category
    mitigation_strategies: List[RiskMitigationStrategy]
    residual_risk_score: float
    risk_appetite_threshold: float
    recommendations: List[str]
    next_assessment_due: datetime
    metadata: Dict[str, Any]

@dataclass
class RiskScenario:
    """Risk scenario analysis"""
    scenario_id: str
    name: str
    description: str
    probability: float
    impact_assessment: Dict[str, Any]
    cascading_effects: List[str]
    mitigation_requirements: List[str]
    worst_case_impact: str
    realistic_impact: str
    best_case_impact: str

class AIRiskAssessment:
    """
    Comprehensive AI Risk Assessment System
    
    Provides systematic risk identification, assessment, and mitigation
    planning for AI models across multiple risk dimensions:
    - Technical risks (performance, reliability, robustness)
    - Ethical risks (bias, fairness, transparency)
    - Operational risks (deployment, maintenance, monitoring)
    - Compliance risks (regulatory, legal, policy)
    - Security risks (adversarial attacks, data breaches)
    - Business risks (financial, reputational, strategic)
    """
    
    def __init__(self):
        self.risk_profiles: Dict[str, AIRiskProfile] = {}
        self.risk_factors_database: Dict[str, RiskFactor] = {}
        self.mitigation_strategies_database: Dict[str, RiskMitigationStrategy] = {}
        
        # Integration with existing systems
        self.audit_logger = get_security_audit_logger()
        self.compliance_manager = get_compliance_manager()
        
        # Risk assessment configuration
        self.risk_config = {
            "default_risk_appetite": 0.3,  # 30% risk tolerance
            "high_risk_threshold": 0.7,
            "critical_risk_threshold": 0.9,
            "assessment_frequency_days": 90,
            "automated_assessment": True,
            "require_manual_review": True
        }
        
        # Initialize risk factor templates
        self._initialize_risk_factors_database()
        
        # Risk matrices and weights
        self.risk_weights = self._initialize_risk_weights()
        
        # Start continuous risk monitoring
        asyncio.create_task(self._continuous_risk_monitoring())
        
        logger.info("AI Risk Assessment system initialized")
    
    def _initialize_risk_factors_database(self):
        """Initialize database of common AI risk factors"""
        
        # Technical Risk Factors
        self.risk_factors_database.update({
            "model_performance_degradation": RiskFactor(
                factor_id="model_performance_degradation",
                name="Model Performance Degradation",
                description="Risk of model accuracy declining over time due to data drift",
                category=RiskCategory.TECHNICAL_RISK,
                likelihood=0.6,
                impact_score=7.0,
                risk_score=4.2,
                impact_areas=[ImpactArea.USER_SAFETY, ImpactArea.BUSINESS_OPERATIONS],
                indicators=["accuracy_decline", "increased_error_rate", "prediction_confidence_drop"],
                evidence={},
                detection_confidence=0.8,
                metadata={"common": True, "category": "performance"}
            ),
            
            "adversarial_vulnerability": RiskFactor(
                factor_id="adversarial_vulnerability",
                name="Adversarial Attack Vulnerability",
                description="Susceptibility to adversarial inputs designed to fool the model",
                category=RiskCategory.SECURITY_RISK,
                likelihood=0.4,
                impact_score=8.0,
                risk_score=3.2,
                impact_areas=[ImpactArea.SYSTEM_SECURITY, ImpactArea.USER_SAFETY],
                indicators=["unusual_input_patterns", "confidence_manipulation", "decision_boundary_exploitation"],
                evidence={},
                detection_confidence=0.7,
                metadata={"attack_type": "adversarial", "mitigation_priority": "high"}
            ),
            
            "data_privacy_breach": RiskFactor(
                factor_id="data_privacy_breach",
                name="Data Privacy Breach",
                description="Risk of exposing sensitive personal data through model inferences",
                category=RiskCategory.COMPLIANCE_RISK,
                likelihood=0.3,
                impact_score=9.0,
                risk_score=2.7,
                impact_areas=[ImpactArea.DATA_PRIVACY, ImpactArea.REGULATORY_COMPLIANCE, ImpactArea.REPUTATION],
                indicators=["membership_inference_success", "attribute_inference", "model_inversion_attacks"],
                evidence={},
                detection_confidence=0.6,
                metadata={"compliance_frameworks": ["GDPR", "HIPAA"], "severity": "high"}
            ),
            
            "algorithmic_bias": RiskFactor(
                factor_id="algorithmic_bias",
                name="Algorithmic Bias",
                description="Systematic bias in model decisions affecting protected groups",
                category=RiskCategory.ETHICAL_RISK,
                likelihood=0.5,
                impact_score=8.5,
                risk_score=4.25,
                impact_areas=[ImpactArea.STAKEHOLDER_TRUST, ImpactArea.REGULATORY_COMPLIANCE, ImpactArea.REPUTATION],
                indicators=["demographic_parity_violation", "equalized_odds_violation", "disparate_impact"],
                evidence={},
                detection_confidence=0.9,
                metadata={"bias_types": ["demographic", "outcome", "representation"], "priority": "critical"}
            ),
            
            "model_explainability_gap": RiskFactor(
                factor_id="model_explainability_gap",
                name="Model Explainability Gap",
                description="Inability to explain model decisions to stakeholders",
                category=RiskCategory.COMPLIANCE_RISK,
                likelihood=0.7,
                impact_score=6.0,
                risk_score=4.2,
                impact_areas=[ImpactArea.REGULATORY_COMPLIANCE, ImpactArea.STAKEHOLDER_TRUST],
                indicators=["no_explanation_capability", "poor_explanation_quality", "stakeholder_confusion"],
                evidence={},
                detection_confidence=0.95,
                metadata={"regulatory_requirement": True, "frameworks": ["GDPR", "AI_Act"]}
            ),
            
            "operational_dependency": RiskFactor(
                factor_id="operational_dependency",
                name="Operational Dependency Risk",
                description="Over-reliance on AI system for critical business operations",
                category=RiskCategory.OPERATIONAL_RISK,
                likelihood=0.6,
                impact_score=7.5,
                risk_score=4.5,
                impact_areas=[ImpactArea.BUSINESS_OPERATIONS, ImpactArea.FINANCIAL_PERFORMANCE],
                indicators=["single_point_of_failure", "no_fallback_procedures", "operational_coupling"],
                evidence={},
                detection_confidence=0.8,
                metadata={"business_critical": True, "requires_contingency": True}
            )
        })
    
    def _initialize_risk_weights(self) -> Dict[str, float]:
        """Initialize risk category weights"""
        return {
            RiskCategory.TECHNICAL_RISK.value: 0.20,
            RiskCategory.ETHICAL_RISK.value: 0.25,
            RiskCategory.OPERATIONAL_RISK.value: 0.15,
            RiskCategory.COMPLIANCE_RISK.value: 0.20,
            RiskCategory.SECURITY_RISK.value: 0.15,
            RiskCategory.BUSINESS_RISK.value: 0.05
        }
    
    async def assess_model_risk(self, model_id: str, model_name: str, framework: str,
                              use_cases: List[str], data_sources: List[str],
                              deployment_context: Dict[str, Any],
                              assessor: str = "automated_system") -> AIRiskProfile:
        """
        Conduct comprehensive risk assessment for an AI model.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            framework: AI framework used
            use_cases: List of model use cases
            data_sources: List of data sources used
            deployment_context: Context information about deployment
            assessor: Who is conducting the assessment
            
        Returns:
            Comprehensive risk profile for the model
        """
        try:
            profile_id = f"risk_profile_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting risk assessment {profile_id} for model {model_name}")
            
            # Identify applicable risk factors
            applicable_risks = await self._identify_applicable_risks(
                model_id, use_cases, data_sources, deployment_context
            )
            
            # Assess each risk factor
            assessed_risks = []
            for risk_template in applicable_risks:
                assessed_risk = await self._assess_risk_factor(
                    risk_template, model_id, use_cases, data_sources, deployment_context
                )
                assessed_risks.append(assessed_risk)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(assessed_risks)
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Calculate risk distribution by category
            risk_distribution = self._calculate_risk_distribution(assessed_risks)
            
            # Generate mitigation strategies
            mitigation_strategies = await self._generate_mitigation_strategies(assessed_risks)
            
            # Calculate residual risk after mitigation
            residual_risk_score = self._calculate_residual_risk(overall_risk_score, mitigation_strategies)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(assessed_risks, overall_risk_score)
            
            # Create risk profile
            risk_profile = AIRiskProfile(
                profile_id=profile_id,
                model_id=model_id,
                model_name=model_name,
                framework=framework,
                assessment_date=datetime.now(timezone.utc),
                assessor=assessor,
                risk_factors=assessed_risks,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                risk_distribution=risk_distribution,
                mitigation_strategies=mitigation_strategies,
                residual_risk_score=residual_risk_score,
                risk_appetite_threshold=self.risk_config["default_risk_appetite"],
                recommendations=recommendations,
                next_assessment_due=datetime.now(timezone.utc) + timedelta(days=self.risk_config["assessment_frequency_days"]),
                metadata={
                    "use_cases": use_cases,
                    "data_sources": data_sources,
                    "deployment_context": deployment_context,
                    "assessment_method": "comprehensive",
                    "automated": assessor == "automated_system"
                }
            )
            
            # Store risk profile
            self.risk_profiles[profile_id] = risk_profile
            
            # Log risk assessment
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.SUCCESS,
                actor_id=assessor,
                actor_type="user" if assessor != "automated_system" else "system",
                resource=f"ai_model:{model_id}",
                action="assess_risk",
                description=f"Risk assessment completed for model {model_name}",
                details={
                    "profile_id": profile_id,
                    "overall_risk_score": overall_risk_score,
                    "risk_level": risk_level.value,
                    "risk_factors_identified": len(assessed_risks),
                    "mitigation_strategies": len(mitigation_strategies)
                },
                risk_score=int(overall_risk_score * 10)  # Convert to 1-10 scale
            )
            
            uap_logger.log_ai_event(
                f"Risk assessment completed for {model_name}",
                model=model_name,
                framework=framework,
                metadata={
                    "profile_id": profile_id,
                    "risk_level": risk_level.value,
                    "overall_risk_score": overall_risk_score
                }
            )
            
            return risk_profile
            
        except Exception as e:
            logger.error(f"Failed to assess model risk: {e}")
            raise
    
    async def _identify_applicable_risks(self, model_id: str, use_cases: List[str],
                                       data_sources: List[str], deployment_context: Dict[str, Any]) -> List[RiskFactor]:
        """Identify which risk factors apply to this model"""
        applicable_risks = []
        
        # Always applicable risks
        base_risks = [
            "model_performance_degradation",
            "operational_dependency"
        ]
        
        for risk_id in base_risks:
            if risk_id in self.risk_factors_database:
                applicable_risks.append(self.risk_factors_database[risk_id])
        
        # Risk factors based on use cases
        high_stakes_use_cases = [
            "financial_decisions", "medical_diagnosis", "legal_analysis",
            "hiring_decisions", "credit_scoring", "autonomous_systems"
        ]
        
        if any(any(high_stake in use_case.lower() for high_stake in high_stakes_use_cases) 
               for use_case in use_cases):
            applicable_risks.append(self.risk_factors_database["algorithmic_bias"])
            applicable_risks.append(self.risk_factors_database["model_explainability_gap"])
        
        # Risk factors based on data sources
        sensitive_data_indicators = [
            "personal_data", "medical_records", "financial_data",
            "biometric_data", "location_data", "behavioral_data"
        ]
        
        if any(any(sensitive in data_source.lower() for sensitive in sensitive_data_indicators)
               for data_source in data_sources):
            applicable_risks.append(self.risk_factors_database["data_privacy_breach"])
        
        # Risk factors based on deployment context
        if deployment_context.get("internet_facing", False):
            applicable_risks.append(self.risk_factors_database["adversarial_vulnerability"])
        
        # Remove duplicates
        seen_ids = set()
        unique_risks = []
        for risk in applicable_risks:
            if risk.factor_id not in seen_ids:
                unique_risks.append(risk)
                seen_ids.add(risk.factor_id)
        
        return unique_risks
    
    async def _assess_risk_factor(self, risk_template: RiskFactor, model_id: str,
                                use_cases: List[str], data_sources: List[str],
                                deployment_context: Dict[str, Any]) -> RiskFactor:
        """Assess a specific risk factor for the model"""
        # Create a copy of the template and customize for this model
        assessed_risk = RiskFactor(
            factor_id=f"{risk_template.factor_id}_{model_id}",
            name=risk_template.name,
            description=risk_template.description,
            category=risk_template.category,
            likelihood=risk_template.likelihood,  # Will be adjusted
            impact_score=risk_template.impact_score,  # Will be adjusted
            risk_score=0.0,  # Will be calculated
            impact_areas=risk_template.impact_areas.copy(),
            indicators=risk_template.indicators.copy(),
            evidence={},
            detection_confidence=risk_template.detection_confidence,
            metadata=risk_template.metadata.copy()
        )
        
        # Adjust likelihood based on model-specific factors
        likelihood_adjustments = await self._calculate_likelihood_adjustments(
            assessed_risk, model_id, use_cases, data_sources, deployment_context
        )
        assessed_risk.likelihood = min(1.0, max(0.0, assessed_risk.likelihood + likelihood_adjustments))
        
        # Adjust impact based on model-specific factors
        impact_adjustments = await self._calculate_impact_adjustments(
            assessed_risk, model_id, use_cases, data_sources, deployment_context
        )
        assessed_risk.impact_score = min(10.0, max(0.0, assessed_risk.impact_score + impact_adjustments))
        
        # Calculate final risk score
        assessed_risk.risk_score = assessed_risk.likelihood * assessed_risk.impact_score
        
        # Collect evidence
        assessed_risk.evidence = await self._collect_risk_evidence(
            assessed_risk, model_id, deployment_context
        )
        
        # Update detection confidence based on available evidence
        assessed_risk.detection_confidence = self._calculate_detection_confidence(assessed_risk.evidence)
        
        return assessed_risk
    
    async def _calculate_likelihood_adjustments(self, risk: RiskFactor, model_id: str,
                                              use_cases: List[str], data_sources: List[str],
                                              deployment_context: Dict[str, Any]) -> float:
        """Calculate likelihood adjustments based on model specifics"""
        adjustments = 0.0
        
        # Deployment context adjustments
        if deployment_context.get("high_availability_required", False):
            if risk.category == RiskCategory.OPERATIONAL_RISK:
                adjustments += 0.2
        
        if deployment_context.get("public_facing", False):
            if risk.category == RiskCategory.SECURITY_RISK:
                adjustments += 0.3
        
        # Use case adjustments
        critical_use_cases = ["safety_critical", "financial", "medical", "legal"]
        if any(critical in " ".join(use_cases).lower() for critical in critical_use_cases):
            if risk.category in [RiskCategory.ETHICAL_RISK, RiskCategory.COMPLIANCE_RISK]:
                adjustments += 0.2
        
        # Data source adjustments
        if any("real_time" in ds.lower() for ds in data_sources):
            if risk.factor_id.endswith("model_performance_degradation"):
                adjustments += 0.1
        
        return adjustments
    
    async def _calculate_impact_adjustments(self, risk: RiskFactor, model_id: str,
                                          use_cases: List[str], data_sources: List[str],
                                          deployment_context: Dict[str, Any]) -> float:
        """Calculate impact adjustments based on model specifics"""
        adjustments = 0.0
        
        # Scale adjustments
        scale = deployment_context.get("expected_users", 1000)
        if scale > 100000:  # Large scale deployment
            adjustments += 1.0
        elif scale > 10000:  # Medium scale
            adjustments += 0.5
        
        # Business criticality adjustments
        if deployment_context.get("business_critical", False):
            adjustments += 1.5
        
        # Regulatory environment adjustments
        regulated_domains = ["healthcare", "finance", "government", "legal"]
        if any(domain in " ".join(use_cases).lower() for domain in regulated_domains):
            if risk.category == RiskCategory.COMPLIANCE_RISK:
                adjustments += 1.0
        
        return adjustments
    
    async def _collect_risk_evidence(self, risk: RiskFactor, model_id: str,
                                   deployment_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence for risk assessment"""
        evidence = {}
        
        # Collect evidence based on risk type
        if risk.category == RiskCategory.TECHNICAL_RISK:
            evidence.update(await self._collect_technical_evidence(model_id))
        elif risk.category == RiskCategory.SECURITY_RISK:
            evidence.update(await self._collect_security_evidence(model_id))
        elif risk.category == RiskCategory.ETHICAL_RISK:
            evidence.update(await self._collect_ethical_evidence(model_id))
        elif risk.category == RiskCategory.COMPLIANCE_RISK:
            evidence.update(await self._collect_compliance_evidence(model_id))
        
        # Add general evidence
        evidence.update({
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "deployment_context": deployment_context,
            "model_id": model_id
        })
        
        return evidence
    
    async def _collect_technical_evidence(self, model_id: str) -> Dict[str, Any]:
        """Collect technical evidence"""
        # Simulate technical metrics collection
        return {
            "current_accuracy": 0.87 + np.random.uniform(-0.1, 0.05),
            "error_rate": 0.03 + np.random.uniform(-0.01, 0.02),
            "response_time_p95": 1.2 + np.random.uniform(-0.5, 1.0),
            "model_complexity": "high",
            "training_data_age_days": 45,
            "last_performance_check": "2024-06-01"
        }
    
    async def _collect_security_evidence(self, model_id: str) -> Dict[str, Any]:
        """Collect security evidence"""
        return {
            "adversarial_testing_performed": False,
            "input_validation_strength": "medium",
            "access_controls": "basic",
            "data_encryption": "in_transit_only",
            "vulnerability_scan_date": None
        }
    
    async def _collect_ethical_evidence(self, model_id: str) -> Dict[str, Any]:
        """Collect ethical evidence"""
        return {
            "bias_testing_performed": True,
            "fairness_metrics": {
                "demographic_parity": 0.08,
                "equalized_odds": 0.05
            },
            "explainability_available": True,
            "stakeholder_review_completed": False
        }
    
    async def _collect_compliance_evidence(self, model_id: str) -> Dict[str, Any]:
        """Collect compliance evidence"""
        return {
            "gdpr_compliance_status": "partial",
            "data_retention_policy": "undefined",
            "consent_management": "manual",
            "audit_trail_completeness": 0.75,
            "documentation_status": "incomplete"
        }
    
    def _calculate_detection_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence in risk detection based on available evidence"""
        base_confidence = 0.5
        
        # Increase confidence based on evidence quality and quantity
        evidence_quality_score = 0.0
        
        # Technical evidence
        if "current_accuracy" in evidence:
            evidence_quality_score += 0.1
        if "error_rate" in evidence:
            evidence_quality_score += 0.1
        
        # Security evidence
        if "adversarial_testing_performed" in evidence:
            evidence_quality_score += 0.15
        
        # Ethical evidence
        if "bias_testing_performed" in evidence and evidence["bias_testing_performed"]:
            evidence_quality_score += 0.2
        
        # Compliance evidence
        if "audit_trail_completeness" in evidence:
            evidence_quality_score += 0.1 * evidence["audit_trail_completeness"]
        
        return min(1.0, base_confidence + evidence_quality_score)
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual risk factors"""
        if not risk_factors:
            return 0.0
        
        # Group risks by category
        category_scores = {}
        for risk in risk_factors:
            category = risk.category.value
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(risk.risk_score)
        
        # Calculate weighted average by category
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, scores in category_scores.items():
            category_weight = self.risk_weights.get(category, 0.1)
            category_score = max(scores) if scores else 0.0  # Use maximum risk in category
            weighted_score += category_weight * category_score
            total_weight += category_weight
        
        # Normalize by total weight
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return statistics.mean([risk.risk_score for risk in risk_factors])
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        if risk_score >= 9.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 7.0:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 5.0:
            return RiskLevel.HIGH
        elif risk_score >= 3.0:
            return RiskLevel.MEDIUM
        elif risk_score >= 1.0:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _calculate_risk_distribution(self, risk_factors: List[RiskFactor]) -> Dict[str, float]:
        """Calculate risk distribution across categories"""
        category_totals = {}
        
        for risk in risk_factors:
            category = risk.category.value
            if category not in category_totals:
                category_totals[category] = 0.0
            category_totals[category] += risk.risk_score
        
        # Normalize to percentages
        total_risk = sum(category_totals.values())
        if total_risk > 0:
            return {category: (score / total_risk) * 100 for category, score in category_totals.items()}
        else:
            return {}
    
    async def _generate_mitigation_strategies(self, risk_factors: List[RiskFactor]) -> List[RiskMitigationStrategy]:
        """Generate mitigation strategies for identified risks"""
        mitigation_strategies = []
        
        for risk in risk_factors:
            # Generate strategies based on risk type and severity
            strategies = await self._generate_risk_specific_strategies(risk)
            mitigation_strategies.extend(strategies)
        
        return mitigation_strategies
    
    async def _generate_risk_specific_strategies(self, risk: RiskFactor) -> List[RiskMitigationStrategy]:
        """Generate mitigation strategies for a specific risk"""
        strategies = []
        
        if risk.category == RiskCategory.TECHNICAL_RISK:
            if "performance_degradation" in risk.factor_id:
                strategies.append(RiskMitigationStrategy(
                    strategy_id=f"strategy_{risk.factor_id}_monitoring",
                    risk_id=risk.factor_id,
                    strategy_type="detective",
                    title="Continuous Performance Monitoring",
                    description="Implement continuous monitoring of model performance metrics",
                    implementation_steps=[
                        "Set up automated performance tracking",
                        "Configure alerting for performance degradation",
                        "Establish baseline performance metrics",
                        "Implement data drift detection"
                    ],
                    cost_estimate=5000.0,
                    timeline_days=30,
                    effectiveness_score=0.8,
                    responsible_party="ml_engineering_team",
                    dependencies=["monitoring_infrastructure"],
                    monitoring_metrics=["accuracy", "precision", "recall", "data_drift_score"],
                    success_criteria=["Performance alerts functioning", "Baseline metrics established"],
                    implemented=False,
                    implementation_date=None,
                    metadata={"priority": "high", "type": "monitoring"}
                ))
        
        elif risk.category == RiskCategory.ETHICAL_RISK:
            if "bias" in risk.factor_id:
                strategies.append(RiskMitigationStrategy(
                    strategy_id=f"strategy_{risk.factor_id}_bias_mitigation",
                    risk_id=risk.factor_id,
                    strategy_type="preventive",
                    title="Bias Detection and Mitigation",
                    description="Implement comprehensive bias detection and mitigation measures",
                    implementation_steps=[
                        "Conduct thorough bias assessment",
                        "Implement fairness constraints in model training",
                        "Deploy bias monitoring in production",
                        "Establish bias remediation procedures"
                    ],
                    cost_estimate=15000.0,
                    timeline_days=60,
                    effectiveness_score=0.9,
                    responsible_party="ai_ethics_team",
                    dependencies=["bias_detection_tools", "fairness_metrics"],
                    monitoring_metrics=["demographic_parity", "equalized_odds", "calibration"],
                    success_criteria=["Bias below threshold", "Fairness metrics established"],
                    implemented=False,
                    implementation_date=None,
                    metadata={"priority": "critical", "type": "fairness"}
                ))
        
        elif risk.category == RiskCategory.SECURITY_RISK:
            if "adversarial" in risk.factor_id:
                strategies.append(RiskMitigationStrategy(
                    strategy_id=f"strategy_{risk.factor_id}_adversarial_defense",
                    risk_id=risk.factor_id,
                    strategy_type="preventive",
                    title="Adversarial Defense Implementation",
                    description="Implement defenses against adversarial attacks",
                    implementation_steps=[
                        "Conduct adversarial vulnerability assessment",
                        "Implement input validation and sanitization",
                        "Deploy adversarial training techniques",
                        "Set up anomaly detection for unusual inputs"
                    ],
                    cost_estimate=20000.0,
                    timeline_days=90,
                    effectiveness_score=0.75,
                    responsible_party="security_team",
                    dependencies=["security_tools", "adversarial_datasets"],
                    monitoring_metrics=["adversarial_detection_rate", "input_anomaly_score"],
                    success_criteria=["Adversarial robustness improved", "Input validation deployed"],
                    implemented=False,
                    implementation_date=None,
                    metadata={"priority": "high", "type": "security"}
                ))
        
        elif risk.category == RiskCategory.COMPLIANCE_RISK:
            strategies.append(RiskMitigationStrategy(
                strategy_id=f"strategy_{risk.factor_id}_compliance",
                risk_id=risk.factor_id,
                strategy_type="preventive",
                title="Compliance Framework Implementation",
                description="Implement comprehensive compliance measures",
                implementation_steps=[
                    "Complete compliance gap analysis",
                    "Implement required documentation",
                    "Establish audit procedures",
                    "Train staff on compliance requirements"
                ],
                cost_estimate=10000.0,
                timeline_days=45,
                effectiveness_score=0.85,
                responsible_party="compliance_team",
                dependencies=["compliance_tools", "legal_review"],
                monitoring_metrics=["compliance_score", "audit_readiness"],
                success_criteria=["Full compliance achieved", "Audit procedures operational"],
                implemented=False,
                implementation_date=None,
                metadata={"priority": "high", "type": "compliance"}
            ))
        
        return strategies
    
    def _calculate_residual_risk(self, original_risk_score: float,
                               mitigation_strategies: List[RiskMitigationStrategy]) -> float:
        """Calculate residual risk after mitigation strategies"""
        if not mitigation_strategies:
            return original_risk_score
        
        # Calculate average effectiveness of implemented strategies
        implemented_strategies = [s for s in mitigation_strategies if s.implemented]
        
        if not implemented_strategies:
            # If no strategies are implemented yet, assume partial effectiveness
            avg_effectiveness = statistics.mean([s.effectiveness_score for s in mitigation_strategies]) * 0.5
        else:
            avg_effectiveness = statistics.mean([s.effectiveness_score for s in implemented_strategies])
        
        # Calculate residual risk
        risk_reduction = original_risk_score * avg_effectiveness
        residual_risk = max(0.0, original_risk_score - risk_reduction)
        
        return residual_risk
    
    def _generate_risk_recommendations(self, risk_factors: List[RiskFactor], overall_risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Overall risk level recommendations
        if overall_risk_score >= 7.0:
            recommendations.append("URGENT: Overall risk level is very high - immediate action required")
            recommendations.append("Consider delaying deployment until critical risks are mitigated")
        elif overall_risk_score >= 5.0:
            recommendations.append("High risk level detected - comprehensive mitigation plan needed")
        
        # Category-specific recommendations
        high_risk_categories = set()
        for risk in risk_factors:
            if risk.risk_score >= 6.0:
                high_risk_categories.add(risk.category)
        
        if RiskCategory.ETHICAL_RISK in high_risk_categories:
            recommendations.append("Implement comprehensive bias testing and fairness measures")
        
        if RiskCategory.SECURITY_RISK in high_risk_categories:
            recommendations.append("Conduct security audit and implement robust security controls")
        
        if RiskCategory.COMPLIANCE_RISK in high_risk_categories:
            recommendations.append("Complete compliance review and address regulatory requirements")
        
        # General recommendations
        recommendations.extend([
            "Establish continuous risk monitoring processes",
            "Implement automated risk detection and alerting",
            "Regular risk assessment reviews (quarterly)",
            "Maintain comprehensive risk documentation",
            "Provide risk management training to relevant staff"
        ])
        
        return recommendations
    
    async def conduct_scenario_analysis(self, model_id: str, scenarios: List[Dict[str, Any]]) -> List[RiskScenario]:
        """Conduct risk scenario analysis"""
        scenario_results = []
        
        for i, scenario_data in enumerate(scenarios):
            scenario = RiskScenario(
                scenario_id=f"scenario_{model_id}_{i}",
                name=scenario_data.get("name", f"Scenario {i+1}"),
                description=scenario_data.get("description", "Risk scenario analysis"),
                probability=scenario_data.get("probability", 0.1),
                impact_assessment=await self._assess_scenario_impact(scenario_data),
                cascading_effects=scenario_data.get("cascading_effects", []),
                mitigation_requirements=await self._identify_scenario_mitigations(scenario_data),
                worst_case_impact="Severe business disruption and regulatory penalties",
                realistic_impact="Moderate performance degradation and user dissatisfaction",
                best_case_impact="Minimal impact with quick recovery"
            )
            scenario_results.append(scenario)
        
        return scenario_results
    
    async def _assess_scenario_impact(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact of a risk scenario"""
        return {
            "financial_impact": scenario_data.get("financial_impact", "medium"),
            "operational_impact": scenario_data.get("operational_impact", "medium"),
            "reputational_impact": scenario_data.get("reputational_impact", "medium"),
            "regulatory_impact": scenario_data.get("regulatory_impact", "low"),
            "user_impact": scenario_data.get("user_impact", "medium"),
            "recovery_time": scenario_data.get("recovery_time", "24-48 hours")
        }
    
    async def _identify_scenario_mitigations(self, scenario_data: Dict[str, Any]) -> List[str]:
        """Identify mitigation requirements for a scenario"""
        return [
            "Implement monitoring and early warning systems",
            "Develop incident response procedures",
            "Establish rollback and recovery mechanisms",
            "Create communication and notification protocols",
            "Prepare stakeholder management procedures"
        ]
    
    async def _continuous_risk_monitoring(self):
        """Continuous monitoring of AI risks"""
        while True:
            try:
                # Check all active risk profiles
                for profile_id, profile in self.risk_profiles.items():
                    if profile.next_assessment_due <= datetime.now(timezone.utc):
                        logger.info(f"Risk assessment due for model {profile.model_id}")
                        # Would trigger risk reassessment in production
                
                # Monitor for emerging risks
                await self._monitor_emerging_risks()
                
                # Check risk thresholds
                await self._check_risk_thresholds()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in continuous risk monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_emerging_risks(self):
        """Monitor for new and emerging AI risks"""
        # This would integrate with threat intelligence feeds, research updates, etc.
        pass
    
    async def _check_risk_thresholds(self):
        """Check if any models exceed risk thresholds"""
        for profile in self.risk_profiles.values():
            if profile.overall_risk_score > self.risk_config["critical_risk_threshold"]:
                logger.critical(f"Model {profile.model_id} exceeds critical risk threshold")
            elif profile.overall_risk_score > self.risk_config["high_risk_threshold"]:
                logger.warning(f"Model {profile.model_id} exceeds high risk threshold")
    
    # Public API methods
    
    def get_risk_profile(self, model_id: str) -> Optional[AIRiskProfile]:
        """Get the latest risk profile for a model"""
        model_profiles = [
            profile for profile in self.risk_profiles.values()
            if profile.model_id == model_id
        ]
        
        if model_profiles:
            return max(model_profiles, key=lambda p: p.assessment_date)
        return None
    
    def get_all_risk_profiles(self, limit: int = 100) -> List[AIRiskProfile]:
        """Get all risk profiles"""
        return sorted(
            self.risk_profiles.values(),
            key=lambda p: p.assessment_date,
            reverse=True
        )[:limit]
    
    async def generate_risk_dashboard(self) -> Dict[str, Any]:
        """Generate risk management dashboard data"""
        all_profiles = list(self.risk_profiles.values())
        
        if not all_profiles:
            return {
                "dashboard_id": f"risk_dash_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_models": 0,
                "risk_summary": {},
                "alerts": [],
                "recommendations": ["No models assessed yet - begin risk assessments"]
            }
        
        # Calculate summary statistics
        risk_levels = [profile.risk_level.value for profile in all_profiles]
        overall_scores = [profile.overall_risk_score for profile in all_profiles]
        
        # Risk level distribution
        from collections import Counter
        risk_level_distribution = dict(Counter(risk_levels))
        
        # High-risk models
        high_risk_models = [
            profile for profile in all_profiles
            if profile.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]
        ]
        
        # Overdue assessments
        overdue_assessments = [
            profile for profile in all_profiles
            if profile.next_assessment_due < datetime.now(timezone.utc)
        ]
        
        return {
            "dashboard_id": f"risk_dash_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_models": len(all_profiles),
            "risk_summary": {
                "average_risk_score": statistics.mean(overall_scores),
                "highest_risk_score": max(overall_scores),
                "risk_level_distribution": risk_level_distribution,
                "high_risk_models": len(high_risk_models),
                "overdue_assessments": len(overdue_assessments)
            },
            "recent_assessments": [
                {
                    "model_id": profile.model_id,
                    "model_name": profile.model_name,
                    "risk_level": profile.risk_level.value,
                    "risk_score": profile.overall_risk_score,
                    "assessment_date": profile.assessment_date.isoformat()
                }
                for profile in sorted(all_profiles, key=lambda p: p.assessment_date, reverse=True)[:10]
            ],
            "alerts": [
                {
                    "type": "high_risk_model",
                    "severity": "high",
                    "message": f"Model {profile.model_id} has {profile.risk_level.value} risk level",
                    "model_id": profile.model_id
                }
                for profile in high_risk_models[:5]
            ] + [
                {
                    "type": "overdue_assessment",
                    "severity": "medium",
                    "message": f"Risk assessment overdue for model {profile.model_id}",
                    "model_id": profile.model_id
                }
                for profile in overdue_assessments[:5]
            ],
            "recommendations": self._generate_dashboard_risk_recommendations(all_profiles)
        }
    
    def _generate_dashboard_risk_recommendations(self, profiles: List[AIRiskProfile]) -> List[str]:
        """Generate recommendations for risk dashboard"""
        recommendations = []
        
        high_risk_count = len([p for p in profiles if p.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]])
        
        if high_risk_count > 0:
            recommendations.append(f"Address {high_risk_count} high-risk models immediately")
        
        avg_risk = statistics.mean([p.overall_risk_score for p in profiles])
        if avg_risk > 5.0:
            recommendations.append("Overall risk level is elevated - implement comprehensive risk management")
        
        recommendations.extend([
            "Maintain regular risk assessment schedule",
            "Implement automated risk monitoring",
            "Provide risk management training",
            "Review and update risk policies quarterly"
        ])
        
        return recommendations

# Global instance
_global_ai_risk_assessment = None

def get_ai_risk_assessment() -> AIRiskAssessment:
    """Get global AI risk assessment instance"""
    global _global_ai_risk_assessment
    if _global_ai_risk_assessment is None:
        _global_ai_risk_assessment = AIRiskAssessment()
    return _global_ai_risk_assessment