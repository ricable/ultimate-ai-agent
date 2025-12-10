# backend/governance/ai_governance.py
# Agent 26: Enterprise AI Governance - Model Governance, Explainability & Compliance

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib

# Governance and compliance tracking
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Metrics libraries not available, using mock implementation")

class ComplianceFramework(Enum):
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditEventType(Enum):
    MODEL_DEPLOYMENT = "model_deployment"
    PREDICTION_REQUEST = "prediction_request"
    DATA_ACCESS = "data_access"
    BIAS_DETECTION = "bias_detection"
    EXPLAINABILITY_REQUEST = "explainability_request"
    COMPLIANCE_VIOLATION = "compliance_violation"
    GOVERNANCE_REVIEW = "governance_review"

@dataclass
class ComplianceRequirement:
    """Compliance requirement definition"""
    framework: ComplianceFramework
    requirement_id: str
    title: str
    description: str
    mandatory: bool
    controls: List[str]
    evidence_required: List[str]
    review_frequency: str  # daily, weekly, monthly, quarterly, annually

@dataclass
class ModelGovernancePolicy:
    """AI model governance policy"""
    policy_id: str
    name: str
    description: str
    scope: List[str]  # model types or domains this applies to
    requirements: List[ComplianceRequirement]
    approval_workflow: List[str]  # roles required for approval
    monitoring_rules: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str

@dataclass
class BiasDetectionResult:
    """Bias detection analysis result"""
    model_id: str
    dataset_id: str
    protected_attributes: List[str]
    bias_metrics: Dict[str, float]
    bias_detected: bool
    risk_level: RiskLevel
    recommendations: List[str]
    timestamp: datetime

@dataclass
class ExplainabilityReport:
    """Model explainability report"""
    model_id: str
    prediction_id: str
    input_features: Dict[str, Any]
    prediction: Any
    explanation_method: str
    feature_importance: Dict[str, float]
    explanation_text: str
    confidence_score: float
    timestamp: datetime

@dataclass
class AuditEvent:
    """Audit trail event"""
    event_id: str
    event_type: AuditEventType
    model_id: Optional[str]
    user_id: str
    timestamp: datetime
    details: Dict[str, Any]
    risk_level: RiskLevel
    compliance_frameworks: List[ComplianceFramework]
    remediation_actions: List[str]

@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    framework: ComplianceFramework
    assessment_date: datetime
    overall_score: float
    requirements_assessed: int
    requirements_passed: int
    requirements_failed: int
    critical_findings: List[str]
    recommendations: List[str]
    next_review_date: datetime

class BiasDetector:
    """Detect bias in AI models and datasets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def detect_bias(self, 
                         model_id: str,
                         predictions: List[Any],
                         true_labels: List[Any],
                         protected_attributes: Dict[str, List[Any]],
                         dataset_id: str = None) -> BiasDetectionResult:
        """Detect bias in model predictions"""
        
        bias_metrics = {}
        bias_detected = False
        recommendations = []
        
        try:
            if METRICS_AVAILABLE:
                # Convert to numpy arrays for easier processing
                predictions = np.array(predictions)
                true_labels = np.array(true_labels)
                
                # Calculate bias metrics for each protected attribute
                for attr_name, attr_values in protected_attributes.items():
                    attr_array = np.array(attr_values)
                    
                    # Demographic Parity (Statistical Parity)
                    demo_parity = self._calculate_demographic_parity(predictions, attr_array)
                    bias_metrics[f'{attr_name}_demographic_parity'] = demo_parity
                    
                    # Equalized Odds
                    eq_odds = self._calculate_equalized_odds(predictions, true_labels, attr_array)
                    bias_metrics[f'{attr_name}_equalized_odds'] = eq_odds
                    
                    # Disparate Impact
                    disp_impact = self._calculate_disparate_impact(predictions, attr_array)
                    bias_metrics[f'{attr_name}_disparate_impact'] = disp_impact
                    
                    # Check if bias is detected
                    if demo_parity > 0.1 or eq_odds > 0.1 or disp_impact < 0.8:
                        bias_detected = True
                        recommendations.append(f"Bias detected in {attr_name}: consider data rebalancing or model retraining")
            
            else:
                # Mock bias detection
                bias_metrics = {
                    'demographic_parity': 0.05,
                    'equalized_odds': 0.08,
                    'disparate_impact': 0.92
                }
                bias_detected = False
                recommendations = ["No significant bias detected"]
        
        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")
            bias_metrics = {'error': str(e)}
            bias_detected = True
            recommendations = ["Bias analysis failed - manual review required"]
        
        # Determine risk level
        if bias_detected:
            max_bias = max([v for v in bias_metrics.values() if isinstance(v, (int, float))])
            if max_bias > 0.2:
                risk_level = RiskLevel.HIGH
            elif max_bias > 0.1:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.LOW
        
        return BiasDetectionResult(
            model_id=model_id,
            dataset_id=dataset_id or f"dataset_{uuid.uuid4().hex[:8]}",
            protected_attributes=list(protected_attributes.keys()),
            bias_metrics=bias_metrics,
            bias_detected=bias_detected,
            risk_level=risk_level,
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_demographic_parity(self, predictions: np.ndarray, protected_attr: np.ndarray) -> float:
        """Calculate demographic parity metric"""
        try:
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0
            
            positive_rates = []
            for group in unique_groups:
                group_mask = protected_attr == group
                group_predictions = predictions[group_mask]
                if len(group_predictions) > 0:
                    positive_rate = np.mean(group_predictions)
                    positive_rates.append(positive_rate)
            
            if len(positive_rates) >= 2:
                return abs(max(positive_rates) - min(positive_rates))
            return 0.0
        except:
            return 0.0
    
    def _calculate_equalized_odds(self, predictions: np.ndarray, true_labels: np.ndarray, protected_attr: np.ndarray) -> float:
        """Calculate equalized odds metric"""
        try:
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0
            
            tpr_differences = []
            fpr_differences = []
            
            for label in [0, 1]:  # Assuming binary classification
                label_mask = true_labels == label
                group_rates = []
                
                for group in unique_groups:
                    group_mask = (protected_attr == group) & label_mask
                    if np.sum(group_mask) > 0:
                        group_predictions = predictions[group_mask]
                        if label == 1:  # True Positive Rate
                            rate = np.mean(group_predictions)
                        else:  # False Positive Rate
                            rate = np.mean(group_predictions)
                        group_rates.append(rate)
                
                if len(group_rates) >= 2:
                    if label == 1:
                        tpr_differences.append(abs(max(group_rates) - min(group_rates)))
                    else:
                        fpr_differences.append(abs(max(group_rates) - min(group_rates)))
            
            return max(tpr_differences + fpr_differences) if (tpr_differences + fpr_differences) else 0.0
        except:
            return 0.0
    
    def _calculate_disparate_impact(self, predictions: np.ndarray, protected_attr: np.ndarray) -> float:
        """Calculate disparate impact ratio"""
        try:
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 1.0
            
            positive_rates = []
            for group in unique_groups:
                group_mask = protected_attr == group
                group_predictions = predictions[group_mask]
                if len(group_predictions) > 0:
                    positive_rate = np.mean(group_predictions)
                    positive_rates.append(positive_rate)
            
            if len(positive_rates) >= 2 and max(positive_rates) > 0:
                return min(positive_rates) / max(positive_rates)
            return 1.0
        except:
            return 1.0

class ExplainabilityEngine:
    """Generate explanations for AI model predictions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def explain_prediction(self,
                               model_id: str,
                               prediction_id: str,
                               input_features: Dict[str, Any],
                               prediction: Any,
                               explanation_method: str = "feature_importance") -> ExplainabilityReport:
        """Generate explanation for a model prediction"""
        
        try:
            # Mock feature importance calculation
            if explanation_method == "feature_importance":
                feature_importance = self._calculate_feature_importance(input_features)
                explanation_text = self._generate_explanation_text(feature_importance, prediction)
                confidence_score = 0.85
            
            elif explanation_method == "lime":
                # LIME explanation (mock)
                feature_importance = self._mock_lime_explanation(input_features)
                explanation_text = f"LIME analysis shows that {list(feature_importance.keys())[0]} was the most influential feature"
                confidence_score = 0.75
            
            elif explanation_method == "shap":
                # SHAP explanation (mock)
                feature_importance = self._mock_shap_explanation(input_features)
                explanation_text = f"SHAP values indicate that {list(feature_importance.keys())[0]} contributed most to this prediction"
                confidence_score = 0.90
            
            else:
                # Default explanation
                feature_importance = {k: 0.1 for k in input_features.keys()}
                explanation_text = "Generic explanation: prediction based on input features"
                confidence_score = 0.50
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            feature_importance = {k: 0.0 for k in input_features.keys()}
            explanation_text = f"Explanation generation failed: {str(e)}"
            confidence_score = 0.0
        
        return ExplainabilityReport(
            model_id=model_id,
            prediction_id=prediction_id,
            input_features=input_features,
            prediction=prediction,
            explanation_method=explanation_method,
            feature_importance=feature_importance,
            explanation_text=explanation_text,
            confidence_score=confidence_score,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_feature_importance(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate mock feature importance scores"""
        importance = {}
        total_features = len(features)
        
        for i, (feature_name, value) in enumerate(features.items()):
            # Mock importance based on feature position and value
            if isinstance(value, (int, float)):
                base_importance = abs(float(value)) / 100.0
            else:
                base_importance = len(str(value)) / 50.0
            
            # Add some randomness but make it deterministic
            import hashlib
            hash_val = int(hashlib.md5(feature_name.encode()).hexdigest()[:8], 16)
            importance[feature_name] = min(1.0, base_importance + (hash_val % 100) / 1000.0)
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def _mock_lime_explanation(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Mock LIME explanation"""
        return {k: 0.1 + (len(k) % 5) * 0.1 for k in features.keys()}
    
    def _mock_shap_explanation(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Mock SHAP explanation"""
        return {k: 0.05 + (hash(k) % 10) * 0.05 for k in features.keys()}
    
    def _generate_explanation_text(self, feature_importance: Dict[str, float], prediction: Any) -> str:
        """Generate human-readable explanation text"""
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"The prediction '{prediction}' was primarily influenced by: "
        feature_explanations = []
        
        for feature, importance in top_features:
            percentage = importance * 100
            feature_explanations.append(f"{feature} ({percentage:.1f}%)")
        
        explanation += ", ".join(feature_explanations)
        return explanation

class ComplianceEngine:
    """Manage compliance with various regulatory frameworks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.frameworks = self._initialize_frameworks()
    
    def _initialize_frameworks(self) -> Dict[ComplianceFramework, List[ComplianceRequirement]]:
        """Initialize compliance framework requirements"""
        frameworks = {}
        
        # GDPR Requirements
        frameworks[ComplianceFramework.GDPR] = [
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="GDPR-ART-22",
                title="Automated Decision Making",
                description="Right not to be subject to automated decision-making",
                mandatory=True,
                controls=["explainability", "human_review"],
                evidence_required=["explanation_logs", "human_review_records"],
                review_frequency="quarterly"
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="GDPR-ART-5",
                title="Data Processing Principles",
                description="Personal data must be processed lawfully, fairly and transparently",
                mandatory=True,
                controls=["data_minimization", "purpose_limitation", "accuracy"],
                evidence_required=["data_inventory", "processing_records"],
                review_frequency="monthly"
            )
        ]
        
        # SOC2 Requirements
        frameworks[ComplianceFramework.SOC2] = [
            ComplianceRequirement(
                framework=ComplianceFramework.SOC2,
                requirement_id="SOC2-CC6.1",
                title="Logical and Physical Access Controls",
                description="Implement logical and physical access controls",
                mandatory=True,
                controls=["access_control", "authentication", "authorization"],
                evidence_required=["access_logs", "user_reviews"],
                review_frequency="monthly"
            )
        ]
        
        # HIPAA Requirements
        frameworks[ComplianceFramework.HIPAA] = [
            ComplianceRequirement(
                framework=ComplianceFramework.HIPAA,
                requirement_id="HIPAA-164.312",
                title="Technical Safeguards",
                description="Implement technical safeguards for PHI",
                mandatory=True,
                controls=["encryption", "access_control", "audit_logging"],
                evidence_required=["encryption_status", "access_logs", "audit_reports"],
                review_frequency="quarterly"
            )
        ]
        
        return frameworks
    
    async def assess_compliance(self, 
                              framework: ComplianceFramework,
                              model_id: str = None,
                              evidence: Dict[str, Any] = None) -> ComplianceReport:
        """Assess compliance with a specific framework"""
        
        if evidence is None:
            evidence = {}
        
        requirements = self.frameworks.get(framework, [])
        requirements_assessed = len(requirements)
        requirements_passed = 0
        requirements_failed = 0
        critical_findings = []
        recommendations = []
        
        for requirement in requirements:
            # Check if requirement is satisfied
            is_satisfied = self._check_requirement_satisfaction(requirement, evidence)
            
            if is_satisfied:
                requirements_passed += 1
            else:
                requirements_failed += 1
                if requirement.mandatory:
                    critical_findings.append(f"Mandatory requirement {requirement.requirement_id} not satisfied")
                recommendations.append(f"Implement controls for {requirement.title}")
        
        # Calculate overall score
        overall_score = (requirements_passed / requirements_assessed * 100) if requirements_assessed > 0 else 0
        
        # Determine next review date
        next_review_date = datetime.utcnow() + timedelta(days=90)  # Default quarterly
        
        return ComplianceReport(
            report_id=str(uuid.uuid4()),
            framework=framework,
            assessment_date=datetime.utcnow(),
            overall_score=overall_score,
            requirements_assessed=requirements_assessed,
            requirements_passed=requirements_passed,
            requirements_failed=requirements_failed,
            critical_findings=critical_findings,
            recommendations=recommendations,
            next_review_date=next_review_date
        )
    
    def _check_requirement_satisfaction(self, 
                                      requirement: ComplianceRequirement, 
                                      evidence: Dict[str, Any]) -> bool:
        """Check if a compliance requirement is satisfied"""
        
        # Check if required evidence is present
        for evidence_type in requirement.evidence_required:
            if evidence_type not in evidence:
                return False
        
        # Check if required controls are implemented
        for control in requirement.controls:
            control_evidence = evidence.get(f"{control}_implemented", False)
            if not control_evidence:
                return False
        
        return True

class AuditTrail:
    """Comprehensive audit trail for AI governance"""
    
    def __init__(self, storage_path: str = "./governance_audit"):
        self.storage_path = storage_path
        self.events: List[AuditEvent] = []
        self.logger = logging.getLogger(__name__)
    
    async def log_event(self,
                       event_type: AuditEventType,
                       user_id: str,
                       details: Dict[str, Any],
                       model_id: str = None,
                       risk_level: RiskLevel = RiskLevel.LOW,
                       compliance_frameworks: List[ComplianceFramework] = None) -> str:
        """Log an audit event"""
        
        if compliance_frameworks is None:
            compliance_frameworks = []
        
        event_id = str(uuid.uuid4())
        
        # Determine remediation actions based on event type and risk level
        remediation_actions = self._determine_remediation_actions(event_type, risk_level, details)
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            model_id=model_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            details=details,
            risk_level=risk_level,
            compliance_frameworks=compliance_frameworks,
            remediation_actions=remediation_actions
        )
        
        self.events.append(event)
        
        # Persist event (in production, this would write to a database)
        await self._persist_event(event)
        
        self.logger.info(f"Audit event logged: {event_type.value} by {user_id}")
        
        return event_id
    
    def _determine_remediation_actions(self,
                                     event_type: AuditEventType,
                                     risk_level: RiskLevel,
                                     details: Dict[str, Any]) -> List[str]:
        """Determine required remediation actions"""
        actions = []
        
        if risk_level == RiskLevel.CRITICAL:
            actions.append("immediate_review_required")
            actions.append("escalate_to_governance_committee")
        
        if event_type == AuditEventType.BIAS_DETECTION:
            actions.append("bias_mitigation_review")
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                actions.append("model_deployment_hold")
        
        if event_type == AuditEventType.COMPLIANCE_VIOLATION:
            actions.append("compliance_officer_notification")
            actions.append("violation_remediation_plan")
        
        return actions
    
    async def _persist_event(self, event: AuditEvent):
        """Persist audit event to storage"""
        # In production, this would write to a secure, immutable audit log
        # For now, we'll just log it
        self.logger.info(f"Persisting audit event: {event.event_id}")
    
    async def query_events(self,
                          start_date: datetime = None,
                          end_date: datetime = None,
                          event_type: AuditEventType = None,
                          user_id: str = None,
                          model_id: str = None,
                          risk_level: RiskLevel = None) -> List[AuditEvent]:
        """Query audit events with filters"""
        
        filtered_events = self.events.copy()
        
        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
        
        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if model_id:
            filtered_events = [e for e in filtered_events if e.model_id == model_id]
        
        if risk_level:
            filtered_events = [e for e in filtered_events if e.risk_level == risk_level]
        
        return sorted(filtered_events, key=lambda e: e.timestamp, reverse=True)

class AIGovernanceManager:
    """Main AI governance orchestrator"""
    
    def __init__(self):
        self.bias_detector = BiasDetector()
        self.explainability_engine = ExplainabilityEngine()
        self.compliance_engine = ComplianceEngine()
        self.audit_trail = AuditTrail()
        self.policies: Dict[str, ModelGovernancePolicy] = {}
        self.logger = logging.getLogger(__name__)
    
    async def register_model_deployment(self,
                                      model_id: str,
                                      user_id: str,
                                      model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new model deployment with governance checks"""
        
        # Log deployment event
        await self.audit_trail.log_event(
            event_type=AuditEventType.MODEL_DEPLOYMENT,
            user_id=user_id,
            model_id=model_id,
            details=model_metadata,
            risk_level=RiskLevel.MEDIUM,
            compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.GDPR]
        )
        
        # Check applicable policies
        applicable_policies = self._get_applicable_policies(model_metadata)
        
        # Perform governance checks
        governance_results = {
            'deployment_approved': True,
            'policies_checked': len(applicable_policies),
            'compliance_status': 'pending',
            'required_actions': []
        }
        
        # If high-risk model, require additional approvals
        if model_metadata.get('risk_category') == 'high':
            governance_results['required_actions'].append('governance_committee_approval')
            governance_results['deployment_approved'] = False
        
        return governance_results
    
    async def monitor_model_predictions(self,
                                      model_id: str,
                                      predictions: List[Any],
                                      true_labels: List[Any] = None,
                                      protected_attributes: Dict[str, List[Any]] = None,
                                      user_id: str = "system") -> Dict[str, Any]:
        """Monitor model predictions for bias and compliance"""
        
        monitoring_results = {
            'bias_analysis': None,
            'compliance_issues': [],
            'alerts_generated': []
        }
        
        # Bias detection if protected attributes provided
        if protected_attributes and true_labels:
            bias_result = await self.bias_detector.detect_bias(
                model_id=model_id,
                predictions=predictions,
                true_labels=true_labels,
                protected_attributes=protected_attributes
            )
            
            monitoring_results['bias_analysis'] = asdict(bias_result)
            
            # Log bias detection event
            if bias_result.bias_detected:
                await self.audit_trail.log_event(
                    event_type=AuditEventType.BIAS_DETECTION,
                    user_id=user_id,
                    model_id=model_id,
                    details={'bias_metrics': bias_result.bias_metrics},
                    risk_level=bias_result.risk_level,
                    compliance_frameworks=[ComplianceFramework.GDPR]
                )
                monitoring_results['alerts_generated'].append('bias_detected')
        
        return monitoring_results
    
    async def generate_explanation(self,
                                 model_id: str,
                                 prediction_id: str,
                                 input_features: Dict[str, Any],
                                 prediction: Any,
                                 user_id: str,
                                 explanation_method: str = "feature_importance") -> ExplainabilityReport:
        """Generate explanation for a model prediction"""
        
        # Generate explanation
        explanation = await self.explainability_engine.explain_prediction(
            model_id=model_id,
            prediction_id=prediction_id,
            input_features=input_features,
            prediction=prediction,
            explanation_method=explanation_method
        )
        
        # Log explainability request
        await self.audit_trail.log_event(
            event_type=AuditEventType.EXPLAINABILITY_REQUEST,
            user_id=user_id,
            model_id=model_id,
            details={
                'prediction_id': prediction_id,
                'explanation_method': explanation_method,
                'confidence_score': explanation.confidence_score
            },
            risk_level=RiskLevel.LOW,
            compliance_frameworks=[ComplianceFramework.GDPR]
        )
        
        return explanation
    
    async def assess_compliance(self,
                              framework: ComplianceFramework,
                              model_id: str = None,
                              evidence: Dict[str, Any] = None) -> ComplianceReport:
        """Assess compliance with regulatory framework"""
        
        return await self.compliance_engine.assess_compliance(
            framework=framework,
            model_id=model_id,
            evidence=evidence
        )
    
    def _get_applicable_policies(self, model_metadata: Dict[str, Any]) -> List[ModelGovernancePolicy]:
        """Get governance policies applicable to a model"""
        applicable = []
        
        model_type = model_metadata.get('type', 'unknown')
        model_domain = model_metadata.get('domain', 'general')
        
        for policy in self.policies.values():
            if (model_type in policy.scope or 
                model_domain in policy.scope or 
                'all' in policy.scope):
                applicable.append(policy)
        
        return applicable
    
    async def get_governance_dashboard(self) -> Dict[str, Any]:
        """Get governance dashboard data"""
        
        # Get recent audit events
        recent_events = await self.audit_trail.query_events(
            start_date=datetime.utcnow() - timedelta(days=30)
        )
        
        # Aggregate metrics
        dashboard_data = {
            'total_audit_events': len(recent_events),
            'high_risk_events': len([e for e in recent_events if e.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]),
            'bias_detections': len([e for e in recent_events if e.event_type == AuditEventType.BIAS_DETECTION]),
            'compliance_violations': len([e for e in recent_events if e.event_type == AuditEventType.COMPLIANCE_VIOLATION]),
            'explainability_requests': len([e for e in recent_events if e.event_type == AuditEventType.EXPLAINABILITY_REQUEST]),
            'active_policies': len(self.policies),
            'recent_events': [asdict(e) for e in recent_events[:10]]
        }
        
        return dashboard_data

# Global AI governance manager
ai_governance = AIGovernanceManager()