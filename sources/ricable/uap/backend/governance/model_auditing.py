# File: backend/governance/model_auditing.py
"""
AI Model Auditing Service

Comprehensive auditing system for AI models including performance auditing,
behavior analysis, compliance verification, and security assessment.
"""

import asyncio
import json
import logging
import numpy as np
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import uuid

# Import existing systems
from ..security.audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger
from ..ai.explainability import get_ai_explainability_engine, ExplanationMethod, BiasType

logger = logging.getLogger(__name__)

class AuditType(Enum):
    """Types of AI model audits"""
    PERFORMANCE_AUDIT = "performance_audit"
    COMPLIANCE_AUDIT = "compliance_audit"
    SECURITY_AUDIT = "security_audit"
    BIAS_AUDIT = "bias_audit"
    EXPLAINABILITY_AUDIT = "explainability_audit"
    DATA_QUALITY_AUDIT = "data_quality_audit"
    BEHAVIORAL_AUDIT = "behavioral_audit"
    COMPREHENSIVE_AUDIT = "comprehensive_audit"

class AuditStatus(Enum):
    """Status of audit processes"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_REVIEW = "needs_review"

class AuditSeverity(Enum):
    """Severity levels for audit findings"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditFinding:
    """Individual audit finding"""
    finding_id: str
    audit_id: str
    finding_type: str
    severity: AuditSeverity
    title: str
    description: str
    evidence: Dict[str, Any]
    impact_assessment: str
    recommendations: List[str]
    compliance_frameworks_affected: List[str]
    detected_at: datetime
    category: str
    automated: bool
    verified: bool
    metadata: Dict[str, Any]

@dataclass
class ModelAuditReport:
    """Comprehensive model audit report"""
    audit_id: str
    model_id: str
    model_name: str
    framework: str
    audit_type: AuditType
    audit_status: AuditStatus
    started_at: datetime
    completed_at: Optional[datetime]
    auditor: str
    audit_scope: List[str]
    findings: List[AuditFinding]
    overall_assessment: str
    compliance_status: str
    security_status: str
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    next_audit_due: datetime
    audit_evidence: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class AuditConfiguration:
    """Configuration for model audits"""
    model_id: str
    audit_types: List[AuditType]
    frequency: timedelta
    compliance_frameworks: List[str]
    performance_thresholds: Dict[str, float]
    bias_thresholds: Dict[str, float]
    automated_checks: List[str]
    manual_review_required: bool
    stakeholders: List[str]
    notification_settings: Dict[str, Any]

class ModelAuditingService:
    """
    Comprehensive AI Model Auditing Service
    
    Provides thorough auditing capabilities for AI models including:
    - Performance and accuracy assessment
    - Bias and fairness evaluation
    - Security vulnerability analysis
    - Compliance verification
    - Behavioral pattern analysis
    - Data quality assessment
    """
    
    def __init__(self):
        self.audit_configurations: Dict[str, AuditConfiguration] = {}
        self.audit_reports: List[ModelAuditReport] = []
        self.audit_findings: List[AuditFinding] = {}
        
        # Integration with existing systems
        self.audit_logger = get_security_audit_logger()
        self.explainability_engine = get_ai_explainability_engine()
        
        # Audit templates and checklists
        self.audit_templates = self._initialize_audit_templates()
        
        # Performance thresholds
        self.default_thresholds = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.80,
            "auc_roc": 0.80,
            "response_time": 2.0,  # seconds
            "throughput": 100,  # requests/second
            "error_rate": 0.05,
            "bias_score": 0.10,
            "fairness_score": 0.80
        }
        
        # Background audit scheduler
        asyncio.create_task(self._audit_scheduler())
        
        logger.info("Model Auditing Service initialized")
    
    def _initialize_audit_templates(self) -> Dict[AuditType, Dict[str, Any]]:
        """Initialize audit templates for different audit types"""
        return {
            AuditType.PERFORMANCE_AUDIT: {
                "checklist": [
                    "Model accuracy assessment",
                    "Response time analysis",
                    "Throughput measurement",
                    "Error rate evaluation",
                    "Resource utilization analysis",
                    "Scalability assessment"
                ],
                "metrics": ["accuracy", "precision", "recall", "f1_score", "response_time", "throughput"],
                "automated": True,
                "duration_hours": 2
            },
            AuditType.BIAS_AUDIT: {
                "checklist": [
                    "Demographic parity assessment",
                    "Equalized odds evaluation",
                    "Calibration analysis",
                    "Representation bias check",
                    "Historical bias assessment",
                    "Intersectional bias analysis"
                ],
                "metrics": ["demographic_parity", "equalized_odds", "calibration", "bias_score"],
                "automated": True,
                "duration_hours": 4
            },
            AuditType.SECURITY_AUDIT: {
                "checklist": [
                    "Model adversarial robustness",
                    "Input validation security",
                    "Model extraction vulnerability",
                    "Privacy leakage assessment",
                    "Access control verification",
                    "Data encryption compliance"
                ],
                "metrics": ["adversarial_robustness", "privacy_score", "security_score"],
                "automated": False,
                "duration_hours": 8
            },
            AuditType.COMPLIANCE_AUDIT: {
                "checklist": [
                    "Regulatory compliance verification",
                    "Documentation completeness",
                    "Process adherence check",
                    "Audit trail verification",
                    "Consent management review",
                    "Data retention compliance"
                ],
                "metrics": ["compliance_score", "documentation_completeness"],
                "automated": False,
                "duration_hours": 6
            },
            AuditType.EXPLAINABILITY_AUDIT: {
                "checklist": [
                    "Explanation capability assessment",
                    "Decision transparency evaluation",
                    "Stakeholder understandability",
                    "Explanation consistency check",
                    "Method appropriateness review"
                ],
                "metrics": ["explainability_score", "transparency_score"],
                "automated": True,
                "duration_hours": 3
            }
        }
    
    async def configure_model_audit(self, model_id: str, model_name: str, framework: str,
                                  audit_types: List[AuditType], frequency: timedelta = None,
                                  compliance_frameworks: List[str] = None,
                                  custom_thresholds: Dict[str, float] = None) -> str:
        """
        Configure auditing for a specific AI model.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            framework: AI framework (copilot, agno, mastra, mlx)
            audit_types: Types of audits to perform
            frequency: How often to perform audits
            compliance_frameworks: Applicable compliance frameworks
            custom_thresholds: Custom performance thresholds
            
        Returns:
            Configuration ID
        """
        try:
            config_id = f"audit_config_{model_id}"
            
            # Set defaults
            if frequency is None:
                frequency = timedelta(days=30)  # Monthly audits by default
            
            if compliance_frameworks is None:
                compliance_frameworks = ["SOC2", "GDPR"]
            
            # Merge custom thresholds with defaults
            thresholds = self.default_thresholds.copy()
            if custom_thresholds:
                thresholds.update(custom_thresholds)
            
            configuration = AuditConfiguration(
                model_id=model_id,
                audit_types=audit_types,
                frequency=frequency,
                compliance_frameworks=compliance_frameworks,
                performance_thresholds=thresholds,
                bias_thresholds={
                    "demographic_parity": 0.10,
                    "equalized_odds": 0.10,
                    "calibration": 0.05
                },
                automated_checks=[
                    "performance_metrics",
                    "bias_detection",
                    "error_rate_monitoring",
                    "response_time_tracking"
                ],
                manual_review_required=AuditType.SECURITY_AUDIT in audit_types or AuditType.COMPLIANCE_AUDIT in audit_types,
                stakeholders=["model_owner", "compliance_team", "security_team"],
                notification_settings={
                    "email_alerts": True,
                    "dashboard_notifications": True,
                    "critical_issue_escalation": True
                }
            )
            
            self.audit_configurations[config_id] = configuration
            
            # Log configuration
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.SUCCESS,
                actor_id="model_auditing_service",
                actor_type="system",
                resource=f"ai_model:{model_id}",
                action="configure_audit",
                description=f"Audit configuration created for model {model_name}",
                details={
                    "config_id": config_id,
                    "audit_types": [at.value for at in audit_types],
                    "frequency_days": frequency.days,
                    "compliance_frameworks": compliance_frameworks
                },
                risk_score=3
            )
            
            # Schedule initial audit
            await self._schedule_audit(model_id, AuditType.COMPREHENSIVE_AUDIT)
            
            logger.info(f"Audit configuration created for model {model_name}: {config_id}")
            
            return config_id
            
        except Exception as e:
            logger.error(f"Failed to configure model audit: {e}")
            raise
    
    async def conduct_audit(self, model_id: str, audit_type: AuditType,
                          auditor: str = "automated_system") -> ModelAuditReport:
        """
        Conduct a comprehensive audit of an AI model.
        
        Args:
            model_id: ID of the model to audit
            audit_type: Type of audit to perform
            auditor: Who is conducting the audit
            
        Returns:
            Comprehensive audit report
        """
        try:
            audit_id = f"audit_{model_id}_{audit_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting {audit_type.value} audit {audit_id} for model {model_id}")
            
            # Get model configuration
            config = self.audit_configurations.get(f"audit_config_{model_id}")
            if not config:
                raise ValueError(f"No audit configuration found for model {model_id}")
            
            # Initialize audit report
            audit_report = ModelAuditReport(
                audit_id=audit_id,
                model_id=model_id,
                model_name=f"model_{model_id}",  # Would be retrieved from model registry
                framework="unknown",  # Would be retrieved from model registry
                audit_type=audit_type,
                audit_status=AuditStatus.IN_PROGRESS,
                started_at=datetime.now(timezone.utc),
                completed_at=None,
                auditor=auditor,
                audit_scope=self._get_audit_scope(audit_type),
                findings=[],
                overall_assessment="",
                compliance_status="",
                security_status="",
                performance_metrics={},
                recommendations=[],
                next_audit_due=datetime.now(timezone.utc) + config.frequency,
                audit_evidence={},
                metadata={
                    "audit_configuration": config_id if (config_id := f"audit_config_{model_id}") else None,
                    "automated": audit_type in [AuditType.PERFORMANCE_AUDIT, AuditType.BIAS_AUDIT]
                }
            )
            
            # Perform audit based on type
            if audit_type == AuditType.PERFORMANCE_AUDIT:
                await self._conduct_performance_audit(audit_report, config)
            elif audit_type == AuditType.BIAS_AUDIT:
                await self._conduct_bias_audit(audit_report, config)
            elif audit_type == AuditType.SECURITY_AUDIT:
                await self._conduct_security_audit(audit_report, config)
            elif audit_type == AuditType.COMPLIANCE_AUDIT:
                await self._conduct_compliance_audit(audit_report, config)
            elif audit_type == AuditType.EXPLAINABILITY_AUDIT:
                await self._conduct_explainability_audit(audit_report, config)
            elif audit_type == AuditType.COMPREHENSIVE_AUDIT:
                await self._conduct_comprehensive_audit(audit_report, config)
            else:
                raise ValueError(f"Unsupported audit type: {audit_type}")
            
            # Finalize audit
            audit_report.completed_at = datetime.now(timezone.utc)
            audit_report.audit_status = AuditStatus.COMPLETED
            
            # Generate overall assessment
            audit_report.overall_assessment = self._generate_overall_assessment(audit_report)
            audit_report.recommendations = self._generate_audit_recommendations(audit_report)
            
            # Store audit report
            self.audit_reports.append(audit_report)
            
            # Log audit completion
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.SUCCESS,
                actor_id=auditor,
                actor_type="user" if auditor != "automated_system" else "system",
                resource=f"ai_model:{model_id}",
                action="complete_audit",
                description=f"{audit_type.value} audit completed for model {model_id}",
                details={
                    "audit_id": audit_id,
                    "findings_count": len(audit_report.findings),
                    "critical_findings": len([f for f in audit_report.findings if f.severity == AuditSeverity.CRITICAL]),
                    "overall_assessment": audit_report.overall_assessment
                },
                risk_score=2 + len([f for f in audit_report.findings if f.severity == AuditSeverity.CRITICAL])
            )
            
            uap_logger.log_ai_event(
                f"Model audit completed: {audit_type.value}",
                model=model_id,
                metadata={
                    "audit_id": audit_id,
                    "findings_count": len(audit_report.findings),
                    "duration_minutes": int((audit_report.completed_at - audit_report.started_at).total_seconds() / 60)
                }
            )
            
            return audit_report
            
        except Exception as e:
            logger.error(f"Failed to conduct audit: {e}")
            
            # Update audit status to failed
            if 'audit_report' in locals():
                audit_report.audit_status = AuditStatus.FAILED
                audit_report.completed_at = datetime.now(timezone.utc)
                audit_report.metadata["error"] = str(e)
            
            raise
    
    def _get_audit_scope(self, audit_type: AuditType) -> List[str]:
        """Get audit scope based on audit type"""
        template = self.audit_templates.get(audit_type, {})
        return template.get("checklist", [])
    
    async def _conduct_performance_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct performance audit"""
        try:
            logger.info(f"Conducting performance audit for {audit_report.model_id}")
            
            # Simulate performance metrics collection
            performance_metrics = await self._collect_performance_metrics(audit_report.model_id)
            audit_report.performance_metrics = performance_metrics
            
            # Check against thresholds
            for metric, value in performance_metrics.items():
                threshold = config.performance_thresholds.get(metric)
                if threshold:
                    if metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "fairness_score"]:
                        # Higher is better
                        if value < threshold:
                            finding = self._create_finding(
                                audit_report.audit_id,
                                "performance_issue",
                                AuditSeverity.MEDIUM if value < threshold * 0.9 else AuditSeverity.LOW,
                                f"Low {metric}",
                                f"{metric.title()} ({value:.3f}) is below threshold ({threshold:.3f})",
                                {"metric": metric, "value": value, "threshold": threshold},
                                f"Poor {metric} may impact model reliability and user trust",
                                [f"Investigate factors affecting {metric}", "Consider model retraining", "Review data quality"],
                                ["SOC2"] if metric in ["accuracy", "precision", "recall"] else []
                            )
                            audit_report.findings.append(finding)
                    
                    elif metric in ["response_time", "error_rate"]:
                        # Lower is better
                        if value > threshold:
                            severity = AuditSeverity.HIGH if value > threshold * 2 else AuditSeverity.MEDIUM
                            finding = self._create_finding(
                                audit_report.audit_id,
                                "performance_issue",
                                severity,
                                f"High {metric}",
                                f"{metric.title()} ({value:.3f}) exceeds threshold ({threshold:.3f})",
                                {"metric": metric, "value": value, "threshold": threshold},
                                f"High {metric} impacts user experience and system reliability",
                                [f"Optimize {metric}", "Review infrastructure capacity", "Implement performance monitoring"],
                                ["SOC2"]
                            )
                            audit_report.findings.append(finding)
            
        except Exception as e:
            logger.error(f"Error in performance audit: {e}")
            raise
    
    async def _conduct_bias_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct bias audit"""
        try:
            logger.info(f"Conducting bias audit for {audit_report.model_id}")
            
            # Simulate bias detection
            bias_results = await self._detect_model_bias(audit_report.model_id)
            
            for bias_type, bias_data in bias_results.items():
                bias_score = bias_data.get("score", 0.0)
                threshold = config.bias_thresholds.get(bias_type, 0.10)
                
                if bias_score > threshold:
                    severity = AuditSeverity.CRITICAL if bias_score > threshold * 2 else AuditSeverity.HIGH
                    
                    finding = self._create_finding(
                        audit_report.audit_id,
                        "bias_detection",
                        severity,
                        f"Bias detected: {bias_type}",
                        f"Significant bias detected in {bias_type} (score: {bias_score:.3f})",
                        bias_data,
                        "Bias can lead to unfair treatment and regulatory compliance issues",
                        [
                            "Implement bias mitigation techniques",
                            "Review training data for balance",
                            "Consider algorithmic fairness constraints",
                            "Monitor ongoing predictions for bias"
                        ],
                        ["GDPR", "SOC2"] if "demographic" in bias_type else ["SOC2"]
                    )
                    audit_report.findings.append(finding)
            
        except Exception as e:
            logger.error(f"Error in bias audit: {e}")
            raise
    
    async def _conduct_security_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct security audit"""
        try:
            logger.info(f"Conducting security audit for {audit_report.model_id}")
            
            # Simulate security assessment
            security_assessment = await self._assess_model_security(audit_report.model_id)
            
            audit_report.security_status = security_assessment.get("overall_status", "unknown")
            
            for vulnerability in security_assessment.get("vulnerabilities", []):
                severity = AuditSeverity(vulnerability.get("severity", "medium"))
                
                finding = self._create_finding(
                    audit_report.audit_id,
                    "security_vulnerability",
                    severity,
                    vulnerability.get("title", "Security vulnerability"),
                    vulnerability.get("description", "Security vulnerability detected"),
                    vulnerability,
                    vulnerability.get("impact", "Potential security risk"),
                    vulnerability.get("recommendations", ["Address security vulnerability"]),
                    ["SOC2", "ISO27001"]
                )
                audit_report.findings.append(finding)
            
        except Exception as e:
            logger.error(f"Error in security audit: {e}")
            raise
    
    async def _conduct_compliance_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct compliance audit"""
        try:
            logger.info(f"Conducting compliance audit for {audit_report.model_id}")
            
            # Check compliance for each framework
            for framework in config.compliance_frameworks:
                compliance_status = await self._check_framework_compliance(audit_report.model_id, framework)
                
                if not compliance_status.get("compliant", False):
                    for issue in compliance_status.get("issues", []):
                        finding = self._create_finding(
                            audit_report.audit_id,
                            "compliance_violation",
                            AuditSeverity(issue.get("severity", "medium")),
                            f"{framework} compliance issue",
                            issue.get("description", "Compliance requirement not met"),
                            issue,
                            f"Non-compliance with {framework} requirements",
                            issue.get("recommendations", [f"Address {framework} compliance issue"]),
                            [framework]
                        )
                        audit_report.findings.append(finding)
            
            # Overall compliance status
            compliance_issues = [f for f in audit_report.findings if f.finding_type == "compliance_violation"]
            audit_report.compliance_status = "non_compliant" if compliance_issues else "compliant"
            
        except Exception as e:
            logger.error(f"Error in compliance audit: {e}")
            raise
    
    async def _conduct_explainability_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct explainability audit"""
        try:
            logger.info(f"Conducting explainability audit for {audit_report.model_id}")
            
            # Test explanation capabilities
            explainability_assessment = await self._assess_explainability(audit_report.model_id)
            
            # Check if explanations are available
            if not explainability_assessment.get("explanation_available", False):
                finding = self._create_finding(
                    audit_report.audit_id,
                    "explainability_issue",
                    AuditSeverity.HIGH,
                    "No explanation capability",
                    "Model does not provide explanations for its decisions",
                    explainability_assessment,
                    "Lack of explainability reduces trust and compliance",
                    [
                        "Implement explanation mechanisms",
                        "Use interpretable model architectures",
                        "Add post-hoc explanation methods"
                    ],
                    ["GDPR", "SOC2"]
                )
                audit_report.findings.append(finding)
            
            # Check explanation quality
            explanation_quality = explainability_assessment.get("quality_score", 0.0)
            if explanation_quality < 0.7:
                finding = self._create_finding(
                    audit_report.audit_id,
                    "explainability_issue",
                    AuditSeverity.MEDIUM,
                    "Poor explanation quality",
                    f"Explanation quality score ({explanation_quality:.2f}) is below acceptable threshold",
                    explainability_assessment,
                    "Poor explanation quality reduces effectiveness and trust",
                    [
                        "Improve explanation algorithms",
                        "Enhance explanation presentation",
                        "Validate explanations with domain experts"
                    ],
                    ["GDPR"]
                )
                audit_report.findings.append(finding)
            
        except Exception as e:
            logger.error(f"Error in explainability audit: {e}")
            raise
    
    async def _conduct_comprehensive_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct comprehensive audit covering all areas"""
        try:
            logger.info(f"Conducting comprehensive audit for {audit_report.model_id}")
            
            # Run all audit types
            await self._conduct_performance_audit(audit_report, config)
            await self._conduct_bias_audit(audit_report, config)
            await self._conduct_security_audit(audit_report, config)
            await self._conduct_compliance_audit(audit_report, config)
            await self._conduct_explainability_audit(audit_report, config)
            
            # Additional comprehensive checks
            await self._conduct_data_quality_audit(audit_report, config)
            await self._conduct_behavioral_audit(audit_report, config)
            
        except Exception as e:
            logger.error(f"Error in comprehensive audit: {e}")
            raise
    
    async def _conduct_data_quality_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct data quality audit"""
        # Simulate data quality assessment
        data_quality_issues = [
            {
                "type": "missing_data",
                "severity": "medium",
                "description": "Missing values detected in training data",
                "impact": "May affect model performance and bias"
            }
        ]
        
        for issue in data_quality_issues:
            finding = self._create_finding(
                audit_report.audit_id,
                "data_quality_issue",
                AuditSeverity(issue["severity"]),
                f"Data quality issue: {issue['type']}",
                issue["description"],
                issue,
                issue["impact"],
                ["Review data preprocessing", "Implement data quality checks"],
                ["SOC2"]
            )
            audit_report.findings.append(finding)
    
    async def _conduct_behavioral_audit(self, audit_report: ModelAuditReport, config: AuditConfiguration):
        """Conduct behavioral audit"""
        # Simulate behavioral pattern analysis
        behavioral_issues = []
        
        # Check for edge case handling
        edge_case_performance = 0.65  # Simulated
        if edge_case_performance < 0.7:
            behavioral_issues.append({
                "type": "poor_edge_case_handling",
                "severity": "medium",
                "description": f"Poor performance on edge cases ({edge_case_performance:.2f})",
                "impact": "Model may behave unpredictably in unusual scenarios"
            })
        
        for issue in behavioral_issues:
            finding = self._create_finding(
                audit_report.audit_id,
                "behavioral_issue",
                AuditSeverity(issue["severity"]),
                issue["type"],
                issue["description"],
                issue,
                issue["impact"],
                ["Improve edge case handling", "Expand test coverage"],
                ["SOC2"]
            )
            audit_report.findings.append(finding)
    
    def _create_finding(self, audit_id: str, finding_type: str, severity: AuditSeverity,
                       title: str, description: str, evidence: Dict[str, Any],
                       impact_assessment: str, recommendations: List[str],
                       compliance_frameworks_affected: List[str]) -> AuditFinding:
        """Create an audit finding"""
        finding_id = f"finding_{audit_id}_{len(self.audit_findings)}"
        
        finding = AuditFinding(
            finding_id=finding_id,
            audit_id=audit_id,
            finding_type=finding_type,
            severity=severity,
            title=title,
            description=description,
            evidence=evidence,
            impact_assessment=impact_assessment,
            recommendations=recommendations,
            compliance_frameworks_affected=compliance_frameworks_affected,
            detected_at=datetime.now(timezone.utc),
            category=finding_type.split("_")[0],  # Extract category from type
            automated=True,
            verified=False,
            metadata={
                "auto_generated": True,
                "requires_manual_review": severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]
            }
        )
        
        self.audit_findings[finding_id] = finding
        return finding
    
    # Mock data collection methods (would integrate with actual model monitoring)
    
    async def _collect_performance_metrics(self, model_id: str) -> Dict[str, float]:
        """Collect performance metrics for a model"""
        # Simulate performance metrics
        return {
            "accuracy": 0.87 + np.random.uniform(-0.1, 0.05),
            "precision": 0.84 + np.random.uniform(-0.1, 0.05),
            "recall": 0.82 + np.random.uniform(-0.1, 0.05),
            "f1_score": 0.83 + np.random.uniform(-0.1, 0.05),
            "auc_roc": 0.89 + np.random.uniform(-0.05, 0.05),
            "response_time": 1.2 + np.random.uniform(-0.5, 1.0),
            "throughput": 150 + np.random.uniform(-50, 50),
            "error_rate": 0.03 + np.random.uniform(-0.01, 0.02)
        }
    
    async def _detect_model_bias(self, model_id: str) -> Dict[str, Dict[str, Any]]:
        """Detect bias in model predictions"""
        return {
            "demographic_parity": {
                "score": 0.08 + np.random.uniform(-0.03, 0.05),
                "affected_groups": ["group_a", "group_b"],
                "details": "Slight difference in positive prediction rates"
            },
            "equalized_odds": {
                "score": 0.05 + np.random.uniform(-0.02, 0.08),
                "affected_groups": ["group_a"],
                "details": "Minor differences in true positive rates"
            }
        }
    
    async def _assess_model_security(self, model_id: str) -> Dict[str, Any]:
        """Assess model security"""
        return {
            "overall_status": "secure",
            "vulnerabilities": [
                {
                    "title": "Input validation weakness",
                    "severity": "low",
                    "description": "Model accepts unusually formatted inputs",
                    "impact": "Potential for unexpected behavior",
                    "recommendations": ["Implement strict input validation", "Add input sanitization"]
                }
            ]
        }
    
    async def _check_framework_compliance(self, model_id: str, framework: str) -> Dict[str, Any]:
        """Check compliance with specific framework"""
        # Simulate compliance check
        compliant = np.random.random() > 0.2  # 80% chance of compliance
        
        if compliant:
            return {"compliant": True, "issues": []}
        else:
            return {
                "compliant": False,
                "issues": [
                    {
                        "severity": "medium",
                        "description": f"{framework} documentation incomplete",
                        "recommendations": [f"Complete {framework} documentation"]
                    }
                ]
            }
    
    async def _assess_explainability(self, model_id: str) -> Dict[str, Any]:
        """Assess model explainability"""
        return {
            "explanation_available": True,
            "quality_score": 0.75 + np.random.uniform(-0.1, 0.15),
            "methods_supported": ["feature_importance", "attention_visualization"],
            "stakeholder_comprehension": 0.70
        }
    
    def _generate_overall_assessment(self, audit_report: ModelAuditReport) -> str:
        """Generate overall assessment based on findings"""
        critical_findings = [f for f in audit_report.findings if f.severity == AuditSeverity.CRITICAL]
        high_findings = [f for f in audit_report.findings if f.severity == AuditSeverity.HIGH]
        
        if critical_findings:
            return f"CRITICAL ISSUES FOUND: {len(critical_findings)} critical findings require immediate attention"
        elif high_findings:
            return f"HIGH PRIORITY ISSUES: {len(high_findings)} high-severity findings need resolution"
        elif audit_report.findings:
            return f"MINOR ISSUES: {len(audit_report.findings)} findings identified for improvement"
        else:
            return "COMPLIANT: No significant issues found"
    
    def _generate_audit_recommendations(self, audit_report: ModelAuditReport) -> List[str]:
        """Generate recommendations based on audit findings"""
        recommendations = []
        
        # Extract recommendations from findings
        for finding in audit_report.findings:
            recommendations.extend(finding.recommendations)
        
        # Remove duplicates and add general recommendations
        unique_recommendations = list(set(recommendations))
        
        # Add general recommendations
        unique_recommendations.extend([
            "Implement continuous monitoring for all audit areas",
            "Schedule regular audit reviews",
            "Document all remediation actions taken",
            "Provide training on identified improvement areas"
        ])
        
        return unique_recommendations[:10]  # Limit to top 10
    
    async def _schedule_audit(self, model_id: str, audit_type: AuditType):
        """Schedule an audit for a model"""
        # This would integrate with a job scheduler in production
        logger.info(f"Audit scheduled: {audit_type.value} for model {model_id}")
    
    async def _audit_scheduler(self):
        """Background audit scheduler"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check for overdue audits
                for config_id, config in self.audit_configurations.items():
                    model_id = config.model_id
                    
                    # Check if audit is due
                    last_audit = self._get_last_audit_date(model_id)
                    if last_audit is None or datetime.now(timezone.utc) - last_audit >= config.frequency:
                        # Schedule audit
                        for audit_type in config.audit_types:
                            await self._schedule_audit(model_id, audit_type)
                
            except Exception as e:
                logger.error(f"Error in audit scheduler: {e}")
                await asyncio.sleep(300)
    
    def _get_last_audit_date(self, model_id: str) -> Optional[datetime]:
        """Get the date of the last audit for a model"""
        model_audits = [r for r in self.audit_reports if r.model_id == model_id and r.audit_status == AuditStatus.COMPLETED]
        if model_audits:
            return max(audit.completed_at for audit in model_audits)
        return None
    
    # Public API methods
    
    def get_audit_history(self, model_id: str = None, limit: int = 50) -> List[ModelAuditReport]:
        """Get audit history"""
        audits = self.audit_reports
        if model_id:
            audits = [a for a in audits if a.model_id == model_id]
        return sorted(audits, key=lambda x: x.started_at, reverse=True)[:limit]
    
    def get_audit_findings(self, audit_id: str = None, severity: AuditSeverity = None) -> List[AuditFinding]:
        """Get audit findings"""
        findings = list(self.audit_findings.values())
        if audit_id:
            findings = [f for f in findings if f.audit_id == audit_id]
        if severity:
            findings = [f for f in findings if f.severity == severity]
        return findings
    
    async def generate_audit_summary(self, model_id: str = None, 
                                   start_date: datetime = None,
                                   end_date: datetime = None) -> Dict[str, Any]:
        """Generate audit summary report"""
        audits = self.get_audit_history(model_id)
        
        if start_date:
            audits = [a for a in audits if a.started_at >= start_date]
        if end_date:
            audits = [a for a in audits if a.started_at <= end_date]
        
        all_findings = []
        for audit in audits:
            all_findings.extend(audit.findings)
        
        return {
            "summary_id": f"audit_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "model_id": model_id or "all_models",
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_audits": len(audits),
            "total_findings": len(all_findings),
            "findings_by_severity": {
                severity.value: len([f for f in all_findings if f.severity == severity])
                for severity in AuditSeverity
            },
            "findings_by_type": dict(Counter([f.finding_type for f in all_findings])),
            "audit_types_performed": dict(Counter([a.audit_type.value for a in audits])),
            "compliance_frameworks_affected": list(set([
                framework
                for finding in all_findings
                for framework in finding.compliance_frameworks_affected
            ])),
            "top_recommendations": self._get_top_recommendations(all_findings),
            "audit_frequency_compliance": self._calculate_audit_frequency_compliance()
        }
    
    def _get_top_recommendations(self, findings: List[AuditFinding]) -> List[Dict[str, Any]]:
        """Get top recommendations from findings"""
        recommendation_counts = Counter()
        for finding in findings:
            for rec in finding.recommendations:
                recommendation_counts[rec] += 1
        
        return [
            {"recommendation": rec, "frequency": count}
            for rec, count in recommendation_counts.most_common(10)
        ]
    
    def _calculate_audit_frequency_compliance(self) -> Dict[str, Any]:
        """Calculate audit frequency compliance"""
        total_configs = len(self.audit_configurations)
        if total_configs == 0:
            return {"compliance_rate": 0, "overdue_audits": 0}
        
        overdue_count = 0
        for config in self.audit_configurations.values():
            last_audit = self._get_last_audit_date(config.model_id)
            if last_audit is None or datetime.now(timezone.utc) - last_audit > config.frequency:
                overdue_count += 1
        
        compliance_rate = (total_configs - overdue_count) / total_configs
        
        return {
            "compliance_rate": compliance_rate,
            "overdue_audits": overdue_count,
            "total_models": total_configs
        }

# Global instance
_global_model_auditing_service = None

def get_model_auditing_service() -> ModelAuditingService:
    """Get global model auditing service instance"""
    global _global_model_auditing_service
    if _global_model_auditing_service is None:
        _global_model_auditing_service = ModelAuditingService()
    return _global_model_auditing_service