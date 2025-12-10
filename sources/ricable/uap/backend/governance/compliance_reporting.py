# File: backend/governance/compliance_reporting.py
"""
AI Compliance Reporting System

Automated compliance reporting for AI governance, integrating with existing
compliance frameworks and generating comprehensive reports for regulatory
and audit purposes.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
import statistics
from collections import defaultdict, Counter

# Import existing systems
from ..compliance.compliance_manager import get_compliance_manager, ComplianceFramework, ComplianceStatus
from ..security.audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of compliance reports"""
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    AI_GOVERNANCE_SUMMARY = "ai_governance_summary"
    BIAS_ASSESSMENT_REPORT = "bias_assessment_report"
    RISK_MANAGEMENT_REPORT = "risk_management_report"
    AUDIT_READINESS_REPORT = "audit_readiness_report"
    STAKEHOLDER_TRANSPARENCY = "stakeholder_transparency"
    PERFORMANCE_COMPLIANCE = "performance_compliance"
    DATA_GOVERNANCE_REPORT = "data_governance_report"
    EXECUTIVE_DASHBOARD = "executive_dashboard"

class ReportFrequency(Enum):
    """Report generation frequency"""
    ON_DEMAND = "on_demand"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class ComplianceMetric(Enum):
    """Compliance metrics for reporting"""
    OVERALL_COMPLIANCE_RATE = "overall_compliance_rate"
    BIAS_DETECTION_RATE = "bias_detection_rate"
    AUDIT_COMPLETION_RATE = "audit_completion_rate"
    RISK_MITIGATION_RATE = "risk_mitigation_rate"
    GOVERNANCE_COVERAGE = "governance_coverage"
    POLICY_ADHERENCE = "policy_adherence"
    TRAINING_COMPLETION = "training_completion"
    INCIDENT_RESOLUTION_TIME = "incident_resolution_time"

@dataclass
class ComplianceKPI:
    """Key Performance Indicator for compliance"""
    kpi_id: str
    name: str
    description: str
    metric_type: ComplianceMetric
    target_value: float
    current_value: float
    unit: str
    trend: str  # improving, stable, declining
    last_updated: datetime
    data_source: str
    calculation_method: str
    metadata: Dict[str, Any]

@dataclass
class ComplianceReportSection:
    """Section within a compliance report"""
    section_id: str
    title: str
    content_type: str  # text, table, chart, metrics
    content: Dict[str, Any]
    compliance_status: str
    recommendations: List[str]
    action_items: List[str]
    metadata: Dict[str, Any]

@dataclass
class AIComplianceReport:
    """Comprehensive AI compliance report"""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    reporting_period_start: datetime
    reporting_period_end: datetime
    scope: Dict[str, Any]
    executive_summary: str
    sections: List[ComplianceReportSection]
    kpis: List[ComplianceKPI]
    overall_compliance_score: float
    compliance_frameworks: List[str]
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    next_actions: List[Dict[str, Any]]
    attestations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class ReportTemplate:
    """Template for generating reports"""
    template_id: str
    report_type: ReportType
    title_template: str
    sections: List[Dict[str, Any]]
    required_data_sources: List[str]
    compliance_frameworks: List[str]
    frequency: ReportFrequency
    stakeholders: List[str]
    approval_required: bool
    metadata: Dict[str, Any]

class AIComplianceReporter:
    """
    Comprehensive AI Compliance Reporting System
    
    Provides automated generation of compliance reports for AI governance:
    - Regulatory compliance reports (GDPR, HIPAA, SOC2, etc.)
    - AI governance summaries for stakeholders
    - Bias and fairness assessment reports
    - Risk management reports
    - Audit readiness documentation
    - Executive dashboards and metrics
    """
    
    def __init__(self):
        self.reports: List[AIComplianceReport] = []
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.scheduled_reports: Dict[str, Dict[str, Any]] = {}
        
        # Integration with existing systems
        self.compliance_manager = get_compliance_manager()
        self.audit_logger = get_security_audit_logger()
        
        # Initialize report templates
        self._initialize_report_templates()
        
        # KPI tracking
        self.compliance_kpis: Dict[str, ComplianceKPI] = {}
        self._initialize_compliance_kpis()
        
        # Report configuration
        self.report_config = {
            "auto_generate_monthly": True,
            "auto_generate_quarterly": True,
            "executive_reports_enabled": True,
            "stakeholder_notifications": True,
            "report_retention_days": 2555,  # 7 years
            "encryption_required": True
        }
        
        # Start background report scheduler
        asyncio.create_task(self._report_scheduler())
        
        logger.info("AI Compliance Reporter initialized")
    
    def _initialize_report_templates(self):
        """Initialize report templates for different compliance needs"""
        
        # Regulatory Compliance Report Template
        self.report_templates["regulatory_compliance"] = ReportTemplate(
            template_id="regulatory_compliance",
            report_type=ReportType.REGULATORY_COMPLIANCE,
            title_template="AI Regulatory Compliance Report - {period}",
            sections=[
                {
                    "id": "executive_summary",
                    "title": "Executive Summary",
                    "content_type": "text",
                    "required": True
                },
                {
                    "id": "compliance_overview",
                    "title": "Compliance Framework Overview",
                    "content_type": "metrics",
                    "required": True
                },
                {
                    "id": "ai_model_inventory",
                    "title": "AI Model Inventory and Classification",
                    "content_type": "table",
                    "required": True
                },
                {
                    "id": "risk_assessment",
                    "title": "Risk Assessment Summary",
                    "content_type": "table",
                    "required": True
                },
                {
                    "id": "bias_fairness",
                    "title": "Bias and Fairness Assessment",
                    "content_type": "metrics",
                    "required": True
                },
                {
                    "id": "data_governance",
                    "title": "Data Governance and Privacy",
                    "content_type": "text",
                    "required": True
                },
                {
                    "id": "audit_trail",
                    "title": "Audit Trail and Documentation",
                    "content_type": "text",
                    "required": True
                },
                {
                    "id": "incidents_violations",
                    "title": "Incidents and Violations",
                    "content_type": "table",
                    "required": True
                },
                {
                    "id": "remediation_actions",
                    "title": "Remediation Actions and Timeline",
                    "content_type": "table",
                    "required": True
                }
            ],
            required_data_sources=["governance_manager", "risk_assessment", "audit_trail"],
            compliance_frameworks=["GDPR", "HIPAA", "SOC2", "AI_Act"],
            frequency=ReportFrequency.QUARTERLY,
            stakeholders=["compliance_officer", "ciso", "dpo", "executive_team"],
            approval_required=True,
            metadata={"priority": "high", "regulatory": True}
        )
        
        # AI Governance Summary Template
        self.report_templates["ai_governance_summary"] = ReportTemplate(
            template_id="ai_governance_summary",
            report_type=ReportType.AI_GOVERNANCE_SUMMARY,
            title_template="AI Governance Summary Report - {period}",
            sections=[
                {
                    "id": "governance_metrics",
                    "title": "Governance Metrics Dashboard",
                    "content_type": "metrics",
                    "required": True
                },
                {
                    "id": "model_lifecycle",
                    "title": "Model Lifecycle Management",
                    "content_type": "table",
                    "required": True
                },
                {
                    "id": "policy_compliance",
                    "title": "Policy Compliance Status",
                    "content_type": "metrics",
                    "required": True
                },
                {
                    "id": "training_awareness",
                    "title": "Training and Awareness Programs",
                    "content_type": "text",
                    "required": False
                }
            ],
            required_data_sources=["governance_manager", "model_auditing"],
            compliance_frameworks=["Internal_Policies"],
            frequency=ReportFrequency.MONTHLY,
            stakeholders=["ai_team", "governance_committee", "management"],
            approval_required=False,
            metadata={"internal": True, "dashboard": True}
        )
        
        # Executive Dashboard Template
        self.report_templates["executive_dashboard"] = ReportTemplate(
            template_id="executive_dashboard",
            report_type=ReportType.EXECUTIVE_DASHBOARD,
            title_template="AI Governance Executive Dashboard - {period}",
            sections=[
                {
                    "id": "key_metrics",
                    "title": "Key Performance Indicators",
                    "content_type": "metrics",
                    "required": True
                },
                {
                    "id": "risk_summary",
                    "title": "Risk Summary",
                    "content_type": "chart",
                    "required": True
                },
                {
                    "id": "compliance_status",
                    "title": "Compliance Status",
                    "content_type": "metrics",
                    "required": True
                },
                {
                    "id": "strategic_recommendations",
                    "title": "Strategic Recommendations",
                    "content_type": "text",
                    "required": True
                }
            ],
            required_data_sources=["governance_manager", "risk_assessment", "compliance_manager"],
            compliance_frameworks=["All"],
            frequency=ReportFrequency.MONTHLY,
            stakeholders=["ceo", "cto", "board_of_directors"],
            approval_required=False,
            metadata={"executive": True, "high_level": True}
        )
    
    def _initialize_compliance_kpis(self):
        """Initialize compliance KPIs for tracking"""
        
        kpis = [
            ComplianceKPI(
                kpi_id="overall_compliance_rate",
                name="Overall Compliance Rate",
                description="Percentage of AI models meeting all compliance requirements",
                metric_type=ComplianceMetric.OVERALL_COMPLIANCE_RATE,
                target_value=95.0,
                current_value=0.0,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(timezone.utc),
                data_source="governance_manager",
                calculation_method="(compliant_models / total_models) * 100",
                metadata={"critical": True, "board_metric": True}
            ),
            
            ComplianceKPI(
                kpi_id="bias_detection_rate",
                name="Bias Detection and Mitigation Rate",
                description="Percentage of models with completed bias assessment and mitigation",
                metric_type=ComplianceMetric.BIAS_DETECTION_RATE,
                target_value=100.0,
                current_value=0.0,
                unit="percentage",
                trend="improving",
                last_updated=datetime.now(timezone.utc),
                data_source="explainability_engine",
                calculation_method="(models_with_bias_assessment / total_models) * 100",
                metadata={"ethical": True, "regulatory_required": True}
            ),
            
            ComplianceKPI(
                kpi_id="audit_completion_rate",
                name="Audit Completion Rate",
                description="Percentage of scheduled audits completed on time",
                metric_type=ComplianceMetric.AUDIT_COMPLETION_RATE,
                target_value=98.0,
                current_value=0.0,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(timezone.utc),
                data_source="model_auditing_service",
                calculation_method="(completed_audits / scheduled_audits) * 100",
                metadata={"operational": True, "compliance_required": True}
            ),
            
            ComplianceKPI(
                kpi_id="risk_mitigation_rate",
                name="Risk Mitigation Implementation Rate",
                description="Percentage of identified risks with active mitigation strategies",
                metric_type=ComplianceMetric.RISK_MITIGATION_RATE,
                target_value=90.0,
                current_value=0.0,
                unit="percentage",
                trend="improving",
                last_updated=datetime.now(timezone.utc),
                data_source="risk_assessment",
                calculation_method="(mitigated_risks / total_identified_risks) * 100",
                metadata={"risk_management": True, "strategic": True}
            ),
            
            ComplianceKPI(
                kpi_id="governance_coverage",
                name="AI Governance Coverage",
                description="Percentage of AI models under governance framework",
                metric_type=ComplianceMetric.GOVERNANCE_COVERAGE,
                target_value=100.0,
                current_value=0.0,
                unit="percentage",
                trend="improving",
                last_updated=datetime.now(timezone.utc),
                data_source="governance_manager",
                calculation_method="(governed_models / total_ai_models) * 100",
                metadata={"foundational": True, "mandatory": True}
            )
        ]
        
        for kpi in kpis:
            self.compliance_kpis[kpi.kpi_id] = kpi
    
    async def generate_compliance_report(self, report_type: ReportType,
                                       reporting_period_start: datetime = None,
                                       reporting_period_end: datetime = None,
                                       scope: Dict[str, Any] = None,
                                       compliance_frameworks: List[str] = None,
                                       requester: str = "automated_system") -> AIComplianceReport:
        """
        Generate a comprehensive compliance report.
        
        Args:
            report_type: Type of report to generate
            reporting_period_start: Start of reporting period
            reporting_period_end: End of reporting period
            scope: Scope of the report (models, frameworks, etc.)
            compliance_frameworks: Specific frameworks to include
            requester: Who requested the report
            
        Returns:
            Generated compliance report
        """
        try:
            report_id = f"report_{report_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            # Set default reporting period if not provided
            if reporting_period_end is None:
                reporting_period_end = datetime.now(timezone.utc)
            
            if reporting_period_start is None:
                reporting_period_start = reporting_period_end - timedelta(days=90)  # Default to quarterly
            
            logger.info(f"Generating {report_type.value} compliance report {report_id}")
            
            # Get report template
            template = self.report_templates.get(report_type.value)
            if not template:
                raise ValueError(f"No template found for report type: {report_type.value}")
            
            # Generate report title
            period_str = f"{reporting_period_start.strftime('%Y-%m-%d')} to {reporting_period_end.strftime('%Y-%m-%d')}"
            title = template.title_template.format(period=period_str)
            
            # Collect data from required sources
            report_data = await self._collect_report_data(
                template.required_data_sources,
                reporting_period_start,
                reporting_period_end,
                scope
            )
            
            # Generate report sections
            sections = await self._generate_report_sections(
                template.sections,
                report_data,
                compliance_frameworks or template.compliance_frameworks
            )
            
            # Update KPIs
            await self._update_compliance_kpis(report_data)
            
            # Get relevant KPIs for this report
            report_kpis = [
                kpi for kpi in self.compliance_kpis.values()
                if self._is_kpi_relevant(kpi, report_type, compliance_frameworks)
            ]
            
            # Calculate overall compliance score
            overall_compliance_score = self._calculate_overall_compliance_score(report_data, sections)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                report_type, overall_compliance_score, report_data, sections
            )
            
            # Generate findings
            findings = self._extract_report_findings(sections, report_data)
            
            # Generate recommendations
            recommendations = self._generate_report_recommendations(findings, overall_compliance_score)
            
            # Generate next actions
            next_actions = self._generate_next_actions(findings, recommendations)
            
            # Create compliance report
            report = AIComplianceReport(
                report_id=report_id,
                report_type=report_type,
                title=title,
                generated_at=datetime.now(timezone.utc),
                reporting_period_start=reporting_period_start,
                reporting_period_end=reporting_period_end,
                scope=scope or {"all_models": True},
                executive_summary=executive_summary,
                sections=sections,
                kpis=report_kpis,
                overall_compliance_score=overall_compliance_score,
                compliance_frameworks=compliance_frameworks or template.compliance_frameworks,
                findings=findings,
                recommendations=recommendations,
                next_actions=next_actions,
                attestations=[],  # Would be populated with actual attestations
                metadata={
                    "template_id": template.template_id,
                    "requester": requester,
                    "generated_by": "ai_compliance_reporter",
                    "report_version": "1.0",
                    "data_sources": template.required_data_sources
                }
            )
            
            # Store report
            self.reports.append(report)
            
            # Log report generation
            await self.audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_CHANGE,
                outcome=AuditOutcome.SUCCESS,
                actor_id=requester,
                actor_type="user" if requester != "automated_system" else "system",
                resource="ai_compliance_system",
                action="generate_report",
                description=f"Compliance report generated: {report_type.value}",
                details={
                    "report_id": report_id,
                    "report_type": report_type.value,
                    "compliance_score": overall_compliance_score,
                    "sections_count": len(sections),
                    "findings_count": len(findings)
                },
                risk_score=3
            )
            
            uap_logger.log_ai_event(
                f"Compliance report generated: {report_type.value}",
                metadata={
                    "report_id": report_id,
                    "compliance_score": overall_compliance_score,
                    "period": period_str
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise
    
    async def _collect_report_data(self, data_sources: List[str],
                                 start_date: datetime, end_date: datetime,
                                 scope: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from various sources for report generation"""
        report_data = {}
        
        for source in data_sources:
            try:
                if source == "governance_manager":
                    report_data["governance"] = await self._collect_governance_data(start_date, end_date, scope)
                elif source == "risk_assessment":
                    report_data["risk"] = await self._collect_risk_data(start_date, end_date, scope)
                elif source == "model_auditing":
                    report_data["auditing"] = await self._collect_auditing_data(start_date, end_date, scope)
                elif source == "compliance_manager":
                    report_data["compliance"] = await self._collect_compliance_data(start_date, end_date, scope)
                elif source == "audit_trail":
                    report_data["audit_trail"] = await self._collect_audit_trail_data(start_date, end_date, scope)
                
            except Exception as e:
                logger.warning(f"Failed to collect data from source {source}: {e}")
                report_data[source] = {"error": str(e), "data_available": False}
        
        return report_data
    
    async def _collect_governance_data(self, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Collect governance data"""
        # This would integrate with the actual governance manager
        return {
            "total_models": 15,
            "compliant_models": 12,
            "non_compliant_models": 3,
            "models_under_review": 2,
            "governance_coverage": 0.85,
            "policy_violations": 5,
            "approval_pending": 1,
            "models_by_risk_level": {
                "low": 8,
                "medium": 5,
                "high": 2,
                "critical": 0
            },
            "frameworks_coverage": {
                "SOC2": 15,
                "GDPR": 12,
                "HIPAA": 3,
                "AI_Act": 8
            }
        }
    
    async def _collect_risk_data(self, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Collect risk assessment data"""
        return {
            "total_risk_assessments": 12,
            "high_risk_models": 2,
            "medium_risk_models": 6,
            "low_risk_models": 4,
            "average_risk_score": 4.2,
            "risk_categories": {
                "technical": 3.8,
                "ethical": 4.5,
                "operational": 3.2,
                "compliance": 4.8,
                "security": 3.5
            },
            "mitigation_strategies": {
                "planned": 15,
                "in_progress": 8,
                "completed": 12
            },
            "overdue_assessments": 2
        }
    
    async def _collect_auditing_data(self, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Collect auditing data"""
        return {
            "total_audits": 18,
            "completed_audits": 16,
            "pending_audits": 2,
            "failed_audits": 0,
            "audit_types": {
                "performance": 8,
                "bias": 6,
                "security": 4,
                "compliance": 5
            },
            "findings": {
                "critical": 1,
                "high": 4,
                "medium": 12,
                "low": 8
            },
            "average_audit_score": 7.8,
            "overdue_audits": 1
        }
    
    async def _collect_compliance_data(self, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Collect compliance framework data"""
        return {
            "frameworks": {
                "SOC2": {
                    "compliance_rate": 0.92,
                    "last_assessment": "2024-05-15",
                    "next_review": "2024-08-15",
                    "status": "compliant"
                },
                "GDPR": {
                    "compliance_rate": 0.88,
                    "last_assessment": "2024-06-01",
                    "next_review": "2024-09-01",
                    "status": "partially_compliant"
                },
                "HIPAA": {
                    "compliance_rate": 0.95,
                    "last_assessment": "2024-05-30",
                    "next_review": "2024-08-30",
                    "status": "compliant"
                }
            },
            "overall_compliance_rate": 0.92,
            "compliance_gaps": 8,
            "remediation_items": 5
        }
    
    async def _collect_audit_trail_data(self, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Collect audit trail data"""
        return {
            "total_events": 1247,
            "security_events": 58,
            "authentication_events": 892,
            "data_access_events": 156,
            "system_changes": 89,
            "failed_attempts": 12,
            "audit_coverage": 0.98,
            "integrity_violations": 0
        }
    
    async def _generate_report_sections(self, section_templates: List[Dict[str, Any]],
                                      report_data: Dict[str, Any],
                                      compliance_frameworks: List[str]) -> List[ComplianceReportSection]:
        """Generate report sections based on templates and data"""
        sections = []
        
        for template in section_templates:
            section = await self._generate_section(template, report_data, compliance_frameworks)
            sections.append(section)
        
        return sections
    
    async def _generate_section(self, template: Dict[str, Any],
                              report_data: Dict[str, Any],
                              compliance_frameworks: List[str]) -> ComplianceReportSection:
        """Generate a single report section"""
        section_id = template["id"]
        title = template["title"]
        content_type = template["content_type"]
        
        # Generate content based on section type
        if section_id == "executive_summary":
            content = await self._generate_executive_summary_content(report_data)
            compliance_status = "compliant" if report_data.get("governance", {}).get("governance_coverage", 0) > 0.8 else "non_compliant"
        
        elif section_id == "compliance_overview":
            content = self._generate_compliance_overview_content(report_data, compliance_frameworks)
            compliance_status = "compliant" if report_data.get("compliance", {}).get("overall_compliance_rate", 0) > 0.85 else "partially_compliant"
        
        elif section_id == "ai_model_inventory":
            content = self._generate_model_inventory_content(report_data)
            compliance_status = "compliant"
        
        elif section_id == "risk_assessment":
            content = self._generate_risk_assessment_content(report_data)
            avg_risk = report_data.get("risk", {}).get("average_risk_score", 5.0)
            compliance_status = "compliant" if avg_risk < 5.0 else "requires_attention"
        
        elif section_id == "bias_fairness":
            content = self._generate_bias_fairness_content(report_data)
            compliance_status = "compliant"  # Would be calculated based on actual bias data
        
        elif section_id == "governance_metrics":
            content = self._generate_governance_metrics_content(report_data)
            compliance_status = "compliant" if report_data.get("governance", {}).get("governance_coverage", 0) > 0.8 else "non_compliant"
        
        else:
            # Generic content generation
            content = self._generate_generic_content(section_id, report_data)
            compliance_status = "compliant"
        
        # Generate recommendations and action items
        recommendations = self._generate_section_recommendations(section_id, content, compliance_status)
        action_items = self._generate_section_action_items(section_id, compliance_status)
        
        return ComplianceReportSection(
            section_id=section_id,
            title=title,
            content_type=content_type,
            content=content,
            compliance_status=compliance_status,
            recommendations=recommendations,
            action_items=action_items,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data_quality": "good",
                "completeness": 1.0
            }
        )
    
    async def _generate_executive_summary_content(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary content"""
        governance_data = report_data.get("governance", {})
        compliance_data = report_data.get("compliance", {})
        risk_data = report_data.get("risk", {})
        
        total_models = governance_data.get("total_models", 0)
        compliant_models = governance_data.get("compliant_models", 0)
        compliance_rate = (compliant_models / total_models * 100) if total_models > 0 else 0
        
        summary_text = f"""
        This report provides a comprehensive overview of AI governance and compliance for the reporting period.
        
        Key Highlights:
        • {total_models} AI models under governance framework
        • {compliance_rate:.1f}% overall compliance rate
        • {risk_data.get('high_risk_models', 0)} models identified as high-risk
        • {compliance_data.get('compliance_gaps', 0)} compliance gaps requiring attention
        
        The organization demonstrates a strong commitment to responsible AI with robust governance 
        frameworks in place. Areas for improvement include bias testing coverage and audit completion rates.
        """
        
        return {
            "summary_text": summary_text.strip(),
            "key_metrics": {
                "total_models": total_models,
                "compliance_rate": compliance_rate,
                "high_risk_models": risk_data.get("high_risk_models", 0),
                "compliance_gaps": compliance_data.get("compliance_gaps", 0)
            }
        }
    
    def _generate_compliance_overview_content(self, report_data: Dict[str, Any], frameworks: List[str]) -> Dict[str, Any]:
        """Generate compliance overview content"""
        compliance_data = report_data.get("compliance", {})
        frameworks_data = compliance_data.get("frameworks", {})
        
        framework_status = {}
        for framework in frameworks:
            if framework in frameworks_data:
                framework_status[framework] = frameworks_data[framework]
            else:
                framework_status[framework] = {
                    "compliance_rate": 0.0,
                    "status": "not_assessed"
                }
        
        return {
            "frameworks": framework_status,
            "overall_compliance_rate": compliance_data.get("overall_compliance_rate", 0.0),
            "compliance_gaps": compliance_data.get("compliance_gaps", 0),
            "remediation_items": compliance_data.get("remediation_items", 0)
        }
    
    def _generate_model_inventory_content(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model inventory content"""
        governance_data = report_data.get("governance", {})
        
        return {
            "total_models": governance_data.get("total_models", 0),
            "models_by_status": {
                "compliant": governance_data.get("compliant_models", 0),
                "non_compliant": governance_data.get("non_compliant_models", 0),
                "under_review": governance_data.get("models_under_review", 0)
            },
            "models_by_risk_level": governance_data.get("models_by_risk_level", {}),
            "frameworks_coverage": governance_data.get("frameworks_coverage", {}),
            "governance_coverage": governance_data.get("governance_coverage", 0.0)
        }
    
    def _generate_risk_assessment_content(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment content"""
        risk_data = report_data.get("risk", {})
        
        return {
            "total_assessments": risk_data.get("total_risk_assessments", 0),
            "risk_distribution": {
                "high": risk_data.get("high_risk_models", 0),
                "medium": risk_data.get("medium_risk_models", 0),
                "low": risk_data.get("low_risk_models", 0)
            },
            "average_risk_score": risk_data.get("average_risk_score", 0.0),
            "risk_categories": risk_data.get("risk_categories", {}),
            "mitigation_status": risk_data.get("mitigation_strategies", {}),
            "overdue_assessments": risk_data.get("overdue_assessments", 0)
        }
    
    def _generate_bias_fairness_content(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bias and fairness content"""
        # This would integrate with actual bias detection data
        return {
            "bias_assessments_completed": 12,
            "models_with_bias_issues": 2,
            "fairness_metrics": {
                "demographic_parity": 0.08,
                "equalized_odds": 0.06,
                "calibration": 0.04
            },
            "mitigation_strategies_implemented": 8,
            "bias_monitoring_coverage": 0.92
        }
    
    def _generate_governance_metrics_content(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate governance metrics content"""
        governance_data = report_data.get("governance", {})
        auditing_data = report_data.get("auditing", {})
        
        return {
            "governance_coverage": governance_data.get("governance_coverage", 0.0),
            "policy_violations": governance_data.get("policy_violations", 0),
            "audit_completion_rate": (auditing_data.get("completed_audits", 0) / 
                                    max(auditing_data.get("total_audits", 1), 1)) * 100,
            "average_audit_score": auditing_data.get("average_audit_score", 0.0),
            "overdue_items": {
                "audits": auditing_data.get("overdue_audits", 0),
                "assessments": report_data.get("risk", {}).get("overdue_assessments", 0)
            }
        }
    
    def _generate_generic_content(self, section_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic content for sections"""
        return {
            "section_id": section_id,
            "status": "generated",
            "data_summary": "Section content would be generated based on specific requirements",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_section_recommendations(self, section_id: str, content: Dict[str, Any], status: str) -> List[str]:
        """Generate recommendations for a section"""
        recommendations = []
        
        if status == "non_compliant":
            recommendations.append(f"Address compliance issues identified in {section_id}")
        
        if section_id == "risk_assessment":
            avg_risk = content.get("average_risk_score", 0.0)
            if avg_risk > 5.0:
                recommendations.append("Implement comprehensive risk mitigation strategies")
        
        if section_id == "bias_fairness":
            if content.get("models_with_bias_issues", 0) > 0:
                recommendations.append("Address bias issues in affected models")
        
        # Add general recommendations
        recommendations.extend([
            f"Continue monitoring {section_id} metrics",
            f"Schedule regular reviews of {section_id} compliance"
        ])
        
        return recommendations
    
    def _generate_section_action_items(self, section_id: str, status: str) -> List[str]:
        """Generate action items for a section"""
        action_items = []
        
        if status in ["non_compliant", "requires_attention"]:
            action_items.extend([
                f"Create remediation plan for {section_id}",
                f"Assign responsible parties for {section_id} improvements",
                f"Set timeline for {section_id} compliance achievement"
            ])
        
        action_items.append(f"Schedule next {section_id} review")
        
        return action_items
    
    async def _update_compliance_kpis(self, report_data: Dict[str, Any]):
        """Update compliance KPIs based on report data"""
        governance_data = report_data.get("governance", {})
        auditing_data = report_data.get("auditing", {})
        compliance_data = report_data.get("compliance", {})
        
        # Update overall compliance rate
        if "overall_compliance_rate" in self.compliance_kpis:
            total_models = governance_data.get("total_models", 0)
            compliant_models = governance_data.get("compliant_models", 0)
            compliance_rate = (compliant_models / total_models * 100) if total_models > 0 else 0
            
            kpi = self.compliance_kpis["overall_compliance_rate"]
            previous_value = kpi.current_value
            kpi.current_value = compliance_rate
            kpi.last_updated = datetime.now(timezone.utc)
            
            # Update trend
            if compliance_rate > previous_value:
                kpi.trend = "improving"
            elif compliance_rate < previous_value:
                kpi.trend = "declining"
            else:
                kpi.trend = "stable"
        
        # Update audit completion rate
        if "audit_completion_rate" in self.compliance_kpis:
            total_audits = auditing_data.get("total_audits", 0)
            completed_audits = auditing_data.get("completed_audits", 0)
            completion_rate = (completed_audits / total_audits * 100) if total_audits > 0 else 0
            
            kpi = self.compliance_kpis["audit_completion_rate"]
            kpi.current_value = completion_rate
            kpi.last_updated = datetime.now(timezone.utc)
        
        # Update other KPIs similarly...
    
    def _is_kpi_relevant(self, kpi: ComplianceKPI, report_type: ReportType, frameworks: List[str]) -> bool:
        """Determine if a KPI is relevant for the report"""
        # All KPIs are relevant for comprehensive reports
        if report_type in [ReportType.REGULATORY_COMPLIANCE, ReportType.EXECUTIVE_DASHBOARD]:
            return True
        
        # Specific KPIs for specific report types
        if report_type == ReportType.AI_GOVERNANCE_SUMMARY:
            return kpi.metric_type in [
                ComplianceMetric.OVERALL_COMPLIANCE_RATE,
                ComplianceMetric.GOVERNANCE_COVERAGE,
                ComplianceMetric.POLICY_ADHERENCE
            ]
        
        if report_type == ReportType.RISK_MANAGEMENT_REPORT:
            return kpi.metric_type in [
                ComplianceMetric.RISK_MITIGATION_RATE,
                ComplianceMetric.AUDIT_COMPLETION_RATE
            ]
        
        return True
    
    def _calculate_overall_compliance_score(self, report_data: Dict[str, Any], sections: List[ComplianceReportSection]) -> float:
        """Calculate overall compliance score"""
        section_scores = []
        
        for section in sections:
            if section.compliance_status == "compliant":
                section_scores.append(1.0)
            elif section.compliance_status == "partially_compliant":
                section_scores.append(0.7)
            elif section.compliance_status == "requires_attention":
                section_scores.append(0.5)
            else:  # non_compliant
                section_scores.append(0.0)
        
        if section_scores:
            return statistics.mean(section_scores) * 100  # Convert to percentage
        else:
            return 0.0
    
    async def _generate_executive_summary(self, report_type: ReportType, compliance_score: float,
                                        report_data: Dict[str, Any], sections: List[ComplianceReportSection]) -> str:
        """Generate executive summary"""
        governance_data = report_data.get("governance", {})
        risk_data = report_data.get("risk", {})
        
        total_models = governance_data.get("total_models", 0)
        high_risk_models = risk_data.get("high_risk_models", 0)
        
        # Generate summary based on compliance score
        if compliance_score >= 90:
            status_text = "demonstrates excellent AI governance and compliance"
        elif compliance_score >= 80:
            status_text = "maintains good AI governance with minor areas for improvement"
        elif compliance_score >= 70:
            status_text = "shows adequate AI governance but requires attention in several areas"
        else:
            status_text = "needs significant improvement in AI governance and compliance"
        
        summary = f"""
        EXECUTIVE SUMMARY
        
        The organization {status_text} with an overall compliance score of {compliance_score:.1f}%.
        
        KEY METRICS:
        • {total_models} AI models under governance
        • {high_risk_models} models classified as high-risk
        • Compliance rate: {compliance_score:.1f}%
        
        CRITICAL AREAS:
        • Risk management and mitigation strategies
        • Bias detection and fairness assessment
        • Audit completion and compliance monitoring
        
        RECOMMENDATIONS:
        • Maintain continuous improvement in AI governance
        • Address identified compliance gaps promptly
        • Strengthen risk assessment and mitigation processes
        """
        
        return summary.strip()
    
    def _extract_report_findings(self, sections: List[ComplianceReportSection], report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract findings from report sections"""
        findings = []
        
        for section in sections:
            if section.compliance_status in ["non_compliant", "requires_attention"]:
                findings.append({
                    "finding_id": f"finding_{section.section_id}",
                    "section": section.title,
                    "severity": "high" if section.compliance_status == "non_compliant" else "medium",
                    "description": f"Compliance issues identified in {section.title}",
                    "impact": "May affect regulatory compliance and risk management",
                    "recommendations": section.recommendations[:3]  # Top 3 recommendations
                })
        
        # Add specific findings based on data
        governance_data = report_data.get("governance", {})
        if governance_data.get("policy_violations", 0) > 0:
            findings.append({
                "finding_id": "policy_violations",
                "section": "Policy Compliance",
                "severity": "high",
                "description": f"{governance_data['policy_violations']} policy violations identified",
                "impact": "Risk of regulatory non-compliance and operational issues",
                "recommendations": ["Review policy adherence procedures", "Implement additional controls"]
            })
        
        return findings
    
    def _generate_report_recommendations(self, findings: List[Dict[str, Any]], compliance_score: float) -> List[str]:
        """Generate overall report recommendations"""
        recommendations = []
        
        # Recommendations based on compliance score
        if compliance_score < 80:
            recommendations.append("Implement comprehensive compliance improvement program")
        
        if compliance_score < 70:
            recommendations.append("Consider engaging external compliance consultants")
        
        # Recommendations based on findings
        high_severity_findings = [f for f in findings if f.get("severity") == "high"]
        if high_severity_findings:
            recommendations.append(f"Address {len(high_severity_findings)} high-severity compliance issues immediately")
        
        # General recommendations
        recommendations.extend([
            "Establish regular governance review cycles",
            "Implement automated compliance monitoring",
            "Provide compliance training to all stakeholders",
            "Maintain comprehensive documentation of AI governance activities"
        ])
        
        return recommendations
    
    def _generate_next_actions(self, findings: List[Dict[str, Any]], recommendations: List[str]) -> List[Dict[str, Any]]:
        """Generate next actions based on findings and recommendations"""
        actions = []
        
        # Actions for high-severity findings
        high_findings = [f for f in findings if f.get("severity") == "high"]
        for finding in high_findings[:5]:  # Top 5 high-severity findings
            actions.append({
                "action": f"Address {finding['finding_id']}",
                "description": finding["description"],
                "priority": "high",
                "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
                "responsible_party": "compliance_team",
                "estimated_effort": "2-4 weeks"
            })
        
        # Actions for top recommendations
        for i, recommendation in enumerate(recommendations[:3]):
            actions.append({
                "action": f"Implement recommendation {i+1}",
                "description": recommendation,
                "priority": "medium",
                "due_date": (datetime.now(timezone.utc) + timedelta(days=60)).isoformat(),
                "responsible_party": "governance_committee",
                "estimated_effort": "4-8 weeks"
            })
        
        # Schedule next report
        actions.append({
            "action": "Schedule next compliance review",
            "description": "Plan and schedule the next comprehensive compliance assessment",
            "priority": "low",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),
            "responsible_party": "compliance_manager",
            "estimated_effort": "1-2 days"
        })
        
        return actions
    
    async def _report_scheduler(self):
        """Background scheduler for automated report generation"""
        while True:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                # Check for scheduled reports
                for template_id, template in self.report_templates.items():
                    if await self._should_generate_report(template):
                        try:
                            await self.generate_compliance_report(
                                report_type=template.report_type,
                                requester="automated_scheduler"
                            )
                            logger.info(f"Automatically generated {template.report_type.value} report")
                        except Exception as e:
                            logger.error(f"Failed to auto-generate report {template_id}: {e}")
                
            except Exception as e:
                logger.error(f"Error in report scheduler: {e}")
                await asyncio.sleep(300)
    
    async def _should_generate_report(self, template: ReportTemplate) -> bool:
        """Determine if a report should be automatically generated"""
        if template.frequency == ReportFrequency.ON_DEMAND:
            return False
        
        # Check if it's time to generate the report
        now = datetime.now(timezone.utc)
        
        # Get last report of this type
        last_report = None
        for report in reversed(self.reports):
            if report.report_type == template.report_type:
                last_report = report
                break
        
        if last_report is None:
            return True  # Generate first report
        
        # Check frequency
        time_since_last = now - last_report.generated_at
        
        if template.frequency == ReportFrequency.DAILY and time_since_last >= timedelta(days=1):
            return True
        elif template.frequency == ReportFrequency.WEEKLY and time_since_last >= timedelta(weeks=1):
            return True
        elif template.frequency == ReportFrequency.MONTHLY and time_since_last >= timedelta(days=30):
            return True
        elif template.frequency == ReportFrequency.QUARTERLY and time_since_last >= timedelta(days=90):
            return True
        elif template.frequency == ReportFrequency.ANNUALLY and time_since_last >= timedelta(days=365):
            return True
        
        return False
    
    # Public API methods
    
    def get_reports(self, report_type: ReportType = None, limit: int = 50) -> List[AIComplianceReport]:
        """Get compliance reports"""
        reports = self.reports
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        return sorted(reports, key=lambda r: r.generated_at, reverse=True)[:limit]
    
    def get_report(self, report_id: str) -> Optional[AIComplianceReport]:
        """Get a specific report"""
        for report in self.reports:
            if report.report_id == report_id:
                return report
        return None
    
    async def generate_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard data"""
        recent_reports = self.get_reports(limit=10)
        
        if not recent_reports:
            return {
                "dashboard_id": f"comp_dash_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_reports": 0,
                "compliance_summary": {},
                "alerts": [],
                "recommendations": ["Generate first compliance report"]
            }
        
        # Calculate summary metrics
        avg_compliance_score = statistics.mean([r.overall_compliance_score for r in recent_reports])
        latest_score = recent_reports[0].overall_compliance_score if recent_reports else 0
        
        # Report type distribution
        report_types = [r.report_type.value for r in recent_reports]
        type_distribution = dict(Counter(report_types))
        
        return {
            "dashboard_id": f"comp_dash_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_reports": len(self.reports),
            "compliance_summary": {
                "latest_compliance_score": latest_score,
                "average_compliance_score": avg_compliance_score,
                "compliance_trend": "improving" if len(recent_reports) > 1 and recent_reports[0].overall_compliance_score > recent_reports[1].overall_compliance_score else "stable",
                "report_type_distribution": type_distribution
            },
            "recent_reports": [
                {
                    "report_id": r.report_id,
                    "report_type": r.report_type.value,
                    "compliance_score": r.overall_compliance_score,
                    "generated_at": r.generated_at.isoformat(),
                    "findings_count": len(r.findings)
                }
                for r in recent_reports[:5]
            ],
            "compliance_kpis": [asdict(kpi) for kpi in self.compliance_kpis.values()],
            "alerts": self._generate_compliance_alerts(recent_reports),
            "recommendations": self._generate_dashboard_compliance_recommendations(recent_reports)
        }
    
    def _generate_compliance_alerts(self, reports: List[AIComplianceReport]) -> List[Dict[str, Any]]:
        """Generate compliance alerts"""
        alerts = []
        
        if reports:
            latest_report = reports[0]
            
            if latest_report.overall_compliance_score < 70:
                alerts.append({
                    "type": "low_compliance_score",
                    "severity": "high",
                    "message": f"Latest compliance score ({latest_report.overall_compliance_score:.1f}%) is below threshold",
                    "report_id": latest_report.report_id
                })
            
            high_findings = [f for f in latest_report.findings if f.get("severity") == "high"]
            if high_findings:
                alerts.append({
                    "type": "high_severity_findings",
                    "severity": "high",
                    "message": f"{len(high_findings)} high-severity compliance findings require attention",
                    "findings": len(high_findings)
                })
        
        # Check overdue KPIs
        for kpi in self.compliance_kpis.values():
            if kpi.current_value < kpi.target_value * 0.8:  # Below 80% of target
                alerts.append({
                    "type": "kpi_below_target",
                    "severity": "medium",
                    "message": f"KPI '{kpi.name}' is significantly below target",
                    "kpi_id": kpi.kpi_id
                })
        
        return alerts
    
    def _generate_dashboard_compliance_recommendations(self, reports: List[AIComplianceReport]) -> List[str]:
        """Generate recommendations for compliance dashboard"""
        recommendations = []
        
        if not reports:
            recommendations.append("Generate initial compliance reports to establish baseline")
            return recommendations
        
        latest_report = reports[0]
        
        if latest_report.overall_compliance_score < 80:
            recommendations.append("Focus on improving overall compliance score")
        
        if len(latest_report.findings) > 5:
            recommendations.append("Address multiple compliance findings systematically")
        
        recommendations.extend([
            "Maintain regular compliance reporting schedule",
            "Monitor compliance KPIs continuously",
            "Provide compliance training to stakeholders",
            "Review and update compliance policies regularly"
        ])
        
        return recommendations

# Global instance
_global_ai_compliance_reporter = None

def get_ai_compliance_reporter() -> AIComplianceReporter:
    """Get global AI compliance reporter instance"""
    global _global_ai_compliance_reporter
    if _global_ai_compliance_reporter is None:
        _global_ai_compliance_reporter = AIComplianceReporter()
    return _global_ai_compliance_reporter