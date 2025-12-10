# File: backend/compliance/compliance_manager.py
"""
Unified Compliance Management System
Coordinates multiple compliance frameworks and provides centralized compliance reporting.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .soc2 import get_soc2_framework, SOC2ComplianceFramework
from .gdpr import get_gdpr_framework, GDPRComplianceFramework
from .hipaa import get_hipaa_framework, HIPAAComplianceFramework
from ..monitoring.logs.logger import uap_logger

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    CCPA = "ccpa"
    NIST = "nist"

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    priority: str  # critical, high, medium, low
    status: ComplianceStatus
    last_assessed: Optional[datetime]
    next_assessment: Optional[datetime]
    evidence_collected: bool
    remediation_required: bool
    remediation_deadline: Optional[datetime]
    assigned_to: Optional[str]
    notes: List[str]

@dataclass
class ComplianceGap:
    """Compliance gap identification"""
    gap_id: str
    framework: ComplianceFramework
    requirement_id: str
    gap_type: str  # missing_control, inadequate_implementation, documentation
    severity: str  # critical, high, medium, low
    description: str
    impact_assessment: str
    remediation_plan: List[str]
    estimated_effort: str
    target_completion: datetime
    business_justification: str

@dataclass
class ComplianceReport:
    """Unified compliance report"""
    report_id: str
    report_type: str
    frameworks_covered: List[str]
    reporting_period_start: datetime
    reporting_period_end: datetime
    overall_status: ComplianceStatus
    framework_statuses: Dict[str, Dict[str, Any]]
    key_findings: List[str]
    gaps_identified: List[ComplianceGap]
    recommendations: List[str]
    next_actions: List[Dict[str, Any]]
    executive_summary: str

class ComplianceManager:
    """
    Unified compliance management system.
    
    Provides centralized management of multiple compliance frameworks including:
    - SOC 2 Type II
    - GDPR
    - HIPAA
    - PCI DSS (placeholder)
    - ISO 27001 (placeholder)
    
    Features:
    - Cross-framework requirement mapping
    - Unified compliance reporting
    - Gap analysis and remediation planning
    - Continuous compliance monitoring
    - Executive dashboards and metrics
    """
    
    def __init__(self):
        # Initialize compliance frameworks
        self.frameworks = {
            ComplianceFramework.SOC2: get_soc2_framework(),
            ComplianceFramework.GDPR: get_gdpr_framework(),
            ComplianceFramework.HIPAA: get_hipaa_framework()
            # Note: PCI DSS and ISO 27001 would be added when implemented
        }
        
        self.requirements: List[ComplianceRequirement] = []
        self.gaps: List[ComplianceGap] = []
        self.reports: List[ComplianceReport] = []
        
        # Cross-framework requirement mappings
        self.requirement_mappings = self._initialize_requirement_mappings()
        
        # Initialize requirements from frameworks
        self._initialize_requirements()
        
        # Start continuous monitoring
        asyncio.create_task(self._continuous_monitoring())
    
    def _initialize_requirement_mappings(self) -> Dict[str, List[str]]:
        """Initialize cross-framework requirement mappings"""
        return {
            # Access control mappings
            "access_control": [
                "SOC2:CC6.1",  # Logical Access Security Measures
                "HIPAA:45_CFR_164_312_a_1",  # Access Control
                "ISO27001:A.9.1.1",  # Access control policy
                "PCI:7.1"  # Limit access to system components
            ],
            
            # Encryption mappings
            "encryption": [
                "SOC2:C1.1",  # Confidentiality Protection
                "GDPR:Article_32",  # Security of processing
                "HIPAA:45_CFR_164_312_e_1",  # Transmission Security
                "PCI:3.4"  # Protect stored cardholder data
            ],
            
            # Audit logging mappings
            "audit_logging": [
                "SOC2:CC7.1",  # System Operation Procedures
                "HIPAA:45_CFR_164_312_b",  # Audit Controls
                "ISO27001:A.12.4.1",  # Event logging
                "PCI:10.1"  # Implement audit trails
            ],
            
            # Data protection mappings
            "data_protection": [
                "GDPR:Article_25",  # Data protection by design
                "HIPAA:PHI_Protection",  # PHI safeguards
                "SOC2:PI1.1",  # Data Processing Integrity
                "PCI:3.1"  # Protect stored cardholder data
            ],
            
            # Incident response mappings
            "incident_response": [
                "SOC2:CC1.1",  # Control Environment
                "GDPR:Article_33",  # Notification of breach
                "HIPAA:45_CFR_164_308_a_6",  # Security Incident Procedures
                "ISO27001:A.16.1.1"  # Management of information security incidents
            ],
            
            # Risk management mappings
            "risk_management": [
                "SOC2:CC3.1",  # Risk Identification and Assessment
                "ISO27001:A.5.1.1",  # Information security policy
                "HIPAA:Risk_Assessment",  # Security risk assessment
                "PCI:12.2"  # Implement a risk assessment process
            ]
        }
    
    def _initialize_requirements(self):
        """Initialize compliance requirements from all frameworks"""
        # This would load requirements from each framework
        # For now, create some example requirements
        
        # SOC 2 requirements
        self.requirements.extend([
            ComplianceRequirement(
                requirement_id="SOC2_CC6_1",
                framework=ComplianceFramework.SOC2,
                title="Logical Access Security Measures",
                description="Implement logical access security measures to protect system resources",
                category="access_control",
                priority="critical",
                status=ComplianceStatus.COMPLIANT,
                last_assessed=datetime.now(timezone.utc),
                next_assessment=datetime.now(timezone.utc) + timedelta(days=365),
                evidence_collected=True,
                remediation_required=False,
                assigned_to="security_team",
                notes=[]
            )
        ])
        
        # GDPR requirements
        self.requirements.extend([
            ComplianceRequirement(
                requirement_id="GDPR_ART_25",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design and by Default",
                description="Implement data protection by design and by default",
                category="data_protection",
                priority="critical",
                status=ComplianceStatus.COMPLIANT,
                last_assessed=datetime.now(timezone.utc),
                next_assessment=datetime.now(timezone.utc) + timedelta(days=365),
                evidence_collected=True,
                remediation_required=False,
                assigned_to="privacy_team",
                notes=[]
            )
        ])
        
        # HIPAA requirements
        self.requirements.extend([
            ComplianceRequirement(
                requirement_id="HIPAA_164_312_A1",
                framework=ComplianceFramework.HIPAA,
                title="Access Control Technical Safeguard",
                description="Implement technical policies and procedures for electronic information systems",
                category="access_control",
                priority="critical",
                status=ComplianceStatus.COMPLIANT,
                last_assessed=datetime.now(timezone.utc),
                next_assessment=datetime.now(timezone.utc) + timedelta(days=365),
                evidence_collected=True,
                remediation_required=False,
                assigned_to="compliance_team",
                notes=[]
            )
        ])
    
    async def assess_compliance(self, frameworks: List[ComplianceFramework] = None) -> ComplianceReport:
        """Perform comprehensive compliance assessment across frameworks"""
        if frameworks is None:
            frameworks = list(self.frameworks.keys())
        
        report_id = f"COMP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        period_start = datetime.now(timezone.utc) - timedelta(days=365)
        period_end = datetime.now(timezone.utc)
        
        # Assess each framework
        framework_statuses = {}
        overall_gaps = []
        key_findings = []
        
        for framework in frameworks:
            framework_instance = self.frameworks.get(framework)
            if framework_instance:
                status = await self._assess_framework(framework, framework_instance)
                framework_statuses[framework.value] = status
                
                if status["gaps"]:
                    overall_gaps.extend(status["gaps"])
                
                key_findings.extend(status.get("findings", []))
        
        # Determine overall compliance status
        overall_status = self._determine_overall_status(framework_statuses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(framework_statuses, overall_gaps)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            framework_statuses, overall_status, len(overall_gaps)
        )
        
        report = ComplianceReport(
            report_id=report_id,
            report_type="Comprehensive Compliance Assessment",
            frameworks_covered=[f.value for f in frameworks],
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            overall_status=overall_status,
            framework_statuses=framework_statuses,
            key_findings=key_findings,
            gaps_identified=overall_gaps,
            recommendations=recommendations,
            next_actions=self._generate_next_actions(overall_gaps),
            executive_summary=executive_summary
        )
        
        self.reports.append(report)
        
        # Log assessment completion
        uap_logger.log_security_event(
            f"Compliance assessment completed: {report_id}",
            metadata={
                "report_id": report_id,
                "frameworks": [f.value for f in frameworks],
                "overall_status": overall_status.value,
                "gaps_identified": len(overall_gaps)
            }
        )
        
        return report
    
    async def _assess_framework(self, framework: ComplianceFramework, 
                              framework_instance: Any) -> Dict[str, Any]:
        """Assess individual compliance framework"""
        try:
            if framework == ComplianceFramework.SOC2:
                assessment = await framework_instance.assess_compliance()
                return {
                    "status": assessment.overall_rating,
                    "compliance_rate": assessment.compliant_controls / assessment.total_controls,
                    "total_controls": assessment.total_controls,
                    "compliant_controls": assessment.compliant_controls,
                    "gaps": [
                        ComplianceGap(
                            gap_id=f"SOC2-GAP-{i}",
                            framework=framework,
                            requirement_id="various",
                            gap_type="control_deficiency",
                            severity="medium",
                            description=gap,
                            impact_assessment="Compliance risk",
                            remediation_plan=["Address control deficiency"],
                            estimated_effort="2-4 weeks",
                            target_completion=datetime.now(timezone.utc) + timedelta(days=30),
                            business_justification="Maintain SOC 2 certification"
                        ) for i, gap in enumerate(assessment.gaps_summary)
                    ],
                    "findings": [f"SOC 2: {gap}" for gap in assessment.gaps_summary],
                    "last_assessment": assessment.assessment_date.isoformat()
                }
            
            elif framework == ComplianceFramework.GDPR:
                status = framework_instance.get_compliance_status()
                return {
                    "status": "compliant" if status["compliance_score"] >= 80 else "partially_compliant",
                    "compliance_score": status["compliance_score"],
                    "data_categories": status["data_categories"],
                    "active_consents": status["active_consents"],
                    "pending_requests": status["pending_requests"],
                    "gaps": [],  # Would be populated based on GDPR assessment
                    "findings": [
                        f"GDPR compliance score: {status['compliance_score']:.1f}%",
                        f"Pending data subject requests: {status['pending_requests']}"
                    ],
                    "last_assessment": datetime.now(timezone.utc).isoformat()
                }
            
            elif framework == ComplianceFramework.HIPAA:
                status = framework_instance.get_compliance_status()
                return {
                    "status": "compliant" if status["compliance_score"] >= 80 else "partially_compliant",
                    "compliance_score": status["compliance_score"],
                    "phi_elements": status["phi_elements"],
                    "active_business_associates": status["active_business_associates"],
                    "recent_incidents": status["recent_incidents"],
                    "gaps": [],  # Would be populated based on HIPAA assessment
                    "findings": [
                        f"HIPAA compliance score: {status['compliance_score']:.1f}%",
                        f"Recent security incidents: {status['recent_incidents']}"
                    ],
                    "last_assessment": datetime.now(timezone.utc).isoformat()
                }
            
            else:
                return {
                    "status": "not_assessed",
                    "message": f"Framework {framework.value} not yet implemented",
                    "gaps": [],
                    "findings": [],
                    "last_assessment": None
                }
        
        except Exception as e:
            logger.error(f"Error assessing framework {framework.value}: {e}")
            return {
                "status": "assessment_error",
                "error": str(e),
                "gaps": [],
                "findings": [f"Assessment error: {str(e)}"],
                "last_assessment": None
            }
    
    def _determine_overall_status(self, framework_statuses: Dict[str, Any]) -> ComplianceStatus:
        """Determine overall compliance status across all frameworks"""
        statuses = []
        
        for status_info in framework_statuses.values():
            status = status_info.get("status", "unknown")
            if status in ["compliant", "Compliant"]:
                statuses.append("compliant")
            elif status in ["partially_compliant", "Partially Compliant"]:
                statuses.append("partially_compliant")
            elif status in ["non_compliant", "Non-Compliant"]:
                statuses.append("non_compliant")
            else:
                statuses.append("unknown")
        
        # Determine overall status
        if all(s == "compliant" for s in statuses):
            return ComplianceStatus.COMPLIANT
        elif any(s == "non_compliant" for s in statuses):
            return ComplianceStatus.NON_COMPLIANT
        elif any(s == "partially_compliant" for s in statuses):
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.UNDER_REVIEW
    
    def _generate_recommendations(self, framework_statuses: Dict[str, Any], 
                                gaps: List[ComplianceGap]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Framework-specific recommendations
        for framework, status in framework_statuses.items():
            if status.get("status") != "compliant":
                if framework == "soc2":
                    recommendations.append("Enhance SOC 2 control implementation and testing")
                elif framework == "gdpr":
                    recommendations.append("Improve GDPR data subject request processing")
                elif framework == "hipaa":
                    recommendations.append("Strengthen HIPAA security incident procedures")
        
        # Gap-based recommendations
        if gaps:
            recommendations.append(f"Address {len(gaps)} identified compliance gaps")
            
            # Prioritize by severity
            critical_gaps = [g for g in gaps if g.severity == "critical"]
            if critical_gaps:
                recommendations.append(f"Immediately address {len(critical_gaps)} critical gaps")
        
        # General recommendations
        recommendations.extend([
            "Implement continuous compliance monitoring",
            "Conduct regular compliance training for staff",
            "Establish compliance metrics and reporting dashboards",
            "Review and update compliance policies quarterly"
        ])
        
        return recommendations
    
    def _generate_next_actions(self, gaps: List[ComplianceGap]) -> List[Dict[str, Any]]:
        """Generate next actions based on identified gaps"""
        actions = []
        
        # Sort gaps by severity and target completion
        sorted_gaps = sorted(gaps, key=lambda g: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(g.severity, 3),
            g.target_completion
        ))
        
        for gap in sorted_gaps[:10]:  # Top 10 priority gaps
            actions.append({
                "action": f"Remediate {gap.gap_type} for {gap.framework.value}",
                "description": gap.description,
                "priority": gap.severity,
                "assigned_to": "compliance_team",
                "due_date": gap.target_completion.isoformat(),
                "estimated_effort": gap.estimated_effort
            })
        
        # Add monitoring actions
        actions.append({
            "action": "Schedule next compliance assessment",
            "description": "Plan and schedule the next comprehensive compliance assessment",
            "priority": "medium",
            "assigned_to": "compliance_manager",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),
            "estimated_effort": "1-2 days"
        })
        
        return actions
    
    def _generate_executive_summary(self, framework_statuses: Dict[str, Any],
                                  overall_status: ComplianceStatus, 
                                  gap_count: int) -> str:
        """Generate executive summary of compliance status"""
        compliant_frameworks = sum(
            1 for status in framework_statuses.values()
            if status.get("status") in ["compliant", "Compliant"]
        )
        total_frameworks = len(framework_statuses)
        
        summary = f"""
        EXECUTIVE SUMMARY - COMPLIANCE ASSESSMENT
        
        Overall Status: {overall_status.value.upper()}
        
        Framework Compliance: {compliant_frameworks}/{total_frameworks} frameworks fully compliant
        
        Key Metrics:
        - {compliant_frameworks} frameworks meeting compliance requirements
        - {gap_count} compliance gaps identified requiring attention
        - {total_frameworks - compliant_frameworks} frameworks needing remediation
        
        Priority Actions:
        {'- Immediate attention required for non-compliant frameworks' if overall_status != ComplianceStatus.COMPLIANT else '- Continue monitoring and maintenance of current compliance status'}
        - Address identified gaps within target timelines
        - Maintain continuous monitoring and improvement processes
        
        Risk Assessment: {'LOW' if overall_status == ComplianceStatus.COMPLIANT else 'MEDIUM' if overall_status == ComplianceStatus.PARTIALLY_COMPLIANT else 'HIGH'}
        
        Next Review: {(datetime.now(timezone.utc) + timedelta(days=90)).strftime('%Y-%m-%d')}
        """
        
        return summary.strip()
    
    async def identify_cross_framework_gaps(self) -> List[ComplianceGap]:
        """Identify gaps that affect multiple compliance frameworks"""
        cross_gaps = []
        
        # Analyze requirement mappings to find common gaps
        for category, requirements in self.requirement_mappings.items():
            # Check if any requirements in this category are non-compliant
            category_requirements = [
                req for req in self.requirements
                if req.category == category and req.status != ComplianceStatus.COMPLIANT
            ]
            
            if category_requirements:
                # This affects multiple frameworks
                affected_frameworks = list(set(req.framework for req in category_requirements))
                
                if len(affected_frameworks) > 1:
                    gap = ComplianceGap(
                        gap_id=f"CROSS-{category.upper()}-{datetime.now().strftime('%Y%m%d')}",
                        framework=affected_frameworks[0],  # Primary framework
                        requirement_id=category,
                        gap_type="cross_framework_gap",
                        severity="high",
                        description=f"Gap in {category} affects multiple compliance frameworks",
                        impact_assessment=f"Impacts {len(affected_frameworks)} frameworks: {[f.value for f in affected_frameworks]}",
                        remediation_plan=[
                            f"Implement comprehensive {category} controls",
                            "Address requirements across all affected frameworks",
                            "Document cross-framework compliance mapping"
                        ],
                        estimated_effort="4-8 weeks",
                        target_completion=datetime.now(timezone.utc) + timedelta(days=60),
                        business_justification="Ensure compliance across multiple regulatory requirements"
                    )
                    cross_gaps.append(gap)
        
        return cross_gaps
    
    async def _continuous_monitoring(self):
        """Continuous compliance monitoring across all frameworks"""
        while True:
            try:
                # Check compliance status across frameworks
                for framework, instance in self.frameworks.items():
                    await self._monitor_framework(framework, instance)
                
                # Check for overdue requirements
                overdue_requirements = [
                    req for req in self.requirements
                    if req.remediation_deadline and 
                    datetime.now(timezone.utc) > req.remediation_deadline and
                    req.status != ComplianceStatus.COMPLIANT
                ]
                
                if overdue_requirements:
                    uap_logger.log_security_event(
                        f"Compliance alert: {len(overdue_requirements)} overdue requirements",
                        metadata={"overdue_count": len(overdue_requirements)}
                    )
                
                # Generate monthly compliance dashboard
                if datetime.now().day == 1:  # First day of month
                    await self._generate_monthly_dashboard()
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _monitor_framework(self, framework: ComplianceFramework, instance: Any):
        """Monitor individual framework for compliance issues"""
        try:
            if framework == ComplianceFramework.GDPR:
                status = instance.get_compliance_status()
                if status["overdue_requests"] > 0:
                    uap_logger.log_security_event(
                        f"GDPR compliance alert: {status['overdue_requests']} overdue data subject requests",
                        metadata={"framework": "GDPR", "overdue_requests": status["overdue_requests"]}
                    )
            
            elif framework == ComplianceFramework.HIPAA:
                status = instance.get_compliance_status()
                if status["recent_incidents"] > 0:
                    uap_logger.log_security_event(
                        f"HIPAA compliance alert: {status['recent_incidents']} recent security incidents",
                        metadata={"framework": "HIPAA", "recent_incidents": status["recent_incidents"]}
                    )
        
        except Exception as e:
            logger.error(f"Error monitoring framework {framework.value}: {e}")
    
    async def _generate_monthly_dashboard(self):
        """Generate monthly compliance dashboard"""
        try:
            dashboard_data = {
                "date": datetime.now(timezone.utc).isoformat(),
                "frameworks": {},
                "overall_metrics": {
                    "total_requirements": len(self.requirements),
                    "compliant_requirements": len([
                        req for req in self.requirements 
                        if req.status == ComplianceStatus.COMPLIANT
                    ]),
                    "open_gaps": len(self.gaps),
                    "overdue_items": len([
                        req for req in self.requirements
                        if req.remediation_deadline and 
                        datetime.now(timezone.utc) > req.remediation_deadline and
                        req.status != ComplianceStatus.COMPLIANT
                    ])
                }
            }
            
            # Get status for each framework
            for framework, instance in self.frameworks.items():
                if hasattr(instance, 'get_compliance_status'):
                    dashboard_data["frameworks"][framework.value] = instance.get_compliance_status()
            
            uap_logger.log_security_event(
                "Monthly compliance dashboard generated",
                metadata=dashboard_data
            )
        
        except Exception as e:
            logger.error(f"Error generating monthly dashboard: {e}")
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get real-time compliance dashboard"""
        total_requirements = len(self.requirements)
        compliant_requirements = len([
            req for req in self.requirements 
            if req.status == ComplianceStatus.COMPLIANT
        ])
        
        compliance_rate = compliant_requirements / total_requirements if total_requirements > 0 else 0
        
        # Framework-specific status
        framework_summary = {}
        for framework, instance in self.frameworks.items():
            if hasattr(instance, 'get_compliance_status'):
                framework_summary[framework.value] = instance.get_compliance_status()
        
        return {
            "dashboard_date": datetime.now(timezone.utc).isoformat(),
            "overall_compliance_rate": compliance_rate,
            "total_requirements": total_requirements,
            "compliant_requirements": compliant_requirements,
            "active_gaps": len(self.gaps),
            "frameworks": framework_summary,
            "recent_assessments": len([
                report for report in self.reports
                if report.reporting_period_end >= datetime.now(timezone.utc) - timedelta(days=30)
            ]),
            "next_assessment_due": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
        }

class ComplianceReporter:
    """Compliance reporting and document generation"""
    
    def __init__(self, compliance_manager: ComplianceManager):
        self.manager = compliance_manager
    
    async def generate_executive_report(self, frameworks: List[ComplianceFramework] = None) -> Dict[str, Any]:
        """Generate executive compliance report"""
        assessment = await self.manager.assess_compliance(frameworks)
        
        return {
            "report_type": "Executive Compliance Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executive_summary": assessment.executive_summary,
            "overall_status": assessment.overall_status.value,
            "key_metrics": {
                "frameworks_assessed": len(assessment.frameworks_covered),
                "overall_compliance_rate": self._calculate_overall_rate(assessment.framework_statuses),
                "gaps_identified": len(assessment.gaps_identified),
                "critical_gaps": len([g for g in assessment.gaps_identified if g.severity == "critical"])
            },
            "framework_breakdown": assessment.framework_statuses,
            "priority_actions": assessment.next_actions[:5],  # Top 5 actions
            "recommendations": assessment.recommendations,
            "compliance_trend": "stable"  # Would calculate from historical data
        }
    
    def _calculate_overall_rate(self, framework_statuses: Dict[str, Any]) -> float:
        """Calculate overall compliance rate across frameworks"""
        rates = []
        
        for status in framework_statuses.values():
            if "compliance_rate" in status:
                rates.append(status["compliance_rate"])
            elif "compliance_score" in status:
                rates.append(status["compliance_score"] / 100.0)
        
        return sum(rates) / len(rates) if rates else 0.0
    
    async def generate_audit_package(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate audit package for specific framework"""
        framework_instance = self.manager.frameworks.get(framework)
        
        if not framework_instance:
            return {"error": f"Framework {framework.value} not available"}
        
        package = {
            "framework": framework.value,
            "package_date": datetime.now(timezone.utc).isoformat(),
            "audit_readiness": "ready",
            "evidence_summary": {},
            "control_documentation": {},
            "gaps_and_remediation": []
        }
        
        # Framework-specific audit packages
        if framework == ComplianceFramework.SOC2:
            if hasattr(framework_instance, 'generate_soc2_report'):
                latest_assessment = await framework_instance.assess_compliance()
                package["evidence_summary"] = framework_instance.generate_soc2_report(latest_assessment)
        
        elif framework == ComplianceFramework.GDPR:
            if hasattr(framework_instance, 'generate_ropa_report'):
                package["evidence_summary"] = framework_instance.generate_ropa_report()
        
        elif framework == ComplianceFramework.HIPAA:
            if hasattr(framework_instance, 'generate_hipaa_risk_assessment'):
                package["evidence_summary"] = framework_instance.generate_hipaa_risk_assessment()
        
        return package

# Global compliance manager instance
_global_compliance_manager = None
_global_compliance_reporter = None

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance"""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = ComplianceManager()
    return _global_compliance_manager

def get_compliance_reporter() -> ComplianceReporter:
    """Get global compliance reporter instance"""
    global _global_compliance_reporter
    if _global_compliance_reporter is None:
        _global_compliance_reporter = ComplianceReporter(get_compliance_manager())
    return _global_compliance_reporter