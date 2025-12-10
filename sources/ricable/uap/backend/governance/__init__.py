# File: backend/governance/__init__.py
"""
Enterprise AI Governance System for UAP

This module provides comprehensive AI governance capabilities including:
- AI model governance and compliance tracking
- Explainability and bias detection
- Data lineage tracking and model auditing
- AI risk assessment and mitigation frameworks
- Automated compliance reporting for AI usage

The system integrates with existing security, compliance, and monitoring infrastructure
to provide enterprise-grade AI governance.
"""

from .ai_governance_manager import AIGovernanceManager, get_ai_governance_manager
from .model_auditing import ModelAuditingService, get_model_auditing_service
from .risk_assessment import AIRiskAssessment, get_ai_risk_assessment
from .compliance_reporting import AIComplianceReporter, get_ai_compliance_reporter
from .data_lineage import AIDataLineageTracker, get_ai_data_lineage_tracker

__all__ = [
    'AIGovernanceManager',
    'get_ai_governance_manager',
    'ModelAuditingService', 
    'get_model_auditing_service',
    'AIRiskAssessment',
    'get_ai_risk_assessment',
    'AIComplianceReporter',
    'get_ai_compliance_reporter',
    'AIDataLineageTracker',
    'get_ai_data_lineage_tracker'
]