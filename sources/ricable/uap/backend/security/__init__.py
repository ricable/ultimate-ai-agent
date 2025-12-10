# File: backend/security/__init__.py
"""
UAP Security Module

Advanced security features including:
- AES-256 encryption at rest
- Automated security scanning
- Threat detection and prevention
- Immutable audit trails
- Security monitoring and alerting
"""

from .encryption import DataEncryption, EncryptedDataService
from .scanning import SecurityScanner, VulnerabilityAssessment
from .threat_detection import ThreatDetectionEngine, SecurityIncidentManager
from .audit_trail import ImmutableAuditTrail, SecurityAuditLogger

__all__ = [
    "DataEncryption",
    "EncryptedDataService", 
    "SecurityScanner",
    "VulnerabilityAssessment",
    "ThreatDetectionEngine",
    "SecurityIncidentManager",
    "ImmutableAuditTrail",
    "SecurityAuditLogger"
]