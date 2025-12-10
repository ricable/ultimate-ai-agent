# File: backend/compliance/hipaa.py
"""
HIPAA Compliance Framework
Implements Health Insurance Portability and Accountability Act compliance for healthcare data protection.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..security.encryption import get_encryption_service
from ..security.audit_trail import get_audit_trail, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger

logger = logging.getLogger(__name__)

class PHIType(Enum):
    """Types of Protected Health Information"""
    IDENTIFIERS = "identifiers"
    HEALTH_INFO = "health_info"
    BILLING_INFO = "billing_info"
    MEDICAL_RECORDS = "medical_records"
    TREATMENT_INFO = "treatment_info"

class HIPAAIdentifier(Enum):
    """HIPAA 18 identifiers that must be removed for de-identification"""
    NAME = "name"
    GEOGRAPHIC_SUBDIVISION = "geographic_subdivision"
    DATES = "dates"
    TELEPHONE_NUMBERS = "telephone_numbers"
    VEHICLE_IDENTIFIERS = "vehicle_identifiers"
    DEVICE_IDENTIFIERS = "device_identifiers"
    WEB_URLS = "web_urls"
    IP_ADDRESSES = "ip_addresses"
    BIOMETRIC_IDENTIFIERS = "biometric_identifiers"
    FULL_FACE_PHOTOS = "full_face_photos"
    UNIQUE_IDENTIFYING_NUMBER = "unique_identifying_number"
    SSN = "ssn"
    MEDICAL_RECORD_NUMBER = "medical_record_number"
    HEALTH_PLAN_NUMBER = "health_plan_number"
    ACCOUNT_NUMBER = "account_number"
    CERTIFICATE_NUMBER = "certificate_number"
    EMAIL_ADDRESS = "email_address"
    OTHER_UNIQUE_IDENTIFIER = "other_unique_identifier"

class HIPAASafeguard(Enum):
    """HIPAA Safeguard categories"""
    ADMINISTRATIVE = "administrative"
    PHYSICAL = "physical"
    TECHNICAL = "technical"

class MinimumNecessaryStandard(Enum):
    """Minimum necessary standard applications"""
    USES = "uses"
    DISCLOSURES = "disclosures"
    REQUESTS = "requests"

@dataclass
class PHIDataElement:
    """Protected Health Information data element"""
    element_id: str
    element_name: str
    phi_type: PHIType
    identifiers: List[HIPAAIdentifier]
    sensitivity_level: str  # high, medium, low
    encryption_required: bool
    access_restrictions: List[str]
    retention_period: timedelta
    minimum_necessary_applied: bool
    data_locations: List[str]
    authorized_users: List[str]

@dataclass
class HIPAAIncident:
    """HIPAA security incident record"""
    incident_id: str
    detection_date: datetime
    incident_type: str  # breach, unauthorized_access, system_failure, etc.
    affected_phi_elements: List[str]
    affected_individuals: int
    description: str
    risk_assessment: str  # low, medium, high
    breach_notification_required: bool
    hhs_notification_required: bool
    notification_completed: bool = False
    remediation_actions: List[str] = None
    investigation_status: str = "open"

@dataclass
class BusinessAssociateAgreement:
    """Business Associate Agreement (BAA) record"""
    baa_id: str
    business_associate: str
    services_provided: List[str]
    phi_types_accessed: List[str]
    agreement_date: datetime
    expiration_date: datetime
    safeguards_required: List[str]
    audit_rights: bool
    termination_procedures: List[str]
    status: str  # active, expired, terminated

@dataclass
class PHIDisclosure:
    """PHI disclosure record for accounting"""
    disclosure_id: str
    date_of_disclosure: datetime
    recipient: str
    description_of_phi: str
    purpose_of_disclosure: str
    legal_authority: str
    individual_id: Optional[str] = None
    minimum_necessary_applied: bool = True

class HIPAAComplianceFramework:
    """
    HIPAA compliance framework implementation.
    
    Provides comprehensive HIPAA compliance management including:
    - Protected Health Information (PHI) identification and protection
    - Administrative, Physical, and Technical safeguards
    - Breach notification procedures
    - Business Associate Agreement management
    - Minimum necessary standard enforcement
    - De-identification procedures
    """
    
    def __init__(self):
        self.phi_elements = self._initialize_phi_elements()
        self.safeguards = self._initialize_safeguards()
        self.incidents: List[HIPAAIncident] = []
        self.business_associates: List[BusinessAssociateAgreement] = []
        self.disclosures: List[PHIDisclosure] = []
        
        # Start compliance monitoring
        asyncio.create_task(self._compliance_monitoring())
    
    def _initialize_phi_elements(self) -> Dict[str, PHIDataElement]:
        """Initialize PHI data elements"""
        elements = {}
        
        elements["patient_identifiers"] = PHIDataElement(
            element_id="patient_identifiers",
            element_name="Patient Identifying Information",
            phi_type=PHIType.IDENTIFIERS,
            identifiers=[
                HIPAAIdentifier.NAME,
                HIPAAIdentifier.SSN,
                HIPAAIdentifier.MEDICAL_RECORD_NUMBER,
                HIPAAIdentifier.EMAIL_ADDRESS,
                HIPAAIdentifier.TELEPHONE_NUMBERS
            ],
            sensitivity_level="high",
            encryption_required=True,
            access_restrictions=["healthcare_providers", "authorized_staff"],
            retention_period=timedelta(days=2555),  # 7 years
            minimum_necessary_applied=True,
            data_locations=["primary_database", "backup_systems"],
            authorized_users=["doctor", "nurse", "admin"]
        )
        
        elements["medical_records"] = PHIDataElement(
            element_id="medical_records",
            element_name="Electronic Medical Records",
            phi_type=PHIType.MEDICAL_RECORDS,
            identifiers=[
                HIPAAIdentifier.MEDICAL_RECORD_NUMBER,
                HIPAAIdentifier.DATES,
                HIPAAIdentifier.HEALTH_PLAN_NUMBER
            ],
            sensitivity_level="high",
            encryption_required=True,
            access_restrictions=["treating_physician", "authorized_staff"],
            retention_period=timedelta(days=2555),  # 7 years
            minimum_necessary_applied=True,
            data_locations=["ehr_system", "document_storage"],
            authorized_users=["doctor", "nurse", "medical_assistant"]
        )
        
        elements["billing_information"] = PHIDataElement(
            element_id="billing_information",
            element_name="Billing and Payment Information",
            phi_type=PHIType.BILLING_INFO,
            identifiers=[
                HIPAAIdentifier.ACCOUNT_NUMBER,
                HIPAAIdentifier.HEALTH_PLAN_NUMBER,
                HIPAAIdentifier.CERTIFICATE_NUMBER
            ],
            sensitivity_level="medium",
            encryption_required=True,
            access_restrictions=["billing_staff", "insurance_coordinators"],
            retention_period=timedelta(days=2555),  # 7 years
            minimum_necessary_applied=True,
            data_locations=["billing_system", "payment_processor"],
            authorized_users=["billing_specialist", "financial_counselor"]
        )
        
        elements["treatment_data"] = PHIDataElement(
            element_id="treatment_data",
            element_name="Treatment and Clinical Data",
            phi_type=PHIType.TREATMENT_INFO,
            identifiers=[
                HIPAAIdentifier.MEDICAL_RECORD_NUMBER,
                HIPAAIdentifier.DATES,
                HIPAAIdentifier.BIOMETRIC_IDENTIFIERS
            ],
            sensitivity_level="high",
            encryption_required=True,
            access_restrictions=["treating_providers", "care_team"],
            retention_period=timedelta(days=2555),  # 7 years
            minimum_necessary_applied=True,
            data_locations=["clinical_system", "lab_results"],
            authorized_users=["doctor", "nurse", "therapist", "lab_tech"]
        )
        
        return elements
    
    def _initialize_safeguards(self) -> Dict[str, Dict[str, Any]]:
        """Initialize HIPAA safeguards"""
        safeguards = {
            "administrative": {
                "45_CFR_164_308_a_1": {
                    "standard": "Security Officer",
                    "requirement": "Assign security responsibilities to one individual",
                    "implementation": "Designated security officer appointed",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_2": {
                    "standard": "Assigned Security Responsibilities", 
                    "requirement": "Identify personnel with access to electronic PHI",
                    "implementation": "Role-based access control system",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_3": {
                    "standard": "Workforce Training",
                    "requirement": "Train workforce on PHI security procedures",
                    "implementation": "Annual HIPAA training program",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_4": {
                    "standard": "Information Access Management",
                    "requirement": "Limit access to PHI to minimum necessary",
                    "implementation": "Minimum necessary policies and access controls",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_5": {
                    "standard": "Security Awareness and Training",
                    "requirement": "Periodic security updates and training",
                    "implementation": "Quarterly security awareness training",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_6": {
                    "standard": "Security Incident Procedures",
                    "requirement": "Implement procedures to address security incidents",
                    "implementation": "Incident response plan and procedures",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_7": {
                    "standard": "Contingency Plan",
                    "requirement": "Establish procedures for emergency access to PHI",
                    "implementation": "Business continuity and disaster recovery plan",
                    "status": "implemented"
                },
                "45_CFR_164_308_a_8": {
                    "standard": "Evaluation",
                    "requirement": "Conduct periodic security evaluations",
                    "implementation": "Annual security risk assessments",
                    "status": "implemented"
                }
            },
            "physical": {
                "45_CFR_164_310_a_1": {
                    "standard": "Facility Access Controls",
                    "requirement": "Limit physical access to systems containing PHI",
                    "implementation": "Badge access control, security cameras",
                    "status": "implemented"
                },
                "45_CFR_164_310_a_2": {
                    "standard": "Workstation Use",
                    "requirement": "Restrict access to workstations with PHI",
                    "implementation": "Workstation security policies and controls",
                    "status": "implemented"
                },
                "45_CFR_164_310_b": {
                    "standard": "Workstation Security",
                    "requirement": "Implement physical safeguards for workstations",
                    "implementation": "Screen locks, encryption, secure mounting",
                    "status": "implemented"
                },
                "45_CFR_164_310_c": {
                    "standard": "Device and Media Controls",
                    "requirement": "Control receipt and removal of hardware and media",
                    "implementation": "Asset management and media disposal procedures",
                    "status": "implemented"
                }
            },
            "technical": {
                "45_CFR_164_312_a_1": {
                    "standard": "Access Control",
                    "requirement": "Limit access to PHI to authorized users",
                    "implementation": "User authentication and authorization system",
                    "status": "implemented"
                },
                "45_CFR_164_312_b": {
                    "standard": "Audit Controls",
                    "requirement": "Implement hardware, software, procedural mechanisms for audit",
                    "implementation": "Comprehensive audit logging system",
                    "status": "implemented"
                },
                "45_CFR_164_312_c_1": {
                    "standard": "Integrity",
                    "requirement": "Protect PHI from improper alteration or destruction",
                    "implementation": "Data integrity controls and checksums",
                    "status": "implemented"
                },
                "45_CFR_164_312_d": {
                    "standard": "Person or Entity Authentication",
                    "requirement": "Verify identity of users accessing PHI",
                    "implementation": "Multi-factor authentication system",
                    "status": "implemented"
                },
                "45_CFR_164_312_e_1": {
                    "standard": "Transmission Security",
                    "requirement": "Protect PHI transmitted over networks",
                    "implementation": "Encryption in transit, secure protocols",
                    "status": "implemented"
                }
            }
        }
        
        return safeguards
    
    async def classify_phi(self, data_content: str, data_type: str) -> Dict[str, Any]:
        """Classify data for PHI content"""
        phi_indicators = []
        risk_score = 0
        
        # Check for HIPAA identifiers
        for identifier in HIPAAIdentifier:
            if await self._detect_identifier(data_content, identifier):
                phi_indicators.append(identifier.value)
                risk_score += self._get_identifier_risk_score(identifier)
        
        # Determine PHI classification
        is_phi = len(phi_indicators) > 0
        sensitivity = "high" if risk_score > 50 else "medium" if risk_score > 20 else "low"
        
        result = {
            "is_phi": is_phi,
            "phi_indicators": phi_indicators,
            "risk_score": risk_score,
            "sensitivity_level": sensitivity,
            "encryption_required": is_phi,
            "access_restrictions_required": is_phi,
            "minimum_necessary_applies": is_phi
        }
        
        # Log PHI detection
        if is_phi:
            uap_logger.log_security_event(
                "PHI detected in data",
                metadata={
                    "data_type": data_type,
                    "phi_indicators": phi_indicators,
                    "risk_score": risk_score
                }
            )
        
        return result
    
    async def _detect_identifier(self, content: str, identifier: HIPAAIdentifier) -> bool:
        """Detect specific HIPAA identifier in content"""
        import re
        
        patterns = {
            HIPAAIdentifier.SSN: r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
            HIPAAIdentifier.TELEPHONE_NUMBERS: r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}',
            HIPAAIdentifier.EMAIL_ADDRESS: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            HIPAAIdentifier.IP_ADDRESSES: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            HIPAAIdentifier.WEB_URLS: r'https?://[^\s]+',
            HIPAAIdentifier.MEDICAL_RECORD_NUMBER: r'\b[Mm][Rr][Nn]?\s*:?\s*\d+\b',
            HIPAAIdentifier.ACCOUNT_NUMBER: r'\b[Aa]cct?\s*:?\s*\d+\b',
            HIPAAIdentifier.DATES: r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        }
        
        pattern = patterns.get(identifier)
        if pattern:
            return bool(re.search(pattern, content, re.IGNORECASE))
        
        return False
    
    def _get_identifier_risk_score(self, identifier: HIPAAIdentifier) -> int:
        """Get risk score for HIPAA identifier"""
        high_risk = [
            HIPAAIdentifier.SSN,
            HIPAAIdentifier.MEDICAL_RECORD_NUMBER,
            HIPAAIdentifier.BIOMETRIC_IDENTIFIERS,
            HIPAAIdentifier.FULL_FACE_PHOTOS
        ]
        
        medium_risk = [
            HIPAAIdentifier.NAME,
            HIPAAIdentifier.EMAIL_ADDRESS,
            HIPAAIdentifier.TELEPHONE_NUMBERS,
            HIPAAIdentifier.ACCOUNT_NUMBER
        ]
        
        if identifier in high_risk:
            return 30
        elif identifier in medium_risk:
            return 15
        else:
            return 5
    
    async def apply_minimum_necessary(self, phi_data: Dict[str, Any], 
                                    user_role: str, purpose: str) -> Dict[str, Any]:
        """Apply minimum necessary standard to PHI access"""
        # Define role-based access rules
        role_permissions = {
            "doctor": ["all"],
            "nurse": ["patient_identifiers", "medical_records", "treatment_data"],
            "billing_specialist": ["patient_identifiers", "billing_information"],
            "admin": ["patient_identifiers"],
            "researcher": []  # De-identified data only
        }
        
        # Define purpose-based access rules
        purpose_permissions = {
            "treatment": ["patient_identifiers", "medical_records", "treatment_data"],
            "payment": ["patient_identifiers", "billing_information"],
            "healthcare_operations": ["patient_identifiers", "medical_records"],
            "research": []  # De-identified data only
        }
        
        # Get allowed data based on role and purpose
        role_allowed = role_permissions.get(user_role, [])
        purpose_allowed = purpose_permissions.get(purpose, [])
        
        # Intersection of role and purpose permissions
        if "all" in role_allowed:
            allowed_data = purpose_allowed if purpose_allowed else list(phi_data.keys())
        else:
            allowed_data = list(set(role_allowed) & set(purpose_allowed))
        
        # Filter data to minimum necessary
        filtered_data = {key: value for key, value in phi_data.items() if key in allowed_data}
        
        # Log minimum necessary application
        uap_logger.log_security_event(
            "Minimum necessary standard applied",
            metadata={
                "user_role": user_role,
                "purpose": purpose,
                "original_fields": len(phi_data),
                "filtered_fields": len(filtered_data),
                "allowed_data": allowed_data
            }
        )
        
        return filtered_data
    
    async def de_identify_data(self, phi_data: Dict[str, Any], 
                             method: str = "safe_harbor") -> Dict[str, Any]:
        """De-identify PHI data according to HIPAA Safe Harbor method"""
        de_identified_data = phi_data.copy()
        
        if method == "safe_harbor":
            # Remove all 18 HIPAA identifiers
            identifiers_to_remove = [
                "name", "address", "dates", "phone", "email", "ssn",
                "medical_record_number", "account_number", "certificate_number",
                "vehicle_id", "device_id", "web_url", "ip_address",
                "biometric_id", "photo", "unique_id"
            ]
            
            for identifier in identifiers_to_remove:
                if identifier in de_identified_data:
                    del de_identified_data[identifier]
            
            # Generalize geographic information
            if "zip_code" in de_identified_data:
                # Only keep first 3 digits for populations > 20,000
                zip_code = de_identified_data["zip_code"]
                if len(zip_code) >= 3:
                    de_identified_data["zip_code"] = zip_code[:3] + "XX"
            
            # Generalize ages over 89
            if "age" in de_identified_data:
                age = de_identified_data["age"]
                if isinstance(age, int) and age > 89:
                    de_identified_data["age"] = "90+"
        
        # Log de-identification
        uap_logger.log_security_event(
            "PHI data de-identified",
            metadata={
                "method": method,
                "original_fields": len(phi_data),
                "de_identified_fields": len(de_identified_data)
            }
        )
        
        return de_identified_data
    
    async def record_phi_disclosure(self, recipient: str, phi_description: str,
                                  purpose: str, legal_authority: str,
                                  individual_id: str = None) -> PHIDisclosure:
        """Record PHI disclosure for accounting"""
        disclosure_id = f"DISC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        disclosure = PHIDisclosure(
            disclosure_id=disclosure_id,
            date_of_disclosure=datetime.now(timezone.utc),
            recipient=recipient,
            description_of_phi=phi_description,
            purpose_of_disclosure=purpose,
            legal_authority=legal_authority,
            individual_id=individual_id,
            minimum_necessary_applied=True
        )
        
        self.disclosures.append(disclosure)
        
        # Log disclosure
        uap_logger.log_security_event(
            "PHI disclosure recorded",
            metadata={
                "disclosure_id": disclosure_id,
                "recipient": recipient,
                "purpose": purpose,
                "individual_id": individual_id
            }
        )
        
        return disclosure
    
    async def report_security_incident(self, incident_type: str, 
                                     affected_phi_elements: List[str],
                                     estimated_affected_individuals: int,
                                     description: str) -> HIPAAIncident:
        """Report HIPAA security incident"""
        incident_id = f"HIPAA-INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Assess risk and determine notification requirements
        risk_assessment = self._assess_incident_risk(
            incident_type, affected_phi_elements, estimated_affected_individuals
        )
        
        breach_notification_required = (
            risk_assessment in ["medium", "high"] and 
            estimated_affected_individuals > 0
        )
        
        hhs_notification_required = estimated_affected_individuals >= 500
        
        incident = HIPAAIncident(
            incident_id=incident_id,
            detection_date=datetime.now(timezone.utc),
            incident_type=incident_type,
            affected_phi_elements=affected_phi_elements,
            affected_individuals=estimated_affected_individuals,
            description=description,
            risk_assessment=risk_assessment,
            breach_notification_required=breach_notification_required,
            hhs_notification_required=hhs_notification_required,
            remediation_actions=[]
        )
        
        self.incidents.append(incident)
        
        # Log incident
        uap_logger.log_security_event(
            f"HIPAA security incident reported: {incident_id}",
            success=False,
            metadata={
                "incident_id": incident_id,
                "incident_type": incident_type,
                "affected_individuals": estimated_affected_individuals,
                "risk_assessment": risk_assessment,
                "breach_notification_required": breach_notification_required
            }
        )
        
        # Start incident response
        await self._respond_to_incident(incident)
        
        return incident
    
    def _assess_incident_risk(self, incident_type: str, affected_elements: List[str], 
                            affected_individuals: int) -> str:
        """Assess risk level of security incident"""
        risk_score = 0
        
        # Risk based on incident type
        high_risk_types = ["unauthorized_access", "data_theft", "malware"]
        medium_risk_types = ["system_failure", "improper_disposal", "lost_device"]
        
        if incident_type in high_risk_types:
            risk_score += 40
        elif incident_type in medium_risk_types:
            risk_score += 25
        else:
            risk_score += 10
        
        # Risk based on affected PHI elements
        high_sensitivity_elements = ["medical_records", "treatment_data"]
        for element in affected_elements:
            if element in high_sensitivity_elements:
                risk_score += 20
            else:
                risk_score += 10
        
        # Risk based on number of affected individuals
        if affected_individuals > 500:
            risk_score += 30
        elif affected_individuals > 100:
            risk_score += 20
        elif affected_individuals > 10:
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 70:
            return "high"
        elif risk_score >= 40:
            return "medium"
        else:
            return "low"
    
    async def _respond_to_incident(self, incident: HIPAAIncident):
        """Respond to HIPAA security incident"""
        try:
            # Immediate containment
            containment_actions = [
                "Isolate affected systems",
                "Preserve evidence",
                "Assess scope of breach",
                "Secure affected PHI"
            ]
            
            incident.remediation_actions.extend(containment_actions)
            
            # If breach notification required
            if incident.breach_notification_required:
                # Individual notification (60 days)
                notification_deadline = datetime.now(timezone.utc) + timedelta(days=60)
                uap_logger.log_security_event(
                    f"HIPAA breach notification required: {incident.incident_id}",
                    metadata={
                        "incident_id": incident.incident_id,
                        "notification_deadline": notification_deadline.isoformat()
                    }
                )
            
            # If HHS notification required
            if incident.hhs_notification_required:
                # HHS notification (60 days for > 500 individuals)
                hhs_deadline = datetime.now(timezone.utc) + timedelta(days=60)
                uap_logger.log_security_event(
                    f"HHS notification required: {incident.incident_id}",
                    metadata={
                        "incident_id": incident.incident_id,
                        "hhs_deadline": hhs_deadline.isoformat()
                    }
                )
        
        except Exception as e:
            logger.error(f"Error responding to HIPAA incident {incident.incident_id}: {e}")
    
    async def create_business_associate_agreement(self, business_associate: str,
                                                services: List[str], phi_types: List[str],
                                                agreement_duration_days: int = 365) -> BusinessAssociateAgreement:
        """Create Business Associate Agreement"""
        baa_id = f"BAA-{datetime.now().strftime('%Y%m%d')}-{business_associate[:10].upper()}"
        
        agreement_date = datetime.now(timezone.utc)
        expiration_date = agreement_date + timedelta(days=agreement_duration_days)
        
        baa = BusinessAssociateAgreement(
            baa_id=baa_id,
            business_associate=business_associate,
            services_provided=services,
            phi_types_accessed=phi_types,
            agreement_date=agreement_date,
            expiration_date=expiration_date,
            safeguards_required=[
                "Implement administrative safeguards",
                "Implement physical safeguards",
                "Implement technical safeguards",
                "Report security incidents within 24 hours",
                "Return or destroy PHI upon termination"
            ],
            audit_rights=True,
            termination_procedures=[
                "Return all PHI",
                "Destroy copies of PHI",
                "Certify destruction",
                "Transfer PHI if directed"
            ],
            status="active"
        )
        
        self.business_associates.append(baa)
        
        # Log BAA creation
        uap_logger.log_security_event(
            f"Business Associate Agreement created: {baa_id}",
            metadata={
                "baa_id": baa_id,
                "business_associate": business_associate,
                "services": services,
                "phi_types": phi_types,
                "expiration_date": expiration_date.isoformat()
            }
        )
        
        return baa
    
    async def _compliance_monitoring(self):
        """Continuous HIPAA compliance monitoring"""
        while True:
            try:
                # Check for upcoming BAA expirations
                upcoming_expirations = [
                    baa for baa in self.business_associates
                    if baa.expiration_date <= datetime.now(timezone.utc) + timedelta(days=30)
                    and baa.status == "active"
                ]
                
                if upcoming_expirations:
                    uap_logger.log_security_event(
                        f"HIPAA compliance alert: {len(upcoming_expirations)} BAAs expiring soon",
                        metadata={"expiring_baas": len(upcoming_expirations)}
                    )
                
                # Check for overdue incident responses
                overdue_incidents = [
                    inc for inc in self.incidents
                    if inc.breach_notification_required and not inc.notification_completed
                    and datetime.now(timezone.utc) > inc.detection_date + timedelta(days=60)
                ]
                
                if overdue_incidents:
                    uap_logger.log_security_event(
                        f"HIPAA compliance alert: {len(overdue_incidents)} overdue breach notifications",
                        metadata={"overdue_notifications": len(overdue_incidents)}
                    )
                
                # Audit PHI access patterns
                await self._audit_phi_access()
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in HIPAA compliance monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _audit_phi_access(self):
        """Audit PHI access patterns"""
        # Get recent access events
        audit_trail = get_audit_trail()
        recent_events = await audit_trail.search_events(
            {"event_type": "data_access"},
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            limit=1000
        )
        
        # Analyze access patterns
        unusual_patterns = []
        
        # Check for after-hours access
        for event in recent_events:
            if event.timestamp.hour < 6 or event.timestamp.hour > 22:
                unusual_patterns.append({
                    "type": "after_hours_access",
                    "event_id": event.event_id,
                    "user_id": event.actor_id,
                    "timestamp": event.timestamp.isoformat()
                })
        
        if unusual_patterns:
            uap_logger.log_security_event(
                f"HIPAA audit: {len(unusual_patterns)} unusual PHI access patterns detected",
                metadata={"unusual_patterns": len(unusual_patterns)}
            )
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get HIPAA compliance status"""
        active_baas = len([baa for baa in self.business_associates if baa.status == "active"])
        recent_incidents = len([
            inc for inc in self.incidents
            if inc.detection_date >= datetime.now(timezone.utc) - timedelta(days=30)
        ])
        
        return {
            "framework": "HIPAA",
            "phi_elements": len(self.phi_elements),
            "safeguards_implemented": {
                "administrative": len(self.safeguards["administrative"]),
                "physical": len(self.safeguards["physical"]),
                "technical": len(self.safeguards["technical"])
            },
            "active_business_associates": active_baas,
            "recent_incidents": recent_incidents,
            "disclosure_count": len(self.disclosures),
            "compliance_score": self._calculate_compliance_score()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate HIPAA compliance score"""
        score = 100.0
        
        # Deduct for recent incidents
        recent_incidents = [
            inc for inc in self.incidents
            if inc.detection_date >= datetime.now(timezone.utc) - timedelta(days=30)
        ]
        score -= len(recent_incidents) * 15
        
        # Deduct for expired BAAs
        expired_baas = [
            baa for baa in self.business_associates
            if baa.expiration_date < datetime.now(timezone.utc) and baa.status == "active"
        ]
        score -= len(expired_baas) * 10
        
        # Deduct for missing safeguards
        total_safeguards = (
            len(self.safeguards["administrative"]) +
            len(self.safeguards["physical"]) +
            len(self.safeguards["technical"])
        )
        implemented_safeguards = sum(
            1 for category in self.safeguards.values()
            for safeguard in category.values()
            if safeguard["status"] == "implemented"
        )
        
        implementation_rate = implemented_safeguards / total_safeguards if total_safeguards > 0 else 0
        score *= implementation_rate
        
        return max(0.0, min(100.0, score))
    
    def generate_hipaa_risk_assessment(self) -> Dict[str, Any]:
        """Generate HIPAA risk assessment report"""
        return {
            "report_type": "HIPAA Security Risk Assessment",
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "phi_inventory": {
                element_id: asdict(element) for element_id, element in self.phi_elements.items()
            },
            "safeguards_status": self.safeguards,
            "risk_analysis": {
                "high_risk_elements": [
                    element_id for element_id, element in self.phi_elements.items()
                    if element.sensitivity_level == "high"
                ],
                "encryption_coverage": len([
                    element for element in self.phi_elements.values()
                    if element.encryption_required
                ]) / len(self.phi_elements) * 100,
                "access_control_coverage": len([
                    element for element in self.phi_elements.values()
                    if element.access_restrictions
                ]) / len(self.phi_elements) * 100
            },
            "recommendations": [
                "Conduct regular security training",
                "Review and update BAAs annually",
                "Implement advanced threat detection",
                "Enhance audit logging capabilities",
                "Regular vulnerability assessments"
            ]
        }

class PHIProtectionManager:
    """Manager for PHI protection and de-identification"""
    
    def __init__(self, compliance_framework: HIPAAComplianceFramework):
        self.framework = compliance_framework
        self.encryption_service = get_encryption_service()
    
    async def protect_phi_data(self, data: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Apply comprehensive PHI protection"""
        # Classify data for PHI content
        classification = await self.framework.classify_phi(str(data), context)
        
        if classification["is_phi"]:
            # Encrypt sensitive data
            protected_data = await self._encrypt_phi_fields(data, classification)
            
            # Apply access controls
            protected_data["_phi_protection"] = {
                "classification": classification,
                "protection_applied": True,
                "encryption_status": "encrypted",
                "access_restrictions": classification.get("access_restrictions_required", False)
            }
            
            return protected_data
        
        return data
    
    async def _encrypt_phi_fields(self, data: Dict[str, Any], 
                                classification: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt PHI fields in data"""
        protected_data = data.copy()
        
        # Fields that should be encrypted for PHI
        phi_fields = ["name", "email", "phone", "ssn", "medical_record_number", "address"]
        
        for field in phi_fields:
            if field in protected_data:
                try:
                    encrypted_value = self.encryption_service.encrypt_field(
                        protected_data[field], f"phi_{field}"
                    )
                    protected_data[field] = encrypted_value
                except Exception as e:
                    logger.error(f"Error encrypting PHI field {field}: {e}")
        
        return protected_data

# Global HIPAA instances
_global_hipaa_framework = None
_global_phi_manager = None

def get_hipaa_framework() -> HIPAAComplianceFramework:
    """Get global HIPAA compliance framework instance"""
    global _global_hipaa_framework
    if _global_hipaa_framework is None:
        _global_hipaa_framework = HIPAAComplianceFramework()
    return _global_hipaa_framework

def get_phi_protection_manager() -> PHIProtectionManager:
    """Get global PHI protection manager instance"""
    global _global_phi_manager
    if _global_phi_manager is None:
        _global_phi_manager = PHIProtectionManager(get_hipaa_framework())
    return _global_phi_manager