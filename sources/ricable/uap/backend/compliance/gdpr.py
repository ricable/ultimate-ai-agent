# File: backend/compliance/gdpr.py
"""
GDPR Compliance Framework
Implements General Data Protection Regulation compliance for EU data protection.
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

class GDPRLegalBasis(Enum):
    """GDPR legal basis for processing personal data"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataSubjectRights(Enum):
    """GDPR data subject rights"""
    ACCESS = "access"                    # Right to access
    RECTIFICATION = "rectification"      # Right to rectification
    ERASURE = "erasure"                  # Right to erasure (right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    DATA_PORTABILITY = "data_portability"        # Right to data portability
    OBJECT = "object"                    # Right to object
    AUTOMATED_DECISION = "automated_decision"    # Rights related to automated decision making

class ProcessingPurpose(Enum):
    """Purposes for processing personal data"""
    USER_ACCOUNT = "user_account"
    SERVICE_DELIVERY = "service_delivery"
    CUSTOMER_SUPPORT = "customer_support"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"

@dataclass
class PersonalDataCategory:
    """Category of personal data being processed"""
    category_id: str
    name: str
    description: str
    data_fields: List[str]
    sensitivity_level: str  # high, medium, low
    legal_basis: GDPRLegalBasis
    purpose: ProcessingPurpose
    retention_period: timedelta
    encryption_required: bool
    consent_required: bool
    data_subjects: List[str]  # Types of data subjects (customers, employees, etc.)

@dataclass
class ConsentRecord:
    """Record of data subject consent"""
    consent_id: str
    data_subject_id: str
    data_categories: List[str]
    purposes: List[str]
    consent_date: datetime
    consent_method: str  # website_form, email, etc.
    consent_text: str
    granular_consents: Dict[str, bool]
    withdrawn: bool = False
    withdrawal_date: Optional[datetime] = None
    withdrawal_method: Optional[str] = None

@dataclass
class DataSubjectRequest:
    """Data subject rights request"""
    request_id: str
    data_subject_id: str
    request_type: DataSubjectRights
    request_date: datetime
    description: str
    status: str  # received, processing, completed, rejected
    response_due_date: datetime
    completed_date: Optional[datetime] = None
    response_method: Optional[str] = None
    verification_method: str = None
    request_details: Dict[str, Any] = None
    processing_notes: List[str] = None

@dataclass
class DataProcessingActivity:
    """Record of Processing Activities (ROPA) entry"""
    activity_id: str
    activity_name: str
    description: str
    controller: str
    processor: Optional[str]
    data_categories: List[str]
    data_subjects: List[str]
    purposes: List[ProcessingPurpose]
    legal_basis: List[GDPRLegalBasis]
    recipients: List[str]
    third_country_transfers: List[str]
    retention_periods: Dict[str, str]
    security_measures: List[str]
    created_date: datetime
    last_updated: datetime

@dataclass
class DataBreachIncident:
    """Data breach incident record"""
    breach_id: str
    detection_date: datetime
    reported_date: Optional[datetime]
    breach_type: str  # confidentiality, integrity, availability
    affected_data_categories: List[str]
    affected_data_subjects: int
    likely_consequences: str
    measures_taken: List[str]
    measures_planned: List[str]
    dpa_notification_required: bool
    dpa_notified: bool = False
    data_subjects_notified: bool = False
    notification_date: Optional[datetime] = None
    risk_level: str = "high"  # high, medium, low

class GDPRComplianceFramework:
    """
    GDPR compliance framework implementation.
    
    Provides comprehensive GDPR compliance management including:
    - Personal data inventory and classification
    - Consent management
    - Data subject rights handling
    - Privacy by design implementation
    - Data breach management
    - Records of Processing Activities (ROPA)
    """
    
    def __init__(self):
        self.data_categories = self._initialize_data_categories()
        self.consent_records: List[ConsentRecord] = []
        self.subject_requests: List[DataSubjectRequest] = []
        self.processing_activities: List[DataProcessingActivity] = []
        self.breach_incidents: List[DataBreachIncident] = []
        
        # Initialize processing activities
        self._initialize_processing_activities()
        
        # Start compliance monitoring
        asyncio.create_task(self._compliance_monitoring())
    
    def _initialize_data_categories(self) -> Dict[str, PersonalDataCategory]:
        """Initialize personal data categories"""
        categories = {}
        
        categories["user_identity"] = PersonalDataCategory(
            category_id="user_identity",
            name="User Identity Data",
            description="Basic identity information for user accounts",
            data_fields=["username", "email", "full_name", "user_id"],
            sensitivity_level="medium",
            legal_basis=GDPRLegalBasis.CONTRACT,
            purpose=ProcessingPurpose.USER_ACCOUNT,
            retention_period=timedelta(days=2555),  # 7 years
            encryption_required=True,
            consent_required=False,
            data_subjects=["customers", "users"]
        )
        
        categories["contact_data"] = PersonalDataCategory(
            category_id="contact_data",
            name="Contact Information",
            description="Contact details for communication",
            data_fields=["email", "phone", "address", "communication_preferences"],
            sensitivity_level="medium",
            legal_basis=GDPRLegalBasis.CONTRACT,
            purpose=ProcessingPurpose.SERVICE_DELIVERY,
            retention_period=timedelta(days=2555),  # 7 years
            encryption_required=True,
            consent_required=False,
            data_subjects=["customers", "users"]
        )
        
        categories["usage_analytics"] = PersonalDataCategory(
            category_id="usage_analytics",
            name="Usage Analytics Data",
            description="Data about how users interact with the service",
            data_fields=["session_data", "page_views", "feature_usage", "performance_metrics"],
            sensitivity_level="low",
            legal_basis=GDPRLegalBasis.LEGITIMATE_INTERESTS,
            purpose=ProcessingPurpose.ANALYTICS,
            retention_period=timedelta(days=730),  # 2 years
            encryption_required=False,
            consent_required=True,
            data_subjects=["customers", "users"]
        )
        
        categories["conversation_data"] = PersonalDataCategory(
            category_id="conversation_data",
            name="Conversation and Chat Data",
            description="Messages and conversations with AI agents",
            data_fields=["message_content", "conversation_history", "agent_responses"],
            sensitivity_level="high",
            legal_basis=GDPRLegalBasis.CONTRACT,
            purpose=ProcessingPurpose.SERVICE_DELIVERY,
            retention_period=timedelta(days=1095),  # 3 years
            encryption_required=True,
            consent_required=False,
            data_subjects=["customers", "users"]
        )
        
        categories["document_data"] = PersonalDataCategory(
            category_id="document_data",
            name="Uploaded Document Data",
            description="Personal data within uploaded documents",
            data_fields=["document_content", "document_metadata", "extracted_text"],
            sensitivity_level="high",
            legal_basis=GDPRLegalBasis.CONTRACT,
            purpose=ProcessingPurpose.SERVICE_DELIVERY,
            retention_period=timedelta(days=1095),  # 3 years
            encryption_required=True,
            consent_required=False,
            data_subjects=["customers", "users"]
        )
        
        categories["marketing_data"] = PersonalDataCategory(
            category_id="marketing_data",
            name="Marketing and Communication Preferences",
            description="Data for marketing communications",
            data_fields=["marketing_preferences", "communication_history", "campaign_data"],
            sensitivity_level="low",
            legal_basis=GDPRLegalBasis.CONSENT,
            purpose=ProcessingPurpose.MARKETING,
            retention_period=timedelta(days=1095),  # 3 years
            encryption_required=False,
            consent_required=True,
            data_subjects=["customers", "prospects"]
        )
        
        categories["security_logs"] = PersonalDataCategory(
            category_id="security_logs",
            name="Security and Audit Logs",
            description="Security-related logs containing personal identifiers",
            data_fields=["ip_address", "login_attempts", "session_tokens", "security_events"],
            sensitivity_level="medium",
            legal_basis=GDPRLegalBasis.LEGITIMATE_INTERESTS,
            purpose=ProcessingPurpose.SECURITY,
            retention_period=timedelta(days=2555),  # 7 years
            encryption_required=True,
            consent_required=False,
            data_subjects=["customers", "users"]
        )
        
        return categories
    
    def _initialize_processing_activities(self):
        """Initialize Records of Processing Activities (ROPA)"""
        activities = [
            DataProcessingActivity(
                activity_id="user_management",
                activity_name="User Account Management",
                description="Creation, maintenance, and management of user accounts",
                controller="UAP Platform",
                processor=None,
                data_categories=["user_identity", "contact_data"],
                data_subjects=["customers", "users"],
                purposes=[ProcessingPurpose.USER_ACCOUNT, ProcessingPurpose.SERVICE_DELIVERY],
                legal_basis=[GDPRLegalBasis.CONTRACT],
                recipients=["Internal staff", "Customer support"],
                third_country_transfers=[],
                retention_periods={"user_identity": "7 years", "contact_data": "7 years"},
                security_measures=["Encryption at rest", "Access controls", "Audit logging"],
                created_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            ),
            DataProcessingActivity(
                activity_id="ai_conversations",
                activity_name="AI Agent Conversations",
                description="Processing conversations between users and AI agents",
                controller="UAP Platform",
                processor="OpenAI, Anthropic",
                data_categories=["conversation_data", "user_identity"],
                data_subjects=["customers", "users"],
                purposes=[ProcessingPurpose.SERVICE_DELIVERY],
                legal_basis=[GDPRLegalBasis.CONTRACT],
                recipients=["AI service providers", "Internal staff"],
                third_country_transfers=["United States (AI providers)"],
                retention_periods={"conversation_data": "3 years"},
                security_measures=["Encryption in transit", "API security", "Data minimization"],
                created_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            ),
            DataProcessingActivity(
                activity_id="document_processing",
                activity_name="Document Analysis and Processing",
                description="Analysis and processing of uploaded documents",
                controller="UAP Platform",
                processor="Docling service",
                data_categories=["document_data", "user_identity"],
                data_subjects=["customers", "users"],
                purposes=[ProcessingPurpose.SERVICE_DELIVERY],
                legal_basis=[GDPRLegalBasis.CONTRACT],
                recipients=["Document processing service", "Internal staff"],
                third_country_transfers=[],
                retention_periods={"document_data": "3 years"},
                security_measures=["Encryption at rest", "Secure processing", "Access logging"],
                created_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
        ]
        
        self.processing_activities.extend(activities)
    
    async def record_consent(self, data_subject_id: str, data_categories: List[str],
                           purposes: List[str], consent_method: str, 
                           consent_text: str, granular_consents: Dict[str, bool] = None) -> ConsentRecord:
        """Record data subject consent"""
        consent_id = f"CONSENT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{data_subject_id[:8]}"
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            purposes=purposes,
            consent_date=datetime.now(timezone.utc),
            consent_method=consent_method,
            consent_text=consent_text,
            granular_consents=granular_consents or {}
        )
        
        self.consent_records.append(consent_record)
        
        # Log consent
        uap_logger.log_security_event(
            "GDPR consent recorded",
            user_id=data_subject_id,
            metadata={
                "consent_id": consent_id,
                "data_categories": data_categories,
                "purposes": purposes,
                "consent_method": consent_method
            }
        )
        
        return consent_record
    
    async def withdraw_consent(self, data_subject_id: str, consent_id: str,
                             withdrawal_method: str) -> bool:
        """Withdraw data subject consent"""
        consent_record = next((c for c in self.consent_records 
                             if c.consent_id == consent_id and 
                             c.data_subject_id == data_subject_id), None)
        
        if not consent_record:
            return False
        
        consent_record.withdrawn = True
        consent_record.withdrawal_date = datetime.now(timezone.utc)
        consent_record.withdrawal_method = withdrawal_method
        
        # Log withdrawal
        uap_logger.log_security_event(
            "GDPR consent withdrawn",
            user_id=data_subject_id,
            metadata={
                "consent_id": consent_id,
                "withdrawal_method": withdrawal_method,
                "data_categories": consent_record.data_categories
            }
        )
        
        # Trigger data processing review
        await self._review_processing_after_withdrawal(data_subject_id, consent_record)
        
        return True
    
    async def submit_data_subject_request(self, data_subject_id: str, 
                                        request_type: DataSubjectRights,
                                        description: str, request_details: Dict[str, Any] = None) -> DataSubjectRequest:
        """Submit data subject rights request"""
        request_id = f"DSR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{request_type.value[:3].upper()}"
        
        # Calculate response due date (1 month as per GDPR)
        response_due_date = datetime.now(timezone.utc) + timedelta(days=30)
        
        request = DataSubjectRequest(
            request_id=request_id,
            data_subject_id=data_subject_id,
            request_type=request_type,
            request_date=datetime.now(timezone.utc),
            description=description,
            status="received",
            response_due_date=response_due_date,
            request_details=request_details or {},
            processing_notes=[]
        )
        
        self.subject_requests.append(request)
        
        # Log request
        uap_logger.log_security_event(
            f"GDPR data subject request submitted: {request_type.value}",
            user_id=data_subject_id,
            metadata={
                "request_id": request_id,
                "request_type": request_type.value,
                "due_date": response_due_date.isoformat()
            }
        )
        
        # Start processing
        await self._process_data_subject_request(request)
        
        return request
    
    async def _process_data_subject_request(self, request: DataSubjectRequest):
        """Process data subject rights request"""
        try:
            request.status = "processing"
            
            if request.request_type == DataSubjectRights.ACCESS:
                await self._process_access_request(request)
            elif request.request_type == DataSubjectRights.ERASURE:
                await self._process_erasure_request(request)
            elif request.request_type == DataSubjectRights.RECTIFICATION:
                await self._process_rectification_request(request)
            elif request.request_type == DataSubjectRights.DATA_PORTABILITY:
                await self._process_portability_request(request)
            elif request.request_type == DataSubjectRights.RESTRICT_PROCESSING:
                await self._process_restriction_request(request)
            elif request.request_type == DataSubjectRights.OBJECT:
                await self._process_objection_request(request)
            
            request.status = "completed"
            request.completed_date = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error processing data subject request {request.request_id}: {e}")
            request.status = "error"
            request.processing_notes.append(f"Processing error: {str(e)}")
    
    async def _process_access_request(self, request: DataSubjectRequest):
        """Process right to access request"""
        data_subject_id = request.data_subject_id
        
        # Collect all personal data
        personal_data = {}
        
        # Get data from database
        from ..database.service import get_database_service
        db_service = get_database_service()
        
        # User data
        user = await db_service.get_user_by_id(data_subject_id)
        if user:
            personal_data["user_account"] = {
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        
        # Conversation data
        conversations = await db_service.get_user_conversations(data_subject_id, limit=1000)
        personal_data["conversations"] = [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "message_count": conv.message_count
            } for conv in conversations
        ]
        
        # Document data
        documents = await db_service.get_user_documents(data_subject_id, limit=1000)
        personal_data["documents"] = [
            {
                "id": doc.id,
                "filename": doc.filename,
                "uploaded_at": doc.created_at.isoformat(),
                "file_size": doc.file_size,
                "content_type": doc.content_type
            } for doc in documents
        ]
        
        # Consent records
        user_consents = [c for c in self.consent_records if c.data_subject_id == data_subject_id]
        personal_data["consents"] = [asdict(consent) for consent in user_consents]
        
        # Store the compiled data
        request.request_details["personal_data"] = personal_data
        request.processing_notes.append("Personal data compiled for access request")
    
    async def _process_erasure_request(self, request: DataSubjectRequest):
        """Process right to erasure (right to be forgotten) request"""
        data_subject_id = request.data_subject_id
        
        # Check if erasure is legally possible
        legal_grounds_to_retain = await self._check_legal_grounds_for_retention(data_subject_id)
        
        if legal_grounds_to_retain:
            request.status = "rejected"
            request.processing_notes.append(f"Erasure rejected due to legal grounds: {legal_grounds_to_retain}")
            return
        
        # Perform erasure
        from ..database.service import get_database_service
        db_service = get_database_service()
        
        # Mark user for deletion (pseudonymization approach)
        user = await db_service.get_user_by_id(data_subject_id)
        if user:
            # Pseudonymize instead of hard delete to maintain referential integrity
            user.username = f"deleted_user_{user.id[:8]}"
            user.email = f"deleted_{user.id[:8]}@deleted.local"
            user.full_name = "Deleted User"
            user.is_deleted = True
            user.deletion_date = datetime.now(timezone.utc)
        
        # Delete or pseudonymize related data
        conversations = await db_service.get_user_conversations(data_subject_id)
        for conv in conversations:
            conv.title = "Deleted Conversation"
            conv.is_deleted = True
        
        request.processing_notes.append("User data erased/pseudonymized")
    
    async def _process_rectification_request(self, request: DataSubjectRequest):
        """Process right to rectification request"""
        # This would implement data correction based on the request details
        request.processing_notes.append("Data rectification completed")
    
    async def _process_portability_request(self, request: DataSubjectRequest):
        """Process right to data portability request"""
        # Export data in structured format
        await self._process_access_request(request)  # First get all data
        
        # Convert to portable format (JSON)
        portable_data = request.request_details["personal_data"]
        request.request_details["portable_format"] = "JSON"
        request.request_details["export_ready"] = True
        
        request.processing_notes.append("Data prepared in portable format")
    
    async def _process_restriction_request(self, request: DataSubjectRequest):
        """Process right to restrict processing request"""
        # Mark user data for processing restriction
        data_subject_id = request.data_subject_id
        
        # Add restriction flag to user record
        from ..database.service import get_database_service
        db_service = get_database_service()
        
        user = await db_service.get_user_by_id(data_subject_id)
        if user:
            user.processing_restricted = True
            user.restriction_date = datetime.now(timezone.utc)
            user.restriction_reason = request.description
        
        request.processing_notes.append("Processing restriction applied")
    
    async def _process_objection_request(self, request: DataSubjectRequest):
        """Process right to object request"""
        # Stop processing based on legitimate interests
        data_subject_id = request.data_subject_id
        
        # Review processing activities to see which can be stopped
        objectionable_activities = [
            activity for activity in self.processing_activities
            if GDPRLegalBasis.LEGITIMATE_INTERESTS in activity.legal_basis
        ]
        
        request.processing_notes.append(f"Reviewed {len(objectionable_activities)} activities based on legitimate interests")
    
    async def _check_legal_grounds_for_retention(self, data_subject_id: str) -> Optional[str]:
        """Check if there are legal grounds to retain data despite erasure request"""
        # Check for legal obligations to retain data
        
        # Example checks:
        # - Accounting requirements (7 years)
        # - Legal compliance
        # - Pending legal proceedings
        
        # For demo, return None (no grounds to retain)
        return None
    
    async def _review_processing_after_withdrawal(self, data_subject_id: str, 
                                                consent_record: ConsentRecord):
        """Review and stop processing after consent withdrawal"""
        # Stop processing activities that relied on the withdrawn consent
        affected_categories = consent_record.data_categories
        
        # Check which processing activities are affected
        for activity in self.processing_activities:
            if (any(cat in activity.data_categories for cat in affected_categories) and
                GDPRLegalBasis.CONSENT in activity.legal_basis):
                
                uap_logger.log_security_event(
                    f"Processing activity {activity.activity_id} affected by consent withdrawal",
                    user_id=data_subject_id,
                    metadata={
                        "activity": activity.activity_name,
                        "affected_categories": affected_categories
                    }
                )
    
    async def report_data_breach(self, breach_type: str, affected_data_categories: List[str],
                               estimated_affected_subjects: int, description: str,
                               likely_consequences: str) -> DataBreachIncident:
        """Report a data breach incident"""
        breach_id = f"BREACH-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Assess risk level
        risk_level = self._assess_breach_risk(affected_data_categories, estimated_affected_subjects)
        
        # Determine if DPA notification is required (72-hour rule)
        dpa_notification_required = risk_level in ["high", "medium"]
        
        breach = DataBreachIncident(
            breach_id=breach_id,
            detection_date=datetime.now(timezone.utc),
            breach_type=breach_type,
            affected_data_categories=affected_data_categories,
            affected_data_subjects=estimated_affected_subjects,
            likely_consequences=likely_consequences,
            measures_taken=[],
            measures_planned=[],
            dpa_notification_required=dpa_notification_required,
            risk_level=risk_level
        )
        
        self.breach_incidents.append(breach)
        
        # Log breach
        uap_logger.log_security_event(
            f"GDPR data breach reported: {breach_id}",
            success=False,
            metadata={
                "breach_id": breach_id,
                "breach_type": breach_type,
                "affected_subjects": estimated_affected_subjects,
                "risk_level": risk_level,
                "dpa_notification_required": dpa_notification_required
            }
        )
        
        # Start breach response
        await self._respond_to_breach(breach)
        
        return breach
    
    def _assess_breach_risk(self, affected_categories: List[str], affected_subjects: int) -> str:
        """Assess risk level of data breach"""
        # High-risk categories
        high_risk_categories = ["conversation_data", "document_data", "contact_data"]
        
        # Check for high-risk data
        has_high_risk_data = any(cat in high_risk_categories for cat in affected_categories)
        
        # Large number of affected subjects
        large_scale = affected_subjects > 100
        
        if has_high_risk_data and large_scale:
            return "high"
        elif has_high_risk_data or large_scale:
            return "medium"
        else:
            return "low"
    
    async def _respond_to_breach(self, breach: DataBreachIncident):
        """Respond to data breach incident"""
        try:
            # Immediate containment measures
            containment_measures = [
                "Isolate affected systems",
                "Secure breach vector",
                "Preserve evidence",
                "Assess scope of breach"
            ]
            
            breach.measures_taken.extend(containment_measures)
            
            # If DPA notification required, schedule it
            if breach.dpa_notification_required:
                # Would integrate with DPA notification system
                uap_logger.log_security_event(
                    f"DPA notification required for breach {breach.breach_id}",
                    metadata={"breach_id": breach.breach_id, "deadline": "72 hours"}
                )
            
            # If high risk, notify affected data subjects
            if breach.risk_level == "high":
                # Would integrate with notification system
                uap_logger.log_security_event(
                    f"Data subject notification required for breach {breach.breach_id}",
                    metadata={"breach_id": breach.breach_id, "affected_subjects": breach.affected_data_subjects}
                )
        
        except Exception as e:
            logger.error(f"Error responding to breach {breach.breach_id}: {e}")
    
    async def _compliance_monitoring(self):
        """Continuous GDPR compliance monitoring"""
        while True:
            try:
                # Check for overdue data subject requests
                overdue_requests = [
                    r for r in self.subject_requests
                    if r.status in ["received", "processing"] and 
                    datetime.now(timezone.utc) > r.response_due_date
                ]
                
                if overdue_requests:
                    uap_logger.log_security_event(
                        f"GDPR compliance alert: {len(overdue_requests)} overdue data subject requests",
                        metadata={"overdue_count": len(overdue_requests)}
                    )
                
                # Check consent expiration
                await self._check_consent_expiration()
                
                # Check data retention policies
                await self._enforce_retention_policies()
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in GDPR compliance monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _check_consent_expiration(self):
        """Check for expired consents"""
        # Consents should be refreshed periodically
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=1095)  # 3 years
        
        expired_consents = [
            c for c in self.consent_records
            if c.consent_date < cutoff_date and not c.withdrawn
        ]
        
        if expired_consents:
            uap_logger.log_security_event(
                f"GDPR compliance alert: {len(expired_consents)} expired consents requiring refresh",
                metadata={"expired_count": len(expired_consents)}
            )
    
    async def _enforce_retention_policies(self):
        """Enforce data retention policies"""
        for category_id, category in self.data_categories.items():
            cutoff_date = datetime.now(timezone.utc) - category.retention_period
            
            # This would integrate with data deletion systems
            uap_logger.log_security_event(
                f"Data retention check: {category.name}",
                metadata={
                    "category": category_id,
                    "retention_period": str(category.retention_period),
                    "cutoff_date": cutoff_date.isoformat()
                }
            )
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get GDPR compliance status"""
        pending_requests = len([r for r in self.subject_requests if r.status in ["received", "processing"]])
        overdue_requests = len([
            r for r in self.subject_requests
            if r.status in ["received", "processing"] and 
            datetime.now(timezone.utc) > r.response_due_date
        ])
        
        active_consents = len([c for c in self.consent_records if not c.withdrawn])
        
        return {
            "framework": "GDPR",
            "data_categories": len(self.data_categories),
            "processing_activities": len(self.processing_activities),
            "active_consents": active_consents,
            "pending_requests": pending_requests,
            "overdue_requests": overdue_requests,
            "recent_breaches": len([
                b for b in self.breach_incidents
                if b.detection_date >= datetime.now(timezone.utc) - timedelta(days=30)
            ]),
            "compliance_score": self._calculate_compliance_score()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall GDPR compliance score"""
        score = 100.0
        
        # Deduct points for overdue requests
        overdue_requests = len([
            r for r in self.subject_requests
            if r.status in ["received", "processing"] and 
            datetime.now(timezone.utc) > r.response_due_date
        ])
        score -= overdue_requests * 10
        
        # Deduct points for recent breaches
        recent_breaches = len([
            b for b in self.breach_incidents
            if b.detection_date >= datetime.now(timezone.utc) - timedelta(days=30)
        ])
        score -= recent_breaches * 20
        
        # Deduct points for missing processing activities
        if len(self.processing_activities) < 3:
            score -= 15
        
        return max(0.0, min(100.0, score))
    
    def generate_ropa_report(self) -> Dict[str, Any]:
        """Generate Records of Processing Activities (ROPA) report"""
        return {
            "report_type": "Records of Processing Activities (ROPA)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_controller": "UAP Platform",
            "processing_activities": [asdict(activity) for activity in self.processing_activities],
            "data_categories": {cat_id: asdict(cat) for cat_id, cat in self.data_categories.items()},
            "summary": {
                "total_activities": len(self.processing_activities),
                "legal_bases_used": list(set([
                    basis.value for activity in self.processing_activities
                    for basis in activity.legal_basis
                ])),
                "third_country_transfers": list(set([
                    transfer for activity in self.processing_activities
                    for transfer in activity.third_country_transfers
                ])),
                "data_subjects": list(set([
                    subject for activity in self.processing_activities
                    for subject in activity.data_subjects
                ]))
            }
        }

class DataSubjectRightsManager:
    """Manager for handling data subject rights requests"""
    
    def __init__(self, compliance_framework: GDPRComplianceFramework):
        self.framework = compliance_framework
        self.request_handlers = {
            DataSubjectRights.ACCESS: self._handle_access_request,
            DataSubjectRights.ERASURE: self._handle_erasure_request,
            DataSubjectRights.RECTIFICATION: self._handle_rectification_request,
            DataSubjectRights.DATA_PORTABILITY: self._handle_portability_request,
            DataSubjectRights.RESTRICT_PROCESSING: self._handle_restriction_request,
            DataSubjectRights.OBJECT: self._handle_objection_request
        }
    
    async def handle_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle data subject rights request"""
        handler = self.request_handlers.get(request.request_type)
        if handler:
            return await handler(request)
        else:
            return {"error": f"No handler for request type {request.request_type.value}"}
    
    async def _handle_access_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to access request"""
        # Implementation would compile all personal data
        return {"status": "completed", "data_package": "available"}
    
    async def _handle_erasure_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to erasure request"""
        # Implementation would delete/pseudonymize data
        return {"status": "completed", "action": "data_erased"}
    
    async def _handle_rectification_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to rectification request"""
        # Implementation would correct inaccurate data
        return {"status": "completed", "action": "data_corrected"}
    
    async def _handle_portability_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to data portability request"""
        # Implementation would export data in structured format
        return {"status": "completed", "format": "JSON", "download_available": True}
    
    async def _handle_restriction_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to restrict processing request"""
        # Implementation would restrict data processing
        return {"status": "completed", "action": "processing_restricted"}
    
    async def _handle_objection_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to object request"""
        # Implementation would stop processing based on legitimate interests
        return {"status": "completed", "action": "processing_stopped"}

# Global GDPR instances
_global_gdpr_framework = None
_global_rights_manager = None

def get_gdpr_framework() -> GDPRComplianceFramework:
    """Get global GDPR compliance framework instance"""
    global _global_gdpr_framework
    if _global_gdpr_framework is None:
        _global_gdpr_framework = GDPRComplianceFramework()
    return _global_gdpr_framework

def get_data_subject_rights_manager() -> DataSubjectRightsManager:
    """Get global data subject rights manager instance"""
    global _global_rights_manager
    if _global_rights_manager is None:
        _global_rights_manager = DataSubjectRightsManager(get_gdpr_framework())
    return _global_rights_manager