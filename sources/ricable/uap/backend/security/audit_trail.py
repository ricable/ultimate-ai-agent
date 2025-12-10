# File: backend/security/audit_trail.py
"""
Immutable Audit Trail System for comprehensive security logging and compliance.
Provides tamper-proof logging with cryptographic integrity verification.
"""

import json
import hashlib
import hmac
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import asyncio
from collections import deque

from .encryption import DataEncryption
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CHANGE = "system_change"
    SECURITY_EVENT = "security_event"
    ADMIN_ACTION = "admin_action"
    API_ACCESS = "api_access"
    USER_MANAGEMENT = "user_management"
    CONFIGURATION_CHANGE = "configuration_change"

class AuditOutcome(Enum):
    """Audit event outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    DENIED = "denied"

@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    outcome: AuditOutcome
    actor_id: Optional[str]
    actor_type: str  # user, system, service, admin
    resource: str
    action: str
    description: str
    source_ip: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    details: Dict[str, Any]
    risk_score: int  # 1-10 scale
    
    # Integrity fields
    previous_hash: Optional[str] = None
    content_hash: str = ""
    signature: str = ""
    
    def __post_init__(self):
        """Calculate content hash after initialization"""
        if not self.content_hash:
            self.content_hash = self._calculate_content_hash()
    
    def _calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of event content"""
        # Create content string excluding hash and signature fields
        content = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "outcome": self.outcome.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "resource": self.resource,
            "action": self.action,
            "description": self.description,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "details": self.details,
            "risk_score": self.risk_score,
            "previous_hash": self.previous_hash
        }
        
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

class ImmutableAuditTrail:
    """
    Immutable audit trail system with cryptographic integrity protection.
    
    Features:
    - Cryptographic hash chaining for tamper detection
    - Digital signatures for authenticity verification
    - Encrypted storage of sensitive audit data
    - Real-time integrity monitoring
    - Compliance reporting capabilities
    """
    
    def __init__(self, signing_key: Optional[str] = None, storage_path: Optional[str] = None):
        self.signing_key = signing_key or self._generate_signing_key()
        self.storage_path = Path(storage_path or "logs/audit_trail")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory chain for fast access (last 1000 events)
        self.recent_events: deque = deque(maxlen=1000)
        self.last_event_hash: Optional[str] = None
        
        # Encryption service for sensitive data
        self.encryption = DataEncryption()
        
        # Event counter for sequence tracking
        self.event_counter = 0
        
        # Background tasks
        self._integrity_check_task = None
        self._storage_task = None
        
        # Load existing audit trail
        asyncio.create_task(self._initialize_audit_trail())
    
    def _generate_signing_key(self) -> str:
        """Generate HMAC signing key for audit event signatures"""
        import secrets
        return secrets.token_hex(32)
    
    async def _initialize_audit_trail(self):
        """Initialize audit trail from existing storage"""
        try:
            # Load the most recent events to establish chain continuity
            recent_files = sorted(
                self.storage_path.glob("audit_*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            if recent_files:
                # Load last file to get the chain head
                with open(recent_files[0], 'r') as f:
                    events_data = json.load(f)
                
                if events_data:
                    last_event_data = events_data[-1]
                    self.last_event_hash = last_event_data.get("content_hash")
                    self.event_counter = last_event_data.get("event_id", "0").split("-")[-1]
                    try:
                        self.event_counter = int(self.event_counter)
                    except ValueError:
                        self.event_counter = 0
            
            # Start background tasks
            self._integrity_check_task = asyncio.create_task(self._periodic_integrity_check())
            self._storage_task = asyncio.create_task(self._periodic_storage())
            
            uap_logger.log_security_event(
                "Audit trail system initialized",
                metadata={
                    "last_event_hash": self.last_event_hash,
                    "event_counter": self.event_counter
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize audit trail: {e}")
    
    async def log_event(self, event_type: AuditEventType, outcome: AuditOutcome,
                       actor_id: Optional[str], actor_type: str, resource: str,
                       action: str, description: str, details: Dict[str, Any] = None,
                       source_ip: Optional[str] = None, user_agent: Optional[str] = None,
                       session_id: Optional[str] = None, request_id: Optional[str] = None,
                       risk_score: int = 5) -> AuditEvent:
        """
        Log an audit event with immutable record.
        
        Args:
            event_type: Type of audit event
            outcome: Success/failure/etc
            actor_id: ID of the actor performing the action
            actor_type: Type of actor (user, system, service, admin)
            resource: Resource being accessed/modified
            action: Action being performed
            description: Human-readable description
            details: Additional event details
            source_ip: Source IP address
            user_agent: User agent string
            session_id: Session identifier
            request_id: Request identifier
            risk_score: Risk score (1-10)
            
        Returns:
            Created audit event
        """
        try:
            # Generate event ID
            self.event_counter += 1
            event_id = f"AUD-{datetime.now().strftime('%Y%m%d')}-{self.event_counter:06d}"
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                outcome=outcome,
                actor_id=actor_id,
                actor_type=actor_type,
                resource=resource,
                action=action,
                description=description,
                source_ip=source_ip,
                user_agent=user_agent,
                session_id=session_id,
                request_id=request_id,
                details=details or {},
                risk_score=risk_score,
                previous_hash=self.last_event_hash
            )
            
            # Calculate content hash (done in __post_init__)
            # Generate digital signature
            event.signature = self._sign_event(event)
            
            # Add to chain
            self.recent_events.append(event)
            self.last_event_hash = event.content_hash
            
            # Log to standard logging system
            uap_logger.log_security_event(
                f"Audit event: {action} on {resource}",
                user_id=actor_id,
                ip_address=source_ip,
                success=(outcome == AuditOutcome.SUCCESS),
                metadata={
                    "event_id": event_id,
                    "event_type": event_type.value,
                    "risk_score": risk_score,
                    "resource": resource,
                    "action": action
                }
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            # Create emergency log entry
            uap_logger.log_security_event(
                "Audit logging failure",
                success=False,
                metadata={"error": str(e), "action": action, "resource": resource}
            )
            raise
    
    def _sign_event(self, event: AuditEvent) -> str:
        """Generate HMAC signature for audit event"""
        message = f"{event.event_id}:{event.content_hash}:{event.timestamp.isoformat()}"
        signature = hmac.new(
            self.signing_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_event_signature(self, event: AuditEvent) -> bool:
        """Verify the digital signature of an audit event"""
        expected_signature = self._sign_event(event)
        return hmac.compare_digest(event.signature, expected_signature)
    
    def verify_chain_integrity(self, events: List[AuditEvent] = None) -> Dict[str, Any]:
        """
        Verify the integrity of the audit chain.
        
        Args:
            events: Events to verify (if None, uses recent events)
            
        Returns:
            Verification result with details
        """
        if events is None:
            events = list(self.recent_events)
        
        if not events:
            return {"valid": True, "message": "No events to verify"}
        
        result = {
            "valid": True,
            "total_events": len(events),
            "verified_events": 0,
            "signature_failures": [],
            "hash_chain_failures": [],
            "timestamp_issues": []
        }
        
        previous_hash = None
        previous_timestamp = None
        
        for i, event in enumerate(events):
            # Verify digital signature
            if not self.verify_event_signature(event):
                result["valid"] = False
                result["signature_failures"].append({
                    "event_id": event.event_id,
                    "position": i
                })
            
            # Verify content hash
            expected_hash = event._calculate_content_hash()
            if event.content_hash != expected_hash:
                result["valid"] = False
                result["hash_chain_failures"].append({
                    "event_id": event.event_id,
                    "position": i,
                    "expected_hash": expected_hash,
                    "actual_hash": event.content_hash
                })
            
            # Verify hash chain linkage
            if previous_hash is not None and event.previous_hash != previous_hash:
                result["valid"] = False
                result["hash_chain_failures"].append({
                    "event_id": event.event_id,
                    "position": i,
                    "expected_previous_hash": previous_hash,
                    "actual_previous_hash": event.previous_hash
                })
            
            # Verify timestamp ordering
            if previous_timestamp is not None and event.timestamp < previous_timestamp:
                result["valid"] = False
                result["timestamp_issues"].append({
                    "event_id": event.event_id,
                    "position": i,
                    "timestamp": event.timestamp.isoformat(),
                    "previous_timestamp": previous_timestamp.isoformat()
                })
            
            if result["valid"] or len(result["signature_failures"]) == 0:
                result["verified_events"] += 1
            
            previous_hash = event.content_hash
            previous_timestamp = event.timestamp
        
        return result
    
    async def _periodic_integrity_check(self):
        """Periodic integrity check of audit trail"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                verification_result = self.verify_chain_integrity()
                
                if not verification_result["valid"]:
                    uap_logger.log_security_event(
                        "Audit trail integrity violation detected",
                        success=False,
                        metadata=verification_result
                    )
                    
                    # Alert administrators
                    await self._send_integrity_alert(verification_result)
                else:
                    uap_logger.log_security_event(
                        "Audit trail integrity verification passed",
                        metadata={
                            "verified_events": verification_result["verified_events"],
                            "total_events": verification_result["total_events"]
                        }
                    )
                
            except Exception as e:
                logger.error(f"Error in periodic integrity check: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes on error
    
    async def _periodic_storage(self):
        """Periodic storage of audit events to disk"""
        while True:
            try:
                await asyncio.sleep(300)  # Store every 5 minutes
                
                if self.recent_events:
                    await self._store_events_to_disk()
                
            except Exception as e:
                logger.error(f"Error in periodic storage: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute on error
    
    async def _store_events_to_disk(self):
        """Store recent events to disk storage"""
        if not self.recent_events:
            return
        
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"audit_{timestamp}.json"
        filepath = self.storage_path / filename
        
        # Prepare events for storage
        events_data = []
        for event in self.recent_events:
            event_dict = asdict(event)
            event_dict["timestamp"] = event.timestamp.isoformat()
            
            # Encrypt sensitive details if needed
            if self._contains_sensitive_data(event.details):
                event_dict["details"] = self.encryption.encrypt_data(
                    json.dumps(event.details),
                    context={"event_id": event.event_id}
                )
                event_dict["details_encrypted"] = True
            
            events_data.append(event_dict)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2, default=str)
        
        # Clear recent events after storage
        self.recent_events.clear()
        
        uap_logger.log_security_event(
            "Audit events stored to disk",
            metadata={
                "filename": filename,
                "event_count": len(events_data)
            }
        )
    
    def _contains_sensitive_data(self, details: Dict[str, Any]) -> bool:
        """Check if event details contain sensitive information"""
        sensitive_keys = [
            "password", "token", "secret", "key", "credential",
            "ssn", "credit_card", "personal_data", "private"
        ]
        
        details_str = json.dumps(details).lower()
        return any(key in details_str for key in sensitive_keys)
    
    async def _send_integrity_alert(self, verification_result: Dict[str, Any]):
        """Send alert about integrity violations"""
        # This would integrate with alerting system
        uap_logger.log_security_event(
            "CRITICAL: Audit trail integrity violation - immediate investigation required",
            success=False,
            metadata={
                "alert_type": "integrity_violation",
                "verification_result": verification_result,
                "requires_immediate_attention": True
            }
        )
    
    async def search_events(self, filters: Dict[str, Any], 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 100) -> List[AuditEvent]:
        """
        Search audit events with filters.
        
        Args:
            filters: Search filters (actor_id, event_type, resource, etc.)
            start_date: Start date for search
            end_date: End date for search
            limit: Maximum number of results
            
        Returns:
            List of matching audit events
        """
        matching_events = []
        
        # Search recent events in memory
        for event in self.recent_events:
            if self._event_matches_filters(event, filters, start_date, end_date):
                matching_events.append(event)
                if len(matching_events) >= limit:
                    break
        
        # If we need more results, search disk storage
        if len(matching_events) < limit:
            disk_events = await self._search_disk_storage(
                filters, start_date, end_date, limit - len(matching_events)
            )
            matching_events.extend(disk_events)
        
        return matching_events[:limit]
    
    def _event_matches_filters(self, event: AuditEvent, filters: Dict[str, Any],
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> bool:
        """Check if event matches search filters"""
        # Date range check
        if start_date and event.timestamp < start_date:
            return False
        if end_date and event.timestamp > end_date:
            return False
        
        # Filter checks
        for key, value in filters.items():
            if hasattr(event, key):
                event_value = getattr(event, key)
                if isinstance(event_value, Enum):
                    event_value = event_value.value
                
                if isinstance(value, list):
                    if event_value not in value:
                        return False
                elif event_value != value:
                    return False
            elif key in event.details:
                if event.details[key] != value:
                    return False
        
        return True
    
    async def _search_disk_storage(self, filters: Dict[str, Any],
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 limit: int = 100) -> List[AuditEvent]:
        """Search audit events in disk storage"""
        matching_events = []
        
        # Get relevant files based on date range
        files_to_search = self._get_relevant_files(start_date, end_date)
        
        for filepath in files_to_search:
            try:
                with open(filepath, 'r') as f:
                    events_data = json.load(f)
                
                for event_dict in events_data:
                    # Reconstruct event object
                    event = self._reconstruct_event_from_dict(event_dict)
                    
                    if self._event_matches_filters(event, filters, start_date, end_date):
                        matching_events.append(event)
                        if len(matching_events) >= limit:
                            return matching_events
                            
            except Exception as e:
                logger.error(f"Error searching file {filepath}: {e}")
                continue
        
        return matching_events
    
    def _get_relevant_files(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[Path]:
        """Get audit files relevant to date range"""
        all_files = list(self.storage_path.glob("audit_*.json"))
        
        if not start_date and not end_date:
            return sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Filter files by date (simple approach - could be optimized)
        relevant_files = []
        for filepath in all_files:
            file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
            
            if start_date and file_mtime < start_date - timedelta(days=1):
                continue
            if end_date and file_mtime > end_date + timedelta(days=1):
                continue
            
            relevant_files.append(filepath)
        
        return sorted(relevant_files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    def _reconstruct_event_from_dict(self, event_dict: Dict[str, Any]) -> AuditEvent:
        """Reconstruct AuditEvent from dictionary"""
        # Handle encrypted details
        if event_dict.get("details_encrypted"):
            try:
                encrypted_details = event_dict["details"]
                decrypted_details = self.encryption.decrypt_data(
                    encrypted_details,
                    context={"event_id": event_dict["event_id"]}
                )
                event_dict["details"] = json.loads(decrypted_details)
            except Exception as e:
                logger.error(f"Failed to decrypt event details: {e}")
                event_dict["details"] = {"decryption_error": str(e)}
        
        # Convert timestamp back to datetime
        event_dict["timestamp"] = datetime.fromisoformat(event_dict["timestamp"])
        
        # Convert enum strings back to enums
        event_dict["event_type"] = AuditEventType(event_dict["event_type"])
        event_dict["outcome"] = AuditOutcome(event_dict["outcome"])
        
        # Remove encryption flag
        event_dict.pop("details_encrypted", None)
        
        return AuditEvent(**event_dict)
    
    async def generate_compliance_report(self, report_type: str,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report for audit trail.
        
        Args:
            report_type: Type of compliance report (soc2, gdpr, hipaa, etc.)
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report data
        """
        # Search all events in date range
        all_events = await self.search_events(
            {},
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        report = {
            "report_type": report_type,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(all_events),
            "integrity_status": self.verify_chain_integrity(all_events),
            "event_breakdown": self._analyze_events_for_compliance(all_events),
            "security_incidents": self._identify_security_incidents(all_events),
            "access_patterns": self._analyze_access_patterns(all_events),
            "compliance_status": self._assess_compliance_status(report_type, all_events)
        }
        
        return report
    
    def _analyze_events_for_compliance(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze events for compliance reporting"""
        breakdown = {
            "by_type": {},
            "by_outcome": {},
            "by_actor_type": {},
            "by_risk_score": {},
            "failed_authentications": 0,
            "data_access_events": 0,
            "admin_actions": 0,
            "system_changes": 0
        }
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            breakdown["by_type"][event_type] = breakdown["by_type"].get(event_type, 0) + 1
            
            # Count by outcome
            outcome = event.outcome.value
            breakdown["by_outcome"][outcome] = breakdown["by_outcome"].get(outcome, 0) + 1
            
            # Count by actor type
            actor_type = event.actor_type
            breakdown["by_actor_type"][actor_type] = breakdown["by_actor_type"].get(actor_type, 0) + 1
            
            # Count by risk score
            risk_bucket = f"risk_{event.risk_score}"
            breakdown["by_risk_score"][risk_bucket] = breakdown["by_risk_score"].get(risk_bucket, 0) + 1
            
            # Specific compliance metrics
            if event.event_type == AuditEventType.AUTHENTICATION and event.outcome == AuditOutcome.FAILURE:
                breakdown["failed_authentications"] += 1
            elif event.event_type == AuditEventType.DATA_ACCESS:
                breakdown["data_access_events"] += 1
            elif event.actor_type == "admin":
                breakdown["admin_actions"] += 1
            elif event.event_type == AuditEventType.SYSTEM_CHANGE:
                breakdown["system_changes"] += 1
        
        return breakdown
    
    def _identify_security_incidents(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Identify potential security incidents from audit events"""
        incidents = []
        
        # Group events by actor and time window
        from collections import defaultdict
        
        # Failed authentication attempts
        failed_auth_by_actor = defaultdict(list)
        for event in events:
            if (event.event_type == AuditEventType.AUTHENTICATION and 
                event.outcome == AuditOutcome.FAILURE):
                failed_auth_by_actor[event.actor_id or event.source_ip].append(event)
        
        # Check for brute force attempts
        for actor, failed_events in failed_auth_by_actor.items():
            if len(failed_events) >= 5:
                incidents.append({
                    "type": "potential_brute_force",
                    "actor": actor,
                    "event_count": len(failed_events),
                    "time_span": {
                        "start": min(e.timestamp for e in failed_events).isoformat(),
                        "end": max(e.timestamp for e in failed_events).isoformat()
                    }
                })
        
        # High-risk events
        high_risk_events = [e for e in events if e.risk_score >= 8]
        if high_risk_events:
            incidents.append({
                "type": "high_risk_activities",
                "event_count": len(high_risk_events),
                "events": [e.event_id for e in high_risk_events[:10]]  # Sample
            })
        
        return incidents
    
    def _analyze_access_patterns(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze access patterns for compliance"""
        patterns = {
            "unique_users": set(),
            "unique_ips": set(),
            "peak_activity_hours": {},
            "resource_access_frequency": {},
            "unusual_access_patterns": []
        }
        
        for event in events:
            if event.actor_id:
                patterns["unique_users"].add(event.actor_id)
            if event.source_ip:
                patterns["unique_ips"].add(event.source_ip)
            
            # Track activity by hour
            hour = event.timestamp.hour
            patterns["peak_activity_hours"][hour] = patterns["peak_activity_hours"].get(hour, 0) + 1
            
            # Track resource access
            resource = event.resource
            patterns["resource_access_frequency"][resource] = patterns["resource_access_frequency"].get(resource, 0) + 1
        
        # Convert sets to counts
        patterns["unique_users"] = len(patterns["unique_users"])
        patterns["unique_ips"] = len(patterns["unique_ips"])
        
        return patterns
    
    def _assess_compliance_status(self, report_type: str, events: List[AuditEvent]) -> Dict[str, Any]:
        """Assess compliance status based on audit events"""
        status = {
            "overall_status": "compliant",
            "issues": [],
            "recommendations": []
        }
        
        if report_type.lower() == "soc2":
            # SOC 2 specific checks
            auth_events = [e for e in events if e.event_type == AuditEventType.AUTHENTICATION]
            if not auth_events:
                status["issues"].append("No authentication events found")
                status["overall_status"] = "non_compliant"
            
            admin_events = [e for e in events if e.actor_type == "admin"]
            if len(admin_events) > len(events) * 0.1:  # More than 10% admin actions
                status["issues"].append("High percentage of admin actions detected")
                status["recommendations"].append("Review admin activity for necessity")
        
        elif report_type.lower() == "gdpr":
            # GDPR specific checks
            data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
            data_mod_events = [e for e in events if e.event_type == AuditEventType.DATA_MODIFICATION]
            
            if not data_access_events:
                status["recommendations"].append("Ensure all data access is properly logged")
        
        return status
    
    def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit trail statistics for the last N days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_events = [e for e in self.recent_events if e.timestamp >= cutoff_date]
        
        stats = {
            "total_events": len(recent_events),
            "events_by_type": {},
            "events_by_outcome": {},
            "unique_actors": len(set(e.actor_id for e in recent_events if e.actor_id)),
            "unique_resources": len(set(e.resource for e in recent_events)),
            "average_risk_score": 0,
            "integrity_status": "valid",
            "storage_status": {
                "files_count": len(list(self.storage_path.glob("audit_*.json"))),
                "storage_path": str(self.storage_path)
            }
        }
        
        if recent_events:
            # Calculate breakdowns
            for event in recent_events:
                event_type = event.event_type.value
                stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
                
                outcome = event.outcome.value
                stats["events_by_outcome"][outcome] = stats["events_by_outcome"].get(outcome, 0) + 1
            
            # Calculate average risk score
            stats["average_risk_score"] = sum(e.risk_score for e in recent_events) / len(recent_events)
            
            # Check integrity
            integrity_result = self.verify_chain_integrity(recent_events)
            stats["integrity_status"] = "valid" if integrity_result["valid"] else "invalid"
        
        return stats

class SecurityAuditLogger:
    """
    High-level security audit logger that integrates with the immutable audit trail.
    Provides convenient methods for logging common security events.
    """
    
    def __init__(self, audit_trail: ImmutableAuditTrail):
        self.audit_trail = audit_trail
    
    async def log_authentication_attempt(self, user_id: str, success: bool,
                                       source_ip: str, user_agent: str = None,
                                       session_id: str = None, details: Dict[str, Any] = None):
        """Log authentication attempt"""
        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE
        risk_score = 3 if success else 7
        
        await self.audit_trail.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            outcome=outcome,
            actor_id=user_id,
            actor_type="user",
            resource="authentication_system",
            action="login_attempt",
            description=f"User {'successfully authenticated' if success else 'failed to authenticate'}",
            source_ip=source_ip,
            user_agent=user_agent,
            session_id=session_id,
            details=details or {},
            risk_score=risk_score
        )
    
    async def log_authorization_check(self, user_id: str, resource: str, 
                                    action: str, allowed: bool, 
                                    source_ip: str = None, details: Dict[str, Any] = None):
        """Log authorization check"""
        outcome = AuditOutcome.SUCCESS if allowed else AuditOutcome.DENIED
        risk_score = 4 if allowed else 8
        
        await self.audit_trail.log_event(
            event_type=AuditEventType.AUTHORIZATION,
            outcome=outcome,
            actor_id=user_id,
            actor_type="user",
            resource=resource,
            action=action,
            description=f"Access {'granted' if allowed else 'denied'} to {resource}",
            source_ip=source_ip,
            details=details or {},
            risk_score=risk_score
        )
    
    async def log_data_access(self, user_id: str, resource: str, 
                            operation: str, success: bool,
                            source_ip: str = None, details: Dict[str, Any] = None):
        """Log data access event"""
        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE
        risk_score = 5 if success else 8
        
        await self.audit_trail.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            outcome=outcome,
            actor_id=user_id,
            actor_type="user",
            resource=resource,
            action=operation,
            description=f"Data {operation} operation on {resource}",
            source_ip=source_ip,
            details=details or {},
            risk_score=risk_score
        )
    
    async def log_data_modification(self, user_id: str, resource: str,
                                  operation: str, success: bool,
                                  source_ip: str = None, details: Dict[str, Any] = None):
        """Log data modification event"""
        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE
        risk_score = 6 if success else 9
        
        await self.audit_trail.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            outcome=outcome,
            actor_id=user_id,
            actor_type="user",
            resource=resource,
            action=operation,
            description=f"Data {operation} operation on {resource}",
            source_ip=source_ip,
            details=details or {},
            risk_score=risk_score
        )
    
    async def log_admin_action(self, admin_id: str, action: str, target: str,
                             success: bool, source_ip: str = None,
                             details: Dict[str, Any] = None):
        """Log administrative action"""
        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE
        risk_score = 7 if success else 9
        
        await self.audit_trail.log_event(
            event_type=AuditEventType.ADMIN_ACTION,
            outcome=outcome,
            actor_id=admin_id,
            actor_type="admin",
            resource=target,
            action=action,
            description=f"Administrator performed {action} on {target}",
            source_ip=source_ip,
            details=details or {},
            risk_score=risk_score
        )
    
    async def log_security_event(self, event_description: str, severity: str,
                               user_id: str = None, source_ip: str = None,
                               details: Dict[str, Any] = None):
        """Log security event"""
        severity_map = {
            "low": 4,
            "medium": 6,
            "high": 8,
            "critical": 10
        }
        
        risk_score = severity_map.get(severity.lower(), 6)
        
        await self.audit_trail.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            outcome=AuditOutcome.SUCCESS,  # Security events are informational
            actor_id=user_id,
            actor_type="system" if not user_id else "user",
            resource="security_system",
            action="security_event",
            description=event_description,
            source_ip=source_ip,
            details=details or {},
            risk_score=risk_score
        )

# Global audit trail instance
_global_audit_trail = None
_global_audit_logger = None

def get_audit_trail() -> ImmutableAuditTrail:
    """Get global audit trail instance"""
    global _global_audit_trail
    if _global_audit_trail is None:
        _global_audit_trail = ImmutableAuditTrail()
    return _global_audit_trail

def get_security_audit_logger() -> SecurityAuditLogger:
    """Get global security audit logger instance"""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = SecurityAuditLogger(get_audit_trail())
    return _global_audit_logger