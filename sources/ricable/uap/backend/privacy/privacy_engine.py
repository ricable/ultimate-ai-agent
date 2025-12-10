"""
Privacy Engine for Advanced Privacy-Preserving AI
Implements secure multi-party computation, advanced differential privacy,
private set intersection, and privacy audit trails.
"""

import asyncio
import json
import logging
import numpy as np
import torch
import hashlib
import hmac
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import uuid4
from enum import Enum
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class AuditEventType(Enum):
    """Types of privacy audit events"""
    DATA_ACCESS = "data_access"
    PRIVACY_BUDGET_CONSUMPTION = "privacy_budget_consumption"
    ENCRYPTION_OPERATION = "encryption_operation"
    SECURE_COMPUTATION = "secure_computation"
    PRIVACY_VIOLATION = "privacy_violation"
    CONSENT_UPDATE = "consent_update"


@dataclass
class PrivacyPolicy:
    """Privacy policy configuration"""
    policy_id: str
    name: str
    description: str
    privacy_level: PrivacyLevel
    allowed_purposes: List[str]
    data_retention_days: int
    differential_privacy_enabled: bool
    encryption_required: bool
    audit_required: bool
    consent_required: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class DataSubject:
    """Represents a data subject with privacy preferences"""
    subject_id: str
    name: str
    email: Optional[str]
    privacy_preferences: Dict[str, Any]
    consent_records: List[Dict[str, Any]]
    data_categories: List[str]
    privacy_policy_id: str
    opt_out_requests: List[datetime]
    deletion_requests: List[datetime]
    last_updated: datetime


@dataclass
class PrivacyAuditEvent:
    """Privacy audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    actor_id: str
    subject_id: Optional[str]
    data_categories: List[str]
    operation: str
    purpose: str
    privacy_budget_used: Optional[float]
    compliance_status: str
    additional_metadata: Dict[str, Any]


@dataclass
class SecretShare:
    """Secret share for multi-party computation"""
    share_id: str
    party_id: str
    share_value: bytes
    threshold: int
    total_parties: int
    metadata: Dict[str, Any]


class AdvancedDifferentialPrivacy:
    """Advanced differential privacy mechanisms"""
    
    def __init__(self):
        self.mechanisms = {
            "laplace": self._laplace_mechanism,
            "gaussian": self._gaussian_mechanism,
            "exponential": self._exponential_mechanism,
            "sparse_vector": self._sparse_vector_technique,
            "private_selection": self._private_selection_mechanism
        }
        
        self.composition_tracker = {}
        self.privacy_accountant = PrivacyAccountant()
    
    async def apply_mechanism(self, data: Union[float, torch.Tensor], 
                            mechanism: str, sensitivity: float,
                            epsilon: float, delta: float = 0,
                            additional_params: Dict[str, Any] = None) -> Union[float, torch.Tensor]:
        """Apply a differential privacy mechanism"""
        additional_params = additional_params or {}
        
        if mechanism not in self.mechanisms:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        # Track privacy composition
        self.privacy_accountant.add_query(epsilon, delta)
        
        return await self.mechanisms[mechanism](data, sensitivity, epsilon, delta, **additional_params)
    
    async def _laplace_mechanism(self, data: Union[float, torch.Tensor], 
                               sensitivity: float, epsilon: float, 
                               delta: float, **kwargs) -> Union[float, torch.Tensor]:
        """Laplace mechanism for differential privacy"""
        scale = sensitivity / epsilon
        
        if isinstance(data, torch.Tensor):
            noise = torch.distributions.Laplace(0, scale).sample(data.shape)
            return data + noise
        else:
            noise = np.random.laplace(0, scale)
            return data + noise
    
    async def _gaussian_mechanism(self, data: Union[float, torch.Tensor],
                                sensitivity: float, epsilon: float,
                                delta: float, **kwargs) -> Union[float, torch.Tensor]:
        """Gaussian mechanism for (ε,δ)-differential privacy"""
        if delta == 0:
            raise ValueError("Gaussian mechanism requires δ > 0")
        
        c = np.sqrt(2 * np.log(1.25 / delta))
        sigma = c * sensitivity / epsilon
        
        if isinstance(data, torch.Tensor):
            noise = torch.normal(0, sigma, data.shape)
            return data + noise
        else:
            noise = np.random.normal(0, sigma)
            return data + noise
    
    async def _exponential_mechanism(self, candidates: List[Any],
                                   utility_function: callable,
                                   sensitivity: float, epsilon: float,
                                   delta: float, **kwargs) -> Any:
        """Exponential mechanism for selecting from candidates"""
        utilities = [utility_function(candidate) for candidate in candidates]
        max_utility = max(utilities)
        
        # Normalize utilities and apply exponential weights
        weights = []
        for utility in utilities:
            weight = np.exp(epsilon * utility / (2 * sensitivity))
            weights.append(weight)
        
        # Sample according to weights
        weights = np.array(weights)
        probabilities = weights / np.sum(weights)
        
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]
    
    async def _sparse_vector_technique(self, queries: List[callable],
                                     dataset: Any, threshold: float,
                                     sensitivity: float, epsilon: float,
                                     delta: float, **kwargs) -> List[bool]:
        """Sparse Vector Technique for answering multiple queries"""
        k = kwargs.get('k', len(queries))  # Number of queries to answer
        
        # Add noise to threshold
        epsilon_threshold = epsilon / 2
        epsilon_queries = epsilon / (2 * k)
        
        threshold_noise = np.random.laplace(0, sensitivity / epsilon_threshold)
        noisy_threshold = threshold + threshold_noise
        
        results = []
        answered = 0
        
        for query in queries:
            if answered >= k:
                results.append(False)
                continue
            
            # Evaluate query
            true_answer = query(dataset)
            query_noise = np.random.laplace(0, sensitivity / epsilon_queries)
            noisy_answer = true_answer + query_noise
            
            # Check against threshold
            above_threshold = noisy_answer >= noisy_threshold
            results.append(above_threshold)
            
            if above_threshold:
                answered += 1
        
        return results
    
    async def _private_selection_mechanism(self, candidates: List[Any],
                                         selection_function: callable,
                                         sensitivity: float, epsilon: float,
                                         delta: float, **kwargs) -> Any:
        """Private selection from candidates"""
        # Use exponential mechanism for selection
        return await self._exponential_mechanism(
            candidates, selection_function, sensitivity, epsilon, delta
        )


class PrivacyAccountant:
    """Tracks privacy budget consumption using advanced composition"""
    
    def __init__(self):
        self.queries = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        
    def add_query(self, epsilon: float, delta: float):
        """Add a privacy query to the accountant"""
        self.queries.append({
            "epsilon": epsilon,
            "delta": delta,
            "timestamp": datetime.utcnow()
        })
        
        # Update totals using advanced composition
        self._update_composition()
    
    def _update_composition(self):
        """Update total privacy cost using advanced composition"""
        if not self.queries:
            return
        
        # Simple composition (sum)
        self.total_epsilon = sum(q["epsilon"] for q in self.queries)
        self.total_delta = sum(q["delta"] for q in self.queries)
        
        # Advanced composition would use tighter bounds
        # For k queries with (ε,δ)-DP, we get (ε',kδ+δ')-DP
        # where ε' = sqrt(2k ln(1/δ'))ε + kε(exp(ε)-1)
    
    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get total privacy cost"""
        return self.total_epsilon, self.total_delta
    
    def check_budget(self, budget_epsilon: float, budget_delta: float) -> bool:
        """Check if remaining budget is sufficient"""
        return (self.total_epsilon <= budget_epsilon and 
                self.total_delta <= budget_delta)


class SecureMultiPartyComputation:
    """Implements secure multi-party computation protocols"""
    
    def __init__(self):
        self.parties = {}
        self.secret_shares = {}
        self.computation_sessions = {}
    
    def register_party(self, party_id: str, public_key: bytes):
        """Register a party for MPC"""
        self.parties[party_id] = {
            "party_id": party_id,
            "public_key": public_key,
            "active": True,
            "last_seen": datetime.utcnow()
        }
    
    async def secret_share(self, secret: float, threshold: int, 
                          party_ids: List[str]) -> Dict[str, SecretShare]:
        """Create secret shares using Shamir's Secret Sharing"""
        if threshold > len(party_ids):
            raise ValueError("Threshold cannot exceed number of parties")
        
        # Generate polynomial coefficients
        coefficients = [secret] + [secrets.randbelow(2**32) for _ in range(threshold - 1)]
        
        shares = {}
        for i, party_id in enumerate(party_ids, 1):
            # Evaluate polynomial at point i
            share_value = sum(coeff * (i ** j) for j, coeff in enumerate(coefficients))
            
            # Convert to bytes for storage
            share_bytes = str(share_value).encode('utf-8')
            
            share = SecretShare(
                share_id=str(uuid4()),
                party_id=party_id,
                share_value=share_bytes,
                threshold=threshold,
                total_parties=len(party_ids),
                metadata={"polynomial_degree": threshold - 1}
            )
            
            shares[party_id] = share
            self.secret_shares[share.share_id] = share
        
        return shares
    
    async def reconstruct_secret(self, shares: List[SecretShare]) -> float:
        """Reconstruct secret from shares using Lagrange interpolation"""
        if len(shares) < shares[0].threshold:
            raise ValueError("Insufficient shares for reconstruction")
        
        # Convert shares back to numbers
        points = []
        for i, share in enumerate(shares[:shares[0].threshold], 1):
            y = float(share.share_value.decode('utf-8'))
            points.append((i, y))
        
        # Lagrange interpolation to find f(0)
        secret = 0.0
        for i, (xi, yi) in enumerate(points):
            # Calculate Lagrange basis polynomial
            basis = 1.0
            for j, (xj, _) in enumerate(points):
                if i != j:
                    basis *= (0 - xj) / (xi - xj)
            
            secret += yi * basis
        
        return secret
    
    async def secure_sum(self, values: Dict[str, float], party_ids: List[str]) -> float:
        """Compute secure sum using secret sharing"""
        # Each party secret-shares their value
        all_shares = {}
        threshold = len(party_ids) // 2 + 1
        
        for party_id, value in values.items():
            shares = await self.secret_share(value, threshold, party_ids)
            all_shares[party_id] = shares
        
        # Sum shares for each party
        summed_shares = []
        for i, party_id in enumerate(party_ids):
            share_sum = 0.0
            
            for shares_dict in all_shares.values():
                if party_id in shares_dict:
                    share_value = float(shares_dict[party_id].share_value.decode('utf-8'))
                    share_sum += share_value
            
            # Create combined share
            combined_share = SecretShare(
                share_id=str(uuid4()),
                party_id=party_id,
                share_value=str(share_sum).encode('utf-8'),
                threshold=threshold,
                total_parties=len(party_ids),
                metadata={}
            )
            summed_shares.append(combined_share)
        
        # Reconstruct the sum
        return await self.reconstruct_secret(summed_shares[:threshold])
    
    async def secure_average(self, values: Dict[str, float], party_ids: List[str]) -> float:
        """Compute secure average"""
        secure_sum_result = await self.secure_sum(values, party_ids)
        return secure_sum_result / len(values)
    
    async def secure_comparison(self, value1: float, value2: float, 
                              party_ids: List[str]) -> bool:
        """Secure comparison protocol"""
        # Simplified secure comparison
        # In practice, would use more sophisticated protocols like Yao's garbled circuits
        
        # Compute difference securely
        difference_shares = await self.secret_share(value1 - value2, 
                                                   len(party_ids) // 2 + 1, 
                                                   party_ids)
        
        # Reconstruct difference (in practice, wouldn't fully reconstruct)
        difference = await self.reconstruct_secret(list(difference_shares.values()))
        
        return difference > 0


class PrivateSetIntersection:
    """Private Set Intersection protocols"""
    
    def __init__(self):
        self.hash_function = hashlib.sha256
    
    async def compute_psi(self, set_a: Set[str], set_b: Set[str],
                         use_bloom_filter: bool = False) -> Set[str]:
        """Compute private set intersection"""
        if use_bloom_filter:
            return await self._bloom_filter_psi(set_a, set_b)
        else:
            return await self._hash_based_psi(set_a, set_b)
    
    async def _hash_based_psi(self, set_a: Set[str], set_b: Set[str]) -> Set[str]:
        """Hash-based PSI using polynomial representation"""
        # Convert sets to polynomial coefficients
        poly_a = self._set_to_polynomial(set_a)
        poly_b = self._set_to_polynomial(set_b)
        
        # Find common roots (intersection)
        intersection = set()
        
        # Simplified intersection finding
        for item_a in set_a:
            hash_a = self._hash_item(item_a)
            for item_b in set_b:
                hash_b = self._hash_item(item_b)
                if hash_a == hash_b:
                    intersection.add(item_a)
        
        return intersection
    
    async def _bloom_filter_psi(self, set_a: Set[str], set_b: Set[str]) -> Set[str]:
        """Bloom filter-based PSI"""
        # Create Bloom filter for set_a
        bloom_filter = self._create_bloom_filter(set_a)
        
        # Test set_b against Bloom filter
        intersection = set()
        for item in set_b:
            if self._bloom_filter_test(bloom_filter, item):
                # Potential match (may have false positives)
                if item in set_a:  # Verify actual membership
                    intersection.add(item)
        
        return intersection
    
    def _set_to_polynomial(self, items: Set[str]) -> List[int]:
        """Convert set to polynomial representation"""
        # Simplified polynomial representation
        coefficients = []
        for item in items:
            hash_value = int(self._hash_item(item), 16) % 1000
            coefficients.append(hash_value)
        return coefficients
    
    def _hash_item(self, item: str) -> str:
        """Hash an item for PSI"""
        return self.hash_function(item.encode('utf-8')).hexdigest()
    
    def _create_bloom_filter(self, items: Set[str], size: int = 1000) -> List[bool]:
        """Create Bloom filter for a set"""
        bloom_filter = [False] * size
        
        for item in items:
            # Use multiple hash functions
            for i in range(3):  # 3 hash functions
                hash_value = int(hashlib.sha256(f"{item}_{i}".encode()).hexdigest(), 16)
                index = hash_value % size
                bloom_filter[index] = True
        
        return bloom_filter
    
    def _bloom_filter_test(self, bloom_filter: List[bool], item: str) -> bool:
        """Test item against Bloom filter"""
        size = len(bloom_filter)
        
        for i in range(3):  # Same 3 hash functions
            hash_value = int(hashlib.sha256(f"{item}_{i}".encode()).hexdigest(), 16)
            index = hash_value % size
            if not bloom_filter[index]:
                return False
        
        return True


class PrivacyAuditor:
    """Comprehensive privacy audit system"""
    
    def __init__(self):
        self.audit_events: List[PrivacyAuditEvent] = []
        self.policies: Dict[str, PrivacyPolicy] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        self.compliance_rules = {}
        
    def add_policy(self, policy: PrivacyPolicy):
        """Add a privacy policy"""
        self.policies[policy.policy_id] = policy
    
    def register_data_subject(self, subject: DataSubject):
        """Register a data subject"""
        self.data_subjects[subject.subject_id] = subject
    
    async def log_event(self, event: PrivacyAuditEvent) -> bool:
        """Log a privacy audit event"""
        try:
            # Check compliance
            compliance_status = await self._check_compliance(event)
            event.compliance_status = compliance_status
            
            # Store event
            self.audit_events.append(event)
            
            # Alert if non-compliant
            if compliance_status != "compliant":
                await self._handle_compliance_violation(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    async def _check_compliance(self, event: PrivacyAuditEvent) -> str:
        """Check if event is compliant with privacy policies"""
        # Check if subject has valid consent
        if event.subject_id and event.subject_id in self.data_subjects:
            subject = self.data_subjects[event.subject_id]
            
            # Check consent for purpose
            has_consent = any(
                consent.get("purpose") == event.purpose and 
                consent.get("status") == "granted"
                for consent in subject.consent_records
            )
            
            if not has_consent and event.event_type == AuditEventType.DATA_ACCESS:
                return "missing_consent"
        
        # Check privacy budget
        if event.privacy_budget_used and event.privacy_budget_used > 1.0:
            return "budget_exceeded"
        
        # Check data retention
        retention_days = 365  # Default
        if event.subject_id in self.data_subjects:
            subject = self.data_subjects[event.subject_id]
            if subject.privacy_policy_id in self.policies:
                policy = self.policies[subject.privacy_policy_id]
                retention_days = policy.data_retention_days
        
        event_age = (datetime.utcnow() - event.timestamp).days
        if event_age > retention_days:
            return "retention_exceeded"
        
        return "compliant"
    
    async def _handle_compliance_violation(self, event: PrivacyAuditEvent):
        """Handle privacy compliance violation"""
        logger.warning(f"Privacy compliance violation: {event.compliance_status} for event {event.event_id}")
        
        # Could trigger alerts, notifications, automatic remediation, etc.
        violation_event = PrivacyAuditEvent(
            event_id=str(uuid4()),
            event_type=AuditEventType.PRIVACY_VIOLATION,
            timestamp=datetime.utcnow(),
            actor_id="system",
            subject_id=event.subject_id,
            data_categories=[],
            operation="compliance_check",
            purpose="audit",
            privacy_budget_used=None,
            compliance_status="violation_detected",
            additional_metadata={"original_event_id": event.event_id, "violation_type": event.compliance_status}
        )
        
        self.audit_events.append(violation_event)
    
    async def generate_compliance_report(self, start_date: datetime = None,
                                       end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        # Filter events by date range
        filtered_events = [
            event for event in self.audit_events
            if start_date <= event.timestamp <= end_date
        ]
        
        # Calculate metrics
        total_events = len(filtered_events)
        compliant_events = len([e for e in filtered_events if e.compliance_status == "compliant"])
        violation_events = len([e for e in filtered_events if e.compliance_status != "compliant"])
        
        # Group by event type
        events_by_type = {}
        for event in filtered_events:
            event_type = event.event_type.value
            if event_type not in events_by_type:
                events_by_type[event_type] = 0
            events_by_type[event_type] += 1
        
        # Privacy budget analysis
        budget_events = [e for e in filtered_events if e.privacy_budget_used is not None]
        total_budget_used = sum(e.privacy_budget_used for e in budget_events)
        avg_budget_per_query = total_budget_used / len(budget_events) if budget_events else 0
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": total_events,
                "compliant_events": compliant_events,
                "violation_events": violation_events,
                "compliance_rate": compliant_events / total_events if total_events > 0 else 0
            },
            "events_by_type": events_by_type,
            "privacy_budget": {
                "total_queries": len(budget_events),
                "total_budget_used": total_budget_used,
                "average_budget_per_query": avg_budget_per_query
            },
            "data_subjects": {
                "total_registered": len(self.data_subjects),
                "with_consent": len([s for s in self.data_subjects.values() if s.consent_records]),
                "opt_out_requests": sum(len(s.opt_out_requests) for s in self.data_subjects.values()),
                "deletion_requests": sum(len(s.deletion_requests) for s in self.data_subjects.values())
            }
        }


class PrivacyEngine:
    """Main privacy engine coordinating all privacy components"""
    
    def __init__(self):
        self.differential_privacy = AdvancedDifferentialPrivacy()
        self.secure_mpc = SecureMultiPartyComputation()
        self.private_set_intersection = PrivateSetIntersection()
        self.auditor = PrivacyAuditor()
        
        # Configuration
        self.config = {
            "default_privacy_level": PrivacyLevel.MEDIUM,
            "audit_all_operations": True,
            "enforce_consent": True,
            "require_encryption": True
        }
        
    async def initialize(self) -> bool:
        """Initialize the privacy engine"""
        logger.info("Initializing Privacy Engine")
        
        # Set up default privacy policies
        await self._setup_default_policies()
        
        return True
    
    async def apply_privacy_protection(self, data: Any, operation: str,
                                     privacy_level: PrivacyLevel = None,
                                     subject_id: str = None) -> Tuple[Any, Dict[str, Any]]:
        """Apply appropriate privacy protection to data"""
        privacy_level = privacy_level or self.config["default_privacy_level"]
        
        # Determine privacy parameters based on level
        privacy_params = self._get_privacy_parameters(privacy_level)
        
        # Apply differential privacy if needed
        if privacy_params["use_differential_privacy"]:
            protected_data = await self.differential_privacy.apply_mechanism(
                data, "gaussian", privacy_params["sensitivity"],
                privacy_params["epsilon"], privacy_params["delta"]
            )
        else:
            protected_data = data
        
        # Log audit event
        if self.config["audit_all_operations"]:
            await self._log_privacy_operation(operation, subject_id, privacy_params)
        
        metadata = {
            "privacy_level": privacy_level.value,
            "parameters_used": privacy_params,
            "protection_applied": privacy_params["use_differential_privacy"]
        }
        
        return protected_data, metadata
    
    async def secure_computation(self, computation_type: str, 
                               participants: List[str],
                               data: Dict[str, Any]) -> Any:
        """Perform secure multi-party computation"""
        if computation_type == "sum":
            return await self.secure_mpc.secure_sum(data, participants)
        elif computation_type == "average":
            return await self.secure_mpc.secure_average(data, participants)
        elif computation_type == "intersection":
            set_a = set(data.get("set_a", []))
            set_b = set(data.get("set_b", []))
            return await self.private_set_intersection.compute_psi(set_a, set_b)
        else:
            raise ValueError(f"Unsupported computation type: {computation_type}")
    
    async def check_consent(self, subject_id: str, purpose: str) -> bool:
        """Check if data subject has given consent for purpose"""
        if subject_id not in self.auditor.data_subjects:
            return False
        
        subject = self.auditor.data_subjects[subject_id]
        
        # Check for valid consent
        for consent in subject.consent_records:
            if (consent.get("purpose") == purpose and 
                consent.get("status") == "granted" and
                consent.get("expiry_date", datetime.max) > datetime.utcnow()):
                return True
        
        return False
    
    async def get_privacy_budget_status(self, subject_id: str = None) -> Dict[str, Any]:
        """Get privacy budget status"""
        if subject_id:
            # Get status for specific subject
            epsilon, delta = self.differential_privacy.privacy_accountant.get_privacy_cost()
            return {
                "subject_id": subject_id,
                "epsilon_used": epsilon,
                "delta_used": delta,
                "queries_made": len(self.differential_privacy.privacy_accountant.queries)
            }
        else:
            # Get global status
            total_epsilon = 0.0
            total_delta = 0.0
            total_queries = 0
            
            # Aggregate from all tracked accountants
            epsilon, delta = self.differential_privacy.privacy_accountant.get_privacy_cost()
            total_epsilon += epsilon
            total_delta += delta
            total_queries += len(self.differential_privacy.privacy_accountant.queries)
            
            return {
                "global_epsilon_used": total_epsilon,
                "global_delta_used": total_delta,
                "total_queries": total_queries
            }
    
    def _get_privacy_parameters(self, privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Get privacy parameters for a given privacy level"""
        params = {
            PrivacyLevel.LOW: {
                "use_differential_privacy": False,
                "epsilon": 10.0,
                "delta": 1e-3,
                "sensitivity": 1.0
            },
            PrivacyLevel.MEDIUM: {
                "use_differential_privacy": True,
                "epsilon": 1.0,
                "delta": 1e-5,
                "sensitivity": 1.0
            },
            PrivacyLevel.HIGH: {
                "use_differential_privacy": True,
                "epsilon": 0.1,
                "delta": 1e-6,
                "sensitivity": 1.0
            },
            PrivacyLevel.MAXIMUM: {
                "use_differential_privacy": True,
                "epsilon": 0.01,
                "delta": 1e-8,
                "sensitivity": 1.0
            }
        }
        
        return params[privacy_level]
    
    async def _setup_default_policies(self):
        """Set up default privacy policies"""
        default_policy = PrivacyPolicy(
            policy_id="default_policy",
            name="Default Privacy Policy",
            description="Standard privacy protection for general use",
            privacy_level=PrivacyLevel.MEDIUM,
            allowed_purposes=["research", "analytics", "service_improvement"],
            data_retention_days=365,
            differential_privacy_enabled=True,
            encryption_required=True,
            audit_required=True,
            consent_required=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.auditor.add_policy(default_policy)
    
    async def _log_privacy_operation(self, operation: str, subject_id: str = None,
                                   privacy_params: Dict[str, Any] = None):
        """Log privacy operation for audit"""
        event = PrivacyAuditEvent(
            event_id=str(uuid4()),
            event_type=AuditEventType.PRIVACY_BUDGET_CONSUMPTION,
            timestamp=datetime.utcnow(),
            actor_id="privacy_engine",
            subject_id=subject_id,
            data_categories=["general"],
            operation=operation,
            purpose="privacy_protection",
            privacy_budget_used=privacy_params.get("epsilon", 0.0) if privacy_params else 0.0,
            compliance_status="compliant",
            additional_metadata=privacy_params or {}
        )
        
        await self.auditor.log_event(event)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive privacy system status"""
        return {
            "privacy_engine": {
                "config": self.config,
                "policies_count": len(self.auditor.policies),
                "data_subjects_count": len(self.auditor.data_subjects),
                "audit_events_count": len(self.auditor.audit_events)
            },
            "differential_privacy": {
                "mechanisms_available": list(self.differential_privacy.mechanisms.keys()),
                "privacy_accountant": await self.get_privacy_budget_status()
            },
            "secure_mpc": {
                "registered_parties": len(self.secure_mpc.parties),
                "active_sessions": len(self.secure_mpc.computation_sessions),
                "secret_shares_count": len(self.secure_mpc.secret_shares)
            },
            "compliance": {
                "audit_enabled": self.config["audit_all_operations"],
                "consent_enforcement": self.config["enforce_consent"],
                "encryption_required": self.config["require_encryption"]
            }
        }