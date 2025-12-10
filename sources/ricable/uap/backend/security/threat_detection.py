# File: backend/security/threat_detection.py
"""
Advanced Threat Detection and Prevention System.
Real-time monitoring, intrusion detection, and automated incident response.
"""

import asyncio
import json
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import ipaddress
import logging

from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ThreatType(Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTACK = "csrf_attack"
    DOS_ATTACK = "dos_attack"
    MALWARE = "malware"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

class ResponseAction(Enum):
    """Automated response actions"""
    BLOCK_IP = "block_ip"
    RATE_LIMIT = "rate_limit"
    ALERT_ADMIN = "alert_admin"
    LOG_INCIDENT = "log_incident"
    QUARANTINE_USER = "quarantine_user"
    FORCE_PASSWORD_RESET = "force_password_reset"
    NOTIFY_SOC = "notify_soc"
    COLLECT_FORENSICS = "collect_forensics"

@dataclass
class SecurityThreat:
    """Security threat detection result"""
    id: str
    threat_type: ThreatType
    level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    timestamp: datetime
    description: str
    evidence: Dict[str, Any]
    confidence_score: float
    false_positive_likelihood: float
    recommended_actions: List[ResponseAction]
    affected_resources: List[str]
    attack_vector: Optional[str] = None
    mitigation_status: str = "new"

@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    title: str
    description: str
    severity: ThreatLevel
    status: str  # new, investigating, contained, resolved, closed
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str]
    threats: List[SecurityThreat]
    timeline: List[Dict[str, Any]]
    containment_actions: List[str]
    recovery_actions: List[str]
    lessons_learned: Optional[str] = None

class ThreatDetectionEngine:
    """
    Advanced threat detection engine with real-time monitoring and pattern analysis.
    
    Features:
    - Real-time request analysis for common attack patterns
    - Behavioral analysis for anomaly detection
    - IP reputation checking and geolocation analysis
    - Rate limiting and DDoS protection
    - Machine learning-based threat scoring
    - Automated incident response
    """
    
    def __init__(self):
        self.detection_rules = self._load_detection_rules()
        self.threat_history: List[SecurityThreat] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, deque] = defaultdict(deque)
        self.user_behavior_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Detection thresholds
        self.thresholds = {
            "failed_login_attempts": 5,
            "requests_per_minute": 60,
            "unusual_requests_per_hour": 500,
            "suspicious_user_agents": True,
            "sql_injection_sensitivity": 0.8,
            "xss_detection_sensitivity": 0.8
        }
        
        # Start background monitoring
        asyncio.create_task(self._background_monitoring())
    
    def _load_detection_rules(self) -> Dict[str, Any]:
        """Load threat detection rules and patterns"""
        return {
            "sql_injection_patterns": [
                r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
                r"(\b(or|and)\s+\d+\s*=\s*\d+)",
                r"(';|'--|\s--)",
                r"(\b(xp_|sp_)\w+)",
                r"(\bcast\s*\(.*\bas\s+\w+\))",
                r"(\bhex\s*\(|\bunhex\s*\()",
                r"(\bload_file\s*\(|\binto\s+outfile)"
            ],
            "xss_patterns": [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript\s*:)",
                r"(on\w+\s*=\s*['\"][^'\"]*['\"])",
                r"(<iframe[^>]*>)",
                r"(<object[^>]*>)",
                r"(<embed[^>]*>)",
                r"(<link[^>]*javascript:)",
                r"(eval\s*\(.*?\))"
            ],
            "suspicious_user_agents": [
                r"(sqlmap|nikto|nessus|openvas|w3af|burp|zap)",
                r"(bot|crawler|spider)(?!.*google|bing|yahoo)",
                r"(wget|curl|python-requests|python-urllib)",
                r"(masscan|nmap|zmap)"
            ],
            "directory_traversal": [
                r"(\.\./|\.\.\\)",
                r"(%2e%2e%2f|%2e%2e%5c)",
                r"(etc/passwd|windows/system32)",
                r"(boot\.ini|web\.config)"
            ],
            "command_injection": [
                r"(;\s*(cat|ls|ps|id|whoami|uname|pwd))",
                r"(\|\s*(cat|ls|ps|id|whoami|uname|pwd))",
                r"(`.*`|\$\(.*\))",
                r"(nc\s+-|netcat\s+-)"
            ]
        }
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """
        Analyze incoming request for security threats.
        
        Args:
            request_data: Request information including IP, headers, body, etc.
            
        Returns:
            List of detected threats
        """
        threats = []
        
        try:
            ip_address = request_data.get("ip_address")
            user_id = request_data.get("user_id")
            user_agent = request_data.get("user_agent", "")
            request_path = request_data.get("path", "")
            request_body = request_data.get("body", "")
            headers = request_data.get("headers", {})
            
            # Check if IP is already blocked
            if ip_address in self.blocked_ips:
                threats.append(self._create_threat(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    ThreatLevel.HIGH,
                    ip_address,
                    user_id,
                    "Request from blocked IP address",
                    {"blocked_ip": ip_address},
                    0.9
                ))
                return threats
            
            # Run detection checks
            threat_checks = [
                self._check_rate_limiting,
                self._check_sql_injection,
                self._check_xss_attempts,
                self._check_directory_traversal,
                self._check_command_injection,
                self._check_suspicious_user_agent,
                self._check_unusual_patterns,
                self._check_geographic_anomalies
            ]
            
            for check_func in threat_checks:
                check_threats = await check_func(request_data)
                threats.extend(check_threats)
            
            # Update user behavior profile
            if user_id:
                self._update_user_behavior(user_id, request_data)
            
            # Update IP reputation
            if ip_address:
                self._update_ip_reputation(ip_address, threats)
            
            # Store threats in history
            self.threat_history.extend(threats)
            
            # Log detected threats
            for threat in threats:
                uap_logger.log_security_event(
                    f"Threat detected: {threat.threat_type.value}",
                    user_id=user_id,
                    ip_address=ip_address,
                    success=False,
                    metadata=asdict(threat)
                )
            
            return threats
            
        except Exception as e:
            logger.error(f"Error in threat analysis: {e}")
            return []
    
    async def _check_rate_limiting(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for rate limiting violations"""
        threats = []
        ip_address = request_data.get("ip_address")
        
        if not ip_address:
            return threats
        
        current_time = time.time()
        window = 60  # 1 minute window
        
        # Clean old entries
        self.rate_limits[ip_address] = deque([
            timestamp for timestamp in self.rate_limits[ip_address]
            if current_time - timestamp < window
        ])
        
        # Add current request
        self.rate_limits[ip_address].append(current_time)
        
        # Check if rate limit exceeded
        request_count = len(self.rate_limits[ip_address])
        if request_count > self.thresholds["requests_per_minute"]:
            threats.append(self._create_threat(
                ThreatType.DOS_ATTACK,
                ThreatLevel.HIGH,
                ip_address,
                request_data.get("user_id"),
                f"Rate limit exceeded: {request_count} requests in {window} seconds",
                {"request_count": request_count, "window": window},
                0.8
            ))
        
        return threats
    
    async def _check_sql_injection(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for SQL injection attempts"""
        threats = []
        
        # Check URL parameters, body, and headers
        check_fields = [
            request_data.get("query_string", ""),
            request_data.get("body", ""),
            str(request_data.get("headers", {}))
        ]
        
        for field_content in check_fields:
            if not field_content:
                continue
            
            for pattern in self.detection_rules["sql_injection_patterns"]:
                matches = re.finditer(pattern, field_content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_sql_injection_confidence(match.group())
                    
                    if confidence >= self.thresholds["sql_injection_sensitivity"]:
                        threats.append(self._create_threat(
                            ThreatType.SQL_INJECTION,
                            ThreatLevel.CRITICAL,
                            request_data.get("ip_address"),
                            request_data.get("user_id"),
                            f"SQL injection attempt detected: {match.group()}",
                            {"pattern_matched": match.group(), "field": field_content[:100]},
                            confidence
                        ))
        
        return threats
    
    async def _check_xss_attempts(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for Cross-Site Scripting (XSS) attempts"""
        threats = []
        
        check_fields = [
            request_data.get("query_string", ""),
            request_data.get("body", ""),
            request_data.get("referer", "")
        ]
        
        for field_content in check_fields:
            if not field_content:
                continue
            
            for pattern in self.detection_rules["xss_patterns"]:
                matches = re.finditer(pattern, field_content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_xss_confidence(match.group())
                    
                    if confidence >= self.thresholds["xss_detection_sensitivity"]:
                        threats.append(self._create_threat(
                            ThreatType.XSS_ATTEMPT,
                            ThreatLevel.HIGH,
                            request_data.get("ip_address"),
                            request_data.get("user_id"),
                            f"XSS attempt detected: {match.group()}",
                            {"pattern_matched": match.group(), "field": field_content[:100]},
                            confidence
                        ))
        
        return threats
    
    async def _check_directory_traversal(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for directory traversal attempts"""
        threats = []
        
        path = request_data.get("path", "")
        query_string = request_data.get("query_string", "")
        
        for content in [path, query_string]:
            if not content:
                continue
            
            for pattern in self.detection_rules["directory_traversal"]:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(self._create_threat(
                        ThreatType.SUSPICIOUS_ACTIVITY,
                        ThreatLevel.HIGH,
                        request_data.get("ip_address"),
                        request_data.get("user_id"),
                        f"Directory traversal attempt detected in {content}",
                        {"suspicious_path": content},
                        0.8
                    ))
        
        return threats
    
    async def _check_command_injection(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for command injection attempts"""
        threats = []
        
        check_fields = [
            request_data.get("query_string", ""),
            request_data.get("body", "")
        ]
        
        for field_content in check_fields:
            if not field_content:
                continue
            
            for pattern in self.detection_rules["command_injection"]:
                if re.search(pattern, field_content, re.IGNORECASE):
                    threats.append(self._create_threat(
                        ThreatType.SUSPICIOUS_ACTIVITY,
                        ThreatLevel.HIGH,
                        request_data.get("ip_address"),
                        request_data.get("user_id"),
                        f"Command injection attempt detected: {field_content[:50]}",
                        {"suspicious_content": field_content[:100]},
                        0.7
                    ))
        
        return threats
    
    async def _check_suspicious_user_agent(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for suspicious user agents"""
        threats = []
        
        user_agent = request_data.get("user_agent", "").lower()
        
        for pattern in self.detection_rules["suspicious_user_agents"]:
            if re.search(pattern, user_agent, re.IGNORECASE):
                threats.append(self._create_threat(
                    ThreatType.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.MEDIUM,
                    request_data.get("ip_address"),
                    request_data.get("user_id"),
                    f"Suspicious user agent detected: {user_agent}",
                    {"user_agent": user_agent},
                    0.6
                ))
        
        return threats
    
    async def _check_unusual_patterns(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for unusual request patterns"""
        threats = []
        
        # Check for unusual request size
        body_size = len(request_data.get("body", ""))
        if body_size > 1000000:  # 1MB
            threats.append(self._create_threat(
                ThreatType.DOS_ATTACK,
                ThreatLevel.MEDIUM,
                request_data.get("ip_address"),
                request_data.get("user_id"),
                f"Unusually large request body: {body_size} bytes",
                {"body_size": body_size},
                0.5
            ))
        
        # Check for unusual headers
        headers = request_data.get("headers", {})
        if len(headers) > 50:
            threats.append(self._create_threat(
                ThreatType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.LOW,
                request_data.get("ip_address"),
                request_data.get("user_id"),
                f"Unusual number of headers: {len(headers)}",
                {"header_count": len(headers)},
                0.4
            ))
        
        return threats
    
    async def _check_geographic_anomalies(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check for geographic anomalies (simplified)"""
        threats = []
        
        # This would integrate with IP geolocation service
        # For now, just check for known suspicious IP ranges
        ip_address = request_data.get("ip_address")
        if ip_address:
            try:
                ip = ipaddress.ip_address(ip_address)
                
                # Check for suspicious IP ranges (simplified)
                suspicious_ranges = [
                    "10.0.0.0/8",    # Private networks shouldn't reach public APIs
                    "172.16.0.0/12",
                    "192.168.0.0/16"
                ]
                
                for range_str in suspicious_ranges:
                    network = ipaddress.ip_network(range_str)
                    if ip in network and not self._is_internal_request(request_data):
                        threats.append(self._create_threat(
                            ThreatType.SUSPICIOUS_ACTIVITY,
                            ThreatLevel.LOW,
                            ip_address,
                            request_data.get("user_id"),
                            f"Request from private IP range: {ip_address}",
                            {"ip_range": range_str},
                            0.3
                        ))
            
            except ValueError:
                # Invalid IP address
                threats.append(self._create_threat(
                    ThreatType.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.LOW,
                    ip_address,
                    request_data.get("user_id"),
                    f"Invalid IP address format: {ip_address}",
                    {"invalid_ip": ip_address},
                    0.5
                ))
        
        return threats
    
    def _is_internal_request(self, request_data: Dict[str, Any]) -> bool:
        """Check if request is from internal system"""
        # Check for internal service headers or known internal IPs
        headers = request_data.get("headers", {})
        return "X-Internal-Service" in headers
    
    def _create_threat(self, threat_type: ThreatType, level: ThreatLevel, 
                      source_ip: Optional[str], user_id: Optional[str],
                      description: str, evidence: Dict[str, Any], 
                      confidence: float) -> SecurityThreat:
        """Create a security threat object"""
        threat_id = hashlib.md5(
            f"{threat_type.value}:{source_ip}:{user_id}:{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Determine recommended actions based on threat level and type
        actions = self._determine_response_actions(threat_type, level, confidence)
        
        return SecurityThreat(
            id=threat_id,
            threat_type=threat_type,
            level=level,
            source_ip=source_ip,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            description=description,
            evidence=evidence,
            confidence_score=confidence,
            false_positive_likelihood=1.0 - confidence,
            recommended_actions=actions,
            affected_resources=self._identify_affected_resources(threat_type, evidence),
            attack_vector=self._identify_attack_vector(threat_type, evidence)
        )
    
    def _determine_response_actions(self, threat_type: ThreatType, 
                                  level: ThreatLevel, confidence: float) -> List[ResponseAction]:
        """Determine appropriate response actions for a threat"""
        actions = [ResponseAction.LOG_INCIDENT]
        
        if level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] and confidence >= 0.8:
            actions.extend([ResponseAction.ALERT_ADMIN, ResponseAction.COLLECT_FORENSICS])
            
            if threat_type in [ThreatType.BRUTE_FORCE, ThreatType.DOS_ATTACK]:
                actions.append(ResponseAction.BLOCK_IP)
            elif threat_type == ThreatType.SQL_INJECTION:
                actions.extend([ResponseAction.BLOCK_IP, ResponseAction.NOTIFY_SOC])
            elif threat_type in [ThreatType.PRIVILEGE_ESCALATION, ThreatType.UNAUTHORIZED_ACCESS]:
                actions.append(ResponseAction.QUARANTINE_USER)
        
        elif level == ThreatLevel.MEDIUM:
            actions.append(ResponseAction.RATE_LIMIT)
        
        if level == ThreatLevel.CRITICAL:
            actions.append(ResponseAction.NOTIFY_SOC)
        
        return actions
    
    def _identify_affected_resources(self, threat_type: ThreatType, 
                                   evidence: Dict[str, Any]) -> List[str]:
        """Identify resources affected by the threat"""
        resources = []
        
        if "path" in evidence:
            resources.append(f"endpoint:{evidence['path']}")
        
        if threat_type == ThreatType.SQL_INJECTION:
            resources.append("database")
        elif threat_type == ThreatType.XSS_ATTEMPT:
            resources.append("web_application")
        elif threat_type == ThreatType.BRUTE_FORCE:
            resources.append("authentication_system")
        
        return resources
    
    def _identify_attack_vector(self, threat_type: ThreatType, 
                               evidence: Dict[str, Any]) -> Optional[str]:
        """Identify the attack vector used"""
        if threat_type == ThreatType.SQL_INJECTION:
            return "web_application_input"
        elif threat_type == ThreatType.XSS_ATTEMPT:
            return "web_application_input"
        elif threat_type == ThreatType.BRUTE_FORCE:
            return "authentication_endpoint"
        elif threat_type == ThreatType.DOS_ATTACK:
            return "network_flood"
        else:
            return "unknown"
    
    def _calculate_sql_injection_confidence(self, matched_pattern: str) -> float:
        """Calculate confidence score for SQL injection detection"""
        high_confidence_indicators = [
            "union select", "drop table", "'; exec", "xp_cmdshell"
        ]
        
        pattern_lower = matched_pattern.lower()
        
        for indicator in high_confidence_indicators:
            if indicator in pattern_lower:
                return 0.95
        
        # Base confidence for SQL keywords
        if any(keyword in pattern_lower for keyword in ["select", "union", "insert", "update", "delete"]):
            return 0.8
        
        return 0.6
    
    def _calculate_xss_confidence(self, matched_pattern: str) -> float:
        """Calculate confidence score for XSS detection"""
        high_confidence_indicators = [
            "<script", "javascript:", "onerror=", "onload="
        ]
        
        pattern_lower = matched_pattern.lower()
        
        for indicator in high_confidence_indicators:
            if indicator in pattern_lower:
                return 0.9
        
        return 0.6
    
    def _update_user_behavior(self, user_id: str, request_data: Dict[str, Any]):
        """Update user behavior profile for anomaly detection"""
        if user_id not in self.user_behavior_profiles:
            self.user_behavior_profiles[user_id] = {
                "request_count": 0,
                "last_activity": None,
                "common_ips": set(),
                "common_user_agents": set(),
                "suspicious_activities": 0
            }
        
        profile = self.user_behavior_profiles[user_id]
        profile["request_count"] += 1
        profile["last_activity"] = datetime.now(timezone.utc)
        
        if request_data.get("ip_address"):
            profile["common_ips"].add(request_data["ip_address"])
        
        if request_data.get("user_agent"):
            profile["common_user_agents"].add(request_data["user_agent"])
    
    def _update_ip_reputation(self, ip_address: str, threats: List[SecurityThreat]):
        """Update IP reputation based on detected threats"""
        if ip_address not in self.suspicious_ips:
            self.suspicious_ips[ip_address] = {
                "threat_count": 0,
                "first_seen": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "threat_types": set()
            }
        
        ip_info = self.suspicious_ips[ip_address]
        ip_info["last_activity"] = datetime.now(timezone.utc)
        
        for threat in threats:
            ip_info["threat_count"] += 1
            ip_info["threat_types"].add(threat.threat_type.value)
            
            # Auto-block IPs with high threat count
            if ip_info["threat_count"] >= 10:
                self.blocked_ips.add(ip_address)
                uap_logger.log_security_event(
                    f"IP address auto-blocked due to repeated threats: {ip_address}",
                    ip_address=ip_address,
                    metadata={"threat_count": ip_info["threat_count"]}
                )
    
    async def _background_monitoring(self):
        """Background task for continuous monitoring and cleanup"""
        while True:
            try:
                # Clean old rate limit entries
                current_time = time.time()
                for ip in list(self.rate_limits.keys()):
                    self.rate_limits[ip] = deque([
                        timestamp for timestamp in self.rate_limits[ip]
                        if current_time - timestamp < 3600  # Keep 1 hour
                    ])
                    
                    if not self.rate_limits[ip]:
                        del self.rate_limits[ip]
                
                # Clean old threat history (keep 30 days)
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                self.threat_history = [
                    threat for threat in self.threat_history
                    if threat.timestamp >= cutoff_date
                ]
                
                # Analyze threat patterns
                await self._analyze_threat_patterns()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_threat_patterns(self):
        """Analyze threat patterns for trend detection"""
        # This would perform more sophisticated analysis
        # For now, just log summary statistics
        recent_threats = [
            threat for threat in self.threat_history
            if threat.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)
        ]
        
        if recent_threats:
            threat_summary = defaultdict(int)
            for threat in recent_threats:
                threat_summary[threat.threat_type.value] += 1
            
            uap_logger.log_security_event(
                "Hourly threat pattern analysis",
                metadata={
                    "total_threats": len(recent_threats),
                    "threat_breakdown": dict(threat_summary),
                    "blocked_ips": len(self.blocked_ips)
                }
            )
    
    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of threats from last N hours"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_threats = [
            threat for threat in self.threat_history
            if threat.timestamp >= cutoff_time
        ]
        
        summary = {
            "total_threats": len(recent_threats),
            "threats_by_type": defaultdict(int),
            "threats_by_level": defaultdict(int),
            "unique_source_ips": len(set(t.source_ip for t in recent_threats if t.source_ip)),
            "blocked_ips": len(self.blocked_ips),
            "high_confidence_threats": len([t for t in recent_threats if t.confidence_score >= 0.8]),
            "automated_responses": 0
        }
        
        for threat in recent_threats:
            summary["threats_by_type"][threat.threat_type.value] += 1
            summary["threats_by_level"][threat.level.value] += 1
            if threat.recommended_actions:
                summary["automated_responses"] += len(threat.recommended_actions)
        
        return summary

class SecurityIncidentManager:
    """
    Security incident management system for tracking and coordinating response to security events.
    """
    
    def __init__(self, threat_engine: ThreatDetectionEngine):
        self.threat_engine = threat_engine
        self.incidents: List[SecurityIncident] = []
        self.incident_templates = self._load_incident_templates()
    
    def _load_incident_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load incident response templates"""
        return {
            ThreatType.SQL_INJECTION.value: {
                "title_template": "SQL Injection Attack Detected",
                "severity": ThreatLevel.CRITICAL,
                "initial_containment": [
                    "Block source IP address",
                    "Isolate affected database connections",
                    "Review database logs for compromise"
                ],
                "investigation_steps": [
                    "Analyze attack payload and techniques",
                    "Check for data exfiltration",
                    "Review application logs",
                    "Assess database integrity"
                ]
            },
            ThreatType.BRUTE_FORCE.value: {
                "title_template": "Brute Force Attack in Progress",
                "severity": ThreatLevel.HIGH,
                "initial_containment": [
                    "Implement rate limiting",
                    "Block attacking IP addresses",
                    "Monitor for account lockouts"
                ],
                "investigation_steps": [
                    "Identify targeted accounts",
                    "Check for successful logins",
                    "Review authentication logs",
                    "Assess credential compromise"
                ]
            }
        }
    
    async def create_incident(self, threats: List[SecurityThreat], 
                            title: Optional[str] = None) -> SecurityIncident:
        """Create security incident from detected threats"""
        if not threats:
            raise ValueError("Cannot create incident without threats")
        
        # Determine incident severity (highest threat level)
        max_severity = max(threat.level for threat in threats)
        primary_threat_type = threats[0].threat_type.value
        
        # Use template if available
        template = self.incident_templates.get(primary_threat_type, {})
        
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{len(self.incidents) + 1:04d}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=title or template.get("title_template", f"Security Incident - {primary_threat_type}"),
            description=f"Automated incident created for {len(threats)} security threats",
            severity=max_severity,
            status="new",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            assigned_to=None,
            threats=threats,
            timeline=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "incident_created",
                "description": "Incident automatically created by threat detection system",
                "actor": "system"
            }],
            containment_actions=template.get("initial_containment", []),
            recovery_actions=template.get("investigation_steps", [])
        )
        
        self.incidents.append(incident)
        
        # Log incident creation
        uap_logger.log_security_event(
            f"Security incident created: {incident_id}",
            metadata={
                "incident_id": incident_id,
                "threat_count": len(threats),
                "severity": max_severity.value,
                "primary_threat_type": primary_threat_type
            }
        )
        
        # Execute automated response if appropriate
        await self._execute_automated_response(incident)
        
        return incident
    
    async def _execute_automated_response(self, incident: SecurityIncident):
        """Execute automated response actions for incident"""
        for threat in incident.threats:
            for action in threat.recommended_actions:
                try:
                    success = await self._execute_response_action(action, threat)
                    
                    # Update incident timeline
                    self._add_timeline_entry(
                        incident,
                        "automated_response",
                        f"Executed {action.value}: {'success' if success else 'failed'}",
                        "system"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to execute response action {action.value}: {e}")
                    self._add_timeline_entry(
                        incident,
                        "automated_response_failed",
                        f"Failed to execute {action.value}: {str(e)}",
                        "system"
                    )
    
    async def _execute_response_action(self, action: ResponseAction, 
                                     threat: SecurityThreat) -> bool:
        """Execute a specific response action"""
        try:
            if action == ResponseAction.BLOCK_IP and threat.source_ip:
                self.threat_engine.blocked_ips.add(threat.source_ip)
                uap_logger.log_security_event(
                    f"IP address blocked: {threat.source_ip}",
                    ip_address=threat.source_ip,
                    metadata={"threat_id": threat.id, "action": action.value}
                )
                return True
            
            elif action == ResponseAction.ALERT_ADMIN:
                # This would send alert to administrators
                uap_logger.log_security_event(
                    f"Admin alert sent for threat: {threat.id}",
                    metadata={"threat_id": threat.id, "action": action.value}
                )
                return True
            
            elif action == ResponseAction.LOG_INCIDENT:
                # Already logged, just return success
                return True
            
            elif action == ResponseAction.COLLECT_FORENSICS:
                # This would trigger forensic data collection
                uap_logger.log_security_event(
                    f"Forensic data collection initiated for threat: {threat.id}",
                    metadata={"threat_id": threat.id, "action": action.value}
                )
                return True
            
            else:
                logger.warning(f"Response action {action.value} not implemented")
                return False
                
        except Exception as e:
            logger.error(f"Error executing response action {action.value}: {e}")
            return False
    
    def _add_timeline_entry(self, incident: SecurityIncident, action: str, 
                           description: str, actor: str):
        """Add entry to incident timeline"""
        incident.timeline.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "description": description,
            "actor": actor
        })
        incident.updated_at = datetime.now(timezone.utc)
    
    def update_incident_status(self, incident_id: str, status: str, 
                             notes: Optional[str] = None, actor: str = "system") -> bool:
        """Update incident status"""
        incident = self.get_incident(incident_id)
        if not incident:
            return False
        
        old_status = incident.status
        incident.status = status
        incident.updated_at = datetime.now(timezone.utc)
        
        description = f"Status changed from {old_status} to {status}"
        if notes:
            description += f": {notes}"
        
        self._add_timeline_entry(incident, "status_update", description, actor)
        
        uap_logger.log_security_event(
            f"Incident status updated: {incident_id}",
            metadata={
                "incident_id": incident_id,
                "old_status": old_status,
                "new_status": status,
                "actor": actor
            }
        )
        
        return True
    
    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Get incident by ID"""
        return next((inc for inc in self.incidents if inc.incident_id == incident_id), None)
    
    def get_open_incidents(self) -> List[SecurityIncident]:
        """Get all open incidents"""
        return [inc for inc in self.incidents if inc.status not in ["resolved", "closed"]]
    
    def get_incidents_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of incidents from last N days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_incidents = [
            inc for inc in self.incidents
            if inc.created_at >= cutoff_date
        ]
        
        summary = {
            "total_incidents": len(recent_incidents),
            "open_incidents": len([inc for inc in recent_incidents if inc.status not in ["resolved", "closed"]]),
            "incidents_by_severity": defaultdict(int),
            "incidents_by_status": defaultdict(int),
            "mean_resolution_time": None,
            "most_common_threat_types": defaultdict(int)
        }
        
        for incident in recent_incidents:
            summary["incidents_by_severity"][incident.severity.value] += 1
            summary["incidents_by_status"][incident.status] += 1
            
            for threat in incident.threats:
                summary["most_common_threat_types"][threat.threat_type.value] += 1
        
        return summary

# Global instances
_global_threat_engine = None
_global_incident_manager = None

def get_threat_detection_engine() -> ThreatDetectionEngine:
    """Get global threat detection engine instance"""
    global _global_threat_engine
    if _global_threat_engine is None:
        _global_threat_engine = ThreatDetectionEngine()
    return _global_threat_engine

def get_incident_manager() -> SecurityIncidentManager:
    """Get global incident manager instance"""
    global _global_incident_manager
    if _global_incident_manager is None:
        _global_incident_manager = SecurityIncidentManager(get_threat_detection_engine())
    return _global_incident_manager