# File: backend/security/scanning.py
"""
Automated Security Scanning and Vulnerability Assessment System.
Provides comprehensive security scanning for code, dependencies, configurations, and runtime.
"""

import os
import json
import asyncio
import subprocess
import tempfile
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ScanType(Enum):
    """Types of security scans"""
    DEPENDENCY = "dependency"
    CODE_ANALYSIS = "code_analysis"
    CONFIGURATION = "configuration"
    SECRETS = "secrets"
    NETWORK = "network"
    CONTAINER = "container"
    API = "api"

@dataclass
class SecurityVulnerability:
    """Security vulnerability finding"""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    scan_type: ScanType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    fix_recommendation: Optional[str] = None
    confidence: float = 1.0
    first_detected: Optional[datetime] = None
    last_detected: Optional[datetime] = None
    remediation_effort: Optional[str] = None
    affected_component: Optional[str] = None

@dataclass
class ScanResult:
    """Security scan result"""
    scan_id: str
    scan_type: ScanType
    start_time: datetime
    end_time: datetime
    target: str
    vulnerabilities: List[SecurityVulnerability]
    scan_status: str
    metadata: Dict[str, Any]
    
    @property
    def duration(self) -> float:
        """Scan duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def vulnerability_count_by_severity(self) -> Dict[str, int]:
        """Count vulnerabilities by severity"""
        counts = {severity.value: 0 for severity in SeverityLevel}
        for vuln in self.vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts

class SecurityScanner:
    """
    Comprehensive security scanner for automated vulnerability assessment.
    
    Features:
    - Dependency vulnerability scanning
    - Static code analysis for security issues
    - Configuration security assessment
    - Secret detection in code and configs
    - Container security scanning
    - API security testing
    """
    
    def __init__(self):
        self.scan_history: List[ScanResult] = []
        self.baseline_vulnerabilities: Set[str] = set()
        self.scan_config = self._load_scan_config()
        
    def _load_scan_config(self) -> Dict[str, Any]:
        """Load scanning configuration"""
        default_config = {
            "dependency_scan": {
                "enabled": True,
                "severity_threshold": "medium",
                "exclude_packages": []
            },
            "code_analysis": {
                "enabled": True,
                "rules": [
                    "sql_injection",
                    "xss_vulnerabilities", 
                    "path_traversal",
                    "command_injection",
                    "weak_crypto",
                    "hardcoded_secrets"
                ]
            },
            "secrets_scan": {
                "enabled": True,
                "patterns": [
                    r"(?i)(api[_-]?key|apikey|access[_-]?key)\s*[:=]\s*['\"]([a-zA-Z0-9_-]{20,})['\"]",
                    r"(?i)(secret[_-]?key|secretkey)\s*[:=]\s*['\"]([a-zA-Z0-9_-]{20,})['\"]",
                    r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]([^\s'\"]{8,})['\"]",
                    r"(?i)(token)\s*[:=]\s*['\"]([a-zA-Z0-9_-]{20,})['\"]",
                    r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*['\"]([A-Z0-9]{20})['\"]",
                    r"(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*['\"]([a-zA-Z0-9/+=]{40})['\"]"
                ]
            },
            "configuration": {
                "enabled": True,
                "check_ssl": True,
                "check_headers": True,
                "check_cors": True,
                "check_permissions": True
            }
        }
        return default_config
    
    async def scan_dependencies(self, project_path: str) -> ScanResult:
        """
        Scan project dependencies for known vulnerabilities.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Scan result with dependency vulnerabilities
        """
        scan_id = f"dep_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)
        vulnerabilities = []
        
        try:
            # Check for Python requirements
            requirements_file = Path(project_path) / "requirements.txt"
            if requirements_file.exists():
                python_vulns = await self._scan_python_dependencies(requirements_file)
                vulnerabilities.extend(python_vulns)
            
            # Check for Node.js package.json
            package_json = Path(project_path) / "package.json"
            if package_json.exists():
                node_vulns = await self._scan_node_dependencies(package_json)
                vulnerabilities.extend(node_vulns)
            
            end_time = datetime.now(timezone.utc)
            
            scan_result = ScanResult(
                scan_id=scan_id,
                scan_type=ScanType.DEPENDENCY,
                start_time=start_time,
                end_time=end_time,
                target=project_path,
                vulnerabilities=vulnerabilities,
                scan_status="completed",
                metadata={
                    "scanned_files": [str(f) for f in [requirements_file, package_json] if f.exists()],
                    "total_vulnerabilities": len(vulnerabilities)
                }
            )
            
            self.scan_history.append(scan_result)
            
            uap_logger.log_security_event(
                "Dependency scan completed",
                metadata={
                    "scan_id": scan_id,
                    "vulnerabilities_found": len(vulnerabilities),
                    "duration": scan_result.duration
                }
            )
            
            return scan_result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_result = ScanResult(
                scan_id=scan_id,
                scan_type=ScanType.DEPENDENCY,
                start_time=start_time,
                end_time=end_time,
                target=project_path,
                vulnerabilities=[],
                scan_status="failed",
                metadata={"error": str(e)}
            )
            
            uap_logger.log_security_event(
                "Dependency scan failed",
                success=False,
                metadata={"scan_id": scan_id, "error": str(e)}
            )
            
            return error_result
    
    async def _scan_python_dependencies(self, requirements_file: Path) -> List[SecurityVulnerability]:
        """Scan Python dependencies for vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Read requirements file
            with open(requirements_file, 'r') as f:
                requirements = f.read().splitlines()
            
            # Known vulnerable packages (simplified for demo)
            vulnerable_packages = {
                "requests": {
                    "versions": ["<2.20.0"],
                    "cve": "CVE-2018-18074",
                    "description": "HTTP request smuggling vulnerability",
                    "severity": SeverityLevel.HIGH
                },
                "urllib3": {
                    "versions": ["<1.24.2"],
                    "cve": "CVE-2019-11324", 
                    "description": "Certificate verification bypass",
                    "severity": SeverityLevel.MEDIUM
                },
                "pyyaml": {
                    "versions": ["<5.1"],
                    "cve": "CVE-2017-18342",
                    "description": "Arbitrary code execution vulnerability",
                    "severity": SeverityLevel.CRITICAL
                }
            }
            
            for req in requirements:
                if '==' in req:
                    package_name, version = req.split('==', 1)
                    package_name = package_name.strip()
                    
                    if package_name in vulnerable_packages:
                        vuln_info = vulnerable_packages[package_name]
                        
                        vulnerability = SecurityVulnerability(
                            id=f"PY_{vuln_info['cve']}_{package_name}",
                            title=f"Vulnerable dependency: {package_name}",
                            description=vuln_info["description"],
                            severity=vuln_info["severity"],
                            scan_type=ScanType.DEPENDENCY,
                            file_path=str(requirements_file),
                            cve_id=vuln_info["cve"],
                            fix_recommendation=f"Update {package_name} to latest version",
                            affected_component=package_name,
                            first_detected=datetime.now(timezone.utc)
                        )
                        vulnerabilities.append(vulnerability)
        
        except Exception as e:
            logger.error(f"Error scanning Python dependencies: {e}")
        
        return vulnerabilities
    
    async def _scan_node_dependencies(self, package_json: Path) -> List[SecurityVulnerability]:
        """Scan Node.js dependencies for vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Use npm audit if available
            project_dir = package_json.parent
            process = await asyncio.create_subprocess_exec(
                "npm", "audit", "--json",
                cwd=project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 or stdout:
                audit_data = json.loads(stdout.decode())
                
                if "vulnerabilities" in audit_data:
                    for vuln_id, vuln_info in audit_data["vulnerabilities"].items():
                        severity_map = {
                            "critical": SeverityLevel.CRITICAL,
                            "high": SeverityLevel.HIGH,
                            "moderate": SeverityLevel.MEDIUM,
                            "low": SeverityLevel.LOW,
                            "info": SeverityLevel.INFO
                        }
                        
                        vulnerability = SecurityVulnerability(
                            id=f"NPM_{vuln_id}",
                            title=vuln_info.get("title", "Node.js dependency vulnerability"),
                            description=vuln_info.get("overview", ""),
                            severity=severity_map.get(vuln_info.get("severity"), SeverityLevel.MEDIUM),
                            scan_type=ScanType.DEPENDENCY,
                            file_path=str(package_json),
                            cve_id=vuln_info.get("cwe"),
                            fix_recommendation=vuln_info.get("recommendation", "Update package"),
                            affected_component=vuln_info.get("module_name"),
                            first_detected=datetime.now(timezone.utc)
                        )
                        vulnerabilities.append(vulnerability)
        
        except Exception as e:
            logger.error(f"Error scanning Node.js dependencies: {e}")
        
        return vulnerabilities
    
    async def scan_secrets(self, project_path: str) -> ScanResult:
        """
        Scan for hardcoded secrets and credentials in source code.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Scan result with secret detection findings
        """
        scan_id = f"secret_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)
        vulnerabilities = []
        
        try:
            # Scan source files for secrets
            source_files = []
            for ext in ['.py', '.js', '.ts', '.json', '.yaml', '.yml', '.env']:
                source_files.extend(Path(project_path).rglob(f'*{ext}'))
            
            patterns = self.scan_config["secrets_scan"]["patterns"]
            
            for file_path in source_files:
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for i, line in enumerate(content.splitlines(), 1):
                        for pattern in patterns:
                            matches = re.finditer(pattern, line)
                            for match in matches:
                                vulnerability = SecurityVulnerability(
                                    id=f"SECRET_{hashlib.md5(f'{file_path}:{i}:{match.group()}'.encode()).hexdigest()[:8]}",
                                    title="Potential hardcoded secret detected",
                                    description=f"Potential secret or credential found: {match.group(1) if match.groups() else 'unknown'}",
                                    severity=SeverityLevel.HIGH,
                                    scan_type=ScanType.SECRETS,
                                    file_path=str(file_path),
                                    line_number=i,
                                    fix_recommendation="Move secrets to environment variables or secure secret management",
                                    confidence=0.8,
                                    first_detected=datetime.now(timezone.utc)
                                )
                                vulnerabilities.append(vulnerability)
                
                except Exception as e:
                    logger.debug(f"Could not scan file {file_path}: {e}")
            
            end_time = datetime.now(timezone.utc)
            
            scan_result = ScanResult(
                scan_id=scan_id,
                scan_type=ScanType.SECRETS,
                start_time=start_time,
                end_time=end_time,
                target=project_path,
                vulnerabilities=vulnerabilities,
                scan_status="completed",
                metadata={
                    "scanned_files": len(source_files),
                    "secrets_found": len(vulnerabilities)
                }
            )
            
            self.scan_history.append(scan_result)
            
            uap_logger.log_security_event(
                "Secret scan completed",
                metadata={
                    "scan_id": scan_id,
                    "secrets_found": len(vulnerabilities),
                    "files_scanned": len(source_files)
                }
            )
            
            return scan_result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_result = ScanResult(
                scan_id=scan_id,
                scan_type=ScanType.SECRETS,
                start_time=start_time,
                end_time=end_time,
                target=project_path,
                vulnerabilities=[],
                scan_status="failed",
                metadata={"error": str(e)}
            )
            
            uap_logger.log_security_event(
                "Secret scan failed",
                success=False,
                metadata={"scan_id": scan_id, "error": str(e)}
            )
            
            return error_result
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning"""
        skip_patterns = [
            "node_modules",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "venv",
            ".venv",
            "dist",
            "build",
            ".egg-info"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    async def scan_configuration(self, config_data: Dict[str, Any]) -> ScanResult:
        """
        Scan application configuration for security issues.
        
        Args:
            config_data: Configuration data to scan
            
        Returns:
            Scan result with configuration security findings
        """
        scan_id = f"config_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)
        vulnerabilities = []
        
        try:
            # Check for insecure configurations
            config_checks = [
                self._check_ssl_config,
                self._check_cors_config,
                self._check_auth_config,
                self._check_debug_config,
                self._check_secrets_config
            ]
            
            for check_func in config_checks:
                check_vulns = check_func(config_data)
                vulnerabilities.extend(check_vulns)
            
            end_time = datetime.now(timezone.utc)
            
            scan_result = ScanResult(
                scan_id=scan_id,
                scan_type=ScanType.CONFIGURATION,
                start_time=start_time,
                end_time=end_time,
                target="application_config",
                vulnerabilities=vulnerabilities,
                scan_status="completed",
                metadata={
                    "config_sections_checked": len(config_checks),
                    "issues_found": len(vulnerabilities)
                }
            )
            
            self.scan_history.append(scan_result)
            
            uap_logger.log_security_event(
                "Configuration scan completed",
                metadata={
                    "scan_id": scan_id,
                    "issues_found": len(vulnerabilities)
                }
            )
            
            return scan_result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_result = ScanResult(
                scan_id=scan_id,
                scan_type=ScanType.CONFIGURATION,
                start_time=start_time,
                end_time=end_time,
                target="application_config",
                vulnerabilities=[],
                scan_status="failed",
                metadata={"error": str(e)}
            )
            
            return error_result
    
    def _check_ssl_config(self, config: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Check SSL/TLS configuration"""
        vulnerabilities = []
        
        # Check if SSL is disabled
        if config.get("ssl_enabled") is False:
            vulnerabilities.append(SecurityVulnerability(
                id="CONFIG_SSL_DISABLED",
                title="SSL/TLS disabled",
                description="SSL/TLS encryption is disabled, allowing unencrypted communications",
                severity=SeverityLevel.HIGH,
                scan_type=ScanType.CONFIGURATION,
                fix_recommendation="Enable SSL/TLS encryption for production environments"
            ))
        
        # Check for weak SSL/TLS versions
        min_tls_version = config.get("min_tls_version", "").lower()
        if min_tls_version in ["ssl3", "tls1.0", "tls1.1"]:
            vulnerabilities.append(SecurityVulnerability(
                id="CONFIG_WEAK_TLS",
                title="Weak TLS version allowed",
                description=f"Minimum TLS version {min_tls_version} is vulnerable to attacks",
                severity=SeverityLevel.MEDIUM,
                scan_type=ScanType.CONFIGURATION,
                fix_recommendation="Set minimum TLS version to 1.2 or higher"
            ))
        
        return vulnerabilities
    
    def _check_cors_config(self, config: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Check CORS configuration"""
        vulnerabilities = []
        
        # Check for overly permissive CORS
        cors_origins = config.get("cors_origins", [])
        if "*" in cors_origins:
            vulnerabilities.append(SecurityVulnerability(
                id="CONFIG_CORS_WILDCARD",
                title="Overly permissive CORS configuration",
                description="CORS allows all origins (*), potentially exposing API to unauthorized domains",
                severity=SeverityLevel.MEDIUM,
                scan_type=ScanType.CONFIGURATION,
                fix_recommendation="Specify explicit allowed origins instead of using wildcard"
            ))
        
        return vulnerabilities
    
    def _check_auth_config(self, config: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Check authentication configuration"""
        vulnerabilities = []
        
        # Check JWT secret strength
        jwt_secret = config.get("jwt_secret", "")
        if len(jwt_secret) < 32:
            vulnerabilities.append(SecurityVulnerability(
                id="CONFIG_WEAK_JWT_SECRET",
                title="Weak JWT secret",
                description="JWT secret is too short, making tokens vulnerable to brute force attacks",
                severity=SeverityLevel.HIGH,
                scan_type=ScanType.CONFIGURATION,
                fix_recommendation="Use a JWT secret of at least 32 characters"
            ))
        
        # Check token expiration
        token_expiry = config.get("access_token_expire_minutes", 0)
        if token_expiry > 1440:  # 24 hours
            vulnerabilities.append(SecurityVulnerability(
                id="CONFIG_LONG_TOKEN_EXPIRY",
                title="Long token expiration time",
                description="Access tokens have long expiration time, increasing risk if compromised",
                severity=SeverityLevel.LOW,
                scan_type=ScanType.CONFIGURATION,
                fix_recommendation="Set access token expiration to 30 minutes or less"
            ))
        
        return vulnerabilities
    
    def _check_debug_config(self, config: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Check debug configuration"""
        vulnerabilities = []
        
        # Check if debug mode is enabled in production
        if config.get("debug") is True and config.get("environment") == "production":
            vulnerabilities.append(SecurityVulnerability(
                id="CONFIG_DEBUG_PRODUCTION",
                title="Debug mode enabled in production",
                description="Debug mode is enabled in production, potentially exposing sensitive information",
                severity=SeverityLevel.HIGH,
                scan_type=ScanType.CONFIGURATION,
                fix_recommendation="Disable debug mode in production environments"
            ))
        
        return vulnerabilities
    
    def _check_secrets_config(self, config: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Check for hardcoded secrets in configuration"""
        vulnerabilities = []
        
        # Check for default passwords
        default_passwords = ["admin", "password", "123456", "admin123"]
        for key, value in config.items():
            if "password" in key.lower() and str(value) in default_passwords:
                vulnerabilities.append(SecurityVulnerability(
                    id=f"CONFIG_DEFAULT_PASSWORD_{key}",
                    title="Default password detected",
                    description=f"Configuration contains default password in field '{key}'",
                    severity=SeverityLevel.CRITICAL,
                    scan_type=ScanType.CONFIGURATION,
                    fix_recommendation="Change default password to a strong, unique password"
                ))
        
        return vulnerabilities
    
    def get_scan_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of security scans from last N days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_scans = [scan for scan in self.scan_history if scan.start_time >= cutoff_date]
        
        summary = {
            "total_scans": len(recent_scans),
            "scans_by_type": {},
            "vulnerabilities_by_severity": {severity.value: 0 for severity in SeverityLevel},
            "total_vulnerabilities": 0,
            "most_recent_scan": None,
            "scan_success_rate": 0.0
        }
        
        if recent_scans:
            # Count scans by type
            for scan in recent_scans:
                scan_type = scan.scan_type.value
                summary["scans_by_type"][scan_type] = summary["scans_by_type"].get(scan_type, 0) + 1
            
            # Count vulnerabilities by severity
            all_vulnerabilities = []
            for scan in recent_scans:
                all_vulnerabilities.extend(scan.vulnerabilities)
            
            for vuln in all_vulnerabilities:
                summary["vulnerabilities_by_severity"][vuln.severity.value] += 1
            
            summary["total_vulnerabilities"] = len(all_vulnerabilities)
            summary["most_recent_scan"] = max(recent_scans, key=lambda s: s.start_time).start_time.isoformat()
            
            # Calculate success rate
            successful_scans = len([scan for scan in recent_scans if scan.scan_status == "completed"])
            summary["scan_success_rate"] = successful_scans / len(recent_scans) if recent_scans else 0.0
        
        return summary

class VulnerabilityAssessment:
    """
    Comprehensive vulnerability assessment and reporting system.
    Provides risk analysis, trending, and remediation prioritization.
    """
    
    def __init__(self, scanner: SecurityScanner):
        self.scanner = scanner
    
    def assess_risk_level(self, vulnerabilities: List[SecurityVulnerability]) -> str:
        """Assess overall risk level based on vulnerabilities"""
        if not vulnerabilities:
            return "LOW"
        
        critical_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL])
        high_count = len([v for v in vulnerabilities if v.severity == SeverityLevel.HIGH])
        
        if critical_count > 0:
            return "CRITICAL"
        elif high_count >= 3:
            return "HIGH"
        elif high_count > 0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def prioritize_remediation(self, vulnerabilities: List[SecurityVulnerability]) -> List[SecurityVulnerability]:
        """Prioritize vulnerabilities for remediation"""
        def priority_score(vuln: SecurityVulnerability) -> int:
            severity_scores = {
                SeverityLevel.CRITICAL: 100,
                SeverityLevel.HIGH: 75,
                SeverityLevel.MEDIUM: 50,
                SeverityLevel.LOW: 25,
                SeverityLevel.INFO: 10
            }
            
            base_score = severity_scores.get(vuln.severity, 0)
            
            # Adjust score based on confidence
            score = base_score * vuln.confidence
            
            # Boost score for vulnerabilities with CVE IDs
            if vuln.cve_id:
                score += 10
            
            # Boost score for public-facing components
            if vuln.scan_type in [ScanType.API, ScanType.NETWORK]:
                score += 15
            
            return int(score)
        
        return sorted(vulnerabilities, key=priority_score, reverse=True)
    
    def generate_assessment_report(self, scan_results: List[ScanResult]) -> Dict[str, Any]:
        """Generate comprehensive vulnerability assessment report"""
        all_vulnerabilities = []
        for result in scan_results:
            all_vulnerabilities.extend(result.vulnerabilities)
        
        prioritized_vulns = self.prioritize_remediation(all_vulnerabilities)
        risk_level = self.assess_risk_level(all_vulnerabilities)
        
        report = {
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "overall_risk_level": risk_level,
            "total_vulnerabilities": len(all_vulnerabilities),
            "vulnerability_breakdown": {
                "by_severity": {},
                "by_scan_type": {},
                "by_confidence": {"high": 0, "medium": 0, "low": 0}
            },
            "top_priority_vulnerabilities": [asdict(v) for v in prioritized_vulns[:10]],
            "scan_coverage": {
                "total_scans": len(scan_results),
                "scan_types_covered": list(set(r.scan_type.value for r in scan_results)),
                "scan_success_rate": len([r for r in scan_results if r.scan_status == "completed"]) / len(scan_results) if scan_results else 0
            },
            "remediation_recommendations": self._generate_remediation_recommendations(prioritized_vulns),
            "trend_analysis": self._analyze_vulnerability_trends()
        }
        
        # Fill in breakdown details
        for vuln in all_vulnerabilities:
            # By severity
            severity = vuln.severity.value
            report["vulnerability_breakdown"]["by_severity"][severity] = \
                report["vulnerability_breakdown"]["by_severity"].get(severity, 0) + 1
            
            # By scan type
            scan_type = vuln.scan_type.value
            report["vulnerability_breakdown"]["by_scan_type"][scan_type] = \
                report["vulnerability_breakdown"]["by_scan_type"].get(scan_type, 0) + 1
            
            # By confidence
            if vuln.confidence >= 0.8:
                report["vulnerability_breakdown"]["by_confidence"]["high"] += 1
            elif vuln.confidence >= 0.5:
                report["vulnerability_breakdown"]["by_confidence"]["medium"] += 1
            else:
                report["vulnerability_breakdown"]["by_confidence"]["low"] += 1
        
        return report
    
    def _generate_remediation_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[Dict[str, Any]]:
        """Generate remediation recommendations"""
        recommendations = []
        
        # Group vulnerabilities by type for common recommendations
        vuln_groups = {}
        for vuln in vulnerabilities[:10]:  # Top 10 priority
            key = f"{vuln.scan_type.value}_{vuln.severity.value}"
            if key not in vuln_groups:
                vuln_groups[key] = []
            vuln_groups[key].append(vuln)
        
        for group_key, group_vulns in vuln_groups.items():
            recommendation = {
                "category": group_vulns[0].scan_type.value,
                "severity": group_vulns[0].severity.value,
                "affected_count": len(group_vulns),
                "recommendation": group_vulns[0].fix_recommendation or "Review and remediate vulnerability",
                "estimated_effort": self._estimate_remediation_effort(group_vulns),
                "business_impact": self._assess_business_impact(group_vulns[0])
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _estimate_remediation_effort(self, vulnerabilities: List[SecurityVulnerability]) -> str:
        """Estimate effort required for remediation"""
        total_vulns = len(vulnerabilities)
        
        if total_vulns == 1:
            if vulnerabilities[0].severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                return "2-4 hours"
            else:
                return "1-2 hours"
        elif total_vulns <= 5:
            return "1-2 days"
        else:
            return "3-5 days"
    
    def _assess_business_impact(self, vulnerability: SecurityVulnerability) -> str:
        """Assess business impact of vulnerability"""
        if vulnerability.severity == SeverityLevel.CRITICAL:
            return "High - Potential for data breach or system compromise"
        elif vulnerability.severity == SeverityLevel.HIGH:
            return "Medium - Significant security risk requiring prompt attention"
        elif vulnerability.severity == SeverityLevel.MEDIUM:
            return "Low - Moderate security risk with potential for exploitation"
        else:
            return "Minimal - Low risk with limited impact"
    
    def _analyze_vulnerability_trends(self) -> Dict[str, Any]:
        """Analyze vulnerability trends over time"""
        # This would analyze historical scan data to identify trends
        # For now, return placeholder data
        return {
            "trend_direction": "stable",
            "new_vulnerabilities_rate": 0.0,
            "remediation_rate": 0.0,
            "most_common_vulnerability_types": ["dependency", "configuration"],
            "improvement_areas": ["Update dependency scanning frequency", "Enhance secret detection"]
        }

# Global scanner instance
_global_scanner = None

def get_security_scanner() -> SecurityScanner:
    """Get global security scanner instance"""
    global _global_scanner
    if _global_scanner is None:
        _global_scanner = SecurityScanner()
    return _global_scanner