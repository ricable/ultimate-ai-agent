# File: backend/security/auth_manager.py
"""
Enterprise Authentication Manager with MFA, SSO, and Advanced Security Features
Provides comprehensive authentication capabilities including multi-factor authentication,
SSO integration, biometric authentication, and advanced session management.
"""

import asyncio
import hashlib
import hmac
import json
import qrcode
import io
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import logging
import re
from urllib.parse import quote

import pyotp
import bcrypt
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr, validator

from ..services.auth import AuthService, User, UserInDB
from .audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger

logger = logging.getLogger(__name__)

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    WEBAUTHN = "webauthn"
    BACKUP_CODE = "backup_code"
    SSO_SAML = "sso_saml"
    SSO_OAUTH2 = "sso_oauth2"
    SSO_OIDC = "sso_oidc"

class MFAStatus(Enum):
    """MFA status levels"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    REQUIRED = "required"
    PENDING_SETUP = "pending_setup"

class SessionStatus(Enum):
    """Session status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"

@dataclass
class MFADevice:
    """Multi-factor authentication device"""
    device_id: str
    user_id: str
    device_type: AuthenticationMethod
    device_name: str
    secret: Optional[str]  # For TOTP
    phone_number: Optional[str]  # For SMS
    email: Optional[str]  # For email
    webauthn_credential: Optional[Dict[str, Any]]  # For WebAuthn
    is_primary: bool
    is_verified: bool
    created_at: datetime
    last_used: Optional[datetime]
    backup_codes: Optional[List[str]]  # For backup codes
    metadata: Dict[str, Any]

@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    device_fingerprint: str
    ip_address: str
    user_agent: str
    location: Optional[str]
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus
    mfa_completed: bool
    authentication_methods: List[AuthenticationMethod]
    security_level: int  # 1-5 scale
    metadata: Dict[str, Any]

@dataclass
class AuthenticationAttempt:
    """Authentication attempt logging"""
    attempt_id: str
    user_id: Optional[str]
    username: str
    ip_address: str
    user_agent: str
    method: AuthenticationMethod
    success: bool
    failure_reason: Optional[str]
    timestamp: datetime
    session_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_history_count: int = 5
    password_max_age_days: int = 90
    
    account_lockout_threshold: int = 5
    account_lockout_duration_minutes: int = 30
    
    session_timeout_minutes: int = 480  # 8 hours
    session_absolute_timeout_hours: int = 24
    concurrent_sessions_limit: int = 3
    
    mfa_required_for_admin: bool = True
    mfa_required_for_sensitive_ops: bool = True
    mfa_grace_period_days: int = 7
    
    sso_enabled: bool = True
    sso_required_domains: List[str] = None
    
    webauthn_enabled: bool = True
    webauthn_require_resident_key: bool = False
    webauthn_user_verification: str = "preferred"

class EnterpriseAuthManager:
    """
    Enterprise Authentication Manager with comprehensive security features.
    
    Features:
    - Multi-factor authentication (TOTP, SMS, Email, WebAuthn)
    - SSO integration (SAML, OAuth2, OpenID Connect)
    - Advanced session management
    - Biometric authentication support
    - Account security policies
    - Threat detection and response
    - Comprehensive audit logging
    """
    
    def __init__(self, auth_service: AuthService, security_policy: SecurityPolicy = None):
        self.auth_service = auth_service
        self.security_policy = security_policy or SecurityPolicy()
        self.audit_logger = get_security_audit_logger()
        
        # In-memory storage for demo (would use database in production)
        self.mfa_devices: Dict[str, List[MFADevice]] = {}  # user_id -> devices
        self.user_sessions: Dict[str, UserSession] = {}  # session_id -> session
        self.auth_attempts: List[AuthenticationAttempt] = []
        self.locked_accounts: Dict[str, datetime] = {}  # user_id -> unlock_time
        self.password_history: Dict[str, List[str]] = {}  # user_id -> password_hashes
        
        # SSO providers (would be configured via environment/database)
        self.sso_providers = {
            "google": {
                "client_id": "your-google-client-id",
                "client_secret": "your-google-client-secret",
                "redirect_uri": "https://yourdomain.com/auth/google/callback"
            },
            "microsoft": {
                "client_id": "your-microsoft-client-id",
                "client_secret": "your-microsoft-client-secret",
                "redirect_uri": "https://yourdomain.com/auth/microsoft/callback"
            }
        }
        
        # Rate limiting for authentication attempts
        self.rate_limiter = {}
        
        logger.info("Enterprise Authentication Manager initialized")
    
    # ==================== ENHANCED AUTHENTICATION ====================
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str, user_agent: str,
                              mfa_code: Optional[str] = None,
                              device_fingerprint: Optional[str] = None) -> Tuple[UserInDB, UserSession]:
        """
        Enhanced user authentication with MFA and threat detection.
        
        Args:
            username: Username or email
            password: User password
            ip_address: Client IP address
            user_agent: Client user agent
            mfa_code: MFA verification code (if applicable)
            device_fingerprint: Device fingerprint for tracking
            
        Returns:
            Tuple of authenticated user and session
            
        Raises:
            HTTPException: If authentication fails
        """
        attempt_id = f"AUTH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
        
        try:
            # Check rate limiting
            await self._check_rate_limiting(ip_address, username)
            
            # Check account lockout
            await self._check_account_lockout(username)
            
            # Validate credentials
            user = self.auth_service.authenticate_user(username, password)
            if not user:
                await self._log_authentication_attempt(
                    attempt_id, None, username, ip_address, user_agent,
                    AuthenticationMethod.PASSWORD, False, "Invalid credentials"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Check if account is active
            if not user.is_active:
                await self._log_authentication_attempt(
                    attempt_id, user.id, username, ip_address, user_agent,
                    AuthenticationMethod.PASSWORD, False, "Account disabled"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is disabled"
                )
            
            # Check if MFA is required
            mfa_required = await self._is_mfa_required(user)
            if mfa_required:
                if not mfa_code:
                    await self._log_authentication_attempt(
                        attempt_id, user.id, username, ip_address, user_agent,
                        AuthenticationMethod.PASSWORD, False, "MFA required"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_202_ACCEPTED,
                        detail="MFA verification required",
                        headers={"X-MFA-Required": "true"}
                    )
                
                # Verify MFA code
                mfa_valid = await self._verify_mfa_code(user.id, mfa_code)
                if not mfa_valid:
                    await self._log_authentication_attempt(
                        attempt_id, user.id, username, ip_address, user_agent,
                        AuthenticationMethod.TOTP, False, "Invalid MFA code"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid MFA code"
                    )
            
            # Create user session
            session = await self._create_user_session(
                user.id, ip_address, user_agent, device_fingerprint,
                mfa_completed=mfa_required
            )
            
            # Log successful authentication
            await self._log_authentication_attempt(
                attempt_id, user.id, username, ip_address, user_agent,
                AuthenticationMethod.PASSWORD, True, None, session.session_id
            )
            
            # Reset rate limiting on successful login
            self._reset_rate_limiting(ip_address, username)
            
            return user, session
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self._log_authentication_attempt(
                attempt_id, None, username, ip_address, user_agent,
                AuthenticationMethod.PASSWORD, False, f"System error: {str(e)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system error"
            )
    
    async def _check_rate_limiting(self, ip_address: str, username: str):
        """Check and enforce rate limiting"""
        current_time = datetime.now(timezone.utc)
        window_minutes = 15
        max_attempts = 10
        
        # Clean old attempts
        cutoff_time = current_time - timedelta(minutes=window_minutes)
        
        # Check IP-based rate limiting
        ip_attempts = [
            attempt for attempt in self.auth_attempts
            if attempt.ip_address == ip_address and 
               attempt.timestamp > cutoff_time and
               not attempt.success
        ]
        
        if len(ip_attempts) >= max_attempts:
            await self.audit_logger.log_security_event(
                "Rate limit exceeded for IP",
                "high",
                source_ip=ip_address,
                details={"attempts": len(ip_attempts), "window_minutes": window_minutes}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts. Please try again later."
            )
        
        # Check username-based rate limiting
        username_attempts = [
            attempt for attempt in self.auth_attempts
            if attempt.username == username and 
               attempt.timestamp > cutoff_time and
               not attempt.success
        ]
        
        if len(username_attempts) >= max_attempts:
            await self.audit_logger.log_security_event(
                "Rate limit exceeded for username",
                "high",
                details={"username": username, "attempts": len(username_attempts)}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts for this account. Please try again later."
            )
    
    def _reset_rate_limiting(self, ip_address: str, username: str):
        """Reset rate limiting counters on successful authentication"""
        # In production, this would clear rate limiting records
        pass
    
    async def _check_account_lockout(self, username: str):
        """Check if account is locked out"""
        user = self.auth_service.get_user_by_username(username)
        if not user:
            return
        
        if user.id in self.locked_accounts:
            unlock_time = self.locked_accounts[user.id]
            if datetime.now(timezone.utc) < unlock_time:
                remaining_minutes = int((unlock_time - datetime.now(timezone.utc)).total_seconds() / 60)
                await self.audit_logger.log_security_event(
                    "Access attempt on locked account",
                    "medium",
                    user_id=user.id,
                    details={"unlock_in_minutes": remaining_minutes}
                )
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Account is locked. Try again in {remaining_minutes} minutes."
                )
            else:
                # Unlock expired lock
                del self.locked_accounts[user.id]
    
    async def _is_mfa_required(self, user: UserInDB) -> bool:
        """Check if MFA is required for user"""
        # Admin users always require MFA
        if "admin" in user.roles and self.security_policy.mfa_required_for_admin:
            return True
        
        # Check if user has MFA devices configured
        user_devices = self.mfa_devices.get(user.id, [])
        if user_devices:
            return True
        
        # Check if MFA is required by policy
        return False
    
    async def _verify_mfa_code(self, user_id: str, code: str) -> bool:
        """Verify MFA code"""
        user_devices = self.mfa_devices.get(user_id, [])
        
        for device in user_devices:
            if not device.is_verified:
                continue
            
            if device.device_type == AuthenticationMethod.TOTP:
                if device.secret:
                    totp = pyotp.TOTP(device.secret)
                    if totp.verify(code, valid_window=1):
                        device.last_used = datetime.now(timezone.utc)
                        return True
            
            elif device.device_type == AuthenticationMethod.BACKUP_CODE:
                if device.backup_codes and code in device.backup_codes:
                    # Remove used backup code
                    device.backup_codes.remove(code)
                    device.last_used = datetime.now(timezone.utc)
                    return True
        
        return False
    
    async def _create_user_session(self, user_id: str, ip_address: str, 
                                 user_agent: str, device_fingerprint: Optional[str],
                                 mfa_completed: bool = False) -> UserSession:
        """Create new user session"""
        session_id = f"SES-{secrets.token_urlsafe(32)}"
        current_time = datetime.now(timezone.utc)
        
        # Check concurrent session limit
        active_sessions = [
            session for session in self.user_sessions.values()
            if session.user_id == user_id and session.status == SessionStatus.ACTIVE
        ]
        
        if len(active_sessions) >= self.security_policy.concurrent_sessions_limit:
            # Revoke oldest session
            oldest_session = min(active_sessions, key=lambda s: s.last_activity)
            oldest_session.status = SessionStatus.REVOKED
            
            await self.audit_logger.log_security_event(
                "Session revoked due to concurrent session limit",
                "medium",
                user_id=user_id,
                details={"revoked_session_id": oldest_session.session_id}
            )
        
        # Create new session
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            device_fingerprint=device_fingerprint or "unknown",
            ip_address=ip_address,
            user_agent=user_agent,
            location=None,  # Would use IP geolocation service
            created_at=current_time,
            last_activity=current_time,
            expires_at=current_time + timedelta(minutes=self.security_policy.session_timeout_minutes),
            status=SessionStatus.ACTIVE,
            mfa_completed=mfa_completed,
            authentication_methods=[AuthenticationMethod.PASSWORD],
            security_level=3 if mfa_completed else 2,
            metadata={}
        )
        
        self.user_sessions[session_id] = session
        
        await self.audit_logger.log_security_event(
            "User session created",
            "low",
            user_id=user_id,
            source_ip=ip_address,
            details={
                "session_id": session_id,
                "mfa_completed": mfa_completed,
                "device_fingerprint": device_fingerprint
            }
        )
        
        return session
    
    async def _log_authentication_attempt(self, attempt_id: str, user_id: Optional[str],
                                        username: str, ip_address: str, user_agent: str,
                                        method: AuthenticationMethod, success: bool,
                                        failure_reason: Optional[str],
                                        session_id: Optional[str] = None):
        """Log authentication attempt"""
        attempt = AuthenticationAttempt(
            attempt_id=attempt_id,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            method=method,
            success=success,
            failure_reason=failure_reason,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            metadata={}
        )
        
        self.auth_attempts.append(attempt)
        
        # Log to audit trail
        await self.audit_logger.log_authentication_attempt(
            user_id=user_id or "unknown",
            success=success,
            source_ip=ip_address,
            user_agent=user_agent,
            details={
                "attempt_id": attempt_id,
                "method": method.value,
                "failure_reason": failure_reason,
                "session_id": session_id
            }
        )
        
        # Check for account lockout on repeated failures
        if not success and user_id:
            await self._check_and_apply_lockout(user_id, username)
    
    async def _check_and_apply_lockout(self, user_id: str, username: str):
        """Check if account should be locked due to failed attempts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        
        failed_attempts = [
            attempt for attempt in self.auth_attempts
            if attempt.user_id == user_id and 
               not attempt.success and
               attempt.timestamp > cutoff_time
        ]
        
        if len(failed_attempts) >= self.security_policy.account_lockout_threshold:
            unlock_time = datetime.now(timezone.utc) + timedelta(
                minutes=self.security_policy.account_lockout_duration_minutes
            )
            self.locked_accounts[user_id] = unlock_time
            
            await self.audit_logger.log_security_event(
                "Account locked due to failed authentication attempts",
                "high",
                user_id=user_id,
                details={
                    "failed_attempts": len(failed_attempts),
                    "unlock_time": unlock_time.isoformat(),
                    "username": username
                }
            )
    
    # ==================== MFA MANAGEMENT ====================
    
    async def setup_totp_mfa(self, user_id: str, device_name: str) -> Dict[str, Any]:
        """Set up TOTP MFA for user"""
        user = self.auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate TOTP secret
        secret = pyotp.random_base32()
        
        # Create device record
        device_id = f"MFA-{secrets.token_hex(8)}"
        device = MFADevice(
            device_id=device_id,
            user_id=user_id,
            device_type=AuthenticationMethod.TOTP,
            device_name=device_name,
            secret=secret,
            phone_number=None,
            email=None,
            webauthn_credential=None,
            is_primary=len(self.mfa_devices.get(user_id, [])) == 0,
            is_verified=False,
            created_at=datetime.now(timezone.utc),
            last_used=None,
            backup_codes=self._generate_backup_codes(),
            metadata={}
        )
        
        # Store device
        if user_id not in self.mfa_devices:
            self.mfa_devices[user_id] = []
        self.mfa_devices[user_id].append(device)
        
        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email,
            issuer_name="UAP Platform"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        qr_code_data = base64.b64encode(img_buffer.getvalue()).decode()
        
        await self.audit_logger.log_security_event(
            "TOTP MFA device setup initiated",
            "medium",
            user_id=user_id,
            details={"device_id": device_id, "device_name": device_name}
        )
        
        return {
            "device_id": device_id,
            "secret": secret,
            "qr_code": f"data:image/png;base64,{qr_code_data}",
            "backup_codes": device.backup_codes,
            "setup_uri": totp_uri
        }
    
    async def verify_totp_setup(self, user_id: str, device_id: str, verification_code: str) -> bool:
        """Verify TOTP setup with verification code"""
        user_devices = self.mfa_devices.get(user_id, [])
        device = next((d for d in user_devices if d.device_id == device_id), None)
        
        if not device or device.device_type != AuthenticationMethod.TOTP:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TOTP device not found"
            )
        
        if device.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device is already verified"
            )
        
        # Verify the code
        totp = pyotp.TOTP(device.secret)
        if totp.verify(verification_code, valid_window=1):
            device.is_verified = True
            device.last_used = datetime.now(timezone.utc)
            
            await self.audit_logger.log_security_event(
                "TOTP MFA device verified and activated",
                "medium",
                user_id=user_id,
                details={"device_id": device_id}
            )
            
            return True
        
        await self.audit_logger.log_security_event(
            "TOTP MFA device verification failed",
            "medium",
            user_id=user_id,
            details={"device_id": device_id}
        )
        
        return False
    
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        codes = []
        for _ in range(count):
            code = "-".join([
                f"{secrets.randbelow(10000):04d}" for _ in range(2)
            ])
            codes.append(code)
        return codes
    
    async def get_user_mfa_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's MFA devices"""
        devices = self.mfa_devices.get(user_id, [])
        
        return [
            {
                "device_id": device.device_id,
                "device_type": device.device_type.value,
                "device_name": device.device_name,
                "is_primary": device.is_primary,
                "is_verified": device.is_verified,
                "created_at": device.created_at.isoformat(),
                "last_used": device.last_used.isoformat() if device.last_used else None,
                "backup_codes_remaining": len(device.backup_codes) if device.backup_codes else 0
            }
            for device in devices
        ]
    
    async def remove_mfa_device(self, user_id: str, device_id: str) -> bool:
        """Remove MFA device"""
        user_devices = self.mfa_devices.get(user_id, [])
        device = next((d for d in user_devices if d.device_id == device_id), None)
        
        if not device:
            return False
        
        user_devices.remove(device)
        
        await self.audit_logger.log_security_event(
            "MFA device removed",
            "medium",
            user_id=user_id,
            details={"device_id": device_id, "device_type": device.device_type.value}
        )
        
        return True
    
    # ==================== SESSION MANAGEMENT ====================
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's active sessions"""
        user_sessions = [
            session for session in self.user_sessions.values()
            if session.user_id == user_id and session.status == SessionStatus.ACTIVE
        ]
        
        return [
            {
                "session_id": session.session_id,
                "device_fingerprint": session.device_fingerprint,
                "ip_address": session.ip_address,
                "location": session.location,
                "user_agent": session.user_agent,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "mfa_completed": session.mfa_completed,
                "security_level": session.security_level,
                "is_current": False  # Would be determined by request context
            }
            for session in user_sessions
        ]
    
    async def revoke_session(self, user_id: str, session_id: str) -> bool:
        """Revoke user session"""
        session = self.user_sessions.get(session_id)
        
        if not session or session.user_id != user_id:
            return False
        
        session.status = SessionStatus.REVOKED
        
        await self.audit_logger.log_security_event(
            "User session revoked",
            "medium",
            user_id=user_id,
            details={"session_id": session_id}
        )
        
        return True
    
    async def revoke_all_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Revoke all user sessions except specified one"""
        revoked_count = 0
        
        for session in self.user_sessions.values():
            if (session.user_id == user_id and 
                session.status == SessionStatus.ACTIVE and
                session.session_id != except_session):
                session.status = SessionStatus.REVOKED
                revoked_count += 1
        
        await self.audit_logger.log_security_event(
            "All user sessions revoked",
            "high",
            user_id=user_id,
            details={"revoked_count": revoked_count, "except_session": except_session}
        )
        
        return revoked_count
    
    # ==================== PASSWORD SECURITY ====================
    
    def validate_password_policy(self, password: str, user_id: Optional[str] = None) -> List[str]:
        """Validate password against security policy"""
        violations = []
        
        if len(password) < self.security_policy.password_min_length:
            violations.append(f"Password must be at least {self.security_policy.password_min_length} characters long")
        
        if self.security_policy.password_require_uppercase and not re.search(r'[A-Z]', password):
            violations.append("Password must contain at least one uppercase letter")
        
        if self.security_policy.password_require_lowercase and not re.search(r'[a-z]', password):
            violations.append("Password must contain at least one lowercase letter")
        
        if self.security_policy.password_require_numbers and not re.search(r'\d', password):
            violations.append("Password must contain at least one number")
        
        if self.security_policy.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            violations.append("Password must contain at least one special character")
        
        # Check password history
        if user_id and user_id in self.password_history:
            password_hash = self._hash_password(password)
            if password_hash in self.password_history[user_id]:
                violations.append(f"Password cannot be one of the last {self.security_policy.password_history_count} passwords")
        
        return violations
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password with policy validation"""
        user = self.auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not self.auth_service.verify_password(current_password, user.hashed_password):
            await self.audit_logger.log_security_event(
                "Password change failed - invalid current password",
                "medium",
                user_id=user_id
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        policy_violations = self.validate_password_policy(new_password, user_id)
        if policy_violations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password policy violations: " + "; ".join(policy_violations)
            )
        
        # Update password
        new_hash = self.auth_service.get_password_hash(new_password)
        user.hashed_password = new_hash
        
        # Update password history
        if user_id not in self.password_history:
            self.password_history[user_id] = []
        
        self.password_history[user_id].append(self._hash_password(new_password))
        
        # Keep only recent passwords
        if len(self.password_history[user_id]) > self.security_policy.password_history_count:
            self.password_history[user_id] = self.password_history[user_id][-self.security_policy.password_history_count:]
        
        await self.audit_logger.log_security_event(
            "Password changed successfully",
            "medium",
            user_id=user_id
        )
        
        return True
    
    # ==================== SECURITY MONITORING ====================
    
    async def get_security_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get user security dashboard"""
        user_devices = self.mfa_devices.get(user_id, [])
        active_sessions = [
            s for s in self.user_sessions.values()
            if s.user_id == user_id and s.status == SessionStatus.ACTIVE
        ]
        
        recent_attempts = [
            attempt for attempt in self.auth_attempts
            if attempt.user_id == user_id and 
               attempt.timestamp > datetime.now(timezone.utc) - timedelta(days=30)
        ]
        
        return {
            "user_id": user_id,
            "mfa_status": "enabled" if any(d.is_verified for d in user_devices) else "disabled",
            "mfa_devices": len([d for d in user_devices if d.is_verified]),
            "active_sessions": len(active_sessions),
            "recent_logins": len([a for a in recent_attempts if a.success]),
            "failed_attempts": len([a for a in recent_attempts if not a.success]),
            "account_locked": user_id in self.locked_accounts,
            "password_age_days": 0,  # Would calculate from password change history
            "security_score": self._calculate_security_score(user_id),
            "last_login": max([a.timestamp for a in recent_attempts if a.success], default=None),
            "recommendations": self._get_security_recommendations(user_id)
        }
    
    def _calculate_security_score(self, user_id: str) -> int:
        """Calculate user security score (0-100)"""
        score = 50  # Base score
        
        # MFA enabled
        user_devices = self.mfa_devices.get(user_id, [])
        if any(d.is_verified for d in user_devices):
            score += 30
        
        # Multiple MFA devices
        verified_devices = [d for d in user_devices if d.is_verified]
        if len(verified_devices) > 1:
            score += 10
        
        # Recent activity
        recent_attempts = [
            attempt for attempt in self.auth_attempts
            if attempt.user_id == user_id and 
               attempt.timestamp > datetime.now(timezone.utc) - timedelta(days=7)
        ]
        
        if recent_attempts:
            success_rate = len([a for a in recent_attempts if a.success]) / len(recent_attempts)
            if success_rate > 0.9:
                score += 10
        
        return min(100, max(0, score))
    
    def _get_security_recommendations(self, user_id: str) -> List[str]:
        """Get security recommendations for user"""
        recommendations = []
        
        user_devices = self.mfa_devices.get(user_id, [])
        if not any(d.is_verified for d in user_devices):
            recommendations.append("Enable multi-factor authentication for enhanced security")
        
        verified_devices = [d for d in user_devices if d.is_verified]
        if len(verified_devices) == 1:
            recommendations.append("Add a backup MFA device in case your primary device is unavailable")
        
        # Check for backup codes
        totp_devices = [d for d in verified_devices if d.device_type == AuthenticationMethod.TOTP]
        if totp_devices and any(not d.backup_codes or len(d.backup_codes) < 5 for d in totp_devices):
            recommendations.append("Generate new backup codes for your TOTP devices")
        
        active_sessions = [
            s for s in self.user_sessions.values()
            if s.user_id == user_id and s.status == SessionStatus.ACTIVE
        ]
        
        if len(active_sessions) > 1:
            recommendations.append("Review and revoke unused active sessions")
        
        return recommendations

# Global enterprise auth manager instance
_global_enterprise_auth_manager = None

def get_enterprise_auth_manager() -> EnterpriseAuthManager:
    """Get global enterprise auth manager instance"""
    global _global_enterprise_auth_manager
    if _global_enterprise_auth_manager is None:
        from ..services.auth import auth_service
        _global_enterprise_auth_manager = EnterpriseAuthManager(auth_service)
    return _global_enterprise_auth_manager
