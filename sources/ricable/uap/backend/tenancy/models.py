# File: backend/tenancy/models.py
"""
Multi-tenancy data models with organization isolation
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, EmailStr, validator
from enum import Enum
import uuid
import json

class TenantStatus(str, Enum):
    """Tenant status enumeration"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    CANCELLED = "cancelled"
    TRIAL = "trial"

class OrganizationType(str, Enum):
    """Organization type enumeration"""
    ENTERPRISE = "enterprise"
    BUSINESS = "business"
    STARTUP = "startup"
    INDIVIDUAL = "individual"
    NONPROFIT = "nonprofit"

class BillingCycle(str, Enum):
    """Billing cycle enumeration"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ONE_TIME = "one_time"

class Organization(BaseModel):
    """Organization model for multi-tenancy"""
    id: str
    name: str
    slug: str  # URL-friendly identifier
    domain: Optional[str] = None  # Custom domain for white-labeling
    organization_type: OrganizationType = OrganizationType.BUSINESS
    status: TenantStatus = TenantStatus.ACTIVE
    
    # Contact Information
    email: EmailStr
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = {}
    
    # Organization Settings
    settings: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    # White-label Configuration
    branding: Dict[str, Any] = {}  # Logo, colors, custom CSS
    custom_domain: Optional[str] = None
    white_label_enabled: bool = False
    
    # Limits and Quotas
    user_limit: Optional[int] = None
    storage_limit_gb: Optional[float] = None
    api_rate_limit: Optional[int] = None
    feature_flags: Dict[str, bool] = {}
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('slug')
    def validate_slug(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Slug must be alphanumeric with hyphens and underscores allowed')
        return v.lower()
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

class Tenant(BaseModel):
    """Tenant model representing a customer instance"""
    id: str
    organization_id: str
    name: str
    slug: str
    subdomain: Optional[str] = None  # For multi-tenant subdomains
    
    # Tenant Configuration
    status: TenantStatus = TenantStatus.ACTIVE
    tier: str = "basic"  # Subscription tier
    features: List[str] = []
    limits: Dict[str, Any] = {}
    
    # Database Configuration
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    connection_string: Optional[str] = None
    
    # Isolation Settings
    isolation_level: str = "schema"  # database, schema, row
    encryption_key: Optional[str] = None
    
    # Usage Tracking
    current_users: int = 0
    storage_used_gb: float = 0.0
    api_calls_this_month: int = 0
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    last_accessed: Optional[datetime] = None
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

class TenantUser(BaseModel):
    """User model with tenant association"""
    id: str
    tenant_id: str
    organization_id: str
    user_id: str  # Reference to main user record
    
    # Tenant-specific roles and permissions
    tenant_roles: List[str] = []
    tenant_permissions: List[str] = []
    is_tenant_admin: bool = False
    
    # Access Control
    is_active: bool = True
    can_invite_users: bool = False
    can_manage_billing: bool = False
    
    # Usage Tracking
    last_login: Optional[datetime] = None
    login_count: int = 0
    api_usage: Dict[str, int] = {}
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

class TenantInvitation(BaseModel):
    """Invitation to join a tenant"""
    id: str
    tenant_id: str
    organization_id: str
    inviter_id: str
    
    # Invitation Details
    email: EmailStr
    roles: List[str] = []
    message: Optional[str] = None
    
    # Status
    status: str = "pending"  # pending, accepted, expired, cancelled
    expires_at: datetime
    
    # Timestamps
    created_at: datetime
    accepted_at: Optional[datetime] = None
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)

class TenantSettings(BaseModel):
    """Tenant-specific configuration settings"""
    tenant_id: str
    
    # UI Customization
    theme: Dict[str, Any] = {}
    logo_url: Optional[str] = None
    favicon_url: Optional[str] = None
    custom_css: Optional[str] = None
    
    # Feature Configuration
    enabled_features: List[str] = []
    feature_config: Dict[str, Any] = {}
    
    # Integration Settings
    integrations: Dict[str, Dict[str, Any]] = {}
    webhooks: List[Dict[str, Any]] = []
    
    # Security Settings
    session_timeout: int = 3600  # seconds
    password_policy: Dict[str, Any] = {}
    mfa_required: bool = False
    ip_whitelist: List[str] = []
    
    # Notification Settings
    email_notifications: bool = True
    slack_webhook: Optional[str] = None
    
    # Data Retention
    data_retention_days: int = 365
    backup_enabled: bool = True
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

# Request/Response Models

class OrganizationCreate(BaseModel):
    """Organization creation request"""
    name: str
    slug: str
    email: EmailStr
    organization_type: OrganizationType = OrganizationType.BUSINESS
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    
    @validator('slug')
    def validate_slug(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Slug must be alphanumeric with hyphens and underscores allowed')
        return v.lower()

class TenantCreate(BaseModel):
    """Tenant creation request"""
    name: str
    slug: str
    tier: str = "basic"
    subdomain: Optional[str] = None
    features: List[str] = []

class TenantUpdate(BaseModel):
    """Tenant update request"""
    name: Optional[str] = None
    status: Optional[TenantStatus] = None
    tier: Optional[str] = None
    features: Optional[List[str]] = None
    limits: Optional[Dict[str, Any]] = None

class TenantUserCreate(BaseModel):
    """Tenant user creation request"""
    user_id: str
    tenant_roles: List[str] = ["user"]
    is_tenant_admin: bool = False
    can_invite_users: bool = False
    can_manage_billing: bool = False

class TenantInvitationCreate(BaseModel):
    """Tenant invitation creation request"""
    email: EmailStr
    roles: List[str] = ["user"]
    message: Optional[str] = None
    expires_in_hours: int = 72

class TenantUsageReport(BaseModel):
    """Tenant usage report"""
    tenant_id: str
    reporting_period: str
    
    # Usage Metrics
    active_users: int
    total_api_calls: int
    storage_used_gb: float
    bandwidth_used_gb: float
    
    # Feature Usage
    feature_usage: Dict[str, int] = {}
    
    # Cost Information
    estimated_cost: float = 0.0
    billing_tier: str = "basic"
    
    # Timestamps
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    @validator('generated_at', pre=True, always=True)
    def set_generated_at(cls, v):
        return v or datetime.now(timezone.utc)