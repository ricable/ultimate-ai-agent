# File: backend/billing/models.py
"""
Billing and subscription models for enterprise multi-tenancy
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, EmailStr, validator
from enum import Enum
from decimal import Decimal
import uuid

class BillingCycle(str, Enum):
    """Billing cycle enumeration"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ONE_TIME = "one_time"

class SubscriptionStatus(str, Enum):
    """Subscription status enumeration"""
    ACTIVE = "active"
    TRIAL = "trial"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

class PaymentStatus(str, Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

class InvoiceStatus(str, Enum):
    """Invoice status enumeration"""
    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    VOID = "void"

class UsageUnit(str, Enum):
    """Usage tracking units"""
    API_CALLS = "api_calls"
    USERS = "users"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    COMPUTE_HOURS = "compute_hours"
    DOCUMENTS = "documents"
    AGENTS = "agents"

class BillingPlan(BaseModel):
    """Billing plan definition"""
    id: str
    name: str
    description: str
    tier: str  # basic, pro, enterprise, custom
    
    # Pricing
    base_price: Decimal
    currency: str = "USD"
    billing_cycle: BillingCycle
    
    # Features and Limits
    features: List[str] = []
    limits: Dict[str, Any] = {}
    
    # Usage-based pricing
    usage_rates: Dict[str, Decimal] = {}  # unit -> price per unit
    included_usage: Dict[str, int] = {}   # unit -> included amount
    
    # Metadata
    is_active: bool = True
    is_public: bool = True
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

class Subscription(BaseModel):
    """Customer subscription"""
    id: str
    tenant_id: str
    organization_id: str
    plan_id: str
    
    # Status and Timing
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    
    # Pricing
    base_amount: Decimal
    discount_amount: Decimal = Decimal('0.00')
    tax_amount: Decimal = Decimal('0.00')
    total_amount: Decimal
    currency: str = "USD"
    
    # Usage Limits and Tracking
    usage_limits: Dict[str, int] = {}
    current_usage: Dict[str, int] = {}
    
    # Payment
    payment_method_id: Optional[str] = None
    billing_email: EmailStr
    
    # Flags
    auto_renew: bool = True
    proration_enabled: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    cancelled_at: Optional[datetime] = None
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

class Invoice(BaseModel):
    """Billing invoice"""
    id: str
    subscription_id: str
    tenant_id: str
    organization_id: str
    
    # Invoice Details
    invoice_number: str
    status: InvoiceStatus
    
    # Amounts
    subtotal: Decimal
    discount_amount: Decimal = Decimal('0.00')
    tax_amount: Decimal = Decimal('0.00')
    total_amount: Decimal
    amount_paid: Decimal = Decimal('0.00')
    amount_due: Decimal
    currency: str = "USD"
    
    # Line Items
    line_items: List[Dict[str, Any]] = []
    
    # Dates
    invoice_date: datetime
    due_date: datetime
    paid_date: Optional[datetime] = None
    
    # Customer Information
    billing_address: Dict[str, str] = {}
    customer_email: EmailStr
    
    # Payment Information
    payment_attempts: int = 0
    last_payment_error: Optional[str] = None
    
    # Document URLs
    pdf_url: Optional[str] = None
    hosted_url: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('invoice_number', pre=True, always=True)
    def set_invoice_number(cls, v, values):
        if not v:
            # Generate invoice number: INV-YYYY-MM-XXXXXX
            now = datetime.now(timezone.utc)
            return f"INV-{now.strftime('%Y-%m')}-{str(uuid.uuid4())[:8].upper()}"
        return v
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

class Payment(BaseModel):
    """Payment record"""
    id: str
    invoice_id: Optional[str] = None
    subscription_id: Optional[str] = None
    tenant_id: str
    organization_id: str
    
    # Payment Details
    amount: Decimal
    currency: str = "USD"
    status: PaymentStatus
    
    # Payment Method
    payment_method_type: str  # card, bank_transfer, paypal, etc.
    payment_method_id: Optional[str] = None
    
    # Transaction Information
    transaction_id: Optional[str] = None
    gateway: str = "stripe"  # payment processor
    gateway_transaction_id: Optional[str] = None
    
    # Failure Information
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now(timezone.utc)

class UsageRecord(BaseModel):
    """Usage tracking record"""
    id: str
    tenant_id: str
    subscription_id: str
    
    # Usage Details
    usage_type: UsageUnit
    quantity: int
    unit_price: Optional[Decimal] = None
    
    # Timing
    usage_date: datetime
    billing_period_start: datetime
    billing_period_end: datetime
    
    # Aggregation
    is_billable: bool = True
    is_aggregated: bool = False
    
    # Source Information
    source_id: Optional[str] = None  # agent_id, document_id, etc.
    source_type: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    # Timestamps
    created_at: datetime
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)

class PaymentMethod(BaseModel):
    """Payment method information"""
    id: str
    tenant_id: str
    
    # Method Details
    type: str  # card, bank_account, paypal
    provider: str = "stripe"
    provider_id: str  # External provider ID
    
    # Card Information (if applicable)
    card_last4: Optional[str] = None
    card_brand: Optional[str] = None
    card_exp_month: Optional[int] = None
    card_exp_year: Optional[int] = None
    
    # Bank Account Information (if applicable)
    bank_name: Optional[str] = None
    account_last4: Optional[str] = None
    routing_number: Optional[str] = None
    
    # Status
    is_default: bool = False
    is_verified: bool = False
    is_active: bool = True
    
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

# Request/Response Models

class SubscriptionCreate(BaseModel):
    """Create subscription request"""
    plan_id: str
    billing_email: EmailStr
    payment_method_id: Optional[str] = None
    trial_days: int = 0
    metadata: Dict[str, Any] = {}

class SubscriptionUpdate(BaseModel):
    """Update subscription request"""
    plan_id: Optional[str] = None
    billing_email: Optional[EmailStr] = None
    payment_method_id: Optional[str] = None
    auto_renew: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class PaymentMethodCreate(BaseModel):
    """Create payment method request"""
    type: str
    provider_token: str  # Token from payment provider
    is_default: bool = False
    metadata: Dict[str, Any] = {}

class UsageReportRequest(BaseModel):
    """Usage report request"""
    start_date: datetime
    end_date: datetime
    usage_types: Optional[List[UsageUnit]] = None
    include_non_billable: bool = False

class BillingReport(BaseModel):
    """Billing report response"""
    tenant_id: str
    subscription_id: str
    period_start: datetime
    period_end: datetime
    
    # Summary
    total_base_cost: Decimal
    total_usage_cost: Decimal
    total_cost: Decimal
    
    # Breakdown
    usage_breakdown: Dict[str, Dict[str, Any]] = {}
    line_items: List[Dict[str, Any]] = []
    
    # Statistics
    total_api_calls: int = 0
    total_storage_gb: float = 0.0
    total_bandwidth_gb: float = 0.0
    active_users: int = 0
    
    # Metadata
    generated_at: datetime
    
    @validator('generated_at', pre=True, always=True)
    def set_generated_at(cls, v):
        return v or datetime.now(timezone.utc)

class BillingPreview(BaseModel):
    """Billing preview for plan changes"""
    current_plan_id: str
    new_plan_id: str
    
    # Cost Impact
    immediate_charge: Decimal
    next_invoice_change: Decimal
    proration_amount: Decimal
    
    # Usage Impact
    new_limits: Dict[str, int] = {}
    feature_changes: Dict[str, bool] = {}
    
    # Timing
    effective_date: datetime
    next_billing_date: datetime

class CreditNote(BaseModel):
    """Credit note for refunds/adjustments"""
    id: str
    invoice_id: str
    tenant_id: str
    
    # Credit Details
    amount: Decimal
    currency: str = "USD"
    reason: str
    
    # Status
    status: str = "pending"  # pending, applied, cancelled
    
    # Timestamps
    created_at: datetime
    applied_at: Optional[datetime] = None
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now(timezone.utc)