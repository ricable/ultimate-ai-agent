# File: backend/billing/billing_api.py
"""
Enterprise billing API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, EmailStr
from decimal import Decimal
import uuid
import logging

from .models import (
    BillingPlan, Subscription, Invoice, Payment, PaymentMethod, UsageRecord,
    SubscriptionCreate, SubscriptionUpdate, PaymentMethodCreate, BillingReport,
    BillingPreview, CreditNote, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    BillingCycle, UsageUnit
)
from .subscription_manager import subscription_manager
from .payment_processors import payment_processor_manager, PaymentResult
from ..tenancy.tenant_context import (
    require_tenant_context, require_tenant_admin, get_current_tenant,
    TenantContext
)
from ..tenancy.organization_manager import organization_manager
from ..services.auth import UserInDB, require_permission
from ..monitoring.logs.logger import uap_logger, EventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["Billing"])

# Request/Response Models

class PaymentMethodResponse(BaseModel):
    """Payment method response"""
    id: str
    type: str
    provider: str
    card_last4: Optional[str] = None
    card_brand: Optional[str] = None
    card_exp_month: Optional[int] = None
    card_exp_year: Optional[int] = None
    is_default: bool
    is_verified: bool
    is_active: bool
    created_at: datetime

class SubscriptionResponse(BaseModel):
    """Subscription response"""
    id: str
    tenant_id: str
    plan_id: str
    plan_name: str
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    base_amount: Decimal
    total_amount: Decimal
    currency: str
    billing_email: EmailStr
    auto_renew: bool
    usage_limits: Dict[str, int]
    current_usage: Dict[str, int]
    created_at: datetime

class InvoiceResponse(BaseModel):
    """Invoice response"""
    id: str
    invoice_number: str
    status: InvoiceStatus
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    amount_paid: Decimal
    amount_due: Decimal
    currency: str
    invoice_date: datetime
    due_date: datetime
    paid_date: Optional[datetime] = None
    line_items: List[Dict[str, Any]]
    pdf_url: Optional[str] = None
    hosted_url: Optional[str] = None

class PaymentRequest(BaseModel):
    """Payment request"""
    payment_method_id: str
    amount: Optional[Decimal] = None  # If None, pay full invoice amount
    currency: str = "USD"
    metadata: Dict[str, Any] = {}

class UsageTrackingRequest(BaseModel):
    """Usage tracking request"""
    usage_type: UsageUnit
    quantity: int
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Dict[str, Any] = {}

# Billing Plan Endpoints

@router.get("/plans", response_model=List[BillingPlan])
async def list_billing_plans(
    include_inactive: bool = Query(False),
    current_user: UserInDB = Depends(require_permission("billing:read"))
):
    """List available billing plans"""
    try:
        plans = await subscription_manager.list_plans(include_inactive)
        return plans
        
    except Exception as e:
        logger.error(f"Failed to list billing plans: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list billing plans")

@router.get("/plans/{plan_id}", response_model=BillingPlan)
async def get_billing_plan(
    plan_id: str,
    current_user: UserInDB = Depends(require_permission("billing:read"))
):
    """Get billing plan details"""
    try:
        plan = await subscription_manager.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Billing plan not found")
        
        return plan
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get billing plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get billing plan")

# Subscription Management

@router.get("/subscription", response_model=SubscriptionResponse)
async def get_tenant_subscription(
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """Get current tenant subscription"""
    try:
        subscription = await subscription_manager.get_tenant_subscription(tenant_context.tenant_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No subscription found")
        
        # Get plan details
        plan = await subscription_manager.get_plan(subscription.plan_id)
        plan_name = plan.name if plan else "Unknown Plan"
        
        return SubscriptionResponse(
            id=subscription.id,
            tenant_id=subscription.tenant_id,
            plan_id=subscription.plan_id,
            plan_name=plan_name,
            status=subscription.status,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            trial_end=subscription.trial_end,
            base_amount=subscription.base_amount,
            total_amount=subscription.total_amount,
            currency=subscription.currency,
            billing_email=subscription.billing_email,
            auto_renew=subscription.auto_renew,
            usage_limits=subscription.usage_limits,
            current_usage=subscription.current_usage,
            created_at=subscription.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get subscription")

@router.post("/subscription", response_model=SubscriptionResponse)
async def create_subscription(
    subscription_data: SubscriptionCreate,
    tenant_context: TenantContext = Depends(require_tenant_admin)
):
    """Create new subscription for tenant"""
    try:
        subscription = await subscription_manager.create_subscription(
            tenant_context.tenant_id,
            tenant_context.organization_id,
            subscription_data
        )
        
        # Get plan details
        plan = await subscription_manager.get_plan(subscription.plan_id)
        plan_name = plan.name if plan else "Unknown Plan"
        
        # Log subscription creation
        uap_logger.log_billing_event(
            f"Subscription created: {subscription.id}",
            "billing_api",
            {
                "subscription_id": subscription.id,
                "tenant_id": tenant_context.tenant_id,
                "plan_id": subscription.plan_id
            }
        )
        
        return SubscriptionResponse(
            id=subscription.id,
            tenant_id=subscription.tenant_id,
            plan_id=subscription.plan_id,
            plan_name=plan_name,
            status=subscription.status,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            trial_end=subscription.trial_end,
            base_amount=subscription.base_amount,
            total_amount=subscription.total_amount,
            currency=subscription.currency,
            billing_email=subscription.billing_email,
            auto_renew=subscription.auto_renew,
            usage_limits=subscription.usage_limits,
            current_usage=subscription.current_usage,
            created_at=subscription.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create subscription")

@router.put("/subscription", response_model=SubscriptionResponse)
async def update_subscription(
    updates: SubscriptionUpdate,
    tenant_context: TenantContext = Depends(require_tenant_admin)
):
    """Update tenant subscription"""
    try:
        # Get current subscription
        subscription = await subscription_manager.get_tenant_subscription(tenant_context.tenant_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No subscription found")
        
        # Update subscription
        updated_subscription = await subscription_manager.update_subscription(
            subscription.id, updates
        )
        
        # Get plan details
        plan = await subscription_manager.get_plan(updated_subscription.plan_id)
        plan_name = plan.name if plan else "Unknown Plan"
        
        # Log subscription update
        uap_logger.log_billing_event(
            f"Subscription updated: {subscription.id}",
            "billing_api",
            {
                "subscription_id": subscription.id,
                "tenant_id": tenant_context.tenant_id,
                "changes": updates.dict(exclude_unset=True)
            }
        )
        
        return SubscriptionResponse(
            id=updated_subscription.id,
            tenant_id=updated_subscription.tenant_id,
            plan_id=updated_subscription.plan_id,
            plan_name=plan_name,
            status=updated_subscription.status,
            current_period_start=updated_subscription.current_period_start,
            current_period_end=updated_subscription.current_period_end,
            trial_end=updated_subscription.trial_end,
            base_amount=updated_subscription.base_amount,
            total_amount=updated_subscription.total_amount,
            currency=updated_subscription.currency,
            billing_email=updated_subscription.billing_email,
            auto_renew=updated_subscription.auto_renew,
            usage_limits=updated_subscription.usage_limits,
            current_usage=updated_subscription.current_usage,
            created_at=updated_subscription.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update subscription")

@router.delete("/subscription")
async def cancel_subscription(
    immediate: bool = Query(False),
    tenant_context: TenantContext = Depends(require_tenant_admin)
):
    """Cancel tenant subscription"""
    try:
        # Get current subscription
        subscription = await subscription_manager.get_tenant_subscription(tenant_context.tenant_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No subscription found")
        
        # Cancel subscription
        cancelled_subscription = await subscription_manager.cancel_subscription(
            subscription.id, immediate
        )
        
        # Log cancellation
        uap_logger.log_billing_event(
            f"Subscription cancelled: {subscription.id}",
            "billing_api",
            {
                "subscription_id": subscription.id,
                "tenant_id": tenant_context.tenant_id,
                "immediate": immediate
            }
        )
        
        return {
            "message": "Subscription cancelled successfully",
            "subscription_id": subscription.id,
            "immediate": immediate,
            "effective_date": cancelled_subscription.current_period_end if not immediate else datetime.now(timezone.utc)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")

# Payment Method Management

@router.get("/payment-methods", response_model=List[PaymentMethodResponse])
async def list_payment_methods(
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """List payment methods for tenant"""
    try:
        # Get payment methods for tenant
        payment_methods = [
            pm for pm in subscription_manager.payment_methods.values()
            if pm.tenant_id == tenant_context.tenant_id
        ]
        
        response_methods = []
        for pm in payment_methods:
            response_methods.append(PaymentMethodResponse(
                id=pm.id,
                type=pm.type,
                provider=pm.provider,
                card_last4=pm.card_last4,
                card_brand=pm.card_brand,
                card_exp_month=pm.card_exp_month,
                card_exp_year=pm.card_exp_year,
                is_default=pm.is_default,
                is_verified=pm.is_verified,
                is_active=pm.is_active,
                created_at=pm.created_at
            ))
        
        return response_methods
        
    except Exception as e:
        logger.error(f"Failed to list payment methods: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list payment methods")

@router.post("/payment-methods", response_model=PaymentMethodResponse)
async def create_payment_method(
    payment_method_data: PaymentMethodCreate,
    tenant_context: TenantContext = Depends(require_tenant_admin)
):
    """Create new payment method"""
    try:
        # Get organization and tenant info
        org = await organization_manager.get_organization(tenant_context.organization_id)
        tenant = await organization_manager.get_tenant(tenant_context.tenant_id)
        
        if not org or not tenant:
            raise HTTPException(status_code=404, detail="Organization or tenant not found")
        
        # Prepare customer data
        customer_data = {
            "email": org.email,
            "name": org.name,
            "tenant_id": tenant_context.tenant_id,
            "organization_id": tenant_context.organization_id
        }
        
        # Create payment method with processor
        processor_result = await payment_processor_manager.create_payment_method(
            payment_method_data.provider_token,  # This would be the provider (stripe, paypal, etc.)
            customer_data,
            payment_method_data.dict()
        )
        
        # Create payment method record
        payment_method = PaymentMethod(
            id=str(uuid.uuid4()),
            tenant_id=tenant_context.tenant_id,
            type=payment_method_data.type,
            provider=processor_result.get("provider", "stripe"),
            provider_id=processor_result["payment_method_id"],
            card_last4=processor_result.get("card_last4"),
            card_brand=processor_result.get("card_brand"),
            card_exp_month=processor_result.get("card_exp_month"),
            card_exp_year=processor_result.get("card_exp_year"),
            is_default=payment_method_data.is_default,
            is_verified=True,
            metadata=payment_method_data.metadata
        )
        
        # Set as default if it's the first payment method or explicitly set
        tenant_payment_methods = [
            pm for pm in subscription_manager.payment_methods.values()
            if pm.tenant_id == tenant_context.tenant_id
        ]
        
        if not tenant_payment_methods or payment_method_data.is_default:
            payment_method.is_default = True
            
            # Unset other default payment methods
            for pm in tenant_payment_methods:
                if pm.is_default:
                    pm.is_default = False
                    subscription_manager.payment_methods[pm.id] = pm
        
        subscription_manager.payment_methods[payment_method.id] = payment_method
        
        return PaymentMethodResponse(
            id=payment_method.id,
            type=payment_method.type,
            provider=payment_method.provider,
            card_last4=payment_method.card_last4,
            card_brand=payment_method.card_brand,
            card_exp_month=payment_method.card_exp_month,
            card_exp_year=payment_method.card_exp_year,
            is_default=payment_method.is_default,
            is_verified=payment_method.is_verified,
            is_active=payment_method.is_active,
            created_at=payment_method.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create payment method: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create payment method")

# Invoice Management

@router.get("/invoices", response_model=List[InvoiceResponse])
async def list_invoices(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[InvoiceStatus] = Query(None),
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """List invoices for tenant"""
    try:
        invoices = await subscription_manager.list_invoices(
            tenant_id=tenant_context.tenant_id,
            status=status
        )
        
        # Apply pagination
        paginated_invoices = invoices[offset:offset + limit]
        
        response_invoices = []
        for invoice in paginated_invoices:
            response_invoices.append(InvoiceResponse(
                id=invoice.id,
                invoice_number=invoice.invoice_number,
                status=invoice.status,
                subtotal=invoice.subtotal,
                tax_amount=invoice.tax_amount,
                total_amount=invoice.total_amount,
                amount_paid=invoice.amount_paid,
                amount_due=invoice.amount_due,
                currency=invoice.currency,
                invoice_date=invoice.invoice_date,
                due_date=invoice.due_date,
                paid_date=invoice.paid_date,
                line_items=invoice.line_items,
                pdf_url=invoice.pdf_url,
                hosted_url=invoice.hosted_url
            ))
        
        return response_invoices
        
    except Exception as e:
        logger.error(f"Failed to list invoices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list invoices")

@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: str,
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """Get invoice details"""
    try:
        invoice = await subscription_manager.get_invoice(invoice_id)
        if not invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")
        
        # Verify invoice belongs to tenant
        if invoice.tenant_id != tenant_context.tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return InvoiceResponse(
            id=invoice.id,
            invoice_number=invoice.invoice_number,
            status=invoice.status,
            subtotal=invoice.subtotal,
            tax_amount=invoice.tax_amount,
            total_amount=invoice.total_amount,
            amount_paid=invoice.amount_paid,
            amount_due=invoice.amount_due,
            currency=invoice.currency,
            invoice_date=invoice.invoice_date,
            due_date=invoice.due_date,
            paid_date=invoice.paid_date,
            line_items=invoice.line_items,
            pdf_url=invoice.pdf_url,
            hosted_url=invoice.hosted_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get invoice: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get invoice")

@router.post("/invoices/{invoice_id}/pay")
async def pay_invoice(
    invoice_id: str,
    payment_request: PaymentRequest,
    tenant_context: TenantContext = Depends(require_tenant_admin)
):
    """Pay an invoice"""
    try:
        # Get invoice
        invoice = await subscription_manager.get_invoice(invoice_id)
        if not invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")
        
        # Verify invoice belongs to tenant
        if invoice.tenant_id != tenant_context.tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if invoice is payable
        if invoice.status != InvoiceStatus.PENDING:
            raise HTTPException(status_code=400, detail="Invoice is not payable")
        
        # Process payment
        payment_result = await subscription_manager.pay_invoice(
            invoice_id, payment_request.payment_method_id
        )
        
        return {
            "message": "Payment processed successfully" if payment_result.status == PaymentStatus.COMPLETED else "Payment failed",
            "payment_id": payment_result.id,
            "status": payment_result.status,
            "amount": float(payment_result.amount),
            "currency": payment_result.currency
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pay invoice: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process payment")

# Usage Tracking

@router.post("/usage")
async def track_usage(
    usage_request: UsageTrackingRequest,
    background_tasks: BackgroundTasks,
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """Track usage for billing"""
    try:
        # Record usage asynchronously
        background_tasks.add_task(
            subscription_manager.record_usage,
            tenant_context.tenant_id,
            usage_request.usage_type,
            usage_request.quantity,
            usage_request.source_id,
            usage_request.source_type,
            usage_request.metadata
        )
        
        return {
            "message": "Usage tracked successfully",
            "usage_type": usage_request.usage_type,
            "quantity": usage_request.quantity
        }
        
    except Exception as e:
        logger.error(f"Failed to track usage: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to track usage")

@router.get("/usage/report", response_model=BillingReport)
async def get_usage_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """Get usage report for billing period"""
    try:
        report = await subscription_manager.get_usage_report(
            tenant_context.tenant_id, start_date, end_date
        )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get usage report")

# Billing Preview

@router.post("/preview", response_model=BillingPreview)
async def preview_plan_change(
    new_plan_id: str,
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """Preview billing impact of plan change"""
    try:
        # Get current subscription
        subscription = await subscription_manager.get_tenant_subscription(tenant_context.tenant_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No subscription found")
        
        # Get new plan
        new_plan = await subscription_manager.get_plan(new_plan_id)
        if not new_plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        # Get current plan
        current_plan = await subscription_manager.get_plan(subscription.plan_id)
        if not current_plan:
            raise HTTPException(status_code=404, detail="Current plan not found")
        
        # Calculate proration (simplified)
        now = datetime.now(timezone.utc)
        days_remaining = (subscription.current_period_end - now).days
        total_days = (subscription.current_period_end - subscription.current_period_start).days
        
        if total_days > 0:
            proration_factor = Decimal(str(days_remaining)) / Decimal(str(total_days))
            
            # Credit for unused portion
            current_credit = subscription.base_amount * proration_factor
            
            # Charge for new plan
            new_charge = new_plan.base_price * proration_factor
            
            # Calculate differences
            immediate_charge = max(Decimal('0'), new_charge - current_credit)
            next_invoice_change = new_plan.base_price - current_plan.base_price
        else:
            immediate_charge = Decimal('0')
            next_invoice_change = new_plan.base_price - current_plan.base_price
        
        # Feature changes
        feature_changes = {}
        current_features = set(current_plan.features)
        new_features = set(new_plan.features)
        
        for feature in current_features.union(new_features):
            if feature in current_features and feature not in new_features:
                feature_changes[feature] = False  # Losing feature
            elif feature not in current_features and feature in new_features:
                feature_changes[feature] = True   # Gaining feature
        
        return BillingPreview(
            current_plan_id=subscription.plan_id,
            new_plan_id=new_plan_id,
            immediate_charge=immediate_charge,
            next_invoice_change=next_invoice_change,
            proration_amount=immediate_charge,
            new_limits=new_plan.limits,
            feature_changes=feature_changes,
            effective_date=now,
            next_billing_date=subscription.current_period_end
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview plan change: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to preview plan change")

# Billing Status

@router.get("/status")
async def get_billing_status(
    tenant_context: TenantContext = Depends(require_tenant_context)
):
    """Get billing status for tenant"""
    try:
        # Get subscription
        subscription = await subscription_manager.get_tenant_subscription(tenant_context.tenant_id)
        
        # Get recent invoices
        recent_invoices = await subscription_manager.list_invoices(
            tenant_id=tenant_context.tenant_id
        )
        recent_invoices = recent_invoices[:5]  # Last 5 invoices
        
        # Get payment methods
        payment_methods = [
            pm for pm in subscription_manager.payment_methods.values()
            if pm.tenant_id == tenant_context.tenant_id and pm.is_active
        ]
        
        # Calculate current usage
        current_usage = subscription.current_usage if subscription else {}
        usage_limits = subscription.usage_limits if subscription else {}
        
        # Calculate usage percentages
        usage_percentages = {}
        for usage_type, current in current_usage.items():
            limit = usage_limits.get(usage_type, 0)
            if limit > 0:
                usage_percentages[usage_type] = (current / limit) * 100
            else:
                usage_percentages[usage_type] = 0
        
        return {
            "subscription": {
                "id": subscription.id if subscription else None,
                "status": subscription.status if subscription else None,
                "plan_id": subscription.plan_id if subscription else None,
                "current_period_end": subscription.current_period_end if subscription else None,
                "auto_renew": subscription.auto_renew if subscription else None
            },
            "payment_methods_count": len(payment_methods),
            "has_default_payment_method": any(pm.is_default for pm in payment_methods),
            "recent_invoices_count": len(recent_invoices),
            "unpaid_invoices_count": len([
                inv for inv in recent_invoices 
                if inv.status in [InvoiceStatus.PENDING, InvoiceStatus.OVERDUE]
            ]),
            "current_usage": current_usage,
            "usage_limits": usage_limits,
            "usage_percentages": usage_percentages,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Failed to get billing status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get billing status")
