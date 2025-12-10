# File: backend/billing/subscription_manager.py
"""
Subscription management with enterprise billing features
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fastapi import HTTPException, status
from decimal import Decimal
import uuid
import logging
import asyncio
from dataclasses import dataclass

from .models import (
    BillingPlan, Subscription, Invoice, Payment, PaymentMethod, UsageRecord,
    SubscriptionCreate, SubscriptionUpdate, PaymentMethodCreate, BillingReport,
    BillingPreview, CreditNote, SubscriptionStatus, PaymentStatus, InvoiceStatus,
    BillingCycle, UsageUnit
)

logger = logging.getLogger(__name__)

@dataclass
class BillingConfig:
    """Billing system configuration"""
    default_currency: str = "USD"
    trial_period_days: int = 14
    invoice_due_days: int = 30
    payment_retry_attempts: int = 3
    late_fee_percentage: Decimal = Decimal('5.0')
    tax_rate: Decimal = Decimal('8.25')  # Default tax rate

class SubscriptionManager:
    """Enterprise subscription and billing management"""
    
    def __init__(self, config: BillingConfig = None):
        self.config = config or BillingConfig()
        
        # In-memory storage for demo (replace with database in production)
        self.plans: Dict[str, BillingPlan] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.payments: Dict[str, Payment] = {}
        self.payment_methods: Dict[str, PaymentMethod] = {}
        self.usage_records: Dict[str, List[UsageRecord]] = {}  # tenant_id -> [records]
        self.credit_notes: Dict[str, CreditNote] = {}
        
        # Initialize default plans
        self._create_default_plans()
    
    def _create_default_plans(self):
        """Create default billing plans"""
        plans = [
            BillingPlan(
                id="basic",
                name="Basic Plan",
                description="Essential features for small teams",
                tier="basic",
                base_price=Decimal('29.00'),
                billing_cycle=BillingCycle.MONTHLY,
                features=[
                    "Up to 5 users",
                    "Basic agent access",
                    "Email support",
                    "5GB storage"
                ],
                limits={
                    "users": 5,
                    "storage_gb": 5,
                    "api_calls_per_month": 10000,
                    "agents": 3
                },
                usage_rates={
                    "api_calls": Decimal('0.001'),
                    "storage_gb": Decimal('2.00'),
                    "users": Decimal('5.00')
                },
                included_usage={
                    "api_calls": 10000,
                    "storage_gb": 5,
                    "users": 5
                }
            ),
            BillingPlan(
                id="pro",
                name="Professional Plan", 
                description="Advanced features for growing businesses",
                tier="pro",
                base_price=Decimal('99.00'),
                billing_cycle=BillingCycle.MONTHLY,
                features=[
                    "Up to 25 users",
                    "Advanced agent features",
                    "Priority support",
                    "50GB storage",
                    "API access",
                    "Custom integrations"
                ],
                limits={
                    "users": 25,
                    "storage_gb": 50,
                    "api_calls_per_month": 100000,
                    "agents": 10
                },
                usage_rates={
                    "api_calls": Decimal('0.0008'),
                    "storage_gb": Decimal('1.50'),
                    "users": Decimal('4.00')
                },
                included_usage={
                    "api_calls": 100000,
                    "storage_gb": 50,
                    "users": 25
                }
            ),
            BillingPlan(
                id="enterprise",
                name="Enterprise Plan",
                description="Full-featured solution for large organizations",
                tier="enterprise",
                base_price=Decimal('299.00'),
                billing_cycle=BillingCycle.MONTHLY,
                features=[
                    "Unlimited users",
                    "All agent features",
                    "24/7 support",
                    "Unlimited storage",
                    "Full API access",
                    "Custom integrations",
                    "White-label options",
                    "SLA guarantee"
                ],
                limits={
                    "users": -1,  # Unlimited
                    "storage_gb": -1,
                    "api_calls_per_month": -1,
                    "agents": -1
                },
                usage_rates={
                    "api_calls": Decimal('0.0005'),
                    "storage_gb": Decimal('1.00'),
                    "users": Decimal('3.00')
                },
                included_usage={
                    "api_calls": 1000000,
                    "storage_gb": 500,
                    "users": 100
                }
            )
        ]
        
        for plan in plans:
            self.plans[plan.id] = plan
        
        logger.info(f"Created {len(plans)} default billing plans")
    
    # Plan Management
    
    async def create_plan(self, plan_data: Dict[str, Any]) -> BillingPlan:
        """Create a new billing plan"""
        plan = BillingPlan(**plan_data)
        self.plans[plan.id] = plan
        logger.info(f"Created billing plan: {plan.id}")
        return plan
    
    async def get_plan(self, plan_id: str) -> Optional[BillingPlan]:
        """Get billing plan by ID"""
        return self.plans.get(plan_id)
    
    async def list_plans(self, include_inactive: bool = False) -> List[BillingPlan]:
        """List available billing plans"""
        plans = list(self.plans.values())
        if not include_inactive:
            plans = [p for p in plans if p.is_active]
        return sorted(plans, key=lambda x: x.base_price)
    
    async def update_plan(self, plan_id: str, updates: Dict[str, Any]) -> BillingPlan:
        """Update billing plan"""
        if plan_id not in self.plans:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found"
            )
        
        plan = self.plans[plan_id]
        for field, value in updates.items():
            if hasattr(plan, field):
                setattr(plan, field, value)
        
        plan.updated_at = datetime.now(timezone.utc)
        self.plans[plan_id] = plan
        return plan
    
    # Subscription Management
    
    async def create_subscription(
        self, 
        tenant_id: str, 
        organization_id: str,
        subscription_data: SubscriptionCreate
    ) -> Subscription:
        """Create new subscription"""
        
        # Validate plan exists
        plan = await self.get_plan(subscription_data.plan_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Billing plan not found"
            )
        
        # Check for existing active subscription
        existing = await self.get_tenant_subscription(tenant_id)
        if existing and existing.status == SubscriptionStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant already has an active subscription"
            )
        
        # Calculate dates
        now = datetime.now(timezone.utc)
        trial_end = None
        if subscription_data.trial_days > 0:
            trial_end = now + timedelta(days=subscription_data.trial_days)
            period_start = trial_end
        else:
            period_start = now
        
        # Calculate period end based on billing cycle
        if plan.billing_cycle == BillingCycle.MONTHLY:
            period_end = period_start + timedelta(days=30)
        elif plan.billing_cycle == BillingCycle.QUARTERLY:
            period_end = period_start + timedelta(days=90)
        elif plan.billing_cycle == BillingCycle.YEARLY:
            period_end = period_start + timedelta(days=365)
        else:
            period_end = period_start + timedelta(days=30)
        
        # Calculate amounts
        base_amount = plan.base_price
        tax_amount = base_amount * (self.config.tax_rate / 100)
        total_amount = base_amount + tax_amount
        
        # Create subscription
        subscription = Subscription(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            organization_id=organization_id,
            plan_id=plan.id,
            status=SubscriptionStatus.TRIAL if trial_end else SubscriptionStatus.ACTIVE,
            current_period_start=period_start,
            current_period_end=period_end,
            trial_end=trial_end,
            base_amount=base_amount,
            tax_amount=tax_amount,
            total_amount=total_amount,
            billing_email=subscription_data.billing_email,
            payment_method_id=subscription_data.payment_method_id,
            usage_limits=plan.limits.copy(),
            metadata=subscription_data.metadata
        )
        
        self.subscriptions[subscription.id] = subscription
        
        # Create initial invoice if not in trial
        if not trial_end:
            await self._create_subscription_invoice(subscription, plan)
        
        logger.info(f"Created subscription {subscription.id} for tenant {tenant_id}")
        return subscription
    
    async def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID"""
        return self.subscriptions.get(subscription_id)
    
    async def get_tenant_subscription(self, tenant_id: str) -> Optional[Subscription]:
        """Get active subscription for tenant"""
        for subscription in self.subscriptions.values():
            if (subscription.tenant_id == tenant_id and 
                subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]):
                return subscription
        return None
    
    async def update_subscription(
        self, 
        subscription_id: str, 
        updates: SubscriptionUpdate
    ) -> Subscription:
        """Update subscription"""
        if subscription_id not in self.subscriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found"
            )
        
        subscription = self.subscriptions[subscription_id]
        
        # Handle plan changes
        if updates.plan_id and updates.plan_id != subscription.plan_id:
            await self._change_subscription_plan(subscription, updates.plan_id)
        
        # Update other fields
        update_data = updates.dict(exclude_unset=True, exclude={'plan_id'})
        for field, value in update_data.items():
            if hasattr(subscription, field):
                setattr(subscription, field, value)
        
        subscription.updated_at = datetime.now(timezone.utc)
        self.subscriptions[subscription_id] = subscription
        
        return subscription
    
    async def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> Subscription:
        """Cancel subscription"""
        if subscription_id not in self.subscriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found"
            )
        
        subscription = self.subscriptions[subscription_id]
        
        if immediate:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.current_period_end = datetime.now(timezone.utc)
        else:
            # Cancel at end of current period
            subscription.auto_renew = False
        
        subscription.cancelled_at = datetime.now(timezone.utc)
        subscription.updated_at = datetime.now(timezone.utc)
        self.subscriptions[subscription_id] = subscription
        
        logger.info(f"Cancelled subscription {subscription_id}")
        return subscription
    
    # Invoice Management
    
    async def create_invoice(
        self, 
        subscription_id: str, 
        line_items: List[Dict[str, Any]] = None
    ) -> Invoice:
        """Create invoice for subscription"""
        subscription = await self.get_subscription(subscription_id)
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found"
            )
        
        plan = await self.get_plan(subscription.plan_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Billing plan not found"
            )
        
        # Calculate amounts
        subtotal = subscription.base_amount
        
        # Add usage charges if line items provided
        if line_items:
            for item in line_items:
                subtotal += Decimal(str(item.get('amount', 0)))
        
        # Calculate tax and total
        tax_amount = subtotal * (self.config.tax_rate / 100)
        total_amount = subtotal + tax_amount - subscription.discount_amount
        
        # Create invoice
        invoice = Invoice(
            id=str(uuid.uuid4()),
            subscription_id=subscription_id,
            tenant_id=subscription.tenant_id,
            organization_id=subscription.organization_id,
            status=InvoiceStatus.PENDING,
            subtotal=subtotal,
            tax_amount=tax_amount,
            total_amount=total_amount,
            amount_due=total_amount,
            currency=subscription.currency,
            line_items=line_items or [],
            invoice_date=datetime.now(timezone.utc),
            due_date=datetime.now(timezone.utc) + timedelta(days=self.config.invoice_due_days),
            customer_email=subscription.billing_email
        )
        
        self.invoices[invoice.id] = invoice
        logger.info(f"Created invoice {invoice.id} for subscription {subscription_id}")
        return invoice
    
    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID"""
        return self.invoices.get(invoice_id)
    
    async def list_invoices(
        self, 
        tenant_id: str = None,
        subscription_id: str = None,
        status: InvoiceStatus = None
    ) -> List[Invoice]:
        """List invoices with optional filters"""
        invoices = list(self.invoices.values())
        
        if tenant_id:
            invoices = [i for i in invoices if i.tenant_id == tenant_id]
        
        if subscription_id:
            invoices = [i for i in invoices if i.subscription_id == subscription_id]
        
        if status:
            invoices = [i for i in invoices if i.status == status]
        
        return sorted(invoices, key=lambda x: x.created_at, reverse=True)
    
    async def pay_invoice(self, invoice_id: str, payment_method_id: str) -> Payment:
        """Process payment for invoice"""
        invoice = await self.get_invoice(invoice_id)
        if not invoice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        # Create payment record
        payment = Payment(
            id=str(uuid.uuid4()),
            invoice_id=invoice_id,
            subscription_id=invoice.subscription_id,
            tenant_id=invoice.tenant_id,
            organization_id=invoice.organization_id,
            amount=invoice.amount_due,
            currency=invoice.currency,
            status=PaymentStatus.PROCESSING,
            payment_method_id=payment_method_id,
            description=f"Payment for invoice {invoice.invoice_number}"
        )
        
        # Simulate payment processing (integrate with real payment processor)
        success = await self._process_payment(payment)
        
        if success:
            payment.status = PaymentStatus.COMPLETED
            payment.processed_at = datetime.now(timezone.utc)
            
            # Update invoice
            invoice.status = InvoiceStatus.PAID
            invoice.amount_paid = invoice.total_amount
            invoice.amount_due = Decimal('0.00')
            invoice.paid_date = datetime.now(timezone.utc)
            self.invoices[invoice_id] = invoice
            
        else:
            payment.status = PaymentStatus.FAILED
            payment.failure_message = "Payment processing failed"
            
            # Update invoice payment attempts
            invoice.payment_attempts += 1
            invoice.last_payment_error = "Payment failed"
            self.invoices[invoice_id] = invoice
        
        self.payments[payment.id] = payment
        return payment
    
    # Usage Tracking
    
    async def record_usage(
        self,
        tenant_id: str,
        usage_type: UsageUnit,
        quantity: int,
        source_id: str = None,
        source_type: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Record usage for billing"""
        subscription = await self.get_tenant_subscription(tenant_id)
        if not subscription:
            logger.warning(f"No subscription found for tenant {tenant_id}")
            return
        
        usage_record = UsageRecord(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            subscription_id=subscription.id,
            usage_type=usage_type,
            quantity=quantity,
            usage_date=datetime.now(timezone.utc),
            billing_period_start=subscription.current_period_start,
            billing_period_end=subscription.current_period_end,
            source_id=source_id,
            source_type=source_type,
            metadata=metadata or {}
        )
        
        if tenant_id not in self.usage_records:
            self.usage_records[tenant_id] = []
        
        self.usage_records[tenant_id].append(usage_record)
        
        # Update subscription current usage
        usage_key = usage_type.value
        if usage_key not in subscription.current_usage:
            subscription.current_usage[usage_key] = 0
        
        subscription.current_usage[usage_key] += quantity
        self.subscriptions[subscription.id] = subscription
    
    async def get_usage_report(self, tenant_id: str, period_start: datetime, period_end: datetime) -> BillingReport:
        """Generate usage report for billing period"""
        subscription = await self.get_tenant_subscription(tenant_id)
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found for tenant"
            )
        
        plan = await self.get_plan(subscription.plan_id)
        
        # Get usage records for period
        tenant_usage = self.usage_records.get(tenant_id, [])
        period_usage = [
            r for r in tenant_usage 
            if period_start <= r.usage_date <= period_end
        ]
        
        # Calculate usage costs
        usage_breakdown = {}
        total_usage_cost = Decimal('0.00')
        
        for usage_type in UsageUnit:
            type_usage = [r for r in period_usage if r.usage_type == usage_type]
            total_quantity = sum(r.quantity for r in type_usage)
            
            if total_quantity > 0:
                # Get included amount and rate
                included = plan.included_usage.get(usage_type.value, 0)
                rate = plan.usage_rates.get(usage_type.value, Decimal('0'))
                
                # Calculate billable usage (above included amount)
                billable_quantity = max(0, total_quantity - included)
                cost = Decimal(str(billable_quantity)) * rate
                
                usage_breakdown[usage_type.value] = {
                    "total_quantity": total_quantity,
                    "included_quantity": included,
                    "billable_quantity": billable_quantity,
                    "unit_rate": float(rate),
                    "cost": float(cost)
                }
                
                total_usage_cost += cost
        
        return BillingReport(
            tenant_id=tenant_id,
            subscription_id=subscription.id,
            period_start=period_start,
            period_end=period_end,
            total_base_cost=subscription.base_amount,
            total_usage_cost=total_usage_cost,
            total_cost=subscription.base_amount + total_usage_cost,
            usage_breakdown=usage_breakdown,
            active_users=subscription.current_usage.get("users", 0),
            total_api_calls=subscription.current_usage.get("api_calls", 0),
            total_storage_gb=subscription.current_usage.get("storage_gb", 0.0)
        )
    
    # Internal Helper Methods
    
    async def _create_subscription_invoice(self, subscription: Subscription, plan: BillingPlan) -> Invoice:
        """Create initial invoice for subscription"""
        line_items = [
            {
                "description": f"{plan.name} - {plan.billing_cycle.value} subscription",
                "quantity": 1,
                "unit_price": float(plan.base_price),
                "amount": float(plan.base_price)
            }
        ]
        
        return await self.create_invoice(subscription.id, line_items)
    
    async def _change_subscription_plan(self, subscription: Subscription, new_plan_id: str):
        """Handle subscription plan change with proration"""
        new_plan = await self.get_plan(new_plan_id)
        if not new_plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="New plan not found"
            )
        
        old_plan = await self.get_plan(subscription.plan_id)
        
        # Calculate proration
        now = datetime.now(timezone.utc)
        days_remaining = (subscription.current_period_end - now).days
        total_days = (subscription.current_period_end - subscription.current_period_start).days
        
        if total_days > 0:
            proration_factor = Decimal(str(days_remaining)) / Decimal(str(total_days))
            
            # Credit for unused portion of old plan
            old_credit = subscription.base_amount * proration_factor
            
            # Charge for new plan
            new_charge = new_plan.base_price * proration_factor
            
            # Calculate difference
            proration_amount = new_charge - old_credit
            
            # Update subscription
            subscription.plan_id = new_plan_id
            subscription.base_amount = new_plan.base_price
            subscription.usage_limits = new_plan.limits.copy()
            
            # Create proration invoice if needed
            if proration_amount > 0:
                line_items = [
                    {
                        "description": f"Plan change proration ({old_plan.name} to {new_plan.name})",
                        "quantity": 1,
                        "unit_price": float(proration_amount),
                        "amount": float(proration_amount)
                    }
                ]
                await self.create_invoice(subscription.id, line_items)
    
    async def _process_payment(self, payment: Payment) -> bool:
        """Simulate payment processing (integrate with real payment processor)"""
        # In production, this would integrate with Stripe, PayPal, etc.
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simulate 95% success rate
        import random
        return random.random() < 0.95

# Global subscription manager instance
subscription_manager = SubscriptionManager()