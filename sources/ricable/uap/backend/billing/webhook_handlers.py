# File: backend/billing/webhook_handlers.py
"""
Webhook handlers for payment processor events
"""

from fastapi import APIRouter, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime, timezone

from .payment_processors import payment_processor_manager
from .subscription_manager import subscription_manager
from .models import PaymentStatus, SubscriptionStatus, InvoiceStatus
from ..tenancy.organization_manager import organization_manager
from ..monitoring.logs.logger import uap_logger, EventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks/billing", tags=["Billing Webhooks"])

class WebhookEventProcessor:
    """Process webhook events from payment processors"""
    
    async def process_stripe_event(self, event_data: Dict[str, Any]):
        """Process Stripe webhook event"""
        event_type = event_data.get("type")
        event_object = event_data.get("data", {}).get("object", {})
        
        logger.info(f"Processing Stripe event: {event_type}")
        
        try:
            if event_type == "payment_intent.succeeded":
                await self._handle_payment_succeeded("stripe", event_object)
            elif event_type == "payment_intent.payment_failed":
                await self._handle_payment_failed("stripe", event_object)
            elif event_type == "customer.subscription.created":
                await self._handle_subscription_created("stripe", event_object)
            elif event_type == "customer.subscription.updated":
                await self._handle_subscription_updated("stripe", event_object)
            elif event_type == "customer.subscription.deleted":
                await self._handle_subscription_cancelled("stripe", event_object)
            elif event_type == "invoice.payment_succeeded":
                await self._handle_invoice_paid("stripe", event_object)
            elif event_type == "invoice.payment_failed":
                await self._handle_invoice_payment_failed("stripe", event_object)
            else:
                logger.info(f"Unhandled Stripe event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing Stripe event {event_type}: {str(e)}")
            raise
    
    async def process_paypal_event(self, event_data: Dict[str, Any]):
        """Process PayPal webhook event"""
        event_type = event_data.get("event_type")
        resource = event_data.get("resource", {})
        
        logger.info(f"Processing PayPal event: {event_type}")
        
        try:
            if event_type == "PAYMENT.CAPTURE.COMPLETED":
                await self._handle_payment_succeeded("paypal", resource)
            elif event_type == "PAYMENT.CAPTURE.DENIED":
                await self._handle_payment_failed("paypal", resource)
            elif event_type == "BILLING.SUBSCRIPTION.CREATED":
                await self._handle_subscription_created("paypal", resource)
            elif event_type == "BILLING.SUBSCRIPTION.UPDATED":
                await self._handle_subscription_updated("paypal", resource)
            elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
                await self._handle_subscription_cancelled("paypal", resource)
            else:
                logger.info(f"Unhandled PayPal event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing PayPal event {event_type}: {str(e)}")
            raise
    
    async def _handle_payment_succeeded(self, provider: str, payment_data: Dict[str, Any]):
        """Handle successful payment"""
        try:
            # Extract payment information
            transaction_id = payment_data.get("id")
            amount = payment_data.get("amount")
            currency = payment_data.get("currency", "USD")
            
            # Find corresponding payment in our system
            for payment in subscription_manager.payments.values():
                if payment.gateway_transaction_id == transaction_id:
                    # Update payment status
                    payment.status = PaymentStatus.COMPLETED
                    payment.processed_at = datetime.now(timezone.utc)
                    subscription_manager.payments[payment.id] = payment
                    
                    # Update invoice if linked
                    if payment.invoice_id:
                        invoice = subscription_manager.invoices.get(payment.invoice_id)
                        if invoice:
                            invoice.status = InvoiceStatus.PAID
                            invoice.amount_paid = payment.amount
                            invoice.amount_due = invoice.total_amount - invoice.amount_paid
                            invoice.paid_date = datetime.now(timezone.utc)
                            subscription_manager.invoices[payment.invoice_id] = invoice
                    
                    # Log successful payment
                    uap_logger.log_billing_event(
                        f"Payment successful: {payment.id}",
                        "webhook_handler",
                        {
                            "payment_id": payment.id,
                            "tenant_id": payment.tenant_id,
                            "amount": float(payment.amount),
                            "provider": provider
                        }
                    )
                    
                    break
            
        except Exception as e:
            logger.error(f"Error handling payment success: {str(e)}")
            raise
    
    async def _handle_payment_failed(self, provider: str, payment_data: Dict[str, Any]):
        """Handle failed payment"""
        try:
            transaction_id = payment_data.get("id")
            error_message = payment_data.get("last_payment_error", {}).get("message", "Payment failed")
            
            # Find corresponding payment in our system
            for payment in subscription_manager.payments.values():
                if payment.gateway_transaction_id == transaction_id:
                    # Update payment status
                    payment.status = PaymentStatus.FAILED
                    payment.failure_message = error_message
                    subscription_manager.payments[payment.id] = payment
                    
                    # Update invoice if linked
                    if payment.invoice_id:
                        invoice = subscription_manager.invoices.get(payment.invoice_id)
                        if invoice:
                            invoice.payment_attempts += 1
                            invoice.last_payment_error = error_message
                            subscription_manager.invoices[payment.invoice_id] = invoice
                    
                    # Log failed payment
                    uap_logger.log_billing_event(
                        f"Payment failed: {payment.id}",
                        "webhook_handler",
                        {
                            "payment_id": payment.id,
                            "tenant_id": payment.tenant_id,
                            "error": error_message,
                            "provider": provider
                        }
                    )
                    
                    break
            
        except Exception as e:
            logger.error(f"Error handling payment failure: {str(e)}")
            raise
    
    async def _handle_subscription_created(self, provider: str, subscription_data: Dict[str, Any]):
        """Handle subscription creation"""
        try:
            # This would typically be called when a subscription is created
            # directly in the payment processor (not through our API)
            logger.info(f"External subscription created in {provider}: {subscription_data.get('id')}")
            
        except Exception as e:
            logger.error(f"Error handling subscription creation: {str(e)}")
            raise
    
    async def _handle_subscription_updated(self, provider: str, subscription_data: Dict[str, Any]):
        """Handle subscription update"""
        try:
            external_sub_id = subscription_data.get("id")
            status = subscription_data.get("status")
            
            # Find subscription by external ID
            for subscription in subscription_manager.subscriptions.values():
                # You'd store the external subscription ID in metadata
                if subscription.metadata.get(f"{provider}_subscription_id") == external_sub_id:
                    # Update subscription status based on provider status
                    if status in ["active", "trialing"]:
                        subscription.status = SubscriptionStatus.ACTIVE
                    elif status in ["past_due"]:
                        subscription.status = SubscriptionStatus.PAST_DUE
                    elif status in ["canceled", "cancelled"]:
                        subscription.status = SubscriptionStatus.CANCELLED
                    elif status in ["unpaid"]:
                        subscription.status = SubscriptionStatus.SUSPENDED
                    
                    subscription.updated_at = datetime.now(timezone.utc)
                    subscription_manager.subscriptions[subscription.id] = subscription
                    
                    logger.info(f"Updated subscription {subscription.id} status to {subscription.status}")
                    break
            
        except Exception as e:
            logger.error(f"Error handling subscription update: {str(e)}")
            raise
    
    async def _handle_subscription_cancelled(self, provider: str, subscription_data: Dict[str, Any]):
        """Handle subscription cancellation"""
        try:
            external_sub_id = subscription_data.get("id")
            
            # Find and cancel subscription
            for subscription in subscription_manager.subscriptions.values():
                if subscription.metadata.get(f"{provider}_subscription_id") == external_sub_id:
                    subscription.status = SubscriptionStatus.CANCELLED
                    subscription.cancelled_at = datetime.now(timezone.utc)
                    subscription.updated_at = datetime.now(timezone.utc)
                    subscription_manager.subscriptions[subscription.id] = subscription
                    
                    # Log cancellation
                    uap_logger.log_billing_event(
                        f"Subscription cancelled: {subscription.id}",
                        "webhook_handler",
                        {
                            "subscription_id": subscription.id,
                            "tenant_id": subscription.tenant_id,
                            "provider": provider
                        }
                    )
                    
                    break
            
        except Exception as e:
            logger.error(f"Error handling subscription cancellation: {str(e)}")
            raise
    
    async def _handle_invoice_paid(self, provider: str, invoice_data: Dict[str, Any]):
        """Handle invoice payment"""
        try:
            # Similar to payment success handling
            await self._handle_payment_succeeded(provider, invoice_data)
            
        except Exception as e:
            logger.error(f"Error handling invoice payment: {str(e)}")
            raise
    
    async def _handle_invoice_payment_failed(self, provider: str, invoice_data: Dict[str, Any]):
        """Handle invoice payment failure"""
        try:
            # Similar to payment failure handling
            await self._handle_payment_failed(provider, invoice_data)
            
        except Exception as e:
            logger.error(f"Error handling invoice payment failure: {str(e)}")
            raise

# Global event processor
event_processor = WebhookEventProcessor()

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle Stripe webhook events"""
    try:
        # Get request body and headers
        body = await request.body()
        signature = request.headers.get("stripe-signature")
        
        if not signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing Stripe signature"
            )
        
        # Verify webhook signature
        is_valid = await payment_processor_manager.verify_webhook(
            "stripe", body.decode(), signature
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Stripe signature"
            )
        
        # Parse event data
        try:
            event_data = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload"
            )
        
        # Process event in background
        background_tasks.add_task(
            event_processor.process_stripe_event,
            event_data
        )
        
        return JSONResponse(
            status_code=200,
            content={"received": True, "event_id": event_data.get("id")}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Stripe webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )

@router.post("/paypal")
async def paypal_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle PayPal webhook events"""
    try:
        # Get request body and headers
        body = await request.body()
        
        # PayPal webhook verification would go here
        # For now, we'll skip verification in mock implementation
        
        # Parse event data
        try:
            event_data = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload"
            )
        
        # Process event in background
        background_tasks.add_task(
            event_processor.process_paypal_event,
            event_data
        )
        
        return JSONResponse(
            status_code=200,
            content={"received": True, "event_id": event_data.get("id")}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PayPal webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )

@router.post("/mock")
async def mock_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle mock webhook events for testing"""
    try:
        body = await request.body()
        
        try:
            event_data = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload"
            )
        
        # Log mock webhook event
        logger.info(f"Mock webhook event: {event_data.get('type', 'unknown')}")
        
        # Process mock events (for testing)
        event_type = event_data.get("type")
        if event_type == "payment.succeeded":
            await event_processor._handle_payment_succeeded("mock", event_data.get("data", {}))
        elif event_type == "payment.failed":
            await event_processor._handle_payment_failed("mock", event_data.get("data", {}))
        
        return JSONResponse(
            status_code=200,
            content={"received": True, "event_id": event_data.get("id", "mock")}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing mock webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )

@router.get("/status")
async def webhook_status():
    """Get webhook handler status"""
    return {
        "status": "healthy",
        "processors": payment_processor_manager.list_processors(),
        "timestamp": datetime.now(timezone.utc)
    }
