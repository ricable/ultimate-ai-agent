# File: backend/billing/payment_processors.py
"""
Payment processor integrations for enterprise billing
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from decimal import Decimal
import logging
import os
import hashlib
import hmac
import asyncio
import json
from fastapi import HTTPException, status
from pydantic import BaseModel

from .models import Payment, PaymentMethod, PaymentStatus

logger = logging.getLogger(__name__)

class PaymentProcessorError(Exception):
    """Payment processor specific error"""
    pass

class PaymentResult(BaseModel):
    """Payment processing result"""
    success: bool
    transaction_id: Optional[str] = None
    gateway_transaction_id: Optional[str] = None
    amount_charged: Optional[Decimal] = None
    currency: str = "USD"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}

class PaymentProcessor(ABC):
    """Abstract base class for payment processors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = "base"
    
    @abstractmethod
    async def create_payment_method(
        self, 
        customer_data: Dict[str, Any], 
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create payment method for customer"""
        pass
    
    @abstractmethod
    async def charge_payment_method(
        self, 
        payment_method_id: str, 
        amount: Decimal, 
        currency: str = "USD",
        metadata: Dict[str, Any] = None
    ) -> PaymentResult:
        """Charge a payment method"""
        pass
    
    @abstractmethod
    async def refund_payment(
        self, 
        transaction_id: str, 
        amount: Optional[Decimal] = None
    ) -> PaymentResult:
        """Refund a payment"""
        pass
    
    @abstractmethod
    async def get_payment_status(self, transaction_id: str) -> PaymentStatus:
        """Get payment status"""
        pass
    
    @abstractmethod
    async def create_webhook_endpoint(self, url: str, events: List[str]) -> str:
        """Create webhook endpoint"""
        pass
    
    @abstractmethod
    async def verify_webhook(self, payload: str, signature: str) -> bool:
        """Verify webhook signature"""
        pass

class StripeProcessor(PaymentProcessor):
    """Stripe payment processor integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "stripe"
        self.api_key = config.get("api_key")
        self.webhook_secret = config.get("webhook_secret")
        self.api_version = "2023-10-16"
        
        if not self.api_key:
            raise PaymentProcessorError("Stripe API key is required")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Stripe API"""
        # In production, use actual HTTP client (e.g., httpx)
        # This is a mock implementation
        
        logger.info(f"Mock Stripe API request: {method} {endpoint}")
        
        # Simulate successful responses for demo
        if endpoint == "payment_methods":
            return {
                "id": f"pm_mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "card",
                "card": {
                    "brand": "visa",
                    "last4": "4242",
                    "exp_month": 12,
                    "exp_year": 2025
                },
                "created": int(datetime.now().timestamp())
            }
        elif endpoint == "payment_intents":
            return {
                "id": f"pi_mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "status": "succeeded",
                "amount": data.get("amount", 0),
                "currency": data.get("currency", "usd"),
                "created": int(datetime.now().timestamp())
            }
        elif endpoint.startswith("payment_intents/") and endpoint.endswith("/refund"):
            return {
                "id": f"re_mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "status": "succeeded",
                "amount": data.get("amount", 0),
                "created": int(datetime.now().timestamp())
            }
        
        return {"success": True}
    
    async def create_payment_method(
        self, 
        customer_data: Dict[str, Any], 
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Stripe payment method"""
        try:
            # Create customer first if needed
            customer_id = customer_data.get("stripe_customer_id")
            if not customer_id:
                customer_response = await self._make_request(
                    "POST", 
                    "customers",
                    {
                        "email": customer_data.get("email"),
                        "name": customer_data.get("name"),
                        "metadata": {
                            "tenant_id": customer_data.get("tenant_id"),
                            "organization_id": customer_data.get("organization_id")
                        }
                    }
                )
                customer_id = customer_response["id"]
            
            # Create payment method
            pm_response = await self._make_request(
                "POST",
                "payment_methods",
                {
                    "type": payment_data.get("type", "card"),
                    "card": payment_data.get("card", {}),
                    "customer": customer_id
                }
            )
            
            return {
                "payment_method_id": pm_response["id"],
                "customer_id": customer_id,
                "type": pm_response["type"],
                "card_last4": pm_response.get("card", {}).get("last4"),
                "card_brand": pm_response.get("card", {}).get("brand"),
                "card_exp_month": pm_response.get("card", {}).get("exp_month"),
                "card_exp_year": pm_response.get("card", {}).get("exp_year")
            }
            
        except Exception as e:
            logger.error(f"Stripe payment method creation failed: {str(e)}")
            raise PaymentProcessorError(f"Failed to create payment method: {str(e)}")
    
    async def charge_payment_method(
        self, 
        payment_method_id: str, 
        amount: Decimal, 
        currency: str = "USD",
        metadata: Dict[str, Any] = None
    ) -> PaymentResult:
        """Charge Stripe payment method"""
        try:
            # Convert amount to cents
            amount_cents = int(amount * 100)
            
            response = await self._make_request(
                "POST",
                "payment_intents",
                {
                    "amount": amount_cents,
                    "currency": currency.lower(),
                    "payment_method": payment_method_id,
                    "confirmation_method": "manual",
                    "confirm": True,
                    "metadata": metadata or {}
                }
            )
            
            return PaymentResult(
                success=response["status"] == "succeeded",
                transaction_id=response["id"],
                gateway_transaction_id=response["id"],
                amount_charged=Decimal(str(response["amount"])) / 100,
                currency=response["currency"].upper(),
                metadata=response.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Stripe charge failed: {str(e)}")
            return PaymentResult(
                success=False,
                error_code="charge_failed",
                error_message=str(e)
            )
    
    async def refund_payment(
        self, 
        transaction_id: str, 
        amount: Optional[Decimal] = None
    ) -> PaymentResult:
        """Refund Stripe payment"""
        try:
            refund_data = {"payment_intent": transaction_id}
            
            if amount:
                refund_data["amount"] = int(amount * 100)
            
            response = await self._make_request(
                "POST",
                f"payment_intents/{transaction_id}/refund",
                refund_data
            )
            
            return PaymentResult(
                success=response["status"] == "succeeded",
                transaction_id=response["id"],
                gateway_transaction_id=response["id"],
                amount_charged=Decimal(str(response["amount"])) / 100 if amount else None
            )
            
        except Exception as e:
            logger.error(f"Stripe refund failed: {str(e)}")
            return PaymentResult(
                success=False,
                error_code="refund_failed",
                error_message=str(e)
            )
    
    async def get_payment_status(self, transaction_id: str) -> PaymentStatus:
        """Get Stripe payment status"""
        try:
            response = await self._make_request(
                "GET",
                f"payment_intents/{transaction_id}"
            )
            
            stripe_status = response.get("status", "unknown")
            
            # Map Stripe statuses to our enum
            status_mapping = {
                "requires_payment_method": PaymentStatus.PENDING,
                "requires_confirmation": PaymentStatus.PENDING,
                "requires_action": PaymentStatus.PENDING,
                "processing": PaymentStatus.PROCESSING,
                "succeeded": PaymentStatus.COMPLETED,
                "canceled": PaymentStatus.CANCELLED,
                "payment_failed": PaymentStatus.FAILED
            }
            
            return status_mapping.get(stripe_status, PaymentStatus.FAILED)
            
        except Exception as e:
            logger.error(f"Failed to get Stripe payment status: {str(e)}")
            return PaymentStatus.FAILED
    
    async def create_webhook_endpoint(self, url: str, events: List[str]) -> str:
        """Create Stripe webhook endpoint"""
        try:
            response = await self._make_request(
                "POST",
                "webhook_endpoints",
                {
                    "url": url,
                    "enabled_events": events
                }
            )
            
            return response["id"]
            
        except Exception as e:
            logger.error(f"Failed to create Stripe webhook: {str(e)}")
            raise PaymentProcessorError(f"Failed to create webhook: {str(e)}")
    
    async def verify_webhook(self, payload: str, signature: str) -> bool:
        """Verify Stripe webhook signature"""
        try:
            if not self.webhook_secret:
                logger.warning("No webhook secret configured for Stripe")
                return False
            
            # Extract timestamp and signature from header
            elements = signature.split(",")
            timestamp = None
            v1_signature = None
            
            for element in elements:
                if element.startswith("t="):
                    timestamp = element[2:]
                elif element.startswith("v1="):
                    v1_signature = element[3:]
            
            if not timestamp or not v1_signature:
                return False
            
            # Create expected signature
            signed_payload = f"{timestamp}.{payload}"
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                signed_payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, v1_signature)
            
        except Exception as e:
            logger.error(f"Stripe webhook verification failed: {str(e)}")
            return False

class PayPalProcessor(PaymentProcessor):
    """PayPal payment processor integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "paypal"
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.sandbox = config.get("sandbox", True)
        
        if not self.client_id or not self.client_secret:
            raise PaymentProcessorError("PayPal credentials are required")
    
    async def _get_access_token(self) -> str:
        """Get PayPal access token"""
        # Mock implementation - in production use actual PayPal API
        return f"mock_paypal_token_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    async def create_payment_method(
        self, 
        customer_data: Dict[str, Any], 
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create PayPal payment method (saved payment source)"""
        try:
            access_token = await self._get_access_token()
            
            # Mock PayPal payment method creation
            return {
                "payment_method_id": f"paypal_pm_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "paypal",
                "paypal_email": payment_data.get("paypal_email", customer_data.get("email"))
            }
            
        except Exception as e:
            logger.error(f"PayPal payment method creation failed: {str(e)}")
            raise PaymentProcessorError(f"Failed to create PayPal payment method: {str(e)}")
    
    async def charge_payment_method(
        self, 
        payment_method_id: str, 
        amount: Decimal, 
        currency: str = "USD",
        metadata: Dict[str, Any] = None
    ) -> PaymentResult:
        """Charge PayPal payment method"""
        try:
            access_token = await self._get_access_token()
            
            # Mock PayPal payment processing
            transaction_id = f"paypal_txn_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            return PaymentResult(
                success=True,
                transaction_id=transaction_id,
                gateway_transaction_id=transaction_id,
                amount_charged=amount,
                currency=currency,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"PayPal charge failed: {str(e)}")
            return PaymentResult(
                success=False,
                error_code="paypal_charge_failed",
                error_message=str(e)
            )
    
    async def refund_payment(
        self, 
        transaction_id: str, 
        amount: Optional[Decimal] = None
    ) -> PaymentResult:
        """Refund PayPal payment"""
        try:
            access_token = await self._get_access_token()
            
            # Mock PayPal refund
            refund_id = f"paypal_refund_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            return PaymentResult(
                success=True,
                transaction_id=refund_id,
                gateway_transaction_id=refund_id,
                amount_charged=amount
            )
            
        except Exception as e:
            logger.error(f"PayPal refund failed: {str(e)}")
            return PaymentResult(
                success=False,
                error_code="paypal_refund_failed",
                error_message=str(e)
            )
    
    async def get_payment_status(self, transaction_id: str) -> PaymentStatus:
        """Get PayPal payment status"""
        try:
            # Mock status check - in production query PayPal API
            return PaymentStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Failed to get PayPal payment status: {str(e)}")
            return PaymentStatus.FAILED
    
    async def create_webhook_endpoint(self, url: str, events: List[str]) -> str:
        """Create PayPal webhook"""
        try:
            # Mock webhook creation
            return f"paypal_webhook_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        except Exception as e:
            logger.error(f"Failed to create PayPal webhook: {str(e)}")
            raise PaymentProcessorError(f"Failed to create webhook: {str(e)}")
    
    async def verify_webhook(self, payload: str, signature: str) -> bool:
        """Verify PayPal webhook signature"""
        try:
            # Mock verification - in production use PayPal's verification process
            return True
            
        except Exception as e:
            logger.error(f"PayPal webhook verification failed: {str(e)}")
            return False

class MockProcessor(PaymentProcessor):
    """Mock payment processor for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "mock"
        self.success_rate = config.get("success_rate", 0.95)
    
    async def create_payment_method(
        self, 
        customer_data: Dict[str, Any], 
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create mock payment method"""
        return {
            "payment_method_id": f"mock_pm_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": "card",
            "card_last4": "1234",
            "card_brand": "test"
        }
    
    async def charge_payment_method(
        self, 
        payment_method_id: str, 
        amount: Decimal, 
        currency: str = "USD",
        metadata: Dict[str, Any] = None
    ) -> PaymentResult:
        """Mock charge payment method"""
        # Simulate success/failure based on success rate
        import random
        success = random.random() < self.success_rate
        
        if success:
            return PaymentResult(
                success=True,
                transaction_id=f"mock_txn_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                gateway_transaction_id=f"mock_gateway_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                amount_charged=amount,
                currency=currency
            )
        else:
            return PaymentResult(
                success=False,
                error_code="mock_failure",
                error_message="Simulated payment failure"
            )
    
    async def refund_payment(
        self, 
        transaction_id: str, 
        amount: Optional[Decimal] = None
    ) -> PaymentResult:
        """Mock refund payment"""
        return PaymentResult(
            success=True,
            transaction_id=f"mock_refund_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            amount_charged=amount
        )
    
    async def get_payment_status(self, transaction_id: str) -> PaymentStatus:
        """Mock payment status"""
        return PaymentStatus.COMPLETED
    
    async def create_webhook_endpoint(self, url: str, events: List[str]) -> str:
        """Mock webhook creation"""
        return f"mock_webhook_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    async def verify_webhook(self, payload: str, signature: str) -> bool:
        """Mock webhook verification"""
        return True

class PaymentProcessorManager:
    """Manager for multiple payment processors"""
    
    def __init__(self):
        self.processors: Dict[str, PaymentProcessor] = {}
        self.default_processor = "mock"
        
        # Initialize processors from environment
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize payment processors from configuration"""
        # Mock processor (always available for testing)
        mock_config = {"success_rate": 0.95}
        self.processors["mock"] = MockProcessor(mock_config)
        
        # Stripe processor
        stripe_api_key = os.getenv("STRIPE_API_KEY")
        if stripe_api_key:
            stripe_config = {
                "api_key": stripe_api_key,
                "webhook_secret": os.getenv("STRIPE_WEBHOOK_SECRET")
            }
            try:
                self.processors["stripe"] = StripeProcessor(stripe_config)
                self.default_processor = "stripe"
                logger.info("Stripe payment processor initialized")
            except PaymentProcessorError as e:
                logger.error(f"Failed to initialize Stripe: {e}")
        
        # PayPal processor
        paypal_client_id = os.getenv("PAYPAL_CLIENT_ID")
        paypal_client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
        if paypal_client_id and paypal_client_secret:
            paypal_config = {
                "client_id": paypal_client_id,
                "client_secret": paypal_client_secret,
                "sandbox": os.getenv("PAYPAL_SANDBOX", "true").lower() == "true"
            }
            try:
                self.processors["paypal"] = PayPalProcessor(paypal_config)
                logger.info("PayPal payment processor initialized")
            except PaymentProcessorError as e:
                logger.error(f"Failed to initialize PayPal: {e}")
        
        logger.info(f"Initialized {len(self.processors)} payment processors")
        logger.info(f"Default processor: {self.default_processor}")
    
    def get_processor(self, provider: str = None) -> PaymentProcessor:
        """Get payment processor by provider name"""
        provider = provider or self.default_processor
        
        if provider not in self.processors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment processor '{provider}' not available"
            )
        
        return self.processors[provider]
    
    def list_processors(self) -> List[str]:
        """List available payment processors"""
        return list(self.processors.keys())
    
    async def create_payment_method(
        self,
        provider: str,
        customer_data: Dict[str, Any],
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create payment method using specified processor"""
        processor = self.get_processor(provider)
        return await processor.create_payment_method(customer_data, payment_data)
    
    async def process_payment(
        self,
        provider: str,
        payment_method_id: str,
        amount: Decimal,
        currency: str = "USD",
        metadata: Dict[str, Any] = None
    ) -> PaymentResult:
        """Process payment using specified processor"""
        processor = self.get_processor(provider)
        return await processor.charge_payment_method(
            payment_method_id, amount, currency, metadata
        )
    
    async def refund_payment(
        self,
        provider: str,
        transaction_id: str,
        amount: Optional[Decimal] = None
    ) -> PaymentResult:
        """Refund payment using specified processor"""
        processor = self.get_processor(provider)
        return await processor.refund_payment(transaction_id, amount)
    
    async def verify_webhook(
        self,
        provider: str,
        payload: str,
        signature: str
    ) -> bool:
        """Verify webhook signature"""
        processor = self.get_processor(provider)
        return await processor.verify_webhook(payload, signature)

# Global payment processor manager
payment_processor_manager = PaymentProcessorManager()
