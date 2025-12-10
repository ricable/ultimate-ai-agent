# File: backend/webhooks/receiver.py
"""
Webhook receiver for handling incoming webhooks from third-party services.
"""

import asyncio
import json
import hashlib
import hmac
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from fastapi import Request, HTTPException
from pydantic import BaseModel, Field
import uuid

from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class WebhookError(Exception):
    """Webhook processing error"""
    def __init__(self, message: str, webhook_id: str = None, status_code: int = 400):
        super().__init__(message)
        self.webhook_id = webhook_id
        self.status_code = status_code


class WebhookPayload(BaseModel):
    """Standardized webhook payload structure"""
    webhook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    integration_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_ip: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    raw_payload: Dict[str, Any]
    verified: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookReceiver:
    """
    Handles incoming webhooks from third-party services.
    
    Provides security verification, payload parsing, and event routing.
    """
    
    def __init__(self, integration_manager):
        self.integration_manager = integration_manager
        self.webhook_secrets: Dict[str, str] = {}  # integration_id -> secret
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # IP -> rate limit info
        self._processing_queue = asyncio.Queue(maxsize=1000)
        self._workers_started = False
    
    async def start_workers(self, num_workers: int = 5):
        """Start background workers for processing webhooks."""
        if self._workers_started:
            return
        
        for i in range(num_workers):
            asyncio.create_task(self._webhook_worker(f"worker-{i}"))
        
        self._workers_started = True
        uap_logger.log_event(
            LogLevel.INFO,
            f"Started {num_workers} webhook workers",
            EventType.SYSTEM,
            {"workers": num_workers},
            "webhook_receiver"
        )
    
    async def receive_webhook(self, request: Request, integration_id: str) -> Dict[str, Any]:
        """
        Receive and process incoming webhook.
        
        Args:
            request: FastAPI request object
            integration_id: Target integration ID
            
        Returns:
            Processing response
        """
        try:
            # Extract request information
            source_ip = self._get_client_ip(request)
            headers = dict(request.headers)
            
            # Check rate limits
            if not self._check_rate_limit(source_ip):
                raise WebhookError(
                    "Rate limit exceeded",
                    status_code=429
                )
            
            # Read raw payload
            raw_body = await request.body()
            
            # Parse JSON payload
            try:
                raw_payload = json.loads(raw_body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise WebhookError(
                    f"Invalid JSON payload: {str(e)}",
                    status_code=400
                )
            
            # Create webhook payload
            webhook_payload = WebhookPayload(
                integration_id=integration_id,
                event_type=self._extract_event_type(raw_payload, integration_id),
                source_ip=source_ip,
                headers=headers,
                raw_payload=raw_payload
            )
            
            # Verify webhook signature
            webhook_payload.verified = await self._verify_webhook_signature(
                integration_id, raw_body, headers
            )
            
            if not webhook_payload.verified:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    f"Webhook signature verification failed for {integration_id}",
                    EventType.SECURITY,
                    {
                        "integration_id": integration_id,
                        "source_ip": source_ip,
                        "webhook_id": webhook_payload.webhook_id
                    },
                    "webhook_receiver"
                )
                # Continue processing but mark as unverified
            
            # Queue for processing
            try:
                await self._processing_queue.put(webhook_payload)
            except asyncio.QueueFull:
                raise WebhookError(
                    "Webhook processing queue full",
                    webhook_payload.webhook_id,
                    status_code=503
                )
            
            # Log successful receipt
            uap_logger.log_event(
                LogLevel.INFO,
                f"Webhook received from {integration_id}",
                EventType.WEBHOOK,
                {
                    "integration_id": integration_id,
                    "webhook_id": webhook_payload.webhook_id,
                    "event_type": webhook_payload.event_type,
                    "verified": webhook_payload.verified,
                    "source_ip": source_ip
                },
                "webhook_receiver"
            )
            
            return {
                "success": True,
                "webhook_id": webhook_payload.webhook_id,
                "message": "Webhook received and queued for processing",
                "verified": webhook_payload.verified
            }
            
        except WebhookError:
            raise
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Unexpected error receiving webhook: {str(e)}",
                EventType.ERROR,
                {
                    "integration_id": integration_id,
                    "error": str(e),
                    "source_ip": self._get_client_ip(request)
                },
                "webhook_receiver"
            )
            raise WebhookError(
                "Internal server error",
                status_code=500
            )
    
    async def _webhook_worker(self, worker_name: str):
        """Background worker for processing webhook queue."""
        uap_logger.log_event(
            LogLevel.INFO,
            f"Webhook worker {worker_name} started",
            EventType.SYSTEM,
            {"worker": worker_name},
            "webhook_receiver"
        )
        
        while True:
            try:
                # Get webhook from queue
                webhook_payload = await self._processing_queue.get()
                
                # Process webhook
                await self._process_webhook(webhook_payload, worker_name)
                
                # Mark task done
                self._processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Webhook worker {worker_name} error: {str(e)}",
                    EventType.ERROR,
                    {"worker": worker_name, "error": str(e)},
                    "webhook_receiver"
                )
    
    async def _process_webhook(self, payload: WebhookPayload, worker_name: str):
        """Process individual webhook payload."""
        try:
            # Get integration
            integration = self.integration_manager.get_integration(payload.integration_id)
            if not integration:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Integration {payload.integration_id} not found for webhook",
                    EventType.ERROR,
                    {
                        "integration_id": payload.integration_id,
                        "webhook_id": payload.webhook_id
                    },
                    "webhook_receiver"
                )
                return
            
            # Create integration event
            from ..integrations.base import IntegrationEvent
            event = IntegrationEvent(
                integration_id=payload.integration_id,
                event_type=payload.event_type,
                source=integration.name,
                timestamp=payload.timestamp,
                data=payload.raw_payload,
                metadata={
                    "webhook_id": payload.webhook_id,
                    "verified": payload.verified,
                    "source_ip": payload.source_ip,
                    "worker": worker_name
                }
            )
            
            # Process with integration
            response = await integration.receive_webhook(event)
            
            if response.success:
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Webhook processed successfully by {payload.integration_id}",
                    EventType.WEBHOOK,
                    {
                        "integration_id": payload.integration_id,
                        "webhook_id": payload.webhook_id,
                        "worker": worker_name
                    },
                    "webhook_receiver"
                )
            else:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    f"Webhook processing failed: {response.error}",
                    EventType.WARNING,
                    {
                        "integration_id": payload.integration_id,
                        "webhook_id": payload.webhook_id,
                        "error": response.error,
                        "worker": worker_name
                    },
                    "webhook_receiver"
                )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Error processing webhook {payload.webhook_id}: {str(e)}",
                EventType.ERROR,
                {
                    "webhook_id": payload.webhook_id,
                    "integration_id": payload.integration_id,
                    "error": str(e),
                    "worker": worker_name
                },
                "webhook_receiver"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, ip: str) -> bool:
        """Check rate limit for IP address."""
        now = datetime.now(timezone.utc)
        
        if ip not in self.rate_limits:
            self.rate_limits[ip] = {
                "count": 1,
                "window_start": now,
                "last_request": now
            }
            return True
        
        rate_info = self.rate_limits[ip]
        
        # Reset window if needed (1 minute window)
        if (now - rate_info["window_start"]).total_seconds() > 60:
            rate_info["count"] = 1
            rate_info["window_start"] = now
            rate_info["last_request"] = now
            return True
        
        # Check if under limit (100 requests per minute)
        if rate_info["count"] < 100:
            rate_info["count"] += 1
            rate_info["last_request"] = now
            return True
        
        return False
    
    def _extract_event_type(self, payload: Dict[str, Any], integration_id: str) -> str:
        """Extract event type from payload based on integration."""
        # Common event type fields
        common_fields = ["type", "event", "event_type", "action", "kind"]
        
        for field in common_fields:
            if field in payload:
                return str(payload[field])
        
        # Integration-specific extraction
        if integration_id == "slack":
            return payload.get("event", {}).get("type", "unknown")
        elif integration_id == "github":
            return payload.get("action", "unknown")
        elif integration_id == "discord":
            return payload.get("t", "unknown")
        
        return "webhook"
    
    async def _verify_webhook_signature(self, integration_id: str, 
                                       raw_body: bytes, headers: Dict[str, str]) -> bool:
        """Verify webhook signature for security."""
        try:
            secret = self.webhook_secrets.get(integration_id)
            if not secret:
                # No secret configured, skip verification
                return False
            
            # Integration-specific signature verification
            if integration_id == "slack":
                return self._verify_slack_signature(raw_body, headers, secret)
            elif integration_id == "github":
                return self._verify_github_signature(raw_body, headers, secret)
            elif integration_id == "discord":
                return self._verify_discord_signature(raw_body, headers, secret)
            else:
                # Generic HMAC-SHA256 verification
                return self._verify_hmac_signature(raw_body, headers, secret)
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Signature verification error for {integration_id}: {str(e)}",
                EventType.ERROR,
                {"integration_id": integration_id, "error": str(e)},
                "webhook_receiver"
            )
            return False
    
    def _verify_slack_signature(self, body: bytes, headers: Dict[str, str], secret: str) -> bool:
        """Verify Slack webhook signature."""
        timestamp = headers.get("X-Slack-Request-Timestamp")
        signature = headers.get("X-Slack-Signature")
        
        if not timestamp or not signature:
            return False
        
        # Check timestamp (prevent replay attacks)
        try:
            request_time = int(timestamp)
            if abs(datetime.now().timestamp() - request_time) > 300:  # 5 minutes
                return False
        except ValueError:
            return False
        
        # Verify signature
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected_signature = "v0=" + hmac.new(
            secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _verify_github_signature(self, body: bytes, headers: Dict[str, str], secret: str) -> bool:
        """Verify GitHub webhook signature."""
        signature = headers.get("X-Hub-Signature-256")
        if not signature:
            return False
        
        expected_signature = "sha256=" + hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _verify_discord_signature(self, body: bytes, headers: Dict[str, str], secret: str) -> bool:
        """Verify Discord webhook signature."""
        signature = headers.get("X-Signature-Ed25519")
        timestamp = headers.get("X-Signature-Timestamp")
        
        if not signature or not timestamp:
            return False
        
        # Discord uses Ed25519 signatures (simplified for demo)
        # In production, use nacl.signing or similar library
        return True  # Placeholder
    
    def _verify_hmac_signature(self, body: bytes, headers: Dict[str, str], secret: str) -> bool:
        """Generic HMAC-SHA256 signature verification."""
        signature = headers.get("X-Signature") or headers.get("X-Hub-Signature")
        if not signature:
            return False
        
        # Remove algorithm prefix if present
        if signature.startswith("sha256="):
            signature = signature[7:]
        
        expected_signature = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def register_webhook_secret(self, integration_id: str, secret: str):
        """Register webhook secret for integration."""
        self.webhook_secrets[integration_id] = secret
        uap_logger.log_event(
            LogLevel.INFO,
            f"Webhook secret registered for {integration_id}",
            EventType.SECURITY,
            {"integration_id": integration_id},
            "webhook_receiver"
        )
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook processing statistics."""
        return {
            "queue_size": self._processing_queue.qsize(),
            "max_queue_size": self._processing_queue.maxsize,
            "workers_started": self._workers_started,
            "registered_secrets": len(self.webhook_secrets),
            "rate_limited_ips": len(self.rate_limits)
        }