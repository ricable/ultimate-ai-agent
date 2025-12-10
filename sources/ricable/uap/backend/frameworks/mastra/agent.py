# File: backend/frameworks/mastra/agent.py
# Mastra Agent Manager - Real Implementation
# Integrates with Mastra TypeScript framework via HTTP API

import httpx
import json
import logging
from typing import Dict, Any, Optional
import asyncio
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class MastraAgentManager:
    """Real implementation for Mastra agent framework integration.
    
    This class communicates with the Mastra TypeScript framework via HTTP API.
    Mastra specializes in workflow-based operations and support tasks.
    
    The implementation assumes a Mastra service is running and accessible via HTTP.
    """
    
    def __init__(self, 
                 base_url: str = None, 
                 api_key: str = None,
                 timeout: int = 30,
                 agent_id: str = "support-workflow-agent"):
        """Initialize the Mastra agent manager.
        
        Args:
            base_url: Base URL for the Mastra service (default: localhost:4111)
            api_key: API key for authentication (optional for development)
            timeout: Request timeout in seconds
            agent_id: Default agent ID for workflow and support operations
        """
        self.framework_name = "Mastra"
        
        # Configuration
        self.base_url = base_url or os.getenv("MASTRA_BASE_URL", "http://localhost:4111")
        self.api_key = api_key or os.getenv("MASTRA_API_KEY")
        self.timeout = timeout
        self.agent_id = agent_id
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        self.last_health_check = None
        
        # Workflow and support capabilities
        self.workflow_types = {
            "support": "customer-support-workflow",
            "help": "help-desk-workflow", 
            "troubleshoot": "troubleshooting-workflow",
            "guide": "step-by-step-guide-workflow",
            "process": "business-process-workflow"
        }
        
        logger.info(f"{self.framework_name} manager initialized with base_url: {self.base_url}")
    
    async def initialize(self) -> None:
        """Initialize the Mastra framework resources and HTTP client."""
        try:
            # Create HTTP client with appropriate configuration
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            # Perform health check
            await self._health_check()
            
            self.is_initialized = True
            logger.info(f"{self.framework_name} initialization complete - connected to {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.framework_name}: {e}")
            # Don't raise exception - allow graceful degradation
            self.is_initialized = False
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user message using Mastra workflow capabilities.
        
        Args:
            message: The user's input message
            context: Additional context for processing
            
        Returns:
            Dict containing 'content' and 'metadata' keys
        """
        # Always try to initialize first if not already done
        if not self.is_initialized:
            await self.initialize()
        
        # If still not initialized or no client, use enhanced fallback
        if not self.is_initialized or not self.client:
            return await self._fallback_response(message, "Service not initialized")
        
        try:
            # Determine workflow type based on message content
            workflow_type = self._detect_workflow_type(message)
            
            # Prepare request payload for Mastra agent
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "context": context,
                "workflow_type": workflow_type,
                "agent_config": {
                    "specialization": "workflow_support",
                    "enable_workflows": True,
                    "max_iterations": 5
                }
            }
            
            # Call Mastra agent stream endpoint
            response = await self.client.post(
                f"/agents/{self.agent_id}/stream",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content from Mastra response
                content = self._extract_content_from_response(result)
                
                return {
                    "content": content,
                    "metadata": {
                        "source": self.framework_name,
                        "workflow_type": workflow_type,
                        "context_received": bool(context),
                        "specialization": "workflow_support",
                        "response_time": response.elapsed.total_seconds(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            else:
                logger.warning(f"Mastra API returned status {response.status_code}: {response.text}")
                return await self._fallback_response(message, f"API error: {response.status_code}")
                
        except httpx.TimeoutException:
            logger.error("Mastra request timed out")
            return await self._fallback_response(message, "Request timeout")
        except Exception as e:
            logger.error(f"Error processing message with Mastra: {e}")
            return await self._fallback_response(message, f"Processing error: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Mastra manager.
        
        Returns:
            Dict containing status information
        """
        status = "active" if self.is_initialized and self.client else "inactive"
        
        return {
            "status": status,
            "framework": self.framework_name,
            "base_url": self.base_url,
            "agent_id": self.agent_id,
            "specialization": "workflow_support",
            "workflow_types": list(self.workflow_types.keys()),
            "initialized": self.is_initialized,
            "last_health_check": self.last_health_check,
            "capabilities": [
                "customer_support",
                "help_desk",
                "troubleshooting", 
                "step_by_step_guides",
                "business_processes",
                "workflow_automation"
            ]
        }
    
    async def cleanup(self) -> None:
        """Clean up resources when shutting down."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.is_initialized = False
        logger.info(f"{self.framework_name} cleanup complete")
    
    def _detect_workflow_type(self, message: str) -> str:
        """Detect the appropriate workflow type based on message content."""
        message_lower = message.lower()
        
        # Check for workflow keywords
        for keyword, workflow_type in self.workflow_types.items():
            if keyword in message_lower:
                return workflow_type
        
        # Default workflow for general support
        if any(word in message_lower for word in ["help", "support", "assist", "guide"]):
            return "customer-support-workflow"
        elif any(word in message_lower for word in ["problem", "issue", "error", "bug", "troubleshoot"]):
            return "troubleshooting-workflow"
        elif any(word in message_lower for word in ["how to", "steps", "process", "procedure"]):
            return "step-by-step-guide-workflow"
        else:
            return "general-workflow"
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from Mastra API response."""
        # Handle different response formats from Mastra
        if isinstance(response, dict):
            # Try common response formats
            if "content" in response:
                return response["content"]
            elif "message" in response:
                return response["message"]
            elif "text" in response:
                return response["text"]
            elif "response" in response:
                return response["response"]
            elif "choices" in response and len(response["choices"]) > 0:
                # OpenAI-style response format
                choice = response["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]
        
        # Fallback to string representation
        return str(response)
    
    async def _health_check(self) -> bool:
        """Perform health check on Mastra service."""
        try:
            # Try to ping the service
            response = await self.client.get("/health", timeout=5.0)
            is_healthy = response.status_code == 200
            
            if not is_healthy:
                # Try alternative endpoints
                response = await self.client.get("/", timeout=5.0)
                is_healthy = response.status_code in [200, 404]  # 404 is OK if no root endpoint
                
            self.last_health_check = datetime.utcnow().isoformat()
            logger.info(f"Mastra health check: {'PASS' if is_healthy else 'FAIL'}")
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Mastra health check failed: {e}")
            self.last_health_check = f"FAILED: {str(e)}"
            return False
    
    async def _fallback_response(self, message: str, error_reason: str) -> Dict[str, Any]:
        """Generate intelligent fallback response when Mastra service is unavailable."""
        message_lower = message.lower()
        
        # Enhanced intelligent responses based on content analysis
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            content = "Hello! I'm your Customer Support Agent powered by Mastra workflows. I'm here to help you with any questions, troubleshooting, or support tasks. How can I assist you today?"
        
        elif any(question in message_lower for question in ['how are you', 'how do you do', 'what\'s up']):
            content = "I'm doing great, thank you for asking! I'm ready to help you with support requests, troubleshooting, workflow automation, and any questions you might have. What can I help you with?"
        
        elif 'support' in message_lower or 'help' in message_lower:
            content = f"I'd be happy to help you with '{message}'. As your Customer Support Agent, I can assist with troubleshooting, step-by-step guides, workflow creation, and support processes. What specific assistance do you need?"
        
        elif any(word in message_lower for word in ['workflow', 'process', 'automation', 'steps']):
            content = f"Great! I can help you create and manage workflows for '{message}'. I specialize in workflow automation, business processes, and step-by-step guides. Let me assist you in designing an efficient workflow for your needs."
        
        elif any(word in message_lower for word in ['troubleshoot', 'problem', 'issue', 'error', 'bug', 'not working']):
            content = f"I understand you're experiencing an issue with '{message}'. Let me help you troubleshoot this. Can you provide more details about what's happening? I can guide you through diagnostic steps and potential solutions."
        
        elif any(word in message_lower for word in ['guide', 'tutorial', 'how to', 'instructions']):
            content = f"I'd be happy to provide a step-by-step guide for '{message}'. I specialize in creating clear, actionable instructions and tutorials. What specific process would you like me to walk you through?"
        
        elif '?' in message:
            content = f"Great question about '{message}'! I'm here to provide answers and support. While I analyze your question, I can help with troubleshooting, workflows, step-by-step guides, and general support. Would you like me to break this down into manageable steps?"
        
        elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            content = "You're very welcome! I'm always here to help with any support needs, workflow automation, or troubleshooting. Feel free to ask if you have any other questions or need assistance with anything else."
        
        else:
            content = f"I understand you're asking about '{message}'. As your Customer Support Agent, I'm here to provide workflow-based assistance, troubleshooting support, and step-by-step guidance. Let me help you with this request - could you provide a bit more detail about what you need?"
        
        return {
            "content": content,
            "metadata": {
                "source": self.framework_name,
                "specialization": "workflow_support",
                "fallback_mode": True,
                "intelligent_response": True,
                "message_analysis": self._detect_workflow_type(message),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

# Global instance management
_mastra_manager_instance = None

def get_mastra_manager() -> MastraAgentManager:
    """Get or create the global Mastra manager instance."""
    global _mastra_manager_instance
    if _mastra_manager_instance is None:
        _mastra_manager_instance = MastraAgentManager()
    return _mastra_manager_instance

logger.info("Mastra agent module loaded with HTTP API integration")