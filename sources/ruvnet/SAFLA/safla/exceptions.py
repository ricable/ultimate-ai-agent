"""
SAFLA Exception Classes

This module defines custom exception classes for the SAFLA system,
providing structured error handling and debugging information.
"""

from typing import Optional, Dict, Any


class SAFLAError(Exception):
    """Base exception class for all SAFLA-related errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        error_str = self.message
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        return error_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class MemoryError(SAFLAError):
    """Exception raised for memory-related errors."""
    
    def __init__(
        self, 
        message: str, 
        memory_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.memory_type = memory_type
        self.operation = operation
        if memory_type:
            self.context["memory_type"] = memory_type
        if operation:
            self.context["operation"] = operation


class VectorMemoryError(MemoryError):
    """Exception raised for vector memory operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, memory_type="vector", **kwargs)


class EpisodicMemoryError(MemoryError):
    """Exception raised for episodic memory operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, memory_type="episodic", **kwargs)


class SemanticMemoryError(MemoryError):
    """Exception raised for semantic memory operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, memory_type="semantic", **kwargs)


class WorkingMemoryError(MemoryError):
    """Exception raised for working memory operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, memory_type="working", **kwargs)


class SafetyViolationError(SAFLAError):
    """Exception raised when safety constraints are violated."""
    
    def __init__(
        self, 
        message: str, 
        constraint_name: Optional[str] = None,
        violation_type: Optional[str] = None,
        severity: str = "high",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.constraint_name = constraint_name
        self.violation_type = violation_type
        self.severity = severity
        
        self.context.update({
            "constraint_name": constraint_name,
            "violation_type": violation_type,
            "severity": severity
        })


class HardConstraintViolationError(SafetyViolationError):
    """Exception raised when hard safety constraints are violated."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, violation_type="hard", severity="critical", **kwargs)


class SoftConstraintViolationError(SafetyViolationError):
    """Exception raised when soft safety constraints are violated."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, violation_type="soft", severity="medium", **kwargs)


class MCPError(SAFLAError):
    """Exception raised for MCP-related errors."""
    
    def __init__(
        self, 
        message: str, 
        server_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.server_name = server_name
        self.operation = operation
        
        if server_name:
            self.context["server_name"] = server_name
        if operation:
            self.context["operation"] = operation


class MCPConnectionError(MCPError):
    """Exception raised for MCP connection errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, operation="connection", **kwargs)


class MCPTimeoutError(MCPError):
    """Exception raised for MCP timeout errors."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(message, operation="timeout", **kwargs)
        if timeout_duration:
            self.context["timeout_duration"] = timeout_duration


class MCPServerError(MCPError):
    """Exception raised for MCP server-side errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, operation="server_error", **kwargs)
        if status_code:
            self.context["status_code"] = status_code


class MetaCognitiveError(SAFLAError):
    """Exception raised for meta-cognitive engine errors."""
    
    def __init__(
        self, 
        message: str, 
        component: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.component = component
        self.operation = operation
        
        if component:
            self.context["component"] = component
        if operation:
            self.context["operation"] = operation


class GoalManagementError(MetaCognitiveError):
    """Exception raised for goal management errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="goal_manager", **kwargs)


class StrategySelectionError(MetaCognitiveError):
    """Exception raised for strategy selection errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="strategy_selector", **kwargs)


class SelfAwarenessError(MetaCognitiveError):
    """Exception raised for self-awareness module errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="self_awareness", **kwargs)


class ConfigurationError(SAFLAError):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self, 
        message: str, 
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.config_key = config_key
        
        if config_section:
            self.context["config_section"] = config_section
        if config_key:
            self.context["config_key"] = config_key


class ValidationError(SAFLAError):
    """Exception raised for validation errors."""
    
    def __init__(
        self, 
        message: str, 
        validation_type: Optional[str] = None,
        failed_checks: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []
        
        if validation_type:
            self.context["validation_type"] = validation_type
        if failed_checks:
            self.context["failed_checks"] = failed_checks


class DeltaEvaluationError(SAFLAError):
    """Exception raised for delta evaluation errors."""
    
    def __init__(
        self, 
        message: str, 
        delta_type: Optional[str] = None,
        evaluation_context: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.delta_type = delta_type
        self.evaluation_context = evaluation_context
        
        if delta_type:
            self.context["delta_type"] = delta_type
        if evaluation_context:
            self.context["evaluation_context"] = evaluation_context


class ResourceError(SAFLAError):
    """Exception raised for resource-related errors."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        resource_limit: Optional[float] = None,
        current_usage: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_limit = resource_limit
        self.current_usage = current_usage
        
        self.context.update({
            "resource_type": resource_type,
            "resource_limit": resource_limit,
            "current_usage": current_usage
        })


class MemoryLimitError(ResourceError):
    """Exception raised when memory limits are exceeded."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, resource_type="memory", **kwargs)


class CPULimitError(ResourceError):
    """Exception raised when CPU limits are exceeded."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, resource_type="cpu", **kwargs)


class NetworkError(SAFLAError):
    """Exception raised for network-related errors."""
    
    def __init__(
        self, 
        message: str, 
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.status_code = status_code
        
        if endpoint:
            self.context["endpoint"] = endpoint
        if status_code:
            self.context["status_code"] = status_code


class SerializationError(SAFLAError):
    """Exception raised for serialization/deserialization errors."""
    
    def __init__(
        self, 
        message: str, 
        data_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.operation = operation
        
        if data_type:
            self.context["data_type"] = data_type
        if operation:
            self.context["operation"] = operation


# Exception handling utilities
def handle_safla_exception(exception: Exception) -> Dict[str, Any]:
    """
    Handle SAFLA exceptions and convert to structured format.
    
    Args:
        exception: Exception to handle
        
    Returns:
        Dictionary with exception information
    """
    if isinstance(exception, SAFLAError):
        return exception.to_dict()
    else:
        return {
            "error_type": exception.__class__.__name__,
            "message": str(exception),
            "error_code": None,
            "context": {}
        }


def create_error_response(exception: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        exception: Exception to create response for
        include_traceback: Whether to include traceback information
        
    Returns:
        Standardized error response dictionary
    """
    import traceback
    
    response = {
        "success": False,
        "error": handle_safla_exception(exception),
        "timestamp": None
    }
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    # Add timestamp
    try:
        import time
        response["timestamp"] = time.time()
    except ImportError:
        pass
    
    return response