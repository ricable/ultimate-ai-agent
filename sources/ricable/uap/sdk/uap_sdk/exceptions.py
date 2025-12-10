# UAP SDK Exceptions Module
"""
Custom exceptions for UAP SDK operations.
"""


class UAPException(Exception):
    """Base exception for UAP SDK operations."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class UAPConnectionError(UAPException):
    """Exception raised when connection to UAP fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONNECTION_ERROR", details)


class UAPAuthError(UAPException):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUTH_ERROR", details)


class UAPConfigError(UAPException):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)


class UAPPluginError(UAPException):
    """Exception raised when plugin operations fail."""
    
    def __init__(self, message: str, plugin_name: str = None, details: dict = None):
        super().__init__(message, "PLUGIN_ERROR", details)
        self.plugin_name = plugin_name


class UAPTimeoutError(UAPException):
    """Exception raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: float = None, details: dict = None):
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_seconds = timeout_seconds


class UAPValidationError(UAPException):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, field_name: str = None, details: dict = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field_name = field_name