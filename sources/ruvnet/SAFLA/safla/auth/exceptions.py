"""Authentication exceptions for SAFLA MCP Server."""


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    
    def __init__(self, message: str = "Authentication failed", status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class TokenExpiredError(AuthenticationError):
    """Raised when a JWT token has expired."""
    
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, 401)


class InvalidTokenError(AuthenticationError):
    """Raised when a JWT token is invalid."""
    
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, 401)


class MissingTokenError(AuthenticationError):
    """Raised when no token is provided."""
    
    def __init__(self, message: str = "No authentication token provided"):
        super().__init__(message, 401)


class InsufficientPermissionsError(AuthenticationError):
    """Raised when user lacks required permissions."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, 403)