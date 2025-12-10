"""Security validators and sanitizers"""

import re
import os
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Validate and sanitize file paths to prevent traversal attacks.
    
    Args:
        path: Path to validate
        base_dir: Base directory to restrict access to
        
    Returns:
        Sanitized path
        
    Raises:
        ValueError: If path is invalid or attempts traversal
    """
    # Remove any null bytes
    path = path.replace('\0', '')
    
    # Convert to Path object for validation
    try:
        p = Path(path)
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")
    
    # Resolve to absolute path
    if base_dir:
        base = Path(base_dir).resolve()
        full_path = (base / p).resolve()
        
        # Ensure path is within base directory
        try:
            full_path.relative_to(base)
        except ValueError:
            raise ValueError("Path traversal detected")
    else:
        full_path = p.resolve()
    
    # Additional checks
    if '..' in str(path):
        raise ValueError("Path traversal detected")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.\./',           # Parent directory
        r'\.\.\\',          # Parent directory (Windows)
        r'^/',              # Absolute path
        r'^[A-Za-z]:',      # Windows drive
        r'\0',              # Null byte
        r'[\r\n]',          # Line breaks
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, path):
            raise ValueError(f"Suspicious path pattern detected")
    
    return str(full_path)


def validate_tool_name(name: str) -> str:
    """
    Validate tool name to prevent injection.
    
    Args:
        name: Tool name to validate
        
    Returns:
        Validated tool name
        
    Raises:
        ValueError: If name is invalid
    """
    # Allow only alphanumeric, underscore, and hyphen
    if not re.match(r'^[a-zA-Z0-9_\-]+$', name):
        raise ValueError("Invalid tool name format")
    
    # Check length
    if len(name) < 1 or len(name) > 100:
        raise ValueError("Tool name must be between 1 and 100 characters")
    
    # Check against reserved names
    reserved = {'__init__', '__main__', '__pycache__', 'None', 'True', 'False'}
    if name in reserved:
        raise ValueError("Reserved tool name")
    
    return name


def validate_resource_uri(uri: str) -> str:
    """
    Validate resource URI.
    
    Args:
        uri: Resource URI to validate
        
    Returns:
        Validated URI
        
    Raises:
        ValueError: If URI is invalid
    """
    # Basic format validation
    if not re.match(r'^[a-zA-Z0-9_\-/:\.]+$', uri):
        raise ValueError("Invalid URI format")
    
    # Prevent common injection patterns
    dangerous_patterns = [
        r'javascript:',     # JavaScript protocol
        r'data:',          # Data URI
        r'vbscript:',      # VBScript protocol
        r'file://',        # File protocol
        r'\.\./',          # Path traversal
        r'<script',        # Script tag
        r'onerror=',       # Event handler
        r'onclick=',       # Event handler
    ]
    
    uri_lower = uri.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, uri_lower, re.IGNORECASE):
            raise ValueError(f"Dangerous URI pattern detected")
    
    # Length check
    if len(uri) > 500:
        raise ValueError("URI too long")
    
    return uri


def sanitize_error_message(message: str) -> str:
    """
    Sanitize error messages to prevent information leakage.
    
    Args:
        message: Error message to sanitize
        
    Returns:
        Sanitized error message
    """
    # Remove file paths
    message = re.sub(r'(/[a-zA-Z0-9_\-/]+)+\.[a-zA-Z]+', '[FILE]', message)
    message = re.sub(r'[A-Za-z]:\\[^\s]+', '[FILE]', message)  # Windows paths
    
    # Remove email addresses
    message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', message)
    
    # Remove IP addresses
    message = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', message)
    
    # Remove port numbers
    message = re.sub(r':\d{2,5}\b', ':[PORT]', message)
    
    # Remove potential secrets (API keys, tokens, etc.)
    message = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[REDACTED]', message)
    
    # Remove stack trace details
    message = re.sub(r'File ".*", line \d+', 'File "[REDACTED]", line [N]', message)
    message = re.sub(r'at 0x[0-9a-fA-F]+', 'at [ADDR]', message)
    
    # Remove environment variables
    for env_var in os.environ:
        if env_var in message:
            message = message.replace(os.environ[env_var], f'[{env_var}]')
    
    # Truncate if too long
    max_length = 500
    if len(message) > max_length:
        message = message[:max_length] + '... [truncated]'
    
    return message


def validate_json_size(data: Any, max_size_mb: float = 10.0) -> None:
    """
    Validate JSON data size to prevent memory exhaustion.
    
    Args:
        data: JSON data to validate
        max_size_mb: Maximum allowed size in megabytes
        
    Raises:
        ValueError: If data exceeds size limit
    """
    import json
    
    # Convert to JSON string to check size
    json_str = json.dumps(data)
    size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
    
    if size_mb > max_size_mb:
        raise ValueError(f"JSON data too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")


def validate_batch_size(size: int, max_size: int = 1000) -> int:
    """
    Validate batch size to prevent resource exhaustion.
    
    Args:
        size: Requested batch size
        max_size: Maximum allowed batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ValueError: If batch size is invalid
    """
    if not isinstance(size, int):
        raise ValueError("Batch size must be an integer")
    
    if size < 1:
        raise ValueError("Batch size must be positive")
    
    if size > max_size:
        raise ValueError(f"Batch size too large: {size} (max: {max_size})")
    
    return size