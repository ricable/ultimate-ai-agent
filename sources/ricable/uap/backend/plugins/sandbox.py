# File: backend/plugins/sandbox.py
"""
Plugin Sandbox - Secure execution environment for plugins.
"""

import asyncio
import resource
import signal
import sys
import traceback
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager

from .plugin_base import PluginResponse
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class SandboxError(Exception):
    """Exception raised by sandbox operations"""
    pass


class ResourceLimits:
    """Resource limits for plugin execution"""
    
    def __init__(self,
                 memory_mb: int = 128,
                 cpu_time_seconds: int = 30,
                 wall_time_seconds: int = 60,
                 file_descriptors: int = 100,
                 processes: int = 5):
        self.memory_mb = memory_mb
        self.cpu_time_seconds = cpu_time_seconds
        self.wall_time_seconds = wall_time_seconds
        self.file_descriptors = file_descriptors
        self.processes = processes


class PluginSandbox:
    """
    Secure execution environment for plugins.
    
    Provides resource limiting, timeout handling, and security isolation.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_active = False
        self._default_limits = ResourceLimits()
    
    async def initialize(self):
        """Initialize the sandbox environment."""
        try:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="plugin_sandbox"
            )
            self.is_active = True
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Plugin sandbox initialized",
                EventType.SYSTEM,
                {"max_workers": self.max_workers},
                "plugin_sandbox"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize plugin sandbox: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "plugin_sandbox"
            )
            raise SandboxError(f"Sandbox initialization failed: {str(e)}")
    
    async def execute_safe(self, func: Callable, *args, resource_limits: Dict[str, Any] = None, **kwargs) -> PluginResponse:
        """
        Execute a function safely within the sandbox.
        
        Args:
            func: Function to execute
            *args: Function arguments
            resource_limits: Resource limits for execution
            **kwargs: Function keyword arguments
            
        Returns:
            PluginResponse with execution results
        """
        if not self.is_active or not self.executor:
            return PluginResponse(
                success=False,
                error="Sandbox not initialized",
                error_code="sandbox_not_ready"
            )
        
        # Parse resource limits
        limits = self._parse_resource_limits(resource_limits)
        
        try:
            # Execute function in thread pool with timeout
            future = self.executor.submit(
                self._execute_with_limits,
                func,
                limits,
                *args,
                **kwargs
            )
            
            # Wait for completion with wall time limit
            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=limits.wall_time_seconds
                )
                
                if isinstance(result, PluginResponse):
                    return result
                else:
                    return PluginResponse(
                        success=True,
                        data=result if isinstance(result, dict) else {"result": result}
                    )
                    
            except asyncio.TimeoutError:
                # Cancel the future if possible
                future.cancel()
                
                return PluginResponse(
                    success=False,
                    error=f"Execution timed out after {limits.wall_time_seconds} seconds",
                    error_code="execution_timeout"
                )
                
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Sandbox execution failed: {str(e)}",
                error_code="sandbox_execution_failed"
            )
    
    def _parse_resource_limits(self, resource_limits: Dict[str, Any] = None) -> ResourceLimits:
        """Parse resource limits from configuration."""
        if not resource_limits:
            return self._default_limits
        
        return ResourceLimits(
            memory_mb=resource_limits.get("memory_mb", self._default_limits.memory_mb),
            cpu_time_seconds=resource_limits.get("cpu_time_seconds", self._default_limits.cpu_time_seconds),
            wall_time_seconds=resource_limits.get("wall_time_seconds", self._default_limits.wall_time_seconds),
            file_descriptors=resource_limits.get("file_descriptors", self._default_limits.file_descriptors),
            processes=resource_limits.get("processes", self._default_limits.processes)
        )
    
    def _execute_with_limits(self, func: Callable, limits: ResourceLimits, *args, **kwargs):
        """
        Execute function with resource limits applied.
        
        This runs in a separate thread to isolate resource usage.
        """
        try:
            # Apply resource limits
            self._apply_resource_limits(limits)
            
            # Set up timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError("CPU time limit exceeded")
            
            # Set CPU time alarm
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(limits.cpu_time_seconds)
            
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    # Handle async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(func(*args, **kwargs))
                    finally:
                        loop.close()
                else:
                    # Handle sync functions
                    result = func(*args, **kwargs)
                
                return result
                
            finally:
                # Clear the alarm
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                
        except TimeoutError:
            raise SandboxError("CPU time limit exceeded")
        except MemoryError:
            raise SandboxError("Memory limit exceeded")
        except Exception as e:
            # Log the full traceback for debugging
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Plugin execution error: {str(e)}",
                EventType.ERROR,
                {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                "plugin_sandbox"
            )
            raise
    
    def _apply_resource_limits(self, limits: ResourceLimits):
        """
        Apply resource limits to the current process.
        
        Note: This only works on Unix-like systems.
        """
        try:
            # Memory limit (RSS)
            if hasattr(resource, 'RLIMIT_RSS'):
                memory_bytes = limits.memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_RSS, (memory_bytes, memory_bytes))
            
            # CPU time limit
            if hasattr(resource, 'RLIMIT_CPU'):
                resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_time_seconds, limits.cpu_time_seconds))
            
            # File descriptor limit
            if hasattr(resource, 'RLIMIT_NOFILE'):
                resource.setrlimit(resource.RLIMIT_NOFILE, (limits.file_descriptors, limits.file_descriptors))
            
            # Process limit
            if hasattr(resource, 'RLIMIT_NPROC'):
                resource.setrlimit(resource.RLIMIT_NPROC, (limits.processes, limits.processes))
                
        except (OSError, ValueError) as e:
            # Resource limits may not be supported on all platforms
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Failed to apply resource limits: {str(e)}",
                EventType.WARNING,
                {"error": str(e)},
                "plugin_sandbox"
            )
    
    @contextmanager
    def restricted_environment(self):
        """
        Context manager for restricted execution environment.
        
        Temporarily restricts certain Python features.
        """
        # Store original values
        original_import = __builtins__['__import__']
        original_open = __builtins__['open']
        original_exec = __builtins__['exec']
        original_eval = __builtins__['eval']
        
        # Restricted import function
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Allow only safe modules
            safe_modules = {
                'json', 'datetime', 'typing', 'uuid', 're', 'math', 
                'collections', 'itertools', 'functools', 'operator',
                'asyncio', 'aiohttp', 'pydantic'
            }
            
            if name.split('.')[0] not in safe_modules:
                raise ImportError(f"Import of '{name}' is not allowed in sandbox")
            
            return original_import(name, globals, locals, fromlist, level)
        
        # Restricted open function
        def restricted_open(*args, **kwargs):
            raise OSError("File operations are not allowed in sandbox")
        
        # Restricted exec/eval
        def restricted_exec(*args, **kwargs):
            raise RuntimeError("Dynamic code execution is not allowed in sandbox")
        
        def restricted_eval(*args, **kwargs):
            raise RuntimeError("Dynamic code evaluation is not allowed in sandbox")
        
        try:
            # Apply restrictions
            __builtins__['__import__'] = restricted_import
            __builtins__['open'] = restricted_open
            __builtins__['exec'] = restricted_exec
            __builtins__['eval'] = restricted_eval
            
            yield
            
        finally:
            # Restore original functions
            __builtins__['__import__'] = original_import
            __builtins__['open'] = original_open
            __builtins__['exec'] = original_exec
            __builtins__['eval'] = original_eval
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage statistics.
        
        Returns:
            Dictionary with resource usage information
        """
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            
            return {
                "memory_peak_kb": usage.ru_maxrss,
                "cpu_time_user": usage.ru_utime,
                "cpu_time_system": usage.ru_stime,
                "page_faults_minor": usage.ru_minflt,
                "page_faults_major": usage.ru_majflt,
                "context_switches_voluntary": usage.ru_nvcsw,
                "context_switches_involuntary": usage.ru_nivcsw,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def cleanup(self):
        """Clean up sandbox resources."""
        try:
            if self.executor:
                # Shutdown executor gracefully
                self.executor.shutdown(wait=True, timeout=30)
                self.executor = None
            
            self.is_active = False
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Plugin sandbox cleanup complete",
                EventType.SYSTEM,
                {},
                "plugin_sandbox"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Error during sandbox cleanup: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "plugin_sandbox"
            )


class SecurityValidator:
    """
    Security validation for plugin code and operations.
    """
    
    @staticmethod
    def validate_code(code: str) -> bool:
        """
        Validate plugin code for security issues.
        
        Args:
            code: Python code to validate
            
        Returns:
            True if code appears safe
        """
        # Dangerous patterns to check for
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'import socket',
            'import urllib',
            'import requests',
            '__import__',
            'exec(',
            'eval(',
            'compile(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
            'globals()',
            'locals()',
            'vars(',
            'dir(',
            'getattr(',
            'setattr(',
            'delattr(',
            'hasattr(',
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    f"Potentially dangerous code pattern detected: {pattern}",
                    EventType.SECURITY,
                    {"pattern": pattern},
                    "security_validator"
                )
                return False
        
        return True
    
    @staticmethod
    def validate_permissions(requested_permissions: list, allowed_permissions: list) -> bool:
        """
        Validate that requested permissions are allowed.
        
        Args:
            requested_permissions: List of permissions requested by plugin
            allowed_permissions: List of permissions allowed for user/context
            
        Returns:
            True if all requested permissions are allowed
        """
        for permission in requested_permissions:
            if permission not in allowed_permissions:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    f"Unauthorized permission requested: {permission}",
                    EventType.SECURITY,
                    {"permission": permission},
                    "security_validator"
                )
                return False
        
        return True
    
    @staticmethod
    def sanitize_input(data: Any) -> Any:
        """
        Sanitize input data to prevent injection attacks.
        
        Args:
            data: Input data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '&', '"', "'", '\x00']
            for char in dangerous_chars:
                data = data.replace(char, '')
        elif isinstance(data, dict):
            return {k: SecurityValidator.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [SecurityValidator.sanitize_input(item) for item in data]
        
        return data