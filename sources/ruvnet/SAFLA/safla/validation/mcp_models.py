"""Pydantic models for MCP request/response validation"""

from typing import Any, Dict, List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, field_validator
from pydantic.types import StringConstraints
from pathlib import Path
import re


class MCPRequest(BaseModel):
    """Base MCP request model"""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str = Field(..., min_length=1, max_length=100)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    id: Optional[Union[str, int]] = None
    
    @field_validator('method')
    def validate_method(cls, v):
        """Validate method name format"""
        if not re.match(r'^[a-zA-Z0-9_/\-]+$', v):
            raise ValueError('Invalid method name format')
        return v


class ToolCallRequest(BaseModel):
    """Tool call request parameters"""
    name: Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_\-]+$')]
    arguments: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('arguments')
    def validate_arguments(cls, v):
        """Validate tool arguments"""
        # Prevent deeply nested objects
        def check_depth(obj, depth=0, max_depth=5):
            if depth > max_depth:
                raise ValueError('Arguments too deeply nested')
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, depth + 1)
        
        check_depth(v)
        return v


class ResourceReadRequest(BaseModel):
    """Resource read request parameters"""
    uri: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    
    @field_validator('uri')
    def validate_uri(cls, v):
        """Validate resource URI"""
        # Basic URI validation
        if not re.match(r'^[a-zA-Z0-9_\-/:\.]+$', v):
            raise ValueError('Invalid URI format')
        
        # Prevent path traversal
        if '..' in v or v.startswith('/'):
            raise ValueError('Path traversal not allowed')
        
        return v


class DeployRequest(BaseModel):
    """Deployment request parameters"""
    instance_count: Annotated[int, Field(ge=1, le=100)] = 1
    deployment_mode: Annotated[str, Field(pattern=r'^(local|cloud|hybrid)$')]
    config_path: Optional[str] = None
    
    @field_validator('config_path')
    def validate_config_path(cls, v):
        """Validate config path"""
        if v:
            # Prevent path traversal
            if '..' in v or v.startswith('/'):
                raise ValueError('Invalid config path')
            
            # Check file extension
            if not v.endswith(('.json', '.yaml', '.yml')):
                raise ValueError('Config must be JSON or YAML')
        
        return v


class BackupRequest(BaseModel):
    """Backup request parameters"""
    backup_path: str
    include_models: bool = True
    include_data: bool = True
    compress: bool = True
    
    @field_validator('backup_path')
    def validate_backup_path(cls, v):
        """Validate backup path"""
        # Prevent path traversal
        if '..' in v:
            raise ValueError('Path traversal not allowed')
        
        # Ensure it's a valid path
        try:
            Path(v)
        except Exception:
            raise ValueError('Invalid backup path')
        
        return v


class BenchmarkRequest(BaseModel):
    """Benchmark request parameters"""
    benchmark_type: Annotated[str, Field(pattern=r'^(vector_ops|memory|mcp_throughput|all)$')]
    iterations: Annotated[int, Field(ge=1, le=10000)] = 100
    batch_size: Annotated[int, Field(ge=1, le=1000)] = 10
    
    class Config:
        extra = 'forbid'  # Reject unknown fields


class AgentSessionRequest(BaseModel):
    """Agent session request parameters"""
    session_id: Optional[Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_\-]+$')]] = None
    agent_type: Optional[Annotated[str, Field(pattern=r'^(general|specialized|research|code)$')]] = "general"
    capabilities: Optional[List[str]] = Field(default_factory=list)
    
    @field_validator('capabilities')
    def validate_capabilities(cls, v):
        """Validate agent capabilities"""
        allowed = {'research', 'analysis', 'code', 'planning', 'execution'}
        for cap in v:
            if cap not in allowed:
                raise ValueError(f'Invalid capability: {cap}')
        return v


class MCPResponse(BaseModel):
    """Base MCP response model"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    @field_validator('error')
    def validate_error(cls, v):
        """Validate error format"""
        if v:
            required_fields = {'code', 'message'}
            if not all(field in v for field in required_fields):
                raise ValueError('Error must contain code and message')
            
            # Validate error code
            if not isinstance(v['code'], int):
                raise ValueError('Error code must be integer')
            
            # Sanitize error message
            if isinstance(v.get('message'), str):
                # Remove sensitive information patterns
                v['message'] = re.sub(r'(/[a-zA-Z0-9_\-/]+)+', '[PATH]', v['message'])
                v['message'] = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', v['message'])
                v['message'] = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', v['message'])
        
        return v


class ToolResponse(BaseModel):
    """Tool execution response"""
    content: List[Dict[str, Any]]
    isError: Optional[bool] = False
    
    @field_validator('content')
    def validate_content(cls, v):
        """Validate tool response content"""
        for item in v:
            if 'type' not in item:
                raise ValueError('Content item must have type')
            
            # Validate based on type
            if item['type'] == 'text' and 'text' not in item:
                raise ValueError('Text content must have text field')
            elif item['type'] == 'image' and 'data' not in item:
                raise ValueError('Image content must have data field')
        
        return v


class ResourceResponse(BaseModel):
    """Resource read response"""
    contents: List[Dict[str, Any]]
    
    @field_validator('contents')
    def validate_contents(cls, v):
        """Validate resource contents"""
        for item in v:
            required_fields = {'uri', 'mimeType', 'text'}
            if not all(field in item for field in required_fields):
                raise ValueError('Resource content missing required fields')
        
        return v