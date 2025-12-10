# MCP Intelligent Coder - Practical Examples

## Example 1: Building a FastAPI Authentication System

This example demonstrates the complete workflow of using MCP Intelligent Coder mode to build a secure authentication system.

### Step 1: Initial Research and Planning

```xml
<!-- Research current authentication best practices -->
<use_mcp_tool>
<server_name>perplexity</server_name>
<tool_name>PERPLEXITYAI_PERPLEXITY_AI_SEARCH</tool_name>
<arguments>{
  "userContent": "FastAPI JWT authentication best practices 2024 with security considerations",
  "systemContent": "Provide comprehensive implementation guidelines with security best practices and code examples",
  "model": "llama-3.1-sonar-large-128k-online",
  "return_citations": true,
  "temperature": 0.3
}</arguments>
</use_mcp_tool>
```

### Step 2: Documentation Retrieval

```xml
<!-- Resolve FastAPI library ID -->
<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>resolve-library-id</tool_name>
<arguments>{"libraryName": "fastapi"}</arguments>
</use_mcp_tool>

<!-- Get FastAPI security documentation -->
<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>get-library-docs</tool_name>
<arguments>{
  "context7CompatibleLibraryID": "/tiangolo/fastapi",
  "topic": "security",
  "tokens": 8000
}</arguments>
</use_mcp_tool>

<!-- Get JWT library documentation -->
<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>resolve-library-id</tool_name>
<arguments>{"libraryName": "pyjwt"}</arguments>
</use_mcp_tool>

<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>get-library-docs</tool_name>
<arguments>{
  "context7CompatibleLibraryID": "/jpadilla/pyjwt",
  "topic": "authentication",
  "tokens": 5000
}</arguments>
</use_mcp_tool>
```

### Step 3: System Validation

```xml
<!-- Validate system capabilities -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>get_system_info</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>

<!-- Check for any existing performance baselines -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>benchmark_memory_performance</tool_name>
<arguments>{
  "test_duration": 30,
  "memory_patterns": ["sequential", "random"]
}</arguments>
</use_mcp_tool>
```

### Step 4: Implementation

Based on the research and documentation, implement the authentication system:

```python
# auth/models.py
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    email: Optional[str] = None
```

```python
# auth/security.py
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

# Configuration from research and documentation
SECRET_KEY = "your-secret-key-here"  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return email."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except jwt.PyJWTError:
        return None
```

### Step 5: Performance Validation

```xml
<!-- Test the implementation performance -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>run_integration_tests</tool_name>
<arguments>{
  "test_suite": "auth_performance",
  "parallel": true,
  "verbose": false
}</arguments>
</use_mcp_tool>

<!-- Monitor memory usage during authentication operations -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>optimize_memory_usage</tool_name>
<arguments>{
  "optimization_level": "balanced",
  "target_memory_mb": 512
}</arguments>
</use_mcp_tool>
```

## Example 2: Optimizing a Machine Learning Pipeline

This example shows how to use MCP tools to optimize a vector similarity search system.

### Step 1: Performance Baseline

```xml
<!-- Establish current performance metrics -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>benchmark_vector_operations</tool_name>
<arguments>{
  "vector_count": 100000,
  "vector_dimensions": 768,
  "operations": ["similarity", "clustering", "indexing"]
}</arguments>
</use_mcp_tool>

<!-- Analyze existing bottlenecks -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>analyze_performance_bottlenecks</tool_name>
<arguments>{
  "duration_seconds": 180,
  "include_memory_profile": true
}</arguments>
</use_mcp_tool>
```

### Step 2: Research Optimization Strategies

```xml
<!-- Research vector optimization techniques -->
<use_mcp_tool>
<server_name>perplexity</server_name>
<tool_name>PERPLEXITYAI_PERPLEXITY_AI_SEARCH</tool_name>
<arguments>{
  "userContent": "FAISS vector similarity search optimization for large-scale embeddings",
  "systemContent": "Focus on practical implementation strategies, indexing methods, and performance benchmarks",
  "model": "llama-3.1-sonar-large-128k-online",
  "return_citations": true,
  "temperature": 0.2
}</arguments>
</use_mcp_tool>

<!-- Get FAISS documentation -->
<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>resolve-library-id</tool_name>
<arguments>{"libraryName": "faiss"}</arguments>
</use_mcp_tool>

<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>get-library-docs</tool_name>
<arguments>{
  "context7CompatibleLibraryID": "/facebookresearch/faiss",
  "topic": "optimization",
  "tokens": 10000
}</arguments>
</use_mcp_tool>
```

### Step 3: Implementation with Optimization

```python
# vector_search/optimized_index.py
import faiss
import numpy as np
from typing import List, Tuple, Optional
import logging

class OptimizedVectorIndex:
    """
    Optimized vector similarity search using FAISS
    Based on research findings and official documentation
    """
    
    def __init__(self, dimension: int, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.is_trained = False
        
        # Initialize index based on research recommendations
        if index_type == "IVF":
            # Use IVF (Inverted File) for large datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "HNSW":
            # Use HNSW for high-precision requirements
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            # Fallback to flat index
            self.index = faiss.IndexFlatL2(dimension)
    
    def train_and_add(self, vectors: np.ndarray) -> None:
        """Train index and add vectors."""
        if not self.is_trained and hasattr(self.index, 'train'):
            logging.info(f"Training index with {len(vectors)} vectors")
            self.index.train(vectors)
            self.is_trained = True
        
        self.index.add(vectors)
        logging.info(f"Added {len(vectors)} vectors to index")
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained
        }
```

### Step 4: Performance Validation

```xml
<!-- Benchmark the optimized implementation -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>benchmark_vector_operations</tool_name>
<arguments>{
  "vector_count": 100000,
  "vector_dimensions": 768,
  "operations": ["similarity", "clustering", "indexing"]
}</arguments>
</use_mcp_tool>

<!-- Validate memory efficiency -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>validate_memory_operations</tool_name>
<arguments>{
  "test_data_size": 100,
  "include_stress_test": true
}</arguments>
</use_mcp_tool>
```

## Example 3: Real-time WebSocket Implementation

This example demonstrates building a scalable real-time communication system.

### Step 1: Architecture Research

```xml
<!-- Research WebSocket scaling patterns -->
<use_mcp_tool>
<server_name>perplexity</server_name>
<tool_name>PERPLEXITYAI_PERPLEXITY_AI_SEARCH</tool_name>
<arguments>{
  "userContent": "WebSocket scaling patterns for real-time applications with Redis and FastAPI",
  "systemContent": "Provide architectural guidance, implementation patterns, and scaling considerations",
  "model": "llama-3.1-sonar-large-128k-online",
  "return_citations": true
}</arguments>
</use_mcp_tool>

<!-- Get WebSocket documentation -->
<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>resolve-library-id</tool_name>
<arguments>{"libraryName": "websockets"}</arguments>
</use_mcp_tool>

<use_mcp_tool>
<server_name>context7</server_name>
<tool_name>get-library-docs</tool_name>
<arguments>{
  "context7CompatibleLibraryID": "/python-websockets/websockets",
  "topic": "server",
  "tokens": 6000
}</arguments>
</use_mcp_tool>
```

### Step 2: System Health Monitoring

```xml
<!-- Monitor system during real-time operations -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>monitor_system_health</tool_name>
<arguments>{
  "check_interval": 10,
  "alert_thresholds": {
    "memory_usage": 0.8,
    "cpu_usage": 0.7,
    "connection_count": 1000
  }
}</arguments>
</use_mcp_tool>
```

### Step 3: Implementation

```python
# websocket/connection_manager.py
from fastapi import WebSocket
from typing import List, Dict, Set
import json
import asyncio
import redis.asyncio as redis
from datetime import datetime

class ConnectionManager:
    """
    WebSocket connection manager with Redis pub/sub for scaling
    Based on research findings for real-time applications
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.redis_client = None
        self.redis_url = redis_url
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept WebSocket connection and register it."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Initialize Redis connection if not exists
        if not self.redis_client:
            self.redis_client = redis.from_url(self.redis_url)
            
        # Subscribe to user-specific channel
        await self._subscribe_to_channel(f"user:{user_id}")
        
    async def disconnect(self, connection_id: str, user_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message)
    
    async def broadcast_to_user(self, message: str, user_id: str):
        """Send message to all connections of a user."""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_all(self, message: str):
        """Broadcast message to all active connections."""
        for connection_id in self.active_connections:
            await self.send_personal_message(message, connection_id)
    
    async def _subscribe_to_channel(self, channel: str):
        """Subscribe to Redis channel for distributed messaging."""
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        # Start background task to handle Redis messages
        asyncio.create_task(self._handle_redis_messages(pubsub))
    
    async def _handle_redis_messages(self, pubsub):
        """Handle incoming Redis pub/sub messages."""
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await self._process_redis_message(data)
    
    async def _process_redis_message(self, data: dict):
        """Process Redis message and forward to appropriate connections."""
        message_type = data.get('type')
        target = data.get('target')
        content = data.get('content')
        
        if message_type == 'user_message' and target:
            await self.broadcast_to_user(content, target)
        elif message_type == 'broadcast':
            await self.broadcast_to_all(content)
```

### Step 4: Performance Testing

```xml
<!-- Test WebSocket performance under load -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>benchmark_mcp_throughput</tool_name>
<arguments>{
  "request_count": 1000,
  "concurrent_connections": 50,
  "payload_size": "medium"
}</arguments>
</use_mcp_tool>

<!-- Monitor system health during load testing -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>analyze_performance_bottlenecks</tool_name>
<arguments>{
  "duration_seconds": 300,
  "include_memory_profile": true
}</arguments>
</use_mcp_tool>
```

## Example 4: Error Handling and Recovery

This example shows how to implement robust error handling with MCP integration.

### Step 1: Research Error Handling Patterns

```xml
<!-- Research error handling best practices -->
<use_mcp_tool>
<server_name>perplexity</server_name>
<tool_name>PERPLEXITYAI_PERPLEXITY_AI_SEARCH</tool_name>
<arguments>{
  "userContent": "Python error handling patterns for production applications with logging and monitoring",
  "systemContent": "Focus on practical implementation strategies and monitoring integration",
  "temperature": 0.3
}</arguments>
</use_mcp_tool>
```

### Step 2: Implementation with MCP Integration

```python
# utils/error_handling.py
import logging
import traceback
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime

class MCPErrorHandler:
    """
    Error handling utility with MCP integration for monitoring and recovery
    """
    
    def __init__(self, safla_client=None):
        self.safla_client = safla_client
        self.error_count = 0
        self.last_error_time = None
        
    def handle_mcp_error(self, operation: str, error: Exception) -> Dict[str, Any]:
        """Handle MCP-related errors with fallback strategies."""
        self.error_count += 1
        self.last_error_time = datetime.utcnow()
        
        error_info = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": self.last_error_time.isoformat(),
            "error_count": self.error_count
        }
        
        # Log error details
        logging.error(f"MCP Error in {operation}: {error}", exc_info=True)
        
        # Report to SAFLA if available
        if self.safla_client:
            try:
                self._report_error_to_safla(error_info)
            except Exception as report_error:
                logging.warning(f"Failed to report error to SAFLA: {report_error}")
        
        return error_info
    
    def _report_error_to_safla(self, error_info: Dict[str, Any]):
        """Report error to SAFLA monitoring system."""
        # This would use the SAFLA MCP client to report errors
        # Implementation depends on SAFLA's error reporting capabilities
        pass
    
    def with_fallback(self, primary_func, fallback_func, *args, **kwargs):
        """Execute function with fallback on error."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.handle_mcp_error(primary_func.__name__, e)
            return fallback_func(*args, **kwargs)

def mcp_error_handler(fallback_result=None):
    """Decorator for MCP operation error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return fallback_result
        return wrapper
    return decorator

# Example usage
@mcp_error_handler(fallback_result={"error": "Documentation unavailable"})
async def get_documentation_with_fallback(library_name: str):
    """Get documentation with automatic fallback."""
    # Primary: Use Context7
    try:
        library_id = await resolve_library_id(library_name)
        docs = await get_library_docs(library_id)
        return docs
    except Exception:
        # Fallback: Use Perplexity
        research_result = await search_with_perplexity(
            f"Official documentation for {library_name}"
        )
        return research_result
```

### Step 3: Monitoring Integration

```xml
<!-- Set up continuous monitoring -->
<use_mcp_tool>
<server_name>safla</server_name>
<tool_name>monitor_system_health</tool_name>
<arguments>{
  "check_interval": 30,
  "alert_thresholds": {
    "error_rate": 0.05,
    "response_time": 5000
  }
}</arguments>
</use_mcp_tool>
```

## Best Practices Summary

### 1. Always Use MCP Tools in Sequence
- Research first with Perplexity for current best practices
- Resolve library IDs before getting documentation
- Validate system capabilities before implementation
- Monitor performance throughout development

### 2. Implement Robust Error Handling
- Use fallback strategies for MCP tool failures
- Log all errors with sufficient context
- Implement circuit breaker patterns for external services
- Monitor error rates and system health

### 3. Optimize Performance Continuously
- Establish baselines before optimization
- Use SAFLA tools for performance monitoring
- Implement caching for frequently accessed data
- Monitor memory usage and optimize as needed

### 4. Security-First Approach
- Research security best practices for all implementations
- Validate all external inputs and outputs
- Use secure communication channels
- Implement proper authentication and authorization

### 5. Documentation and Knowledge Sharing
- Document all MCP integration patterns
- Share successful optimization strategies
- Maintain up-to-date dependency information
- Create reusable templates for common patterns

These examples demonstrate the practical application of the MCP Intelligent Coder mode, showing how to leverage external tools and services to create robust, well-documented, and optimized software solutions.