# MCP Tutorial Mode - Interactive Examples

## Overview

This document provides detailed interactive examples and hands-on exercises for the MCP Tutorial mode. Each example includes step-by-step instructions, code templates, validation criteria, and extension challenges to support progressive learning.

## Beginner Level Examples

### Example 1: "Hello MCP" - Your First Server

#### Learning Objectives
- Understand basic MCP server structure
- Implement a simple tool
- Test server-client communication
- Handle basic error scenarios

#### Interactive Tutorial Structure

```yaml
tutorial_flow:
  introduction:
    duration: "15 minutes"
    content:
      - "What is an MCP server?"
      - "Basic architecture overview"
      - "Tool vs Resource concepts"
      - "Communication protocol basics"
    
    interactive_elements:
      - "Architecture diagram exploration"
      - "Protocol message examples"
      - "Live server demonstration"

  hands_on_coding:
    duration: "45 minutes"
    phases:
      - "Server setup and initialization"
      - "Tool implementation"
      - "Client connection and testing"
      - "Error handling and debugging"

  validation_and_extension:
    duration: "30 minutes"
    activities:
      - "Functionality testing"
      - "Code review and optimization"
      - "Extension challenges"
      - "Reflection and documentation"
```

#### Step-by-Step Implementation

**Step 1: Server Foundation**
```python
# File: hello_mcp_server.py
"""
Hello MCP Server - A beginner's introduction to MCP
This server demonstrates basic MCP concepts with simple tools
"""

import asyncio
import json
from typing import Any, Dict, List
from mcp import Server, Tool, Resource
from mcp.types import TextContent, ImageContent

class HelloMCPServer(Server):
    """
    A simple MCP server for learning basic concepts
    """
    
    def __init__(self):
        super().__init__(
            name="hello-mcp-server",
            version="1.0.0",
            description="A beginner-friendly MCP server for learning"
        )
        
        # Register our tools
        self.register_tools()
        
        # Initialize any state we need
        self.greeting_count = 0
        self.user_preferences = {}
    
    def register_tools(self):
        """Register all available tools"""
        
        # Simple greeting tool
        @self.tool("say_hello")
        async def say_hello(name: str = "World") -> str:
            """
            A simple greeting tool that says hello to someone
            
            Args:
                name: The name to greet (default: "World")
            
            Returns:
                A personalized greeting message
            """
            self.greeting_count += 1
            return f"Hello, {name}! This is greeting #{self.greeting_count}"
        
        # Calculator tool with error handling
        @self.tool("calculate")
        async def calculate(operation: str, a: float, b: float) -> Dict[str, Any]:
            """
            Perform basic mathematical operations
            
            Args:
                operation: The operation to perform (+, -, *, /)
                a: First number
                b: Second number
            
            Returns:
                Dictionary with result and operation details
            """
            try:
                if operation == "+":
                    result = a + b
                elif operation == "-":
                    result = a - b
                elif operation == "*":
                    result = a * b
                elif operation == "/":
                    if b == 0:
                        raise ValueError("Division by zero is not allowed")
                    result = a / b
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                return {
                    "result": result,
                    "operation": f"{a} {operation} {b}",
                    "success": True
                }
            
            except Exception as e:
                return {
                    "error": str(e),
                    "operation": f"{a} {operation} {b}",
                    "success": False
                }

# TODO: Add your implementation here
# Students will implement the server startup and client connection code
```

**Student Exercise: Complete the Implementation**
```python
# Students complete this section:

async def main():
    """
    Main function to start the MCP server
    TODO: Implement server startup logic
    """
    # Hint: Create server instance, start it, and handle connections
    pass

if __name__ == "__main__":
    # TODO: Run the main function
    pass
```

**Validation Checklist:**
- [ ] Server starts without errors
- [ ] Tools are properly registered
- [ ] `say_hello` tool works with and without parameters
- [ ] `calculate` tool handles all operations correctly
- [ ] Error handling works for division by zero
- [ ] Invalid operations return appropriate errors

**Extension Challenges:**
1. **Beginner**: Add a `get_server_info` tool that returns server statistics
2. **Intermediate**: Implement user preference storage and retrieval
3. **Advanced**: Add input validation and sanitization

### Example 2: File System Explorer

#### Learning Objectives
- Work with file system operations
- Implement resource providers
- Handle security considerations
- Create user-friendly interfaces

#### Interactive Tutorial

```yaml
tutorial_progression:
  concept_introduction:
    topics:
      - "File system security in MCP"
      - "Resource vs Tool for file operations"
      - "Path validation and sanitization"
      - "Error handling for file operations"
  
  implementation_phases:
    phase_1: "Basic file listing tool"
    phase_2: "File content reading resource"
    phase_3: "Search functionality"
    phase_4: "Security and validation"
```

**Implementation Template:**
```python
# File: file_explorer_server.py
import os
import pathlib
from typing import List, Dict, Any, Optional
from mcp import Server, Tool, Resource

class FileExplorerServer(Server):
    """
    MCP server for safe file system exploration
    """
    
    def __init__(self, allowed_paths: List[str] = None):
        super().__init__(
            name="file-explorer",
            version="1.0.0",
            description="Safe file system exploration server"
        )
        
        # Security: Define allowed paths
        self.allowed_paths = allowed_paths or [os.getcwd()]
        self.allowed_paths = [pathlib.Path(p).resolve() for p in self.allowed_paths]
        
        self.register_tools()
        self.register_resources()
    
    def _is_path_allowed(self, path: str) -> bool:
        """
        Check if a path is within allowed directories
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is allowed, False otherwise
        """
        try:
            resolved_path = pathlib.Path(path).resolve()
            return any(
                str(resolved_path).startswith(str(allowed))
                for allowed in self.allowed_paths
            )
        except Exception:
            return False
    
    def register_tools(self):
        """Register file system tools"""
        
        @self.tool("list_directory")
        async def list_directory(
            path: str = ".",
            show_hidden: bool = False,
            include_details: bool = False
        ) -> Dict[str, Any]:
            """
            List contents of a directory
            
            Args:
                path: Directory path to list
                show_hidden: Include hidden files
                include_details: Include file size, modification time, etc.
            
            Returns:
                Dictionary with directory contents and metadata
            """
            # TODO: Student implements this
            # Hints:
            # 1. Validate path using _is_path_allowed
            # 2. Use pathlib for path operations
            # 3. Handle permissions errors gracefully
            # 4. Return structured data with files and directories
            pass
        
        @self.tool("search_files")
        async def search_files(
            pattern: str,
            path: str = ".",
            file_types: List[str] = None,
            max_results: int = 100
        ) -> Dict[str, Any]:
            """
            Search for files matching a pattern
            
            Args:
                pattern: Search pattern (supports wildcards)
                path: Directory to search in
                file_types: List of file extensions to include
                max_results: Maximum number of results to return
            
            Returns:
                Dictionary with search results and metadata
            """
            # TODO: Student implements this
            # Hints:
            # 1. Use glob or regex for pattern matching
            # 2. Implement file type filtering
            # 3. Limit results to prevent overwhelming responses
            # 4. Include match context and relevance scoring
            pass
    
    def register_resources(self):
        """Register file system resources"""
        
        @self.resource("file://")
        async def file_resource(uri: str) -> str:
            """
            Provide file contents as a resource
            
            Args:
                uri: File URI (e.g., "file:///path/to/file.txt")
            
            Returns:
                File contents as string
            """
            # TODO: Student implements this
            # Hints:
            # 1. Parse URI to extract file path
            # 2. Validate path permissions
            # 3. Handle binary vs text files appropriately
            # 4. Implement caching for frequently accessed files
            pass

# Student Exercise Areas:
# 1. Complete the tool implementations
# 2. Add error handling and validation
# 3. Implement the resource provider
# 4. Add security measures and logging
```

**Guided Exercise Questions:**
1. How would you handle symbolic links safely?
2. What security risks exist with file system access?
3. How can you optimize performance for large directories?
4. What metadata would be most useful to include?

## Intermediate Level Examples

### Example 3: Multi-Service Integration Hub

#### Learning Objectives
- Integrate multiple external services
- Manage service state and configuration
- Implement caching and performance optimization
- Handle service failures gracefully

#### Architecture Overview
```yaml
system_design:
  services:
    weather_service:
      purpose: "Provide weather information"
      api: "OpenWeatherMap API"
      caching: "1 hour TTL"
      fallback: "Cached data or default message"
    
    news_service:
      purpose: "Fetch latest news"
      api: "NewsAPI"
      caching: "30 minutes TTL"
      fallback: "Cached headlines"
    
    task_service:
      purpose: "Manage todo items"
      storage: "Local JSON file"
      backup: "Automatic daily backup"
      sync: "Optional cloud sync"
  
  integration_patterns:
    - "Service abstraction layer"
    - "Unified error handling"
    - "Configuration management"
    - "Health monitoring"
```

**Implementation Framework:**
```python
# File: integration_hub_server.py
import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ServiceConfig:
    """Configuration for external services"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    cache_ttl: int = 3600
    retry_attempts: int = 3

class ServiceInterface(ABC):
    """Abstract base class for service integrations"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.cache = {}
        self.last_health_check = 0
        self.is_healthy = True
    
    @abstractmethod
    async def call_api(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to the service"""
        pass
    
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        # TODO: Student implements health check logic
        pass
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data if still valid"""
        # TODO: Student implements caching logic
        pass
    
    def set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with TTL"""
        # TODO: Student implements cache storage
        pass

class WeatherService(ServiceInterface):
    """Weather service integration"""
    
    async def get_current_weather(self, city: str) -> Dict[str, Any]:
        """
        Get current weather for a city
        
        Args:
            city: City name
            
        Returns:
            Weather data dictionary
        """
        # TODO: Student implements weather API integration
        # Hints:
        # 1. Check cache first
        # 2. Make API call if cache miss
        # 3. Handle API errors gracefully
        # 4. Store result in cache
        pass
    
    async def get_forecast(self, city: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast"""
        # TODO: Student implements forecast functionality
        pass

class NewsService(ServiceInterface):
    """News service integration"""
    
    async def get_headlines(self, category: str = "general", country: str = "us") -> List[Dict[str, Any]]:
        """
        Get latest news headlines
        
        Args:
            category: News category
            country: Country code
            
        Returns:
            List of news articles
        """
        # TODO: Student implements news API integration
        pass
    
    async def search_news(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for news articles"""
        # TODO: Student implements news search
        pass

class TaskService(ServiceInterface):
    """Local task management service"""
    
    def __init__(self, config: ServiceConfig, data_file: str = "tasks.json"):
        super().__init__(config)
        self.data_file = data_file
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from file"""
        # TODO: Student implements task loading
        pass
    
    def _save_tasks(self) -> None:
        """Save tasks to file"""
        # TODO: Student implements task saving
        pass
    
    async def add_task(self, title: str, description: str = "", priority: str = "medium") -> Dict[str, Any]:
        """Add a new task"""
        # TODO: Student implements task creation
        pass
    
    async def list_tasks(self, status: str = "all") -> List[Dict[str, Any]]:
        """List tasks with optional status filter"""
        # TODO: Student implements task listing
        pass
    
    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing task"""
        # TODO: Student implements task updates
        pass

class IntegrationHubServer(Server):
    """Main MCP server that orchestrates all services"""
    
    def __init__(self):
        super().__init__(
            name="integration-hub",
            version="2.0.0",
            description="Multi-service integration hub"
        )
        
        # Initialize services
        self.services = self._initialize_services()
        
        # Register tools and resources
        self.register_tools()
        self.register_resources()
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
    
    def _initialize_services(self) -> Dict[str, ServiceInterface]:
        """Initialize all service integrations"""
        # TODO: Student implements service initialization
        # Hints:
        # 1. Load configuration from environment or config file
        # 2. Create service instances with proper config
        # 3. Validate API keys and connectivity
        pass
    
    async def _health_monitor(self):
        """Background task to monitor service health"""
        # TODO: Student implements health monitoring
        # Hints:
        # 1. Periodic health checks for all services
        # 2. Update service status
        # 3. Log health changes
        # 4. Implement circuit breaker pattern
        pass
    
    def register_tools(self):
        """Register all available tools"""
        
        @self.tool("get_weather")
        async def get_weather(city: str, include_forecast: bool = False) -> Dict[str, Any]:
            """Get weather information for a city"""
            # TODO: Student implements weather tool
            pass
        
        @self.tool("get_news")
        async def get_news(category: str = "general", limit: int = 5) -> Dict[str, Any]:
            """Get latest news headlines"""
            # TODO: Student implements news tool
            pass
        
        @self.tool("manage_tasks")
        async def manage_tasks(action: str, **kwargs) -> Dict[str, Any]:
            """Manage tasks (add, list, update, delete)"""
            # TODO: Student implements task management tool
            pass
        
        @self.tool("service_status")
        async def service_status() -> Dict[str, Any]:
            """Get status of all integrated services"""
            # TODO: Student implements status reporting
            pass

# Student Exercise Areas:
# 1. Complete all service implementations
# 2. Add comprehensive error handling
# 3. Implement caching and performance optimization
# 4. Add configuration management
# 5. Create health monitoring and alerting
```

**Learning Challenges:**
1. **Error Handling**: How do you handle cascading failures?
2. **Performance**: What caching strategies work best?
3. **Security**: How do you safely store and use API keys?
4. **Monitoring**: What metrics are most important to track?

## Advanced Level Examples

### Example 4: Real-Time Data Processing Pipeline

#### Learning Objectives
- Implement streaming data processing
- Handle high-throughput scenarios
- Create scalable architectures
- Integrate with message queues and databases

#### System Architecture
```yaml
pipeline_design:
  data_sources:
    - "Real-time sensor data"
    - "API webhooks"
    - "File uploads"
    - "Message queue consumers"
  
  processing_stages:
    - "Data ingestion and validation"
    - "Transformation and enrichment"
    - "Analysis and pattern detection"
    - "Output routing and storage"
  
  scalability_features:
    - "Horizontal scaling support"
    - "Load balancing"
    - "Circuit breakers"
    - "Backpressure handling"
```

**Advanced Implementation:**
```python
# File: data_pipeline_server.py
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
import aioredis
import aiokafka
from sqlalchemy.ext.asyncio import create_async_engine

@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    max_batch_size: int = 1000
    batch_timeout: float = 5.0
    max_queue_size: int = 10000
    worker_count: int = 4
    redis_url: str = "redis://localhost:6379"
    kafka_bootstrap_servers: str = "localhost:9092"
    database_url: str = "postgresql+asyncpg://user:pass@localhost/db"

class DataProcessor:
    """Base class for data processing stages"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.metrics = {
            "processed_count": 0,
            "error_count": 0,
            "processing_time": 0.0
        }
    
    async def process(self, data: Any) -> Any:
        """Process a single data item"""
        start_time = time.time()
        try:
            result = await self._process_impl(data)
            self.metrics["processed_count"] += 1
            return result
        except Exception as e:
            self.metrics["error_count"] += 1
            raise
        finally:
            self.metrics["processing_time"] += time.time() - start_time
    
    async def _process_impl(self, data: Any) -> Any:
        """Override this method in subclasses"""
        return data

class ValidationProcessor(DataProcessor):
    """Validates incoming data"""
    
    async def _process_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data structure and content
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Validated data dictionary
            
        Raises:
            ValueError: If data is invalid
        """
        # TODO: Student implements validation logic
        # Hints:
        # 1. Check required fields
        # 2. Validate data types
        # 3. Check value ranges
        # 4. Sanitize inputs
        pass

class EnrichmentProcessor(DataProcessor):
    """Enriches data with additional information"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.cache = {}
        self.external_apis = {}
    
    async def _process_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich data with additional context
        
        Args:
            data: Validated data dictionary
            
        Returns:
            Enriched data dictionary
        """
        # TODO: Student implements enrichment logic
        # Hints:
        # 1. Add timestamps and metadata
        # 2. Lookup additional data from external sources
        # 3. Calculate derived fields
        # 4. Add geolocation or other context
        pass

class AnalysisProcessor(DataProcessor):
    """Performs analysis and pattern detection"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.pattern_detectors = []
        self.anomaly_threshold = 2.0
        self.historical_data = deque(maxlen=1000)
    
    async def _process_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data for patterns and anomalies
        
        Args:
            data: Enriched data dictionary
            
        Returns:
            Data with analysis results
        """
        # TODO: Student implements analysis logic
        # Hints:
        # 1. Statistical analysis
        # 2. Pattern matching
        # 3. Anomaly detection
        # 4. Trend analysis
        pass

class DataPipeline:
    """Main data processing pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processors: List[DataProcessor] = []
        self.input_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.workers = []
        self.is_running = False
        
        # Initialize external connections
        self.redis = None
        self.kafka_producer = None
        self.database = None
    
    async def initialize(self):
        """Initialize external connections"""
        # TODO: Student implements initialization
        # Hints:
        # 1. Connect to Redis for caching
        # 2. Initialize Kafka producer
        # 3. Set up database connection
        # 4. Validate all connections
        pass
    
    def add_processor(self, processor: DataProcessor):
        """Add a processing stage to the pipeline"""
        self.processors.append(processor)
    
    async def start(self):
        """Start the pipeline workers"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.config.worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start output handler
        asyncio.create_task(self._output_handler())
    
    async def stop(self):
        """Stop the pipeline gracefully"""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def _worker(self, worker_id: str):
        """Worker task that processes data"""
        while self.is_running:
            try:
                # Get data from input queue
                data = await asyncio.wait_for(
                    self.input_queue.get(),
                    timeout=1.0
                )
                
                # Process through all stages
                for processor in self.processors:
                    data = await processor.process(data)
                
                # Put result in output queue
                await self.output_queue.put(data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                # TODO: Student implements error handling
                # Hints:
                # 1. Log the error
                # 2. Send to dead letter queue
                # 3. Update metrics
                # 4. Continue processing
                pass
    
    async def _output_handler(self):
        """Handle processed data output"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Try to get data with timeout
                data = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=1.0
                )
                
                batch.append(data)
                
                # Check if we should flush the batch
                should_flush = (
                    len(batch) >= self.config.max_batch_size or
                    time.time() - last_batch_time >= self.config.batch_timeout
                )
                
                if should_flush and batch:
                    await self._flush_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                    
            except asyncio.TimeoutError:
                # Flush partial batch on timeout
                if batch:
                    await self._flush_batch(batch)
                    batch = []
                    last_batch_time = time.time()
    
    async def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Flush a batch of processed data"""
        # TODO: Student implements batch output
        # Hints:
        # 1. Send to Kafka topic
        # 2. Store in database
        # 3. Update Redis cache
        # 4. Send notifications if needed
        pass
    
    async def ingest_data(self, data: Any):
        """Ingest data into the pipeline"""
        await self.input_queue.put(data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        # TODO: Student implements metrics collection
        # Hints:
        # 1. Aggregate processor metrics
        # 2. Add queue sizes
        # 3. Calculate throughput
        # 4. Include error rates
        pass

class DataPipelineServer(Server):
    """MCP server for data pipeline management"""
    
    def __init__(self):
        super().__init__(
            name="data-pipeline",
            version="3.0.0",
            description="Real-time data processing pipeline"
        )
        
        # Initialize pipeline
        config = PipelineConfig()
        self.pipeline = DataPipeline(config)
        
        # Set up processors
        self._setup_processors()
        
        # Register tools and resources
        self.register_tools()
        self.register_resources()
    
    def _setup_processors(self):
        """Set up the processing pipeline"""
        # TODO: Student implements processor setup
        # Hints:
        # 1. Add validation processor
        # 2. Add enrichment processor
        # 3. Add analysis processor
        # 4. Configure each processor appropriately
        pass
    
    def register_tools(self):
        """Register pipeline management tools"""
        
        @self.tool("ingest_data")
        async def ingest_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """Ingest data into the processing pipeline"""
            # TODO: Student implements data ingestion
            pass
        
        @self.tool("get_pipeline_status")
        async def get_pipeline_status() -> Dict[str, Any]:
            """Get current pipeline status and metrics"""
            # TODO: Student implements status reporting
            pass
        
        @self.tool("configure_pipeline")
        async def configure_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
            """Update pipeline configuration"""
            # TODO: Student implements configuration updates
            pass
        
        @self.tool("start_pipeline")
        async def start_pipeline() -> Dict[str, Any]:
            """Start the data processing pipeline"""
            # TODO: Student implements pipeline startup
            pass
        
        @self.tool("stop_pipeline")
        async def stop_pipeline() -> Dict[str, Any]:
            """Stop the data processing pipeline"""
            # TODO: Student implements pipeline shutdown
            pass

# Student Exercise Areas:
# 1. Complete all processor implementations
# 2. Add comprehensive error handling and recovery
# 3. Implement monitoring and alerting
# 4. Add configuration management
# 5. Create performance optimization features
# 6. Add data quality monitoring
```

**Advanced Challenges:**
1. **Scalability**: How do you handle millions of events per second?
2. **Reliability**: What happens when components fail?
3. **Monitoring**: How do you detect and respond to issues?
4. **Optimization**: How do you minimize latency and maximize throughput?

## Assessment and Validation Framework

### Automated Testing Suite

```python
# File: tutorial_test_framework.py
import asyncio
import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

class TutorialTestFramework:
    """Framework for testing tutorial implementations"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
    
    async def test_hello_mcp_server(self, server_instance) -> Dict[str, Any]:
        """Test the Hello MCP server implementation"""
        results = {
            "basic_functionality": False,
            "error_handling": False,
            "performance": False,
            "code_quality": False
        }
        
        try:
            # Test basic functionality
            greeting = await server_instance.call_tool("say_hello", {"name": "Student"})
            assert "Hello, Student" in greeting
            results["basic_functionality"] = True
            
            # Test calculator
            calc_result = await server_instance.call_tool("calculate", {
                "operation": "+",
                "a": 5,
                "b": 3
            })
            assert calc_result["result"] == 8
            assert calc_result["success"] is True
            
            # Test error handling
            error_result = await server_instance.call_tool("calculate", {
                "operation": "/",
                "a": 5,
                "b": 0
            })
            assert error_result["success"] is False
            assert "Division by zero" in error_result["error"]
            results["error_handling"] = True
            
            # Performance test
            start_time = time.time()
            for _ in range(100):
                await server_instance.call_tool("say_hello", {"name": "Test"})
            duration = time.time() - start_time
            
            if duration < 1.0:  # Should handle 100 calls in under 1 second
                results["performance"] = True
            
            # Code quality checks (simplified)
            # In real implementation, this would use static analysis tools
            results["code_quality"] = True
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    async def test_file_explorer_server(self, server_instance) -> Dict[str, Any]:
        """Test the File Explorer server implementation"""
        # TODO: Implement comprehensive file explorer tests
        pass
    
    async def test_integration_hub_server(self, server_instance) -> Dict[str, Any]:
        """Test the Integration Hub server implementation"""
        # TODO: Implement integration hub tests
        pass
    
    async def test_data_pipeline_server(self, server_instance) -> Dict[str, Any]:
        """Test the Data Pipeline server implementation"""
        # TODO: Implement data pipeline tests
        pass
    
    def generate_feedback(self, test_results: Dict[str, Any]) -> str:
        """Generate personalized feedback based on test results"""
        feedback = []
        
        if not test_results.get("basic_functionality", False):
            feedback.append("❌ Basic functionality needs work. Review the tool implementation patterns.")
        else:
            feedback.append("✅ Basic functionality working correctly!")
        
        if not test_results.get("error_handling", False):
            feedback.append("❌ Error handling needs improvement. Add try-catch blocks and validation.")
        else:
            feedback.append("✅ Error handling implemented well!")
        
        if not test_results.get("performance", False):
            feedback.append("⚠️ Performance could be better. Consider optimization techniques.")
        else:
            feedback.append("✅ Performance meets requirements!")
        
        # Add specific suggestions based on common issues
        if "error" in test_results:
            feedback.append(f"🔍 Debug this error: {test_results['error']}")
        
        return "\n".join(feedback)
```

### Interactive Learning Aids

```yaml
learning_aids:
  visual_debugger:
    features:
      - "Step-through code execution"
      - "Variable state visualization"
      - "Call stack inspection"
      - "Performance profiling"
  
  interactive_documentation:
    features:
      - "Searchable API reference"
      - "Interactive code examples"
      - "Best practices guide"
      - "Common patterns library"
  
  collaboration_tools:
    features:
      - "Shared coding sessions"
      - "Peer review system"
      - "Discussion forums"
      - "Expert office hours"
  
  progress_tracking:
    features:
      - "Skill progression visualization"
      - "Achievement badges"
      - "Learning path recommendations"
      - "Performance analytics"
```

This comprehensive tutorial framework provides hands-on learning experiences that progress from basic concepts to advanced implementations, ensuring learners develop both theoretical understanding and practical skills in MCP development.