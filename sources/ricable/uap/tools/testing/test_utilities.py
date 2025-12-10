# UAP Testing Utilities
"""
Comprehensive testing utilities for UAP development.
Provides mock services, test fixtures, performance testing, and development helpers.
"""

import asyncio
import json
import time
import logging
import random
import string
from typing import Dict, Any, List, Optional, Callable, Union, AsyncGenerator
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path

# Test data generation
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """Represents a test scenario for agent testing."""
    name: str
    description: str
    input_message: str
    expected_response_type: str
    expected_keywords: List[str] = None
    framework: str = "auto"
    context: Dict[str, Any] = None
    timeout_seconds: float = 10.0
    
    def __post_init__(self):
        if self.expected_keywords is None:
            self.expected_keywords = []
        if self.context is None:
            self.context = {}


class MockAgentFramework:
    """Mock agent framework for testing."""
    
    def __init__(self, framework_name: str, response_delay: float = 0.1, 
                 error_rate: float = 0.0, custom_responses: Optional[Dict[str, str]] = None):
        self.framework_name = framework_name
        self.response_delay = response_delay
        self.error_rate = error_rate
        self.custom_responses = custom_responses or {}
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock message processing."""
        self.call_count += 1
        
        # Record call history
        call_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "context": context.copy(),
            "call_number": self.call_count
        }
        self.call_history.append(call_record)
        
        # Simulate processing delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Simulate errors
        if random.random() < self.error_rate:
            raise Exception(f"Mock error from {self.framework_name} framework")
        
        # Check for custom responses
        for pattern, response in self.custom_responses.items():
            if pattern.lower() in message.lower():
                return {
                    "content": response,
                    "metadata": {
                        "framework": self.framework_name,
                        "mock": True,
                        "pattern_matched": pattern
                    }
                }
        
        # Default response
        return {
            "content": f"Mock response from {self.framework_name}: {message}",
            "metadata": {
                "framework": self.framework_name,
                "mock": True,
                "call_count": self.call_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Mock status."""
        return {
            "framework": self.framework_name,
            "status": "mock_active",
            "call_count": self.call_count,
            "error_rate": self.error_rate,
            "response_delay": self.response_delay
        }
    
    async def initialize(self) -> None:
        """Mock initialization."""
        pass
    
    def reset_stats(self) -> None:
        """Reset call statistics."""
        self.call_count = 0
        self.call_history.clear()


class MockWebSocketServer:
    """Mock WebSocket server for testing."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.connections: List[Any] = []
        self.message_history: List[Dict[str, Any]] = []
        
    async def start(self) -> None:
        """Start the mock WebSocket server."""
        import websockets
        
        async def handle_client(websocket, path):
            self.connections.append(websocket)
            try:
                async for message in websocket:
                    # Record message
                    self.message_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "direction": "received",
                        "message": message,
                        "path": path
                    })
                    
                    # Echo response
                    response = {
                        "type": "text_message_content",
                        "content": f"Mock WebSocket echo: {message}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await websocket.send(json.dumps(response))
                    
                    self.message_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "direction": "sent",
                        "message": json.dumps(response),
                        "path": path
                    })
                    
            except Exception as e:
                logger.debug(f"WebSocket connection closed: {e}")
            finally:
                if websocket in self.connections:
                    self.connections.remove(websocket)
        
        self.server = await websockets.serve(handle_client, "localhost", self.port)
        logger.info(f"Mock WebSocket server started on port {self.port}")
    
    async def stop(self) -> None:
        """Stop the mock WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Mock WebSocket server stopped")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if self.connections:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[conn.send(message_json) for conn in self.connections],
                return_exceptions=True
            )


class MockHTTPServer:
    """Mock HTTP server for testing."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.app = None
        self.server = None
        self.request_history: List[Dict[str, Any]] = []
        self.custom_responses: Dict[str, Dict[str, Any]] = {}
        
    def add_endpoint(self, path: str, method: str = "GET", response: Dict[str, Any] = None,
                    status_code: int = 200, delay: float = 0.0) -> None:
        """Add a custom endpoint response."""
        self.custom_responses[f"{method.upper()} {path}"] = {
            "response": response or {"message": f"Mock response for {path}"},
            "status_code": status_code,
            "delay": delay
        }
    
    async def start(self) -> None:
        """Start the mock HTTP server."""
        from aiohttp import web, web_runner
        
        async def handle_request(request):
            # Record request
            self.request_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "query": dict(request.query)
            })
            
            # Check for custom response
            endpoint_key = f"{request.method} {request.path}"
            if endpoint_key in self.custom_responses:
                custom = self.custom_responses[endpoint_key]
                
                # Simulate delay
                if custom["delay"] > 0:
                    await asyncio.sleep(custom["delay"])
                
                return web.json_response(
                    custom["response"],
                    status=custom["status_code"]
                )
            
            # Default responses for common UAP endpoints
            if request.path == "/health":
                return web.json_response({
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "mock": True
                })
            
            elif request.path == "/api/status":
                return web.json_response({
                    "status": "active",
                    "frameworks": {
                        "copilot": {"status": "active", "mock": True},
                        "agno": {"status": "active", "mock": True},
                        "mastra": {"status": "active", "mock": True}
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "mock": True
                })
            
            elif request.path.startswith("/api/agents/") and request.path.endswith("/chat"):
                # Extract agent ID
                agent_id = request.path.split("/")[3]
                
                try:
                    request_data = await request.json()
                except:
                    request_data = {}
                
                return web.json_response({
                    "content": f"Mock response for agent {agent_id}: {request_data.get('message', 'No message')}",
                    "metadata": {
                        "agent_id": agent_id,
                        "framework": request_data.get("framework", "auto"),
                        "mock": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            
            # Default 404 response
            return web.json_response(
                {"error": f"Mock endpoint not found: {request.path}"},
                status=404
            )
        
        # Create application
        self.app = web.Application()
        self.app.router.add_route("*", "/{path:.*}", handle_request)
        
        # Start server
        runner = web_runner.AppRunner(self.app)
        await runner.setup()
        site = web_runner.TCPSite(runner, "localhost", self.port)
        await site.start()
        
        self.server = runner
        logger.info(f"Mock HTTP server started on port {self.port}")
    
    async def stop(self) -> None:
        """Stop the mock HTTP server."""
        if self.server:
            await self.server.cleanup()
            logger.info("Mock HTTP server stopped")


class TestDataGenerator:
    """Generates test data for various scenarios."""
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_agent_id() -> str:
        """Generate a random agent ID."""
        return f"agent-{TestDataGenerator.random_string(8)}"
    
    @staticmethod
    def random_message() -> str:
        """Generate a random test message."""
        messages = [
            "Hello, how are you?",
            "What is the weather like today?",
            "Can you help me with a task?",
            "Tell me a joke",
            "What are your capabilities?",
            "How does machine learning work?",
            "What is the meaning of life?",
            "Can you explain quantum physics?",
            "What should I have for lunch?",
            "How do I improve my productivity?"
        ]
        return random.choice(messages)
    
    @staticmethod
    def generate_conversation(length: int = 5) -> List[Dict[str, str]]:
        """Generate a test conversation."""
        conversation = []
        for i in range(length):
            conversation.append({
                "role": "user",
                "content": TestDataGenerator.random_message(),
                "timestamp": (datetime.utcnow() - timedelta(minutes=length-i)).isoformat()
            })
            conversation.append({
                "role": "assistant",
                "content": f"This is a test response {i+1}",
                "timestamp": (datetime.utcnow() - timedelta(minutes=length-i-0.5)).isoformat()
            })
        return conversation
    
    @staticmethod
    def generate_test_scenarios() -> List[TestScenario]:
        """Generate common test scenarios."""
        return [
            TestScenario(
                name="greeting",
                description="Test basic greeting functionality",
                input_message="Hello",
                expected_response_type="text",
                expected_keywords=["hello", "hi", "greeting"]
            ),
            TestScenario(
                name="capability_query",
                description="Test capability inquiry",
                input_message="What can you do?",
                expected_response_type="text",
                expected_keywords=["can", "help", "capabilities"]
            ),
            TestScenario(
                name="math_question",
                description="Test mathematical reasoning",
                input_message="What is 2 + 2?",
                expected_response_type="text",
                expected_keywords=["4", "four", "math"]
            ),
            TestScenario(
                name="complex_query",
                description="Test complex multi-part question",
                input_message="Can you explain machine learning and provide an example?",
                expected_response_type="text",
                expected_keywords=["machine learning", "example", "algorithm"],
                timeout_seconds=15.0
            ),
            TestScenario(
                name="context_awareness",
                description="Test context awareness",
                input_message="What did I just ask you?",
                expected_response_type="text",
                context={"previous_message": "What is machine learning?"}
            )
        ]
    
    @staticmethod
    def generate_load_test_data(count: int = 100) -> List[str]:
        """Generate messages for load testing."""
        base_messages = [
            "Hello",
            "How are you?",
            "What can you do?",
            "Tell me about yourself",
            "What is AI?",
            "Explain machine learning",
            "Help me solve a problem",
            "What's the weather?",
            "Can you help me?",
            "Thank you"
        ]
        
        messages = []
        for i in range(count):
            base_msg = random.choice(base_messages)
            messages.append(f"{base_msg} (test #{i+1})")
        
        return messages


class PerformanceTester:
    """Handles performance testing for agents."""
    
    def __init__(self, client):
        self.client = client
        self.results: List[Dict[str, Any]] = []
    
    async def run_latency_test(self, agent_id: str, message: str, iterations: int = 10) -> Dict[str, Any]:
        """Run latency test for a specific message."""
        latencies = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                response = await self.client.chat(agent_id, f"{message} (iteration {i+1})")
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                errors += 1
                logger.error(f"Error in latency test iteration {i+1}: {e}")
        
        if latencies:
            return {
                "test_type": "latency",
                "agent_id": agent_id,
                "message": message,
                "iterations": iterations,
                "successful_requests": len(latencies),
                "failed_requests": errors,
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
                "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0]
            }
        else:
            return {
                "test_type": "latency",
                "agent_id": agent_id,
                "error": "All requests failed",
                "failed_requests": errors
            }
    
    async def run_throughput_test(self, agent_id: str, messages: List[str], 
                                concurrent_requests: int = 5, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run throughput test."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        completed_requests = 0
        failed_requests = 0
        response_times = []
        
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def send_request(message: str) -> None:
            nonlocal completed_requests, failed_requests
            
            async with semaphore:
                request_start = time.perf_counter()
                
                try:
                    await self.client.chat(agent_id, message)
                    request_end = time.perf_counter()
                    
                    response_times.append((request_end - request_start) * 1000)
                    completed_requests += 1
                    
                except Exception as e:
                    failed_requests += 1
                    logger.debug(f"Request failed: {e}")
        
        # Send requests continuously for the duration
        tasks = []
        message_index = 0
        
        while time.time() < end_time:
            message = messages[message_index % len(messages)]
            task = asyncio.create_task(send_request(message))
            tasks.append(task)
            
            message_index += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Wait for remaining requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        return {
            "test_type": "throughput",
            "agent_id": agent_id,
            "duration_seconds": actual_duration,
            "concurrent_requests": concurrent_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "requests_per_second": completed_requests / actual_duration,
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "success_rate": (completed_requests / (completed_requests + failed_requests)) * 100 if (completed_requests + failed_requests) > 0 else 0
        }
    
    async def run_stress_test(self, agent_id: str, max_concurrent: int = 50, 
                            ramp_up_seconds: int = 60) -> Dict[str, Any]:
        """Run stress test with gradual load increase."""
        results = []
        messages = TestDataGenerator.generate_load_test_data(1000)
        
        for concurrent in range(1, max_concurrent + 1, 5):
            logger.info(f"Testing with {concurrent} concurrent requests")
            
            # Run short throughput test
            result = await self.run_throughput_test(
                agent_id, messages, concurrent_requests=concurrent, duration_seconds=30
            )
            
            result["concurrent_level"] = concurrent
            results.append(result)
            
            # Check if performance is degrading significantly
            if len(results) > 1:
                prev_rps = results[-2]["requests_per_second"]
                current_rps = result["requests_per_second"]
                
                if current_rps < prev_rps * 0.5:  # 50% degradation
                    logger.warning(f"Significant performance degradation at {concurrent} concurrent requests")
                    break
            
            # Brief pause between tests
            await asyncio.sleep(5)
        
        return {
            "test_type": "stress",
            "agent_id": agent_id,
            "max_concurrent_tested": concurrent,
            "ramp_up_results": results,
            "performance_summary": self._analyze_stress_results(results)
        }
    
    def _analyze_stress_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stress test results."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Find optimal concurrent level
        best_result = max(results, key=lambda r: r["requests_per_second"])
        
        # Find degradation point
        degradation_point = None
        for i in range(1, len(results)):
            prev_rps = results[i-1]["requests_per_second"]
            current_rps = results[i]["requests_per_second"]
            
            if current_rps < prev_rps * 0.8:  # 20% degradation
                degradation_point = results[i]["concurrent_level"]
                break
        
        return {
            "optimal_concurrent_requests": best_result["concurrent_level"],
            "max_requests_per_second": best_result["requests_per_second"],
            "degradation_point": degradation_point,
            "recommended_max_concurrent": degradation_point - 5 if degradation_point else best_result["concurrent_level"]
        }


class TestEnvironment:
    """Manages a complete test environment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mock_http_server = None
        self.mock_ws_server = None
        self.mock_frameworks: Dict[str, MockAgentFramework] = {}
        self.temp_files: List[Path] = []
        
    async def setup(self) -> None:
        """Set up the test environment."""
        logger.info("Setting up test environment")
        
        # Start mock servers
        if self.config.get("mock_http_server", True):
            self.mock_http_server = MockHTTPServer(
                port=self.config.get("http_port", 8001)
            )
            await self.mock_http_server.start()
        
        if self.config.get("mock_websocket_server", True):
            self.mock_ws_server = MockWebSocketServer(
                port=self.config.get("ws_port", 8766)
            )
            await self.mock_ws_server.start()
        
        # Set up mock frameworks
        for framework_name in ["copilot", "agno", "mastra"]:
            self.mock_frameworks[framework_name] = MockAgentFramework(
                framework_name=framework_name,
                response_delay=self.config.get("framework_delay", 0.1),
                error_rate=self.config.get("framework_error_rate", 0.0)
            )
        
        logger.info("Test environment ready")
    
    async def teardown(self) -> None:
        """Tear down the test environment."""
        logger.info("Tearing down test environment")
        
        # Stop servers
        if self.mock_http_server:
            await self.mock_http_server.stop()
        
        if self.mock_ws_server:
            await self.mock_ws_server.stop()
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        logger.info("Test environment cleaned up")
    
    @asynccontextmanager
    async def managed_environment(self):
        """Context manager for test environment."""
        await self.setup()
        try:
            yield self
        finally:
            await self.teardown()
    
    def create_temp_file(self, content: str, suffix: str = ".json") -> Path:
        """Create a temporary file."""
        temp_file = Path(tempfile.mktemp(suffix=suffix))
        with open(temp_file, 'w') as f:
            f.write(content)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def get_mock_framework(self, name: str) -> MockAgentFramework:
        """Get a mock framework by name."""
        return self.mock_frameworks.get(name)


# Pytest fixtures and helpers (if pytest is available)
try:
    import pytest
    
    @pytest.fixture
    async def test_environment():
        """Pytest fixture for test environment."""
        env = TestEnvironment()
        async with env.managed_environment():
            yield env
    
    @pytest.fixture
    def test_data_generator():
        """Pytest fixture for test data generator."""
        return TestDataGenerator()
    
    @pytest.fixture
    def mock_agent_framework():
        """Pytest fixture for mock agent framework."""
        return MockAgentFramework("test_framework")
    
    PYTEST_AVAILABLE = True
    
except ImportError:
    PYTEST_AVAILABLE = False


# Utility functions
async def run_scenario_test(client, scenario: TestScenario) -> Dict[str, Any]:
    """Run a single test scenario."""
    start_time = time.perf_counter()
    
    try:
        response = await asyncio.wait_for(
            client.chat(
                "test-agent",
                scenario.input_message,
                framework=scenario.framework,
                context=scenario.context
            ),
            timeout=scenario.timeout_seconds
        )
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Check if response contains expected keywords
        response_text = response.get("content", "").lower()
        keywords_found = [kw for kw in scenario.expected_keywords if kw.lower() in response_text]
        
        return {
            "scenario": scenario.name,
            "status": "passed",
            "response": response,
            "duration_ms": duration_ms,
            "keywords_found": keywords_found,
            "keywords_expected": scenario.expected_keywords,
            "all_keywords_found": len(keywords_found) == len(scenario.expected_keywords)
        }
        
    except asyncio.TimeoutError:
        return {
            "scenario": scenario.name,
            "status": "timeout",
            "error": f"Timeout after {scenario.timeout_seconds} seconds",
            "duration_ms": scenario.timeout_seconds * 1000
        }
    
    except Exception as e:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        return {
            "scenario": scenario.name,
            "status": "failed",
            "error": str(e),
            "duration_ms": duration_ms
        }


async def run_regression_test_suite(client) -> Dict[str, Any]:
    """Run a complete regression test suite."""
    scenarios = TestDataGenerator.generate_test_scenarios()
    results = []
    
    logger.info(f"Running regression test suite with {len(scenarios)} scenarios")
    
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario.name}")
        result = await run_scenario_test(client, scenario)
        results.append(result)
        
        # Brief pause between tests
        await asyncio.sleep(0.5)
    
    # Calculate summary statistics
    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    timeouts = [r for r in results if r["status"] == "timeout"]
    
    return {
        "test_suite": "regression",
        "timestamp": datetime.utcnow().isoformat(),
        "total_scenarios": len(scenarios),
        "passed": len(passed),
        "failed": len(failed),
        "timeouts": len(timeouts),
        "success_rate": (len(passed) / len(scenarios)) * 100 if scenarios else 0,
        "avg_duration_ms": sum(r["duration_ms"] for r in results) / len(results) if results else 0,
        "results": results
    }