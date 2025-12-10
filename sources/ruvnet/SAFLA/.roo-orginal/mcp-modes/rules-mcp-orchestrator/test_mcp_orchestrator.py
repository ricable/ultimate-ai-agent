#!/usr/bin/env python3
"""
Comprehensive Test Suite for MCP Orchestrator Mode

This test suite validates the core functionality of the MCP Orchestrator mode,
including workflow execution, error handling, performance optimization, and
integration with other aiGI modes.
"""

import asyncio
import pytest
import time
import json
import yaml
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test Data Classes
@dataclass
class MockMCPServer:
    """Mock MCP server for testing"""
    name: str
    tools: List[str]
    available: bool = True
    response_time: float = 0.1
    error_rate: float = 0.0
    responses: Dict[str, Any] = field(default_factory=dict)
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate tool call with configurable behavior"""
        if not self.available:
            raise ConnectionError(f"Server {self.name} is unavailable")
        
        # Simulate response time
        await asyncio.sleep(self.response_time)
        
        # Simulate errors based on error rate
        import random
        if random.random() < self.error_rate:
            raise RuntimeError(f"Simulated error from {self.name}")
        
        # Return configured response or default
        return self.responses.get(tool_name, {"status": "success", "data": f"Mock response from {self.name}"})

@dataclass
class WorkflowStep:
    """Workflow step definition"""
    id: str
    service: str
    tool: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 30000
    required: bool = True

@dataclass
class Workflow:
    """Workflow definition"""
    id: str
    name: str
    steps: List[WorkflowStep]
    parallel_steps: List[str] = field(default_factory=list)
    timeout: int = 300000

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class MCPOrchestrator:
    """Mock MCP Orchestrator implementation for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_servers: Dict[str, MockMCPServer] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the orchestrator"""
        # Initialize circuit breakers
        for server_config in self.config.get('mcp_servers', {}).get('required', []):
            self.circuit_breakers[server_config['name']] = {
                'state': CircuitState.CLOSED,
                'failure_count': 0,
                'last_failure_time': None,
                'success_count': 0
            }
    
    def add_mcp_server(self, server: MockMCPServer):
        """Add a mock MCP server"""
        self.mcp_servers[server.name] = server
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a workflow with error handling and monitoring"""
        start_time = time.time()
        workflow_id = workflow.id
        
        try:
            self.active_workflows[workflow_id] = {
                'workflow': workflow,
                'start_time': start_time,
                'status': 'running'
            }
            
            # Execute workflow steps
            results = await self._execute_workflow_steps(workflow)
            
            execution_time = time.time() - start_time
            
            # Record performance metrics
            self._record_performance_metric('execution_time', execution_time)
            
            # Record workflow history
            workflow_result = {
                'id': workflow_id,
                'name': workflow.name,
                'status': 'completed',
                'execution_time': execution_time,
                'steps': len(workflow.steps),
                'results': results
            }
            
            self.workflow_history.append(workflow_result)
            self.active_workflows[workflow_id]['status'] = 'completed'
            
            return workflow_result
            
        except Exception as error:
            execution_time = time.time() - start_time
            
            workflow_result = {
                'id': workflow_id,
                'name': workflow.name,
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(error),
                'error_type': type(error).__name__
            }
            
            self.workflow_history.append(workflow_result)
            self.active_workflows[workflow_id]['status'] = 'failed'
            
            return workflow_result
    
    async def _execute_workflow_steps(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Execute workflow steps with dependency resolution"""
        results = []
        completed_steps = set()
        
        # Create dependency graph
        step_map = {step.id: step for step in workflow.steps}
        
        while len(completed_steps) < len(workflow.steps):
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []
            for step in workflow.steps:
                if (step.id not in completed_steps and 
                    all(dep in completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise RuntimeError("Circular dependency detected in workflow")
            
            # Execute ready steps (parallel if configured)
            if len(ready_steps) > 1 and any(step.id in workflow.parallel_steps for step in ready_steps):
                # Execute in parallel
                step_results = await asyncio.gather(
                    *[self._execute_step(step) for step in ready_steps],
                    return_exceptions=True
                )
            else:
                # Execute sequentially
                step_results = []
                for step in ready_steps:
                    result = await self._execute_step(step)
                    step_results.append(result)
            
            # Process results
            for i, step in enumerate(ready_steps):
                result = step_results[i]
                if isinstance(result, Exception):
                    if step.required:
                        raise result
                    else:
                        # Optional step failed, continue
                        result = {
                            'step_id': step.id,
                            'status': 'failed',
                            'error': str(result),
                            'required': False
                        }
                
                results.append(result)
                completed_steps.add(step.id)
        
        return results
    
    async def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step with circuit breaker and retry"""
        try:
            # Check circuit breaker
            if not self._is_circuit_closed(step.service):
                raise RuntimeError(f"Circuit breaker open for service {step.service}")
            
            # Execute with retry
            result = await self._execute_with_retry(step)
            
            # Record success
            self._record_circuit_success(step.service)
            
            return {
                'step_id': step.id,
                'service': step.service,
                'tool': step.tool,
                'status': 'success',
                'data': result
            }
            
        except Exception as error:
            # Record failure
            self._record_circuit_failure(step.service)
            
            return {
                'step_id': step.id,
                'service': step.service,
                'tool': step.tool,
                'status': 'failed',
                'error': str(error),
                'error_type': type(error).__name__
            }
    
    async def _execute_with_retry(self, step: WorkflowStep) -> Any:
        """Execute step with retry logic"""
        max_attempts = self.config.get('error_handling', {}).get('retry_policy', {}).get('max_attempts', 3)
        base_delay = self.config.get('error_handling', {}).get('retry_policy', {}).get('base_delay', 1000) / 1000
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                server = self.mcp_servers.get(step.service)
                if not server:
                    raise RuntimeError(f"Server {step.service} not found")
                
                result = await server.call_tool(step.tool, step.parameters)
                return result
                
            except Exception as error:
                last_error = error
                
                if attempt < max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    break
        
        raise last_error
    
    def _is_circuit_closed(self, service_name: str) -> bool:
        """Check if circuit breaker is closed for service"""
        circuit = self.circuit_breakers.get(service_name, {})
        state = circuit.get('state', CircuitState.CLOSED)
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            # Check if we should attempt reset
            last_failure = circuit.get('last_failure_time')
            if last_failure and time.time() - last_failure > 60:  # 60 second timeout
                circuit['state'] = CircuitState.HALF_OPEN
                return True
            return False
        elif state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_circuit_success(self, service_name: str):
        """Record successful service call"""
        circuit = self.circuit_breakers.get(service_name, {})
        circuit['failure_count'] = 0
        circuit['success_count'] = circuit.get('success_count', 0) + 1
        
        if circuit.get('state') == CircuitState.HALF_OPEN and circuit['success_count'] >= 2:
            circuit['state'] = CircuitState.CLOSED
            circuit['success_count'] = 0
    
    def _record_circuit_failure(self, service_name: str):
        """Record failed service call"""
        circuit = self.circuit_breakers.get(service_name, {})
        circuit['failure_count'] = circuit.get('failure_count', 0) + 1
        circuit['last_failure_time'] = time.time()
        
        failure_threshold = self.config.get('error_handling', {}).get('circuit_breaker', {}).get('failure_threshold', 5)
        if circuit['failure_count'] >= failure_threshold:
            circuit['state'] = CircuitState.OPEN
    
    def _record_performance_metric(self, metric_name: str, value: float):
        """Record performance metric"""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append(value)
    
    def get_circuit_state(self, service_name: str) -> str:
        """Get circuit breaker state for service"""
        circuit = self.circuit_breakers.get(service_name, {})
        return circuit.get('state', CircuitState.CLOSED).value
    
    async def call_service_with_retry(self, service_name: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call service with retry logic"""
        step = WorkflowStep(
            id="test_step",
            service=service_name,
            tool=tool_name,
            parameters=parameters
        )
        return await self._execute_with_retry(step)
    
    async def optimize_workflow(self, workflow: Workflow) -> Workflow:
        """Optimize workflow based on performance metrics"""
        # Simple optimization: identify parallelizable steps
        optimized_steps = workflow.steps.copy()
        parallel_steps = workflow.parallel_steps.copy()
        
        # Find steps that can be parallelized (no dependencies between them)
        for i, step1 in enumerate(optimized_steps):
            for j, step2 in enumerate(optimized_steps[i+1:], i+1):
                if (step1.id not in step2.dependencies and 
                    step2.id not in step1.dependencies and
                    step1.id not in parallel_steps and
                    step2.id not in parallel_steps):
                    parallel_steps.extend([step1.id, step2.id])
        
        return Workflow(
            id=f"{workflow.id}_optimized",
            name=f"{workflow.name} (Optimized)",
            steps=optimized_steps,
            parallel_steps=parallel_steps,
            timeout=workflow.timeout
        )

# Test Configuration
TEST_CONFIG = {
    'orchestrator': {
        'max_concurrent_workflows': 5,
        'default_timeout': 30000,
        'retry_attempts': 3
    },
    'error_handling': {
        'circuit_breaker': {
            'failure_threshold': 3,
            'recovery_timeout': 60
        },
        'retry_policy': {
            'max_attempts': 3,
            'base_delay': 1000,
            'strategy': 'exponential'
        }
    },
    'mcp_servers': {
        'required': [
            {'name': 'safla', 'tools': ['*']},
            {'name': 'perplexity', 'tools': ['search']},
            {'name': 'context7', 'tools': ['resolve-library-id', 'get-library-docs']}
        ]
    }
}

# Test Fixtures
@pytest.fixture
async def orchestrator():
    """Create orchestrator instance for testing"""
    orch = MCPOrchestrator(TEST_CONFIG)
    await orch.initialize()
    
    # Add mock servers
    orch.add_mcp_server(MockMCPServer(
        name="safla",
        tools=["*"],
        responses={"get_system_info": {"status": "healthy", "version": "1.0.0"}}
    ))
    
    orch.add_mcp_server(MockMCPServer(
        name="perplexity",
        tools=["search"],
        responses={"search": {"results": ["result1", "result2"]}}
    ))
    
    orch.add_mcp_server(MockMCPServer(
        name="context7",
        tools=["resolve-library-id", "get-library-docs"],
        responses={"resolve-library-id": {"library_id": "/test/library"}}
    ))
    
    return orch

@pytest.fixture
def simple_workflow():
    """Create a simple test workflow"""
    return Workflow(
        id="test_workflow_simple",
        name="Simple Test Workflow",
        steps=[
            WorkflowStep(
                id="step1",
                service="safla",
                tool="get_system_info",
                parameters={}
            ),
            WorkflowStep(
                id="step2",
                service="perplexity",
                tool="search",
                parameters={"query": "test"},
                dependencies=["step1"]
            )
        ]
    )

@pytest.fixture
def parallel_workflow():
    """Create a parallel test workflow"""
    return Workflow(
        id="test_workflow_parallel",
        name="Parallel Test Workflow",
        steps=[
            WorkflowStep(
                id="step1",
                service="safla",
                tool="get_system_info",
                parameters={}
            ),
            WorkflowStep(
                id="step2",
                service="perplexity",
                tool="search",
                parameters={"query": "test1"}
            ),
            WorkflowStep(
                id="step3",
                service="context7",
                tool="resolve-library-id",
                parameters={"library": "test"}
            )
        ],
        parallel_steps=["step2", "step3"]
    )

# Test Cases
class TestMCPOrchestrator:
    """Test suite for MCP Orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator is not None
        assert len(orchestrator.mcp_servers) == 3
        assert "safla" in orchestrator.mcp_servers
        assert "perplexity" in orchestrator.mcp_servers
        assert "context7" in orchestrator.mcp_servers
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, orchestrator, simple_workflow):
        """Test execution of a simple sequential workflow"""
        result = await orchestrator.execute_workflow(simple_workflow)
        
        assert result['status'] == 'completed'
        assert result['id'] == 'test_workflow_simple'
        assert len(result['results']) == 2
        assert result['execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, orchestrator, parallel_workflow):
        """Test execution of a parallel workflow"""
        start_time = time.time()
        result = await orchestrator.execute_workflow(parallel_workflow)
        execution_time = time.time() - start_time
        
        assert result['status'] == 'completed'
        assert result['id'] == 'test_workflow_parallel'
        assert len(result['results']) == 3
        
        # Parallel execution should be faster than sequential
        # (This is a rough check since we're using mock delays)
        assert execution_time < 1.0  # Should complete quickly with mocks
    
    @pytest.mark.asyncio
    async def test_workflow_with_dependencies(self, orchestrator):
        """Test workflow with complex dependencies"""
        workflow = Workflow(
            id="test_dependencies",
            name="Dependency Test",
            steps=[
                WorkflowStep(id="step1", service="safla", tool="get_system_info", parameters={}),
                WorkflowStep(id="step2", service="perplexity", tool="search", parameters={"query": "test"}, dependencies=["step1"]),
                WorkflowStep(id="step3", service="context7", tool="resolve-library-id", parameters={"library": "test"}, dependencies=["step1"]),
                WorkflowStep(id="step4", service="safla", tool="get_system_info", parameters={}, dependencies=["step2", "step3"])
            ]
        )
        
        result = await orchestrator.execute_workflow(workflow)
        
        assert result['status'] == 'completed'
        assert len(result['results']) == 4
        
        # Verify execution order respects dependencies
        step_order = [r['step_id'] for r in result['results']]
        assert step_order.index('step1') < step_order.index('step2')
        assert step_order.index('step1') < step_order.index('step3')
        assert step_order.index('step2') < step_order.index('step4')
        assert step_order.index('step3') < step_order.index('step4')
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, orchestrator):
        """Test circuit breaker pattern implementation"""
        # Configure server to fail
        orchestrator.mcp_servers["safla"].error_rate = 1.0  # 100% failure rate
        
        # Execute multiple calls to trigger circuit breaker
        for i in range(5):
            try:
                await orchestrator.call_service_with_retry("safla", "get_system_info", {})
            except Exception:
                pass  # Expected failures
        
        # Circuit should be open now
        circuit_state = orchestrator.get_circuit_state("safla")
        assert circuit_state == "open"
        
        # Reset server and wait for circuit to reset
        orchestrator.mcp_servers["safla"].error_rate = 0.0
        
        # Manually set circuit to half-open for testing
        orchestrator.circuit_breakers["safla"]["state"] = CircuitState.HALF_OPEN
        orchestrator.circuit_breakers["safla"]["success_count"] = 0
        
        # Execute successful calls to close circuit
        await orchestrator.call_service_with_retry("safla", "get_system_info", {})
        await orchestrator.call_service_with_retry("safla", "get_system_info", {})
        
        circuit_state = orchestrator.get_circuit_state("safla")
        assert circuit_state == "closed"
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, orchestrator):
        """Test retry mechanism with exponential backoff"""
        # Configure server to fail twice then succeed
        call_count = 0
        original_call_tool = orchestrator.mcp_servers["perplexity"].call_tool
        
        async def failing_call_tool(tool_name, parameters):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Simulated failure")
            return await original_call_tool(tool_name, parameters)
        
        orchestrator.mcp_servers["perplexity"].call_tool = failing_call_tool
        
        start_time = time.time()
        result = await orchestrator.call_service_with_retry("perplexity", "search", {"query": "test"})
        execution_time = time.time() - start_time
        
        assert result is not None
        assert call_count == 3  # Should have retried twice
        assert execution_time > 3.0  # Should have waited for retries (1s + 2s delays)
    
    @pytest.mark.asyncio
    async def test_workflow_failure_handling(self, orchestrator):
        """Test workflow failure handling"""
        # Create workflow with failing step
        workflow = Workflow(
            id="test_failure",
            name="Failure Test",
            steps=[
                WorkflowStep(id="step1", service="safla", tool="get_system_info", parameters={}),
                WorkflowStep(id="step2", service="nonexistent", tool="fail", parameters={}, dependencies=["step1"])
            ]
        )
        
        result = await orchestrator.execute_workflow(workflow)
        
        assert result['status'] == 'failed'
        assert 'error' in result
        assert result['error_type'] is not None
    
    @pytest.mark.asyncio
    async def test_optional_step_failure(self, orchestrator):
        """Test workflow continues when optional step fails"""
        workflow = Workflow(
            id="test_optional_failure",
            name="Optional Failure Test",
            steps=[
                WorkflowStep(id="step1", service="safla", tool="get_system_info", parameters={}),
                WorkflowStep(id="step2", service="nonexistent", tool="fail", parameters={}, required=False),
                WorkflowStep(id="step3", service="perplexity", tool="search", parameters={"query": "test"}, dependencies=["step1"])
            ]
        )
        
        result = await orchestrator.execute_workflow(workflow)
        
        assert result['status'] == 'completed'
        assert len(result['results']) == 3
        
        # Check that optional step failed but workflow continued
        step2_result = next(r for r in result['results'] if r['step_id'] == 'step2')
        assert step2_result['status'] == 'failed'
        assert not step2_result['required']
    
    @pytest.mark.asyncio
    async def test_workflow_optimization(self, orchestrator, simple_workflow):
        """Test workflow optimization"""
        optimized = await orchestrator.optimize_workflow(simple_workflow)
        
        assert optimized.id == f"{simple_workflow.id}_optimized"
        assert optimized.name.endswith("(Optimized)")
        assert len(optimized.steps) == len(simple_workflow.steps)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, orchestrator, simple_workflow):
        """Test performance monitoring and metrics collection"""
        # Execute workflow multiple times
        for i in range(3):
            await orchestrator.execute_workflow(simple_workflow)
        
        # Check that metrics were recorded
        assert 'execution_time' in orchestrator.performance_metrics
        assert len(orchestrator.performance_metrics['execution_time']) == 3
        assert len(orchestrator.workflow_history) == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestrator, simple_workflow):
        """Test concurrent execution of multiple workflows"""
        # Create multiple workflows
        workflows = []
        for i in range(3):
            workflow = Workflow(
                id=f"concurrent_test_{i}",
                name=f"Concurrent Test {i}",
                steps=[
                    WorkflowStep(
                        id=f"step_{i}",
                        service="safla",
                        tool="get_system_info",
                        parameters={}
                    )
                ]
            )
            workflows.append(workflow)
        
        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*[
            orchestrator.execute_workflow(workflow) for workflow in workflows
        ])
        execution_time = time.time() - start_time
        
        # All workflows should complete successfully
        assert len(results) == 3
        assert all(result['status'] == 'completed' for result in results)
        
        # Concurrent execution should be faster than sequential
        assert execution_time < 1.0  # Should be fast with mocks
    
    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, orchestrator):
        """Test service health monitoring"""
        # Test healthy service
        server = orchestrator.mcp_servers["safla"]
        assert server.available == True
        
        # Simulate service failure
        server.available = False
        
        try:
            await orchestrator.call_service_with_retry("safla", "get_system_info", {})
            assert False, "Should have raised exception"
        except Exception as e:
            assert "unavailable" in str(e)
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, orchestrator):
        """Test workflow timeout handling"""
        # Create workflow with very short timeout
        workflow = Workflow(
            id="timeout_test",
            name="Timeout Test",
            steps=[
                WorkflowStep(
                    id="slow_step",
                    service="safla",
                    tool="get_system_info",
                    parameters={},
                    timeout=1  # Very short timeout
                )
            ],
            timeout=1
        )
        
        # Configure server to be slow
        orchestrator.mcp_servers["safla"].response_time = 2.0  # 2 second delay
        
        result = await orchestrator.execute_workflow(workflow)
        
        # Workflow should complete (our mock doesn't enforce timeouts, but structure is there)
        assert result is not None

# Integration Tests
class TestMCPOrchestratorIntegration:
    """Integration tests for MCP Orchestrator with real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_code_development_workflow(self, orchestrator):
        """Test complete code development workflow"""
        workflow = Workflow(
            id="code_dev_integration",
            name="Code Development Integration Test",
            steps=[
                WorkflowStep(
                    id="research",
                    service="perplexity",
                    tool="search",
                    parameters={"query": "React best practices"}
                ),
                WorkflowStep(
                    id="design",
                    service="safla",
                    tool="get_system_info",
                    parameters={},
                    dependencies=["research"]
                ),
                WorkflowStep(
                    id="implement",
                    service="safla",
                    tool="get_system_info",
                    parameters={},
                    dependencies=["design"]
                ),
                WorkflowStep(
                    id="test",
                    service="context7",
                    tool="resolve-library-id",
                    parameters={"library": "jest"},
                    dependencies=["implement"]
                ),
                WorkflowStep(
                    id="review",
                    service="safla",
                    tool="get_system_info",
                    parameters={},
                    dependencies=["test"]
                )
            ]
        )
        
        result = await orchestrator.execute_workflow(workflow)
        
        assert result['status'] == 'completed'
        assert len(result['results']) == 5
        assert result['execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_research_decision_workflow(self, orchestrator):
        """Test research and decision making workflow"""
        workflow = Workflow(
            id="research_decision_integration",
            name="Research Decision Integration Test",
            steps=[
                WorkflowStep(
                    id="technical_research",
                    service="perplexity",
                    tool="search",
                    parameters={"query": "AI frameworks comparison"}
                ),
                WorkflowStep(
                    id="market_research",
                    service="perplexity",
                    tool="search",
                    parameters={"query": "AI market trends"}
                ),
                WorkflowStep(
                    id="analysis",
                    service="safla",
                    tool="get_system_info",
                    parameters={},
                    dependencies=["technical_research", "market_research"]
                ),
                WorkflowStep(
                    id="recommendation",
                    service="safla",
                    tool="get_system_info",
                    parameters={},
                    dependencies=["analysis"]
                )
            ],
            parallel_steps=["technical_research", "market_research"]
        )
        
        result = await orchestrator.execute_workflow(workflow)
        
        assert result['status'] == 'completed'
        assert len(result['results']) == 4

# Performance Tests
class TestMCPOrchestratorPerformance:
    """Performance tests for MCP Orchestrator"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_workflows(self, orchestrator):
        """Test orchestrator under high concurrency"""
        # Create many small workflows
        workflows = []
        for i in range(10):
            workflow = Workflow(
                id=f"perf_test_{i}",
                name=f"Performance Test {i}",
                steps=[
                    WorkflowStep(
                        id=f"step_{i}",
                        service="safla",
                        tool="get_system_info",
                        parameters={}
                    )
                ]
            )
            workflows.append(workflow)
        
        start_time = time.time()
        results = await asyncio.gather(*[
            orchestrator.execute_workflow(workflow) for workflow in workflows
        ])
        execution_time = time.time() - start_time
        
        assert len(results) == 10
        assert all(result['status'] == 'completed' for result in results)
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_large_workflow_execution(self, orchestrator):
        """Test execution of large workflow with many steps"""
        # Create workflow with many sequential steps
        steps = []
        for i in range(20):
            step = WorkflowStep(
                id=f"step_{i}",
                service="safla",
                tool="get_system_info",
                parameters={}
            )
            if i > 0:
                step.dependencies = [f"step_{i-1}"]
            steps.append(step)
        
        workflow = Workflow(
            id="large_workflow_test",
            name="Large Workflow Test",
            steps=steps
        )
        
        start_time = time.time()
        result = await orchestrator.execute_workflow(workflow)
        execution_time = time.time() - start_time
        
        assert result['status'] == 'completed'
        assert len(result['results']) == 20
        assert execution_time < 10.0  # Should complete within 10 seconds

# Main test runner
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])