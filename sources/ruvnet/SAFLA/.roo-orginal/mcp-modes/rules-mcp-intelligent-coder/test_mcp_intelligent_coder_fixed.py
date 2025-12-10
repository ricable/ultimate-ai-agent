#!/usr/bin/env python3
"""
Test Suite for MCP Intelligent Coder Mode

This test suite validates the functionality and integration patterns
defined in the MCP Intelligent Coder mode rules and examples.
"""

import asyncio
import pytest
import json
import logging
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime, timezone

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockMCPClient:
    """Mock MCP client for testing MCP tool interactions."""
    
    def __init__(self):
        self.call_history = []
        self.responses = {}
        
    def set_response(self, server_name: str, tool_name: str, response: Any):
        """Set mock response for specific MCP tool call."""
        key = f"{server_name}:{tool_name}"
        self.responses[key] = response
        
    async def use_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Mock MCP tool usage."""
        call_info = {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.call_history.append(call_info)
        
        key = f"{server_name}:{tool_name}"
        if key in self.responses:
            return self.responses[key]
        
        # Default responses for common tools
        if server_name == "context7" and tool_name == "resolve-library-id":
            return {"library_id": f"/{arguments['libraryName']}/{arguments['libraryName']}"}
        elif server_name == "context7" and tool_name == "get-library-docs":
            return {"documentation": f"Mock documentation for {arguments['context7CompatibleLibraryID']}"}
        elif server_name == "safla":
            return {"status": "success", "data": "mock_data"}
        elif server_name == "perplexity":
            return {"response": f"Mock research result for: {arguments.get('userContent', 'query')}"}
        
        return {"status": "success", "mock": True}

class MCPIntelligentCoderWorkflow:
    """Implementation of MCP Intelligent Coder workflow patterns."""
    
    def __init__(self, mcp_client: MockMCPClient):
        self.mcp_client = mcp_client
        self.workflow_state = {}
        
    async def context_driven_development(self, task_description: str, libraries: List[str]) -> Dict[str, Any]:
        """Execute context-driven development workflow."""
        workflow_result = {
            "task": task_description,
            "libraries": libraries,
            "steps": [],
            "documentation": {},
            "performance_metrics": {},
            "validation_results": {}
        }
        
        # Step 1: System validation
        system_info = await self.mcp_client.use_mcp_tool(
            "safla", "get_system_info", {}
        )
        workflow_result["steps"].append("system_validation")
        workflow_result["system_info"] = system_info
        
        # Step 2: Library resolution and documentation
        for library in libraries:
            # Resolve library ID
            library_resolution = await self.mcp_client.use_mcp_tool(
                "context7", "resolve-library-id", {"libraryName": library}
            )
            
            # Get documentation
            library_id = library_resolution.get("library_id", f"/{library}/{library}")
            documentation = await self.mcp_client.use_mcp_tool(
                "context7", "get-library-docs", {
                    "context7CompatibleLibraryID": library_id,
                    "topic": "getting-started",
                    "tokens": 5000
                }
            )
            
            workflow_result["documentation"][library] = {
                "library_id": library_id,
                "docs": documentation
            }
        
        workflow_result["steps"].append("documentation_retrieval")
        
        # Step 3: Performance baseline
        performance_metrics = await self.mcp_client.use_mcp_tool(
            "safla", "benchmark_memory_performance", {
                "test_duration": 30,
                "memory_patterns": ["sequential", "random"]
            }
        )
        workflow_result["performance_metrics"] = performance_metrics
        workflow_result["steps"].append("performance_baseline")
        
        # Step 4: Validation
        validation_result = await self.mcp_client.use_mcp_tool(
            "safla", "validate_memory_operations", {
                "test_data_size": 10,
                "include_stress_test": False
            }
        )
        workflow_result["validation_results"] = validation_result
        workflow_result["steps"].append("validation")
        
        return workflow_result
    
    async def research_enhanced_problem_solving(self, problem_description: str) -> Dict[str, Any]:
        """Execute research-enhanced problem solving workflow."""
        workflow_result = {
            "problem": problem_description,
            "research_results": {},
            "documentation_validation": {},
            "implementation_guidance": {},
            "steps": []
        }
        
        # Step 1: Research solutions
        research_result = await self.mcp_client.use_mcp_tool(
            "perplexity", "PERPLEXITYAI_PERPLEXITY_AI_SEARCH", {
                "userContent": problem_description,
                "systemContent": "Provide technical implementation details with code examples",
                "model": "llama-3.1-sonar-large-128k-online",
                "return_citations": True,
                "temperature": 0.3
            }
        )
        workflow_result["research_results"] = research_result
        workflow_result["steps"].append("research")
        
        # Step 2: Validate with official documentation
        # Extract potential libraries from problem description
        common_libraries = ["fastapi", "react", "tensorflow", "pytorch"]
        relevant_library = None
        for lib in common_libraries:
            if lib.lower() in problem_description.lower():
                relevant_library = lib
                break
        
        if relevant_library:
            library_resolution = await self.mcp_client.use_mcp_tool(
                "context7", "resolve-library-id", {"libraryName": relevant_library}
            )
            
            documentation = await self.mcp_client.use_mcp_tool(
                "context7", "get-library-docs", {
                    "context7CompatibleLibraryID": library_resolution.get("library_id"),
                    "topic": "implementation",
                    "tokens": 8000
                }
            )
            
            workflow_result["documentation_validation"] = {
                "library": relevant_library,
                "documentation": documentation
            }
        
        workflow_result["steps"].append("documentation_validation")
        
        # Step 3: Integration testing
        integration_result = await self.mcp_client.use_mcp_tool(
            "safla", "run_integration_tests", {
                "test_suite": "problem_solving",
                "parallel": True,
                "verbose": False
            }
        )
        workflow_result["implementation_guidance"] = integration_result
        workflow_result["steps"].append("integration_testing")
        
        return workflow_result
    
    async def performance_optimized_development(self, performance_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance-optimized development workflow."""
        workflow_result = {
            "requirements": performance_requirements,
            "baseline_metrics": {},
            "bottleneck_analysis": {},
            "optimization_research": {},
            "benchmark_results": {},
            "steps": []
        }
        
        # Step 1: Performance baseline
        baseline_metrics = await self.mcp_client.use_mcp_tool(
            "safla", "benchmark_memory_performance", {
                "test_duration": 60,
                "memory_patterns": ["sequential", "random", "mixed"]
            }
        )
        workflow_result["baseline_metrics"] = baseline_metrics
        workflow_result["steps"].append("baseline_measurement")
        
        # Step 2: Bottleneck analysis
        bottleneck_analysis = await self.mcp_client.use_mcp_tool(
            "safla", "analyze_performance_bottlenecks", {
                "duration_seconds": 120,
                "include_memory_profile": True
            }
        )
        workflow_result["bottleneck_analysis"] = bottleneck_analysis
        workflow_result["steps"].append("bottleneck_analysis")
        
        # Step 3: Research optimizations
        optimization_query = f"Performance optimization for {performance_requirements.get('domain', 'general')} applications"
        optimization_research = await self.mcp_client.use_mcp_tool(
            "perplexity", "PERPLEXITYAI_PERPLEXITY_AI_SEARCH", {
                "userContent": optimization_query,
                "systemContent": "Focus on practical implementation strategies and benchmarking",
                "model": "llama-3.1-sonar-large-128k-online",
                "temperature": 0.2
            }
        )
        workflow_result["optimization_research"] = optimization_research
        workflow_result["steps"].append("optimization_research")
        
        # Step 4: Vector operations benchmark (if applicable)
        if performance_requirements.get("vector_operations", False):
            vector_benchmark = await self.mcp_client.use_mcp_tool(
                "safla", "benchmark_vector_operations", {
                    "vector_count": performance_requirements.get("vector_count", 10000),
                    "vector_dimensions": performance_requirements.get("vector_dimensions", 512),
                    "operations": ["similarity", "clustering", "indexing"]
                }
            )
            workflow_result["benchmark_results"]["vector_operations"] = vector_benchmark
        
        workflow_result["steps"].append("benchmark_validation")
        
        return workflow_result

class TestMCPIntelligentCoderMode:
    """Test cases for MCP Intelligent Coder mode functionality."""
    
    @pytest.fixture
    def mcp_client(self):
        """Create mock MCP client for testing."""
        return MockMCPClient()
    
    @pytest.fixture
    def workflow(self, mcp_client):
        """Create workflow instance for testing."""
        return MCPIntelligentCoderWorkflow(mcp_client)
    
    @pytest.mark.asyncio
    async def test_context_driven_development_workflow(self, workflow, mcp_client):
        """Test the context-driven development workflow."""
        # Set up mock responses
        mcp_client.set_response("safla", "get_system_info", {
            "status": "healthy",
            "memory_usage": 0.6,
            "cpu_usage": 0.3
        })
        
        mcp_client.set_response("context7", "resolve-library-id", {
            "library_id": "/facebook/react"
        })
        
        mcp_client.set_response("context7", "get-library-docs", {
            "documentation": "React hooks documentation...",
            "examples": ["useState example", "useEffect example"]
        })
        
        # Execute workflow
        result = await workflow.context_driven_development(
            "Build a React component with hooks",
            ["react", "typescript"]
        )
        
        # Validate results
        assert result["task"] == "Build a React component with hooks"
        assert "react" in result["libraries"]
        assert "typescript" in result["libraries"]
        assert "system_validation" in result["steps"]
        assert "documentation_retrieval" in result["steps"]
        assert "performance_baseline" in result["steps"]
        assert "validation" in result["steps"]
        assert "react" in result["documentation"]
        assert "typescript" in result["documentation"]
        
        # Validate MCP call sequence
        calls = mcp_client.call_history
        assert len(calls) >= 6  # At least 6 MCP calls should be made
        
        # Check that system info was called first
        assert calls[0]["server_name"] == "safla"
        assert calls[0]["tool_name"] == "get_system_info"
        
        # Check that library resolution was called for each library
        resolve_calls = [c for c in calls if c["tool_name"] == "resolve-library-id"]
        assert len(resolve_calls) == 2
        
    @pytest.mark.asyncio
    async def test_research_enhanced_problem_solving(self, workflow, mcp_client):
        """Test the research-enhanced problem solving workflow."""
        # Set up mock responses
        mcp_client.set_response("perplexity", "PERPLEXITYAI_PERPLEXITY_AI_SEARCH", {
            "response": "FastAPI authentication best practices...",
            "citations": ["https://fastapi.tiangolo.com/tutorial/security/"]
        })
        
        mcp_client.set_response("context7", "resolve-library-id", {
            "library_id": "/tiangolo/fastapi"
        })
        
        mcp_client.set_response("context7", "get-library-docs", {
            "documentation": "FastAPI security documentation...",
            "code_examples": ["JWT authentication example"]
        })
        
        # Execute workflow
        result = await workflow.research_enhanced_problem_solving(
            "How to implement secure authentication in FastAPI with JWT tokens"
        )
        
        # Validate results
        assert "FastAPI" in result["problem"]
        assert "research" in result["steps"]
        assert "documentation_validation" in result["steps"]
        assert "integration_testing" in result["steps"]
        assert "response" in result["research_results"]
        
        # Validate that research was performed
        research_calls = [c for c in mcp_client.call_history if c["server_name"] == "perplexity"]
        assert len(research_calls) >= 1
        
    @pytest.mark.asyncio
    async def test_performance_optimized_development(self, workflow, mcp_client):
        """Test the performance-optimized development workflow."""
        # Set up mock responses
        mcp_client.set_response("safla", "benchmark_memory_performance", {
            "baseline_memory_mb": 512,
            "peak_memory_mb": 1024,
            "average_latency_ms": 150
        })
        
        mcp_client.set_response("safla", "analyze_performance_bottlenecks", {
            "bottlenecks": ["memory_allocation", "cpu_intensive_operations"],
            "recommendations": ["optimize_memory_usage", "parallel_processing"]
        })
        
        mcp_client.set_response("safla", "benchmark_vector_operations", {
            "similarity_search_ms": 45,
            "indexing_time_ms": 200,
            "memory_efficiency": 0.85
        })
        
        # Execute workflow
        performance_requirements = {
            "domain": "machine_learning",
            "vector_operations": True,
            "vector_count": 50000,
            "vector_dimensions": 768
        }
        
        result = await workflow.performance_optimized_development(performance_requirements)
        
        # Validate results
        assert result["requirements"]["domain"] == "machine_learning"
        assert "baseline_measurement" in result["steps"]
        assert "bottleneck_analysis" in result["steps"]
        assert "optimization_research" in result["steps"]
        assert "benchmark_validation" in result["steps"]
        assert "vector_operations" in result["benchmark_results"]
        
        # Validate performance monitoring calls
        performance_calls = [c for c in mcp_client.call_history if c["server_name"] == "safla"]
        assert len(performance_calls) >= 3
        
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, workflow, mcp_client):
        """Test error handling and fallback mechanisms."""
        # Create a custom failing MCP client
        class FailingMCPClient(MockMCPClient):
            async def use_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
                if server_name == "context7":
                    raise Exception("Context7 server unavailable")
                return await super().use_mcp_tool(server_name, tool_name, arguments)
        
        failing_client = FailingMCPClient()
        failing_workflow = MCPIntelligentCoderWorkflow(failing_client)
        
        # Test that workflow handles failures gracefully
        try:
            result = await failing_workflow.context_driven_development(
                "Test error handling",
                ["react"]
            )
            # Should not reach here if error handling is working
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Context7 server unavailable" in str(e)
    
    def test_mcp_client_call_tracking(self, mcp_client):
        """Test that MCP client properly tracks calls."""
        # Perform some mock calls
        asyncio.run(mcp_client.use_mcp_tool("safla", "get_system_info", {}))
        asyncio.run(mcp_client.use_mcp_tool("context7", "resolve-library-id", {"libraryName": "react"}))
        
        # Validate call history
        assert len(mcp_client.call_history) == 2
        assert mcp_client.call_history[0]["server_name"] == "safla"
        assert mcp_client.call_history[1]["server_name"] == "context7"
        assert "timestamp" in mcp_client.call_history[0]
    
    def test_workflow_state_management(self, workflow):
        """Test workflow state management."""
        # Test initial state
        assert workflow.workflow_state == {}
        
        # Test state updates
        workflow.workflow_state["current_step"] = "documentation_retrieval"
        workflow.workflow_state["libraries_processed"] = ["react", "typescript"]
        
        assert workflow.workflow_state["current_step"] == "documentation_retrieval"
        assert len(workflow.workflow_state["libraries_processed"]) == 2

class TestMCPIntegrationPatterns:
    """Test MCP integration patterns and best practices."""
    
    def test_library_resolution_pattern(self):
        """Test the library resolution pattern."""
        # This pattern should always resolve library ID before getting docs
        pattern_steps = [
            ("context7", "resolve-library-id", {"libraryName": "fastapi"}),
            ("context7", "get-library-docs", {"context7CompatibleLibraryID": "/tiangolo/fastapi"})
        ]
        
        # Validate pattern structure
        assert len(pattern_steps) == 2
        assert pattern_steps[0][1] == "resolve-library-id"
        assert pattern_steps[1][1] == "get-library-docs"
        assert "libraryName" in pattern_steps[0][2]
        assert "context7CompatibleLibraryID" in pattern_steps[1][2]
    
    def test_performance_monitoring_pattern(self):
        """Test the performance monitoring pattern."""
        # This pattern should establish baseline, analyze, and optimize
        pattern_steps = [
            ("safla", "benchmark_memory_performance", {"test_duration": 60}),
            ("safla", "analyze_performance_bottlenecks", {"duration_seconds": 120}),
            ("safla", "optimize_memory_usage", {"optimization_level": "balanced"})
        ]
        
        # Validate pattern structure
        assert len(pattern_steps) == 3
        assert all(step[0] == "safla" for step in pattern_steps)
        assert "benchmark" in pattern_steps[0][1]
        assert "analyze" in pattern_steps[1][1]
        assert "optimize" in pattern_steps[2][1]
    
    def test_research_validation_pattern(self):
        """Test the research and validation pattern."""
        # This pattern should research first, then validate with official docs
        pattern_steps = [
            ("perplexity", "PERPLEXITYAI_PERPLEXITY_AI_SEARCH", {"userContent": "research query"}),
            ("context7", "resolve-library-id", {"libraryName": "library"}),
            ("context7", "get-library-docs", {"context7CompatibleLibraryID": "/org/library"})
        ]
        
        # Validate pattern structure
        assert len(pattern_steps) == 3
        assert pattern_steps[0][0] == "perplexity"
        assert pattern_steps[1][0] == "context7"
        assert pattern_steps[2][0] == "context7"
        assert "userContent" in pattern_steps[0][2]

def run_comprehensive_tests():
    """Run all tests and generate a comprehensive report."""
    import subprocess
    import sys
    
    print("Running MCP Intelligent Coder Mode Test Suite...")
    print("=" * 60)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print("Test Results:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print("-" * 40)
            print(result.stderr)
        
        print(f"\nTest Suite Exit Code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ All tests passed successfully!")
        else:
            print("❌ Some tests failed. Please review the output above.")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_comprehensive_tests()
    exit(0 if success else 1)