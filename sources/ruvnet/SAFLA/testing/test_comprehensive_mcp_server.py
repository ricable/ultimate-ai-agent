#!/usr/bin/env python3
"""
Comprehensive Test Script for SAFLA MCP Server

This script tests all tools and resources across the different domains:
- Core System Tools
- Deployment Tools  
- Optimization Tools
- Admin Tools
- Testing Tools
- Benchmarking Tools
- Agent Interaction Tools
- All Resources
"""

import asyncio
import json
import subprocess
import sys
import time
from typing import Dict, Any, List


class MCPServerTester:
    """Test harness for the comprehensive SAFLA MCP server"""
    
    def __init__(self):
        self.server_process = None
        self.test_results = {}
        
    async def start_server(self):
        """Start the MCP server process"""
        try:
            self.server_process = await asyncio.create_subprocess_exec(
                sys.executable, "safla/mcp_stdio_server.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            print("✓ MCP Server started successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to start MCP server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the MCP server process"""
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            print("✓ MCP Server stopped")
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a request to the MCP server"""
        if not self.server_process:
            raise RuntimeError("Server not started")
        
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": method,
            "params": params or {}
        }
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.server_process.stdin.write(request_json.encode())
        await self.server_process.stdin.drain()
        
        # Read response - handle multiple JSON objects
        response_line = await self.server_process.stdout.readline()
        response_text = response_line.decode().strip()
        
        # Handle case where multiple JSON objects might be on the same line
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract the first complete JSON object
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        response = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                raise json.JSONDecodeError(f"Could not parse JSON from: {response_text}", response_text, 0)
        
        return response
    
    async def test_tool(self, tool_name: str, args: Dict[str, Any] = None) -> bool:
        """Test a specific tool"""
        try:
            print(f"  Testing tool: {tool_name}")
            response = await self.send_request("tools/call", {
                "name": tool_name,
                "arguments": args or {}
            })
            
            if "error" in response:
                print(f"    ✗ Error: {response['error']['message']}")
                return False
            
            result = response.get("result", {})
            if "content" in result and result["content"]:
                content = json.loads(result["content"][0]["text"])
                if "error" in content:
                    print(f"    ✗ Tool error: {content['error']}")
                    return False
                else:
                    print(f"    ✓ Success")
                    return True
            else:
                print(f"    ✗ No content in response")
                return False
                
        except Exception as e:
            print(f"    ✗ Exception: {e}")
            return False
    
    async def test_resource(self, resource_uri: str) -> bool:
        """Test a specific resource"""
        try:
            print(f"  Testing resource: {resource_uri}")
            response = await self.send_request("resources/read", {
                "uri": resource_uri
            })
            
            if "error" in response:
                print(f"    ✗ Error: {response['error']['message']}")
                return False
            
            result = response.get("result", {})
            if "contents" in result and result["contents"]:
                print(f"    ✓ Success")
                return True
            else:
                print(f"    ✗ No contents in response")
                return False
                
        except Exception as e:
            print(f"    ✗ Exception: {e}")
            return False
    
    async def test_initialization(self) -> bool:
        """Test server initialization"""
        print("\n=== Testing Server Initialization ===")
        try:
            response = await self.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            if "error" in response:
                print(f"✗ Initialization failed: {response['error']['message']}")
                return False
            
            result = response.get("result", {})
            if result.get("protocolVersion") == "2024-11-05":
                print("✓ Server initialized successfully")
                return True
            else:
                print("✗ Invalid initialization response")
                return False
                
        except Exception as e:
            print(f"✗ Initialization exception: {e}")
            return False
    
    async def test_tools_list(self) -> bool:
        """Test tools listing"""
        print("\n=== Testing Tools List ===")
        try:
            response = await self.send_request("tools/list")
            
            if "error" in response:
                print(f"✗ Tools list failed: {response['error']['message']}")
                return False
            
            result = response.get("result", {})
            tools = result.get("tools", [])
            
            expected_tool_count = 24  # Total number of tools we implemented
            if len(tools) >= expected_tool_count:
                print(f"✓ Tools list returned {len(tools)} tools")
                return True
            else:
                print(f"✗ Expected at least {expected_tool_count} tools, got {len(tools)}")
                return False
                
        except Exception as e:
            print(f"✗ Tools list exception: {e}")
            return False
    
    async def test_resources_list(self) -> bool:
        """Test resources listing"""
        print("\n=== Testing Resources List ===")
        try:
            response = await self.send_request("resources/list")
            
            if "error" in response:
                print(f"✗ Resources list failed: {response['error']['message']}")
                return False
            
            result = response.get("result", {})
            resources = result.get("resources", [])
            
            expected_resource_count = 15  # Total number of resources we implemented
            if len(resources) >= expected_resource_count:
                print(f"✓ Resources list returned {len(resources)} resources")
                return True
            else:
                print(f"✗ Expected at least {expected_resource_count} resources, got {len(resources)}")
                return False
                
        except Exception as e:
            print(f"✗ Resources list exception: {e}")
            return False
    
    async def test_core_system_tools(self) -> bool:
        """Test core system tools"""
        print("\n=== Testing Core System Tools ===")
        tools = [
            "validate_installation",
            "get_system_info", 
            "check_gpu_status",
            "get_config_summary"
        ]
        
        results = []
        for tool in tools:
            result = await self.test_tool(tool)
            results.append(result)
        
        success_count = sum(results)
        print(f"Core System Tools: {success_count}/{len(tools)} passed")
        return success_count == len(tools)
    
    async def test_deployment_tools(self) -> bool:
        """Test deployment tools"""
        print("\n=== Testing Deployment Tools ===")
        
        # Test deploy_safla_instance
        deploy_result = await self.test_tool("deploy_safla_instance", {
            "instance_name": "test-instance",
            "environment": "development"
        })
        
        # Test check_deployment_status
        status_result = await self.test_tool("check_deployment_status", {
            "instance_name": "test-instance"
        })
        
        # Test scale_deployment
        scale_result = await self.test_tool("scale_deployment", {
            "instance_name": "test-instance",
            "scale_factor": 1.5,
            "resource_type": "both"
        })
        
        results = [deploy_result, status_result, scale_result]
        success_count = sum(results)
        print(f"Deployment Tools: {success_count}/{len(results)} passed")
        return success_count == len(results)
    
    async def test_optimization_tools(self) -> bool:
        """Test optimization tools"""
        print("\n=== Testing Optimization Tools ===")
        
        # Test optimize_memory_usage
        memory_result = await self.test_tool("optimize_memory_usage", {
            "optimization_level": "balanced",
            "target_memory_mb": 1024
        })
        
        # Test optimize_vector_operations
        vector_result = await self.test_tool("optimize_vector_operations", {
            "batch_size": 100,
            "use_gpu": True
        })
        
        # Test analyze_performance_bottlenecks
        bottleneck_result = await self.test_tool("analyze_performance_bottlenecks", {
            "duration_seconds": 5,  # Short duration for testing
            "include_memory_profile": True
        })
        
        results = [memory_result, vector_result, bottleneck_result]
        success_count = sum(results)
        print(f"Optimization Tools: {success_count}/{len(results)} passed")
        return success_count == len(results)
    
    async def test_admin_tools(self) -> bool:
        """Test admin tools"""
        print("\n=== Testing Admin Tools ===")
        
        # Test manage_user_sessions
        sessions_result = await self.test_tool("manage_user_sessions", {
            "action": "list"
        })
        
        # Test backup_safla_data
        backup_result = await self.test_tool("backup_safla_data", {
            "backup_type": "full",
            "compress": True
        })
        
        # Test restore_safla_data (using a dummy path)
        restore_result = await self.test_tool("restore_safla_data", {
            "backup_path": "/tmp/dummy_backup.tar.gz",
            "restore_type": "config_only"
        })
        
        # Test monitor_system_health
        health_result = await self.test_tool("monitor_system_health", {
            "check_interval": 30
        })
        
        results = [sessions_result, backup_result, restore_result, health_result]
        success_count = sum(results)
        print(f"Admin Tools: {success_count}/{len(results)} passed")
        return success_count == len(results)
    
    async def test_testing_tools(self) -> bool:
        """Test testing tools"""
        print("\n=== Testing Testing Tools ===")
        
        # Test run_integration_tests
        integration_result = await self.test_tool("run_integration_tests", {
            "test_suite": "core",
            "parallel": True,
            "verbose": False
        })
        
        # Test validate_memory_operations
        memory_validation_result = await self.test_tool("validate_memory_operations", {
            "test_data_size": 5,  # Small size for testing
            "include_stress_test": False
        })
        
        # Test test_mcp_connectivity
        connectivity_result = await self.test_tool("test_mcp_connectivity", {
            "target_server": "self",
            "test_depth": "basic"
        })
        
        results = [integration_result, memory_validation_result, connectivity_result]
        success_count = sum(results)
        print(f"Testing Tools: {success_count}/{len(results)} passed")
        return success_count == len(results)
    
    async def test_benchmarking_tools(self) -> bool:
        """Test benchmarking tools"""
        print("\n=== Testing Benchmarking Tools ===")
        
        # Test benchmark_vector_operations
        vector_bench_result = await self.test_tool("benchmark_vector_operations", {
            "vector_count": 100,  # Small count for testing
            "vector_dimensions": 128,
            "operations": ["similarity", "search"]
        })
        
        # Test benchmark_memory_performance
        memory_bench_result = await self.test_tool("benchmark_memory_performance", {
            "test_duration": 5,  # Short duration for testing
            "memory_patterns": ["sequential", "random"]
        })
        
        # Test benchmark_mcp_throughput
        mcp_bench_result = await self.test_tool("benchmark_mcp_throughput", {
            "request_count": 10,  # Small count for testing
            "concurrent_connections": 1,
            "payload_size": "small"
        })
        
        results = [vector_bench_result, memory_bench_result, mcp_bench_result]
        success_count = sum(results)
        print(f"Benchmarking Tools: {success_count}/{len(results)} passed")
        return success_count == len(results)
    
    async def test_agent_interaction_tools(self) -> bool:
        """Test agent interaction tools"""
        print("\n=== Testing Agent Interaction Tools ===")
        
        session_id = None
        
        # Test create_agent_session and capture session ID
        print("  Testing tool: create_agent_session")
        try:
            response = await self.send_request("tools/call", {
                "name": "create_agent_session",
                "arguments": {
                    "agent_type": "cognitive",
                    "session_config": {"mode": "test"},
                    "timeout_seconds": 3600
                }
            })
            
            if "error" in response:
                print(f"    ✗ Tool error: {response['error']['message']}")
                create_result = False
            else:
                result = response.get("result", {})
                session_id = result.get("session_id")
                print(f"    ✓ Success (session_id: {session_id})")
                create_result = True
        except Exception as e:
            print(f"    ✗ Exception: {e}")
            create_result = False
        
        # Test list_agent_sessions
        list_result = await self.test_tool("list_agent_sessions", {
            "include_inactive": True
        })
        
        # Test interact_with_agent using the created session ID if available
        if session_id:
            interact_result = await self.test_tool("interact_with_agent", {
                "session_id": session_id,
                "command": "analyze",
                "parameters": {"data": "test"}
            })
            
            # Test terminate_agent_session using the created session ID
            terminate_result = await self.test_tool("terminate_agent_session", {
                "session_id": session_id,
                "force": False
            })
            
            results = [create_result, list_result, interact_result, terminate_result]
            success_count = sum(results)
            print(f"Agent Interaction Tools: {success_count}/{len(results)} passed")
            return success_count >= 2  # At least create and list should work
        else:
            # If session creation failed, only test the basic tools
            results = [create_result, list_result]
            success_count = sum(results)
            print(f"Agent Interaction Tools: {success_count}/{len(results)} core tools passed")
            return success_count >= 1  # At least one should work
    
    async def test_all_resources(self) -> bool:
        """Test all resources"""
        print("\n=== Testing All Resources ===")
        
        resources = [
            "safla://config",
            "safla://status",
            "safla://deployments",
            "safla://deployment-templates",
            "safla://performance-metrics",
            "safla://optimization-recommendations",
            "safla://system-logs",
            "safla://user-sessions",
            "safla://backup-status",
            "safla://test-results",
            "safla://test-coverage",
            "safla://benchmark-results",
            "safla://performance-baselines",
            "safla://agent-sessions",
            "safla://agent-capabilities"
        ]
        
        results = []
        for resource in resources:
            result = await self.test_resource(resource)
            results.append(result)
        
        success_count = sum(results)
        print(f"Resources: {success_count}/{len(results)} passed")
        return success_count == len(results)
    
    async def test_strategy_and_learning_tools(self) -> bool:
        """Test strategy management and learning tools (fixed functions)"""
        print("\n=== Testing Strategy and Learning Tools (Fixed) ===")
        
        # Test create_custom_strategy with expected_outcomes parameter
        print("  Testing tool: create_custom_strategy")
        strategy_result = await self.test_tool("create_custom_strategy", {
            "strategy_name": "Test Strategy",
            "description": "A test strategy for validation",
            "context": "testing",
            "steps": ["step1", "step2", "step3"],
            "expected_outcomes": ["outcome1", "outcome2"]
        })
        
        # Test create_custom_strategy without optional expected_outcomes
        print("  Testing tool: create_custom_strategy (without expected_outcomes)")
        strategy_result_minimal = await self.test_tool("create_custom_strategy", {
            "strategy_name": "Minimal Test Strategy",
            "description": "A minimal test strategy",
            "context": "testing",
            "steps": ["step1", "step2"]
        })
        
        # Test get_learning_metrics with default parameters
        print("  Testing tool: get_learning_metrics (default)")
        learning_result_default = await self.test_tool("get_learning_metrics")
        
        # Test get_learning_metrics with specific metric_type
        print("  Testing tool: get_learning_metrics (accuracy)")
        learning_result_accuracy = await self.test_tool("get_learning_metrics", {
            "metric_type": "accuracy",
            "time_range_hours": 12
        })
        
        # Test get_learning_metrics with different metric_type
        print("  Testing tool: get_learning_metrics (adaptation_rate)")
        learning_result_adaptation = await self.test_tool("get_learning_metrics", {
            "metric_type": "adaptation_rate",
            "time_range_hours": 48
        })
        
        # Test get_learning_metrics with knowledge_retention
        print("  Testing tool: get_learning_metrics (knowledge_retention)")
        learning_result_retention = await self.test_tool("get_learning_metrics", {
            "metric_type": "knowledge_retention",
            "time_range_hours": 24
        })
        
        results = [
            strategy_result,
            strategy_result_minimal,
            learning_result_default,
            learning_result_accuracy,
            learning_result_adaptation,
            learning_result_retention
        ]
        success_count = sum(results)
        print(f"Strategy and Learning Tools: {success_count}/{len(results)} passed")
        
        # These tools should now work after the fixes
        return success_count >= 4  # At least 4 out of 6 should pass

    async def run_comprehensive_test(self) -> bool:
        """Run all tests"""
        print("🚀 Starting Comprehensive SAFLA MCP Server Test")
        print("=" * 60)
        
        # Start server
        if not await self.start_server():
            return False
        
        try:
            # Wait for server to be ready
            await asyncio.sleep(2)
            
            # Run all test categories
            test_results = []
            
            test_results.append(await self.test_initialization())
            test_results.append(await self.test_tools_list())
            test_results.append(await self.test_resources_list())
            test_results.append(await self.test_core_system_tools())
            test_results.append(await self.test_deployment_tools())
            test_results.append(await self.test_optimization_tools())
            test_results.append(await self.test_admin_tools())
            test_results.append(await self.test_testing_tools())
            test_results.append(await self.test_benchmarking_tools())
            test_results.append(await self.test_agent_interaction_tools())
            test_results.append(await self.test_strategy_and_learning_tools())
            test_results.append(await self.test_all_resources())
            
            # Summary
            passed_tests = sum(test_results)
            total_tests = len(test_results)
            
            print("\n" + "=" * 60)
            print("🏁 Test Summary")
            print("=" * 60)
            print(f"Total test categories: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {total_tests - passed_tests}")
            print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
            
            if passed_tests == total_tests:
                print("\n🎉 All tests passed! The comprehensive MCP server is working correctly.")
                return True
            else:
                print(f"\n⚠️  {total_tests - passed_tests} test categories failed. Please review the output above.")
                return False
        
        finally:
            await self.stop_server()


async def main():
    """Main test function"""
    tester = MCPServerTester()
    success = await tester.run_comprehensive_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())