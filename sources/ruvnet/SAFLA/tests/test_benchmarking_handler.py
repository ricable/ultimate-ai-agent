#!/usr/bin/env python3
"""
Test script to verify BenchmarkingHandler functionality.
"""

import asyncio
import json
from safla.utils.config import get_config
from safla.mcp.handlers.benchmarking import BenchmarkingHandler
from safla.mcp.state.manager import StateManager


async def test_benchmarking_handler():
    """Test the BenchmarkingHandler methods."""
    
    # Initialize components
    config = get_config()
    state_manager = StateManager()
    handler = BenchmarkingHandler(config, state_manager)
    
    print("=== Testing BenchmarkingHandler ===\n")
    
    # Test 1: Benchmark vector operations
    print("1. Benchmarking vector operations...")
    try:
        vector_result = await handler._benchmark_vector_operations(
            operation_types=["insert", "search"],
            vector_dimensions=512,
            num_iterations=10
        )
        print(f"Vector benchmark completed: {vector_result}")
    except Exception as e:
        print(f"Error running vector benchmark: {e}")
    print()
    
    # Test 2: Benchmark memory performance
    print("2. Benchmarking memory performance...")
    try:
        memory_result = await handler._benchmark_memory_performance(
            memory_tiers=["L1", "L2", "L3"],
            workload_type="mixed",
            data_size_mb=10.0
        )
        print(f"Memory benchmark completed: {memory_result}")
    except Exception as e:
        print(f"Error running memory benchmark: {e}")
    print()
    
    # Test 3: Benchmark MCP throughput
    print("3. Benchmarking MCP throughput...")
    try:
        mcp_result = await handler._benchmark_mcp_throughput(
            request_types=["tool_call", "resource_read"],
            concurrent_clients=5,
            duration_seconds=5
        )
        print(f"MCP benchmark completed: {mcp_result}")
    except Exception as e:
        print(f"Error running MCP benchmark: {e}")
    print()
    
    # Test 4: List all registered tools
    print("4. Registered tools:")
    tools = handler.get_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n=== BenchmarkingHandler test complete ===")


if __name__ == "__main__":
    asyncio.run(test_benchmarking_handler())