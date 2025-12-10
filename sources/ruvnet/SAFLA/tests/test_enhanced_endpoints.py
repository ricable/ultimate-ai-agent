#!/usr/bin/env python3
"""Test the enhanced SAFLA endpoints on the deployed Fly.io instance."""

import asyncio
import aiohttp
import json
from typing import Dict, Any

async def test_endpoint(session: aiohttp.ClientSession, url: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single endpoint."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return result
    except Exception as e:
        return {"error": str(e)}

async def main():
    """Test all enhanced endpoints."""
    base_url = "https://safla.fly.dev/api/safla"
    
    # Test cases for each enhanced endpoint
    test_cases = [
        {
            "name": "analyze_text",
            "method": "analyze_text",
            "params": {
                "text": "SAFLA is an advanced self-aware AI system with extreme optimization capabilities.",
                "analysis_type": "all",
                "depth": "deep"
            }
        },
        {
            "name": "detect_patterns",
            "method": "detect_patterns",
            "params": {
                "data": ["pattern1", "pattern2", "pattern1", "pattern3", "pattern2"],
                "pattern_type": "frequency",
                "threshold": 0.3
            }
        },
        {
            "name": "build_knowledge_graph",
            "method": "build_knowledge_graph",
            "params": {
                "entities": ["SAFLA", "AI", "optimization", "neural network"],
                "relationships": [
                    ["SAFLA", "is-a", "AI"],
                    ["SAFLA", "uses", "optimization"],
                    ["AI", "based-on", "neural network"]
                ],
                "max_depth": 3
            }
        },
        {
            "name": "batch_process",
            "method": "batch_process",
            "params": {
                "data": ["item1", "item2", "item3", "item4", "item5"],
                "operation": "embed",
                "batch_size": 256
            }
        },
        {
            "name": "consolidate_memories",
            "method": "consolidate_memories",
            "params": {
                "memory_keys": ["test_memory_1", "test_memory_2"],
                "strategy": "merge",
                "max_size": 1000
            }
        },
        {
            "name": "optimize_parameters",
            "method": "optimize_parameters",
            "params": {
                "target_metric": "performance",
                "constraints": {"max_memory": 4096, "max_time": 60},
                "iterations": 10
            }
        },
        {
            "name": "create_session",
            "method": "create_session",
            "params": {
                "session_type": "optimization",
                "duration": 3600,
                "config": {"auto_save": True, "checkpoint_interval": 300}
            }
        },
        {
            "name": "export_memory_snapshot",
            "method": "export_memory_snapshot",
            "params": {
                "namespaces": ["default", "session"],
                "format": "json",
                "include_metadata": True
            }
        },
        {
            "name": "run_benchmark",
            "method": "run_benchmark",
            "params": {
                "benchmark_type": "throughput",
                "duration": 10,
                "warmup": 2
            }
        },
        {
            "name": "monitor_health",
            "method": "monitor_health",
            "params": {
                "include_metrics": True,
                "include_predictions": True,
                "window": 60
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        print("Testing Enhanced SAFLA Endpoints on Fly.io\n")
        print("=" * 80)
        
        for test in test_cases:
            print(f"\nTesting: {test['name']}")
            print(f"Method: {test['method']}")
            print(f"Params: {json.dumps(test['params'], indent=2)}")
            
            result = await test_endpoint(session, base_url, test['method'], test['params'])
            
            if "error" in result:
                if isinstance(result.get("error"), dict):
                    print(f"❌ Error: {result['error'].get('message', 'Unknown error')}")
                else:
                    print(f"❌ Error: {result['error']}")
            elif "result" in result:
                print(f"✅ Success!")
                # Pretty print the result (truncated for readability)
                result_str = json.dumps(result['result'], indent=2)
                if len(result_str) > 500:
                    print(f"Result: {result_str[:500]}...")
                else:
                    print(f"Result: {result_str}")
            else:
                print(f"⚠️  Unexpected response: {json.dumps(result, indent=2)[:200]}...")
            
            print("-" * 40)
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        print("\n" + "=" * 80)
        print("Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())