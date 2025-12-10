#!/usr/bin/env python3
"""Test enhanced endpoints with correct parameter names."""

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
    """Test enhanced endpoints with correct parameters."""
    base_url = "https://safla.fly.dev/api/safla"
    
    # Test cases with corrected parameter names
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
                "texts": ["SAFLA is an AI system", "AI systems use optimization", "Neural networks power AI"],
                "relationship_depth": 2
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
                "memory_type": "recent",
                "time_range": {"hours": 24}
            }
        },
        {
            "name": "optimize_parameters",
            "method": "optimize_parameters",
            "params": {
                "workload_type": "inference",
                "target_metric": "performance"
            }
        },
        {
            "name": "create_session",
            "method": "create_session",
            "params": {
                "session_name": "test_session",
                "context": {"type": "optimization"},
                "expiration": 3600
            }
        },
        {
            "name": "export_memory_snapshot",
            "method": "export_memory_snapshot",
            "params": {
                "memory_types": ["working", "long_term"],
                "format": "json"
            }
        },
        {
            "name": "run_benchmark",
            "method": "run_benchmark",
            "params": {
                "benchmark_type": "throughput",
                "duration": 10
            }
        },
        {
            "name": "monitor_health",
            "method": "monitor_health",
            "params": {
                "include_metrics": True,
                "include_errors": True
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        print("Testing Enhanced SAFLA Endpoints with Correct Parameters")
        print("=" * 80)
        
        success_count = 0
        error_count = 0
        
        for test in test_cases:
            print(f"\n🧪 Testing: {test['name']}")
            print(f"Method: {test['method']}")
            
            result = await test_endpoint(session, base_url, test['method'], test['params'])
            
            if "error" in result:
                if isinstance(result.get("error"), dict):
                    print(f"❌ Error: {result['error'].get('message', 'Unknown error')}")
                else:
                    print(f"❌ Error: {result['error']}")
                error_count += 1
            elif "result" in result:
                print(f"✅ Success!")
                # Show first few fields of result
                result_preview = {}
                count = 0
                for key, value in result['result'].items():
                    if count < 3:
                        result_preview[key] = value
                        count += 1
                    else:
                        break
                print(f"Result preview: {json.dumps(result_preview, indent=2)}")
                success_count += 1
            else:
                print(f"⚠️  Unexpected response format")
                error_count += 1
            
            print("-" * 40)
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        print(f"\n" + "=" * 80)
        print(f"🎯 Testing Complete!")
        print(f"✅ Success: {success_count}/{len(test_cases)} endpoints")
        print(f"❌ Errors: {error_count}/{len(test_cases)} endpoints")
        
        if success_count == len(test_cases):
            print("🎉 All enhanced endpoints working perfectly!")
        elif success_count > len(test_cases) // 2:
            print("🟡 Most endpoints working - minor fixes needed")
        else:
            print("🔴 Significant issues detected - major fixes needed")

if __name__ == "__main__":
    asyncio.run(main())