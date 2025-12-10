#!/usr/bin/env python3
"""
SAFLA MCP Client Setup
=====================

This script sets up and demonstrates how to connect to the SAFLA system
via MCP (Model Context Protocol) hosted on Fly.io.
"""

import asyncio
import json
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAFLAMCPClient:
    """Client for connecting to SAFLA MCP server on Fly.io."""
    
    def __init__(self, base_url: str = "https://safla.fly.dev"):
        self.base_url = base_url
        self.session = None
        self.connected = False
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def connect(self) -> bool:
        """Connect to SAFLA MCP server."""
        logger.info(f"🔗 Connecting to SAFLA MCP server at {self.base_url}")
        
        try:
            # Check health endpoint
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✅ SAFLA server is healthy")
                    logger.info(f"   GPU Available: {health_data.get('gpu_available', False)}")
                    logger.info(f"   Version: {health_data.get('version', 'unknown')}")
                    self.connected = True
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def call_mcp_method(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call an MCP method on the SAFLA server."""
        if not self.connected:
            raise RuntimeError("Not connected to SAFLA server")
        
        mcp_request = {
            "jsonrpc": "2.0",
            "id": datetime.now().isoformat(),
            "method": method,
            "params": params or {}
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/safla",
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"MCP call failed: {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            logger.error(f"MCP call exception: {e}")
            return {"error": str(e)}
    
    async def generate_embeddings(self, texts: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate embeddings using SAFLA's optimized engine."""
        params = {
            "texts": texts,
            "config": config or {
                "batch_size": 256,
                "use_flash_attention_2": True,
                "mixed_precision": "fp32",
                "cache_embeddings": True
            }
        }
        
        logger.info(f"🧠 Generating embeddings for {len(texts)} texts...")
        result = await self.call_mcp_method("generate_embeddings", params)
        
        if "error" not in result:
            logger.info(f"✅ Embeddings generated successfully")
        
        return result
    
    async def optimize_model(self, optimization_type: str = "extreme") -> Dict[str, Any]:
        """Trigger model optimization on the SAFLA server."""
        params = {
            "optimization_type": optimization_type,
            "target_improvement": 100.0
        }
        
        logger.info(f"⚡ Starting {optimization_type} optimization...")
        result = await self.call_mcp_method("optimize_model", params)
        
        return result
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from SAFLA."""
        logger.info("📊 Retrieving performance metrics...")
        result = await self.call_mcp_method("get_performance_metrics")
        
        return result
    
    async def run_benchmark(self, benchmark_type: str = "embedding_performance") -> Dict[str, Any]:
        """Run performance benchmarks."""
        params = {
            "benchmark_type": benchmark_type,
            "num_samples": 100
        }
        
        logger.info(f"🏁 Running {benchmark_type} benchmark...")
        result = await self.call_mcp_method("run_benchmark", params)
        
        return result
    
    async def store_memory(self, content: str, memory_type: str = "episodic") -> Dict[str, Any]:
        """Store data in SAFLA's hybrid memory system."""
        params = {
            "content": content,
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"💾 Storing memory: {content[:50]}...")
        result = await self.call_mcp_method("store_memory", params)
        
        return result
    
    async def retrieve_memories(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Retrieve memories from SAFLA's hybrid memory system."""
        params = {
            "query": query,
            "limit": limit
        }
        
        logger.info(f"🔍 Retrieving memories for: {query}")
        result = await self.call_mcp_method("retrieve_memories", params)
        
        return result
    
    async def safety_validate(self, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate an action using SAFLA's safety framework."""
        params = {
            "action": action,
            "context": context or {}
        }
        
        logger.info(f"🛡️ Safety validating: {action}")
        result = await self.call_mcp_method("safety_validate", params)
        
        return result

async def demonstrate_safla_mcp_usage():
    """Demonstrate various SAFLA MCP capabilities."""
    print("🚀 SAFLA MCP Client Demonstration")
    print("=" * 50)
    
    async with SAFLAMCPClient() as client:
        # Connect to SAFLA server
        connected = await client.connect()
        
        if not connected:
            print("❌ Could not connect to SAFLA server")
            print("💡 Make sure the SAFLA instance is running at https://safla.fly.dev")
            return
        
        print("\n🎯 Testing SAFLA MCP capabilities...")
        
        # 1. Generate embeddings
        print("\n1. 🧠 Testing Embedding Generation")
        embedding_texts = [
            "SAFLA is a self-aware AI system",
            "Machine learning optimization is crucial",
            "Neural embeddings represent semantic meaning"
        ]
        
        embedding_result = await client.generate_embeddings(embedding_texts)
        print(f"   Result: {embedding_result.get('status', 'unknown')}")
        
        # 2. Store and retrieve memories
        print("\n2. 💾 Testing Memory System")
        memory_result = await client.store_memory(
            "SAFLA achieved 178,146% performance improvement through cache optimization"
        )
        print(f"   Store result: {memory_result.get('status', 'unknown')}")
        
        retrieval_result = await client.retrieve_memories("performance improvement")
        print(f"   Retrieval result: {retrieval_result.get('status', 'unknown')}")
        
        # 3. Safety validation
        print("\n3. 🛡️ Testing Safety Framework")
        safety_result = await client.safety_validate(
            "model_optimization",
            {"risk_level": 0.2, "optimization_type": "cache"}
        )
        print(f"   Safety result: {safety_result.get('status', 'unknown')}")
        
        # 4. Performance metrics
        print("\n4. 📊 Getting Performance Metrics")
        metrics_result = await client.get_performance_metrics()
        print(f"   Metrics result: {metrics_result.get('status', 'unknown')}")
        
        # 5. Run benchmark
        print("\n5. 🏁 Running Performance Benchmark")
        benchmark_result = await client.run_benchmark("embedding_performance")
        print(f"   Benchmark result: {benchmark_result.get('status', 'unknown')}")
        
        print("\n✅ SAFLA MCP demonstration complete!")
        print(f"🌐 Server URL: https://safla.fly.dev")
        print(f"📚 Full API documentation available at the server")

def create_mcp_config_file():
    """Create MCP configuration file for easy connection."""
    config = {
        "safla_mcp": {
            "server_url": "https://safla.fly.dev",
            "timeout": 30,
            "retry_attempts": 3,
            "optimization_config": {
                "batch_size": 256,
                "use_flash_attention_2": True,
                "mixed_precision": "fp32",
                "cache_embeddings": True,
                "normalize_embeddings": True
            },
            "endpoints": {
                "health": "/health",
                "api": "/api/safla",
                "optimize": "/api/safla/optimize",
                "benchmark": "/api/safla/benchmark"
            }
        }
    }
    
    config_file = "/workspaces/SAFLA/config/mcp_client_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"📝 MCP client configuration saved to {config_file}")
    return config_file

def create_usage_examples():
    """Create usage examples for different programming languages."""
    
    # Python example
    python_example = '''#!/usr/bin/env python3
"""
Example: Using SAFLA via MCP in Python
"""

import asyncio
import aiohttp

async def use_safla_mcp():
    async with aiohttp.ClientSession() as session:
        # Generate embeddings
        mcp_request = {
            "jsonrpc": "2.0",
            "id": "embed_001",
            "method": "generate_embeddings",
            "params": {
                "texts": ["Hello SAFLA!", "This is a test"],
                "config": {"batch_size": 256, "cache_embeddings": True}
            }
        }
        
        async with session.post(
            "https://safla.fly.dev/api/safla",
            json=mcp_request
        ) as response:
            result = await response.json()
            print(f"Embeddings generated: {result}")

if __name__ == "__main__":
    asyncio.run(use_safla_mcp())
'''
    
    # JavaScript example
    javascript_example = '''// Example: Using SAFLA via MCP in JavaScript/Node.js

const axios = require('axios');

async function useSAFLAMCP() {
    const mcpRequest = {
        jsonrpc: "2.0",
        id: "embed_001",
        method: "generate_embeddings",
        params: {
            texts: ["Hello SAFLA!", "This is a test"],
            config: { batch_size: 256, cache_embeddings: true }
        }
    };
    
    try {
        const response = await axios.post(
            'https://safla.fly.dev/api/safla',
            mcpRequest
        );
        console.log('Embeddings generated:', response.data);
    } catch (error) {
        console.error('Error:', error);
    }
}

useSAFLAMCP();
'''
    
    # cURL example
    curl_example = '''#!/bin/bash
# Example: Using SAFLA via MCP with cURL

curl -X POST https://safla.fly.dev/api/safla \\
  -H "Content-Type: application/json" \\
  -d '{
    "jsonrpc": "2.0",
    "id": "embed_001", 
    "method": "generate_embeddings",
    "params": {
      "texts": ["Hello SAFLA!", "This is a test"],
      "config": {
        "batch_size": 256,
        "cache_embeddings": true
      }
    }
  }'
'''
    
    # Save examples
    examples_dir = "/workspaces/SAFLA/examples/mcp_usage"
    import os
    os.makedirs(examples_dir, exist_ok=True)
    
    with open(f"{examples_dir}/python_example.py", 'w') as f:
        f.write(python_example)
    
    with open(f"{examples_dir}/javascript_example.js", 'w') as f:
        f.write(javascript_example)
    
    with open(f"{examples_dir}/curl_example.sh", 'w') as f:
        f.write(curl_example)
    
    logger.info(f"📚 Usage examples created in {examples_dir}")

async def main():
    """Main function."""
    print("🔧 Setting up SAFLA MCP Client")
    
    # Create configuration files
    config_file = create_mcp_config_file()
    create_usage_examples()
    
    print(f"✅ Configuration created: {config_file}")
    print("✅ Usage examples created in examples/mcp_usage/")
    
    # Run demonstration
    print("\n🚀 Running SAFLA MCP demonstration...")
    await demonstrate_safla_mcp_usage()

if __name__ == "__main__":
    asyncio.run(main())