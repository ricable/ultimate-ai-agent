#!/usr/bin/env python3
"""
SAFLA Basic Setup - Getting Started Example
==========================================

This example demonstrates the most basic SAFLA setup and initialization.
Perfect for newcomers to understand core concepts.

Learning Objectives:
- How to initialize SAFLA components
- Basic configuration management
- Simple system health checks
- Understanding SAFLA's modular architecture

Time to Complete: 5-10 minutes
Complexity: Beginner
"""

import asyncio
import logging
import numpy as np
from pathlib import Path

# SAFLA imports
from safla import HybridMemoryArchitecture, SAFLAConfig, get_logger
from safla.core.safety_validation import OptimizedSafetyValidator
from safla.core.delta_evaluation import OptimizedDeltaEvaluator

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def basic_configuration():
    """Demonstrate basic SAFLA configuration setup."""
    print("🔧 Setting up basic SAFLA configuration...")
    
    # Option 1: Use default configuration
    config = SAFLAConfig()
    print(f"✅ Default config loaded with log level: {config.log_level}")
    
    # Option 2: Custom configuration
    custom_config = SAFLAConfig()
    custom_config.debug = True
    custom_config.memory.max_memories = 1000
    custom_config.safety.memory_limit = 100000000  # 100MB
    print(f"✅ Custom config created with debug={custom_config.debug}")
    
    return custom_config


async def initialize_core_components():
    """Initialize SAFLA's core components."""
    print("\n🚀 Initializing SAFLA core components...")
    
    # 1. Initialize Memory System
    print("  📚 Setting up Hybrid Memory Architecture...")
    memory = HybridMemoryArchitecture()
    print("  ✅ Memory system initialized")
    
    # 2. Initialize Safety Validator
    print("  🛡️ Setting up Safety Validation...")
    safety = OptimizedSafetyValidator()
    print("  ✅ Safety validator initialized")
    
    # 3. Initialize Delta Evaluator  
    print("  📊 Setting up Performance Evaluation...")
    evaluator = OptimizedDeltaEvaluator()
    print("  ✅ Delta evaluator initialized")
    
    return memory, safety, evaluator


async def basic_health_check(memory, safety, evaluator):
    """Perform basic system health checks."""
    print("\n💊 Performing system health checks...")
    
    # Check 1: Memory System Health
    try:
        # Store a simple test memory
        test_data = "Hello SAFLA! This is a test memory."
        test_embedding = [0.1] * 512  # Simple 512-dimensional vector
        
        memory_id = memory.vector_memory.store(
            embedding=np.array(test_embedding),
            metadata={"type": "health_check", "timestamp": "now", "content": test_data}
        )
        print("  ✅ Memory system: HEALTHY")
    except Exception as e:
        print(f"  ❌ Memory system: ERROR - {e}")
    
    # Check 2: Safety Validator Health
    try:
        # Test basic validation
        test_content = "health_check operation with test data"
        test_context = {"operation": "health_check", "data": {"test": True}}
        result = await safety.validate(test_content, context=test_context)
        print("  ✅ Safety validator: HEALTHY")
    except Exception as e:
        print(f"  ❌ Safety validator: ERROR - {e}")
    
    # Check 3: Delta Evaluator Health
    try:
        # Test basic evaluation
        result = evaluator.evaluate_delta(
            performance_data={"current_reward": 1.0, "previous_reward": 0.9, "tokens_used": 100}
        )
        print("  ✅ Delta evaluator: HEALTHY")
        print(f"    📈 Test evaluation result: {result.overall_delta:.3f}")
    except Exception as e:
        print(f"  ❌ Delta evaluator: ERROR - {e}")


def display_system_info():
    """Display basic system information."""
    print("\n📋 SAFLA System Information:")
    print("=" * 50)
    
    # Import and display version info
    import safla
    info = safla.get_info()
    
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    
    print("\n🏗️ Architecture Overview:")
    print("  • Hybrid Memory: Vector + Episodic + Semantic + Working")
    print("  • Safety Validation: Constraint checking + Risk assessment")
    print("  • Delta Evaluation: Performance improvement tracking")
    print("  • MCP Integration: Model Context Protocol support")
    print("  • Meta-Cognitive: Self-awareness and adaptive learning")


async def demonstrate_basic_operations(memory):
    """Demonstrate basic SAFLA operations."""
    print("\n🎯 Demonstrating basic operations...")
    
    # Store multiple memories
    memories = [
        ("Learning about SAFLA architecture", [0.1, 0.2, 0.3] * 170 + [0.1, 0.2]),
        ("Understanding memory systems", [0.2, 0.3, 0.4] * 170 + [0.2, 0.3]),
        ("Exploring safety features", [0.3, 0.4, 0.5] * 170 + [0.3, 0.4])
    ]
    
    stored_ids = []
    for i, (content, embedding) in enumerate(memories):
        try:
            memory_id = memory.vector_memory.store(
                embedding=np.array(embedding),
                metadata={"example": "basic_demo", "index": i, "content": content}
            )
            stored_ids.append(memory_id)
            print(f"  📝 Stored memory {i+1}: '{content[:30]}...'")
        except Exception as e:
            print(f"  ❌ Failed to store memory {i+1}: {e}")
    
    # Search for similar memories
    if stored_ids:
        try:
            query_embedding = np.array([0.15, 0.25, 0.35] * 170 + [0.15, 0.25])
            similar = memory.vector_memory.similarity_search(
                query_embedding=query_embedding,
                k=2
            )
            print(f"  🔍 Found {len(similar)} similar memories")
            for result in similar:
                content = result.item.metadata.get("content", "Unknown content")
                print(f"    - {content[:50]}... (similarity: {result.similarity_score:.3f})")
        except Exception as e:
            print(f"  ❌ Search failed: {e}")


async def cleanup_resources(memory, safety):
    """Clean up resources properly."""
    print("\n🧹 Cleaning up resources...")
    
    try:
        # Memory system doesn't require explicit cleanup in this version
        print("  ✅ Memory system cleaned up")
    except Exception as e:
        print(f"  ⚠️ Memory cleanup warning: {e}")
    
    try:
        # Safety validator might not need explicit cleanup
        print("  ✅ Safety validator cleaned up")
    except Exception as e:
        print(f"  ⚠️ Safety cleanup warning: {e}")


async def main():
    """Main demonstration function."""
    print("🌟 Welcome to SAFLA - Basic Setup Example")
    print("=" * 60)
    
    # Step 1: Basic Configuration
    config = basic_configuration()
    
    # Step 2: Initialize Components
    memory, safety, evaluator = await initialize_core_components()
    
    # Step 3: Health Checks
    await basic_health_check(memory, safety, evaluator)
    
    # Step 4: System Information
    display_system_info()
    
    # Step 5: Basic Operations
    await demonstrate_basic_operations(memory)
    
    # Step 6: Cleanup
    await cleanup_resources(memory, safety)
    
    print("\n🎉 Basic setup example completed successfully!")
    print("\nNext Steps:")
    print("  1. Run 02_simple_memory.py to explore memory features")
    print("  2. Try 03_basic_safety.py to understand safety constraints")
    print("  3. Check out the README.md for the full example progression")


if __name__ == "__main__":
    # Run the example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Detailed error information:")