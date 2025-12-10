# Quick Start Guide

Get SAFLA up and running in minutes with this streamlined guide. This tutorial will have you experiencing SAFLA's core capabilities quickly, with links to deeper documentation for each component.

## Prerequisites

Before starting, ensure you have:

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **pip** for package management
- **8GB+ RAM** recommended for optimal performance
- **Internet connection** for downloading dependencies

## 🚀 5-Minute Setup

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/ruvnet/SAFLA.git
cd SAFLA

# Create a virtual environment (recommended)
python -m venv safla-env
source safla-env/bin/activate  # On Windows: safla-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAFLA in development mode
pip install -e .
```

### Step 2: Basic Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit the configuration (optional for basic usage)
# nano .env  # or your preferred editor
```

**Minimal `.env` configuration:**
```bash
# Memory Configuration
SAFLA_VECTOR_DIMENSIONS=512,768
SAFLA_MAX_MEMORIES=1000
SAFLA_SIMILARITY_THRESHOLD=0.8

# Safety Configuration
SAFLA_MEMORY_LIMIT=1000000000  # 1GB
SAFLA_CPU_LIMIT=0.8
SAFLA_SAFETY_MONITORING_INTERVAL=1.0
```

### Step 3: Verify Installation

```bash
# Run basic tests to verify installation
python -m pytest tests/test_basic_functionality.py -v

# Check system status
python -c "from safla.core.hybrid_memory import HybridMemoryArchitecture; print('SAFLA installed successfully!')"
```

## 🧠 First Steps with SAFLA

### Basic Memory Operations

Create a simple script to test SAFLA's memory capabilities:

```python
# quickstart_demo.py
import asyncio
import numpy as np
from safla.core.hybrid_memory import HybridMemoryArchitecture
from safla.core.meta_cognitive_engine import MetaCognitiveEngine
from safla.core.safety_validation import SafetyValidationFramework

async def basic_memory_demo():
    """Demonstrate basic memory operations."""
    print("🧠 Initializing SAFLA Memory System...")
    
    # Initialize the hybrid memory architecture
    memory = HybridMemoryArchitecture()
    await memory.start()
    
    print("✅ Memory system started successfully!")
    
    # Store some vector memories
    print("\n📝 Storing vector memories...")
    
    # Create sample embeddings (512-dimensional)
    embedding1 = np.random.rand(512).tolist()
    embedding2 = np.random.rand(512).tolist()
    embedding3 = np.random.rand(512).tolist()
    
    # Store memories with metadata
    memory_id1 = await memory.store_vector_memory(
        content="Python is a programming language",
        embedding=embedding1,
        metadata={"type": "fact", "domain": "programming", "confidence": 0.9}
    )
    
    memory_id2 = await memory.store_vector_memory(
        content="Machine learning uses algorithms to learn patterns",
        embedding=embedding2,
        metadata={"type": "fact", "domain": "AI", "confidence": 0.95}
    )
    
    memory_id3 = await memory.store_vector_memory(
        content="SAFLA implements hybrid memory architecture",
        embedding=embedding3,
        metadata={"type": "fact", "domain": "SAFLA", "confidence": 1.0}
    )
    
    print(f"✅ Stored 3 memories: {memory_id1[:8]}..., {memory_id2[:8]}..., {memory_id3[:8]}...")
    
    # Search for similar memories
    print("\n🔍 Searching for similar memories...")
    
    # Create a query embedding similar to the first one
    query_embedding = np.array(embedding1) + np.random.normal(0, 0.1, 512)
    query_embedding = query_embedding.tolist()
    
    similar_memories = await memory.search_similar_memories(
        query_embedding=query_embedding,
        top_k=3,
        similarity_threshold=0.5
    )
    
    print(f"Found {len(similar_memories)} similar memories:")
    for i, mem in enumerate(similar_memories, 1):
        print(f"  {i}. {mem.content[:50]}... (similarity: {mem.similarity:.3f})")
    
    # Store an episodic memory
    print("\n📚 Storing episodic memory...")
    
    episode_id = await memory.store_episodic_memory(
        content="User asked about SAFLA's memory system",
        context={"user_id": "demo_user", "session": "quickstart"},
        outcome="provided_explanation",
        metadata={"duration": 30, "satisfaction": 0.9}
    )
    
    print(f"✅ Stored episode: {episode_id[:8]}...")
    
    # Add semantic knowledge
    print("\n🕸️ Building semantic knowledge...")
    
    # Add nodes to semantic memory
    python_node = await memory.add_semantic_node(
        content="Python Programming Language",
        node_type="concept",
        properties={"category": "programming", "popularity": "high"}
    )
    
    ai_node = await memory.add_semantic_node(
        content="Artificial Intelligence",
        node_type="concept",
        properties={"category": "technology", "complexity": "high"}
    )
    
    # Create relationship
    await memory.add_semantic_edge(
        source_id=python_node,
        target_id=ai_node,
        relationship="used_in",
        weight=0.8
    )
    
    print(f"✅ Created semantic relationship: Python -> AI")
    
    # Cleanup
    await memory.stop()
    print("\n🎉 Basic memory demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(basic_memory_demo())
```

Run the demo:
```bash
python quickstart_demo.py
```

### Meta-Cognitive Capabilities

Test SAFLA's self-awareness and goal management:

```python
# meta_cognitive_demo.py
import asyncio
from safla.core.meta_cognitive_engine import MetaCognitiveEngine

async def meta_cognitive_demo():
    """Demonstrate meta-cognitive capabilities."""
    print("🤖 Initializing Meta-Cognitive Engine...")
    
    # Initialize the meta-cognitive engine
    meta_engine = MetaCognitiveEngine()
    await meta_engine.start()
    
    print("✅ Meta-cognitive engine started!")
    
    # Add a goal
    print("\n🎯 Setting system goals...")
    
    goal_id = await meta_engine.goal_manager.add_goal(
        description="Optimize memory retrieval performance",
        priority=0.8,
        target_metrics={"retrieval_time": 0.1, "accuracy": 0.95}
    )
    
    print(f"✅ Added goal: {goal_id[:8]}...")
    
    # Monitor performance
    print("\n📊 Monitoring performance...")
    
    performance_data = {
        "retrieval_time": 0.15,
        "accuracy": 0.92,
        "memory_usage": 0.6,
        "cpu_usage": 0.4
    }
    
    await meta_engine.performance_monitor.update_metrics(performance_data)
    
    # Get system status
    status = await meta_engine.self_awareness.get_system_status()
    print(f"System Status: {status['overall_health']}")
    print(f"Active Goals: {status['active_goals']}")
    print(f"Performance Score: {status['performance_score']:.3f}")
    
    # Select strategy
    print("\n🧠 Strategy selection...")
    
    context = {
        "task_type": "memory_optimization",
        "current_performance": 0.92,
        "target_performance": 0.95,
        "available_resources": 0.6
    }
    
    available_strategies = [
        "increase_cache_size",
        "optimize_indexing",
        "parallel_processing",
        "memory_consolidation"
    ]
    
    selected_strategy = await meta_engine.strategy_selector.select_strategy(
        context=context,
        available_strategies=available_strategies
    )
    
    print(f"✅ Selected strategy: {selected_strategy}")
    
    # Cleanup
    await meta_engine.stop()
    print("\n🎉 Meta-cognitive demo completed!")

if __name__ == "__main__":
    asyncio.run(meta_cognitive_demo())
```

### Safety Validation

Test SAFLA's safety mechanisms:

```python
# safety_demo.py
import asyncio
from safla.core.safety_validation import SafetyValidationFramework, SafetyConstraint, ConstraintType

async def safety_demo():
    """Demonstrate safety validation capabilities."""
    print("🛡️ Initializing Safety Validation Framework...")
    
    # Initialize safety framework
    safety_framework = SafetyValidationFramework()
    await safety_framework.start()
    
    print("✅ Safety framework started!")
    
    # Add safety constraints
    print("\n⚠️ Adding safety constraints...")
    
    memory_constraint = SafetyConstraint(
        name="memory_limit",
        constraint_type=ConstraintType.HARD,
        description="Maximum memory usage limit",
        rule="memory_usage <= 1000000000",  # 1GB
        threshold=1000000000,
        violation_action="emergency_stop"
    )
    
    cpu_constraint = SafetyConstraint(
        name="cpu_limit",
        constraint_type=ConstraintType.SOFT,
        description="CPU usage warning threshold",
        rule="cpu_usage <= 0.8",
        threshold=0.8,
        violation_action="warning"
    )
    
    safety_framework.constraint_engine.add_constraint(memory_constraint)
    safety_framework.constraint_engine.add_constraint(cpu_constraint)
    
    print("✅ Added memory and CPU constraints")
    
    # Test system modification validation
    print("\n🔍 Testing system modification validation...")
    
    # Safe modification
    safe_modification = {
        "memory_usage": 500000000,  # 500MB
        "cpu_usage": 0.6,
        "new_capabilities": 1,
        "total_capabilities": 10
    }
    
    result = await safety_framework.validate_system_modification(safe_modification)
    print(f"Safe modification result: {'✅ APPROVED' if result.is_approved else '❌ REJECTED'}")
    
    # Unsafe modification
    unsafe_modification = {
        "memory_usage": 1500000000,  # 1.5GB (exceeds limit)
        "cpu_usage": 0.9,
        "new_capabilities": 1,
        "total_capabilities": 10
    }
    
    result = await safety_framework.validate_system_modification(unsafe_modification)
    print(f"Unsafe modification result: {'✅ APPROVED' if result.is_approved else '❌ REJECTED'}")
    if not result.is_approved:
        print(f"  Reason: {result.message}")
    
    # Create safety checkpoint
    print("\n💾 Creating safety checkpoint...")
    
    checkpoint_id = await safety_framework.create_safety_checkpoint(
        name="quickstart_demo",
        description="Checkpoint created during quickstart demo"
    )
    
    print(f"✅ Created checkpoint: {checkpoint_id[:8]}...")
    
    # Cleanup
    await safety_framework.stop()
    print("\n🎉 Safety demo completed!")

if __name__ == "__main__":
    asyncio.run(safety_demo())
```

## 🔗 MCP Integration (Optional)

If you want to test MCP (Model Context Protocol) integration:

```python
# mcp_demo.py
import asyncio
from safla.core.mcp_orchestration import MCPOrchestrator

async def mcp_demo():
    """Demonstrate MCP orchestration capabilities."""
    print("🔗 Initializing MCP Orchestrator...")
    
    # Initialize MCP orchestrator
    orchestrator = MCPOrchestrator()
    await orchestrator.start()
    
    print("✅ MCP orchestrator started!")
    
    # Register a simple MCP server (if available)
    print("\n📡 Registering MCP servers...")
    
    try:
        await orchestrator.server_manager.register_server(
            name="context7",
            connection_config={
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp"]
            }
        )
        print("✅ Registered context7 MCP server")
        
        # Test server health
        health_status = await orchestrator.server_manager.check_server_health("context7")
        print(f"Context7 health: {'✅ HEALTHY' if health_status else '❌ UNHEALTHY'}")
        
    except Exception as e:
        print(f"⚠️ MCP server registration failed: {e}")
        print("   This is normal if MCP servers are not installed")
    
    # Cleanup
    await orchestrator.stop()
    print("\n🎉 MCP demo completed!")

if __name__ == "__main__":
    asyncio.run(mcp_demo())
```

## 🧪 Running Tests

Verify your installation with the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_hybrid_memory.py -v
python -m pytest tests/test_meta_cognitive.py -v
python -m pytest tests/test_safety_validation.py -v

# Run tests with coverage
python -m pytest --cov=safla tests/
```

## 📊 Performance Monitoring

Monitor SAFLA's performance in real-time:

```python
# monitoring_demo.py
import asyncio
import time
from safla.core.delta_evaluation import DeltaEvaluator

async def monitoring_demo():
    """Demonstrate performance monitoring."""
    print("📊 Testing Delta Evaluation System...")
    
    evaluator = DeltaEvaluator()
    
    # Simulate performance improvements
    result = evaluator.evaluate_delta(
        performance_data={
            'current_reward': 0.92,
            'previous_reward': 0.85,
            'tokens_used': 1000
        },
        efficiency_data={
            'current_throughput': 150,
            'previous_throughput': 120,
            'resource_used': 0.8
        },
        stability_data={
            'divergence_score': 0.15
        },
        capability_data={
            'new_capabilities': 2,
            'total_capabilities': 10
        },
        context="quickstart_demo"
    )
    
    print(f"📈 Performance Delta: {result.performance_delta:.4f}")
    print(f"⚡ Efficiency Delta: {result.efficiency_delta:.4f}")
    print(f"🎯 Stability Delta: {result.stability_delta:.4f}")
    print(f"🚀 Capability Delta: {result.capability_delta:.4f}")
    print(f"🏆 Total Delta: {result.total_delta:.4f}")
    print(f"✨ Improvement Detected: {'Yes' if result.is_improvement() else 'No'}")

if __name__ == "__main__":
    asyncio.run(monitoring_demo())
```

## 🎯 Next Steps

Congratulations! You've successfully set up SAFLA and explored its core capabilities. Here's what to do next:

### Immediate Next Steps
1. **[Installation Guide](03-installation.md)** - Set up a production environment
2. **[System Architecture](04-architecture.md)** - Understand SAFLA's design
3. **[Configuration Guide](16-configuration.md)** - Customize SAFLA for your needs

### Explore Core Features
- **[Memory System](05-memory-system.md)** - Deep dive into hybrid memory
- **[Meta-Cognitive Engine](06-meta-cognitive.md)** - Self-awareness and adaptation
- **[Safety Framework](07-safety-validation.md)** - Comprehensive safety mechanisms

### Build Applications
- **[Use Cases](24-use-cases.md)** - Real-world applications
- **[Code Examples](25-examples.md)** - Practical implementations
- **[API Reference](22-api-reference.md)** - Complete API documentation

### Advanced Topics
- **[MCP Orchestration](09-mcp-orchestration.md)** - Distributed agent coordination
- **[Performance Tuning](29-performance-tuning.md)** - Optimization techniques
- **[Extension Development](23-extensions.md)** - Custom extensions

## 🆘 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure SAFLA is properly installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

**Memory Issues**
```bash
# Reduce memory limits in .env
SAFLA_MAX_MEMORIES=500
SAFLA_VECTOR_DIMENSIONS=512
```

**Performance Issues**
```bash
# Enable performance monitoring
SAFLA_PERFORMANCE_MONITORING=true
SAFLA_LOG_LEVEL=INFO
```

For more troubleshooting help, see the [Troubleshooting Guide](27-troubleshooting.md).

## 🤝 Getting Help

- **Documentation**: Browse the complete [documentation guide](README.md)
- **GitHub Issues**: Report bugs or request features
- **Community Forums**: Ask questions and share experiences
- **Examples Repository**: Find practical implementation examples

---

**Next**: [Installation Guide](03-installation.md) - Detailed setup for production environments  
**See Also**: [System Architecture](04-architecture.md) - Understanding SAFLA's design principles