# SAFLA Optimization Guide for Claude-Flow Orchestration

## Overview

This guide provides optimization strategies for agents working on the SAFLA system through the claude-flow orchestration system. Follow these guidelines to ensure consistent, high-quality optimizations across the codebase.

## Key Optimization Principles

### 1. File Size Management (Critical)
- **Hard Limit**: No file should exceed 500 lines
- **Target**: Aim for 200-300 lines per file
- **Strategy**: Use modular decomposition and single responsibility principle

### 2. Performance Optimization Patterns

#### Async-First Design
```python
# ❌ Bad: Synchronous blocking
def process_data(data):
    time.sleep(1)  # Blocking
    return result

# ✅ Good: Asynchronous non-blocking
async def process_data(data):
    await asyncio.sleep(1)  # Non-blocking
    return result
```

#### Resource Pooling
```python
# ✅ Implement connection/resource pools
class ResourcePool:
    def __init__(self, max_size=10):
        self._pool = asyncio.Queue(maxsize=max_size)
        self._created = 0
    
    async def acquire(self):
        # Reuse existing resources
        pass
```

### 3. Memory Optimization Strategies

#### Lazy Loading
- Load large objects only when needed
- Use generators for large datasets
- Implement pagination for collections

#### Caching with TTL
```python
# Use the StateManager with TTL for temporary data
state_manager.set("cache_key", data, namespace="cache", ttl=300)  # 5 min TTL
```

## Claude-Flow Orchestration Commands

### For Optimization Tasks

```bash
# Analyze specific module for optimization
npx claude-flow sparc run refinement-optimization-mode "optimize safla.core.hybrid_memory module"

# Profile performance bottlenecks
npx claude-flow sparc run refinement-optimization-mode "profile SAFLA API endpoints"

# Refactor large files
npx claude-flow sparc run code "refactor meta_cognitive_engine.py into modular structure"
```

### Memory Management

```bash
# Store optimization progress
npx claude-flow memory store optimization_<module> "Completed refactoring of X module, reduced from Y to Z lines"

# Query previous optimizations
npx claude-flow memory query optimization

# Track performance improvements
npx claude-flow memory store perf_<feature> "Improved response time by X% through Y optimization"
```

## Parallel Optimization Workflow

Use BatchTool for concurrent optimization tasks:

```bash
# Parallel module optimization
batchtool run --parallel \
  "npx claude-flow sparc run refinement-optimization-mode 'optimize hybrid memory module' --non-interactive" \
  "npx claude-flow sparc run refinement-optimization-mode 'optimize meta-cognitive engine' --non-interactive" \
  "npx claude-flow sparc run refinement-optimization-mode 'optimize MCP orchestration' --non-interactive"

# Boomerang pattern for iterative optimization
batchtool orchestrate --boomerang \
  --phase1 "npx claude-flow sparc run refinement-optimization-mode 'analyze performance bottlenecks' --non-interactive" \
  --phase2 "npx claude-flow sparc run code 'implement performance fixes' --non-interactive" \
  --phase3 "npx claude-flow sparc run tdd 'create performance tests' --non-interactive" \
  --phase4 "npx claude-flow sparc run refinement-optimization-mode 'validate improvements' --non-interactive"
```

## Optimization Checklist

### Before Starting
- [ ] Analyze current file sizes and structure
- [ ] Profile performance bottlenecks
- [ ] Review existing tests
- [ ] Check memory usage patterns

### During Optimization
- [ ] Keep files under 500 lines
- [ ] Replace synchronous operations with async
- [ ] Implement proper error handling
- [ ] Add caching where beneficial
- [ ] Use resource pooling for expensive operations

### After Optimization
- [ ] Run all tests
- [ ] Benchmark performance improvements
- [ ] Document changes
- [ ] Store progress in claude-flow memory

## Current Optimization Status

### Completed ✅
1. **MCP Server Modularization**
   - Reduced from 3,284 lines to modular structure
   - Average module size: 285 lines
   - Improved testability and maintainability

### In Progress 🔄
1. **Core Module Refactoring**
   - Meta-Cognitive Engine (1,669 lines)
   - Hybrid Memory (1,369 lines)
   - ML Neural Embedding Engine (1,336 lines)

### Pending 📋
1. **Configuration Unification**
   - Migrate to single Pydantic-based system
   - Remove duplicate configuration approaches

2. **Performance Optimizations**
   - Replace time.sleep() with async equivalents
   - Implement connection pooling
   - Add caching layer

## Best Practices for Agents

### 1. Communication
- Store progress updates frequently
- Document architectural decisions
- Share optimization patterns discovered

### 2. Coordination
```bash
# Check what other agents are working on
npx claude-flow memory query optimization

# Avoid conflicts by claiming modules
npx claude-flow memory store optimization_claim_<module> "Agent X working on module Y"
```

### 3. Testing
- Always run tests before and after optimization
- Create benchmarks to measure improvements
- Add new tests for refactored code

## Quick Reference

### File Size Limits
- Maximum: 500 lines (hard limit)
- Target: 200-300 lines
- Critical files (>1000 lines): Urgent refactoring needed

### Performance Targets
- Response time improvement: 30%
- Memory usage reduction: 20%
- Code coverage: >90%

### Key Metrics to Track
1. File sizes before/after
2. Performance benchmarks
3. Memory usage
4. Test coverage
5. Cyclomatic complexity

## Next Steps for Agents

1. **Claim a module** to optimize using claude-flow memory
2. **Analyze** the module structure and dependencies
3. **Plan** the refactoring approach
4. **Implement** optimizations incrementally
5. **Test** thoroughly at each step
6. **Document** changes and improvements
7. **Share** learnings with other agents

Remember: The goal is not just to reduce file sizes, but to improve overall system performance, maintainability, and scalability. Each optimization should make the system better, not just different.