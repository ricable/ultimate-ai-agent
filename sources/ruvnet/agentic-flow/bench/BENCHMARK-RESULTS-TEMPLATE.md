# ReasoningBank Benchmark Results Template

> This template shows expected results format. Actual results are generated after running benchmarks.

## Expected Results (Based on Paper)

### Overall Performance

| Metric | Baseline | ReasoningBank | Improvement |
|--------|----------|---------------|-------------|
| Success Rate | 0-20% | 80-100% | +80% |
| Avg Tokens/Task | 1,200 | 810 | -32.3% |
| Avg Latency | 2.5s | 2.8s | +12% |
| Memory Efficiency | N/A | 85% reuse | N/A |

### Per-Scenario Expectations

#### Coding Tasks
**Tasks**: 10 implementation tasks (deduplication, debounce, LRU cache, etc.)

**Expected Pattern**:
- Iteration 1: Baseline 20%, ReasoningBank 10% (cold start)
- Iteration 2: Baseline 20%, ReasoningBank 60% (learning)
- Iteration 3: Baseline 20%, ReasoningBank 90% (mastery)

**Key Learnings**:
- Use `Set` for deduplication
- Use `setTimeout`/`clearTimeout` for debounce/throttle
- Use `Map` for cache implementations
- Recursive patterns for tree/array operations

#### Debugging Tasks
**Tasks**: 10 bug-fixing tasks (off-by-one, race conditions, memory leaks, etc.)

**Expected Pattern**:
- Iteration 1: Baseline 30%, ReasoningBank 20%
- Iteration 2: Baseline 30%, ReasoningBank 70%
- Iteration 3: Baseline 30%, ReasoningBank 95%

**Key Learnings**:
- Off-by-one: Use `<` instead of `<=` in loops
- Race conditions: Use locks/queues for async operations
- Memory leaks: Always `removeEventListener` in cleanup
- Type coercion: Use `===` instead of `==`

#### API Design Tasks
**Tasks**: 10 API design tasks (authentication, CRUD, pagination, etc.)

**Expected Pattern**:
- Iteration 1: Baseline 40%, ReasoningBank 30%
- Iteration 2: Baseline 40%, ReasoningBank 75%
- Iteration 3: Baseline 40%, ReasoningBank 90%

**Key Learnings**:
- REST verbs: GET (read), POST (create), PUT/PATCH (update), DELETE (remove)
- Authentication: JWT tokens with refresh mechanism
- Pagination: Cursor-based or offset/limit patterns
- Error responses: Consistent schema with status codes

#### Problem Solving Tasks
**Tasks**: 10 algorithmic problems (two sum, valid parentheses, BFS, DP, etc.)

**Expected Pattern**:
- Iteration 1: Baseline 15%, ReasoningBank 5%
- Iteration 2: Baseline 15%, ReasoningBank 50%
- Iteration 3: Baseline 15%, ReasoningBank 85%

**Key Learnings**:
- Hash map for O(n) lookups (two sum)
- Stack for matching problems (parentheses)
- BFS/Queue for level-order traversal
- Dynamic programming for optimization problems

### Learning Velocity Analysis

**Baseline Agent**:
- No improvement across iterations (stateless)
- Each iteration starts from scratch
- Consistent performance at 20-40% success

**ReasoningBank Agent**:
- Cold start penalty in iteration 1 (10-30% success)
- Rapid learning in iteration 2 (50-70% success)
- Mastery in iteration 3 (85-95% success)
- 3-4x faster to consistent high performance

### Token Efficiency Analysis

**Baseline Token Breakdown** (average per task):
```
Problem understanding: 300 tokens
Solution reasoning: 600 tokens
Code generation: 300 tokens
Total: 1,200 tokens
```

**ReasoningBank Token Breakdown** (average per task, after learning):
```
Problem understanding: 200 tokens (memory context reduces need)
Solution reasoning: 250 tokens (memory provides patterns)
Code generation: 300 tokens (similar to baseline)
Memory injection: 60 tokens (3 memories @ 20 tokens each)
Total: 810 tokens (-32.3%)
```

**Savings Breakdown**:
- 100 tokens: Reduced problem analysis (context from memory)
- 350 tokens: Reduced reasoning (patterns from memory)
- Net savings: 450 tokens per task (37.5%)
- Memory overhead: 60 tokens (reduces net to 32.3%)

### Latency Analysis

**Baseline Latency** (average per task):
```
API call: 2,000ms
Processing: 500ms
Total: 2,500ms
```

**ReasoningBank Latency** (average per task):
```
Memory retrieval: 150ms (database + embedding search)
API call: 2,000ms (same model)
Processing: 500ms (same as baseline)
Memory distillation: 100ms (judge + distill)
Consolidation (amortized): 50ms (runs every 100 memories)
Total: 2,800ms (+12%)
```

**Overhead Breakdown**:
- Retrieval: 150ms (6% overhead)
- Distillation: 100ms (4% overhead)
- Consolidation: 50ms (2% overhead)
- Total: 300ms (12% overhead)

**Is 12% acceptable?**
- ✅ Yes: Enables 80% success improvement and 32% token savings
- ✅ Amortizes over time (early iterations pay more, later less)
- ✅ Can be optimized with caching and indexing

### Memory Efficiency Analysis

**Memory Creation**:
- Iteration 1: ~10 memories per scenario (high failure rate)
- Iteration 2: ~8 memories per scenario (improving)
- Iteration 3: ~5 memories per scenario (mostly success)
- Total: ~23 memories per scenario after 3 iterations

**Memory Usage**:
- Iteration 1: 0 memories used (none available)
- Iteration 2: ~15 memory retrievals (reusing from iteration 1)
- Iteration 3: ~20 memory retrievals (reusing from iterations 1-2)
- Usage ratio: 35 uses / 23 created = 1.52x reuse

**Memory Quality**:
- High confidence (>0.8): ~60% of memories
- Medium confidence (0.5-0.8): ~30% of memories
- Low confidence (<0.5): ~10% of memories (pruned by consolidation)

### Consolidation Analysis

**Duplicate Detection**:
- Iteration 1: 0 duplicates (too few memories)
- Iteration 2: ~2 duplicates per scenario (similar solutions)
- Iteration 3: ~3 duplicates per scenario (convergence)
- Total pruned: ~5 duplicates per scenario (22% reduction)

**Contradiction Detection**:
- Expected: Very low (judge filters bad memories)
- Typical: 0-1 contradictions per scenario
- Resolution: Lower confidence memory removed

**Pruning Strategy**:
- Remove memories with confidence < 0.3
- Keep max 1,000 memories (configurable)
- Prioritize: High reliability + recent + high confidence

### Statistical Significance

**Confidence Intervals** (95%):
- Success rate improvement: ±5%
- Token savings: ±3%
- Latency overhead: ±50ms

**P-values** (null hypothesis: no improvement):
- Success rate: p < 0.001 (highly significant)
- Token efficiency: p < 0.001 (highly significant)
- Latency: p < 0.01 (significant)

**Effect Sizes**:
- Success rate: Cohen's d = 3.2 (very large effect)
- Token efficiency: Cohen's d = 2.1 (large effect)
- Latency: Cohen's d = 0.4 (small effect)

## How to Interpret Your Results

### Success Rate
- **Great**: >70% improvement over baseline
- **Good**: 40-70% improvement
- **Needs tuning**: <40% improvement

### Token Efficiency
- **Great**: >25% savings
- **Good**: 15-25% savings
- **Needs tuning**: <15% savings

### Latency Overhead
- **Great**: <15% overhead
- **Good**: 15-30% overhead
- **Needs tuning**: >30% overhead

### Learning Velocity
- **Great**: 3x+ faster than baseline
- **Good**: 2-3x faster
- **Needs tuning**: <2x faster

## Recommendations Based on Results

### If success rate is low:
1. Check success criteria aren't too strict
2. Increase iterations (try 5 instead of 3)
3. Review judge verdicts (are failures being marked correctly?)
4. Check memory quality (`npx agentic-flow reasoningbank list`)

### If token savings are low:
1. Lower k parameter (retrieve fewer memories)
2. Improve distillation (make memories more concise)
3. Check if memories are being reused (usage ratio)

### If latency is high:
1. Enable WAL mode on database
2. Add database indexes
3. Increase consolidation threshold
4. Use hash embeddings instead of neural

### If learning is slow:
1. Check memories are being created (`npx agentic-flow reasoningbank status`)
2. Verify retrieval is working (`npx agentic-flow reasoningbank list --query="test"`)
3. Review distillation quality (are memories actionable?)
4. Increase diversity of training tasks

## Next Steps

1. **Analyze your results**: Compare to expected patterns above
2. **Tune parameters**: Adjust config.json based on observations
3. **Add scenarios**: Create domain-specific benchmarks
4. **Optimize performance**: Profile database and retrieval
5. **Scale up**: Run with more iterations and tasks

---

**Note**: This is a template. Run `./run-benchmark.sh` to generate actual results.
