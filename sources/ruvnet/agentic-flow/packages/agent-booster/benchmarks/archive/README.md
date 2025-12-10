# Agent Booster Benchmarks

Performance benchmarks comparing Morph LLM against Agent Booster for code transformation tasks.

---

## Quick Start

### Run Morph LLM Benchmark
```bash
cd /workspaces/agentic-flow/agent-booster/benchmarks
node morph-benchmark.js
```

### Analyze Results
```bash
node analyze-results.js
```

---

## Directory Structure

```
benchmarks/
├── datasets/
│   └── small-test-dataset.json       # 12 test cases
├── results/
│   └── morph-baseline-results.json   # Morph LLM results
├── morph-benchmark.js                 # Morph LLM benchmark script
├── analyze-results.js                 # Results analysis script
├── RESULTS.md                         # Detailed benchmark report
├── SUMMARY.md                         # Executive summary
├── comparison-template.md             # Comparison template (awaiting Agent Booster)
└── README.md                          # This file
```

---

## Test Dataset

12 JavaScript/TypeScript code transformation tasks covering:

- **TypeScript Conversion** (2 tests): Type annotations, class conversion
- **Error Handling** (2 tests): Async error handling, try-catch wrappers
- **Modernization** (3 tests): var→const/let, arrow functions, destructuring
- **Async Conversion** (2 tests): Callbacks→Promises, Promises→async/await
- **Documentation** (1 test): JSDoc comments
- **Validation** (1 test): Input validation
- **Safety** (1 test): Null checks

---

## Morph LLM Results (Completed)

**Performance:**
- Success Rate: 100% (12/12)
- Average Latency: 352ms
- Total Time: 16.2 seconds

**Resource Usage:**
- Applies: 12/500 (2.4%)
- Input Tokens: 963/300,000 (0.32%)
- Output Tokens: 427/100,000 (0.43%)

**Key Insights:**
- Network latency: ~200ms overhead
- Processing time: ~152ms per test
- Bottleneck: 500 applies/month limit

---

## Agent Booster Results (Pending)

**Status:** Awaiting npm-progress completion

**Next Steps:**
1. Install Agent Booster
2. Run same 12 tests
3. Measure latency, accuracy, memory
4. Generate comparison report

**Expected Improvements:**
- 4.6x speed improvement
- Unlimited usage (no monthly cap)
- Zero marginal cost
- Offline capability

---

## Usage Tracking

**Morph LLM Free Plan:**
- Monthly Applies: 500
- Monthly Input Tokens: 300,000
- Monthly Output Tokens: 100,000

**Current Usage:**
- Applies Used: 12 (2.4%)
- Input Tokens: 963 (0.32%)
- Output Tokens: 427 (0.43%)
- Remaining: 488 applies, 299K input, 99.5K output

**API Key:** sk-ifrAh8auI9Gkk2J_Sw9pCR0EGA81zIS5ZqouMesp_AzHKA-M

---

## Memory Storage

**Namespace:** `agent-booster-swarm`

**Keys:**
- `morphllm-usage` - Usage tracking data
- `benchmark-progress` - Benchmark status and next steps

**Retrieve:**
```bash
npx claude-flow@alpha memory retrieve --namespace agent-booster-swarm --key morphllm-usage
npx claude-flow@alpha memory retrieve --namespace agent-booster-swarm --key benchmark-progress
```

---

## Reports

### Summary Report
- **File:** `SUMMARY.md`
- **Content:** Executive summary with key metrics

### Detailed Report
- **File:** `RESULTS.md`
- **Content:** Comprehensive analysis with tables and projections

### Comparison Report
- **File:** `comparison-template.md`
- **Content:** Side-by-side comparison (awaiting Agent Booster results)

---

## API Configuration

**Endpoint:** `https://api.morphllm.com/v1/chat/completions`
**Model:** `morph-v3-fast`
**Authentication:** Bearer token

---

## Contributing

To add new test cases:

1. Edit `datasets/small-test-dataset.json`
2. Add test with format:
```json
{
  "id": "test-XXX",
  "description": "Task description",
  "input": "Original code",
  "expected_output": "Transformed code",
  "category": "category-name"
}
```
3. Run benchmark: `node morph-benchmark.js`
4. Analyze results: `node analyze-results.js`

---

## License

Part of Agent Booster project - see main repository for license.

---

**Last Updated:** October 7, 2025
**Status:** Morph LLM baseline complete ✅, Agent Booster pending ⏳
