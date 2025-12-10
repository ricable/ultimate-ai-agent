# Agent Booster Usage Guide

## What is Agent Booster?

Agent Booster is a **pattern matching engine** for code edits, NOT an AI. It uses Rust/WASM to perform fast local code transformations without making LLM API calls.

### Performance Benefits

- **Speed**: 85ms average latency (vs 13s with LLM)
- **Cost**: $0.00 per edit (vs ~$0.001 with API)
- **Privacy**: Code stays local, never sent to cloud APIs

### Limitations

Agent Booster is a **specialized tool** for mechanical code changes. It cannot:
- Understand high-level instructions ("make it better", "add feature")
- Reason about code behavior ("fix the bug")
- Make architectural decisions
- Write new functions from scratch

## ✅ Correct Usage (What Works)

Agent Booster requires **exact code snippets** to replace, not vague instructions.

### Example 1: Variable Declaration Conversion

```bash
# ❌ WRONG: Vague instruction
echo '{"code":"var x = 1;","edit":"convert to const"}' | \
  npx agent-booster apply --language javascript
# Result: Low confidence or failure

# ✅ CORRECT: Exact code replacement
echo '{"code":"var x = 1;","edit":"const x = 1;"}' | \
  npx agent-booster apply --language javascript
# Result: {"success":true,"confidence":0.57,"strategy":"insert_after"}
```

### Example 2: Type Annotations

```bash
# ❌ WRONG: High-level instruction
echo '{"code":"function add(a, b) { return a + b; }","edit":"add TypeScript types"}' | \
  npx agent-booster apply --language typescript
# Result: Low confidence or failure

# ✅ CORRECT: Exact typed function
echo '{"code":"function add(a, b) { return a + b; }","edit":"function add(a: number, b: number): number { return a + b; }"}' | \
  npx agent-booster apply --language typescript
# Result: {"success":true,"confidence":0.57,"strategy":"insert_after"}
```

### Example 3: Error Handling

```bash
# ❌ WRONG: Abstract instruction
echo '{"code":"function divide(a, b) { return a / b; }","edit":"fix division by zero"}' | \
  npx agent-booster apply --language javascript
# Result: Low confidence or failure

# ✅ CORRECT: Exact error handling code
echo '{"code":"function divide(a, b) { return a / b; }","edit":"function divide(a, b) { if (b === 0) throw new Error(\"Division by zero\"); return a / b; }"}' | \
  npx agent-booster apply --language javascript
# Result: {"success":true,"confidence":0.90,"strategy":"exact_replace"}
```

### Example 4: Async/Await

```bash
# ❌ WRONG: Refactoring instruction
echo '{"code":"function fetchData() { return fetch(\"/api\"); }","edit":"refactor to async/await"}' | \
  npx agent-booster apply --language javascript
# Result: Low confidence or failure

# ✅ CORRECT: Exact async function
echo '{"code":"function fetchData() { return fetch(\"/api\"); }","edit":"async function fetchData() { return await fetch(\"/api\"); }"}' | \
  npx agent-booster apply --language javascript
# Result: {"success":true,"confidence":0.78,"strategy":"fuzzy_replace"}
```

## ❌ Incorrect Usage (What Doesn't Work)

These patterns will produce low confidence or fail outright:

### Vague Instructions
```bash
# These will all fail with low confidence:
"make it better"
"improve this code"
"optimize performance"
"add best practices"
"refactor"
```

### High-Level Goals
```bash
# These require reasoning, not pattern matching:
"add authentication"
"implement caching"
"fix the bug"
"add error handling" (without exact code)
"convert to async" (without exact code)
```

### Architectural Changes
```bash
# These are too complex for pattern matching:
"convert to class-based component"
"add dependency injection"
"implement factory pattern"
"split into microservices"
```

## Understanding Confidence Scores

Agent Booster returns a confidence score (0-1) indicating match quality:

- **≥0.70**: High confidence - exact or fuzzy match found
- **0.50-0.69**: Medium confidence - partial match, review output
- **<0.50**: Low confidence - use LLM instead

### Strategies Used

- `exact_replace`: Perfect match, direct replacement (best)
- `fuzzy_replace`: Similar code found, replaced with fuzzy matching (good)
- `insert_after`: Appended after original code (review needed)
- `insert_before`: Prepended before original code (review needed)
- `append`: Added to end of file (review needed)

## Best Practices

### 1. Use for Mechanical Changes Only

Agent Booster excels at:
- Adding type annotations (when you provide exact types)
- Converting syntax (when you provide exact new syntax)
- Adding guards/checks (when you provide exact check code)
- Formatting changes (when you provide exact format)

### 2. Always Provide Exact Code

```bash
# Bad: "add null check"
# Good: "if (value === null) throw new Error('Value cannot be null');"

# Bad: "convert to ES6"
# Good: "const myFunc = () => { ... }" (exact ES6 syntax)
```

### 3. Check Confidence Scores

```javascript
const result = await booster.apply({ code, edit, language });

if (result.confidence >= 0.7) {
  // High confidence - use Agent Booster result
  fs.writeFileSync(file, result.output);
} else {
  // Low confidence - fallback to LLM
  const llmResult = await agent.execute({
    agent: 'coder',
    task: `Apply edit to ${file}: ${edit}`
  });
}
```

### 4. Use LLM for Complex Tasks

For anything requiring reasoning, use agentic-flow instead:

```bash
# Instead of Agent Booster for vague tasks:
npx agentic-flow agent coder "convert var to const in src/utils.js"

# Agent Booster is better for exact replacements:
echo '{"code":"var x=1;","edit":"const x=1;"}' | npx agent-booster apply
```

## MCP Integration (Claude Desktop/Cursor)

The agentic-flow MCP server includes 3 Agent Booster tools with automatic LLM fallback:

### Tool 1: `agent_booster_edit_file`

Apply precise code edit to a file:

```json
{
  "target_filepath": "src/utils.js",
  "instructions": "Add null check to getValue function",
  "code_edit": "if (value === null) throw new Error('Value cannot be null');",
  "language": "javascript"
}
```

If confidence < 70%, tool suggests LLM fallback automatically.

### Tool 2: `agent_booster_batch_edit`

Edit multiple files in one operation:

```json
{
  "edits": [
    {
      "filepath": "src/utils.js",
      "code_edit": "const x = 1;",
      "language": "javascript"
    },
    {
      "filepath": "src/types.ts",
      "code_edit": "type User = { id: number };",
      "language": "typescript"
    }
  ]
}
```

### Tool 3: `agent_booster_parse_markdown`

Extract code blocks from AI responses:

```json
{
  "markdown_response": "Here's the code:\n```javascript\nconst x = 1;\n```"
}
```

## Validation Tests

Run comprehensive validation:

```bash
cd agent-booster
node validation/test-published-package.js
```

Expected output:
```
✅ Correct Usage Tests: 4/4 passed
✅ Incorrect Usage Tests: 5/5 correctly rejected
🎯 Overall: 9/9 tests passed
```

## When to Use Agent Booster vs LLM

| Task | Use Agent Booster | Use LLM (agentic-flow) |
|------|-------------------|------------------------|
| Exact code replacement | ✅ Yes | ❌ Overkill |
| Var → const (with exact code) | ✅ Yes | ❌ Overkill |
| Add type annotations (exact) | ✅ Yes | ❌ Overkill |
| Vague instruction ("improve") | ❌ No | ✅ Yes |
| New feature | ❌ No | ✅ Yes |
| Bug fix (requires reasoning) | ❌ No | ✅ Yes |
| Architectural refactor | ❌ No | ✅ Yes |
| Code review | ❌ No | ✅ Yes |

## Performance Benchmarks

Real-world comparison with OpenRouter LLM:

```
Task: Apply 3 code edits (var→const, types, error handling)

OpenRouter (meta-llama/llama-3.1-8b-instruct):
- Latency: 6,738ms average
- Cost: $0.003 for 3 edits
- Success: 4/5 tests passed

Agent Booster (WASM):
- Latency: 85ms average (79x faster)
- Cost: $0.000 (100% savings)
- Success: 4/4 exact replacements passed
```

## Troubleshooting

### "Low confidence" errors

**Problem**: Agent Booster returns confidence < 50%

**Solution**:
1. Check if you're providing exact code (not instructions)
2. If instruction is vague, use LLM instead
3. Try providing more context in the exact code

### "JSON parsing error"

**Problem**: CLI fails to parse input

**Solution**:
1. Ensure JSON is valid: `echo '{"code":"...","edit":"..."}' | jq`
2. Escape special characters in shell: `'"` becomes `'\''`
3. Use file input instead: `cat input.json | npx agent-booster apply`

### "Command not found"

**Problem**: `npx agent-booster` not found

**Solution**:
```bash
npm install -g agent-booster
# or use npx explicitly:
npx agent-booster@latest apply --language javascript
```

## Summary

**Agent Booster is a specialized tool** for mechanical code transformations:

✅ **Use it when you have exact code to replace** (152x faster, $0 cost)

❌ **Don't use it for vague instructions** (use LLM instead)

💡 **Best ROI**: Combine Agent Booster (for exact edits) with LLM (for reasoning) using the MCP tools' automatic fallback mechanism.
