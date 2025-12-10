# Agent Booster Known Issues

## Critical: Insert-Only Behavior

### Issue

Agent Booster uses `insert_after` strategy instead of `exact_replace` for most edits, creating duplicates:

**Input:**
```javascript
var x = 1;
```

**Expected (replace):**
```javascript
const x = 1;
```

**Actual (insert_after):**
```javascript
var x = 1;

const x = 1;  // Duplicate declaration!
```

### Root Cause

The WASM pattern matching engine prefers `insert_after` (safer) over `exact_replace` (risky). Confidence scores:

- `insert_after`: 57% confidence (chosen)
- `exact_replace`: Would require 90%+ confidence
- `fuzzy_replace`: 78% confidence (sometimes works)

### Impact

| Operation | Works? | Strategy | Result |
|-----------|--------|----------|--------|
| Add import | ✅ Yes | insert_after | Appends correctly |
| Add logging | ✅ Yes | insert_after | Inserts correctly |
| var → const | ❌ No | insert_after | Creates duplicate |
| Add types | ⚠️ Partial | insert_after | Duplicates function |
| Error handling | ✅ Yes | exact_replace | Replaces (90% conf) |
| async/await | ✅ Yes | fuzzy_replace | Works (78% conf) |

### Workarounds

#### 1. Use for Insertions Only

**Good:**
```bash
# Add import
echo '{"code":"","edit":"import React from '\''react'\''"}' | npx agent-booster apply

# Add logging
echo '{"code":"function foo(){}","edit":"function foo(){console.log('\''called'\'')}"}' | npx agent-booster apply
```

**Bad:**
```bash
# Replace var → const (creates duplicate!)
echo '{"code":"var x=1","edit":"const x=1"}' | npx agent-booster apply
```

#### 2. Use High-Confidence Edits

Exact replacements work when confidence ≥90%:

```bash
# Error handling (exact_replace, 90% conf)
echo '{"code":"function divide(a,b){return a/b;}","edit":"function divide(a,b){if(b===0)throw new Error();return a/b;}"}' | npx agent-booster apply
# ✅ Works: Replaces function
```

#### 3. Fallback to LLM

For replacements, use LLM instead:

```bash
# Use agentic-flow agent (not Agent Booster)
npx agentic-flow --agent coder --task "convert var to const in file.js"
# ✅ Works: LLM understands replacement
```

### MCP Integration Impact

The MCP tools (`agent_booster_edit_file`, etc.) have this limitation:

```javascript
// User request in Claude Desktop
"Use agent_booster_edit_file to convert var to const in utils.js"

// Agent Booster result (insert_after, 57% conf)
{
  "success": false,  // Confidence < 70%
  "confidence": 0.571,
  "fallback_required": true,
  "suggestion": "Use agentic_flow_agent with coder"
}

// Claude automatically falls back to LLM ✅
```

**Result:** Users get correct behavior (LLM replacement) but not the speed benefit.

### Performance Reality

| Scenario | Agent Booster | LLM | Winner |
|----------|---------------|-----|--------|
| **Insertions** (add import) | 10ms ✅ | 2,000ms | Agent Booster |
| **Replacements** (var→const) | 10ms but duplicates ❌ | 2,000ms but correct ✅ | LLM |
| **Complex** (refactor) | Fails ❌ | 2,000ms ✅ | LLM |

**Effective speedup:** Only for insertions (~10% of coding tasks)

## Other Known Issues

### 2. Vague Instructions

Agent Booster requires exact code, not natural language:

```bash
# ❌ Fails
echo '{"code":"var x=1","edit":"convert to const"}' | npx agent-booster apply

# ✅ Works
echo '{"code":"var x=1","edit":"const x=1"}' | npx agent-booster apply
```

**Solution:** Use LLM for vague tasks.

### 3. CLI vs MCP Confusion

- CLI `--agent-booster` flag does NOTHING (cosmetic only)
- Agent Booster only works via MCP server
- CLI agents use standard LLM (not Agent Booster)

**Solution:** Remove misleading CLI flag, document MCP-only.

### 4. Confidence Threshold Too Low?

Current threshold: 70% for auto-apply

Issues:
- 57% edits create duplicates (insert_after)
- 78% edits sometimes work (fuzzy_replace)
- 90% edits work reliably (exact_replace)

**Recommendation:** Raise threshold to 80% or 90%?

### 5. Network Dependency

MCP tools use `npx --yes agent-booster`, which downloads on first use:

- 30s timeout for download
- Fails in air-gapped environments
- User doesn't know it's downloading

**Solution:** Add optional dependency or pre-download in installation.

## Recommendations

### For Users

1. **Use Agent Booster for insertions only**
   - Add imports
   - Add logging
   - Insert new code

2. **Use LLM for replacements**
   - var → const
   - Add types
   - Refactoring

3. **Let MCP tools decide**
   - Automatic fallback to LLM works well
   - You get correct result, just slower

### For Developers

1. **Fix WASM engine strategy selection**
   - Prefer `exact_replace` for simple substitutions
   - Only use `insert_after` when replacement is ambiguous
   - Improve confidence scoring

2. **Raise confidence threshold**
   - Current: 70% (allows duplicates)
   - Recommended: 80-90% (only high-confidence)

3. **Remove CLI flag**
   - `--agent-booster` is misleading
   - Only works via MCP server
   - CLI agents use LLM

4. **Update documentation**
   - Clarify insert-only behavior
   - Set realistic expectations
   - Show correct use cases

## Test Results

### CLI Validation (v0.1.2)

✅ **Stability:** No JSON parsing errors
✅ **Speed:** 10-13ms (matches claims)
⚠️ **Functionality:** Insert-only (not documented)

### MCP Integration (agentic-flow@1.4.2)

✅ **Architecture:** Uses npx (works remotely)
✅ **Fallback:** LLM fallback for low confidence
⚠️ **Performance:** Only fast for insertions

### Actual Speedup

- **Insertions:** 10ms vs 2,000ms = **200x faster** ✅
- **Replacements:** Fallback to LLM = **0x faster** ❌
- **Complex:** Not supported = **N/A** ❌

**Effective benefit:** ~10% of coding tasks (insertions only)

## Conclusion

Agent Booster works correctly for its **intended use case** (insertions), but is **not a general-purpose replacement** for LLM code editing.

**Status:** ✅ Working as designed (but design is limited)

**Recommendation:** Document limitations clearly, raise confidence threshold, focus on insertion use cases.

---

**Last Updated:** 2025-10-08
**Version:** agent-booster@0.1.2, agentic-flow@1.4.2
