# Agent Booster Strategy Selection Fix

## Problem

Agent Booster uses `insert_after` instead of `fuzzy_replace` for simple substitutions like var→const, creating duplicates:

**Input:**
```javascript
var x = 1;
```

**Current Result (v0.1.2):**
```javascript
var x = 1;

const x = 1;  // Duplicate!
```

**Expected Result (v0.2.1):**
```javascript
const x = 1;  // Replaced
```

## Root Cause

In `crates/agent-booster/src/merge.rs`, the `select_strategy()` function had overly conservative thresholds:

```rust
// BEFORE (v0.1.2 - causes duplicates)
fn select_strategy(similarity: f32, threshold: f32) -> MergeStrategy {
    match similarity {
        s if s >= 0.95 => MergeStrategy::ExactReplace,  // 95%+ required!
        s if s >= 0.80 => MergeStrategy::FuzzyReplace,  // 80-95%
        s if s >= 0.60 => MergeStrategy::InsertAfter,   // 60-80%
        s if s >= threshold => MergeStrategy::InsertBefore,
        _ => MergeStrategy::Append,
    }
}
```

**For var→const:**
- Similarity: ~57%
- Strategy: `InsertAfter` (creates duplicate)
- Confidence: 57%

## Solution

Lower thresholds to prefer `FuzzyReplace` for medium-similarity edits:

```rust
// AFTER (v0.2.1 - fixes duplicates)
fn select_strategy(similarity: f32, threshold: f32) -> MergeStrategy {
    match similarity {
        // Prefer replacement strategies over insertion for simple edits
        s if s >= 0.90 => MergeStrategy::ExactReplace,  // Very high similarity
        s if s >= 0.50 => MergeStrategy::FuzzyReplace,  // Medium-high similarity (var→const, add types)
        s if s >= 0.30 => MergeStrategy::InsertAfter,   // Low-medium similarity
        s if s >= threshold => MergeStrategy::InsertBefore,
        _ => MergeStrategy::Append,
    }
}
```

**For var→const (after fix):**
- Similarity: ~57%
- Strategy: `FuzzyReplace` (replaces!)
- Confidence: ~64%

## Test Results

### Local Testing (Before Publishing)

```bash
# Test 1: var → const
echo '{"code":"var x = 1;","edit":"const x = 1;"}' | node dist/cli.js apply --language javascript
# Result: {"strategy":"fuzzy_replace","confidence":0.6386110782623291,"success":true}
# ✅ PASS: Uses fuzzy_replace, no duplicate

# Test 2: Add type annotations
echo '{"code":"function add(a, b) { return a + b; }","edit":"function add(a: number, b: number): number { return a + b; }"}' | node dist/cli.js apply --language typescript
# Result: {"strategy":"fuzzy_replace","confidence":0.6412500143051147,"success":true}
# ✅ PASS: Uses fuzzy_replace, no duplicate

# Test 3: Error handling (should still use exact_replace)
echo '{"code":"function divide(a, b) { return a / b; }","edit":"function divide(a, b) { if (b === 0) throw new Error(\\"Division by zero\\"); return a / b; }"}' | node dist/cli.js apply --language javascript
# Result: {"strategy":"exact_replace","confidence":0.8999999761581421,"success":true}
# ✅ PASS: Uses exact_replace, 90% confidence (unchanged)
```

### Remote Testing (After Publishing)

```bash
# Published package test
cd /tmp && echo '{"code":"var x = 1;","edit":"const x = 1;"}' | npx --yes agent-booster@0.2.1 apply --language javascript
# Result: {"strategy":"fuzzy_replace","confidence":0.6386110782623291,"success":true}
# ✅ PASS: Works remotely with strategy fix
```

## Expected Impact

| Edit Type | Before (0.1.2) | After (0.2.1) | Improvement |
|-----------|----------------|---------------|-------------|
| var → const | ❌ Duplicate (insert_after) | ✅ Replace (fuzzy_replace) | Fixed |
| Add types | ❌ Duplicate (insert_after) | ✅ Replace (fuzzy_replace) | Fixed |
| Error handling | ✅ Replace (exact_replace) | ✅ Replace (exact_replace) | Same |
| Add import | ✅ Insert (insert_after) | ✅ Insert (insert_after) | Same |
| Complex refactor | ❌ Fail (low conf) | ⚠️ Partial (higher conf) | Slight |
| Confidence | 57% | 64% | +7% |

## Success Criteria

✅ **var → const uses `fuzzy_replace`** (not `insert_after`)
✅ **No duplicate code** in output
✅ **Confidence 64%** for simple substitutions (was 57%)
✅ **All existing tests still pass**
✅ **WASM files included** in npm package (469KB)
✅ **Remote validation** confirms fix works

## Build & Publish Process

### 1. Build WASM
```bash
cd /workspaces/agentic-flow/agent-booster
export PATH="/home/codespace/.cargo/bin:$PATH"
cd crates/agent-booster-wasm
wasm-pack build --target nodejs --out-dir ../../wasm
```

### 2. Build TypeScript
```bash
cd /workspaces/agentic-flow/agent-booster
npm run build:js
```

### 3. Validate Locally
```bash
echo '{"code":"var x = 1;","edit":"const x = 1;"}' | node dist/cli.js apply --language javascript
# Should show: strategy: fuzzy_replace, confidence: ~64%
```

### 4. Validate Package
```bash
npm pack
tar -xzf agent-booster-*.tgz
ls -lh package/wasm/*.wasm  # Should show 1.3MB WASM file
```

### 5. Publish
```bash
npm version patch  # 0.2.0 → 0.2.1
npm publish --access public
```

### 6. Validate Remotely
```bash
cd /tmp
echo '{"code":"var x = 1;","edit":"const x = 1;"}' | npx --yes agent-booster@0.2.1 apply --language javascript
```

## Status

- [x] Root cause identified
- [x] Fix implemented in merge.rs:74-82
- [x] WASM rebuilt with fix
- [x] TypeScript rebuilt
- [x] Tested locally (all 3 scenarios pass)
- [x] Package validated (WASM included)
- [x] Published as v0.2.1
- [x] Tested remotely (confirmed working)
- [ ] agentic-flow updated to use @0.2.1

---

**Published:** 2025-10-08
**Version:** 0.2.1
**npm:** https://www.npmjs.com/package/agent-booster
**Repository:** https://github.com/ruvnet/agentic-flow/tree/main/agent-booster
