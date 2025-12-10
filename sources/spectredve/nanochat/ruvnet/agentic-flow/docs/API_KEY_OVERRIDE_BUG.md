# API Key Override Bug - ANTHROPIC_API_KEY Overrides Provider Arguments

## Summary
When `ANTHROPIC_API_KEY` is exported in the environment, agentic-flow automatically uses it even when `--provider openrouter` (or other providers) is specified via CLI arguments, causing authentication failures.

## Root Cause Analysis

### Issue 1: CLI Arguments Don't Set PROVIDER Environment Variable
**Location:** `agentic-flow/src/index.ts:203-207`

The CLI parsing logic stores provider arguments in `options.provider` but **never sets** the `PROVIDER` environment variable that the agent code relies on:

```typescript
if (options.mode === 'agent') {
  const task = options.task || process.env.TASK || '';
  const agent = options.agent || process.env.AGENT || '';
  const model = options.model || process.env.MODEL;
  await runAgentMode(agent, task, options.stream || false, model);
}
```

Missing: `process.env.PROVIDER = options.provider || process.env.PROVIDER;`

### Issue 2: Fallback to ANTHROPIC_API_KEY
**Location:** `agentic-flow/src/agents/claudeAgent.ts:34, 41, 48`

The `getModelForProvider()` function falls back to `ANTHROPIC_API_KEY` for all providers:

```typescript
case 'gemini':
  return {
    model: process.env.COMPLETION_MODEL || 'gemini-2.0-flash-exp',
    apiKey: process.env.GOOGLE_GEMINI_API_KEY || process.env.ANTHROPIC_API_KEY || '',
    baseURL: process.env.PROXY_URL || undefined
  };

case 'openrouter':
  return {
    model: process.env.COMPLETION_MODEL || 'deepseek/deepseek-chat',
    apiKey: process.env.OPENROUTER_API_KEY || process.env.ANTHROPIC_API_KEY || '',
    baseURL: process.env.PROXY_URL || undefined
  };
```

### Issue 3: Provider Detection Relies on Environment Variables
**Location:** `agentic-flow/src/agents/claudeAgent.ts:8-23` and `agentic-flow/src/agents/directApiAgent.ts:45-57`

```typescript
function getCurrentProvider(): string {
  // Determine provider from environment
  if (process.env.PROVIDER === 'gemini' || process.env.USE_GEMINI === 'true') {
    return 'gemini';
  }
  if (process.env.PROVIDER === 'openrouter' || process.env.USE_OPENROUTER === 'true') {
    return 'openrouter';
  }
  return 'anthropic'; // Default
}
```

Since `process.env.PROVIDER` is never set from CLI args, it defaults to `'anthropic'` even when `--provider openrouter` is passed.

## Reproduction Steps

```bash
# Export Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Try to use OpenRouter with its API key
npx agentic-flow --agent coder \
  --task "Write hello world" \
  --provider openrouter \
  --openrouter-key sk-or-your-key-here

# EXPECTED: Uses OpenRouter API
# ACTUAL: Uses ANTHROPIC_API_KEY, authentication fails
```

## Expected Behavior
1. CLI `--provider openrouter` argument should set `process.env.PROVIDER = 'openrouter'`
2. CLI `--openrouter-key` argument should set `process.env.OPENROUTER_API_KEY`
3. Provider-specific API keys should be required, no fallback to `ANTHROPIC_API_KEY`
4. Clear error message if provider-specific API key is missing

## Current Behavior
1. CLI arguments are parsed but not propagated to environment variables
2. `getCurrentProvider()` returns `'anthropic'` because `process.env.PROVIDER` is undefined
3. `getModelForProvider()` falls back to `ANTHROPIC_API_KEY` even for other providers
4. API calls authenticate against wrong provider, causing 401 errors

## Affected Files
1. **`agentic-flow/src/index.ts`** - Main entry point doesn't set environment variables from CLI args
2. **`agentic-flow/src/agents/claudeAgent.ts`** - Falls back to `ANTHROPIC_API_KEY` for all providers
3. **`agentic-flow/src/agents/directApiAgent.ts`** - Same fallback issue
4. **`agentic-flow/src/utils/cli.ts`** - CLI argument parsing works correctly, but args aren't used

## Proposed Fix

### Fix 1: Propagate CLI Arguments to Environment Variables
**File:** `agentic-flow/src/index.ts`

```typescript
// After parsing CLI options, before running agents
if (options.provider) {
  process.env.PROVIDER = options.provider;
}
if (options.anthropicApiKey) {
  process.env.ANTHROPIC_API_KEY = options.anthropicApiKey;
}
if (options.openrouterApiKey) {
  process.env.OPENROUTER_API_KEY = options.openrouterApiKey;
}
if (options.model) {
  process.env.COMPLETION_MODEL = options.model;
}
```

### Fix 2: Remove ANTHROPIC_API_KEY Fallback for Non-Anthropic Providers
**File:** `agentic-flow/src/agents/claudeAgent.ts`

```typescript
case 'openrouter':
  const openrouterKey = process.env.OPENROUTER_API_KEY;
  if (!openrouterKey) {
    throw new Error('OPENROUTER_API_KEY is required for OpenRouter provider');
  }
  return {
    model: process.env.COMPLETION_MODEL || 'deepseek/deepseek-chat',
    apiKey: openrouterKey,
    baseURL: process.env.PROXY_URL || undefined
  };
```

### Fix 3: Accept Provider as Function Parameter
**Alternative approach:** Pass provider from CLI options directly to agent functions instead of relying on environment variables.

## Impact
- **Severity:** High - Silent authentication failures
- **Users Affected:** Anyone using `--provider` CLI arguments with exported `ANTHROPIC_API_KEY`
- **Workaround:** Unset `ANTHROPIC_API_KEY` or use environment variable `PROVIDER=openrouter` instead of CLI args

## Testing
```bash
# Test 1: OpenRouter with explicit API key
export ANTHROPIC_API_KEY=sk-ant-existing
npx agentic-flow --agent coder --task "test" \
  --provider openrouter \
  --openrouter-key sk-or-v1-test
# Should use OpenRouter, not Anthropic

# Test 2: Missing provider API key should error
npx agentic-flow --agent coder --task "test" \
  --provider openrouter
# Should error: "OPENROUTER_API_KEY is required"

# Test 3: Anthropic provider still works
export ANTHROPIC_API_KEY=sk-ant-real
npx agentic-flow --agent coder --task "test"
# Should use Anthropic (default provider)
```

## Related Issues
- CLI argument parsing in `parseArgs()` works correctly
- Environment variable precedence is confusing
- No validation that provider-specific API keys are set
