# Agent Booster

High-performance prompt optimization library with automatic native/WASM binding detection.

## Features

- **Automatic Runtime Detection**: Tries native bindings first, falls back to WASM
- **Zero Configuration**: Works out of the box with optimal performance
- **Cross-Platform**: Supports macOS, Linux, Windows (x64, ARM64)
- **TypeScript Support**: Full type definitions included
- **Batch Processing**: Optimize multiple prompts efficiently
- **Prompt Analysis**: Get insights and optimization suggestions

## Installation

```bash
npm install agent-booster
```

The package will automatically use native bindings if available for your platform, or fall back to WASM for universal compatibility.

### Force WASM-only (optional)

```bash
npm install agent-booster --no-optional
```

## Quick Start

```javascript
const agentBooster = require('agent-booster');

// Simple optimization
async function optimizePrompt() {
  const prompt = 'Your long, verbose prompt here...';
  const optimized = await agentBooster.apply(prompt);
  console.log(optimized);
}

optimizePrompt();
```

## API Reference

### `apply(prompt, options?)`

Applies prompt optimization transformations.

```javascript
const optimized = await agentBooster.apply(prompt, {
  strategy: 'balanced',      // 'aggressive' | 'balanced' | 'conservative'
  maxTokens: 1000,           // Maximum tokens in output
  preserve: ['important'],   // Preserve specific keywords
  targetModel: 'claude-3'    // Target model for optimization
});
```

**Parameters:**
- `prompt` (string): The input prompt to optimize
- `options` (object, optional): Optimization options

**Returns:** Promise<string> - The optimized prompt

### `batchApply(prompts, options?)`

Optimizes multiple prompts in batch mode for better performance.

```javascript
const prompts = [
  'First prompt...',
  'Second prompt...',
  'Third prompt...'
];

const optimized = await agentBooster.batchApply(prompts, {
  strategy: 'balanced'
});

console.log(optimized); // Array of optimized prompts
```

**Parameters:**
- `prompts` (string[]): Array of prompts to optimize
- `options` (object, optional): Optimization options

**Returns:** Promise<string[]> - Array of optimized prompts

### `analyze(prompt)`

Analyzes a prompt and returns optimization metrics.

```javascript
const analysis = await agentBooster.analyze(prompt);

console.log(analysis);
// {
//   originalTokens: 150,
//   optimizedTokens: 95,
//   savings: 55,
//   savingsPercent: 36.67,
//   patterns: ['repetition', 'verbosity'],
//   suggestions: ['Remove redundant phrases', 'Simplify structure'],
//   complexity: 42
// }
```

**Parameters:**
- `prompt` (string): The prompt to analyze

**Returns:** Promise<AnalysisResult> - Analysis results with metrics and suggestions

### `getVersion()`

Gets version information including the runtime type.

```javascript
const info = agentBooster.getVersion();

console.log(info);
// {
//   version: '0.1.0',
//   runtime: 'native',  // or 'wasm'
//   coreVersion: '0.1.0'
// }
```

**Returns:** VersionInfo - Version information object

### `getRuntime()`

Gets the currently loaded runtime type.

```javascript
const runtime = agentBooster.getRuntime();
console.log(runtime); // 'native' or 'wasm'
```

**Returns:** string | null - Runtime type or null if not yet initialized

## Runtime Detection

Agent Booster automatically detects and loads the best available runtime:

1. **Native Bindings** (Fastest): Platform-specific native modules built with Rust
   - macOS: arm64, x64
   - Linux: arm64, x64
   - Windows: x64

2. **WASM Fallback** (Universal): WebAssembly for maximum compatibility

The detection happens automatically on first use. You can check which runtime is loaded:

```javascript
const agentBooster = require('agent-booster');

// Initialize explicitly (or wait for first API call)
agentBooster.initialize();

console.log(`Using runtime: ${agentBooster.getRuntime()}`);
```

## Optimization Strategies

### Aggressive

Maximum token reduction, may sacrifice some nuance:

```javascript
const optimized = await agentBooster.apply(prompt, {
  strategy: 'aggressive'
});
```

### Balanced (Default)

Good balance between token reduction and prompt quality:

```javascript
const optimized = await agentBooster.apply(prompt, {
  strategy: 'balanced'
});
```

### Conservative

Minimal changes, preserves most original structure:

```javascript
const optimized = await agentBooster.apply(prompt, {
  strategy: 'conservative'
});
```

## TypeScript Support

Full TypeScript definitions are included:

```typescript
import * as agentBooster from 'agent-booster';
import type { OptimizationOptions, AnalysisResult } from 'agent-booster';

const options: OptimizationOptions = {
  strategy: 'balanced',
  maxTokens: 1000
};

const optimized: string = await agentBooster.apply(prompt, options);
const analysis: AnalysisResult = await agentBooster.analyze(prompt);
```

## Performance

Native bindings provide significant performance improvements:

- **Native**: ~100-500x faster than pure JavaScript
- **WASM**: ~10-50x faster than pure JavaScript
- **Batch Processing**: Additional 2-5x speedup for multiple prompts

## Error Handling

```javascript
try {
  const optimized = await agentBooster.apply(prompt);
} catch (error) {
  if (error.message.includes('No runtime available')) {
    console.error('Please install WASM package: npm install @agent-booster/wasm');
  } else {
    console.error('Optimization failed:', error.message);
  }
}
```

## CLI Tool

For command-line usage, install the CLI package:

```bash
npm install -g agent-booster-cli

# Optimize a prompt
agent-booster apply "Your prompt here"

# Batch process files
agent-booster batch prompts.txt -o optimized.txt
```

See [agent-booster-cli](https://www.npmjs.com/package/agent-booster-cli) for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md).

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Support

- GitHub Issues: [Report a bug](https://github.com/yourusername/agentic-flow/issues)
- Documentation: [Full docs](https://github.com/yourusername/agentic-flow/tree/main/agent-booster)

## Related Packages

- `agent-booster-cli` - Command-line interface
- `@agent-booster/wasm` - WASM bindings
- `@agent-booster/native-*` - Platform-specific native bindings
