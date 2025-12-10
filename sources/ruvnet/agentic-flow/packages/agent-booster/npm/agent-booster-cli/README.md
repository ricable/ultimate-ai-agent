# Agent Booster CLI

Command-line interface for high-performance prompt optimization.

## Installation

```bash
npm install -g agent-booster-cli
```

Or use with npx (no installation required):

```bash
npx agent-booster-cli apply "Your prompt here"
```

## Commands

### `apply` - Optimize a single prompt

Optimize a single prompt with various options:

```bash
agent-booster apply "Your long verbose prompt here"
```

#### Options

- `-s, --strategy <type>` - Optimization strategy: `aggressive`, `balanced`, `conservative` (default: `balanced`)
- `-m, --max-tokens <number>` - Maximum tokens in output
- `-t, --target-model <model>` - Target model for optimization
- `-p, --preserve <keywords...>` - Keywords to preserve
- `-a, --analyze` - Show analysis before optimization
- `-o, --output <file>` - Write output to file
- `--json` - Output in JSON format

#### Examples

```bash
# Basic optimization
agent-booster apply "Create a function that processes user input"

# Aggressive optimization with analysis
agent-booster apply "Your prompt" -s aggressive -a

# Preserve specific keywords
agent-booster apply "Your prompt" -p important critical

# Save to file
agent-booster apply "Your prompt" -o optimized.txt

# JSON output
agent-booster apply "Your prompt" --json
```

### `batch` - Optimize multiple prompts

Process multiple prompts from a file efficiently:

```bash
agent-booster batch prompts.txt -o optimized.txt
```

#### Input Formats

**Plain text** (one prompt per line):
```
First prompt here
Second prompt here
Third prompt here
```

**JSONL** (JSON Lines):
```json
{"prompt": "First prompt here"}
{"prompt": "Second prompt here"}
{"prompt": "Third prompt here"}
```

#### Options

- `-s, --strategy <type>` - Optimization strategy (default: `balanced`)
- `-m, --max-tokens <number>` - Maximum tokens in output
- `-t, --target-model <model>` - Target model for optimization
- `-o, --output <file>` - Output file (default: stdout)
- `--json` - Input/output in JSONL format
- `--parallel <number>` - Number of parallel operations (default: 4)
- `--progress` - Show progress bar

#### Examples

```bash
# Process plain text file
agent-booster batch prompts.txt

# Process with progress bar
agent-booster batch prompts.txt --progress

# JSONL input/output
agent-booster batch input.jsonl --json -o output.jsonl

# Aggressive optimization with more parallelism
agent-booster batch prompts.txt -s aggressive --parallel 8

# Save to file with progress
agent-booster batch prompts.txt -o optimized.txt --progress
```

### `analyze` - Analyze a prompt

Get insights and optimization suggestions without modifying:

```bash
agent-booster analyze "Your prompt here"
```

#### Options

- `--json` - Output in JSON format

#### Example Output

```
📊 Prompt Analysis

──────────────────────────────────────────────────
Original Tokens: 150
Estimated Optimized: 95
Potential Savings: 55 tokens (36.7%)
Complexity Score: 42/100

🔍 Detected Patterns:
  • repetition
  • verbosity
  • redundant phrases

💡 Suggestions:
  • Remove redundant phrases
  • Simplify sentence structure
  • Consolidate repeated concepts
```

### `version` - Show version information

Display version and runtime details:

```bash
agent-booster version
```

#### Example Output

```
🚀 Agent Booster

──────────────────────────────────────────────────
CLI Version: 0.1.0
SDK Version: 0.1.0
Core Version: 0.1.0
Runtime: native
```

## Optimization Strategies

### Aggressive

Maximum token reduction, may sacrifice some nuance. Best for:
- Cost-sensitive applications
- Simple prompts
- When maximum compression is needed

```bash
agent-booster apply "Your prompt" -s aggressive
```

### Balanced (Default)

Good balance between token reduction and prompt quality. Best for:
- General use cases
- Most production applications
- When unsure which strategy to use

```bash
agent-booster apply "Your prompt" -s balanced
```

### Conservative

Minimal changes, preserves most original structure. Best for:
- Complex prompts with specific structure
- When prompt quality is critical
- Sensitive applications

```bash
agent-booster apply "Your prompt" -s conservative
```

## Performance

The CLI automatically uses the fastest available runtime:

- **Native bindings** (~100-500x faster than pure JS)
- **WASM fallback** (~10-50x faster than pure JS)

For batch processing, use `--parallel` to control concurrency:

```bash
# Process 8 prompts in parallel
agent-booster batch prompts.txt --parallel 8
```

## Workflows

### Optimizing a Project's Prompts

```bash
# Create a file with all your prompts
cat > prompts.txt <<EOF
Prompt 1 here
Prompt 2 here
Prompt 3 here
EOF

# Optimize them all
agent-booster batch prompts.txt --progress -o optimized.txt

# Review the results
cat optimized.txt
```

### Pre-commit Hook

Automatically optimize prompts before committing:

```bash
#!/bin/bash
# .git/hooks/pre-commit

if [ -f prompts.txt ]; then
  echo "Optimizing prompts..."
  agent-booster batch prompts.txt -o prompts.optimized.txt --progress
  git add prompts.optimized.txt
fi
```

### CI/CD Integration

```yaml
# .github/workflows/optimize-prompts.yml
name: Optimize Prompts

on: [push]

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: npm install -g agent-booster-cli
      - run: agent-booster batch prompts.txt -o optimized.txt
      - uses: actions/upload-artifact@v2
        with:
          name: optimized-prompts
          path: optimized.txt
```

## Error Handling

### Runtime Not Available

```
Error: No runtime available. Please ensure either native bindings or WASM package is installed.

💡 Tip: Install WASM package: npm install @agent-booster/wasm
```

**Solution**: Install the WASM package as a fallback:

```bash
npm install -g @agent-booster/wasm
```

### Invalid Input File

```
Error: Input file not found: prompts.txt
```

**Solution**: Check the file path and ensure the file exists.

### JSON Parse Error

```
Error: Invalid JSON on line 3: Unexpected token
```

**Solution**: Validate your JSONL file format. Each line must be valid JSON.

## Environment Variables

### `AGENT_BOOSTER_RUNTIME`

Force a specific runtime (for testing):

```bash
# Force WASM runtime
export AGENT_BOOSTER_RUNTIME=wasm
agent-booster apply "Your prompt"

# Force native runtime (will error if not available)
export AGENT_BOOSTER_RUNTIME=native
agent-booster apply "Your prompt"
```

### `AGENT_BOOSTER_SILENT`

Disable runtime detection messages:

```bash
export AGENT_BOOSTER_SILENT=1
agent-booster apply "Your prompt"
```

## Tips

1. **Use batch mode for multiple prompts** - Much faster than running `apply` multiple times
2. **Analyze first** - Use `analyze` to understand potential savings before optimizing
3. **Start with balanced strategy** - Only use aggressive if you need maximum compression
4. **Preserve important keywords** - Use `-p` flag to keep critical terms intact
5. **Output to files** - Easier to review and compare results

## Programmatic Usage

For programmatic usage, use the [agent-booster](https://www.npmjs.com/package/agent-booster) SDK directly:

```javascript
const agentBooster = require('agent-booster');

const optimized = await agentBooster.apply(prompt, {
  strategy: 'balanced'
});
```

## Support

- GitHub Issues: [Report a bug](https://github.com/yourusername/agentic-flow/issues)
- Documentation: [Full docs](https://github.com/yourusername/agentic-flow/tree/main/agent-booster)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
