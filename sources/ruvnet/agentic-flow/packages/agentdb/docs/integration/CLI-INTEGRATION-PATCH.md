# CLI Integration Patch for Attention Commands

## Files to Update

### 1. `/workspaces/agentic-flow/packages/agentdb/src/cli/agentdb-cli.ts`

Add the following import at the top (around line 27, after other command imports):

```typescript
import { attentionCommand } from './commands/attention.js';
```

Add the following in the `main()` function (around line 1184, after stats command):

```typescript
  // Handle attention commands
  if (command === 'attention') {
    // Use commander to handle the attention subcommands
    const { program } = await import('commander');
    const attentionProgram = new program.Command();
    attentionProgram.addCommand(attentionCommand);

    // Parse and execute
    await attentionProgram.parseAsync(['node', 'agentdb', ...args], { from: 'user' });
    return;
  }
```

Add to the help text (around line 2644, before ENVIRONMENT section):

```typescript
${colors.bright}ATTENTION MECHANISM COMMANDS:${colors.reset}
  agentdb attention init [--mechanism <type>] [--force]
    Initialize attention configuration
    Options:
      --mechanism <type>  Attention mechanism (flash, hyperbolic, sparse, linear, performer)
      --force             Force overwrite existing configuration
      --json              Output as JSON
    Example: agentdb attention init --mechanism flash

  agentdb attention compute --mechanism <type> --query <text> --keys-file <path>
    Compute attention for query-key-value triplets
    Options:
      --mechanism <type>  Attention mechanism (flash, hyperbolic, sparse, linear, performer)
      --query <text>      Query text or vector
      --keys-file <path>  Path to keys JSON file
      --values-file <path> Path to values JSON file (optional)
      --heads <n>         Number of attention heads (default: 8)
      --dimension <n>     Attention dimension (default: 384)
      --output <path>     Output file path
      --json              Output as JSON
    Example: agentdb attention compute --mechanism flash --query "search query" --keys-file keys.json

  agentdb attention benchmark [--mechanism <type>] [--all] [--iterations <n>]
    Benchmark attention mechanism performance
    Options:
      --mechanism <type>  Specific mechanism to benchmark
      --all               Benchmark all mechanisms
      --iterations <n>    Number of iterations (default: 100)
      --output <path>     Output file path for results
      --json              Output as JSON
      --verbose           Verbose output
    Example: agentdb attention benchmark --all --iterations 100 --output results.json

  agentdb attention optimize --mechanism <type> [--curvature <n>] [--sparsity <n>]
    Optimize attention mechanism parameters
    Options:
      --mechanism <type>  Attention mechanism (hyperbolic, sparse)
      --curvature <n>     Hyperbolic curvature (default: -1.0)
      --sparsity <n>      Sparsity ratio 0-1 (default: 0.9)
      --output <path>     Output file path for optimized config
      --json              Output as JSON
    Example: agentdb attention optimize --mechanism hyperbolic --curvature -1.0

```

Add to the EXAMPLES section (around line 2690):

```typescript
  # Attention Mechanisms: Configure and benchmark
  agentdb attention init --mechanism flash
  agentdb attention compute --mechanism flash --query "search query" --keys-file keys.json
  agentdb attention benchmark --all --iterations 100 --output benchmark.json
  agentdb attention optimize --mechanism hyperbolic --curvature -1.0 --output optimized.json

```

## Testing the Integration

After making these changes:

1. Build the package:
```bash
cd /workspaces/agentic-flow/packages/agentdb
npm run build
```

2. Test the commands:
```bash
# Initialize attention configuration
npx agentdb attention init --mechanism flash

# Run benchmark
npx agentdb attention benchmark --all --iterations 50

# Get help
npx agentdb attention --help
npx agentdb attention init --help
npx agentdb attention compute --help
```

## Notes

- The attention command uses Commander.js just like other CLI commands
- All attention subcommands support `--help` and `--json` flags
- Configuration is stored in `.agentdb/attention-config.json`
- Benchmark results can be saved to JSON for further analysis
