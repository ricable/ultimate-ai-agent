# SPARC CLI Advanced Features

## Provider Configuration

### Supported Providers

- Anthropic (Claude)
- OpenAI (GPT-4, GPT-3.5)
- OpenRouter
- OpenAI-compatible endpoints

### Environment Variables

```bash
# Anthropic
export ANTHROPIC_API_KEY=your_key

# OpenAI
export OPENAI_API_KEY=your_key

# OpenRouter
export OPENROUTER_API_KEY=your_key

# OpenAI-compatible
export OPENAI_API_KEY=your_key
export OPENAI_API_BASE=your_endpoint
```

## Expert Knowledge System

The expert system provides specialized knowledge for complex analysis:

```bash
sparc -m "Review authentication implementation" --expert-model gpt-4
```

### Expert Configuration

- `--expert-provider`: Provider for expert queries
- `--expert-model`: Model for expert knowledge
- Separate API keys for expert system (EXPERT_*)

## Human-in-the-Loop Controls

Enable interactive approval for changes:

```bash
sparc -m "Refactor database schema" --hil
```

Features:
- Review proposed changes
- Approve/reject modifications
- Interactive guidance
- Progress monitoring

## Cowboy Mode

Enable autonomous shell command execution:

```bash
sparc -m "Run test suite" --cowboy-mode
```

Features:
- Skips command approval
- Displays cowboy messages
- Session-level toggle
- Safety constraints

## Advanced Examples

```bash
# Complex refactoring with expert system
sparc -m "Optimize database queries" --expert-model gpt-4 --hil

# Autonomous testing with cowboy mode
sparc -m "Run full test suite and fix failures" --cowboy-mode

# Multi-provider setup
sparc -m "Security audit" --provider openai --model gpt-4 --expert-provider anthropic
```

For system architecture details, see the [Architecture Guide](architecture.md).
