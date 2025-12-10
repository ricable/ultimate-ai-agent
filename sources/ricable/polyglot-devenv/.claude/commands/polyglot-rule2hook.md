# Task: Convert Polyglot Project Rules to Environment-Aware Claude Code Hooks

You are an expert at converting natural language project rules into Claude Code hook configurations for polyglot development environments. Your task is to analyze rules and generate environment-aware hook configurations that integrate with devbox environments and Nushell automation.

## Instructions

1. If rules are provided as arguments, analyze those rules
2. If no arguments are provided, read and analyze the CLAUDE.md file from these locations:
   - `./CLAUDE.md` (project memory)
   - `./CLAUDE.local.md` (local project memory)  
   - `~/.claude/CLAUDE.md` (user memory)

3. For each rule, determine:
   - The appropriate hook event (PreToolUse, PostToolUse, Stop, Notification)
   - The tool matcher pattern (exact tool names or regex)
   - Environment-aware commands using devbox and Nushell integration
   - File type and path-based triggers

4. Generate polyglot-aware hook configurations
5. Save to `.claude/settings.json` (project-specific) or merge with `~/.claude/settings.json`
6. Provide a summary of configured automations

## Polyglot Environment Structure

This project uses isolated devbox environments:
```
polyglot-project/
├── python-env/     # Python + uv + ruff + mypy + pytest
├── typescript-env/ # Node.js + TypeScript + ESLint + Prettier + Jest
├── rust-env/       # Rust + Cargo + Clippy + rustfmt
├── go-env/         # Go + golangci-lint + goimports
├── nushell-env/    # Nushell automation scripts
└── CLAUDE.md       # This documentation
```

## Environment-Aware Hook Patterns

### Python Environment Hooks
- **Format**: `cd python-env && devbox run format` (ruff format)
- **Lint**: `cd python-env && devbox run lint` (ruff check)
- **Type Check**: `cd python-env && devbox run type-check` (mypy)
- **Test**: `cd python-env && devbox run test` (pytest)

### TypeScript Environment Hooks  
- **Format**: `cd typescript-env && devbox run format` (prettier)
- **Lint**: `cd typescript-env && devbox run lint` (eslint)
- **Test**: `cd typescript-env && devbox run test` (jest)

### Rust Environment Hooks
- **Format**: `cd rust-env && devbox run format` (rustfmt)
- **Lint**: `cd rust-env && devbox run lint` (clippy)
- **Test**: `cd rust-env && devbox run test` (cargo test)

### Go Environment Hooks
- **Format**: `cd go-env && devbox run format` (goimports)
- **Lint**: `cd go-env && devbox run lint` (golangci-lint)
- **Test**: `cd go-env && devbox run test` (go test)

### Cross-Language Automation
- **Validate All**: `nu nushell-env/scripts/validate-all.nu`
- **Parallel Validation**: `nu nushell-env/scripts/validate-all.nu parallel`

## Smart Environment Detection

Generate commands that detect the current context:

```bash
# Environment-aware formatting
if [[ "$PWD" =~ python-env ]] || [[ "$(jq -r '.tool_input.file_path // ""')" =~ \.py$ ]]; then
  cd python-env && devbox run format 2>/dev/null || true
elif [[ "$PWD" =~ typescript-env ]] || [[ "$(jq -r '.tool_input.file_path // ""')" =~ \.(ts|js)$ ]]; then
  cd typescript-env && devbox run format 2>/dev/null || true
elif [[ "$PWD" =~ rust-env ]] || [[ "$(jq -r '.tool_input.file_path // ""')" =~ \.rs$ ]]; then
  cd rust-env && devbox run format 2>/dev/null || true
elif [[ "$PWD" =~ go-env ]] || [[ "$(jq -r '.tool_input.file_path // ""')" =~ \.go$ ]]; then
  cd go-env && devbox run format 2>/dev/null || true
fi
```

## Hook Events for Polyglot Environment

### PreToolUse (Validation & Checks)
- **Keywords**: "before", "check", "validate", "prevent", "scan", "verify"
- **Use Cases**: 
  - Lint before saves/commits
  - Security scans before file writes
  - Type checking before builds
  - Environment validation

### PostToolUse (Formatting & Testing)
- **Keywords**: "after", "following", "once done", "when finished"
- **Use Cases**:
  - Auto-format after file edits
  - Run tests after test file changes
  - Update dependencies after package changes
  - Regenerate docs after code changes

### Stop (Cross-Language Validation)
- **Keywords**: "finish", "complete", "end task", "done", "wrap up"
- **Use Cases**:
  - Run full validation suite
  - Generate summary reports
  - Clean up temporary files
  - Check git status

## Example Rule Conversions

### Example 1: Auto-Format Rules
**Rule**: "Format code files after editing"
**Generated Hook**:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|MultiEdit|Write",
      "hooks": [{
        "type": "command",
        "command": "bash -c 'file_path=$(echo \"$0\" | jq -r \".tool_input.file_path // \\\"\\\"\" 2>/dev/null || echo \"\"); if [[ \"$file_path\" =~ \\.py$ ]] || [[ \"$PWD\" =~ python-env ]]; then cd python-env && devbox run format --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.(ts|js)$ ]] || [[ \"$PWD\" =~ typescript-env ]]; then cd typescript-env && devbox run format --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.rs$ ]] || [[ \"$PWD\" =~ rust-env ]]; then cd rust-env && devbox run format --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.go$ ]] || [[ \"$PWD\" =~ go-env ]]; then cd go-env && devbox run format --quiet 2>/dev/null || true; fi'"
      }]
    }]
  }
}
```

### Example 2: Cross-Language Validation
**Rule**: "Run validation across all environments when finishing tasks"
**Generated Hook**:
```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "cd nushell-env && nu scripts/validate-all.nu 2>/dev/null || echo 'Validation completed'"
      }]
    }]
  }
}
```

### Example 3: Environment-Specific Testing
**Rule**: "Run tests after modifying test files"
**Generated Hook**:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|MultiEdit|Write",
      "hooks": [{
        "type": "command",
        "command": "bash -c 'file_path=$(echo \"$0\" | jq -r \".tool_input.file_path // \\\"\\\"\" 2>/dev/null || echo \"\"); if [[ \"$file_path\" =~ test.*\\.py$ ]]; then cd python-env && devbox run test --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ .*\\.(test|spec)\\.(ts|js)$ ]]; then cd typescript-env && devbox run test --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ .*_test\\.rs$ ]]; then cd rust-env && devbox run test --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ .*_test\\.go$ ]]; then cd go-env && devbox run test --quiet 2>/dev/null || true; fi'"
      }]
    }]
  }
}
```

## Advanced Features

### Nushell Integration
For complex automation, generate hooks that call Nushell scripts:
```bash
# Example: Complex validation hook
nu nushell-env/scripts/validate-file.nu --file "$(jq -r '.tool_input.file_path' || echo '')"
```

### Performance Optimization
- Use `--quiet` flags to minimize output
- Add `2>/dev/null || true` for error handling
- Check file existence before running commands
- Use specific matchers to avoid unnecessary executions

### Security Considerations
- Validate file paths to prevent injection
- Use absolute paths for script execution
- Limit hook scope with specific matchers
- Test hooks thoroughly before production use

## Common Polyglot Rule Patterns

- **"Format [language] files after editing"** → PostToolUse + environment detection
- **"Run tests for [language] after test changes"** → PostToolUse + file pattern matching  
- **"Lint [language] code before saves"** → PreToolUse + environment-specific commands
- **"Validate all environments when done"** → Stop + Nushell validation scripts
- **"Check [pattern] across projects"** → PreToolUse + cross-language commands

## Configuration Storage

For project-specific hooks (recommended):
- Save to `.claude/settings.json` in project root
- Include in version control for team consistency

For user-global hooks:
- Save to `~/.claude/settings.json`
- Apply across all projects

## User Input
$ARGUMENTS