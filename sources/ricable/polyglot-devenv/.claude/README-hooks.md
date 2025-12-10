# Claude Code Hooks for Polyglot Development Environment

This directory contains Claude Code hooks integration for the polyglot development environment, providing automated quality gates, formatting, testing, and validation across Python, TypeScript, Rust, Go, and Nushell environments.

## 🚀 Quick Start

### Install Hooks (Recommended - Project-Specific)
```bash
# Install hooks for this project only
./.claude/install-hooks.sh

# Or install globally for all your projects
./.claude/install-hooks.sh --user
```

### Using the Rule2Hook Command
```bash
# In Claude Code, use the enhanced rule2hook command
/project:polyglot-rule2hook "Format code after editing files"

# Or convert existing CLAUDE.md rules
/project:polyglot-rule2hook
```

### Managing Hooks with Nushell
```bash
# Check current hooks status
nu nushell-env/scripts/hooks.nu status

# Generate hooks from CLAUDE.md
nu nushell-env/scripts/hooks.nu generate

# Validate hook configuration
nu nushell-env/scripts/hooks.nu validate

# Test hook functionality
nu nushell-env/scripts/hooks.nu test --hook-type format

# Backup current configuration
nu nushell-env/scripts/hooks.nu backup --name "pre-update"
```

## 🎯 Automated Features

### 📝 Auto-Formatting (PostToolUse)
Automatically formats code after editing files:

- **Python**: `ruff format` via `devbox run format`
- **TypeScript/JavaScript**: `prettier` via `devbox run format`  
- **Rust**: `rustfmt` via `devbox run format`
- **Go**: `goimports` via `devbox run format`
- **Nushell**: `nu format` via `devbox run format`

**Triggers**: After using Edit, MultiEdit, or Write tools on relevant files

### 🧪 Auto-Testing (PostToolUse)  
Automatically runs tests when test files are modified:

- **Python**: `pytest` for `test_*.py`, `*_test.py`, `*.test.py`
- **TypeScript/JS**: `jest` for `*.test.ts`, `*.spec.js`, etc.
- **Rust**: `cargo test` for `*_test.rs`, `tests/*.rs`
- **Go**: `go test` for `*_test.go`, `*.test.go`
- **Nushell**: `nu test` for `test_*.nu`, `*_test.nu`

**Triggers**: After editing test files

### 🔍 Pre-Commit Validation (PreToolUse)
Runs validation before git commits:

- **Environment-specific linting**: Runs `devbox run lint` in current environment
- **Cross-language validation**: Runs `nu scripts/validate-all.nu` for comprehensive checks
- **Secret scanning**: Scans configuration files for potential secrets

**Triggers**: Before executing `git commit` commands

### 🎯 Task Completion Automation (Stop)
Runs when Claude Code finishes responding:

- **Git status check**: Shows current repository status
- **Cross-language validation**: Runs parallel validation across all environments
- **Summary reporting**: Provides overview of project health

**Triggers**: When Claude Code stops responding

### 🔔 Notification Logging (Notification)
Logs all Claude Code notifications to `~/.claude/notifications.log`

**Triggers**: When Claude Code sends notifications

## 🏗️ Architecture

### Environment Detection
Hooks intelligently detect the current environment using:

1. **File extension matching**: `.py` → Python, `.ts/.js` → TypeScript, etc.
2. **Directory context**: `python-env/` → Python environment
3. **PWD analysis**: Current working directory path

### Command Structure  
```bash
# Example: Environment-aware formatting command
if [[ "$file_path" =~ \.py$ ]] || [[ "$PWD" =~ python-env ]]; then
  cd python-env && devbox run format --quiet 2>/dev/null || true
elif [[ "$file_path" =~ \.(ts|js)$ ]] || [[ "$PWD" =~ typescript-env ]]; then
  cd typescript-env && devbox run format --quiet 2>/dev/null || true
# ... additional environments
fi
```

### Error Handling
- **Non-blocking**: Uses `|| true` to prevent hook failures from stopping Claude
- **Quiet mode**: Uses `--quiet` flags to minimize output noise
- **Graceful fallbacks**: Continues execution even if specific environments aren't available

## 📁 File Structure

```
.claude/
├── commands/
│   └── polyglot-rule2hook.md     # Enhanced rule2hook command
├── polyglot-hooks-config.json    # Predefined hook configuration
├── install-hooks.sh              # Installation script
├── README-hooks.md               # This documentation
└── settings.json                 # Project-specific hook settings (created on install)
```

## ⚙️ Configuration

### Project Settings (`.claude/settings.json`)
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "environment-aware-formatting-command"
          }
        ]
      }
    ],
    "PreToolUse": [...],
    "Stop": [...],
    "Notification": [...]
  }
}
```

### Hook Events

| Event | When | Use Cases |
|-------|------|-----------|
| `PreToolUse` | Before tool execution | Validation, linting, security checks |
| `PostToolUse` | After tool completion | Formatting, testing, building |
| `Stop` | When Claude finishes | Status checks, summaries, cleanup |
| `Notification` | On Claude notifications | Logging, external integrations |

## 🔧 Customization

### Adding New Rules
1. **Via Rule2Hook**: Use `/project:polyglot-rule2hook "your rule"`
2. **Via Nushell**: Add rules to CLAUDE.md and run `nu scripts/hooks.nu generate`
3. **Manual**: Edit `.claude/settings.json` directly

### Environment-Specific Hooks
Create hooks that target specific environments:

```json
{
  "matcher": "Edit|MultiEdit|Write",
  "hooks": [
    {
      "type": "command",
      "command": "if [[ \"$PWD\" =~ rust-env ]]; then cargo clippy; fi"
    }
  ]
}
```

### File-Type Specific Hooks
Target specific file types:

```json
{
  "matcher": "Write",
  "hooks": [
    {
      "type": "command", 
      "command": "if [[ \"$(jq -r '.tool_input.file_path')\" =~ \\.toml$ ]]; then toml-sort \"$file\"; fi"
    }
  ]
}
```

## 🧪 Testing

### Test Individual Hook Types
```bash
# Test formatting hooks
nu nushell-env/scripts/hooks.nu test --hook-type format

# Test linting hooks  
nu nushell-env/scripts/hooks.nu test --hook-type lint

# Test all hooks
nu nushell-env/scripts/hooks.nu test --hook-type test
```

### Test Environment Availability
```bash
# Check which environments are ready
./.claude/install-hooks.sh --test
```

### Manual Testing
1. **Format test**: Edit a Python file, save it, check if `ruff format` ran
2. **Test run**: Modify a test file, verify appropriate test runner executed
3. **Commit test**: Run `git commit`, confirm linting/validation occurred
4. **Task completion**: Finish a task in Claude Code, see status summary

## 🔒 Security Considerations

### Safe Commands
- All commands use `|| true` to prevent blocking Claude
- File paths are validated and quoted to prevent injection
- Secret scanning is included for configuration files

### Permissions
- Hooks run with your user permissions
- No sudo or elevated privileges required
- All operations are confined to project directory

### Best Practices
1. **Review generated hooks** before applying them
2. **Backup configurations** before major changes
3. **Test in development** before production use
4. **Monitor hook performance** to avoid slowdowns

## 🐛 Troubleshooting

### Common Issues

**Hooks not executing:**
- Check `.claude/settings.json` exists and is valid JSON
- Verify you're running Claude Code in the project directory
- Use `nu scripts/hooks.nu validate` to check configuration

**Environment not detected:**
- Ensure `devbox.json` exists in environment directories
- Check that you're in the correct directory context
- Verify file extensions match expected patterns

**Commands failing:**
- Check if devbox environments are properly set up
- Ensure required tools are installed in each environment
- Use `--test` flag to verify environment availability

**Performance issues:**
- Hooks should complete within 60 seconds
- Use `--quiet` flags to reduce output
- Consider reducing hook scope with specific matchers

### Debug Commands
```bash
# Check hook status
nu nushell-env/scripts/hooks.nu status

# Validate configuration
nu nushell-env/scripts/hooks.nu validate

# Test specific environments
./.claude/install-hooks.sh --test

# View recent notifications
tail ~/.claude/notifications.log
```

## 📚 Resources

- [Claude Code Hooks Documentation](https://docs.anthropic.com/en/docs/claude-code/hooks)
- [Slash Commands Guide](https://docs.anthropic.com/en/docs/claude-code/slash-commands)
- [Devbox Documentation](https://www.jetify.com/devbox/docs/)
- [Nushell Documentation](https://www.nushell.sh/book/)

## 🤝 Contributing

To improve the hooks system:

1. **Test thoroughly** with various file types and scenarios
2. **Add new environment support** by extending detection logic
3. **Optimize performance** by reducing command execution time
4. **Improve error handling** with better fallback strategies
5. **Document patterns** for common automation needs

## 📝 Examples

### Example 1: Custom Python Hook
```bash
/project:polyglot-rule2hook "Run black and isort on Python files after editing"
```

### Example 2: Cross-Language Validation  
```bash
/project:polyglot-rule2hook "Validate all environments when finishing tasks"
```

### Example 3: Git Workflow
```bash
/project:polyglot-rule2hook "Check for TODO comments before committing code"
```

### Example 4: Environment-Specific Testing
```bash
/project:polyglot-rule2hook "Run coverage reports after running Python tests"
```

---

**🎉 Enjoy automated, intelligent development workflows with Claude Code hooks!**