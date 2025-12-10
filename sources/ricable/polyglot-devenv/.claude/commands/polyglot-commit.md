# /polyglot-commit

Creates optimized git commits for polyglot development environments with automatic language detection, conventional commit formatting, and intelligent commit scoping.

## Usage
```
/polyglot-commit [message]
```

## Features
- **Auto-detects changed languages** based on file extensions and directories
- **Conventional commit format** with appropriate scopes (python, typescript, rust, go, nushell)
- **Intelligent commit types** (feat, fix, refactor, test, docs, chore, perf, style)
- **Cross-language validation** before commit
- **Hooks integration** leverages existing automation system
- **Performance tracking** integrates with analytics scripts

## Language Detection Rules
- `.py` files or `python-env/` → `python` scope
- `.ts/.js` files or `typescript-env/` → `typescript` scope  
- `.rs` files or `rust-env/` → `rust` scope
- `.go` files or `go-env/` → `go` scope
- `.nu` files or `nushell-env/` → `nushell` scope
- Multiple languages → `polyglot` scope
- Config/hooks → `config` scope

## Commit Type Detection
- New files/features → `feat`
- Bug fixes → `fix`  
- Code cleanup → `refactor`
- Test files → `test`
- Documentation → `docs`
- Build/config → `chore`
- Performance → `perf`
- Formatting → `style`

## Instructions
1. **Analyze staged changes** using `git status` and `git diff --cached`
2. **Detect languages** from changed file paths and extensions
3. **Determine commit type** based on file changes and context
4. **Run pre-commit validation**:
   - Execute `devbox run lint` in affected environments
   - Run `devbox run test` for test files
   - Check for security issues with existing scanners
5. **Generate commit message** using format: `<type>(<scope>): <description>`
6. **Create commit** with proper formatting and emoji icons
7. **Track performance** using existing analytics if available

## Examples
```bash
# Single language change
feat(python): add user authentication endpoint

# Multiple languages  
feat(polyglot): implement cross-language API client

# Configuration change
chore(config): update devbox environments and hooks

# Test addition
test(rust): add integration tests for payment service

# Performance improvement
perf(typescript): optimize data processing pipeline

# Documentation update
docs(nushell): add automation script examples
```

## Pre-commit Validation
- Lint changed files in their respective environments
- Run tests if test files are modified
- Security scan for sensitive patterns
- Dependency vulnerability check
- Format validation across all languages
- Cross-environment consistency check

## Integration Points
- Uses existing devbox environments for validation
- Leverages `.claude/polyglot-hooks-config.json` settings
- Integrates with performance analytics scripts
- Works with GitHub integration for issue linking
- Supports existing secret scanning via teller

## Error Handling
- If validation fails, provide specific fix recommendations
- Show which environment commands to run manually
- Suggest using `/polyglot-clean` for formatting issues
- Recommend `/polyglot-check` for comprehensive analysis