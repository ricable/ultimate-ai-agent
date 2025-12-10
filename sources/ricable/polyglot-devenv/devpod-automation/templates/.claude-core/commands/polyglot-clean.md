# /polyglot-clean

Automatically fixes formatting issues, resolves linting problems, and addresses type errors across all languages in the polyglot development environment.

## Usage
```
/polyglot-clean [--env <environment>] [--format-only] [--fix-imports] [--type-errors]
```

## Features
- **Auto-formatting** using language-specific tools
- **Import organization** and unused import removal
- **Type error resolution** with intelligent fixes
- **Linting issue fixes** where possible
- **Cross-language consistency** enforcement
- **Safe operation** with backup and rollback capability
- **Performance tracking** during cleanup operations

## Language-Specific Cleaning

### Python (`python-env/`)
- **Format**: `ruff format` (88 char line length)
- **Fix imports**: `ruff check --fix --select I`
- **Auto-fix linting**: `ruff check --fix`
- **Type issues**: `mypy` analysis with suggestions
- **Sort imports**: isort-compatible via ruff

### TypeScript (`typescript-env/`)
- **Format**: `prettier --write`
- **Fix imports**: `eslint --fix --rule @typescript-eslint/no-unused-vars`
- **Auto-fix linting**: `eslint --fix`
- **Type issues**: `tsc --noEmit` with error analysis
- **Organize imports**: VS Code style import sorting

### Rust (`rust-env/`)
- **Format**: `rustfmt`
- **Fix imports**: `cargo fmt` with import grouping
- **Auto-fix linting**: `cargo clippy --fix --allow-dirty`
- **Type issues**: `cargo check` with suggestions
- **Module organization**: `rustfmt` import configuration

### Go (`go-env/`)
- **Format**: `gofmt -w`
- **Fix imports**: `goimports -w`
- **Auto-fix linting**: `golangci-lint run --fix`
- **Type issues**: `go vet` analysis with fixes
- **Module cleanup**: `go mod tidy`

### Nushell (`nushell-env/`)
- **Format**: `nu --check` with style enforcement
- **Fix imports**: Module path validation and correction
- **Script validation**: Syntax checking and auto-correction
- **Function organization**: Consistent parameter formatting
- **Variable naming**: snake_case enforcement

## Cleaning Operations

### Level 1: Formatting Only
- Code formatting without logic changes
- Import sorting and organization
- Whitespace and indentation fixes
- Comment formatting consistency

### Level 2: Safe Auto-fixes
- Remove unused imports
- Fix obvious linting violations
- Correct simple type annotations
- Organize module structure

### Level 3: Intelligent Fixes
- Resolve type errors with context analysis
- Fix complex linting issues
- Update deprecated API usage
- Modernize code patterns

## Instructions
1. **Environment Detection**:
   - Identify changed files from `git status`
   - Map files to appropriate environments
   - Check devbox environment availability

2. **Pre-clean Analysis**:
   - Run quick linting to identify issues
   - Create backup of current state
   - Analyze fix complexity and safety

3. **Incremental Cleaning**:
   - Start with formatting-only operations
   - Progress to safe auto-fixes
   - Apply intelligent fixes with validation

4. **Environment-Specific Execution**:
   ```bash
   # Python
   cd python-env && devbox run format
   cd python-env && devbox run lint --fix
   
   # TypeScript  
   cd typescript-env && devbox run format
   cd typescript-env && devbox run lint --fix
   
   # Rust
   cd rust-env && devbox run format
   cd rust-env && devbox run lint --fix
   
   # Go
   cd go-env && devbox run format
   cd go-env && devbox run lint --fix
   
   # Nushell
   cd nushell-env && devbox run format
   cd nushell-env && devbox run check --fix
   ```

5. **Validation and Verification**:
   - Run tests to ensure no functionality broken
   - Verify code still compiles/runs
   - Check that formatting is consistent
   - Validate import organization

6. **Results Reporting**:
   - Show files modified and changes made
   - Report any issues that couldn't be auto-fixed
   - Provide manual fix recommendations
   - Suggest follow-up actions

## Cleaning Report
```
🧹 POLYGLOT CLEAN RESULTS

📊 SUMMARY
✅ Files processed: 23
🔧 Auto-fixed: 18 issues
⚠️  Manual review: 5 issues
🎯 Environments: 4/5 cleaned

🐍 PYTHON (python-env/)
✅ Formatted: 8 files with ruff
✅ Fixed imports: 3 files organized
✅ Linting: 12 issues auto-fixed
⚠️  Type errors: 2 require manual review

📘 TYPESCRIPT (typescript-env/)  
✅ Formatted: 5 files with prettier
✅ Fixed imports: 2 unused imports removed
✅ Linting: 8 ESLint issues fixed
⚠️  Type errors: 1 complex generic needs review

🦀 RUST (rust-env/)
✅ Formatted: 4 files with rustfmt
✅ Imports: Module organization improved
✅ Clippy: 6 warnings auto-fixed
✅ All type issues resolved

🐹 GO (go-env/)
✅ Formatted: 3 files with gofmt
✅ Imports: goimports applied
✅ Linting: 4 issues fixed
✅ go mod tidy completed

🐚 NUSHELL (nushell-env/)
✅ Scripts: 3 files formatted
✅ Syntax: All validation passed
✅ Modules: Import paths corrected
✅ Style: Consistent parameter formatting

⚠️  MANUAL REVIEW REQUIRED
1. python-env/src/auth.py:42 - Complex type annotation
2. python-env/src/models.py:156 - Generic type constraint
3. typescript-env/src/types.ts:89 - Union type simplification

💡 RECOMMENDATIONS
- Run /polyglot-check to verify all fixes
- Review manual items before committing
- Consider updating type definitions
```

## Safety Features
- **Backup creation** before major changes
- **Incremental fixes** to isolate issues
- **Rollback capability** if cleaning breaks functionality
- **Test validation** after each cleaning phase
- **Git integration** to show exact changes made

## Integration Points
- Uses existing devbox environments and scripts
- Works with current hooks automation system
- Integrates with performance monitoring
- Supports existing linting configurations
- Compatible with IDE formatting settings

## Error Handling
- Continue cleaning other files if one fails
- Provide specific error context and solutions
- Suggest manual commands for complex issues
- Offer to skip problematic files and continue
- Recommend environment-specific debugging steps

## Performance Optimization
- Process files in parallel where safe
- Skip unchanged files based on git status
- Use incremental formatting for large files
- Cache formatting results when possible
- Monitor cleanup performance and report metrics