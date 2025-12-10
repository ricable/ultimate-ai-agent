# /polyglot-check

Performs comprehensive code quality, security, and performance checks across all languages in the polyglot development environment.

## Usage
```
/polyglot-check [--env <environment>] [--security] [--performance] [--dependencies]
```

## Features
- **Cross-language validation** across Python, TypeScript, Rust, Go, Nushell
- **Security analysis** using existing scanner scripts
- **Performance monitoring** with analytics integration
- **Dependency health** checks and vulnerability scanning
- **Environment consistency** validation
- **Intelligent reporting** with actionable recommendations
- **GitHub integration** for critical issue tracking

## Check Categories

### Code Quality Checks
- **Python**: `ruff check`, `mypy`, `pytest --co -q`
- **TypeScript**: `eslint`, `tsc --noEmit`, `jest --passWithNoTests`
- **Rust**: `clippy`, `cargo check`, `cargo test --no-run`
- **Go**: `golangci-lint`, `go vet`, `go test -compile-packages`
- **Nushell**: syntax validation, module consistency

### Security Analysis
- Secret detection across all files
- Security anti-pattern analysis
- Vulnerability scanning in dependencies
- Configuration security audit
- Environment variable validation

### Performance Analysis  
- Build time monitoring
- Test execution performance
- Resource usage analysis
- Memory leak detection
- Optimization recommendations

### Dependency Health
- Outdated package detection
- Security vulnerability scanning
- License compliance checking
- Dependency tree analysis
- Update recommendations

## Environment Detection
Automatically detects and validates environments based on:
- Directory structure (`*-env/` folders)
- Configuration files (`devbox.json`, `pyproject.toml`, etc.)
- Active devbox shells
- File modification patterns

## Instructions
1. **Environment Discovery**:
   - Scan for `*-env/` directories
   - Check `devbox.json` files for package configurations
   - Identify active development environments

2. **Quality Validation**:
   - Run `devbox run lint` in each environment
   - Execute `devbox run test --quiet` for test validation
   - Check code formatting and style consistency

3. **Security Analysis**:
   - Execute `nu nushell-env/scripts/security-scanner.nu scan-all`
   - Run secret detection with existing teller configuration
   - Validate environment variable security

4. **Performance Assessment**:
   - Use `nu nushell-env/scripts/performance-analytics.nu measure`
   - Monitor resource usage during checks
   - Generate optimization recommendations

5. **Dependency Health**:
   - Run `nu nushell-env/scripts/dependency-monitor.nu scan-all`
   - Check for security vulnerabilities
   - Analyze update requirements

6. **Report Generation**:
   - Consolidate findings across all environments
   - Prioritize issues by severity and impact
   - Provide specific fix recommendations
   - Create GitHub issues for critical problems

## Report Structure
```
🔍 POLYGLOT ENVIRONMENT CHECK REPORT

📊 SUMMARY
✅ Environments: 4/5 passing
⚠️  Issues: 12 warnings, 2 errors
🛡️  Security: 1 critical vulnerability
📈 Performance: 3 optimization opportunities

🐍 PYTHON ENVIRONMENT
✅ Code Quality: Passed (ruff, mypy)
✅ Tests: 127 passed, 0 failed
⚠️  Dependencies: 3 outdated packages
✅ Security: No issues found

📘 TYPESCRIPT ENVIRONMENT  
✅ Code Quality: Passed (eslint, tsc)
❌ Tests: 2 failed, build errors
⚠️  Performance: Build time +25% slower
✅ Security: No issues found

🦀 RUST ENVIRONMENT
✅ Code Quality: Passed (clippy, check)
✅ Tests: All integration tests passed
✅ Performance: Optimal
✅ Security: No issues found

🐹 GO ENVIRONMENT
✅ Code Quality: Passed (golangci-lint)
✅ Tests: Coverage 87%
⚠️  Dependencies: 1 security vulnerability
✅ Performance: Good

🐚 NUSHELL ENVIRONMENT
✅ Scripts: All syntax valid
✅ Modules: Consistent structure
✅ Performance: Automation optimized
✅ Security: Configuration secure

🎯 RECOMMENDATIONS
1. Fix TypeScript build errors in auth module
2. Update Python packages: fastapi, pydantic
3. Address Go dependency vulnerability (severity: medium)
4. Optimize TypeScript build configuration

🔗 GITHUB INTEGRATION
- Created issue #123 for critical TypeScript build failure
- Updated issue #98 with dependency vulnerability details
```

## Integration Points
- Uses existing devbox environments and scripts
- Leverages intelligence monitoring scripts in `nushell-env/scripts/`
- Integrates with GitHub via existing automation
- Works with teller for secret management
- Supports existing hooks configuration

## Performance Optimization
- Runs checks in parallel where possible
- Uses `--quiet` flags to minimize output
- Caches results for faster subsequent runs
- Skips unchanged environments when appropriate
- Provides incremental check options

## Error Recovery
- Continues checking other environments if one fails
- Provides specific error context and solutions
- Suggests manual commands for failed checks
- Offers to run `/polyglot-clean` for fixable issues
- Recommends environment-specific debugging steps