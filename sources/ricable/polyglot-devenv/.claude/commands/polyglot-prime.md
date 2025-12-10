# /polyglot-prime

Primes Claude with comprehensive understanding of the polyglot development environment by loading project structure, configurations, and context across all languages.

## Usage
```
/polyglot-prime [--deep] [--env <environment>] [--focus <area>]
```

## Features
- **Multi-environment context** loading across Python, TypeScript, Rust, Go, Nushell
- **Configuration analysis** of devbox, project files, and tooling setup
- **Dependency mapping** across all language ecosystems
- **Architecture understanding** of polyglot project structure
- **Development workflow** comprehension including hooks and automation
- **Performance baseline** establishment for intelligent monitoring
- **Security context** including secret management and scanning setup

## Context Loading Strategy

### Project Structure Analysis
- Repository root and environment organization
- Devbox configuration and package management
- Cross-language dependencies and integrations
- Automation scripts and workflow orchestration

### Environment-Specific Context
- **Python**: `pyproject.toml`, dependencies, virtual environment setup
- **TypeScript**: `package.json`, `tsconfig.json`, Node.js configuration  
- **Rust**: `Cargo.toml`, workspace configuration, crate dependencies
- **Go**: `go.mod`, module structure, build configuration
- **Nushell**: Scripts, modules, automation workflows, configuration files

### Development Infrastructure
- Claude Code hooks and automation system
- Intelligence monitoring scripts and analytics
- Security scanning and secret management
- Performance monitoring and optimization tools
- GitHub integration and workflow automation

## Instructions
1. **Repository Structure Mapping**:
   - Read project root structure and organization
   - Identify all `*-env/` directories and their purposes
   - Understand `.claude/` configuration and hooks setup
   - Map cross-environment dependencies and workflows

2. **Configuration Analysis**:
   ```bash
   # Read all devbox configurations
   python-env/devbox.json, typescript-env/devbox.json, etc.
   
   # Language-specific configurations
   python-env/pyproject.toml
   typescript-env/package.json, tsconfig.json
   rust-env/Cargo.toml
   go-env/go.mod
   nushell-env/config.nu, common.nu
   ```

3. **CLAUDE.md Deep Analysis**:
   - Project overview and development principles
   - Environment setup and usage patterns
   - Automation and intelligence features
   - Development workflow and best practices
   - Integration points and dependencies

4. **Hooks and Automation Context**:
   - Read `.claude/polyglot-hooks-config.json`
   - Understand automation triggers and workflows
   - Map intelligence monitoring capabilities
   - Analyze security and performance tracking

5. **Intelligence Scripts Inventory**:
   ```bash
   nushell-env/scripts/performance-analytics.nu
   nushell-env/scripts/resource-monitor.nu  
   nushell-env/scripts/dependency-monitor.nu
   nushell-env/scripts/security-scanner.nu
   nushell-env/scripts/environment-drift.nu
   nushell-env/scripts/failure-pattern-learning.nu
   nushell-env/scripts/test-intelligence.nu
   ```

6. **Development State Assessment**:
   - Current git status and recent changes
   - Active development areas and focus
   - Performance baselines and metrics
   - Security posture and compliance status
   - Cross-environment consistency validation

## Context Report Structure
```
🧠 POLYGLOT ENVIRONMENT PRIME COMPLETE

📁 PROJECT STRUCTURE
✅ Root: polyglot-devenv with 5 environments
✅ Environments: python-env, typescript-env, rust-env, go-env, nushell-env
✅ Configuration: .claude/ with hooks automation
✅ Intelligence: 7 monitoring scripts active

📦 ENVIRONMENT OVERVIEW
🐍 Python: FastAPI project, uv package management, ruff/mypy tooling
📘 TypeScript: Node.js 20, strict mode, Jest testing, ESLint/Prettier
🦀 Rust: Workspace config, tokio async, clippy/rustfmt tooling  
🐹 Go: Module-based, 1.22 toolchain, golangci-lint standards
🐚 Nushell: Automation hub, 7 intelligence scripts, devbox orchestration

🔧 DEVELOPMENT INFRASTRUCTURE
✅ Devbox: Isolated environments with reproducible builds
✅ Hooks: 8 intelligent automation triggers active
✅ Monitoring: Performance, security, dependency, environment drift
✅ GitHub: Integration for automated issue creation and tracking
✅ Security: Teller secret management, vulnerability scanning

⚡ CURRENT STATE
🔄 Branch: feature/claude-hooks-automation
📝 Changes: 15 modified files, hooks configuration active
🎯 Focus: Intelligence system integration and automation
📊 Performance: Baseline metrics established
🛡️  Security: All scans passing, no critical vulnerabilities

🎛️  AUTOMATION CAPABILITIES
• Auto-formatting after file edits (ruff, prettier, rustfmt, goimports)
• Test execution on test file changes
• Security scanning on code modifications  
• Pre-commit validation with performance tracking
• Cross-environment consistency checking
• GitHub issue creation for critical failures
• Performance regression detection
• Dependency vulnerability monitoring

💡 DEVELOPMENT CONTEXT
• Primary languages: Python (FastAPI), TypeScript (Node.js), Rust, Go
• Orchestration: Nushell scripts for cross-language automation
• Architecture: Microservices-ready with shared DevOps workflows
• Testing: Comprehensive coverage across all environments
• Deployment: Container-ready with Kubernetes integration scripts

🎯 OPTIMIZATION OPPORTUNITIES
• Performance: Build time tracking shows TypeScript +15% slower
• Dependencies: 12 packages across environments need updates  
• Security: All environments passing, proactive monitoring active
• Consistency: Environment drift detection shows minimal variance

🚀 READY FOR DEVELOPMENT
Claude now has comprehensive understanding of:
✓ All 5 development environments and their tooling
✓ Cross-language automation and workflow orchestration  
✓ Intelligence monitoring and performance optimization
✓ Security scanning and vulnerability management
✓ Development best practices and coding standards
✓ GitHub integration and issue management workflows
```

## Deep Prime Features (--deep flag)
- **Code Pattern Analysis**: Common patterns and anti-patterns across languages
- **Dependency Graph**: Cross-environment dependency relationships
- **Performance Profiling**: Historical build times and optimization opportunities
- **Security Audit**: Comprehensive security posture assessment
- **Architecture Mapping**: Service boundaries and integration points
- **Test Coverage Analysis**: Cross-language testing strategy and gaps

## Focus Areas (--focus flag)
- `--focus security`: Deep dive into security configuration and practices
- `--focus performance`: Detailed performance analysis and optimization
- `--focus dependencies`: Comprehensive dependency health and updates
- `--focus architecture`: Service design and integration patterns
- `--focus testing`: Test strategy and coverage analysis
- `--focus automation`: Hooks and workflow optimization

## Integration Benefits
- **Contextual Assistance**: Better code suggestions based on project patterns
- **Cross-Language Awareness**: Understand polyglot integration requirements
- **Workflow Optimization**: Leverage existing automation and intelligence
- **Problem Prevention**: Proactive issue detection and resolution
- **Performance Optimization**: Data-driven improvement recommendations
- **Security Enhancement**: Context-aware security best practices

## Performance Optimization
- Loads most critical context first
- Caches frequently accessed information
- Skips unchanged configurations
- Provides incremental context updates
- Uses intelligent content filtering based on current focus