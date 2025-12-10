# Polyglot Development Environment Makefile
# Comprehensive automation for multi-language development

# Variables
NU = nu
NODE = node
DEVBOX = devbox
COUNT ?= 1

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
PURPLE = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Default target
.PHONY: help
help:
	@echo "$(CYAN)Polyglot Development Environment$(NC)"
	@echo "$(YELLOW)================================$(NC)"
	@echo ""
	@echo "$(GREEN)🚀 Setup & Installation:$(NC)"
	@echo "  make setup           - Complete automated setup"
	@echo "  make install-deps    - Install system dependencies"
	@echo "  make setup-envs      - Setup all development environments"
	@echo "  make validate        - Validate all environments"
	@echo ""
	@echo "$(GREEN)🤖 MCP Server:$(NC)"
	@echo "  make mcp-build       - Build MCP server"
	@echo "  make mcp-start       - Start MCP server"
	@echo "  make mcp-test        - Test MCP server"
	@echo ""
	@echo "$(GREEN)🐳 DevPod Containers:$(NC)"
	@echo "  make devpod-python   - Create Python workspace(s) [COUNT=n]"
	@echo "  make devpod-typescript - Create TypeScript workspace(s) [COUNT=n]"
	@echo "  make devpod-rust     - Create Rust workspace(s) [COUNT=n]"
	@echo "  make devpod-go       - Create Go workspace(s) [COUNT=n]"
	@echo "  make devpod-list     - List all DevPod workspaces"
	@echo "  make devpod-clean    - Clean up all DevPod workspaces"
	@echo ""
	@echo "$(GREEN)🔍 Validation & Testing:$(NC)"
	@echo "  make test            - Run tests in all environments"
	@echo "  make test-python     - Run Python tests"
	@echo "  make test-typescript - Run TypeScript tests"
	@echo "  make test-rust       - Run Rust tests"
	@echo "  make test-go         - Run Go tests"
	@echo "  make test-nushell    - Run Nushell tests"
	@echo "  make validate-quick  - Quick validation check"
	@echo "  make validate-parallel - Parallel validation"
	@echo ""
	@echo "$(GREEN)🛠️ Development:$(NC)"
	@echo "  make format          - Format code in all environments"
	@echo "  make lint            - Lint code in all environments"
	@echo "  make clean           - Clean all environments"
	@echo "  make deps-update     - Update dependencies"
	@echo ""
	@echo "$(GREEN)📊 Monitoring & Analytics:$(NC)"
	@echo "  make perf-dashboard  - Show performance dashboard"
	@echo "  make perf-report     - Generate performance report"
	@echo "  make security-scan   - Run security scan"
	@echo "  make resource-monitor - Monitor system resources"
	@echo ""
	@echo "$(GREEN)🔧 Utilities:$(NC)"
	@echo "  make env-status      - Show environment status"
	@echo "  make install-hooks   - Install Claude Code hooks"
	@echo "  make update-docs     - Update documentation"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make devpod-python COUNT=3    # Create 3 Python workspaces"
	@echo "  make test-rust                # Run Rust tests only"
	@echo "  make validate-parallel        # Fast parallel validation"

# ============================================================================
# Setup & Installation
# ============================================================================

.PHONY: setup
setup: install-deps setup-envs validate mcp-build
	@echo "$(GREEN)✅ Complete setup finished!$(NC)"
	@echo "$(CYAN)Next steps:$(NC)"
	@echo "  1. Start MCP server: make mcp-start"
	@echo "  2. Test environments: make validate"
	@echo "  3. Create DevPod workspace: make devpod-python"

.PHONY: install-deps
install-deps:
	@echo "$(BLUE)📦 Installing system dependencies...$(NC)"
	@if ! command -v nu >/dev/null 2>&1; then \
		echo "$(RED)❌ Nushell not found. Please install it first.$(NC)"; \
		echo "$(YELLOW)See README.md for installation instructions.$(NC)"; \
		exit 1; \
	fi
	@if ! command -v devbox >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️  Installing DevBox...$(NC)"; \
		curl -fsSL https://get.jetify.com/devbox | bash; \
	fi
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

.PHONY: setup-envs
setup-envs:
	@echo "$(BLUE)🏗️  Setting up development environments...$(NC)"
	@cd dev-env/python && $(DEVBOX) run install || true
	@cd dev-env/typescript && $(DEVBOX) run install || true
	@cd dev-env/rust && $(DEVBOX) run build || true
	@cd dev-env/go && $(DEVBOX) run build || true
	@cd dev-env/nushell && $(DEVBOX) run setup || true
	@echo "$(GREEN)✅ All environments setup complete$(NC)"

# ============================================================================
# MCP Server Management
# ============================================================================

.PHONY: mcp-build
mcp-build:
	@echo "$(BLUE)🔨 Building MCP server...$(NC)"
	@cd mcp && npm install && npm run build
	@echo "$(GREEN)✅ MCP server built successfully$(NC)"

.PHONY: mcp-start
mcp-start:
	@echo "$(BLUE)🚀 Starting MCP server...$(NC)"
	@cd mcp && npm run start

.PHONY: mcp-test
mcp-test:
	@echo "$(BLUE)🧪 Testing MCP server...$(NC)"
	@echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | $(NODE) mcp/dist/index.js

.PHONY: mcp-watch
mcp-watch:
	@echo "$(BLUE)👀 Starting MCP server in watch mode...$(NC)"
	@cd mcp && npm run watch

# ============================================================================
# DevPod Container Management
# ============================================================================

.PHONY: devpod-python
devpod-python:
	@echo "$(BLUE)🐍 Creating $(COUNT) Python DevPod workspace(s)...$(NC)"
	@$(NU) .claude/commands/devpod-python.md $(COUNT)

.PHONY: devpod-typescript
devpod-typescript:
	@echo "$(BLUE)📘 Creating $(COUNT) TypeScript DevPod workspace(s)...$(NC)"
	@$(NU) .claude/commands/devpod-typescript.md $(COUNT)

.PHONY: devpod-rust
devpod-rust:
	@echo "$(BLUE)🦀 Creating $(COUNT) Rust DevPod workspace(s)...$(NC)"
	@$(NU) .claude/commands/devpod-rust.md $(COUNT)

.PHONY: devpod-go
devpod-go:
	@echo "$(BLUE)🐹 Creating $(COUNT) Go DevPod workspace(s)...$(NC)"
	@$(NU) .claude/commands/devpod-go.md $(COUNT)

.PHONY: devpod-list
devpod-list:
	@echo "$(BLUE)📋 Listing DevPod workspaces...$(NC)"
	@devpod list

.PHONY: devpod-clean
devpod-clean:
	@echo "$(YELLOW)🧹 Cleaning up DevPod workspaces...$(NC)"
	@$(NU) dev-env/nushell/scripts/containers.nu cleanup --all

# ============================================================================
# Testing & Validation
# ============================================================================

.PHONY: validate
validate:
	@echo "$(BLUE)✅ Running full validation...$(NC)"
	@$(NU) scripts/validate-all.nu --parallel

.PHONY: validate-quick
validate-quick:
	@echo "$(BLUE)⚡ Running quick validation...$(NC)"
	@$(NU) scripts/validate-all.nu quick

.PHONY: validate-parallel
validate-parallel:
	@echo "$(BLUE)🚀 Running parallel validation...$(NC)"
	@$(NU) scripts/validate-all.nu --parallel

.PHONY: test
test: test-python test-typescript test-rust test-go test-nushell

.PHONY: test-python
test-python:
	@echo "$(BLUE)🐍 Running Python tests...$(NC)"
	@cd dev-env/python && $(DEVBOX) run test

.PHONY: test-typescript
test-typescript:
	@echo "$(BLUE)📘 Running TypeScript tests...$(NC)"
	@cd dev-env/typescript && $(DEVBOX) run test

.PHONY: test-rust
test-rust:
	@echo "$(BLUE)🦀 Running Rust tests...$(NC)"
	@cd dev-env/rust && $(DEVBOX) run test

.PHONY: test-go
test-go:
	@echo "$(BLUE)🐹 Running Go tests...$(NC)"
	@cd dev-env/go && $(DEVBOX) run test

.PHONY: test-nushell
test-nushell:
	@echo "$(BLUE)🐚 Running Nushell tests...$(NC)"
	@cd dev-env/nushell && $(DEVBOX) run test

# ============================================================================
# Code Quality
# ============================================================================

.PHONY: format
format:
	@echo "$(BLUE)🎨 Formatting code in all environments...$(NC)"
	@cd dev-env/python && $(DEVBOX) run format
	@cd dev-env/typescript && $(DEVBOX) run format
	@cd dev-env/rust && $(DEVBOX) run format
	@cd dev-env/go && $(DEVBOX) run format
	@cd dev-env/nushell && $(DEVBOX) run format

.PHONY: lint
lint:
	@echo "$(BLUE)🔍 Linting code in all environments...$(NC)"
	@cd dev-env/python && $(DEVBOX) run lint
	@cd dev-env/typescript && $(DEVBOX) run lint
	@cd dev-env/rust && $(DEVBOX) run lint
	@cd dev-env/go && $(DEVBOX) run lint
	@cd dev-env/nushell && $(DEVBOX) run check

.PHONY: clean
clean:
	@echo "$(YELLOW)🧹 Cleaning all environments...$(NC)"
	@$(NU) dev-env/nushell/scripts/deploy.nu clean-all

.PHONY: deps-update
deps-update:
	@echo "$(BLUE)📦 Updating dependencies...$(NC)"
	@cd dev-env/python && $(DEVBOX) update
	@cd dev-env/typescript && $(DEVBOX) update
	@cd dev-env/rust && $(DEVBOX) update
	@cd dev-env/go && $(DEVBOX) update
	@cd dev-env/nushell && $(DEVBOX) update

# ============================================================================
# Monitoring & Analytics
# ============================================================================

.PHONY: perf-dashboard
perf-dashboard:
	@echo "$(BLUE)📊 Showing performance dashboard...$(NC)"
	@$(NU) dev-env/nushell/scripts/performance-analytics.nu dashboard

.PHONY: perf-report
perf-report:
	@echo "$(BLUE)📈 Generating performance report...$(NC)"
	@$(NU) dev-env/nushell/scripts/performance-analytics.nu report --days 7

.PHONY: security-scan
security-scan:
	@echo "$(BLUE)🔒 Running security scan...$(NC)"
	@$(NU) dev-env/nushell/scripts/security-scanner.nu scan-all

.PHONY: resource-monitor
resource-monitor:
	@echo "$(BLUE)📊 Monitoring system resources...$(NC)"
	@$(NU) dev-env/nushell/scripts/resource-monitor.nu watch --interval 30

# ============================================================================
# Utilities
# ============================================================================

.PHONY: env-status
env-status:
	@echo "$(BLUE)📋 Environment status:$(NC)"
	@$(NU) dev-env/nushell/scripts/list.nu

.PHONY: install-hooks
install-hooks:
	@echo "$(BLUE)🪝 Installing Claude Code hooks...$(NC)"
	@./.claude/install-hooks.sh

.PHONY: update-docs
update-docs:
	@echo "$(BLUE)📚 Updating documentation...$(NC)"
	@$(NU) dev-env/nushell/scripts/docs-automation.nu generate

.PHONY: polyglot-check
polyglot-check:
	@echo "$(BLUE)🔍 Running polyglot health check...$(NC)"
	@$(NU) -c "use dev-env/nushell/common.nu *; polyglot-check"

.PHONY: logs
logs:
	@echo "$(BLUE)📝 Showing recent logs...$(NC)"
	@tail -f ~/.claude/notifications.log

# ============================================================================
# Development Shortcuts
# ============================================================================

.PHONY: dev-python
dev-python:
	@echo "$(BLUE)🐍 Entering Python development environment...$(NC)"
	@cd dev-env/python && $(DEVBOX) shell

.PHONY: dev-typescript
dev-typescript:
	@echo "$(BLUE)📘 Entering TypeScript development environment...$(NC)"
	@cd dev-env/typescript && $(DEVBOX) shell

.PHONY: dev-rust
dev-rust:
	@echo "$(BLUE)🦀 Entering Rust development environment...$(NC)"
	@cd dev-env/rust && $(DEVBOX) shell

.PHONY: dev-go
dev-go:
	@echo "$(BLUE)🐹 Entering Go development environment...$(NC)"
	@cd dev-env/go && $(DEVBOX) shell

.PHONY: dev-nushell
dev-nushell:
	@echo "$(BLUE)🐚 Entering Nushell development environment...$(NC)"
	@cd dev-env/nushell && $(DEVBOX) shell

# ============================================================================
# CI/CD Targets
# ============================================================================

.PHONY: ci-test
ci-test: validate test lint
	@echo "$(GREEN)✅ All CI tests passed$(NC)"

.PHONY: pre-commit
pre-commit: format lint test-nushell
	@echo "$(GREEN)✅ Pre-commit checks passed$(NC)"

.PHONY: release-check
release-check: clean setup validate mcp-test
	@echo "$(GREEN)✅ Release checks passed$(NC)"

# ============================================================================
# Emergency & Recovery
# ============================================================================

.PHONY: emergency-reset
emergency-reset:
	@echo "$(RED)🚨 Emergency reset - this will clean everything!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or wait 5 seconds to continue...$(NC)"
	@sleep 5
	@$(NU) dev-env/nushell/scripts/deploy.nu emergency-reset

.PHONY: backup
backup:
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	@$(NU) dev-env/nushell/scripts/deploy.nu backup

# ============================================================================
# Help for specific topics
# ============================================================================

.PHONY: help-setup
help-setup:
	@echo "$(CYAN)Setup Help$(NC)"
	@echo "$(YELLOW)==========$(NC)"
	@echo "1. Install Nushell: See README.md for OS-specific instructions"
	@echo "2. Run: make setup"
	@echo "3. Verify: make validate"
	@echo "4. Start MCP: make mcp-start"

.PHONY: help-devpod
help-devpod:
	@echo "$(CYAN)DevPod Help$(NC)"
	@echo "$(YELLOW)===========$(NC)"
	@echo "Create workspaces: make devpod-python COUNT=3"
	@echo "List workspaces:   make devpod-list"
	@echo "Clean workspaces:  make devpod-clean"

.PHONY: help-mcp
help-mcp:
	@echo "$(CYAN)MCP Server Help$(NC)"
	@echo "$(YELLOW)===============$(NC)"
	@echo "Build:  make mcp-build"
	@echo "Start:  make mcp-start"
	@echo "Test:   make mcp-test"
	@echo "Watch:  make mcp-watch"

# Make sure intermediate files are not deleted
.SECONDARY:

# Default goal
.DEFAULT_GOAL := help