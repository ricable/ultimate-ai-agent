# Polyglot DevPod Functional Test Suite

Comprehensive functional testing infrastructure for the polyglot-dev MCP server, testing all 64+ MCP tools across 10 DevPod environments with full integration validation.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run full functional test suite
npm run test

# Run specific test suite
npm run test:devpod
npm run test:mcp-tools
npm run test:agentic
npm run test:performance

# Run with specific configuration
npm run test:parallel
npm run test:sequential
```

## 📋 Test Coverage

### Test Suites

1. **DevPod Swarm Tests** (`devpod-swarm-tests.ts`)
   - DevPod environment provisioning (10 environments)
   - Tool validation and connectivity
   - Claude-Flow integration
   - .claude/ auto-installation
   - Resource monitoring

2. **Environment-Specific Tests** (`environment-specific-tests.ts`)
   - Deep validation of language-specific tools
   - Package manager verification
   - Build/test/lint command availability
   - Minimal project setup validation

3. **MCP Tool Matrix Tests** (`mcp-tool-matrix-tests.ts`)
   - 64+ MCP tools across 12 categories
   - Parameter validation and execution
   - Environment compatibility testing
   - Performance benchmarking

4. **AI Integration Tests** (`ai-integration-tests.ts`)
   - Claude commands and slash commands
   - AI hooks (auto-format, auto-test, context engineering)
   - Claude-Flow CLI integration
   - RUV-Swarm CLI functionality
   - Enhanced AI hooks system

5. **Agentic Environment Tests** (`agentic-environment-tests.ts`)
   - Agentic environment validation
   - AG-UI protocol features
   - CopilotKit integration
   - Cross-environment agent communication
   - Agent lifecycle management

6. **Performance and Load Tests** (`performance-load-tests.ts`)
   - DevPod provisioning performance
   - MCP tool execution benchmarks
   - Concurrent operation load testing
   - Resource utilization analysis
   - Scaling tests

### Environments Tested

#### Standard Environments (5)
- **Python**: Python 3.12, uv, ruff, mypy, pytest, FastAPI
- **TypeScript**: Node.js 20, npm, TypeScript, ESLint, Jest, Next.js
- **Rust**: rustc, cargo, clippy, rustfmt, tokio, serde
- **Go**: Go 1.22, gofmt, golangci-lint, gin, gorm
- **Nushell**: Nushell 0.105.1, git, teller, pipeline processing

#### Agentic Environments (5)
- **agentic-python**: Python + FastAPI + async agents + CopilotKit + Claude-Flow
- **agentic-typescript**: TypeScript + Next.js + CopilotKit + AG-UI protocol
- **agentic-rust**: Rust + Tokio + high-performance agents + AG-UI support
- **agentic-go**: Go + HTTP server + agent middleware + microservices
- **agentic-nushell**: Nushell + pipeline agents + automation scripting

### MCP Tools Tested (64+ tools)

#### Categories
- **Environment Tools** (3): environment_detect, environment_info, environment_validate
- **DevBox Tools** (6): devbox_shell, devbox_start, devbox_run, devbox_status, devbox_add_package, devbox_quick_start
- **DevPod Tools** (4): devpod_provision, devpod_list, devpod_status, devpod_start
- **Cross-Language Tools** (3): polyglot_check, polyglot_validate, polyglot_clean
- **Performance Tools** (2): performance_measure, performance_report
- **Security Tools** (1): security_scan
- **Hook Tools** (2): hook_status, hook_trigger
- **PRP Tools** (2): prp_generate, prp_execute
- **AG-UI Tools** (9): agui_provision, agui_agent_create, agui_agent_list, agui_agent_invoke, agui_chat, agui_generate_ui, agui_shared_state, agui_status, agui_workflow
- **Claude-Flow Tools** (10): claude_flow_init, claude_flow_wizard, claude_flow_start, claude_flow_stop, claude_flow_status, claude_flow_monitor, claude_flow_spawn, claude_flow_logs, claude_flow_hive_mind, claude_flow_terminal_mgmt
- **Enhanced AI Hooks Tools** (8): enhanced_hook_context_triggers, enhanced_hook_error_resolution, enhanced_hook_env_orchestration, enhanced_hook_dependency_tracking, enhanced_hook_performance_integration, enhanced_hook_quality_gates, enhanced_hook_devpod_manager, enhanced_hook_prp_lifecycle
- **Docker MCP Tools** (15): docker_mcp_gateway_start, docker_mcp_gateway_status, docker_mcp_tools_list, docker_mcp_http_bridge, docker_mcp_client_list, docker_mcp_server_list, docker_mcp_gemini_config, docker_mcp_test, docker_mcp_demo, docker_mcp_security_scan, docker_mcp_resource_limits, docker_mcp_network_isolation, docker_mcp_signature_verify, docker_mcp_logs, docker_mcp_cleanup

## 🏗️ Architecture

### Test Execution Flow

```
1. Global Setup (jest.global-setup.js)
   ├── Verify required tools (DevPod, Nushell, Docker)
   ├── Check centralized DevPod management script
   ├── Clean existing test workspaces
   └── Initialize test directories

2. Test Sequencer (jest.test-sequencer.js)
   ├── Order tests by priority and dependencies
   ├── Handle dependency resolution
   └── Break circular dependencies

3. Test Execution (6 test suites)
   ├── DevPod Swarm Tests (foundation)
   ├── Environment-Specific Tests
   ├── MCP Tool Matrix Tests
   ├── AI Integration Tests
   ├── Agentic Environment Tests
   └── Performance and Load Tests

4. Global Teardown (jest.global-teardown.js)
   ├── Clean up all test workspaces
   ├── Remove test containers and volumes
   ├── Docker system cleanup
   └── Generate cleanup summary
```

### Helper Utilities

#### DevPodManager (`test-helpers.ts`)
- Workspace lifecycle management
- Health checks and validation
- Resource usage monitoring
- Centralized cleanup operations

#### TestDataGenerator
- Generate workspace configurations
- Create validation commands
- Generate test files for environments

#### PerformanceMeasurer
- Track operation durations
- Generate performance reports
- Benchmark comparisons

#### TestContextManager
- Manage test isolation
- Track workspace associations
- Coordinate cleanup operations

## 🔧 Configuration

### Test Configuration (`jest.config.js`)
```javascript
{
  maxWorkers: '50%',           // Use 50% of CPU cores
  testTimeout: 300000,         // 5 minutes default
  globalTimeout: 3600000,      // 1 hour total
  maxConcurrentSuites: 3,      // Parallel test suites
  maxWorkspaces: 15,           // DevPod workspace limit
  cleanupBetweenSuites: true   // Clean between suites
}
```

### Environment Variables
```bash
# Test configuration
FUNCTIONAL_TEST_MAX_WORKSPACES=15
FUNCTIONAL_TEST_CLEANUP_ENABLED=true
FUNCTIONAL_TEST_PERFORMANCE_TRACKING=true

# DevPod configuration
DEVPOD_WORKSPACE_PREFIX=functional-test
DEVPOD_TIMEOUT=600000

# Performance thresholds
DEVPOD_PROVISIONING_MAX_TIME=300000
MCP_TOOL_EXECUTION_MAX_TIME=30000
```

## 📊 Performance Benchmarks

### Expected Performance
- **DevPod Provisioning**: < 5 minutes per environment
- **MCP Tool Execution**: < 30 seconds per tool
- **Environment Switching**: < 10 seconds
- **Memory per Container**: < 2GB
- **CPU per Container**: < 2 cores
- **Max Concurrent Containers**: 15

### Load Testing Scenarios
1. **Rapid Provisioning**: 5 concurrent environments in 1 minute
2. **MCP Tool Stress**: 10 concurrent tools for 2 minutes
3. **Agentic Environment Load**: 3 concurrent agentic environments for 3 minutes
4. **Cross-Environment Coordination**: 8 concurrent operations for 1.5 minutes

## 🚀 Usage Examples

### Run Full Test Suite
```bash
# Complete functional validation
npm run test

# With custom configuration
npm run test -- --maxWorkers=2 --testTimeout=600000
```

### Run Specific Test Categories
```bash
# Core DevPod functionality
npm run test:devpod

# MCP tool validation
npm run test:mcp-tools

# Agentic features
npm run test:agentic

# Performance benchmarks
npm run test:performance

# AI integration features
npm run test:ai
```

### Development and Debug Mode
```bash
# Run single test file
npm run test devpod-swarm-tests.ts

# Run with verbose output
npm run test:verbose

# Run with coverage
npm run test:coverage

# Run specific environment tests
npm run test -- --testNamePattern="python"
```

### Continuous Integration
```bash
# CI pipeline execution
npm run test:ci

# Generate reports
npm run test:report

# Cleanup only
npm run cleanup
```

## 📈 Reporting

### Test Reports Generated
1. **HTML Report**: `./test-reports/functional-test-report.html`
2. **JUnit XML**: `./test-reports/functional-test-results.xml`
3. **Coverage Report**: `./coverage/lcov-report/index.html`
4. **Performance Report**: Console output with detailed metrics

### Report Contents
- Test execution summary
- Environment validation results
- MCP tool performance metrics
- Resource utilization analysis
- Workspace provisioning statistics
- Integration test results

## 🛠️ Troubleshooting

### Common Issues

#### DevPod Not Found
```bash
# Install DevPod
curl -fsSL https://get.jetify.com/devbox | bash
```

#### Workspace Provisioning Failures
```bash
# Check DevPod status
devpod list
devpod version

# Clean up failed workspaces
devpod list | grep test | xargs devpod delete --force
```

#### Memory/Resource Issues
```bash
# Clean up Docker resources
docker system prune -f --volumes

# Check resource usage
docker stats

# Adjust test configuration
export FUNCTIONAL_TEST_MAX_WORKSPACES=10
```

#### Test Timeouts
```bash
# Increase timeouts
npm run test -- --testTimeout=900000

# Run tests sequentially
npm run test:sequential
```

### Debug Commands
```bash
# Check centralized DevPod management
nu ../../host-tooling/devpod-management/manage-devpod.nu help

# Validate environment tools
devpod --version
nu --version
docker --version

# Manual workspace provisioning test
nu ../../host-tooling/devpod-management/manage-devpod.nu provision python

# Clean up test resources
npm run cleanup
```

## 🔒 Security Considerations

### Test Isolation
- Each test runs in isolated DevPod containers
- Workspaces are uniquely named with timestamps
- Network isolation between test environments
- Automatic cleanup prevents resource leaks

### Resource Limits
- Memory limits enforced per container (2GB)
- CPU limits enforced per container (2 cores)
- Maximum concurrent workspace limits (15)
- Timeout protection for long-running operations

### Cleanup Procedures
- Automatic workspace cleanup after tests
- Docker container and volume cleanup
- Temporary file cleanup
- Test artifact removal

## 📚 Dependencies

### Required Tools
- **DevPod**: Container development environments
- **Nushell**: Shell scripting and automation
- **Docker**: Container runtime
- **Node.js**: JavaScript runtime (v20+)
- **Git**: Version control

### NPM Dependencies
- **Jest**: Testing framework
- **TypeScript**: Type safety
- **ts-jest**: TypeScript Jest transformer
- **jest-html-reporters**: HTML test reports
- **jest-junit**: JUnit XML reports

### Development Dependencies
- **@types/jest**: TypeScript definitions
- **@types/node**: Node.js TypeScript definitions

## 🤝 Contributing

### Adding New Tests
1. Create test file in appropriate category
2. Follow naming convention: `*-tests.ts`
3. Use helper utilities from `test-helpers.ts`
4. Add to test sequencer if dependencies exist
5. Update documentation

### Performance Testing
1. Use `PerformanceMeasurer` for timing
2. Validate against performance thresholds
3. Include resource usage monitoring
4. Document expected performance

### Environment Testing
1. Use `DevPodManager` for workspace management
2. Include health checks
3. Test both standard and agentic variants
4. Validate tool availability and functionality

---

*Comprehensive functional testing infrastructure for the polyglot-dev MCP server with full DevPod integration and intelligent automation.*