# MCP Issue #65 Fixes - Docker Test Guide

## Overview

This Docker test suite validates the fixes for MCP Issue #65, which addresses:
1. Logger output contaminating stdio communication
2. Database persistence issues
3. Error handling improvements
4. Concurrent operation stability

## Test Components

### 1. Docker Images

#### `Dockerfile.mcp-stdio-test`
- Base image for all MCP stdio mode tests
- Includes SQLite for database testing
- Configured with proper environment variables for stdio mode

### 2. Test Scripts

#### `test-mcp-stdio.js`
Main test script that validates:
- Logger stderr output separation
- Database persistence for all tables
- MCP stdio communication
- Error handling
- Concurrent operations

#### `run-mcp-fixes-tests.sh`
Shell script to orchestrate the entire test suite:
```bash
./docker/run-mcp-fixes-tests.sh
```

### 3. Docker Compose Configuration

#### `docker-compose.mcp-fixes-test.yml`
Defines multiple test services:
- **mcp-stdio-test**: Main stdio mode validation
- **db-persistence-test**: Database table verification
- **logger-test**: Output stream separation
- **error-handling-test**: Error condition handling
- **concurrent-test**: Concurrent operation testing
- **integration-test**: Full end-to-end validation
- **report-generator**: Consolidates test results

## Running Tests

### Quick Start
```bash
cd /workspaces/ruv-FANN/ruv-swarm/npm
./docker/run-mcp-fixes-tests.sh
```

### Individual Tests
```bash
# Run specific test container
docker-compose -f docker/docker-compose.mcp-fixes-test.yml run mcp-stdio-test

# View logs
docker-compose -f docker/docker-compose.mcp-fixes-test.yml logs logger-test

# Interactive debugging
docker-compose -f docker/docker-compose.mcp-fixes-test.yml run --entrypoint /bin/sh mcp-stdio-test
```

### Manual Testing
```bash
# Build the test image
docker build -f docker/Dockerfile.mcp-stdio-test -t ruv-swarm-mcp-test .

# Run with stdio mode
docker run -e MCP_MODE=stdio -it ruv-swarm-mcp-test node bin/ruv-swarm-clean.js mcp start

# Test with piped input
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  docker run -e MCP_MODE=stdio -i ruv-swarm-mcp-test node bin/ruv-swarm-clean.js mcp start
```

## Test Validation Points

### 1. Logger Output Separation
- ✅ Stdout contains only JSON-RPC messages
- ✅ All logs directed to stderr
- ✅ No contamination of MCP communication

### 2. Database Persistence
Tables verified:
- ✅ swarms
- ✅ agents  
- ✅ tasks
- ✅ neural_states
- ✅ memory
- ✅ daa_agents

### 3. Error Handling
- ✅ Invalid JSON handled gracefully
- ✅ Missing fields return proper errors
- ✅ Database errors don't crash server
- ✅ Concurrent writes handled properly

### 4. MCP Protocol
- ✅ Initialize method works
- ✅ Tool listing returns all tools
- ✅ Tool calls execute successfully
- ✅ Responses properly formatted

## Test Results

Results are saved in multiple locations:
- `docker/test-results/` - Individual test outputs
- `docker/reports/` - Consolidated reports
- `docker/logs/` - Container logs

### Report Files
- `consolidated-test-report.json` - Machine-readable results
- `mcp-fixes-test-report.md` - Human-readable summary
- `mcp-stdio-test-*.json` - Detailed test results

## Debugging Failed Tests

### Check Container Logs
```bash
# View all logs
docker-compose -f docker/docker-compose.mcp-fixes-test.yml logs

# View specific service
docker-compose -f docker/docker-compose.mcp-fixes-test.yml logs mcp-stdio-test
```

### Inspect Database
```bash
# Run SQLite shell
docker run -v $(pwd)/data:/data -it alpine:latest \
  sh -c "apk add sqlite && sqlite3 /data/ruv-swarm.db"

# Check tables
.tables
SELECT * FROM swarms;
SELECT * FROM agents;
SELECT * FROM memory;
```

### Test Stdout/Stderr Separation
```bash
# Run test capturing streams separately
docker run -e MCP_MODE=stdio ruv-swarm-mcp-test \
  node bin/ruv-swarm-clean.js mcp start \
  1>stdout.log 2>stderr.log

# Verify stdout is clean
cat stdout.log | jq .  # Should only show JSON-RPC

# Verify logs in stderr
grep -i "info\|debug\|error" stderr.log
```

## Common Issues

### 1. Database Lock Errors
- Ensure only one process accesses the database
- Check file permissions on data directory

### 2. Container Build Failures
- Verify Docker daemon is running
- Check available disk space
- Clear Docker cache: `docker system prune`

### 3. Test Timeouts
- Increase timeout values in test scripts
- Check system resource availability
- Verify network connectivity

## CI/CD Integration

Add to GitHub Actions:
```yaml
- name: Run MCP Fixes Tests
  run: |
    cd ruv-swarm/npm
    ./docker/run-mcp-fixes-tests.sh
  
- name: Upload Test Results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: mcp-test-results
    path: |
      ruv-swarm/npm/docker/test-results/
      ruv-swarm/npm/docker/reports/
```

## Next Steps

1. Monitor test results over multiple runs
2. Add performance benchmarks
3. Implement stress testing scenarios
4. Create automated regression tests