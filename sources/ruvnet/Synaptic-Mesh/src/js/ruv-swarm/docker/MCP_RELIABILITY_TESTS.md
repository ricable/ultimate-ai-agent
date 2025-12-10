# MCP Reliability Test Environment

This Docker-based test environment is designed to reproduce and test the MCP server reliability issues reported in issue #65.

## Overview

The test environment simulates various failure scenarios to identify when and why MCP tools become unavailable during Claude Code sessions:

- **Connection stability** over long-running sessions
- **Reconnection resilience** after network failures
- **High load conditions** with parallel requests
- **Network chaos** (latency, packet loss, disconnections)
- **Resource constraints** and memory pressure

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Claude Simulator   │────▶│   MCP Server     │────▶│   Monitoring    │
│  (Client behavior)  │     │  (ruv-swarm)     │     │  (Prometheus)   │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
         │                           │                         │
         │                           │                         ▼
         │                    ┌──────────────────┐     ┌─────────────────┐
         │                    │  Network Chaos   │     │    Grafana      │
         └───────────────────▶│   (Pumba)        │     │  (Dashboards)   │
                              └──────────────────┘     └─────────────────┘
```

## Components

### 1. MCP Server (`mcp-server`)
- The ruv-swarm MCP server under test
- Configured with debug logging and health checks
- Exposes metrics for monitoring

### 2. Claude Simulator (`claude-simulator`)
- Simulates Claude Code client behavior
- Executes test scenarios automatically
- Tracks connection metrics and failures

### 3. Network Chaos (`network-chaos`)
- Uses Pumba to inject network failures
- Simulates:
  - Network latency (100ms ± 50ms)
  - Packet loss (5%)
  - Connection drops
  - Container pauses

### 4. Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Loki**: Log aggregation
- **Promtail**: Log forwarding

### 5. Test Runner (`test-runner`)
- Orchestrates test scenarios
- Collects results and generates reports
- Validates expected behavior

## Test Scenarios

### 1. Connection Stability (`connection-stability.json`)
- Establishes connection and maintains it for 1 hour
- Performs regular health checks
- Monitors for silent failures

### 2. Reconnection Resilience (`reconnection-resilience.json`)
- Tests reconnection after various disconnection types
- Verifies state persistence
- Multiple disconnect/reconnect cycles

### 3. High Load Stress (`high-load-stress.json`)
- Spawns many agents in parallel
- Executes concurrent tasks
- Rapid-fire status checks

### 4. Network Chaos (`network-chaos.json`)
- Tests under degraded network conditions
- Verifies behavior during packet loss
- Handles connection timeouts

## Usage

### Quick Start

```bash
# Run with default settings (1 hour test)
./run-mcp-reliability-tests.sh

# Run for 30 minutes
./run-mcp-reliability-tests.sh --duration 1800

# Keep services running after tests
./run-mcp-reliability-tests.sh --no-cleanup
```

### Manual Testing

```bash
# Start all services
docker-compose -f docker-compose.mcp-reliability.yml up -d

# View logs
docker-compose -f docker-compose.mcp-reliability.yml logs -f mcp-server
docker-compose -f docker-compose.mcp-reliability.yml logs -f claude-simulator

# Access services
# - MCP Server: http://localhost:3001
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3002 (admin/admin)

# Stop services
docker-compose -f docker-compose.mcp-reliability.yml down -v
```

### Monitoring

Access Grafana at http://localhost:3002 to view:
- MCP connection metrics
- Request rates and latencies
- Disconnection reasons
- Reconnection attempts
- Session durations

## Key Metrics

The following metrics help identify reliability issues:

1. **`mcp_connections_total`**: Total connection attempts (by status)
2. **`mcp_disconnections_total`**: Disconnections (by reason)
3. **`mcp_reconnect_attempts_total`**: Reconnection attempts
4. **`mcp_requests_total`**: Request counts (by method/status)
5. **`mcp_request_duration_seconds`**: Request latencies
6. **`mcp_session_duration_seconds`**: Session lifetimes

## Reproducing Issue #65

To reproduce the specific issue where MCP tools become unavailable:

1. Start the test environment
2. Monitor the Claude simulator logs for successful initial connections
3. Watch for disconnection events in the logs
4. Check if reconnection attempts succeed
5. Verify if tools remain available after reconnection

Look for patterns like:
- Silent WebSocket closures without proper error handling
- Failed reconnection attempts that don't retry
- State loss after reconnection
- Memory leaks causing server instability

## Test Results

Results are saved in:
- `test-results/`: Raw test output and metrics
- `logs/`: Application logs from all containers
- `reports/`: Generated test reports (JSON and Markdown)

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose -f docker-compose.mcp-reliability.yml logs <service-name>

# Rebuild images
docker-compose -f docker-compose.mcp-reliability.yml build --no-cache
```

### Can't connect to services
```bash
# Check if ports are already in use
netstat -tulpn | grep -E '3001|9090|3002|3100'

# Use different ports in docker-compose.yml
```

### Out of memory
```bash
# Increase Docker memory limits
# Edit docker-compose.yml mem_limit values
```

## Contributing

To add new test scenarios:

1. Create a new JSON file in `scenarios/`
2. Define the test steps using available actions
3. Add the scenario to the test suite
4. Update documentation

## Next Steps

Based on test results, potential fixes include:

1. **Implement proper WebSocket reconnection logic**
   - Exponential backoff
   - State preservation
   - Event emission for connection status

2. **Add connection health checks**
   - Periodic ping/pong
   - Timeout detection
   - Automatic recovery

3. **Improve error handling**
   - Graceful degradation
   - Clear error messages
   - Recovery strategies

4. **Add connection pooling**
   - Multiple connection attempts
   - Load balancing
   - Failover support