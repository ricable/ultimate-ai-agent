# Federation Multi-Agent Collaboration Test

## Overview

This test simulates **5 agents collaborating** through a shared **Federation Hub** using Docker containers. It validates:

- ✅ Multi-agent memory sharing
- ✅ Tenant isolation
- ✅ Real-time synchronization
- ✅ WebSocket-based communication
- ✅ Episode storage and retrieval

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Federation Hub                     │
│              (WebSocket Server)                     │
│            Port: 8443                               │
└────────────┬────────────────────────────────────────┘
             │
      ┌──────┴──────┬──────────┬──────────┬──────────┐
      │             │          │          │          │
┌─────▼────┐  ┌────▼─────┐ ┌──▼──────┐ ┌▼─────────┐ ┌▼──────────┐
│Researcher│  │  Coder   │ │ Tester  │ │ Reviewer │ │  Isolated │
│  Agent   │  │  Agent   │ │  Agent  │ │  Agent   │ │   Agent   │
└──────────┘  └──────────┘ └─────────┘ └──────────┘ └───────────┘
   Tenant: test-collaboration              Tenant: different-tenant
```

## Components

### 1. Federation Hub (`federation-hub`)
- **Role**: Central memory synchronization server
- **Port**: 8443 (WebSocket)
- **Database**: SQLite at `/data/hub.db`
- **Features**:
  - Tenant-isolated episode storage
  - Vector clock sync
  - Change log tracking
  - Real-time broadcasting

### 2. Researcher Agent (`agent-researcher`)
- **Role**: Find patterns and insights
- **Task**: Analyze codebase for collaboration opportunities
- **Output**: Pattern analysis reports

### 3. Coder Agent (`agent-coder`)
- **Role**: Implement solutions
- **Task**: Build features based on researcher findings
- **Output**: Code implementations

### 4. Tester Agent (`agent-tester`)
- **Role**: Validate work
- **Task**: Test implementations
- **Output**: Test results and coverage reports

### 5. Reviewer Agent (`agent-reviewer`)
- **Role**: Quality assurance
- **Task**: Review code quality and security
- **Output**: Quality reports and recommendations

### 6. Isolated Agent (`agent-isolated`)
- **Role**: Tenant isolation validation
- **Tenant**: `different-tenant` (separate from others)
- **Expected**: Cannot access other tenant's memories

### 7. Monitor Dashboard (`monitor`)
- **Role**: Real-time visualization
- **Port**: 3000 (HTTP)
- **URL**: http://localhost:3000
- **Features**:
  - Connected agents count
  - Total episodes stored
  - Active tenants
  - Real-time activity log

## Running the Test

### Quick Start

```bash
cd docker/federation-test
./run-test.sh
```

### Manual Steps

```bash
# 1. Build project
npm run build

# 2. Build Docker images
cd docker/federation-test
docker-compose build

# 3. Start all services
docker-compose up

# 4. Open monitor dashboard
open http://localhost:3000

# 5. Stop test (Ctrl+C)
docker-compose down
```

## Expected Test Flow

### Phase 1: Connection (0-10s)
```
[Hub] Federation Hub Server started on port 8443
[Hub] Agent authenticated: researcher-001 (tenant: test-collaboration)
[Hub] Agent authenticated: coder-001 (tenant: test-collaboration)
[Hub] Agent authenticated: tester-001 (tenant: test-collaboration)
[Hub] Agent authenticated: reviewer-001 (tenant: test-collaboration)
[Hub] Agent authenticated: isolated-001 (tenant: different-tenant)
```

### Phase 2: Collaboration (10-60s)
```
[researcher] Iteration 1: Identified optimal async/await patterns (reward: 0.87)
[coder] Iteration 1: Implemented async data fetcher (reward: 0.92)
[tester] Iteration 1: Unit tests: 45 passed (reward: 0.88)
[reviewer] Iteration 1: Code quality: A+ (reward: 0.94)
[isolated] Iteration 1: Found performance bottleneck (reward: 0.79)
```

### Phase 3: Memory Sharing
```
[Hub] Push completed: researcher-001 (1 episodes)
[Hub] Pull completed: coder-001 (1 episodes from researcher)
[Hub] Push completed: coder-001 (1 episodes)
[Hub] Pull completed: tester-001 (2 episodes from researcher+coder)
```

### Phase 4: Tenant Isolation Validation
```
[Hub] Tenant stats: test-collaboration (4 agents, 40 episodes)
[Hub] Tenant stats: different-tenant (1 agent, 10 episodes)
✓ No cross-tenant data leakage detected
```

## Validation Checklist

After running the test, verify:

- [ ] All 5 agents connected successfully
- [ ] Hub reports 2 distinct tenants
- [ ] `test-collaboration` tenant has 4 agents
- [ ] `different-tenant` has 1 agent
- [ ] Episodes stored in hub database
- [ ] No cross-tenant data access
- [ ] Monitor dashboard shows real-time updates
- [ ] Agents complete 10-12 iterations each
- [ ] Average reward > 0.75 for all agents
- [ ] No connection errors or timeouts

## Inspecting Results

### View Hub Database

```bash
# Access hub container
docker exec -it federation-hub sh

# Query database
sqlite3 /data/hub.db

# Check episodes
SELECT COUNT(*), tenant_id FROM episodes GROUP BY tenant_id;

# Check agents
SELECT * FROM agents;

# Exit
.exit
```

### View Agent Logs

```bash
# Researcher
docker logs agent-researcher

# Coder
docker logs agent-coder

# Tester
docker logs agent-tester

# Reviewer
docker logs agent-reviewer

# Isolated
docker logs agent-isolated
```

### Monitor Hub Stats

```bash
# Via HTTP endpoint
curl http://localhost:8444/stats

# Expected output:
{
  "connectedAgents": 5,
  "totalEpisodes": 50,
  "tenants": 2,
  "uptime": 60
}
```

## Performance Metrics

### Expected Latencies
- **Agent connection**: <100ms
- **Authentication**: <50ms
- **Memory sync (pull)**: <50ms
- **Memory sync (push)**: <100ms
- **Episode storage**: <20ms

### Expected Throughput
- **Sync rate**: 1 sync/5s per agent (0.2 Hz)
- **Total syncs**: ~60 syncs over 60s test
- **Episodes**: 50-60 total (10-12 per agent)

## Troubleshooting

### Hub Not Starting
```bash
# Check port 8443 is available
lsof -i :8443

# View hub logs
docker logs federation-hub
```

### Agents Not Connecting
```bash
# Check network connectivity
docker network inspect federation-test_federation-network

# Verify hub is ready
curl http://localhost:8444/health
```

### Database Issues
```bash
# Reset hub database
docker-compose down -v
docker-compose up
```

## Clean Up

```bash
# Stop all containers
docker-compose down

# Remove volumes (reset database)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Success Criteria

The test is considered **successful** if:

1. ✅ All 5 agents connect to hub within 10 seconds
2. ✅ Agents complete at least 10 iterations each
3. ✅ Hub stores 50+ episodes total
4. ✅ `test-collaboration` tenant has 40+ episodes
5. ✅ `different-tenant` has 10+ episodes (isolated)
6. ✅ No cross-tenant data access detected
7. ✅ Average sync latency <100ms
8. ✅ No connection errors or failures
9. ✅ Monitor dashboard shows real-time updates
10. ✅ All agents disconnect gracefully

## Next Steps

After successful test:

1. **Scale Test**: Increase to 10, 50, 100 agents
2. **Load Test**: Simulate high-frequency syncs
3. **Failure Test**: Test hub recovery and agent reconnection
4. **Multi-Region**: Deploy hubs in different regions
5. **Production**: Deploy to Kubernetes with real AgentDB

## Files

```
docker/federation-test/
├── README.md                    # This file
├── docker-compose.yml           # Service orchestration
├── Dockerfile.hub              # Hub server image
├── Dockerfile.agent            # Agent image
├── Dockerfile.monitor          # Monitor dashboard image
├── run-hub.ts                  # Hub server entrypoint
├── run-agent.ts                # Agent entrypoint
├── run-monitor.ts              # Monitor entrypoint
└── run-test.sh                 # Test execution script
```

---

**Built with ❤️ by [@ruvnet](https://github.com/ruvnet)**

**Test Duration**: ~60 seconds
**Agents**: 5 concurrent
**Shared Memories**: 50+ episodes
**Tenant Isolation**: Verified
