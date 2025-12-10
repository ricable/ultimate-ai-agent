# Phase 4: Validation Prompt

## Objective
Thoroughly test and validate all implementations before deployment.

---

## Prompt Template

```
You are a QA engineer validating an AI agent platform implementation.

## Context
The implementation phase is complete. We need to validate:
- Functionality against requirements
- Performance under load
- Security vulnerabilities
- Integration correctness

## Your Task
Perform comprehensive validation:

### 1. Functional Testing
- Verify all acceptance criteria are met
- Test edge cases and error conditions
- Validate input handling and sanitization
- Check cross-browser/platform compatibility (if applicable)

### 2. Performance Testing
- Benchmark critical operations
- Load test with realistic traffic patterns
- Identify bottlenecks and memory leaks
- Measure latency percentiles (p50, p95, p99)

### 3. Security Testing
- Review authentication/authorization
- Test for common vulnerabilities (OWASP Top 10)
- Validate input sanitization
- Check secrets management

### 4. Integration Testing
- Test all API endpoints
- Verify MCP protocol compliance
- Test database operations
- Validate external service integrations

## Output Format
1. **Test Results Summary** (pass/fail counts)
2. **Issues Found** (severity, description, reproduction)
3. **Performance Metrics** (benchmarks, graphs)
4. **Security Report** (findings, recommendations)
5. **Sign-off Recommendation** (go/no-go)
```

---

## Test Suites

### Unit Test Suite

```typescript
// tests/unit/agent.test.ts
describe('Agent', () => {
  describe('constructor', () => {
    it('should initialize with valid config', () => {
      const agent = new Agent(validConfig);
      expect(agent.id).toBeDefined();
      expect(agent.status).toBe('idle');
    });

    it('should throw on invalid config', () => {
      expect(() => new Agent(invalidConfig)).toThrow(ConfigError);
    });
  });

  describe('execute', () => {
    it('should complete task successfully', async () => {
      const result = await agent.execute(task);
      expect(result.status).toBe('completed');
    });

    it('should handle timeout', async () => {
      const result = await agent.execute(longTask, { timeout: 100 });
      expect(result.status).toBe('timeout');
    });

    it('should retry on transient errors', async () => {
      mockProvider.failOnce();
      const result = await agent.execute(task);
      expect(result.status).toBe('completed');
      expect(mockProvider.callCount).toBe(2);
    });
  });
});
```

### Integration Test Suite

```typescript
// tests/integration/api.test.ts
describe('API Integration', () => {
  let app: Express;

  beforeAll(async () => {
    app = await createTestApp();
  });

  describe('POST /agents', () => {
    it('should create agent', async () => {
      const response = await request(app)
        .post('/agents')
        .send({ role: 'researcher' })
        .expect(201);

      expect(response.body.id).toBeDefined();
    });

    it('should require authentication', async () => {
      await request(app)
        .post('/agents')
        .send({ role: 'researcher' })
        .expect(401);
    });
  });

  describe('GET /agents/:id/execute', () => {
    it('should execute task', async () => {
      const response = await request(app)
        .post(`/agents/${agentId}/execute`)
        .set('Authorization', `Bearer ${token}`)
        .send({ task: 'Research AI trends' })
        .expect(200);

      expect(response.body.result).toBeDefined();
    });
  });
});
```

### Performance Test Suite

```typescript
// tests/performance/load.test.ts
import { check, sleep } from 'k6';
import http from 'k6/http';

export const options = {
  stages: [
    { duration: '30s', target: 20 },   // Ramp up
    { duration: '1m', target: 100 },   // Peak load
    { duration: '30s', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% under 500ms
    http_req_failed: ['rate<0.01'],    // Error rate under 1%
  },
};

export default function () {
  const response = http.post(
    'http://localhost:8080/agents/execute',
    JSON.stringify({ task: 'Simple query' }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

### Security Test Checklist

```markdown
## Authentication & Authorization
- [ ] JWT tokens expire appropriately
- [ ] Refresh token rotation works
- [ ] Role-based access control enforced
- [ ] API keys can be revoked

## Input Validation
- [ ] SQL injection prevented
- [ ] XSS attacks mitigated
- [ ] Command injection blocked
- [ ] Path traversal prevented

## Data Protection
- [ ] Sensitive data encrypted at rest
- [ ] TLS enforced for all connections
- [ ] PII properly handled
- [ ] Secrets not logged

## Rate Limiting
- [ ] Per-user rate limits enforced
- [ ] Per-endpoint rate limits work
- [ ] Graceful degradation under load
- [ ] DDoS protection in place
```

## Validation Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Unit Test Coverage | > 80% | | |
| Integration Test Pass Rate | 100% | | |
| P95 Latency | < 500ms | | |
| P99 Latency | < 1000ms | | |
| Error Rate | < 0.1% | | |
| Security Vulnerabilities | 0 critical | | |

## Issue Template

```markdown
## Issue: [Title]

### Severity
- [ ] Critical - Blocks release
- [ ] High - Must fix before release
- [ ] Medium - Should fix before release
- [ ] Low - Can fix after release

### Description
[Detailed description of the issue]

### Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Environment
- OS:
- Node version:
- Package versions:

### Screenshots/Logs
[If applicable]
```

## Next Phase
Once validation is complete, proceed to [05-deployment.md](05-deployment.md) for production deployment.
