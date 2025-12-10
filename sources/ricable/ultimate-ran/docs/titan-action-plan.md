# TITAN RAN Platform - Action Plan

**Date:** 2025-12-06
**Status:** 68/100 (Conditional Go - Fix Critical Issues)
**Timeline:** 8 weeks to production readiness

---

## 🎯 MISSION

**Transform TITAN from 68/100 to 98/100 production-ready in 8 weeks by systematically resolving critical blockers, completing integration, and hardening the system.**

---

## 📅 WEEK-BY-WEEK PLAN

### WEEK 1 (Dec 6-13): CRITICAL BLOCKER FIXES

**Goal:** Restore functionality, eliminate P0 blockers → **Target: 75/100**

#### Day 1-2: GNN Uplink Optimizer Fix
**Owner:** ML Engineer
**Effort:** 2-4 hours

```typescript
// File: src/gnn/uplink-optimizer-v2.ts
// BEFORE (incorrect):
export default function UplinkOptimizer() { ... }

// AFTER (correct):
export class UplinkOptimizer {
  constructor() { ... }
}

// File: tests/gnn/uplink-optimizer.test.ts
import { UplinkOptimizer } from '../../src/gnn/uplink-optimizer-v2';
```

**Success Criteria:**
- ✅ 31 tests now passing (0% → 100%)
- ✅ All GNN functionality restored
- ✅ CI/CD green for uplink optimizer

#### Day 2-4: LLM Council API Integration (Part 1)
**Owner:** Senior Backend Engineer #1
**Effort:** 16-24 hours (across 3 days)

**Tasks:**
1. Install latest SDKs:
   ```bash
   npm install @anthropic-ai/sdk@latest @google/generative-ai@latest
   ```

2. Implement Anthropic API (Claude 3.7 Sonnet):
   ```typescript
   // File: src/council/providers/anthropic.ts
   import Anthropic from '@anthropic-ai/sdk';

   export async function invokeClaudeAnalyst(prompt: string) {
     const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
     const response = await client.messages.create({
       model: 'claude-3-sonnet-20250219',
       max_tokens: 4096,
       messages: [{ role: 'user', content: prompt }]
     });
     return response.content[0].text;
   }
   ```

3. Update Council Orchestrator:
   ```typescript
   // File: src/council/orchestrator.ts (line 200-250)
   async invokeCouncilMember(member: CouncilMember, context: string): Promise<DebateProposal> {
     let proposalContent: string;

     switch (member.model) {
       case 'claude-3-sonnet-20250219':
         proposalContent = await invokeClaudeAnalyst(context);
         break;
       case 'gemini-1.5-pro':
         proposalContent = await invokeGeminiHistorian(context);
         break;
       case 'deepseek-r1':
         proposalContent = await invokeDeepSeekStrategist(context);
         break;
     }

     return {
       member_id: member.id,
       content: proposalContent,
       timestamp: Date.now(),
       // ... rest of proposal
     };
   }
   ```

**Success Criteria:**
- ✅ Claude API integration working
- ✅ Gemini API integration working
- ✅ DeepSeek API integration working
- ✅ End-to-end debate protocol test passing

#### Day 3-4: AgentDB Persistence
**Owner:** Senior Backend Engineer #2
**Effort:** 8-12 hours

**Tasks:**
1. Initialize database:
   ```bash
   npm run db:status
   npm run db:train
   ```

2. Add storage hooks to agents:
   ```typescript
   // File: src/agents/guardian/index.ts
   async performSafetyCheck(artifact: any) {
     const result = await this.analyzeLyapunov(artifact);

     // STORE IN AGENTDB
     await agentdb.store({
       table: 'safety_checks',
       data: {
         artifact_id: artifact.id,
         lyapunov_exponent: result.lyapunov,
         verdict: result.safe ? 'APPROVED' : 'REJECTED',
         timestamp: Date.now()
       }
     });

     return result;
   }
   ```

3. Implement reflexion logging:
   ```typescript
   // File: src/cognitive/agentdb-client.js
   async storeReflexion(episode: Episode) {
     await this.db.insert('reflexion_log', {
       episode_id: episode.id,
       action: episode.action,
       observation: episode.observation,
       critique: this.generateCritique(episode),
       learned_constraint: this.extractConstraint(episode),
       embedding: await this.generateEmbedding(episode)
     });
   }
   ```

**Success Criteria:**
- ✅ AgentDB initialized and operational
- ✅ Data persisting to SQLite
- ✅ Reflexion logs accumulating
- ✅ Vector embeddings stored

#### Day 5: Bug Fixes & Testing
**Owner:** Test Engineer
**Effort:** 8 hours

**Tasks:**
1. Fix interference graph neighbor detection
2. Fix P0/Alpha controller rationale text
3. Update dependencies
4. Run full test suite
5. Validate coverage improvements

**Success Criteria:**
- ✅ Test pass rate: 75.8% → 90%+
- ✅ No new regressions introduced
- ✅ Dependencies up to date

**WEEK 1 TARGET: 75/100 → 80/100** ✅

---

### WEEK 2 (Dec 14-20): COMPLETE CRITICAL FIXES

**Goal:** Finish P0 items, begin P1 integration → **Target: 80/100**

#### Day 1-2: LLM Council Vector Embeddings
**Owner:** ML Engineer
**Effort:** 8-12 hours

```typescript
// File: src/council/embeddings.ts
import Anthropic from '@anthropic-ai/sdk';

export async function generateProposalEmbedding(proposal: string): Promise<number[]> {
  // Use Claude's text embedding API (768 dimensions)
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const response = await client.embeddings.create({
    model: 'claude-text-embedding',
    input: proposal
  });
  return response.data[0].embedding;
}
```

**Success Criteria:**
- ✅ 768-dim embeddings generated
- ✅ Stored in AgentDB
- ✅ HNSW index created
- ✅ Similarity search working

#### Day 3-5: ENM REST Client (Initial)
**Owner:** Senior Backend Engineer #2
**Effort:** 24 hours

**Tasks:**
1. Create ENM client class:
   ```typescript
   // File: src/enm/client.ts
   export class ENMClient {
     async fetchPMCounters(cells: string[], startTime: Date, endTime: Date) {
       const response = await fetch(`${this.enmUrl}/pm/v1/counters`, {
         method: 'POST',
         headers: { 'Authorization': `Bearer ${this.token}` },
         body: JSON.stringify({ cells, startTime, endTime })
       });
       return response.json();
     }

     async subscribeFMAlarms(cells: string[]) {
       // TS 28.532 alarm subscription
     }

     async updateParameter(cell: string, param: string, value: any) {
       // TS 28.532 configuration management
     }
   }
   ```

2. Replace mock data in PM Collector:
   ```typescript
   // File: src/smo/pm-collector.ts
   // BEFORE:
   const pmData = this.generateMockPMCounters();

   // AFTER:
   const enmClient = new ENMClient(config.enmUrl);
   const pmData = await enmClient.fetchPMCounters(this.cells, startROP, endROP);
   ```

**Success Criteria:**
- ✅ ENM client implemented (TS 28.552/28.532)
- ✅ PM counter collection working
- ✅ FM alarm subscription working
- ✅ Mock data removed from production code

**WEEK 2 TARGET: 80/100** ✅

---

### WEEK 3 (Dec 21-27): INTEGRATION PHASE 1

**Goal:** Complete ENM integration, expand test coverage → **Target: 84/100**

#### Day 1-3: ENM 3-ROP Governance
**Owner:** Senior Backend Engineer #1
**Effort:** 24 hours

**Tasks:**
1. Implement 3-ROP monitoring:
   ```typescript
   // File: src/enm/rop-governance.ts
   export class ROPGovernance {
     async monitor3ROPCycle(change: ParameterChange) {
       // ROP 1: Baseline collection
       const baseline = await this.collectROP(change.cell);

       // Apply change
       await this.enmClient.updateParameter(change);

       // ROP 2: Measure impact
       const rop2 = await this.collectROP(change.cell);
       const prediction = this.validatePrediction(baseline, rop2, change.expectedImpact);

       if (!prediction.withinConfidenceInterval) {
         // ROP 3: Confirm or rollback
         const rop3 = await this.collectROP(change.cell);
         if (!this.confirmSuccess(rop2, rop3)) {
           await this.rollback(change);
         }
       }
     }
   }
   ```

**Success Criteria:**
- ✅ 3-ROP governance implemented
- ✅ Automatic rollback working
- ✅ Confidence interval validation
- ✅ QuDAG ledger integration

#### Day 4-5: Test Coverage Expansion (Part 1)
**Owner:** Test Engineer
**Effort:** 16 hours

**New Test Files:**
1. `tests/safety.test.ts` (Lyapunov analysis, SPARC gates)
2. `tests/enm-integration.test.ts` (ENM workflows, 3-ROP)
3. `tests/council-integration.test.ts` (LLM debate protocol)

**Success Criteria:**
- ✅ 50+ new tests added
- ✅ Coverage: 68% → 73%
- ✅ All new tests passing

**WEEK 3 TARGET: 84/100** ✅

---

### WEEK 4 (Dec 28 - Jan 3): INTEGRATION PHASE 2

**Goal:** Finalize integration, hit 80% coverage → **Target: 88/100**

#### Day 1-3: Test Coverage Expansion (Part 2)
**Owner:** Test Engineer + ML Engineer
**Effort:** 24 hours

**New Test Files:**
1. Complete `tests/phase2.test.js` (multi-cell swarm)
2. Add GNN integration tests (end-to-end optimization)
3. Add performance regression tests

**Success Criteria:**
- ✅ 100+ new tests added
- ✅ Coverage: 73% → 80%+
- ✅ All critical paths tested

#### Day 4-5: 3GPP Compliance Validation
**Owner:** Senior Backend Engineer #2
**Effort:** 12 hours

**Tasks:**
1. Implement real spec checks:
   ```typescript
   // File: src/sparc/validator.js
   async validate3GPPCompliance(artifact: any): Promise<boolean> {
     // TS 38.331 parameter ranges
     if (artifact.p0NominalPusch < -130 || artifact.p0NominalPusch > -70) {
       throw new Error('P0 outside 3GPP range (-130 to -70 dBm)');
     }

     // TS 38.300 BLER limits
     if (artifact.targetBLER > 0.1) {
       throw new Error('BLER exceeds 3GPP limit (0.1)');
     }

     // TS 38.214 power control
     if (artifact.pMax > 46) {
       throw new Error('Power exceeds 3GPP max (46 dBm)');
     }

     return true;
   }
   ```

**Success Criteria:**
- ✅ Real 3GPP validation
- ✅ Replace hardcoded `compliant: true`
- ✅ All parameter ranges enforced

**WEEK 4 TARGET: 88/100** ✅

---

### WEEK 5 (Jan 4-10): HARDENING PHASE 1

**Goal:** Transport layer, refactoring, logging → **Target: 91/100**

#### Day 1-2: Transport Layer Decision & Implementation
**Owner:** DevOps Engineer + Senior Backend Engineer #1
**Effort:** 16 hours

**Option A: gRPC (Recommended)**
```bash
npm install @grpc/grpc-js @grpc/proto-loader
```

```typescript
// File: src/transport/grpc-transport.ts
import * as grpc from '@grpc/grpc-js';

export class GRPCTransport {
  async sendToAgent(agentId: string, message: any) {
    const client = this.getClient(agentId);
    return new Promise((resolve, reject) => {
      client.processMessage(message, (error, response) => {
        if (error) reject(error);
        else resolve(response);
      });
    });
  }
}
```

**Success Criteria:**
- ✅ Transport protocol selected
- ✅ Inter-agent communication working
- ✅ Performance benchmarked (<50ms p95)

#### Day 3-4: Structured Logging
**Owner:** DevOps Engineer
**Effort:** 12 hours

```typescript
// File: src/utils/logger.ts
import winston from 'winston';

export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({ format: winston.format.simple() })
  ]
});

// Replace all console.log calls:
// console.log('Agent spawned') → logger.info('Agent spawned', { agentId, type })
```

**Success Criteria:**
- ✅ Winston installed and configured
- ✅ 100+ console.log calls replaced
- ✅ Structured JSON logging
- ✅ Log levels (debug, info, warn, error)

#### Day 5: Refactoring Large Classes
**Owner:** Senior Backend Engineer #2
**Effort:** 8 hours

**Targets:**
1. Split `smo/fm-handler.ts` (858 lines)
   - Extract `AlarmCorrelator` class
   - Extract `AnomalyDetector` class

2. Split `council/orchestrator.ts` (722 lines)
   - Extract `DebateManager` class
   - Extract `ConsensusCalculator` class

**Success Criteria:**
- ✅ All files <500 lines
- ✅ Better modularity
- ✅ No regression in tests

**WEEK 5 TARGET: 91/100** ✅

---

### WEEK 6 (Jan 11-17): HARDENING PHASE 2

**Goal:** AG-UI frontend, TypeScript migration → **Target: 93/100**

#### Day 1-4: AG-UI Frontend
**Owner:** Frontend Engineer (new) + Senior Backend Engineer #1
**Effort:** 32 hours

**Tasks:**
1. WebSocket server:
   ```typescript
   // File: src/agui/websocket-server.ts
   import WebSocket from 'ws';

   const wss = new WebSocket.Server({ port: 8080 });

   wss.on('connection', (ws) => {
     ws.on('message', (message) => {
       // Handle HITL approval responses
     });

     // Subscribe to AG-UI events
     agui.on('request_approval', (payload) => {
       ws.send(JSON.stringify({ type: 'approval_request', data: payload }));
     });
   });
   ```

2. React dashboard:
   ```tsx
   // File: src/ui/frontend/App.tsx
   import React from 'react';
   import { InterferenceHeatmap } from './components/InterferenceHeatmap';
   import { ApprovalCard } from './components/ApprovalCard';

   export default function App() {
     return (
       <div>
         <InterferenceHeatmap cells={cells} matrix={interferenceMatrix} />
         <ApprovalCard requests={pendingApprovals} onApprove={handleApprove} />
       </div>
     );
   }
   ```

**Success Criteria:**
- ✅ WebSocket server working
- ✅ React dashboard deployed
- ✅ Real-time heatmaps rendering
- ✅ HITL approval workflow functional

#### Day 5: TypeScript Migration
**Owner:** Senior Backend Engineer #2
**Effort:** 8 hours

**Files to convert:**
1. `agents/base-agent.js` → `agents/base-agent.ts`
2. `consensus/voting.js` → `consensus/voting.ts`
3. `transport/quic-transport.js` → `transport/grpc-transport.ts`

**Success Criteria:**
- ✅ All core files now TypeScript
- ✅ Type safety across codebase
- ✅ No runtime errors

**WEEK 6 TARGET: 93/100** ✅

---

### WEEK 7 (Jan 18-24): PRODUCTION PREP 1

**Goal:** Benchmarking, monitoring setup → **Target: 95/100**

#### Day 1-2: Performance Benchmarking
**Owner:** ML Engineer + DevOps Engineer
**Effort:** 16 hours

**Benchmarks to run:**
1. LLM Council consensus time (target: <5s)
2. Safety check execution (target: <100ms)
3. Vector search (target: <10ms p95) ✅ already met
4. End-to-end optimization latency
5. Multi-cell coordination overhead

```typescript
// File: tests/benchmarks/performance.test.ts
test('LLM Council consensus completes in <5s', async () => {
  const start = Date.now();
  const result = await councilOrchestrator.runDebate(proposal);
  const elapsed = Date.now() - start;
  expect(elapsed).toBeLessThan(5000);
});
```

**Success Criteria:**
- ✅ All performance targets validated
- ✅ Latency percentiles measured (p50, p95, p99)
- ✅ Optimization opportunities identified

#### Day 3-4: Monitoring Setup
**Owner:** DevOps Engineer
**Effort:** 16 hours

**Stack:**
- Prometheus (metrics collection)
- Grafana (visualization)
- AlertManager (alerting)

```typescript
// File: src/monitoring/metrics.ts
import prometheus from 'prom-client';

const register = new prometheus.Registry();

export const agentMetrics = {
  taskDuration: new prometheus.Histogram({
    name: 'agent_task_duration_seconds',
    help: 'Agent task execution time',
    labelNames: ['agent_type', 'task_type'],
    registers: [register]
  }),

  councilConsensusTime: new prometheus.Histogram({
    name: 'council_consensus_duration_seconds',
    help: 'LLM Council consensus time',
    registers: [register]
  })
};
```

**Grafana Dashboards:**
1. System health (uptime, errors, latency)
2. Agent activity (tasks, duration, success rate)
3. LLM Council (debates, consensus, model usage)
4. Network KPIs (SINR, BLER, throughput)

**Success Criteria:**
- ✅ Prometheus scraping metrics
- ✅ Grafana dashboards deployed
- ✅ Alerts configured (CRITICAL, HIGH)

**WEEK 7 TARGET: 95/100** ✅

---

### WEEK 8 (Jan 25-31): PRODUCTION PREP 2

**Goal:** Security, load testing, final validation → **Target: 98/100 PRODUCTION READY**

#### Day 1-2: Security Hardening
**Owner:** Security Engineer (consultant) + Senior Backend Engineer #1
**Effort:** 16 hours

**Tasks:**
1. ML-DSA-87 signature implementation
2. ML-KEM-768 encryption for sensitive data
3. QuDAG ledger integration (immutable audit trail)
4. Security audit and penetration testing

```typescript
// File: src/security/quantum-resistant.ts
import { ml_dsa_87 } from 'post-quantum-crypto';

export async function signParameterChange(change: ParameterChange): Promise<Signature> {
  const privateKey = await loadPrivateKey();
  const signature = ml_dsa_87.sign(JSON.stringify(change), privateKey);

  // Store in QuDAG ledger
  await qudagLedger.append({
    type: 'parameter_change',
    data: change,
    signature,
    timestamp: Date.now()
  });

  return signature;
}
```

**Success Criteria:**
- ✅ Quantum-resistant signatures working
- ✅ QuDAG ledger operational
- ✅ Security audit passed
- ✅ No critical vulnerabilities

#### Day 3-4: Load Testing
**Owner:** Test Engineer + DevOps Engineer
**Effort:** 16 hours

**Test Scenarios:**
1. 10-cell optimization (baseline)
2. 50-cell optimization (Phase 3 simulation)
3. 100 concurrent LLM Council debates
4. Stress test (200% nominal load)
5. Failure injection (chaos engineering)

```bash
# Load test script
npm run benchmark -- --scenario multi_cell --cells 50 --duration 300s
```

**Success Criteria:**
- ✅ System stable under 2x load
- ✅ Graceful degradation under stress
- ✅ Auto-recovery from failures
- ✅ No memory leaks

#### Day 5: Final Validation & Sign-Off
**Owner:** All engineers + Project Manager
**Effort:** 8 hours

**Checklist:**
- ✅ All 281 tests passing (100%)
- ✅ Test coverage ≥80%
- ✅ All performance targets met
- ✅ Security audit passed
- ✅ Documentation complete
- ✅ Deployment runbook ready
- ✅ Rollback procedure tested
- ✅ Production environment ready

**WEEK 8 TARGET: 98/100** ✅ **PRODUCTION READY**

---

## 🎯 SUCCESS METRICS

### Technical Metrics

| Metric | Week 0 | Week 2 | Week 4 | Week 6 | Week 8 |
|:-------|:-------|:-------|:-------|:-------|:-------|
| **System Readiness** | 68 | 80 | 88 | 93 | **98** ✅ |
| **Test Pass Rate** | 75.8% | 90% | 95% | 98% | **100%** ✅ |
| **Test Coverage** | 68% | 70% | 80% | 82% | **85%** ✅ |
| **Critical Blockers** | 5 | 1 | 0 | 0 | **0** ✅ |
| **Integration Score** | 48% | 60% | 70% | 80% | **90%** ✅ |

### Business Metrics

| Metric | Target | Week 8 |
|:-------|:-------|:-------|
| **UL SINR Improvement** | +26% | Validated ✅ |
| **System Uptime** | ≥99.9% | Measured ✅ |
| **URLLC Packet Loss** | ≤10⁻⁵ | Measured ✅ |
| **LLM Council Consensus** | <5s | Benchmarked ✅ |
| **Safety Check Time** | <100ms | Benchmarked ✅ |

---

## 📋 DAILY STAND-UP FORMAT

**Time:** 9:00 AM daily
**Duration:** 15 minutes
**Attendees:** All engineers + PM

**Format:**
1. **Yesterday:** What did you complete?
2. **Today:** What will you work on?
3. **Blockers:** Any impediments?
4. **Metrics:** Current readiness score

**Example:**
```
ML Engineer:
  Yesterday: Fixed Uplink Optimizer constructor bug (31 tests now passing)
  Today: Implement vector embedding generation for LLM proposals
  Blockers: Need Anthropic API key for embedding endpoint
  Metrics: GNN test pass rate 0% → 100% ✅
```

---

## 🚨 ESCALATION PROTOCOL

### Level 1: Team Resolution (0-2 hours)
- Engineer identifies blocker
- Discusses with peer engineer
- Self-resolve or escalate

### Level 2: Technical Lead (2-8 hours)
- Technical lead provides guidance
- Allocates additional resources if needed
- Escalates if critical path blocked

### Level 3: Project Manager (8-24 hours)
- PM assesses schedule impact
- Reprioritizes tasks if necessary
- Escalates to executive if needed

### Level 4: Executive (>24 hours)
- Critical path blocked >1 day
- Major scope change needed
- Budget/resource reallocation required

---

## 📊 RISK MITIGATION

### Top 5 Risks & Mitigations

1. **Risk:** LLM API rate limits slow development
   **Mitigation:** Request increased quotas immediately, use caching

2. **Risk:** ENM integration blocked by firewall/network issues
   **Mitigation:** VPN setup, mock ENM server for parallel development

3. **Risk:** Test coverage expansion takes longer than estimated
   **Mitigation:** Add temporary test engineer, use AI test generation

4. **Risk:** Performance targets not met after optimization
   **Mitigation:** Allocate buffer week, engage performance specialist

5. **Risk:** Security audit identifies critical vulnerabilities
   **Mitigation:** Engage security consultant early (Week 6), not Week 8

---

## ✅ DEFINITION OF DONE

### For Each Week
- ✅ All planned tasks completed
- ✅ Target readiness score achieved
- ✅ No new critical bugs introduced
- ✅ Documentation updated
- ✅ Code reviewed and merged
- ✅ Stand-up notes captured

### For Production Deployment (Week 8)
- ✅ All 281+ tests passing (100%)
- ✅ Test coverage ≥80%
- ✅ System readiness ≥95/100
- ✅ All performance targets validated
- ✅ Security audit passed
- ✅ Load testing completed
- ✅ Monitoring dashboards deployed
- ✅ Deployment runbook approved
- ✅ Rollback procedure tested
- ✅ Executive sign-off received

---

## 📞 CONTACTS

**Project Manager:** [TBD]
**Technical Lead:** [TBD]
**Senior Backend Engineer #1:** [TBD]
**Senior Backend Engineer #2:** [TBD]
**ML Engineer:** [TBD]
**Test Engineer:** [TBD]
**DevOps Engineer:** [TBD]
**Security Consultant:** [TBD]

---

**Next Review:** End of Week 1 (December 13, 2025)
**Status Report:** Weekly on Fridays at 4:00 PM

---

*This action plan provides a clear, week-by-week roadmap to achieve 98/100 production readiness in 8 weeks. All tasks are assigned, estimated, and have clear success criteria.*
