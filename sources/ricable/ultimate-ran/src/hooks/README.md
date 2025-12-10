# SPARC Safety Hooks - Track D: Governance

Physics-Aware Safety Layer with 3GPP Compliance for the Titan Gen 7.0 Platform

## Overview

The Safety Hooks module implements a critical governance layer that prevents dangerous parameter changes before they reach the network. It combines:

1. **Symbolic Verification**: 3GPP standards compliance checking
2. **Physics Verification**: Real-time network state analysis to prevent feedback loops
3. **Hardware Validation**: Equipment-specific limit enforcement
4. **Audit Trail**: Complete logging of all safety decisions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Execution Request                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Pre-Tool-Use Hook                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. 3GPP Compliance Check                            │  │
│  │     - Power limits (TS 38.104)                       │  │
│  │     - BLER limits (TS 38.331)                        │  │
│  │     - CIO range validation                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Hardware Limits Verification                     │  │
│  │     - Radio 6630: 46 dBm max TX power               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. Physics Verification                             │  │
│  │     - Interference storm detection                   │  │
│  │     - Chaos detection (Lyapunov exponent)            │  │
│  │     - Positive feedback loop prevention              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
           ✅ ALLOW              ❌ DENY
                │                   │
                ▼                   │
    ┌───────────────────┐          │
    │  Execute Change   │          │
    └─────────┬─────────┘          │
              │                     │
              ▼                     ▼
    ┌──────────────────────────────────┐
    │   Post-Tool-Use Hook             │
    │   - Audit Logging                │
    │   - Reflexion Storage            │
    └──────────────────────────────────┘
```

## Critical Safety Rules

### Rule 1: Interference Storm Prevention

**Physics**: Increasing transmit power during high interference creates a positive feedback loop that can destabilize the entire network cluster.

**Implementation**:
```typescript
if (interference_level > -90 dBm && tx_power > 40 dBm) {
  DENY: "Power boost prohibited during interference storm"
}
```

**Example**:
- Cell A experiences interference: -85 dBm (above threshold)
- Agent requests power boost to 43 dBm
- **BLOCKED**: This would amplify interference to neighbors, creating cascade

### Rule 2: 3GPP Power Limits

**Standard**: 3GPP TS 38.104 specifies maximum transmit power for 5G NR base stations.

**Implementation**:
```typescript
if (tx_power > 46 dBm) {
  DENY: "Power exceeds 3GPP TS 38.104 limit"
}
```

### Rule 3: Hardware Limits

**Equipment**: Ericsson Radio 6630 has hardware-enforced maximum TX power of 46 dBm.

**Implementation**:
```typescript
if (tx_power > radio_6630.tx_power_max_dbm) {
  DENY: "Power exceeds Radio 6630 hardware limit"
}
```

### Rule 4: Chaos Detection

**Physics**: When Lyapunov exponent > 1.0, the system is in chaotic state and parameter changes can cause unpredictable behavior.

**Implementation**:
```typescript
if (lyapunov_exponent > 1.0) {
  DENY: "System chaos detected. Parameter changes blocked."
}
```

## Usage

### Basic Integration

```typescript
import { safetyHooks, HookManager } from './hooks/safety';

// Hooks are auto-initialized on import

// Execute a parameter change
const result = await HookManager.execute(
  'pre_tool_use',
  'execute_parameter_change',
  {
    cellId: 'cell-001',
    params: { tx_power: 43 },
    hardware: 'radio_6630'
  },
  {
    cellId: 'cell-001',
    tools: {
      ruvector_query_neighbors: async (cellId) => {
        // Query actual network state
        return {
          interference_level: -95,
          lyapunov_exponent: 0.3
        };
      }
    }
  }
);

if (result.action === 'deny') {
  console.error('Safety violation:', result.reason);
  console.error('Violations:', result.violations);
} else {
  // Proceed with change
  await executeParameterChange(...);
}
```

### Custom Guard Configuration

```typescript
import { SafetyHooks, PsychoSymbolicGuardInterface } from './hooks/safety';

const customGuard = new PsychoSymbolicGuardInterface({
  '3gpp_constraints': {
    power_max_dbm: 43, // Custom limit
    bler_max: 0.05
  },
  'physics_thresholds': {
    interference_critical_dbm: -85,
    power_boost_max_dbm: 38
  }
});

const hooks = new SafetyHooks(customGuard);
hooks.initialize();
```

### Audit Trail Access

```typescript
import { safetyHooks } from './hooks/safety';

// Get audit statistics
const stats = safetyHooks.getAuditStats();
console.log(`Total operations: ${stats.total}`);
console.log(`Denied: ${stats.denied}`);
console.log(`Critical violations: ${stats.criticalViolations}`);

// Export full audit trail
const trail = safetyHooks.exportAuditTrail();
fs.writeFileSync('audit-trail.json', trail);
```

## Testing

Run the comprehensive test suite:

```bash
# Build TypeScript
npm run build

# Run safety hooks tests
npm run test:safety
```

### Test Coverage

- ✅ Interference storm blocking
- ✅ 3GPP power limit enforcement
- ✅ Hardware limit validation (Radio 6630)
- ✅ Safe parameter changes allowed
- ✅ Chaos detection and blocking
- ✅ Audit trail logging
- ✅ Post-execution hooks

## Integration with SPARC Workflow

The safety hooks integrate seamlessly with the SPARC methodology:

1. **Specification**: Defines safety constraints (3GPP, physics, hardware)
2. **Pseudocode**: Validates logic against constraints
3. **Architecture**: Enforces architectural patterns
4. **Refinement**: TDD with physics-based test cases
5. **Completion**: Audit trail for compliance reporting

## Performance

- **Pre-check latency**: < 10ms (including network state query)
- **Memory overhead**: ~1KB per audit entry
- **Audit log rotation**: 10,000 entries maximum

## Compliance

The Safety Hooks module ensures compliance with:

- **3GPP TS 38.104**: Radio transmission and reception
- **3GPP TS 38.331**: Radio Resource Control (RRC)
- **Ericsson Radio 6630 Specifications**: Hardware limits
- **Internal Safety Policy**: Physics-based constraints

## Audit Trail Format

Each audit entry contains:

```typescript
{
  timestamp: "2025-12-06T10:30:00.000Z",
  toolName: "execute_parameter_change",
  args: {
    cellId: "cell-001",
    params: { tx_power: 43 },
    hardware: "radio_6630"
  },
  context: {
    cellId: "cell-001",
    agentId: "agent-strategist-01",
    sessionId: "session-123"
  },
  result: {
    action: "deny",
    reason: "Safety violations detected: INTERFERENCE_STORM",
    violations: [
      {
        type: "INTERFERENCE_STORM",
        severity: "CRITICAL",
        message: "Power boost prohibited during interference storm...",
        constraint: "Physics: Prevent positive feedback loop",
        value: { interference: -85, requestedPower: 43 },
        limit: { maxInterference: -90, maxPowerDuringStorm: 40 }
      }
    ]
  },
  executionTime: 8
}
```

## Future Enhancements

- [ ] ML-DSA signature verification for critical approvals
- [ ] Integration with strange-loops for advanced chaos analysis
- [ ] Real-time visualization in AG-UI Dojo
- [ ] Automated incident response (circuit breakers)
- [ ] Cross-cluster coordination for multi-cell scenarios

## References

- Plan: `/home/user/ultimate-ran/plan.md` (Track D)
- Psycho-Symbolic Guard: `/home/user/ultimate-ran/src/security/psycho-symbolic.js`
- 3GPP Specifications: https://www.3gpp.org/specifications-technologies

---

**Track D: SPARC Governance**
**Agent-08 Implementation**
**Ericsson Gen 7.0 Neuro-Symbolic Titan Platform**
