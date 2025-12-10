# TITAN Agent System

The TITAN platform relies on a multi-agent "Cognitive Mesh" architecture where specialized agents collaborate to optimize, secure, and evolve the network.

## Implemented Agents

### 1. Architect Agent
**Role:** Strategic Planner & Cognitive Decomposition
**Type:** `architect`

The Architect is responsible for breaking down high-level intents (e.g., "Optimize downtown cluster for heavy video traffic") into structured Product Requirements Prompts (PRPs). It does not write implementation code but focuses on planning and constraints.

**Key Capabilities:**
- **Cognitive Decomposition:** Breaks objectives into interfaces, data structures, and constraints.
- **PRP Generation:** Creates formal specifications for other agents to execute.
- **Risk Assessment:** Identifies potential risks (e.g., interference escalation) and assigns mitigations.
- **Constraint Identification:** Enforces 3GPP and operational safety limits (e.g., max BLER, max power).

### 2. Guardian Agent
**Role:** Adversarial Safety & Pre-Commit Audit
**Type:** `guardian`

The Guardian acts as a safety gatekeeper *before* any code or configuration is deployed. It runs simulations in a digital twin environment to detect dangerous behaviors or "hallucinations" in agent-generated code.

**Key Capabilities:**
- **Pre-Commit Simulation:** Runs artifacts in a digital twin to observe effects over time.
- **Hallucination Detection:** Scans for logic errors like infinite power loops or physics violations.
- **Lyapunov Analysis:** Calculates Lyapunov exponents to detect the onset of chaotic behavior in the system dynamics.
- **Safety Verification:** Enforces hard safety bounds (e.g., transmission power limits).

### 3. Sentinel Agent
**Role:** System Observer & Circuit Breaker (RIV Pattern)
**Type:** `sentinel`

The Sentinel is a persistent observer that monitors the global state of the live network. It runs a "Strange Loop" monitoring cycle to ensure system stability and has the authority to halt operations if things go wrong.

**Key Capabilities:**
- **Global Monitoring:** Tracks real-time metrics (Throughput, BLER, Interference).
- **Chaos Detection:** Computes real-time Lyapunov exponents on live data.
- **Circuit Breaker:** Triggers system-wide freeze if stability drops or critical thresholds are breached.
- **Auto-Recovery:** Manages the transition from "Open" (frozen) to "Half-Open" (testing) to "Closed" (normal) stats.

## Conceptual Agent Roles

These roles are defined in the architecture and are currently being implemented or handled by the broader orchestration system.

| Agent | Role | Focus |
|:------|:-----|:------|
| **Cluster Orchestrator** | Coordination | Decomposes cluster-wide goals into per-cell quotas. (Likely handled by Architect) |
| **Cell Optimizer** | Execution | Proposes specific parameter changes for a single cell. |
| **KPI Advocate** | Specialist | Champions specific KPIs (e.g., "I speak for Retainability") to ensure balanced outcomes. |
| **Slice Agent** | Specialist | Optimizes resources for specific slices (e.g., URLLC, eMBB). |
| **Self-Healing Agent** | Maintenance | Anomaly detection and remediation. Implemented in `src/smo/fm-handler.ts`. Cross-correlates PM anomalies with FM alarms. |
| **Self-Learning Agent** | Optimization | Continuous improvement using Q-Learning. Implemented in `src/learning/self-learner.ts`. Adapts policies based on rewards from network environment. |

## Agent Communication

Agents communicate via the **Action-Observation-Reflexion** loop:
1. **Action:** An agent executes a task (or proposes a change).
2. **Observation:** The system (via Sentinel/Guardian) observes the result.
3. **Reflexion:** The agent critiques its own performance and logs it to `AgentDB` for future learning.

All critical changes are signed with **ML-DSA-87** (Quantum-Resistant Signatures) and logged to **QuDAG**.
