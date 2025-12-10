# Getting Started with TITAN (Ericsson Gen 7.0)

Welcome to **TITAN**, the Neuro-Symbolic RAN Platform developed by the Ericsson Autonomous Networks Division. This guide will help you understand the core philosophy of the system and get your local environment set up for development and simulation.

## Project Philosophy

TITAN represents a paradigm shift from traditional automated networks to **Cognitive Mesh** networks. 

### 1. The Living Network
Instead of static rules or simple ML models, TITAN treats the Radio Access Network (RAN) as a living organism. It utilizes a swarm of specialized AI agents that don't just optimize parameters but *reason* about network state, negotiate trade-offs, and evolve strategies over time.

### 2. Neuro-Symbolic Architecture
We combine the best of two worlds:
-   **Neural (The "Intuition")**: LLMs (DeepSeek, Gemini, Claude) provide creative problem-solving, pattern recognition, and strategic reasoning.
-   **Symbolic (The "Law")**: Deterministic logic, physics simulators (Lyapunov analysis), and digital twins ensure strict adherence to 3GPP standards and safety bounds.

### 3. Safety First (The Guardian)
Autonomy requires trust. The **Guardian Agent** acts as an incorruptible safety layer. No code or parameter change touches the live network without passing a rigorous "digital twin" simulation to prove it won't cause instability or "hallucinations."

### 4. SPARC Methodology
All agent-generated code follows the **SPARC** protocol to ensure quality and determinism:
-   **S**pecification: Define the intent.
-   **P**seudocode: Logic flow.
-   **A**rchitecture: Component interaction.
-   **R**efinement: Optimization & safety checks.
-   **C**ompletion: Final executable code.

## System Prerequisites

-   **Node.js**: >= 18.0.0
-   **npm**: Included with Node.js
-   **Private Keys**: Access to the internal Ericsson agent swarm keys (see your team lead).
-   **Access**: Connectivity to `claude-flow` and `agentdb` alpha registries.

## Installation

TITAN is built as a modular monorepo.

1.  **Clone the repository**:
    ```bash
    git clone https://internal.ericsson.se/titan-ran.git
    cd titan-ran
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```
    *Note: This will fetch private packages from the `ruvnet` registry.*

3.  **Configure environment**:
    Copy the example config and add your API keys.
    ```bash
    cp .env.example .env
    ```

## Quick Start

### 1. Run the Swarm Orchestrator
Start the main orchestration loop locally. This initializes the Architect, Guardian, and Sentinel agents.

```bash
npm start
```

### 2. Launch the Glass Box Interface (AG-UI)
Visualize the cognitive mesh and watch agents "think" in real-time.

```bash
npm run agui:start
```
Then open `src/agui/frontend.html` in your browser.

### 3. Run a Simulation
Trigger a standard "Optimize Downtown Coverage" scenario to see the agents in action.

```bash
npm run swarm:spawn -- --intent="optimize_coverage" --region="downtown_cluster_a"
```

## Key Commands

| Command | Description |
| :--- | :--- |
| `npm run orchestrate` | Run the `claude-flow` orchestration engine directly. |
| `npm run sentient:monitor` | Start the Sentinel agent for real-time chaos detection. |
| `npm run db:status` | Check the status of the local `agentdb` memory. |
| `npm test` | Run the full integration test suite using Vitest. |
| `npm run coverage` | Generate a code coverage report using v8. |

## Next Steps

-   **Meet the Agents**: Read [AGENTS.md](./AGENTS.md) to understand the specific roles of the Architect, Guardian, and Sentinel.
-   **Explore the Plan**: Check [my-prd.md](./my-prd.md) (or `plan.md`) to see the current roadmap and active phases.
-   **Deep Dive**: Look into `src/agents/` to see the actual implementation of the cognitive skills.

---
**Internal Use Only** - Ericsson AB
