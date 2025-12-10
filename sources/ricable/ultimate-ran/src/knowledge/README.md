# Agentic-Flow Knowledge Graph Integration

## Overview

This module provides 3GPP knowledge graph integration for the Titan Council using agentic-flow patterns.

## Components

### 1. ThreeGPPKnowledgeAgent (`agentic-kg.ts`)

**Capabilities:**
- `spec_lookup`: Look up 3GPP specifications by ID
- `parameter_search`: Search for RAN parameters
- `requirement_extraction`: Extract requirements from specs
- `prd_generation`: Generate Product Requirements Documents
- `sparc_research`: Run SPARC research pipeline

**Actions:**
```typescript
await agent.actions.lookupSpec('TS-38.331');
await agent.actions.searchParameters('power');
await agent.actions.extractRequirements('uplink power control');
await agent.actions.generatePRD('feature name', ['TS-38.331']);
await agent.actions.runSPARC('optimization topic', 'all');
```

### 2. KnowledgeEnhancedCouncil (`agentic-kg.ts`)

Enhances Council proposals with 3GPP knowledge:

```typescript
const council = new KnowledgeEnhancedCouncil(knowledgeAgent);

// Enhance proposal with 3GPP knowledge
const enhanced = await council.enhanceProposal(proposal);

// Validate against 3GPP standards
const validation = await council.validateAgainst3GPP(proposal);
```

### 3. MCP Tools

Four MCP tools for knowledge graph operations:

- `kg_search`: Search the knowledge graph
- `kg_traverse`: Traverse graph relationships
- `kg_sparc`: Run SPARC research pipeline
- `kg_prd`: Generate PRD from specs

## SPARC Integration

The knowledge agent integrates with the SPARC methodology:

- **S (Specification)**: Extract formal specs and constraints
- **P (Pseudocode)**: Generate algorithm documentation
- **A (Architecture)**: Define system architecture
- **R (Refinement)**: Define test cases and resource limits
- **C (Completion)**: Verify 3GPP compliance

## Usage Example

```typescript
import { ThreeGPPKnowledgeAgent, KnowledgeEnhancedCouncil } from './knowledge/index.js';

// Initialize agent
const agent = new ThreeGPPKnowledgeAgent();
await agent.initialize();

// Run SPARC research
const research = await agent.actions.runSPARC('Uplink power control', 'all');

// Generate PRD
const prd = await agent.actions.generatePRD('Power control optimization');

// Integrate with Council
const council = new KnowledgeEnhancedCouncil(agent);
const enhanced = await council.enhanceProposal(councilProposal);
```

## Data Model

### 3GPP Spec
- ID (e.g., "TS-38.331")
- Version
- Title
- Category (RRC, NAS, RRM, PHY, MAC, etc.)
- Sections, Parameters, Information Elements

### Parameter
- Name (e.g., "p0-NominalPUSCH")
- Type (integer, enumerated, boolean, etc.)
- Range (min/max or allowed values)
- Unit (dBm, dB, Hz, etc.)
- Description
- 3GPP constraints

### Knowledge Graph
- Nodes: Specs, Sections, Parameters, IEs
- Edges: contains, references, relatedTo, implements, constrains
- Vector embeddings for similarity search

## Testing

Run the test suite:

```bash
node --loader ts-node/esm tests/knowledge.test.ts
```

Test coverage:
- GraphML parsing and KG initialization
- Spec metadata indexing
- SPARC pipeline (all 5 phases)
- Query interface (search, traverse)
- PRD generation
- Requirement extraction
- Council integration
- Event emission
- Full workflow integration

## Architecture

```
┌─────────────────────────────────────┐
│  KnowledgeEnhancedCouncil           │
│  (Enhances proposals with 3GPP)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  ThreeGPPKnowledgeAgent             │
│  - spec_lookup                      │
│  - parameter_search                 │
│  - requirement_extraction           │
│  - prd_generation                   │
│  - sparc_research                   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Knowledge Graph                    │
│  - 3GPP Specs (TS-38.331, etc.)     │
│  - Parameters (p0, alpha, etc.)     │
│  - Information Elements (RRC, NAS)  │
│  - Vector embeddings (ruvector)     │
└─────────────────────────────────────┘
```

## Integration with Council

The Analyst, Historian, and Strategist can leverage the knowledge agent:

- **Analyst**: Query KG for Lyapunov analysis context
- **Historian**: Retrieve past spec implementations
- **Strategist**: Generate spec-compliant proposals

```typescript
// Strategist proposes parameters
const proposal = {
  parameters: {
    'p0-NominalPUSCH': -103,
    'alpha': 0.8
  }
};

// Knowledge agent validates against 3GPP
const enhanced = await council.enhanceProposal(proposal);
// enhanced.complianceScore: 1.0 (100% compliant)
// enhanced.parameterValidation: [...all valid...]
```

## Events

The agent emits events for monitoring:

- `initialized`: Agent initialized
- `spec_lookup`: Spec lookup completed
- `parameter_search`: Parameter search completed
- `requirements_extracted`: Requirements extracted
- `prd_generated`: PRD generated
- `sparc_complete`: SPARC research completed
- `proposal_enhanced`: Proposal enhanced

## Future Enhancements

1. Load actual GraphML from 3GPP dataset
2. Integrate with ruvector for vector similarity search
3. Store in agentdb for reflexion/learning
4. Real-time spec update monitoring
5. ASN.1 schema validation
6. Multi-spec compliance checking
