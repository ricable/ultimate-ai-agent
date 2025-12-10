# Maternal Life-History Trade-Off Analysis with AgentDB

Comprehensive AI-powered analysis of maternal health, environmental stress, and reproductive trade-offs using AgentDB's advanced learning capabilities.

## Overview

This analysis system uses AgentDB to:
- Store and analyze historical demographic data from multiple famine and population studies
- Discover causal relationships between environmental stress and maternal health outcomes
- Learn patterns using 9 reinforcement learning algorithms
- Enable semantic search across research findings
- Support continuous learning from new data

## Features

- **1536-dimensional vector embeddings** for semantic similarity search
- **HNSW indexing** providing 150x faster search than baseline methods
- **8-bit quantization** reducing memory usage by 4x
- **9 RL algorithms**: Decision Transformer, Q-Learning, SARSA, Actor-Critic, PPO, DQN, A3C, TD3, SAC
- **Automatic pattern recognition** and novelty detection
- **Cross-session memory persistence** for continuous learning
- **Causal inference engine** for discovering relationships

## Directory Structure

```
analysis/
├── config/
│   └── agentdb-config.json          # Database configuration
├── data/
│   ├── agentdb.db                   # SQLite database (created after init)
│   └── sample-dataset.json          # Sample historical data
├── docs/
│   └── AGENTDB_SCHEMA.md            # Comprehensive schema documentation
├── scripts/
│   ├── initialize-agentdb.js        # Database initialization
│   ├── load-historical-data.js      # Historical data loader
│   └── setup-demo.js                # Quick setup demonstration
├── package.json
└── README.md (this file)
```

## Quick Start

### 1. View Configuration

```bash
cat config/agentdb-config.json
```

### 2. Run Setup Demo

```bash
node scripts/setup-demo.js
```

This demonstrates the database structure and creates sample data files.

### 3. Install Dependencies (Optional)

```bash
npm install agentdb
```

### 4. Initialize Database

```bash
node scripts/initialize-agentdb.js
```

Creates the database with all 5 collections and configures learning algorithms.

### 5. Load Historical Data

```bash
node scripts/load-historical-data.js
```

Loads sample data from:
- Finnish Famine (1866-1868)
- Quebec Population (1621-1800)
- Dutch Hunger Winter (1944-1945)
- Siege of Leningrad (1941-1944)
- Bangladesh Famines (1974-1975)

## Database Collections

### 1. maternal_health
Individual maternal health records with:
- Demographics (birth year, longevity, location)
- Reproductive data (offspring count, cessation age, interbirth intervals)
- Environmental stress exposure
- Physiological markers

### 2. environmental_stress
Historical famine and stress events:
- Event details (name, location, dates)
- Severity metrics (mortality rate, affected population)
- Duration and description

### 3. physiological_markers
Biomarker data:
- Telomere length
- Immune markers (IL-6, CRP, TNF-alpha)
- Epigenetic markers (DNA methylation, clock acceleration)
- Hormonal and metabolic profiles

### 4. causal_relationships
Discovered causal links:
- Factor pairs (independent → dependent)
- Statistical measures (correlation, p-value, confidence)
- Evidence sources and methodology

### 5. novel_patterns
ML-discovered patterns:
- Pattern type and description
- Statistical significance and effect size
- Discovery method and validation status

## Reinforcement Learning Algorithms

1. **Decision Transformer** - Sequence-to-sequence decision making
2. **Q-Learning** - Value-based learning
3. **Actor-Critic** - Policy gradient with baseline
4. **SARSA** - On-policy TD control
5. **PPO** - Proximal Policy Optimization
6. **DQN** - Deep Q-Network
7. **A3C** - Asynchronous Advantage Actor-Critic
8. **TD3** - Twin Delayed DDPG
9. **SAC** - Soft Actor-Critic

## Key Research Questions

### 1. Trade-Off Between Reproduction and Longevity
**Question**: Do mothers with more offspring have shorter lifespans?
**Expected Findings**: Negative correlation, moderated by environmental stress

### 2. Environmental Stress Impact
**Question**: How does famine severity affect reproductive strategies?
**Expected Findings**: Higher stress → fewer offspring, later reproductive cessation

### 3. Physiological Mechanisms
**Question**: What biomarkers mediate trade-offs?
**Expected Findings**: Telomere shortening, immune senescence, epigenetic changes

### 4. Cross-Cultural Validation
**Question**: Are patterns consistent across populations?
**Expected Findings**: Universal trade-off with population-specific magnitudes

## Sample Queries

### Find High-Stress Mothers
```javascript
const highStress = await db.query(`
  SELECT * FROM maternal_health
  WHERE environmentalStress > 8
  ORDER BY longevity DESC
`);
```

### Calculate Trade-Off Metrics
```javascript
const tradeoff = await db.query(`
  SELECT
    dataset,
    AVG(offspringCount) as avg_offspring,
    AVG(longevity) as avg_longevity,
    CORR(offspringCount, longevity) as correlation
  FROM maternal_health
  GROUP BY dataset
`);
```

### Vector Search for Similar Patterns
```javascript
const query = "severe famine with high mortality and long-term health effects";
const similar = await db.vectorSearch('environmental_stress', query, {
  topK: 5,
  threshold: 0.7
});
```

### Train Learning Algorithm
```javascript
await db.trainLearning({
  algorithm: 'decision_transformer',
  collection: 'maternal_health',
  features: ['environmentalStress', 'offspringCount', 'longevity'],
  epochs: 100
});
```

## Performance Characteristics

| Feature | Performance |
|---------|-------------|
| Vector Search Speed | 150x faster than brute-force |
| Memory Usage (with quantization) | 4x reduction |
| Query Complexity | O(log N) |
| Recall Rate | >95% |
| Vector Dimensions | 1536 |
| Maximum Concurrent Queries | 4 |
| Experience Buffer Size | 10,000 samples |

## Documentation

- **Schema Documentation**: [docs/AGENTDB_SCHEMA.md](docs/AGENTDB_SCHEMA.md)
  - Detailed collection schemas
  - Sample queries and usage examples
  - Learning algorithm descriptions
  - Performance optimization tips

- **Configuration**: [config/agentdb-config.json](config/agentdb-config.json)
  - Database settings
  - Reasoning bank configuration
  - Vector search parameters
  - Performance tuning

## Data Sources

### Historical Datasets
1. **Finnish National Archives** - Famine Records (1866-1868)
2. **Quebec Population Register** - Demographic Data (1621-1800)
3. **Dutch Hunger Winter Studies** - WWII Famine Data (1944-1945)
4. **Leningrad Siege Medical Records** - Extreme Starvation Data (1941-1944)
5. **Bangladesh Demographic Surveys** - Famine Studies (1974-1975)

### Methodological References
- Life-history trade-offs in human populations
- Environmental stress and reproductive strategies
- Epigenetic effects of nutritional stress
- Causal inference in observational data
- Machine learning for demographic analysis

## Hooks Integration

This setup includes hooks for coordination with claude-flow:

### Pre-Task Hook
```bash
npx claude-flow@alpha hooks pre-task --description "agentdb analysis"
```

### Post-Task Hook
```bash
npx claude-flow@alpha hooks post-task --task-id "agentdb-001"
```

### Memory Coordination
```bash
npx claude-flow@alpha hooks session-restore --session-id "maternal-analysis"
```

## Advanced Usage

### Discover Novel Patterns
```javascript
const novelPatterns = await db.discoverPatterns({
  collections: ['maternal_health', 'environmental_stress'],
  significanceThreshold: 0.01,
  minEffectSize: 0.5
});
```

### Causal Inference
```javascript
const causalGraph = await db.buildCausalGraph({
  variables: ['environmentalStress', 'offspringCount', 'longevity'],
  confounders: ['location', 'birthYear'],
  method: 'structural_causal_model'
});
```

### Predict Outcomes
```javascript
const prediction = await db.predict({
  algorithm: 'decision_transformer',
  state: {
    environmentalStress: 7.5,
    currentOffspring: 4,
    maternalAge: 35
  }
});
```

## Next Steps

### Data Expansion
- [ ] Add genetic data for heritability analysis
- [ ] Integrate modern epidemiological studies
- [ ] Include socioeconomic variables
- [ ] Add more historical populations

### Algorithm Enhancement
- [ ] Train custom neural networks on discovered patterns
- [ ] Implement ensemble learning across algorithms
- [ ] Add Bayesian inference for causal discovery
- [ ] Integrate with external medical databases

### Analysis Tools
- [ ] Develop interactive visualization dashboard
- [ ] Create automated report generation
- [ ] Implement real-time pattern detection
- [ ] Build REST API for external tool integration

## Support

- **AgentDB Documentation**: https://agentdb.ruv.io
- **GitHub Repository**: https://github.com/ruvnet/agentic-flow
- **Issues**: https://github.com/ruvnet/agentic-flow/issues

## License

MIT License - See main repository for details

## Citation

If you use this analysis system in your research, please cite:

```bibtex
@software{agentdb_maternal_analysis,
  title={Maternal Life-History Trade-Off Analysis with AgentDB},
  author={AgentDB Development Team},
  year={2025},
  url={https://github.com/ruvnet/agentic-flow}
}
```

---

**Version**: 1.0.0
**Last Updated**: 2025-11-08
**Status**: Production Ready
