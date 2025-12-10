# MCP Researcher Mode

## Overview

The MCP Researcher mode is a comprehensive research assistant that leverages Model Context Protocol (MCP) servers to conduct deep, structured research across multiple domains. It integrates with external services like Perplexity AI for real-time information gathering and Context7 for technical documentation, creating a powerful research ecosystem.

## Features

### 🔍 Multi-Domain Research
- **Technical Research**: Software libraries, APIs, frameworks, development practices
- **Market Intelligence**: Industry analysis, competitive landscape, market opportunities
- **Academic Research**: Literature reviews, methodological analysis, research synthesis
- **Regulatory Analysis**: Compliance requirements, legal frameworks, policy research

### 🔗 MCP Integration
- **Perplexity AI**: Real-time web search and AI-powered analysis
- **Context7**: Technical documentation and library research
- **SAFLA**: Internal system integration and workflow management

### 📊 Quality Assurance
- Source credibility assessment
- Cross-reference validation
- Bias detection and mitigation
- Confidence scoring

### 🚀 Advanced Workflows
- Recursive information gathering
- Multi-perspective analysis
- Parallel research execution
- Intelligent caching

## Quick Start

### 1. Initialize Research Session

```bash
# Start MCP Researcher mode
new_task: mcp-researcher

# Basic research command
"Research [TOPIC] focusing on [ASPECTS] for [PURPOSE]"
```

### 2. Example Research Commands

#### Technical Research
```bash
"Evaluate React state management libraries (Redux, Zustand, Jotai) for enterprise applications, comparing performance, developer experience, and ecosystem maturity."
```

#### Market Research
```bash
"Analyze the AI-powered development tools market, including competitive landscape, growth trends, and emerging opportunities."
```

#### Academic Research
```bash
"Conduct a literature review on explainable AI methods, covering recent advances, methodologies, and practical applications."
```

## File Structure

```
.roo/mcp-modes/rules-mcp-researcher/
├── README.md                 # This file
├── rules.md                  # Comprehensive mode rules and functionality
├── config.yaml              # Configuration settings
├── examples.md               # Detailed usage examples
├── workflows.md              # Workflow templates and patterns
└── test_mcp_researcher.py    # Test suite
```

## Configuration

The mode is configured via `config.yaml`:

```yaml
# MCP Server Dependencies
mcp_servers:
  required:
    - name: "perplexity"
      tools: ["PERPLEXITYAI_PERPLEXITY_AI_SEARCH"]
    - name: "context7"
      tools: ["resolve-library-id", "get-library-docs"]
    - name: "safla"
      tools: ["create_research_session", "manage_user_sessions"]

# Research Workflow Configuration
workflows:
  technical_research:
    phases: ["scope_definition", "documentation_gathering", "practice_analysis", "synthesis"]
    max_depth: 3
    parallel_processing: true
    cache_duration: "24h"
```

## Research Workflows

### Technical Research Workflow

1. **Scope Definition**: Define research parameters and criteria
2. **Documentation Gathering**: Collect authoritative technical documentation
3. **Practice Analysis**: Understand current industry practices
4. **Synthesis**: Generate actionable insights and recommendations

### Market Research Workflow

1. **Landscape Mapping**: Map competitive and market landscape
2. **Competitive Analysis**: Detailed competitor analysis
3. **Trend Analysis**: Identify emerging trends and signals
4. **Strategic Insights**: Generate actionable strategic insights

### Academic Research Workflow

1. **Literature Discovery**: Identify relevant academic sources
2. **Methodological Analysis**: Analyze research methodologies
3. **Knowledge Synthesis**: Synthesize research findings
4. **Research Agenda**: Develop future research directions

## Output Templates

### Executive Summary Template
```markdown
# Research Executive Summary

## Research Objective
[Clear statement of research goals and scope]

## Key Findings
1. **Primary Finding**: [Most important discovery with confidence level]
2. **Secondary Finding**: [Supporting discovery with evidence]

## Strategic Implications
- **Immediate Actions**: [What should be done now]
- **Medium-term Considerations**: [What to plan for]

## Recommendations
1. [Specific, actionable recommendation]
2. [Implementation timeline and resources]
```

### Technical Analysis Template
```markdown
# Technical Research Analysis

## Technology Overview
**Technology**: [Name and version]
**Maturity Level**: [Emerging/Growing/Mature/Declining]

## Performance Characteristics
| Metric | Value | Benchmark | Source |
|--------|-------|-----------|---------|
| [Metric 1] | [Value] | [Comparison] | [Source] |

## Implementation Analysis
### Best Practices
1. [Practice 1]: [Description and rationale]

### Decision Framework
**Recommended Choice**: [Technology/Approach]
**Rationale**: [Detailed reasoning]
```

## Quality Assurance

### Source Credibility Assessment
- Authority scoring (0-10)
- Accuracy validation
- Bias detection
- Currency evaluation

### Validation Framework
```typescript
interface SourceCredibility {
  authority: number;    // 0-10: Author/organization expertise
  accuracy: number;     // 0-10: Factual correctness
  objectivity: number;  // 0-10: Bias assessment
  currency: number;     // 0-10: Information freshness
  overall: number;      // Weighted average
}
```

## Advanced Features

### Recursive Research
Iteratively deepens research scope based on initial findings:

```typescript
function recursiveResearch(topic: string, depth: number = 0): ResearchResult {
  const broadResults = await perplexitySearch(topic, "comprehensive");
  const subtopics = extractSubtopics(broadResults);
  
  if (depth < MAX_DEPTH) {
    const deepResults = await Promise.all(
      subtopics.map(subtopic => recursiveResearch(subtopic, depth + 1))
    );
    return synthesizeResults(broadResults, deepResults);
  }
  
  return broadResults;
}
```

### Multi-Perspective Analysis
Gathers information from multiple viewpoints:

```typescript
const perspectives = ["academic", "industry", "regulatory", "consumer"];
const results = await Promise.all(
  perspectives.map(perspective => 
    perplexitySearch(topic, perspective)
  )
);
```

### Intelligent Caching
Optimizes performance through smart caching:

```typescript
interface CacheStrategy {
  ttl: number;           // Time to live
  priority: number;      // Cache priority
  invalidation: string[]; // Invalidation triggers
  compression: boolean;   // Compress cached data
}
```

## Integration with aiGI Workflow

### Input from Other Modes
- **Architect Mode**: Research requirements and specifications
- **Orchestrator Mode**: Research priorities and resource allocation
- **Memory Manager Mode**: Historical research data and patterns

### Output to Other Modes
- **Code Mode**: Technical implementation guidance
- **Critic Mode**: Validation criteria and benchmarks
- **Final Assembly Mode**: Research findings and documentation

## Performance Optimization

### Parallel Processing
```typescript
// Execute research streams in parallel
const streams = [
  { name: "technical", focus: "technical_aspects" },
  { name: "market", focus: "market_dynamics" },
  { name: "regulatory", focus: "compliance_requirements" }
];

const results = await Promise.all(
  streams.map(stream => executeResearchStream(topic, stream))
);
```

### Resource Management
- Maximum concurrent requests: 5
- Batch processing for related queries
- Intelligent timeout handling
- Graceful degradation on failures

## Testing

Run the comprehensive test suite:

```bash
python test_mcp_researcher.py
```

### Test Coverage
- ✅ Research session management
- ✅ Technical research workflows
- ✅ Market research workflows
- ✅ Academic research workflows
- ✅ Quality validation
- ✅ MCP server integration
- ✅ Error handling
- ✅ Performance testing
- ✅ Concurrent execution

## Best Practices

### Research Planning
1. **Define Clear Objectives**: Start with specific, measurable goals
2. **Scope Appropriately**: Balance depth vs. breadth
3. **Plan Validation Strategy**: Determine verification methods
4. **Consider Time Constraints**: Adjust depth based on available time

### Source Management
1. **Diversify Sources**: Use multiple MCP servers and information types
2. **Validate Credibility**: Apply credibility assessment frameworks
3. **Track Provenance**: Maintain clear citation tracking
4. **Update Regularly**: Refresh time-sensitive information

### Quality Assurance
1. **Cross-Reference Findings**: Validate across multiple sources
2. **Assess Confidence Levels**: Provide confidence scores
3. **Identify Limitations**: State research limitations clearly
4. **Document Methodology**: Maintain transparent processes

## Error Handling

### Connection Resilience
```typescript
class MCPConnectionManager {
  async executeWithResilience<T>(
    serverName: string, 
    operation: () => Promise<T>
  ): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      return await this.handleConnectionError(serverName, operation, error);
    }
  }
}
```

### Fallback Strategies
- Retry with exponential backoff
- Alternative MCP server usage
- Cached result fallback
- Graceful degradation

## Continuous Learning

### Pattern Recognition
```typescript
interface ResearchPattern {
  context: string;
  approach: string;
  effectiveness: number;
  conditions: string[];
  outcomes: string[];
}
```

### Quality Feedback Loop
- User feedback incorporation
- Research strategy optimization
- Workflow refinement
- Performance improvement

## Examples

See `examples.md` for detailed implementation examples including:

1. **Technical Library Evaluation**: Comprehensive React state management comparison
2. **Market Intelligence Research**: AI development tools market analysis
3. **Academic Literature Review**: Explainable AI systematic review
4. **Competitive Intelligence**: Detailed competitor analysis
5. **Regulatory Research**: Compliance landscape analysis

## Workflow Templates

See `workflows.md` for structured workflow templates including:

1. **Technology Evaluation Workflow**
2. **Competitive Intelligence Workflow**
3. **Market Opportunity Assessment**
4. **Academic Literature Review**
5. **Regulatory Compliance Research**

## Contributing

To extend the MCP Researcher mode:

1. **Add New Workflows**: Define in `workflows.md`
2. **Create Templates**: Add output templates for new domains
3. **Extend Tests**: Add test cases in `test_mcp_researcher.py`
4. **Update Configuration**: Modify `config.yaml` for new settings

## License

This implementation is part of the SAFLA project and follows the project's licensing terms.

---

**Ready for real-world research tasks** 🚀

The MCP Researcher mode provides a comprehensive, practical framework for conducting structured research across multiple domains using the power of MCP integration.