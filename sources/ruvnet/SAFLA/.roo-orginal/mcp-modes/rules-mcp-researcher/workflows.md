# MCP Researcher Workflow Templates

## Quick Start Guide

### 1. Initialize Research Session
```bash
# Start MCP Researcher mode
new_task: mcp-researcher

# Basic research command structure
"Research [TOPIC] focusing on [SPECIFIC_ASPECTS] for [PURPOSE/CONTEXT]"
```

### 2. Research Command Templates

#### Technical Research
```bash
"Conduct technical research on [TECHNOLOGY/FRAMEWORK] including:
- Current best practices and implementation patterns
- Performance characteristics and benchmarks
- Ecosystem maturity and community support
- Integration requirements and compatibility
- Comparison with alternatives"
```

#### Market Research
```bash
"Analyze the market for [PRODUCT/SERVICE CATEGORY] including:
- Market size, growth trends, and projections
- Competitive landscape and key players
- Customer segments and adoption patterns
- Pricing strategies and business models
- Emerging opportunities and threats"
```

#### Academic Research
```bash
"Conduct literature review on [RESEARCH TOPIC] covering:
- Recent publications and seminal works
- Research methodologies and approaches
- Key findings and consensus areas
- Identified gaps and future directions
- Practical applications and implications"
```

## Workflow Templates

### Template 1: Technology Evaluation Workflow

```yaml
name: "Technology Evaluation"
description: "Comprehensive evaluation of technologies, frameworks, or tools"
phases:
  - name: "requirements_analysis"
    description: "Define evaluation criteria and requirements"
    tools: ["safla"]
    outputs: ["requirements_matrix", "evaluation_criteria"]
  
  - name: "candidate_identification"
    description: "Identify and shortlist candidate technologies"
    tools: ["context7", "perplexity"]
    outputs: ["candidate_list", "initial_screening"]
  
  - name: "detailed_analysis"
    description: "Deep dive into each candidate"
    tools: ["context7", "perplexity"]
    parallel: true
    outputs: ["technical_specs", "documentation_analysis", "community_assessment"]
  
  - name: "comparative_evaluation"
    description: "Compare candidates against criteria"
    tools: ["safla", "perplexity"]
    outputs: ["comparison_matrix", "scoring_results"]
  
  - name: "recommendation_synthesis"
    description: "Generate final recommendations"
    tools: ["safla"]
    outputs: ["recommendation_report", "implementation_roadmap"]

parameters:
  evaluation_criteria:
    - "performance"
    - "scalability"
    - "maintainability"
    - "community_support"
    - "learning_curve"
    - "cost_of_ownership"
  
  output_format: "technical_analysis_template"
  confidence_threshold: 0.8
  validation_required: true
```

### Template 2: Competitive Intelligence Workflow

```yaml
name: "Competitive Intelligence"
description: "Comprehensive competitor analysis and market positioning"
phases:
  - name: "competitor_identification"
    description: "Identify direct and indirect competitors"
    tools: ["perplexity"]
    outputs: ["competitor_list", "market_map"]
  
  - name: "company_profiling"
    description: "Create detailed profiles for each competitor"
    tools: ["perplexity"]
    parallel: true
    outputs: ["company_profiles", "financial_analysis", "product_analysis"]
  
  - name: "strategic_analysis"
    description: "Analyze strategies and positioning"
    tools: ["perplexity"]
    outputs: ["strategy_assessment", "positioning_analysis"]
  
  - name: "swot_analysis"
    description: "Identify strengths, weaknesses, opportunities, threats"
    tools: ["perplexity", "safla"]
    outputs: ["swot_matrix", "vulnerability_assessment"]
  
  - name: "intelligence_synthesis"
    description: "Generate actionable intelligence"
    tools: ["safla"]
    outputs: ["intelligence_report", "strategic_recommendations"]

parameters:
  analysis_depth: "comprehensive"
  time_horizon: "12_months"
  focus_areas:
    - "product_strategy"
    - "market_positioning"
    - "financial_performance"
    - "technology_stack"
    - "partnerships"
  
  output_format: "market_intelligence_template"
  update_frequency: "monthly"
```

### Template 3: Market Opportunity Assessment

```yaml
name: "Market Opportunity Assessment"
description: "Identify and evaluate market opportunities"
phases:
  - name: "market_landscape"
    description: "Map the overall market landscape"
    tools: ["perplexity"]
    outputs: ["market_overview", "size_and_growth", "key_players"]
  
  - name: "trend_analysis"
    description: "Identify and analyze market trends"
    tools: ["perplexity"]
    outputs: ["trend_report", "signal_analysis", "disruption_factors"]
  
  - name: "customer_analysis"
    description: "Analyze customer segments and needs"
    tools: ["perplexity"]
    outputs: ["customer_segments", "needs_analysis", "pain_points"]
  
  - name: "gap_identification"
    description: "Identify market gaps and opportunities"
    tools: ["perplexity", "safla"]
    outputs: ["gap_analysis", "opportunity_matrix"]
  
  - name: "opportunity_evaluation"
    description: "Evaluate and prioritize opportunities"
    tools: ["safla"]
    outputs: ["opportunity_assessment", "prioritization_matrix", "go_to_market_strategy"]

parameters:
  market_scope: "global"
  time_horizon: "3_years"
  opportunity_criteria:
    - "market_size"
    - "growth_potential"
    - "competitive_intensity"
    - "barriers_to_entry"
    - "strategic_fit"
  
  output_format: "market_intelligence_template"
  risk_assessment: true
```

### Template 4: Academic Literature Review

```yaml
name: "Academic Literature Review"
description: "Systematic review of academic literature"
phases:
  - name: "search_strategy"
    description: "Define search strategy and criteria"
    tools: ["safla"]
    outputs: ["search_strategy", "inclusion_criteria", "quality_criteria"]
  
  - name: "literature_search"
    description: "Comprehensive literature search"
    tools: ["perplexity"]
    outputs: ["paper_list", "source_databases", "search_results"]
  
  - name: "screening_and_selection"
    description: "Screen and select relevant papers"
    tools: ["perplexity", "safla"]
    outputs: ["selected_papers", "exclusion_log", "quality_assessment"]
  
  - name: "data_extraction"
    description: "Extract key data from selected papers"
    tools: ["perplexity"]
    parallel: true
    outputs: ["extracted_data", "methodology_analysis", "findings_summary"]
  
  - name: "synthesis_and_analysis"
    description: "Synthesize findings and identify patterns"
    tools: ["safla"]
    outputs: ["synthesis_report", "meta_analysis", "research_gaps"]
  
  - name: "conclusion_and_recommendations"
    description: "Draw conclusions and future directions"
    tools: ["safla"]
    outputs: ["literature_review", "research_agenda", "methodology_recommendations"]

parameters:
  review_type: "systematic"
  time_period: "last_5_years"
  quality_threshold: "peer_reviewed"
  synthesis_method: "narrative_with_quantitative"
  
  output_format: "academic_synthesis_template"
  citation_style: "apa"
```

### Template 5: Regulatory Compliance Research

```yaml
name: "Regulatory Compliance Research"
description: "Comprehensive regulatory landscape analysis"
phases:
  - name: "regulatory_mapping"
    description: "Map applicable regulations and jurisdictions"
    tools: ["perplexity"]
    outputs: ["regulatory_map", "jurisdiction_analysis", "applicability_matrix"]
  
  - name: "current_requirements"
    description: "Analyze current regulatory requirements"
    tools: ["perplexity"]
    outputs: ["requirement_analysis", "compliance_checklist", "documentation_needs"]
  
  - name: "upcoming_changes"
    description: "Identify upcoming regulatory changes"
    tools: ["perplexity"]
    outputs: ["change_timeline", "impact_assessment", "preparation_requirements"]
  
  - name: "compliance_strategy"
    description: "Develop compliance strategy and implementation plan"
    tools: ["perplexity", "safla"]
    outputs: ["compliance_strategy", "implementation_plan", "risk_mitigation"]
  
  - name: "monitoring_framework"
    description: "Establish ongoing monitoring and updates"
    tools: ["safla"]
    outputs: ["monitoring_plan", "update_procedures", "compliance_dashboard"]

parameters:
  jurisdictions: ["us", "eu", "uk", "canada"]
  regulatory_areas:
    - "data_protection"
    - "financial_services"
    - "healthcare"
    - "environmental"
  
  compliance_level: "enterprise"
  output_format: "regulatory_analysis_template"
  monitoring_frequency: "quarterly"
```

## Advanced Workflow Patterns

### Pattern 1: Multi-Phase Deep Dive

```typescript
// Multi-phase research with increasing specificity
async function multiPhaseDeepDive(topic: string) {
  // Phase 1: Broad landscape scan
  const landscape = await broadLandscapeScan(topic);
  
  // Phase 2: Identify key areas of interest
  const keyAreas = extractKeyAreas(landscape);
  
  // Phase 3: Deep dive into each area
  const deepAnalysis = await Promise.all(
    keyAreas.map(area => conductDeepAnalysis(area))
  );
  
  // Phase 4: Cross-area synthesis
  const synthesis = synthesizeAcrossAreas(deepAnalysis);
  
  // Phase 5: Strategic implications
  const implications = deriveStrategicImplications(synthesis);
  
  return {
    landscape,
    keyAreas,
    deepAnalysis,
    synthesis,
    implications
  };
}
```

### Pattern 2: Parallel Investigation

```typescript
// Parallel research streams with convergence
async function parallelInvestigation(topic: string) {
  // Define parallel research streams
  const streams = [
    { name: "technical", focus: "technical_aspects" },
    { name: "market", focus: "market_dynamics" },
    { name: "regulatory", focus: "compliance_requirements" },
    { name: "competitive", focus: "competitive_landscape" }
  ];
  
  // Execute streams in parallel
  const results = await Promise.all(
    streams.map(stream => executeResearchStream(topic, stream))
  );
  
  // Converge findings
  const convergence = convergeFindings(results);
  
  // Identify synergies and conflicts
  const analysis = analyzeCrossStreamPatterns(convergence);
  
  return {
    streamResults: results,
    convergence,
    analysis
  };
}
```

### Pattern 3: Iterative Refinement

```typescript
// Iterative research with refinement loops
async function iterativeRefinement(topic: string, maxIterations: number = 3) {
  let currentScope = defineInitialScope(topic);
  let iteration = 0;
  let results = [];
  
  while (iteration < maxIterations) {
    // Conduct research for current scope
    const iterationResult = await conductScopedResearch(currentScope);
    results.push(iterationResult);
    
    // Analyze gaps and refinement needs
    const gaps = identifyKnowledgeGaps(iterationResult);
    const refinements = determineRefinements(gaps);
    
    // Check if further refinement is needed
    if (refinements.length === 0 || iteration === maxIterations - 1) {
      break;
    }
    
    // Refine scope for next iteration
    currentScope = refineScope(currentScope, refinements);
    iteration++;
  }
  
  // Synthesize across iterations
  return synthesizeIterativeResults(results);
}
```

### Pattern 4: Validation-Driven Research

```typescript
// Research with continuous validation and fact-checking
async function validationDrivenResearch(topic: string) {
  // Initial research phase
  const initialFindings = await conductInitialResearch(topic);
  
  // Validation phase
  const validationResults = await validateFindings(initialFindings);
  
  // Identify discrepancies and uncertainties
  const discrepancies = identifyDiscrepancies(validationResults);
  const uncertainties = identifyUncertainties(validationResults);
  
  // Additional research to resolve discrepancies
  const clarificationResearch = await resolveDis crepancies(discrepancies);
  
  // Final synthesis with confidence scoring
  const finalResults = synthesizeWithConfidence(
    initialFindings,
    validationResults,
    clarificationResearch
  );
  
  return {
    findings: finalResults,
    validationReport: validationResults,
    confidenceScores: calculateConfidenceScores(finalResults),
    limitations: identifyLimitations(finalResults)
  };
}
```

## Workflow Execution Examples

### Example 1: Technology Stack Evaluation

```bash
# Command
"Evaluate modern web development stacks for a high-traffic e-commerce platform, 
comparing React/Next.js, Vue/Nuxt, and Svelte/SvelteKit across performance, 
developer experience, ecosystem maturity, and scalability."

# Workflow Execution
Phase 1: Requirements Analysis
- Define evaluation criteria
- Establish performance benchmarks
- Identify scalability requirements

Phase 2: Technology Documentation Research
- Context7: Get comprehensive docs for each framework
- Extract performance characteristics
- Analyze ecosystem and tooling

Phase 3: Industry Practice Analysis
- Perplexity: Research real-world implementations
- Gather performance benchmarks
- Analyze adoption patterns

Phase 4: Comparative Evaluation
- Create comparison matrix
- Score against criteria
- Identify trade-offs

Phase 5: Recommendation Generation
- Synthesize findings
- Generate recommendations
- Create implementation roadmap
```

### Example 2: Market Entry Strategy Research

```bash
# Command
"Research market entry opportunities for AI-powered customer service tools 
in the European market, including regulatory requirements, competitive 
landscape, and customer adoption patterns."

# Workflow Execution
Phase 1: Market Landscape Mapping
- Perplexity: European AI/customer service market overview
- Identify market size and growth trends
- Map key players and market segments

Phase 2: Regulatory Analysis
- Research GDPR and AI Act implications
- Analyze data protection requirements
- Identify compliance obligations

Phase 3: Competitive Intelligence
- Profile major competitors in EU market
- Analyze pricing and positioning strategies
- Identify market gaps and opportunities

Phase 4: Customer Analysis
- Research adoption patterns and preferences
- Analyze customer segments and needs
- Identify decision-making factors

Phase 5: Entry Strategy Development
- Synthesize market intelligence
- Develop entry strategy recommendations
- Create implementation timeline
```

## Quality Assurance Workflows

### QA Workflow 1: Source Validation

```yaml
name: "Source Validation Workflow"
steps:
  - name: "credibility_assessment"
    criteria:
      - authority_score: ">= 7"
      - recency: "<= 12_months"
      - peer_review: "preferred"
    
  - name: "cross_reference_validation"
    requirements:
      - minimum_sources: 3
      - source_diversity: "high"
      - consensus_threshold: 0.7
    
  - name: "bias_detection"
    checks:
      - commercial_bias: "flag_if_detected"
      - confirmation_bias: "validate_opposing_views"
      - selection_bias: "ensure_representative_sampling"
    
  - name: "fact_verification"
    methods:
      - primary_source_verification: "when_possible"
      - expert_validation: "for_technical_claims"
      - statistical_verification: "for_quantitative_data"
```

### QA Workflow 2: Completeness Assessment

```yaml
name: "Completeness Assessment Workflow"
dimensions:
  - name: "scope_coverage"
    metrics:
      - topic_breadth: "percentage_covered"
      - depth_adequacy: "sufficient_detail"
      - perspective_diversity: "multiple_viewpoints"
    
  - name: "temporal_coverage"
    requirements:
      - current_state: "latest_information"
      - historical_context: "relevant_background"
      - future_trends: "forward_looking_analysis"
    
  - name: "stakeholder_perspectives"
    coverage:
      - technical_perspective: "implementation_details"
      - business_perspective: "commercial_implications"
      - user_perspective: "adoption_and_usage"
      - regulatory_perspective: "compliance_considerations"
```

## Integration Patterns

### Integration Pattern 1: aiGI Workflow Integration

```typescript
// Integration with other aiGI modes
interface aiGIIntegration {
  inputModes: {
    architect: "research_requirements",
    orchestrator: "research_priorities",
    memoryManager: "historical_context"
  },
  
  outputModes: {
    code: "implementation_guidance",
    critic: "validation_criteria",
    finalAssembly: "research_documentation"
  },
  
  collaborationProtocol: {
    requestFormat: "structured_research_request",
    responseFormat: "standardized_research_output",
    feedbackLoop: "continuous_refinement"
  }
}
```

### Integration Pattern 2: External System Integration

```typescript
// Integration with external systems and APIs
interface ExternalIntegration {
  mcpServers: {
    perplexity: "real_time_search",
    context7: "technical_documentation",
    safla: "session_management"
  },
  
  dataFormats: {
    input: ["json", "yaml", "markdown"],
    output: ["markdown", "json", "pdf", "html"]
  },
  
  apiIntegration: {
    rateLimiting: "respect_api_limits",
    errorHandling: "graceful_degradation",
    caching: "intelligent_caching"
  }
}
```

## Performance Optimization Patterns

### Optimization Pattern 1: Parallel Processing

```typescript
// Optimize research through parallel execution
async function optimizedParallelResearch(topics: string[]) {
  // Group related topics for batch processing
  const topicGroups = groupRelatedTopics(topics);
  
  // Execute groups in parallel with resource management
  const results = await Promise.all(
    topicGroups.map(group => 
      executeWithResourceLimits(group, {
        maxConcurrency: 3,
        timeout: 30000,
        retryPolicy: "exponential_backoff"
      })
    )
  );
  
  // Merge and deduplicate results
  return mergeAndDeduplicate(results);
}
```

### Optimization Pattern 2: Intelligent Caching

```typescript
// Implement intelligent caching for research efficiency
class ResearchCache {
  private cache = new Map<string, CachedResult>();
  
  async get(query: string, maxAge: number = 3600000): Promise<CachedResult | null> {
    const cached = this.cache.get(query);
    
    if (cached && (Date.now() - cached.timestamp) < maxAge) {
      return cached;
    }
    
    return null;
  }
  
  async set(query: string, result: any, metadata: CacheMetadata): Promise<void> {
    this.cache.set(query, {
      result,
      timestamp: Date.now(),
      metadata,
      accessCount: 0
    });
  }
  
  // Implement cache eviction and optimization strategies
  optimizeCache(): void {
    // Remove least recently used items
    // Compress frequently accessed items
    // Update cache based on usage patterns
  }
}
```

This comprehensive workflow system provides structured approaches for conducting various types of research using the MCP Researcher mode, ensuring consistent quality and actionable outcomes across different research domains.