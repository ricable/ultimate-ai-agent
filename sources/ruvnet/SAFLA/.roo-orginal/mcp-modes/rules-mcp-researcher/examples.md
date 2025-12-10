# MCP Researcher Mode Examples

## Example 1: Technical Library Evaluation

### Research Objective
Evaluate and compare React state management libraries for a large-scale enterprise application.

### Workflow Implementation

```typescript
// Technical Research Workflow Example
async function evaluateReactStateManagement() {
  // Phase 1: Scope Definition
  const researchScope = {
    objective: "Compare React state management solutions for enterprise apps",
    criteria: ["performance", "scalability", "developer_experience", "ecosystem"],
    constraints: ["enterprise_requirements", "team_size_100+", "typescript_support"]
  };

  // Phase 2: Library Identification using Context7
  const librarySearch = await use_mcp_tool({
    server_name: "context7",
    tool_name: "resolve-library-id",
    arguments: {
      libraryName: "react state management redux zustand jotai recoil"
    }
  });

  // Phase 3: Documentation Analysis
  const libraries = ["redux", "zustand", "jotai", "recoil"];
  const documentationAnalysis = await Promise.all(
    libraries.map(async (lib) => {
      const libId = await use_mcp_tool({
        server_name: "context7",
        tool_name: "resolve-library-id",
        arguments: { libraryName: lib }
      });

      return use_mcp_tool({
        server_name: "context7",
        tool_name: "get-library-docs",
        arguments: {
          context7CompatibleLibraryID: libId.selectedLibrary,
          topic: "enterprise patterns performance typescript",
          tokens: 8000
        }
      });
    })
  );

  // Phase 4: Community and Industry Analysis
  const industryAnalysis = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "React state management comparison 2024: Redux vs Zustand vs Jotai vs Recoil for enterprise applications",
      systemContent: "Provide detailed technical comparison focusing on performance benchmarks, scalability, TypeScript support, and enterprise adoption patterns. Include recent community discussions and expert opinions.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.3,
      max_tokens: 4000
    }
  });

  // Phase 5: Performance Benchmarking Research
  const performanceData = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "React state management performance benchmarks bundle size memory usage 2024",
      systemContent: "Focus on quantitative performance data, bundle sizes, memory usage, and rendering performance metrics for React state management libraries.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.2
    }
  });

  // Phase 6: Enterprise Adoption Analysis
  const enterpriseAdoption = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "Enterprise companies using Redux Zustand Jotai Recoil case studies migration stories",
      systemContent: "Research enterprise adoption patterns, migration experiences, and case studies from large companies using these state management solutions.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 7: Synthesis and Recommendation
  return synthesizeTechnicalComparison({
    scope: researchScope,
    documentation: documentationAnalysis,
    industry: industryAnalysis,
    performance: performanceData,
    enterprise: enterpriseAdoption
  });
}
```

### Expected Output Structure

```markdown
# React State Management Library Evaluation

## Executive Summary
**Recommendation**: Zustand for new projects, Redux Toolkit for existing Redux codebases
**Confidence Level**: High (8.5/10)
**Key Factors**: Performance, developer experience, migration complexity

## Detailed Analysis

### Library Comparison Matrix
| Library | Bundle Size | Performance | DX Score | Enterprise Adoption | TypeScript Support |
|---------|-------------|-------------|----------|-------------------|-------------------|
| Redux Toolkit | 47KB | 8/10 | 7/10 | 9/10 | 9/10 |
| Zustand | 8KB | 9/10 | 9/10 | 7/10 | 8/10 |
| Jotai | 13KB | 8/10 | 8/10 | 6/10 | 9/10 |
| Recoil | 79KB | 7/10 | 7/10 | 5/10 | 7/10 |

### Implementation Recommendations
1. **New Enterprise Projects**: Start with Zustand
2. **Existing Redux Projects**: Migrate to Redux Toolkit
3. **Complex State Requirements**: Consider Jotai for atomic state management
4. **Facebook Ecosystem**: Recoil if already using other Meta tools
```

## Example 2: Market Intelligence Research

### Research Objective
Analyze the competitive landscape and market opportunities in the AI-powered development tools sector.

### Workflow Implementation

```typescript
// Market Research Workflow Example
async function analyzeAIDevelopmentToolsMarket() {
  // Phase 1: Market Landscape Mapping
  const marketOverview = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "AI-powered development tools market size growth rate key players 2024",
      systemContent: "Provide comprehensive market analysis including market size, growth projections, key players, and market segments. Focus on quantitative data and financial metrics.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      return_images: true,
      temperature: 0.3
    }
  });

  // Phase 2: Competitive Analysis
  const competitorAnalysis = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "GitHub Copilot vs Cursor vs Codeium vs Tabnine competitive analysis features pricing",
      systemContent: "Analyze competitive positioning, feature comparison, pricing strategies, and market share of major AI coding assistants. Include recent product updates and strategic moves.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.2
    }
  });

  // Phase 3: Investment and Funding Analysis
  const fundingTrends = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "AI development tools startups funding rounds venture capital investment 2024",
      systemContent: "Research recent funding rounds, valuations, investor preferences, and emerging startups in the AI development tools space. Include analysis of investment trends and patterns.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 4: Technology Trends Analysis
  const technologyTrends = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "Emerging AI technologies for software development code generation testing debugging 2024",
      systemContent: "Identify emerging technologies, breakthrough innovations, and future trends in AI-powered development tools. Focus on technical capabilities and potential disruptions.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 5: Customer Adoption Patterns
  const adoptionPatterns = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "Developer adoption AI coding tools enterprise integration challenges benefits survey data",
      systemContent: "Research developer adoption rates, enterprise integration patterns, challenges, benefits, and user satisfaction data for AI development tools.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 6: Regulatory and Ethical Considerations
  const regulatoryLandscape = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "AI development tools regulatory compliance intellectual property copyright concerns",
      systemContent: "Analyze regulatory requirements, intellectual property concerns, copyright issues, and compliance challenges for AI-powered development tools.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  return synthesizeMarketIntelligence({
    market: marketOverview,
    competitors: competitorAnalysis,
    funding: fundingTrends,
    technology: technologyTrends,
    adoption: adoptionPatterns,
    regulatory: regulatoryLandscape
  });
}
```

### Expected Output Structure

```markdown
# AI Development Tools Market Intelligence Report

## Executive Summary
**Market Size**: $2.3B (2024) → $8.7B (2029)
**Growth Rate**: 30.2% CAGR
**Key Opportunity**: Enterprise integration and specialized domain tools

## Market Landscape
### Leading Players
1. **GitHub (Microsoft)** - 40% market share, Copilot dominance
2. **Cursor** - 15% market share, rapid growth in enterprise
3. **Codeium** - 12% market share, strong free tier adoption
4. **Tabnine** - 10% market share, enterprise security focus

### Emerging Opportunities
- **Specialized Domain Tools**: 65% of developers want domain-specific AI assistants
- **Enterprise Security**: 78% of enterprises cite security as primary concern
- **Integration Platforms**: 52% want better IDE and workflow integration

## Strategic Recommendations
1. **Focus on Enterprise Security**: Develop on-premise/private cloud solutions
2. **Domain Specialization**: Create vertical-specific AI assistants
3. **Integration Ecosystem**: Build comprehensive developer workflow integration
```

## Example 3: Academic Literature Review

### Research Objective
Conduct a systematic literature review on explainable AI (XAI) methods and their practical applications.

### Workflow Implementation

```typescript
// Academic Research Workflow Example
async function conductXAILiteratureReview() {
  // Phase 1: Literature Discovery
  const literatureSearch = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "explainable AI XAI literature review recent papers 2023 2024 LIME SHAP attention mechanisms",
      systemContent: "Search for recent academic papers on explainable AI, focusing on peer-reviewed publications, conference proceedings, and journal articles. Include seminal works and recent advances.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.2
    }
  });

  // Phase 2: Methodological Analysis
  const methodologyAnalysis = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "explainable AI methods comparison LIME SHAP GradCAM attention visualization techniques evaluation metrics",
      systemContent: "Analyze different XAI methodologies, their strengths, limitations, and evaluation approaches. Focus on technical details and comparative studies.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.1
    }
  });

  // Phase 3: Application Domains Research
  const applicationDomains = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "explainable AI applications healthcare finance autonomous vehicles natural language processing computer vision",
      systemContent: "Research practical applications of explainable AI across different domains, including case studies, implementation challenges, and domain-specific requirements.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 4: Evaluation Frameworks
  const evaluationFrameworks = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "explainable AI evaluation metrics human studies user studies faithfulness plausibility comprehensibility",
      systemContent: "Research evaluation methodologies for XAI systems, including quantitative metrics, human evaluation studies, and standardized benchmarks.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 5: Future Directions and Challenges
  const futureDirections = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: "explainable AI future research directions challenges open problems causality counterfactual explanations",
      systemContent: "Identify current challenges, limitations, and future research directions in explainable AI. Focus on open problems and emerging research areas.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 6: Technical Implementation Analysis
  const technicalImplementation = await use_mcp_tool({
    server_name: "context7",
    tool_name: "resolve-library-id",
    arguments: {
      libraryName: "explainable AI LIME SHAP interpretability"
    }
  });

  const implementationDocs = await use_mcp_tool({
    server_name: "context7",
    tool_name: "get-library-docs",
    arguments: {
      context7CompatibleLibraryID: "/slundberg/shap",
      topic: "explainability methods implementation examples",
      tokens: 6000
    }
  });

  return synthesizeAcademicResearch({
    literature: literatureSearch,
    methodology: methodologyAnalysis,
    applications: applicationDomains,
    evaluation: evaluationFrameworks,
    future: futureDirections,
    implementation: implementationDocs
  });
}
```

### Expected Output Structure

```markdown
# Explainable AI: A Systematic Literature Review

## Abstract
This systematic review analyzes 127 papers on explainable AI published between 2022-2024, 
identifying key methodological advances, application domains, and future research directions.

## Literature Overview
### Seminal Works
- **Ribeiro et al. (2016)**: LIME - Local Interpretable Model-agnostic Explanations
- **Lundberg & Lee (2017)**: SHAP - SHapley Additive exPlanations
- **Selvaraju et al. (2017)**: Grad-CAM for visual explanations

### Recent Advances (2023-2024)
- **Causal Explanations**: 23 papers focusing on causal inference in XAI
- **Counterfactual Methods**: 18 papers on counterfactual explanation generation
- **Multi-modal Explanations**: 15 papers on explaining multi-modal AI systems

## Methodological Analysis
### Method Comparison
| Method | Scope | Fidelity | Efficiency | Human Comprehensibility |
|--------|-------|----------|------------|------------------------|
| LIME | Local | Medium | High | High |
| SHAP | Local/Global | High | Medium | Medium |
| Attention | Local | Low | High | Medium |
| Counterfactuals | Local | High | Low | High |

## Research Gaps and Future Directions
1. **Standardized Evaluation**: Need for unified evaluation frameworks
2. **Causal Explanations**: Integration of causal reasoning in explanations
3. **Interactive Explanations**: Development of interactive explanation interfaces
4. **Domain-Specific Methods**: Tailored approaches for specific application domains
```

## Example 4: Competitive Intelligence Research

### Research Objective
Analyze a specific competitor's strategy, market position, and potential threats/opportunities.

### Workflow Implementation

```typescript
// Competitive Intelligence Workflow Example
async function analyzeCompetitorStrategy(competitorName: string) {
  // Phase 1: Company Overview and Recent Developments
  const companyOverview = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${competitorName} company overview recent news product launches strategy 2024`,
      systemContent: "Provide comprehensive company analysis including recent strategic moves, product launches, partnerships, and market positioning. Focus on factual, recent information.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.2
    }
  });

  // Phase 2: Financial Performance and Funding
  const financialAnalysis = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${competitorName} financial performance revenue growth funding valuation investors`,
      systemContent: "Analyze financial metrics, revenue growth, funding rounds, valuation, and investor information. Include recent financial reports and analyst opinions.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 3: Product and Technology Analysis
  const productAnalysis = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${competitorName} products features technology stack competitive advantages`,
      systemContent: "Analyze product offerings, key features, technology stack, competitive advantages, and differentiation factors. Include user reviews and expert opinions.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 4: Market Position and Customer Base
  const marketPosition = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${competitorName} market share customer base target market pricing strategy`,
      systemContent: "Research market position, customer segments, pricing strategy, and competitive positioning. Include market share data and customer testimonials.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 5: Strategic Partnerships and Ecosystem
  const partnerships = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${competitorName} partnerships integrations ecosystem strategic alliances`,
      systemContent: "Identify key partnerships, integrations, ecosystem relationships, and strategic alliances. Analyze the strategic value and impact of these relationships.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 6: Strengths, Weaknesses, and Vulnerabilities
  const swotAnalysis = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${competitorName} strengths weaknesses challenges criticism user complaints`,
      systemContent: "Analyze company strengths, weaknesses, challenges, and areas of criticism. Include user feedback, expert analysis, and identified vulnerabilities.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  return synthesizeCompetitiveIntelligence({
    company: companyOverview,
    financial: financialAnalysis,
    product: productAnalysis,
    market: marketPosition,
    partnerships: partnerships,
    swot: swotAnalysis
  });
}
```

## Example 5: Regulatory Research

### Research Objective
Analyze regulatory requirements and compliance considerations for a specific industry or technology.

### Workflow Implementation

```typescript
// Regulatory Research Workflow Example
async function analyzeRegulatoryLandscape(domain: string, jurisdiction: string) {
  // Phase 1: Current Regulatory Framework
  const currentRegulations = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${domain} regulations ${jurisdiction} current laws compliance requirements`,
      systemContent: "Research current regulatory framework, laws, and compliance requirements. Focus on official sources and recent regulatory updates.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true,
      temperature: 0.1
    }
  });

  // Phase 2: Upcoming Regulatory Changes
  const upcomingChanges = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${domain} upcoming regulations ${jurisdiction} proposed laws regulatory changes 2024 2025`,
      systemContent: "Identify upcoming regulatory changes, proposed legislation, and regulatory trends. Include timelines and implementation dates.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 3: Compliance Best Practices
  const compliancePractices = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${domain} compliance best practices ${jurisdiction} implementation guidelines industry standards`,
      systemContent: "Research compliance best practices, implementation guidelines, and industry standards. Include case studies and expert recommendations.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  // Phase 4: Enforcement and Penalties
  const enforcement = await use_mcp_tool({
    server_name: "perplexity",
    tool_name: "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    arguments: {
      userContent: `${domain} regulatory enforcement ${jurisdiction} penalties fines violations case studies`,
      systemContent: "Research enforcement actions, penalties, fines, and violation case studies. Analyze enforcement trends and regulatory priorities.",
      model: "llama-3.1-sonar-large-128k-online",
      return_citations: true
    }
  });

  return synthesizeRegulatoryAnalysis({
    current: currentRegulations,
    upcoming: upcomingChanges,
    compliance: compliancePractices,
    enforcement: enforcement
  });
}
```

## Best Practices for MCP Researcher Mode

### 1. Research Planning
- **Define Clear Objectives**: Start with specific, measurable research goals
- **Scope Appropriately**: Balance depth vs. breadth based on time constraints
- **Identify Key Questions**: Formulate specific questions to guide research
- **Plan Validation Strategy**: Determine how findings will be verified

### 2. Source Management
- **Diversify Sources**: Use multiple MCP servers and information types
- **Validate Credibility**: Apply source credibility assessment frameworks
- **Track Provenance**: Maintain clear citation and source tracking
- **Update Regularly**: Refresh information for time-sensitive research

### 3. Quality Assurance
- **Cross-Reference Findings**: Validate information across multiple sources
- **Assess Confidence Levels**: Provide confidence scores for all findings
- **Identify Limitations**: Clearly state research limitations and gaps
- **Document Methodology**: Maintain transparent research processes

### 4. Synthesis and Reporting
- **Structure Findings**: Use consistent templates and formats
- **Highlight Key Insights**: Emphasize actionable findings and recommendations
- **Provide Context**: Include relevant background and contextual information
- **Enable Action**: Ensure findings lead to clear next steps

### 5. Continuous Improvement
- **Learn from Feedback**: Incorporate user feedback into future research
- **Optimize Workflows**: Refine research processes based on outcomes
- **Update Methods**: Adapt to new tools and information sources
- **Share Knowledge**: Document and share successful research patterns