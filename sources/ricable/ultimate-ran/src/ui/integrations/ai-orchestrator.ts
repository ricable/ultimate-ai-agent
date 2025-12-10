/**
 * AI Orchestrator - Hybrid Claude + Gemini Intelligence
 * Combines Claude Agent SDK and Google ADK for enhanced RAN optimization
 *
 * @module ui/integrations/ai-orchestrator
 * @version 7.0.0-alpha.1
 */

import { ClaudeAgentIntegration } from './claude-agent-integration.js';
import { GoogleADKIntegration } from './google-adk-integration.js';
import type {
  CellStatus,
  InterferenceMatrix,
  OptimizationEvent,
  ParameterChange,
  ApprovalRequest
} from '../types.js';

// ============================================================================
// AI Orchestrator Configuration
// ============================================================================

export interface AIOrchestrationConfig {
  claude: {
    apiKey?: string;
    model?: string;
  };
  gemini: {
    apiKey?: string;
    model?: string;
  };
  strategy?: 'claude_primary' | 'gemini_primary' | 'consensus' | 'parallel';
  consensusThreshold?: number;
}

export interface OptimizationResult {
  source: 'claude' | 'gemini' | 'consensus';
  recommendations: ParameterChange[];
  reasoning: string;
  confidence: number;
  analysis?: string;
  risks?: string[];
  timeline?: string;
}

// ============================================================================
// AI Orchestrator Class
// ============================================================================

export class AIOrchestrator {
  private claude: ClaudeAgentIntegration;
  private gemini: GoogleADKIntegration;
  private config: Required<AIOrchestrationConfig>;

  constructor(config: AIOrchestrationConfig) {
    this.config = {
      claude: config.claude,
      gemini: config.gemini,
      strategy: config.strategy || 'consensus',
      consensusThreshold: config.consensusThreshold ?? 0.8
    };

    // Initialize both AI agents
    this.claude = new ClaudeAgentIntegration({
      apiKey: this.config.claude.apiKey,
      model: this.config.claude.model || process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101'
    });

    this.gemini = new GoogleADKIntegration({
      apiKey: this.config.gemini.apiKey,
      model: this.config.gemini.model || process.env.GOOGLE_AI_MODEL || 'gemini-2.0-flash-exp'
    });

    console.log('[AI Orchestrator] Initialized');
    console.log(`  Claude: ${this.config.claude.model || process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101'}`);
    console.log(`  Gemini: ${this.config.gemini.model || process.env.GOOGLE_AI_MODEL || 'gemini-2.0-flash-exp'}`);
    console.log(`  Strategy: ${this.config.strategy}`);
  }

  /**
   * Request RAN optimization using configured strategy
   */
  async requestOptimization(
    cells: CellStatus[],
    objective: string,
    interferenceMatrix?: InterferenceMatrix
  ): Promise<OptimizationResult> {
    switch (this.config.strategy) {
      case 'claude_primary':
        return this.optimizeWithClaude(cells, objective);

      case 'gemini_primary':
        return this.optimizeWithGemini(cells, objective, interferenceMatrix);

      case 'consensus':
        return this.optimizeWithConsensus(cells, objective, interferenceMatrix);

      case 'parallel':
        return this.optimizeInParallel(cells, objective, interferenceMatrix);

      default:
        throw new Error(`Unknown strategy: ${this.config.strategy}`);
    }
  }

  /**
   * Analyze cell performance using both AIs
   */
  async analyzeCellPerformance(cell: CellStatus): Promise<{
    claudeAnalysis: any;
    geminiAnalysis: any;
    combinedInsights: string[];
    priority: 'low' | 'medium' | 'high' | 'critical';
  }> {
    console.log(`[AI Orchestrator] Analyzing cell ${cell.cell_id} with both AIs...`);

    // Run analyses in parallel
    const [claudeResult, geminiResult] = await Promise.all([
      this.claude.analyzeCellPerformance(cell),
      this.gemini.detectAnomalies([cell])
    ]);

    // Combine insights
    const combinedInsights: string[] = [
      ...claudeResult.recommendations.slice(0, 3),
      ...geminiResult.summary.split('\n').filter(l => l.trim()).slice(0, 3)
    ];

    // Determine overall priority (take the higher one)
    const priorities = ['low', 'medium', 'high', 'critical'];
    const maxPriority = Math.max(
      priorities.indexOf(claudeResult.priority),
      geminiResult.anomalies.length > 0
        ? priorities.indexOf(geminiResult.anomalies[0].severity)
        : 0
    );

    return {
      claudeAnalysis: claudeResult,
      geminiAnalysis: geminiResult,
      combinedInsights,
      priority: priorities[maxPriority] as any
    };
  }

  /**
   * Validate approval request with both AIs
   */
  async validateApprovalRequest(request: ApprovalRequest): Promise<{
    claudeDecision: any;
    geminiDecision: any;
    finalDecision: boolean;
    reasoning: string;
  }> {
    console.log(`[AI Orchestrator] Validating approval ${request.id} with both AIs...`);

    // Get both validations
    const [claudeDecision, geminiDecision] = await Promise.all([
      this.claude.validateApprovalRequest(request),
      this.validateWithGemini(request)
    ]);

    // Final decision: both must approve for consensus
    const finalDecision =
      this.config.strategy === 'consensus'
        ? claudeDecision.approved && geminiDecision.approved
        : claudeDecision.approved || geminiDecision.approved;

    const reasoning = `
**Claude Assessment:** ${claudeDecision.approved ? 'APPROVED' : 'REJECTED'}
${claudeDecision.reasoning}

**Gemini Assessment:** ${geminiDecision.approved ? 'APPROVED' : 'REJECTED'}
${geminiDecision.reasoning}

**Final Decision:** ${finalDecision ? 'APPROVED' : 'REJECTED'}
${finalDecision
  ? 'Both AI agents recommend approval.'
  : 'At least one AI agent raised concerns. Review required.'
}`;

    return {
      claudeDecision,
      geminiDecision,
      finalDecision,
      reasoning
    };
  }

  /**
   * Optimize using Claude Agent SDK
   */
  private async optimizeWithClaude(
    cells: CellStatus[],
    objective: string
  ): Promise<OptimizationResult> {
    console.log('[AI Orchestrator] Optimizing with Claude...');

    const result = await this.claude.requestOptimization(cells, objective);

    return {
      source: 'claude',
      recommendations: result.recommendations,
      reasoning: result.reasoning,
      confidence: result.confidence
    };
  }

  /**
   * Optimize using Google ADK
   */
  private async optimizeWithGemini(
    cells: CellStatus[],
    objective: string,
    interferenceMatrix?: InterferenceMatrix
  ): Promise<OptimizationResult> {
    console.log('[AI Orchestrator] Optimizing with Gemini...');

    const [analysis, strategy] = await Promise.all([
      this.gemini.analyzeNetworkPerformance(cells, interferenceMatrix),
      this.gemini.generateOptimizationStrategy(objective, cells)
    ]);

    return {
      source: 'gemini',
      recommendations: analysis.recommendations,
      reasoning: analysis.analysis,
      confidence: analysis.confidence,
      analysis: strategy.strategy,
      risks: strategy.risks,
      timeline: strategy.timeline
    };
  }

  /**
   * Optimize using consensus between both AIs
   */
  private async optimizeWithConsensus(
    cells: CellStatus[],
    objective: string,
    interferenceMatrix?: InterferenceMatrix
  ): Promise<OptimizationResult> {
    console.log('[AI Orchestrator] Optimizing with consensus approach...');

    // Get recommendations from both
    const [claudeResult, geminiResult] = await Promise.all([
      this.claude.requestOptimization(cells, objective),
      this.gemini.analyzeNetworkPerformance(cells, interferenceMatrix)
    ]);

    // Find consensus recommendations (parameters changed in both)
    const consensusRecommendations: ParameterChange[] = [];

    for (const claudeRec of claudeResult.recommendations) {
      const geminiRec = geminiResult.recommendations.find(
        r => r.cell_id === claudeRec.cell_id && r.parameter === claudeRec.parameter
      );

      if (geminiRec) {
        // Both recommend changing this parameter
        // Use average of proposed values
        consensusRecommendations.push({
          cell_id: claudeRec.cell_id,
          parameter: claudeRec.parameter,
          old_value: claudeRec.old_value,
          new_value: (claudeRec.new_value + geminiRec.new_value) / 2,
          bounds: claudeRec.bounds
        });
      }
    }

    // Average confidence
    const avgConfidence = (claudeResult.confidence + geminiResult.confidence) / 2;

    const reasoning = `
**Consensus Analysis**

Claude's Reasoning:
${claudeResult.reasoning}

Gemini's Analysis:
${geminiResult.analysis}

**Consensus Recommendations:**
Found ${consensusRecommendations.length} parameters where both AIs agree on optimization.
Average confidence: ${(avgConfidence * 100).toFixed(1)}%
`;

    return {
      source: 'consensus',
      recommendations: consensusRecommendations,
      reasoning,
      confidence: avgConfidence,
      analysis: geminiResult.analysis
    };
  }

  /**
   * Run both optimizations in parallel and combine results
   */
  private async optimizeInParallel(
    cells: CellStatus[],
    objective: string,
    interferenceMatrix?: InterferenceMatrix
  ): Promise<OptimizationResult> {
    console.log('[AI Orchestrator] Running parallel optimization...');

    // Run both in parallel
    const [claudeResult, geminiResult] = await Promise.all([
      this.claude.requestOptimization(cells, objective),
      this.gemini.analyzeNetworkPerformance(cells, interferenceMatrix)
    ]);

    // Combine all unique recommendations
    const allRecommendations = [...claudeResult.recommendations];
    for (const geminiRec of geminiResult.recommendations) {
      const exists = allRecommendations.find(
        r => r.cell_id === geminiRec.cell_id && r.parameter === geminiRec.parameter
      );
      if (!exists) {
        allRecommendations.push(geminiRec);
      }
    }

    // Weighted average of confidence (Claude gets 60%, Gemini gets 40%)
    const confidence = claudeResult.confidence * 0.6 + geminiResult.confidence * 0.4;

    const reasoning = `
**Parallel Optimization Results**

${allRecommendations.length} total recommendations from both AI agents.

**Claude Recommendations:** ${claudeResult.recommendations.length} parameters
${claudeResult.reasoning}

**Gemini Recommendations:** ${geminiResult.recommendations.length} parameters
${geminiResult.analysis}

Combined confidence: ${(confidence * 100).toFixed(1)}%
`;

    return {
      source: 'consensus',
      recommendations: allRecommendations,
      reasoning,
      confidence,
      analysis: geminiResult.analysis
    };
  }

  /**
   * Validate approval request with Gemini
   */
  private async validateWithGemini(request: ApprovalRequest): Promise<{
    approved: boolean;
    reasoning: string;
  }> {
    const prompt = `Review this RAN parameter change request:

Action: ${request.action}
Risk Level: ${request.risk_level}
Justification: ${request.justification}

Changes:
${request.proposed_changes.map(c =>
  `- ${c.cell_id}: ${c.parameter} from ${c.old_value} to ${c.new_value}`
).join('\n')}

Safety Checks:
${request.safety_checks?.map(s => `- ${s.check_name}: ${s.passed ? 'PASS' : 'FAIL'}`).join('\n') || 'None'}

Should this request be approved or rejected? Provide clear reasoning.`;

    const result = await this.gemini['model'].generateContent(prompt);
    const text = result.response.text();

    const approved = /\b(approve|approved|accept|accepted)\b/i.test(text) &&
                    !/\b(reject|rejected|deny|denied)\b/i.test(text);

    return {
      approved,
      reasoning: text
    };
  }

  /**
   * Get detailed network insights from both AIs
   */
  async getNetworkInsights(
    cells: CellStatus[],
    interferenceMatrix?: InterferenceMatrix
  ): Promise<{
    overview: string;
    keyFindings: string[];
    recommendations: string[];
    risks: string[];
  }> {
    console.log('[AI Orchestrator] Generating network insights...');

    const [geminiAnalysis, anomalies] = await Promise.all([
      this.gemini.analyzeNetworkPerformance(cells, interferenceMatrix),
      this.gemini.detectAnomalies(cells)
    ]);

    const keyFindings = [
      `Active Cells: ${cells.filter(c => c.status === 'active').length}/${cells.length}`,
      `Anomalies Detected: ${anomalies.anomalies.length}`,
      ...(geminiAnalysis.visualInsights || []).slice(0, 3)
    ];

    const recommendations = geminiAnalysis.recommendations.map(
      r => `Optimize ${r.parameter} for ${r.cell_id}: ${r.old_value} → ${r.new_value}`
    ).slice(0, 5);

    return {
      overview: geminiAnalysis.analysis,
      keyFindings,
      recommendations,
      risks: anomalies.anomalies
        .filter(a => a.severity === 'high' || a.severity === 'critical')
        .map(a => `${a.cellId}: ${a.description}`)
    };
  }

  /**
   * Clear conversation history for both AIs
   */
  clearHistory(): void {
    this.claude.clearHistory();
    this.gemini.startNewSession();
    console.log('[AI Orchestrator] Conversation history cleared');
  }
}

// ============================================================================
// Export
// ============================================================================

export default AIOrchestrator;
