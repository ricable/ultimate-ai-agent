/**
 * Claude Agent SDK Integration for TITAN Dashboard
 * Provides AI-powered RAN optimization using Claude's agent capabilities
 *
 * @module ui/integrations/claude-agent-integration
 * @version 7.0.0-alpha.1
 */

import Anthropic from '@anthropic-ai/sdk';
import { z } from 'zod';
import type {
  CellStatus,
  OptimizationEvent,
  ParameterChange,
  ApprovalRequest,
  SafetyCheck
} from '../types.js';

// ============================================================================
// Zod Schemas for Tool Validation
// ============================================================================

const OptimizeParametersSchema = z.object({
  cell_ids: z.array(z.string()).describe('Cell IDs to optimize'),
  target_kpi: z.enum(['rsrp', 'sinr', 'throughput', 'prb_utilization']).describe('Target KPI to improve'),
  constraints: z.object({
    min_power: z.number().min(-130).max(46).optional(),
    max_power: z.number().min(-130).max(46).optional(),
    min_tilt: z.number().min(0).max(15).optional(),
    max_tilt: z.number().min(0).max(15).optional()
  }).optional()
});

const AnalyzeInterferenceSchema = z.object({
  cell_id: z.string().describe('Cell ID to analyze'),
  neighbor_cells: z.array(z.string()).describe('Neighboring cell IDs'),
  threshold_dbm: z.number().describe('Interference threshold in dBm')
});

const PredictKPIImpactSchema = z.object({
  cell_id: z.string(),
  parameter: z.enum(['power_dbm', 'tilt_deg', 'azimuth_deg']),
  current_value: z.number(),
  proposed_value: z.number()
});

// ============================================================================
// Tool Definitions
// ============================================================================

const CLAUDE_TOOLS: Anthropic.Tool[] = [
  {
    name: 'optimize_ran_parameters',
    description: 'Optimize RAN parameters (power, tilt, azimuth) for specified cells to improve target KPI while maintaining 3GPP compliance',
    input_schema: {
      type: 'object',
      properties: {
        cell_ids: {
          type: 'array',
          items: { type: 'string' },
          description: 'Cell IDs to optimize'
        },
        target_kpi: {
          type: 'string',
          enum: ['rsrp', 'sinr', 'throughput', 'prb_utilization'],
          description: 'Target KPI to improve'
        },
        constraints: {
          type: 'object',
          properties: {
            min_power: { type: 'number', minimum: -130, maximum: 46 },
            max_power: { type: 'number', minimum: -130, maximum: 46 },
            min_tilt: { type: 'number', minimum: 0, maximum: 15 },
            max_tilt: { type: 'number', minimum: 0, maximum: 15 }
          }
        }
      },
      required: ['cell_ids', 'target_kpi']
    }
  },
  {
    name: 'analyze_interference_pattern',
    description: 'Analyze interference patterns between cells and identify problematic interactions',
    input_schema: {
      type: 'object',
      properties: {
        cell_id: { type: 'string', description: 'Cell ID to analyze' },
        neighbor_cells: {
          type: 'array',
          items: { type: 'string' },
          description: 'Neighboring cell IDs'
        },
        threshold_dbm: { type: 'number', description: 'Interference threshold in dBm' }
      },
      required: ['cell_id', 'neighbor_cells', 'threshold_dbm']
    }
  },
  {
    name: 'predict_kpi_impact',
    description: 'Predict the impact of a parameter change on cell KPIs using historical data',
    input_schema: {
      type: 'object',
      properties: {
        cell_id: { type: 'string' },
        parameter: {
          type: 'string',
          enum: ['power_dbm', 'tilt_deg', 'azimuth_deg']
        },
        current_value: { type: 'number' },
        proposed_value: { type: 'number' }
      },
      required: ['cell_id', 'parameter', 'current_value', 'proposed_value']
    }
  }
];

// ============================================================================
// Claude Agent Integration Class
// ============================================================================

export interface ClaudeAgentConfig {
  apiKey?: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

export class ClaudeAgentIntegration {
  private client: Anthropic;
  private config: Required<Omit<ClaudeAgentConfig, 'apiKey'>> & { apiKey?: string };
  private conversationHistory: Anthropic.MessageParam[] = [];

  constructor(config: ClaudeAgentConfig = {}) {
    // Support multiple authentication modes:
    // 1. API key from config or environment
    // 2. OAuth token from Claude Code subscription (CLAUDE_CODE_OAUTH_TOKEN)
    // 3. Automatic auth when running inside Claude Code CLI

    const apiKey = config.apiKey || process.env.ANTHROPIC_API_KEY;
    const oauthToken = process.env.CLAUDE_CODE_OAUTH_TOKEN;
    const isClaudeCodeCLI = process.env.CLAUDECODE === '1';

    this.config = {
      apiKey,
      model: config.model || process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101',
      maxTokens: config.maxTokens || 8192,
      temperature: config.temperature || 0.7
    };

    // Initialize client based on available auth method
    if (apiKey) {
      // Direct API key (blocked in subscription-only mode)
      this.client = new Anthropic({ apiKey });
      console.log(`[Claude Agent] Using API key authentication`);
    } else if (oauthToken) {
      // OAuth token from Claude Code subscription
      // The token acts as an API key for the SDK
      this.client = new Anthropic({ apiKey: oauthToken });
      console.log(`[Claude Agent] Using Claude Code OAuth token (subscription)`);
    } else if (isClaudeCodeCLI) {
      // Running inside Claude Code CLI - auth is automatic
      this.client = new Anthropic();
      console.log(`[Claude Agent] Using Claude Code CLI automatic authentication`);
    } else {
      // No auth available - provide helpful error
      throw new Error(
        'Claude authentication required.\n\n' +
        'Options:\n' +
        '1. Run inside Claude Code CLI (automatic auth)\n' +
        '2. Run: claude setup-token and set CLAUDE_CODE_OAUTH_TOKEN in config/.env\n' +
        '3. Set ANTHROPIC_API_KEY (blocked in subscription-only mode)\n'
      );
    }

    console.log(`[Claude Agent] Initialized with model: ${this.config.model}`);
  }

  /**
   * Request RAN optimization recommendations from Claude
   */
  async requestOptimization(
    cells: CellStatus[],
    objective: string,
    constraints?: Record<string, any>
  ): Promise<{
    recommendations: ParameterChange[];
    reasoning: string;
    confidence: number;
  }> {
    const systemPrompt = `You are an expert RAN optimization AI assistant for 5G/6G networks.
You analyze cell performance metrics and recommend parameter adjustments to improve network KPIs.
You must respect 3GPP constraints: power_dbm [-130, 46], tilt_deg [0, 15], azimuth_deg [0, 360].
Use the provided tools to optimize parameters, analyze interference, and predict KPI impacts.`;

    const userMessage = `Analyze the following cells and provide optimization recommendations:

Objective: ${objective}

Cells:
${cells.map(c => `
- ${c.cell_id}:
  Power: ${c.power_dbm} dBm, Tilt: ${c.tilt_deg}°, Azimuth: ${c.azimuth_deg}°
  KPIs: RSRP ${c.kpi.rsrp_avg} dBm, SINR ${c.kpi.sinr_avg} dB, Throughput ${c.kpi.throughput_mbps} Mbps, PRB ${c.kpi.prb_utilization}%
  Status: ${c.status}
`).join('\n')}

${constraints ? `Constraints: ${JSON.stringify(constraints, null, 2)}` : ''}

Please use the available tools to:
1. Analyze interference patterns
2. Predict KPI impacts
3. Recommend optimal parameter changes`;

    const messages: Anthropic.MessageParam[] = [
      ...this.conversationHistory,
      {
        role: 'user',
        content: userMessage
      }
    ];

    const response = await this.client.messages.create({
      model: this.config.model,
      max_tokens: this.config.maxTokens,
      temperature: this.config.temperature,
      system: systemPrompt,
      messages,
      tools: CLAUDE_TOOLS
    });

    // Process tool use if present
    const recommendations: ParameterChange[] = [];
    let reasoning = '';
    let confidence = 0.8;

    for (const content of response.content) {
      if (content.type === 'tool_use') {
        const result = await this.handleToolUse(content, cells);
        if (result.recommendations) {
          recommendations.push(...result.recommendations);
        }
      } else if (content.type === 'text') {
        reasoning += content.text;
      }
    }

    // Update conversation history
    this.conversationHistory.push(
      { role: 'user', content: userMessage },
      { role: 'assistant', content: response.content }
    );

    // Extract confidence from reasoning if mentioned
    const confidenceMatch = reasoning.match(/confidence[:\s]+(\d+(?:\.\d+)?)/i);
    if (confidenceMatch) {
      confidence = parseFloat(confidenceMatch[1]);
      if (confidence > 1) confidence /= 100; // Handle percentage
    }

    return {
      recommendations,
      reasoning,
      confidence
    };
  }

  /**
   * Analyze cell performance and suggest improvements
   */
  async analyzeCellPerformance(cell: CellStatus): Promise<{
    issues: string[];
    recommendations: string[];
    priority: 'low' | 'medium' | 'high' | 'critical';
  }> {
    const prompt = `Analyze this cell's performance and identify issues:

Cell: ${cell.cell_id}
Status: ${cell.status}
Power: ${cell.power_dbm} dBm
Tilt: ${cell.tilt_deg}°
KPIs:
- RSRP: ${cell.kpi.rsrp_avg} dBm (target: > -90 dBm)
- SINR: ${cell.kpi.sinr_avg} dB (target: > 10 dB)
- Throughput: ${cell.kpi.throughput_mbps} Mbps
- PRB Utilization: ${cell.kpi.prb_utilization}%
- Handover Success: ${cell.kpi.ho_success_rate}%
- Drop Rate: ${cell.kpi.drop_rate}%

Provide:
1. List of performance issues
2. Recommendations to fix them
3. Priority level (low/medium/high/critical)`;

    const response = await this.client.messages.create({
      model: this.config.model,
      max_tokens: 2048,
      messages: [{ role: 'user', content: prompt }]
    });

    const text = response.content
      .filter((c): c is Anthropic.TextBlock => c.type === 'text')
      .map(c => c.text)
      .join('\n');

    // Parse response
    const issues: string[] = [];
    const recommendations: string[] = [];
    let priority: 'low' | 'medium' | 'high' | 'critical' = 'medium';

    // Extract issues
    const issuesMatch = text.match(/issues?[:\n]+((?:[-•*]\s*.+\n?)+)/i);
    if (issuesMatch) {
      issues.push(...issuesMatch[1].split('\n').filter(l => l.trim()).map(l => l.replace(/^[-•*]\s*/, '')));
    }

    // Extract recommendations
    const recsMatch = text.match(/recommendations?[:\n]+((?:[-•*]\s*.+\n?)+)/i);
    if (recsMatch) {
      recommendations.push(...recsMatch[1].split('\n').filter(l => l.trim()).map(l => l.replace(/^[-•*]\s*/, '')));
    }

    // Extract priority
    const priorityMatch = text.match(/priority[:\s]+(low|medium|high|critical)/i);
    if (priorityMatch) {
      priority = priorityMatch[1].toLowerCase() as any;
    }

    return { issues, recommendations, priority };
  }

  /**
   * Validate an approval request using Claude's judgment
   */
  async validateApprovalRequest(request: ApprovalRequest): Promise<{
    approved: boolean;
    reasoning: string;
    safetyChecks: SafetyCheck[];
  }> {
    const prompt = `Review this RAN parameter change request for safety and compliance:

Action: ${request.action}
Risk Level: ${request.risk_level}
Target Cells: ${request.target?.join(', ')}

Changes:
${request.proposed_changes.map(c =>
  `- ${c.cell_id}: ${c.parameter} from ${c.old_value} to ${c.new_value} (bounds: [${c.bounds.join(', ')}])`
).join('\n')}

Justification: ${request.justification}

Safety Checks:
${request.safety_checks?.map(s => `- ${s.check_name}: ${s.passed ? 'PASS' : 'FAIL'} (${s.message})`).join('\n') || 'None'}

Assess:
1. Are all changes within 3GPP bounds?
2. Is the risk level appropriate?
3. Are safety checks adequate?
4. Should this be approved or rejected?

Provide your decision with reasoning.`;

    const response = await this.client.messages.create({
      model: this.config.model,
      max_tokens: 2048,
      messages: [{ role: 'user', content: prompt }]
    });

    const text = response.content
      .filter((c): c is Anthropic.TextBlock => c.type === 'text')
      .map(c => c.text)
      .join('\n');

    // Parse decision
    const approved = /\b(approve|approved|accept|accepted)\b/i.test(text) &&
                    !/\b(reject|rejected|deny|denied)\b/i.test(text);

    return {
      approved,
      reasoning: text,
      safetyChecks: request.safety_checks || []
    };
  }

  /**
   * Handle tool use from Claude's response
   */
  private async handleToolUse(
    toolUse: Anthropic.ToolUseBlock,
    cells: CellStatus[]
  ): Promise<{ recommendations?: ParameterChange[] }> {
    switch (toolUse.name) {
      case 'optimize_ran_parameters': {
        const input = OptimizeParametersSchema.parse(toolUse.input);
        // Simulate optimization (in production, this would call actual optimization logic)
        const recommendations: ParameterChange[] = input.cell_ids.map(cellId => {
          const cell = cells.find(c => c.cell_id === cellId);
          if (!cell) return null;

          return {
            cell_id: cellId,
            parameter: 'power_dbm',
            old_value: cell.power_dbm,
            new_value: Math.min(
              input.constraints?.max_power ?? 46,
              Math.max(input.constraints?.min_power ?? -130, cell.power_dbm + 2)
            ),
            bounds: [input.constraints?.min_power ?? -130, input.constraints?.max_power ?? 46]
          };
        }).filter((r): r is ParameterChange => r !== null);

        return { recommendations };
      }

      case 'analyze_interference_pattern': {
        const input = AnalyzeInterferenceSchema.parse(toolUse.input);
        console.log(`[Claude Agent] Analyzing interference for ${input.cell_id}`);
        return {};
      }

      case 'predict_kpi_impact': {
        const input = PredictKPIImpactSchema.parse(toolUse.input);
        console.log(`[Claude Agent] Predicting KPI impact for ${input.cell_id}`);
        return {};
      }

      default:
        return {};
    }
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }

  /**
   * Get conversation history
   */
  getHistory(): Anthropic.MessageParam[] {
    return [...this.conversationHistory];
  }
}

// ============================================================================
// Export
// ============================================================================

export default ClaudeAgentIntegration;
