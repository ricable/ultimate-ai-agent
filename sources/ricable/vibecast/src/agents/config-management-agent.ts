// Configuration Management Agent
// Generates automated parameter change proposals for RAN optimization

import { BaseAgent, AgentResult } from './base-agent.js';
import { ParameterChangeProposal, KpiMeasurement, RanNode } from '../types/ran-models.js';
import { z } from 'zod';

const ConfigManagementInputSchema = z.object({
  node: z.any(),
  kpiMeasurements: z.array(z.any()),
  performanceIssues: z.array(z.string()).optional(),
  optimizationGoals: z.array(z.string()).optional(),
  constraints: z.object({
    maxRiskLevel: z.enum(['Low', 'Medium', 'High', 'Critical']).default('Medium'),
    maxParameterChanges: z.number().default(5),
    allowCriticalParameters: z.boolean().default(false),
  }).optional(),
});

type ConfigManagementInput = z.infer<typeof ConfigManagementInputSchema>;

export interface ConfigManagementOutput {
  proposals: ParameterChangeProposal[];
  summary: string;
  overallImpact: {
    expectedKpiImprovements: Record<string, number>;
    riskAssessment: string;
    rollbackPlan: string[];
  };
  prioritization: Array<{
    proposalId: string;
    priority: number;
    reason: string;
  }>;
}

export class ConfigManagementAgent extends BaseAgent<ConfigManagementInput, ConfigManagementOutput> {
  constructor() {
    super({
      name: 'Config-Management-Agent',
      role: 'Automated Configuration Management Specialist',
      systemPrompt: `You are an expert in RAN automated configuration management and optimization.
Your role is to:
- Analyze network performance data and KPIs
- Identify suboptimal parameter configurations
- Generate parameter change proposals with clear rationale
- Assess risks and expected impacts of changes
- Prioritize changes based on expected benefits and risks
- Provide rollback procedures for all changes

You understand:
- RAN parameter interdependencies
- Impact of parameters on various KPIs
- Safe parameter ranges for different scenarios
- Risk assessment for configuration changes
- Optimization strategies for capacity, coverage, and quality

Always prioritize network stability and service continuity. Never suggest changes that could cause service disruption without explicit acknowledgment of risks.`,
      temperature: 0.4,
    });
  }

  async execute(input: ConfigManagementInput): Promise<AgentResult<ConfigManagementOutput>> {
    try {
      const validatedInput = await this.validateInput(input, ConfigManagementInputSchema);
      this.log('info', 'Generating configuration proposals', {
        nodeId: validatedInput.node?.nodeId,
        kpiCount: validatedInput.kpiMeasurements?.length || 0,
      });

      const prompt = this.buildConfigPrompt(validatedInput);
      const response = await this.sendMessage(prompt);
      const output = this.parseConfigResponse(response);

      return {
        success: true,
        data: output,
        reasoning: 'Generated configuration change proposals',
      };
    } catch (error) {
      this.log('error', 'Failed to generate config proposals', { error });
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private buildConfigPrompt(input: ConfigManagementInput): string {
    let prompt = `Generate automated configuration management proposals for RAN optimization.\n\n`;

    prompt += `Node Information:\n`;
    prompt += `  Node ID: ${input.node?.nodeId || 'N/A'}\n`;
    prompt += `  Node Type: ${input.node?.nodeType || 'N/A'}\n`;
    prompt += `  Number of Cells: ${input.node?.cells?.length || 0}\n\n`;

    if (input.node?.parameters) {
      prompt += `Current Parameters:\n${JSON.stringify(input.node.parameters, null, 2)}\n\n`;
    }

    if (input.kpiMeasurements && input.kpiMeasurements.length > 0) {
      const kpiSummary = this.summarizeKpis(input.kpiMeasurements);
      prompt += `KPI Performance:\n${kpiSummary}\n\n`;
    }

    if (input.performanceIssues && input.performanceIssues.length > 0) {
      prompt += `Identified Performance Issues:\n`;
      input.performanceIssues.forEach((issue, idx) => {
        prompt += `  ${idx + 1}. ${issue}\n`;
      });
      prompt += `\n`;
    }

    if (input.optimizationGoals && input.optimizationGoals.length > 0) {
      prompt += `Optimization Goals:\n`;
      input.optimizationGoals.forEach((goal, idx) => {
        prompt += `  ${idx + 1}. ${goal}\n`;
      });
      prompt += `\n`;
    }

    const constraints = input.constraints || {};
    prompt += `Constraints:\n`;
    prompt += `  Max Risk Level: ${constraints.maxRiskLevel || 'Medium'}\n`;
    prompt += `  Max Parameter Changes: ${constraints.maxParameterChanges || 5}\n`;
    prompt += `  Allow Critical Parameters: ${constraints.allowCriticalParameters ? 'Yes' : 'No'}\n\n`;

    prompt += `Generate parameter change proposals in the following JSON format:
{
  "proposals": [
    {
      "proposalId": "unique-id",
      "timestamp": <unix-timestamp>,
      "nodeId": "node-id",
      "cellId": "cell-id (optional)",
      "parameterName": "parameter-name",
      "currentValue": <current-value>,
      "proposedValue": <proposed-value>,
      "confidenceScore": <0-1>,
      "rationale": "clear explanation",
      "expectedImpact": {
        "kpi-name": <percentage-improvement>
      },
      "riskAssessment": "Low|Medium|High|Critical"
    }
  ],
  "summary": "overall summary",
  "overallImpact": {
    "expectedKpiImprovements": {
      "kpi-name": <percentage>
    },
    "riskAssessment": "assessment",
    "rollbackPlan": ["step 1", "step 2"]
  },
  "prioritization": [
    {
      "proposalId": "id",
      "priority": <1-10>,
      "reason": "reason"
    }
  ]
}

Ensure all proposals include:
1. Clear rationale based on data
2. Expected impact on relevant KPIs
3. Risk assessment
4. Confidence score
5. Rollback procedures`;

    return prompt;
  }

  private summarizeKpis(measurements: any[]): string {
    const kpiGroups: Record<string, any[]> = {};

    for (const m of measurements) {
      const kpiName = m.kpiName || 'unknown';
      if (!kpiGroups[kpiName]) {
        kpiGroups[kpiName] = [];
      }
      kpiGroups[kpiName].push(m.value || 0);
    }

    let summary = '';
    for (const [kpiName, values] of Object.entries(kpiGroups)) {
      const avg = values.reduce((a, b) => a + b, 0) / values.length;
      const min = Math.min(...values);
      const max = Math.max(...values);
      summary += `  ${kpiName}: avg=${avg.toFixed(2)}, min=${min.toFixed(2)}, max=${max.toFixed(2)}\n`;
    }

    return summary;
  }

  private parseConfigResponse(response: string): ConfigManagementOutput {
    const defaultOutput: ConfigManagementOutput = {
      proposals: [],
      summary: response,
      overallImpact: {
        expectedKpiImprovements: {},
        riskAssessment: 'Unknown',
        rollbackPlan: [],
      },
      prioritization: [],
    };

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        if (parsed.proposals && Array.isArray(parsed.proposals)) {
          return {
            proposals: parsed.proposals,
            summary: parsed.summary || 'Configuration proposals generated',
            overallImpact: parsed.overallImpact || defaultOutput.overallImpact,
            prioritization: parsed.prioritization || [],
          };
        }
      }
    } catch (error) {
      this.log('warn', 'Failed to parse config response', { error });
    }

    return defaultOutput;
  }

  /**
   * Generate optimization proposals
   */
  async generateOptimizationProposals(
    node: RanNode,
    kpiMeasurements: KpiMeasurement[],
    optimizationGoals: string[]
  ): Promise<ParameterChangeProposal[]> {
    const result = await this.execute({
      node,
      kpiMeasurements,
      optimizationGoals,
    });
    return result.data?.proposals || [];
  }

  /**
   * Assess risk of parameter change
   */
  async assessParameterChangeRisk(
    node: RanNode,
    parameterName: string,
    proposedValue: any
  ): Promise<{ riskLevel: string; impacts: string[]; recommendations: string[] }> {
    const prompt = `Assess the risk of changing parameter "${parameterName}" to "${proposedValue}" on node ${node.nodeId}.

Node Type: ${node.nodeType}
Current Value: ${node.parameters[parameterName]}

Provide risk assessment in JSON format:
{
  "riskLevel": "Low|Medium|High|Critical",
  "impacts": ["impact 1", "impact 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}`;

    const response = await this.sendMessage(prompt);

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      this.log('warn', 'Failed to parse risk assessment', { error });
    }

    return {
      riskLevel: 'Unknown',
      impacts: [],
      recommendations: [],
    };
  }
}

export default ConfigManagementAgent;
