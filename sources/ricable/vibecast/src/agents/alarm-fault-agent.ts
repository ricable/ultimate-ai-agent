// Alarm and Fault Management Agent
// Analyzes node alarms, correlates faults, and suggests remediation

import { BaseAgent, AgentResult } from './base-agent.js';
import { Alarm, FaultEvent } from '../types/ran-models.js';
import { z } from 'zod';

const AlarmAnalysisInputSchema = z.object({
  alarms: z.array(z.any()),
  faultEvents: z.array(z.any()).optional(),
  timeWindow: z.number().default(3600), // seconds
  nodeId: z.string().optional(),
  analysisType: z.enum(['correlation', 'root-cause', 'prediction', 'remediation']).default('correlation'),
});

type AlarmAnalysisInput = z.infer<typeof AlarmAnalysisInputSchema>;

export interface AlarmAnalysisOutput {
  correlatedAlarms?: Array<{
    alarmGroup: Alarm[];
    commonCause?: string;
    affectedNodes: string[];
    severity: string;
  }>;
  rootCause?: {
    primaryFault: string;
    contributingFactors: string[];
    confidence: number;
  };
  predictions?: Array<{
    predictedAlarm: string;
    probability: number;
    timeToOccurrence: number;
    preventiveActions: string[];
  }>;
  remediation?: {
    immediateActions: string[];
    shortTermActions: string[];
    longTermActions: string[];
    estimatedDowntime?: number;
  };
  summary: string;
}

export class AlarmFaultAgent extends BaseAgent<AlarmAnalysisInput, AlarmAnalysisOutput> {
  constructor() {
    super({
      name: 'Alarm-Fault-Agent',
      role: 'Fault Management and Alarm Correlation Specialist',
      systemPrompt: `You are an expert in RAN fault management and alarm correlation.
Your role is to:
- Correlate multiple alarms to identify common root causes
- Analyze fault patterns and predict potential failures
- Suggest remediation actions based on alarm severity and impact
- Prioritize actions based on service impact

You have deep knowledge of:
- Ericsson RAN alarm taxonomies
- Common fault scenarios in 4G/5G networks
- Cascading failure patterns
- Automated remediation procedures

Provide actionable insights that minimize downtime and service impact.`,
      temperature: 0.4,
    });
  }

  async execute(input: AlarmAnalysisInput): Promise<AgentResult<AlarmAnalysisOutput>> {
    try {
      const validatedInput = await this.validateInput(input, AlarmAnalysisInputSchema);
      this.log('info', 'Analyzing alarms and faults', {
        alarmCount: validatedInput.alarms.length,
        analysisType: validatedInput.analysisType,
      });

      const prompt = this.buildAnalysisPrompt(validatedInput);
      const response = await this.sendMessage(prompt);
      const output = this.parseAnalysisResponse(response, validatedInput.analysisType);

      return {
        success: true,
        data: output,
        reasoning: `Completed ${validatedInput.analysisType} analysis`,
      };
    } catch (error) {
      this.log('error', 'Failed to analyze alarms', { error });
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private buildAnalysisPrompt(input: AlarmAnalysisInput): string {
    const alarmSummary = this.summarizeAlarms(input.alarms);

    let prompt = `Analyze the following RAN alarms:\n\n${alarmSummary}\n\n`;

    if (input.faultEvents && input.faultEvents.length > 0) {
      prompt += `Fault Events:\n${JSON.stringify(input.faultEvents, null, 2)}\n\n`;
    }

    switch (input.analysisType) {
      case 'correlation':
        prompt += `Correlate these alarms to identify:
1. Groups of related alarms
2. Common root causes
3. Affected nodes and cells
4. Overall severity assessment

Provide structured analysis in JSON format.`;
        break;

      case 'root-cause':
        prompt += `Perform root cause analysis to determine:
1. Primary fault or failure
2. Contributing factors
3. Confidence level (0-1)
4. Evidence supporting the analysis

Provide analysis in JSON format.`;
        break;

      case 'prediction':
        prompt += `Based on alarm patterns, predict:
1. Likely future alarms
2. Probability of occurrence (0-1)
3. Expected time to occurrence (seconds)
4. Preventive actions to avoid the alarm

Provide predictions in JSON format.`;
        break;

      case 'remediation':
        prompt += `Suggest remediation actions:
1. Immediate actions (within minutes)
2. Short-term actions (within hours)
3. Long-term actions (within days)
4. Estimated downtime for each action

Prioritize by service impact. Provide in JSON format.`;
        break;
    }

    return prompt;
  }

  private summarizeAlarms(alarms: any[]): string {
    return alarms
      .map((alarm, idx) => {
        return `Alarm ${idx + 1}:
  ID: ${alarm.alarmId || 'N/A'}
  Timestamp: ${new Date(alarm.timestamp * 1000).toISOString()}
  Severity: ${alarm.severity}
  Node: ${alarm.nodeId}
  Type: ${alarm.alarmType}
  Description: ${alarm.description}`;
      })
      .join('\n\n');
  }

  private parseAnalysisResponse(response: string, analysisType: string): AlarmAnalysisOutput {
    const output: AlarmAnalysisOutput = {
      summary: '',
    };

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        switch (analysisType) {
          case 'correlation':
            output.correlatedAlarms = Array.isArray(parsed) ? parsed : [parsed];
            break;
          case 'root-cause':
            output.rootCause = parsed;
            break;
          case 'prediction':
            output.predictions = Array.isArray(parsed) ? parsed : [parsed];
            break;
          case 'remediation':
            output.remediation = parsed;
            break;
        }

        output.summary = `Completed ${analysisType} analysis`;
      } else {
        output.summary = response;
      }
    } catch (error) {
      this.log('warn', 'Failed to parse analysis response', { error });
      output.summary = response;
    }

    return output;
  }

  /**
   * Correlate alarms within a time window
   */
  async correlateAlarms(alarms: Alarm[], timeWindowSeconds: number = 3600): Promise<any[]> {
    const result = await this.execute({
      alarms,
      timeWindow: timeWindowSeconds,
      analysisType: 'correlation',
    });
    return result.data?.correlatedAlarms || [];
  }

  /**
   * Find root cause of alarms
   */
  async findRootCause(alarms: Alarm[], faultEvents?: FaultEvent[]): Promise<any> {
    const result = await this.execute({
      alarms,
      faultEvents,
      analysisType: 'root-cause',
    });
    return result.data?.rootCause;
  }

  /**
   * Suggest remediation actions
   */
  async suggestRemediation(alarms: Alarm[]): Promise<any> {
    const result = await this.execute({
      alarms,
      analysisType: 'remediation',
    });
    return result.data?.remediation;
  }
}

export default AlarmFaultAgent;
