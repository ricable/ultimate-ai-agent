// RAN Technical Documentation Ingestion and Analysis Agent
// Processes Ericsson technical documentation and extracts relevant information

import { BaseAgent, AgentResult } from './base-agent.js';
import { z } from 'zod';
import { logger } from '../core/logger.js';

const RanDocsInputSchema = z.object({
  documentPath: z.string().optional(),
  documentContent: z.string().optional(),
  query: z.string().optional(),
  extractionType: z.enum(['parameters', 'kpis', 'alarms', 'procedures', 'all']).default('all'),
});

type RanDocsInput = z.infer<typeof RanDocsInputSchema>;

export interface RanDocsOutput {
  parameters?: Array<{ name: string; description: string; defaultValue?: string; range?: string }>;
  kpis?: Array<{ name: string; description: string; unit: string; category: string }>;
  alarms?: Array<{ alarmId: string; description: string; severity: string; causes: string[] }>;
  procedures?: Array<{ name: string; steps: string[]; conditions: string[] }>;
  summary?: string;
}

export class RanDocsAgent extends BaseAgent<RanDocsInput, RanDocsOutput> {
  constructor() {
    super({
      name: 'RAN-Docs-Agent',
      role: 'Technical Documentation Specialist',
      systemPrompt: `You are an expert in Ericsson RAN (Radio Access Network) technical documentation.
Your role is to analyze and extract structured information from RAN documentation including:
- Network parameters and their valid ranges
- KPIs (Key Performance Indicators) and counters
- Alarm definitions and troubleshooting procedures
- Configuration management procedures
- Optimization guidelines

You provide precise, structured information that can be used for automated network management.
Always cite the source of information and indicate confidence levels.`,
      temperature: 0.3, // Lower temperature for factual extraction
    });
  }

  async execute(input: RanDocsInput): Promise<AgentResult<RanDocsOutput>> {
    try {
      const validatedInput = await this.validateInput(input, RanDocsInputSchema);
      this.log('info', 'Processing RAN documentation', { extractionType: validatedInput.extractionType });

      let prompt = '';

      if (validatedInput.documentContent) {
        prompt = this.buildExtractionPrompt(validatedInput.documentContent, validatedInput.extractionType);
      } else if (validatedInput.query) {
        prompt = `Query: ${validatedInput.query}\n\nProvide detailed information based on your knowledge of Ericsson RAN systems.`;
      } else {
        throw new Error('Either documentContent or query must be provided');
      }

      const response = await this.sendMessage(prompt);
      const output = this.parseResponse(response, validatedInput.extractionType);

      return {
        success: true,
        data: output,
        reasoning: 'Successfully extracted information from RAN documentation',
      };
    } catch (error) {
      this.log('error', 'Failed to process RAN documentation', { error });
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private buildExtractionPrompt(content: string, extractionType: string): string {
    const basePrompt = `Analyze the following RAN technical documentation and extract structured information.\n\nDocument:\n${content}\n\n`;

    switch (extractionType) {
      case 'parameters':
        return basePrompt + `Extract all network parameters including:
- Parameter name
- Description
- Default value (if mentioned)
- Valid range or allowed values
- Impact on network performance

Format as JSON array.`;

      case 'kpis':
        return basePrompt + `Extract all KPIs and counters including:
- KPI name
- Description
- Unit of measurement
- Category (throughput, latency, capacity, quality, etc.)
- Typical target values

Format as JSON array.`;

      case 'alarms':
        return basePrompt + `Extract all alarm definitions including:
- Alarm ID
- Description
- Severity level
- Possible causes
- Recommended actions

Format as JSON array.`;

      case 'procedures':
        return basePrompt + `Extract all procedures including:
- Procedure name
- Detailed steps
- Prerequisites/conditions
- Expected outcomes
- Rollback procedures (if any)

Format as JSON array.`;

      default:
        return basePrompt + `Extract all relevant information in a structured format including parameters, KPIs, alarms, and procedures.`;
    }
  }

  private parseResponse(response: string, extractionType: string): RanDocsOutput {
    const output: RanDocsOutput = {};

    try {
      // Attempt to parse JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        if (extractionType === 'all') {
          return parsed;
        } else {
          output[extractionType as keyof RanDocsOutput] = Array.isArray(parsed) ? parsed : [parsed];
        }
      } else {
        // Fallback: use the response as summary
        output.summary = response;
      }
    } catch (error) {
      logger.warn('Failed to parse structured response, using as summary', { error });
      output.summary = response;
    }

    return output;
  }

  /**
   * Query specific RAN information
   */
  async queryRanInfo(query: string): Promise<string> {
    const result = await this.execute({ query });
    return result.data?.summary || JSON.stringify(result.data);
  }

  /**
   * Extract parameter definitions
   */
  async extractParameters(documentContent: string): Promise<Array<any>> {
    const result = await this.execute({
      documentContent,
      extractionType: 'parameters',
    });
    return result.data?.parameters || [];
  }

  /**
   * Extract KPI definitions
   */
  async extractKpis(documentContent: string): Promise<Array<any>> {
    const result = await this.execute({
      documentContent,
      extractionType: 'kpis',
    });
    return result.data?.kpis || [];
  }
}

export default RanDocsAgent;
