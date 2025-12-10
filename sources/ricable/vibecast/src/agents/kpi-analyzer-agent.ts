// KPI and Counter Multivariate Analysis Agent
// Analyzes time series KPIs and counters for optimization insights

import { BaseAgent, AgentResult } from './base-agent.js';
import { KpiMeasurement, MultiVariatePoint, Granularity } from '../types/ran-models.js';
import { z } from 'zod';

const KpiAnalysisInputSchema = z.object({
  measurements: z.array(z.any()),
  granularity: z.enum(['Hourly', 'Daily', 'Weekly']).default('Hourly'),
  analysisType: z.enum(['trend', 'anomaly', 'correlation', 'forecast', 'optimization']).default('trend'),
  nodeId: z.string().optional(),
  cellId: z.string().optional(),
  kpiNames: z.array(z.string()).optional(),
  forecastHorizon: z.number().default(168), // hours
});

type KpiAnalysisInput = z.infer<typeof KpiAnalysisInputSchema>;

export interface KpiAnalysisOutput {
  trends?: Array<{
    kpiName: string;
    trend: 'increasing' | 'decreasing' | 'stable' | 'volatile';
    changeRate: number;
    significance: number;
  }>;
  anomalies?: Array<{
    timestamp: number;
    kpiName: string;
    actualValue: number;
    expectedValue: number;
    deviationScore: number;
    severity: string;
  }>;
  correlations?: Array<{
    kpi1: string;
    kpi2: string;
    correlation: number;
    relationship: string;
  }>;
  forecasts?: Array<{
    kpiName: string;
    predictions: Array<{ timestamp: number; value: number; confidence: number }>;
    model: string;
  }>;
  optimizations?: Array<{
    kpiName: string;
    currentPerformance: number;
    targetPerformance: number;
    recommendations: string[];
    expectedImprovement: number;
  }>;
  summary: string;
  insights: string[];
}

export class KpiAnalyzerAgent extends BaseAgent<KpiAnalysisInput, KpiAnalysisOutput> {
  constructor() {
    super({
      name: 'KPI-Analyzer-Agent',
      role: 'KPI and Performance Analysis Specialist',
      systemPrompt: `You are an expert in RAN KPI analysis and network optimization.
Your role is to:
- Analyze multivariate time series data from RAN KPIs and counters
- Identify trends, anomalies, and correlation patterns
- Forecast future KPI values based on historical patterns
- Provide optimization recommendations to improve network performance

You understand:
- RAN KPI taxonomies (throughput, latency, capacity, quality, efficiency)
- Relationships between different KPIs
- Normal ranges and targets for various KPIs
- Impact of network parameters on KPI values
- Seasonality and temporal patterns in network traffic

Provide data-driven insights with quantified confidence levels.`,
      temperature: 0.5,
    });
  }

  async execute(input: KpiAnalysisInput): Promise<AgentResult<KpiAnalysisOutput>> {
    try {
      const validatedInput = await this.validateInput(input, KpiAnalysisInputSchema);
      this.log('info', 'Analyzing KPIs', {
        measurementCount: validatedInput.measurements.length,
        analysisType: validatedInput.analysisType,
        granularity: validatedInput.granularity,
      });

      const prompt = this.buildKpiAnalysisPrompt(validatedInput);
      const response = await this.sendMessage(prompt);
      const output = this.parseKpiResponse(response, validatedInput.analysisType);

      return {
        success: true,
        data: output,
        reasoning: `Completed ${validatedInput.analysisType} analysis for KPIs`,
      };
    } catch (error) {
      this.log('error', 'Failed to analyze KPIs', { error });
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private buildKpiAnalysisPrompt(input: KpiAnalysisInput): string {
    const stats = this.calculateBasicStats(input.measurements);

    let prompt = `Analyze the following RAN KPI measurements:\n\n`;
    prompt += `Granularity: ${input.granularity}\n`;
    prompt += `Number of measurements: ${input.measurements.length}\n`;
    prompt += `Time range: ${new Date(stats.minTimestamp * 1000).toISOString()} to ${new Date(stats.maxTimestamp * 1000).toISOString()}\n\n`;

    prompt += `KPI Statistics:\n${JSON.stringify(stats.kpiStats, null, 2)}\n\n`;

    switch (input.analysisType) {
      case 'trend':
        prompt += `Analyze trends for each KPI:
1. Identify if the KPI is increasing, decreasing, stable, or volatile
2. Calculate the rate of change
3. Assess statistical significance
4. Identify any concerning trends

Provide in JSON array format.`;
        break;

      case 'anomaly':
        prompt += `Detect anomalies in the KPI data:
1. Identify unusual values or patterns
2. Calculate deviation scores
3. Assess severity (critical, major, minor)
4. Suggest possible causes

Provide in JSON array format.`;
        break;

      case 'correlation':
        prompt += `Analyze correlations between KPIs:
1. Identify strongly correlated KPIs (positive or negative)
2. Describe the relationship
3. Assess if correlation implies causation
4. Suggest optimization opportunities based on correlations

Provide in JSON array format.`;
        break;

      case 'forecast':
        prompt += `Forecast KPI values for the next ${input.forecastHorizon} hours:
1. Use appropriate forecasting method (trend, seasonal, etc.)
2. Provide predicted values with confidence intervals
3. Identify the forecasting model used
4. Note any assumptions or limitations

Provide in JSON format.`;
        break;

      case 'optimization':
        prompt += `Provide optimization recommendations:
1. Identify underperforming KPIs
2. Set realistic target performance levels
3. Suggest specific actions to improve performance
4. Estimate expected improvement

Provide in JSON array format.`;
        break;
    }

    return prompt;
  }

  private calculateBasicStats(measurements: any[]): any {
    const kpiStats: Record<string, any> = {};
    let minTimestamp = Infinity;
    let maxTimestamp = -Infinity;

    for (const m of measurements) {
      const kpiName = m.kpiName || 'unknown';
      const value = m.value || 0;
      const timestamp = m.timestamp || 0;

      minTimestamp = Math.min(minTimestamp, timestamp);
      maxTimestamp = Math.max(maxTimestamp, timestamp);

      if (!kpiStats[kpiName]) {
        kpiStats[kpiName] = {
          count: 0,
          sum: 0,
          min: Infinity,
          max: -Infinity,
          values: [],
        };
      }

      kpiStats[kpiName].count++;
      kpiStats[kpiName].sum += value;
      kpiStats[kpiName].min = Math.min(kpiStats[kpiName].min, value);
      kpiStats[kpiName].max = Math.max(kpiStats[kpiName].max, value);
      kpiStats[kpiName].values.push(value);
    }

    // Calculate mean and std dev
    for (const kpiName in kpiStats) {
      const stats = kpiStats[kpiName];
      stats.mean = stats.sum / stats.count;

      const variance = stats.values.reduce((acc: number, v: number) =>
        acc + Math.pow(v - stats.mean, 2), 0) / stats.count;
      stats.stdDev = Math.sqrt(variance);

      delete stats.values; // Remove raw values to reduce prompt size
    }

    return { kpiStats, minTimestamp, maxTimestamp };
  }

  private parseKpiResponse(response: string, analysisType: string): KpiAnalysisOutput {
    const output: KpiAnalysisOutput = {
      summary: '',
      insights: [],
    };

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        switch (analysisType) {
          case 'trend':
            output.trends = Array.isArray(parsed) ? parsed : [parsed];
            break;
          case 'anomaly':
            output.anomalies = Array.isArray(parsed) ? parsed : [parsed];
            break;
          case 'correlation':
            output.correlations = Array.isArray(parsed) ? parsed : [parsed];
            break;
          case 'forecast':
            output.forecasts = Array.isArray(parsed) ? parsed : [parsed];
            break;
          case 'optimization':
            output.optimizations = Array.isArray(parsed) ? parsed : [parsed];
            break;
        }

        output.summary = `Completed ${analysisType} analysis`;
      } else {
        output.summary = response;
      }

      // Extract insights from response
      const insightPatterns = [
        /insight[s]?:/gi,
        /key finding[s]?:/gi,
        /recommendation[s]?:/gi,
      ];

      for (const pattern of insightPatterns) {
        const matches = response.match(new RegExp(`${pattern.source}[^]*?(?=\\n\\n|$)`, 'gi'));
        if (matches) {
          output.insights.push(...matches);
        }
      }
    } catch (error) {
      this.log('warn', 'Failed to parse KPI analysis response', { error });
      output.summary = response;
    }

    return output;
  }

  /**
   * Analyze KPI trends
   */
  async analyzeTrends(measurements: KpiMeasurement[], granularity: Granularity = 'Hourly'): Promise<any[]> {
    const result = await this.execute({
      measurements,
      granularity,
      analysisType: 'trend',
    });
    return result.data?.trends || [];
  }

  /**
   * Detect KPI anomalies
   */
  async detectAnomalies(measurements: KpiMeasurement[], granularity: Granularity = 'Hourly'): Promise<any[]> {
    const result = await this.execute({
      measurements,
      granularity,
      analysisType: 'anomaly',
    });
    return result.data?.anomalies || [];
  }

  /**
   * Forecast KPI values
   */
  async forecastKpis(measurements: KpiMeasurement[], horizonHours: number = 168): Promise<any[]> {
    const result = await this.execute({
      measurements,
      analysisType: 'forecast',
      forecastHorizon: horizonHours,
    });
    return result.data?.forecasts || [];
  }
}

export default KpiAnalyzerAgent;
