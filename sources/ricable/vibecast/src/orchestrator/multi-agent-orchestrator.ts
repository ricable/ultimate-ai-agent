// Multi-Agent Orchestrator using claude-agent-sdk patterns
// Coordinates multiple specialized agents for comprehensive RAN automation

import { RanDocsAgent } from '../agents/ran-docs-agent.js';
import { AlarmFaultAgent } from '../agents/alarm-fault-agent.js';
import { KpiAnalyzerAgent } from '../agents/kpi-analyzer-agent.js';
import { ConfigManagementAgent } from '../agents/config-management-agent.js';
import { BaseAgent, AgentResult } from '../agents/base-agent.js';
import { logger } from '../core/logger.js';
import { AgentTask } from '../types/ran-models.js';
import { EventEmitter } from 'events';

export interface OrchestrationRequest {
  taskType: 'full-analysis' | 'alarm-response' | 'optimization' | 'documentation-query' | 'custom';
  input: any;
  priority?: number;
  requiredAgents?: string[];
}

export interface OrchestrationResult {
  success: boolean;
  results: Map<string, any>;
  aggregatedInsights: string[];
  recommendations: string[];
  timestamp: number;
  executionTimeMs: number;
}

export class MultiAgentOrchestrator extends EventEmitter {
  private agents: Map<string, BaseAgent>;
  private taskQueue: AgentTask[] = [];
  private activeTasksCount: number = 0;
  private maxConcurrentTasks: number;

  constructor(maxConcurrentTasks: number = 10) {
    super();
    this.maxConcurrentTasks = maxConcurrentTasks;
    this.agents = new Map();
    this.initializeAgents();
  }

  private initializeAgents(): void {
    logger.info('Initializing multi-agent system');

    // Register specialized agents
    this.registerAgent('ran-docs', new RanDocsAgent());
    this.registerAgent('alarm-fault', new AlarmFaultAgent());
    this.registerAgent('kpi-analyzer', new KpiAnalyzerAgent());
    this.registerAgent('config-management', new ConfigManagementAgent());

    logger.info(`Registered ${this.agents.size} agents`, {
      agents: Array.from(this.agents.keys()),
    });
  }

  private registerAgent(name: string, agent: BaseAgent): void {
    this.agents.set(name, agent);
    logger.debug(`Registered agent: ${name}`);
  }

  /**
   * Orchestrate a complex task across multiple agents
   */
  async orchestrate(request: OrchestrationRequest): Promise<OrchestrationResult> {
    const startTime = Date.now();
    logger.info('Starting orchestration', { taskType: request.taskType });

    try {
      let results: Map<string, any>;

      switch (request.taskType) {
        case 'full-analysis':
          results = await this.executeFullAnalysis(request.input);
          break;
        case 'alarm-response':
          results = await this.executeAlarmResponse(request.input);
          break;
        case 'optimization':
          results = await this.executeOptimization(request.input);
          break;
        case 'documentation-query':
          results = await this.executeDocumentationQuery(request.input);
          break;
        case 'custom':
          results = await this.executeCustomWorkflow(request);
          break;
        default:
          throw new Error(`Unknown task type: ${request.taskType}`);
      }

      const aggregatedInsights = this.aggregateInsights(results);
      const recommendations = this.generateRecommendations(results);

      const executionTimeMs = Date.now() - startTime;

      logger.info('Orchestration completed', {
        taskType: request.taskType,
        executionTimeMs,
        agentsUsed: results.size,
      });

      return {
        success: true,
        results,
        aggregatedInsights,
        recommendations,
        timestamp: Date.now(),
        executionTimeMs,
      };
    } catch (error) {
      logger.error('Orchestration failed', { error, taskType: request.taskType });

      return {
        success: false,
        results: new Map(),
        aggregatedInsights: [],
        recommendations: [],
        timestamp: Date.now(),
        executionTimeMs: Date.now() - startTime,
      };
    }
  }

  /**
   * Execute full network analysis using all agents
   */
  private async executeFullAnalysis(input: any): Promise<Map<string, any>> {
    const results = new Map<string, any>();

    // Parallel execution of independent analysis tasks
    const tasks = [
      this.executeAgentTask('kpi-analyzer', {
        measurements: input.kpiMeasurements || [],
        analysisType: 'trend',
      }),
      this.executeAgentTask('alarm-fault', {
        alarms: input.alarms || [],
        analysisType: 'correlation',
      }),
    ];

    // Execute in parallel
    const taskResults = await Promise.allSettled(tasks);

    taskResults.forEach((result, idx) => {
      const agentName = idx === 0 ? 'kpi-analyzer' : 'alarm-fault';
      if (result.status === 'fulfilled') {
        results.set(agentName, result.value);
      } else {
        logger.error(`Agent ${agentName} failed`, { error: result.reason });
      }
    });

    // Sequential: Use KPI and alarm insights to generate config proposals
    if (input.node && results.size > 0) {
      const configResult = await this.executeAgentTask('config-management', {
        node: input.node,
        kpiMeasurements: input.kpiMeasurements || [],
        performanceIssues: this.extractPerformanceIssues(results),
      });

      results.set('config-management', configResult);
    }

    return results;
  }

  /**
   * Execute alarm response workflow
   */
  private async executeAlarmResponse(input: any): Promise<Map<string, any>> {
    const results = new Map<string, any>();

    // Step 1: Correlate alarms
    const correlationResult = await this.executeAgentTask('alarm-fault', {
      alarms: input.alarms,
      analysisType: 'correlation',
    });
    results.set('alarm-correlation', correlationResult);

    // Step 2: Find root cause
    const rootCauseResult = await this.executeAgentTask('alarm-fault', {
      alarms: input.alarms,
      faultEvents: input.faultEvents,
      analysisType: 'root-cause',
    });
    results.set('root-cause', rootCauseResult);

    // Step 3: Suggest remediation
    const remediationResult = await this.executeAgentTask('alarm-fault', {
      alarms: input.alarms,
      analysisType: 'remediation',
    });
    results.set('remediation', remediationResult);

    // Step 4: Query documentation for related procedures (if available)
    if (correlationResult.data?.correlatedAlarms?.length > 0) {
      const alarmTypes = correlationResult.data.correlatedAlarms
        .flatMap((g: any) => g.alarmGroup || [])
        .map((a: any) => a.alarmType)
        .join(', ');

      const docsResult = await this.executeAgentTask('ran-docs', {
        query: `Troubleshooting procedures for alarms: ${alarmTypes}`,
      });
      results.set('documentation', docsResult);
    }

    return results;
  }

  /**
   * Execute optimization workflow
   */
  private async executeOptimization(input: any): Promise<Map<string, any>> {
    const results = new Map<string, any>();

    // Step 1: Analyze KPIs to identify issues
    const kpiAnalysis = await this.executeAgentTask('kpi-analyzer', {
      measurements: input.kpiMeasurements,
      analysisType: 'optimization',
    });
    results.set('kpi-analysis', kpiAnalysis);

    // Step 2: Analyze correlations
    const correlationAnalysis = await this.executeAgentTask('kpi-analyzer', {
      measurements: input.kpiMeasurements,
      analysisType: 'correlation',
    });
    results.set('kpi-correlations', correlationAnalysis);

    // Step 3: Generate configuration proposals
    const performanceIssues = this.extractPerformanceIssues(results);
    const configProposals = await this.executeAgentTask('config-management', {
      node: input.node,
      kpiMeasurements: input.kpiMeasurements,
      performanceIssues,
      optimizationGoals: input.optimizationGoals || [],
    });
    results.set('config-proposals', configProposals);

    return results;
  }

  /**
   * Execute documentation query
   */
  private async executeDocumentationQuery(input: any): Promise<Map<string, any>> {
    const results = new Map<string, any>();

    const docsResult = await this.executeAgentTask('ran-docs', input);
    results.set('documentation', docsResult);

    return results;
  }

  /**
   * Execute custom workflow with specified agents
   */
  private async executeCustomWorkflow(request: OrchestrationRequest): Promise<Map<string, any>> {
    const results = new Map<string, any>();

    if (!request.requiredAgents || request.requiredAgents.length === 0) {
      throw new Error('Custom workflow requires specifying agents');
    }

    // Execute agents in parallel
    const tasks = request.requiredAgents.map(agentName =>
      this.executeAgentTask(agentName, request.input[agentName] || request.input)
    );

    const taskResults = await Promise.allSettled(tasks);

    taskResults.forEach((result, idx) => {
      const agentName = request.requiredAgents![idx];
      if (result.status === 'fulfilled') {
        results.set(agentName, result.value);
      } else {
        logger.error(`Agent ${agentName} failed`, { error: result.reason });
      }
    });

    return results;
  }

  /**
   * Execute a single agent task
   */
  private async executeAgentTask(agentName: string, input: any): Promise<any> {
    const agent = this.agents.get(agentName);

    if (!agent) {
      throw new Error(`Agent not found: ${agentName}`);
    }

    logger.debug(`Executing agent: ${agentName}`);
    this.activeTasksCount++;

    try {
      const result = await agent.execute(input);
      return result;
    } finally {
      this.activeTasksCount--;
    }
  }

  /**
   * Extract performance issues from agent results
   */
  private extractPerformanceIssues(results: Map<string, any>): string[] {
    const issues: string[] = [];

    for (const [agentName, result] of results) {
      if (!result.success || !result.data) continue;

      // Extract from KPI analysis
      if (result.data.trends) {
        for (const trend of result.data.trends) {
          if (trend.trend === 'decreasing' && trend.significance > 0.7) {
            issues.push(`${trend.kpiName} showing significant decrease (${trend.changeRate.toFixed(2)}%)`);
          }
        }
      }

      // Extract from anomalies
      if (result.data.anomalies) {
        for (const anomaly of result.data.anomalies) {
          if (anomaly.severity === 'Critical' || anomaly.severity === 'Major') {
            issues.push(`Critical anomaly in ${anomaly.kpiName} at ${new Date(anomaly.timestamp * 1000).toISOString()}`);
          }
        }
      }

      // Extract from alarm correlation
      if (result.data.correlatedAlarms) {
        for (const group of result.data.correlatedAlarms) {
          if (group.severity === 'Critical' || group.severity === 'Major') {
            issues.push(`${group.severity} alarm group affecting ${group.affectedNodes.length} nodes`);
          }
        }
      }
    }

    return issues;
  }

  /**
   * Aggregate insights from multiple agents
   */
  private aggregateInsights(results: Map<string, any>): string[] {
    const insights: string[] = [];

    for (const [agentName, result] of results) {
      if (!result.success || !result.data) continue;

      // Add summary if available
      if (result.data.summary) {
        insights.push(`[${agentName}] ${result.data.summary}`);
      }

      // Add specific insights
      if (result.data.insights && Array.isArray(result.data.insights)) {
        insights.push(...result.data.insights.map((i: string) => `[${agentName}] ${i}`));
      }

      // Add reasoning
      if (result.reasoning) {
        insights.push(`[${agentName}] ${result.reasoning}`);
      }
    }

    return insights;
  }

  /**
   * Generate overall recommendations
   */
  private generateRecommendations(results: Map<string, any>): string[] {
    const recommendations: string[] = [];

    for (const [agentName, result] of results) {
      if (!result.success || !result.data) continue;

      // From config management
      if (result.data.proposals && Array.isArray(result.data.proposals)) {
        for (const proposal of result.data.proposals) {
          recommendations.push(
            `[Config] ${proposal.parameterName}: ${proposal.currentValue} → ${proposal.proposedValue} (${proposal.rationale})`
          );
        }
      }

      // From alarm fault
      if (result.data.remediation) {
        if (result.data.remediation.immediateActions) {
          recommendations.push(...result.data.remediation.immediateActions.map((a: string) => `[Immediate] ${a}`));
        }
        if (result.data.remediation.shortTermActions) {
          recommendations.push(...result.data.remediation.shortTermActions.map((a: string) => `[Short-term] ${a}`));
        }
      }

      // From KPI optimization
      if (result.data.optimizations && Array.isArray(result.data.optimizations)) {
        for (const opt of result.data.optimizations) {
          recommendations.push(
            `[Optimization] ${opt.kpiName}: Target ${opt.targetPerformance} (${opt.expectedImprovement}% improvement)`
          );
        }
      }
    }

    return recommendations;
  }

  /**
   * Get agent by name
   */
  getAgent(name: string): BaseAgent | undefined {
    return this.agents.get(name);
  }

  /**
   * Get all agent names
   */
  getAgentNames(): string[] {
    return Array.from(this.agents.keys());
  }

  /**
   * Get orchestrator status
   */
  getStatus(): {
    totalAgents: number;
    activeAgents: string[];
    activeTasks: number;
    queuedTasks: number;
  } {
    return {
      totalAgents: this.agents.size,
      activeAgents: this.getAgentNames(),
      activeTasks: this.activeTasksCount,
      queuedTasks: this.taskQueue.length,
    };
  }
}

export default MultiAgentOrchestrator;
