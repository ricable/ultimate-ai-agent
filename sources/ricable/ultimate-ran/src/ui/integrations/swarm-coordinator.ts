/**
 * AI Swarm Coordinator - Multi-Agent Consensus-Based Reasoning
 * Orchestrates Claude Agent SDK + Google ADK swarms for distributed decision-making
 *
 * @module ui/integrations/swarm-coordinator
 * @version 7.0.0-alpha.1
 */

import { ClaudeAgentIntegration } from './claude-agent-integration.js';
import { GoogleADKIntegration } from './google-adk-integration.js';
import { AIOrchestrator } from './ai-orchestrator.js';
import type {
  CellStatus,
  InterferenceMatrix,
  OptimizationEvent,
  ParameterChange,
  ApprovalRequest
} from '../types.js';

// ============================================================================
// Swarm Configuration
// ============================================================================

export interface SwarmAgent {
  id: string;
  type: 'claude' | 'gemini';
  role: 'analyzer' | 'optimizer' | 'validator' | 'coordinator';
  status: 'idle' | 'active' | 'busy' | 'error';
  confidence: number;
  tasksCompleted: number;
  currentTask?: string;
}

export interface SwarmConfig {
  claude: {
    apiKey: string;
    model?: string;
  };
  gemini: {
    apiKey: string;
    model?: string;
  };
  topology: 'hierarchical' | 'mesh' | 'consensus';
  consensusThreshold: number;
  maxAgents: number;
  enableLearning: boolean;
}

export interface ConsensusResult {
  decision: 'approved' | 'rejected' | 'needs_review';
  confidence: number;
  votes: {
    agent_id: string;
    vote: boolean;
    confidence: number;
    reasoning: string;
  }[];
  finalReasoning: string;
  timestamp: Date;
}

export interface SwarmTask {
  id: string;
  type: 'optimization' | 'analysis' | 'validation' | 'prediction';
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  assignedAgents: string[];
  result?: any;
  consensus?: ConsensusResult;
  createdAt: Date;
  completedAt?: Date;
}

// ============================================================================
// AI Swarm Coordinator Class
// ============================================================================

export class AISwarmCoordinator {
  private config: Required<SwarmConfig>;
  private agents: Map<string, SwarmAgent>;
  private orchestrator: AIOrchestrator;
  private claudeAgents: ClaudeAgentIntegration[];
  private geminiAgents: GoogleADKIntegration[];
  private tasks: Map<string, SwarmTask>;
  private consensusHistory: ConsensusResult[];

  constructor(config: SwarmConfig) {
    this.config = {
      claude: config.claude,
      gemini: config.gemini,
      topology: config.topology || 'consensus',
      consensusThreshold: config.consensusThreshold ?? 0.75,
      maxAgents: config.maxAgents || 6,
      enableLearning: config.enableLearning !== false
    };

    this.agents = new Map();
    this.tasks = new Map();
    this.consensusHistory = [];
    this.claudeAgents = [];
    this.geminiAgents = [];

    // Initialize orchestrator
    this.orchestrator = new AIOrchestrator({
      claude: this.config.claude,
      gemini: this.config.gemini,
      strategy: 'consensus',
      consensusThreshold: this.config.consensusThreshold
    });

    // Initialize agent swarm
    this.initializeSwarm();

    console.log(`[Swarm Coordinator] Initialized with ${this.agents.size} agents in ${this.config.topology} topology`);
  }

  /**
   * Initialize the agent swarm based on topology
   */
  private initializeSwarm(): void {
    const agentCount = Math.min(this.config.maxAgents, 6);
    const roles: Array<SwarmAgent['role']> = ['analyzer', 'optimizer', 'validator', 'coordinator'];

    // Create Claude agents (50%)
    for (let i = 0; i < Math.ceil(agentCount / 2); i++) {
      const agent = new ClaudeAgentIntegration({
        apiKey: this.config.claude.apiKey,
        model: this.config.claude.model
      });

      this.claudeAgents.push(agent);

      const swarmAgent: SwarmAgent = {
        id: `claude-${i + 1}`,
        type: 'claude',
        role: roles[i % roles.length],
        status: 'idle',
        confidence: 0.85,
        tasksCompleted: 0
      };

      this.agents.set(swarmAgent.id, swarmAgent);
    }

    // Create Gemini agents (50%)
    for (let i = 0; i < Math.floor(agentCount / 2); i++) {
      const agent = new GoogleADKIntegration({
        apiKey: this.config.gemini.apiKey,
        model: this.config.gemini.model
      });

      this.geminiAgents.push(agent);

      const swarmAgent: SwarmAgent = {
        id: `gemini-${i + 1}`,
        type: 'gemini',
        role: roles[i % roles.length],
        status: 'idle',
        confidence: 0.82,
        tasksCompleted: 0
      };

      this.agents.set(swarmAgent.id, swarmAgent);
    }
  }

  /**
   * Request consensus-based RAN optimization
   */
  async requestConsensusOptimization(
    cells: CellStatus[],
    objective: string,
    interferenceMatrix?: InterferenceMatrix
  ): Promise<{
    recommendations: ParameterChange[];
    consensus: ConsensusResult;
    participatingAgents: string[];
  }> {
    const taskId = `opt-${Date.now()}`;
    const task: SwarmTask = {
      id: taskId,
      type: 'optimization',
      status: 'in_progress',
      assignedAgents: [],
      createdAt: new Date()
    };

    this.tasks.set(taskId, task);

    console.log(`[Swarm] Starting consensus optimization task ${taskId}`);

    // Select agents for the task (mix of Claude and Gemini)
    const selectedAgents = this.selectAgentsForTask('optimization');
    task.assignedAgents = selectedAgents.map(a => a.id);

    // Mark agents as busy
    selectedAgents.forEach(agent => {
      agent.status = 'busy';
      agent.currentTask = taskId;
    });

    try {
      // Get recommendations from all agents in parallel
      const agentResults = await Promise.all(
        selectedAgents.map(async (agent) => {
          if (agent.type === 'claude') {
            const claudeAgent = this.claudeAgents[parseInt(agent.id.split('-')[1]) - 1];
            const result = await claudeAgent.requestOptimization(cells, objective);
            return {
              agent_id: agent.id,
              recommendations: result.recommendations,
              confidence: result.confidence,
              reasoning: result.reasoning
            };
          } else {
            const geminiAgent = this.geminiAgents[parseInt(agent.id.split('-')[1]) - 1];
            const result = await geminiAgent.analyzeNetworkPerformance(cells, interferenceMatrix);
            return {
              agent_id: agent.id,
              recommendations: result.recommendations,
              confidence: result.confidence,
              reasoning: result.analysis
            };
          }
        })
      );

      // Build consensus from agent results
      const consensus = await this.buildConsensus(agentResults, 'optimization');

      // Combine recommendations where agents agree
      const consensusRecommendations = this.mergeRecommendations(
        agentResults.map(r => r.recommendations)
      );

      // Update agent stats
      selectedAgents.forEach(agent => {
        agent.status = 'idle';
        agent.currentTask = undefined;
        agent.tasksCompleted++;
        agent.confidence = (agent.confidence + consensus.confidence) / 2; // Update confidence
      });

      // Complete task
      task.status = 'completed';
      task.completedAt = new Date();
      task.consensus = consensus;
      task.result = { recommendations: consensusRecommendations };

      // Store consensus history
      this.consensusHistory.push(consensus);

      console.log(`[Swarm] Task ${taskId} completed with ${consensus.decision} (confidence: ${consensus.confidence})`);

      return {
        recommendations: consensusRecommendations,
        consensus,
        participatingAgents: selectedAgents.map(a => a.id)
      };
    } catch (error) {
      task.status = 'failed';
      selectedAgents.forEach(agent => {
        agent.status = 'error';
        agent.currentTask = undefined;
      });
      throw error;
    }
  }

  /**
   * Validate approval with consensus voting
   */
  async validateWithConsensus(request: ApprovalRequest): Promise<ConsensusResult> {
    const taskId = `val-${Date.now()}`;
    const task: SwarmTask = {
      id: taskId,
      type: 'validation',
      status: 'in_progress',
      assignedAgents: [],
      createdAt: new Date()
    };

    this.tasks.set(taskId, task);

    console.log(`[Swarm] Starting consensus validation task ${taskId}`);

    // Select validator agents
    const validators = this.selectAgentsForTask('validation');
    task.assignedAgents = validators.map(a => a.id);

    validators.forEach(agent => {
      agent.status = 'busy';
      agent.currentTask = taskId;
    });

    try {
      // Get votes from all validators
      const votes = await Promise.all(
        validators.map(async (agent) => {
          if (agent.type === 'claude') {
            const claudeAgent = this.claudeAgents[parseInt(agent.id.split('-')[1]) - 1];
            const result = await claudeAgent.validateApprovalRequest(request);
            return {
              agent_id: agent.id,
              vote: result.approved,
              confidence: result.approved ? 0.9 : 0.8,
              reasoning: result.reasoning
            };
          } else {
            const geminiAgent = this.geminiAgents[parseInt(agent.id.split('-')[1]) - 1];
            // Simplified validation for Gemini
            return {
              agent_id: agent.id,
              vote: request.risk_level !== 'critical',
              confidence: 0.85,
              reasoning: `Risk level: ${request.risk_level}`
            };
          }
        })
      );

      // Calculate consensus
      const approvals = votes.filter(v => v.vote).length;
      const totalVotes = votes.length;
      const approvalRate = approvals / totalVotes;

      const avgConfidence = votes.reduce((sum, v) => sum + v.confidence, 0) / votes.length;

      const decision: ConsensusResult['decision'] =
        approvalRate >= this.config.consensusThreshold
          ? 'approved'
          : approvalRate <= (1 - this.config.consensusThreshold)
          ? 'rejected'
          : 'needs_review';

      const consensus: ConsensusResult = {
        decision,
        confidence: avgConfidence,
        votes,
        finalReasoning: `
**Consensus Validation Result**

Votes: ${approvals}/${totalVotes} approve (${(approvalRate * 100).toFixed(1)}%)
Threshold: ${(this.config.consensusThreshold * 100).toFixed(0)}%
Average Confidence: ${(avgConfidence * 100).toFixed(1)}%

**Agent Votes:**
${votes.map(v => `- ${v.agent_id}: ${v.vote ? 'APPROVE' : 'REJECT'} (${(v.confidence * 100).toFixed(0)}% confidence)`).join('\n')}

**Final Decision:** ${decision.toUpperCase()}
`,
        timestamp: new Date()
      };

      // Update agents
      validators.forEach(agent => {
        agent.status = 'idle';
        agent.currentTask = undefined;
        agent.tasksCompleted++;
      });

      task.status = 'completed';
      task.completedAt = new Date();
      task.consensus = consensus;

      this.consensusHistory.push(consensus);

      console.log(`[Swarm] Validation ${taskId} completed: ${decision}`);

      return consensus;
    } catch (error) {
      task.status = 'failed';
      validators.forEach(agent => {
        agent.status = 'error';
        agent.currentTask = undefined;
      });
      throw error;
    }
  }

  /**
   * Select agents for a specific task based on role and availability
   */
  private selectAgentsForTask(taskType: string): SwarmAgent[] {
    const availableAgents = Array.from(this.agents.values()).filter(a => a.status === 'idle');

    switch (this.config.topology) {
      case 'hierarchical':
        // Coordinator leads, others assist
        const coordinator = availableAgents.find(a => a.role === 'coordinator');
        const assistants = availableAgents.filter(a => a.role !== 'coordinator').slice(0, 2);
        return coordinator ? [coordinator, ...assistants] : assistants.slice(0, 3);

      case 'mesh':
        // All agents participate
        return availableAgents.slice(0, this.config.maxAgents);

      case 'consensus':
      default:
        // Select mix of Claude and Gemini agents
        const claudeAgents = availableAgents.filter(a => a.type === 'claude').slice(0, 2);
        const geminiAgents = availableAgents.filter(a => a.type === 'gemini').slice(0, 2);
        return [...claudeAgents, ...geminiAgents];
    }
  }

  /**
   * Build consensus from multiple agent results
   */
  private async buildConsensus(
    results: Array<{ agent_id: string; confidence: number; reasoning: string }>,
    taskType: string
  ): Promise<ConsensusResult> {
    // Calculate average confidence
    const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;

    // Determine decision based on confidence threshold
    const highConfidence = results.filter(r => r.confidence >= this.config.consensusThreshold);
    const decision: ConsensusResult['decision'] =
      highConfidence.length / results.length >= 0.5
        ? 'approved'
        : 'needs_review';

    const votes = results.map(r => ({
      agent_id: r.agent_id,
      vote: r.confidence >= this.config.consensusThreshold,
      confidence: r.confidence,
      reasoning: r.reasoning.slice(0, 200) // Truncate for display
    }));

    return {
      decision,
      confidence: avgConfidence,
      votes,
      finalReasoning: `Consensus reached with ${(avgConfidence * 100).toFixed(1)}% average confidence across ${results.length} agents.`,
      timestamp: new Date()
    };
  }

  /**
   * Merge recommendations from multiple agents
   */
  private mergeRecommendations(recommendations: ParameterChange[][]): ParameterChange[] {
    const merged: ParameterChange[] = [];
    const seen = new Set<string>();

    for (const agentRecs of recommendations) {
      for (const rec of agentRecs) {
        const key = `${rec.cell_id}:${rec.parameter}`;
        if (!seen.has(key)) {
          // Find all recommendations for this parameter
          const allRecs = recommendations.flat().filter(
            r => r.cell_id === rec.cell_id && r.parameter === rec.parameter
          );

          if (allRecs.length >= 2) {
            // Consensus exists - average the values
            const avgValue = allRecs.reduce((sum, r) => sum + r.new_value, 0) / allRecs.length;
            merged.push({
              ...rec,
              new_value: avgValue
            });
            seen.add(key);
          }
        }
      }
    }

    return merged;
  }

  /**
   * Get swarm status and agent details
   */
  getSwarmStatus(): {
    topology: string;
    totalAgents: number;
    activeAgents: number;
    idleAgents: number;
    busyAgents: number;
    agents: SwarmAgent[];
    tasks: SwarmTask[];
    consensusHistory: ConsensusResult[];
  } {
    const agents = Array.from(this.agents.values());

    return {
      topology: this.config.topology,
      totalAgents: agents.length,
      activeAgents: agents.filter(a => a.status !== 'error').length,
      idleAgents: agents.filter(a => a.status === 'idle').length,
      busyAgents: agents.filter(a => a.status === 'busy').length,
      agents,
      tasks: Array.from(this.tasks.values()),
      consensusHistory: this.consensusHistory.slice(-10) // Last 10 consensus results
    };
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): SwarmAgent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Clear all agent histories and reset swarm
   */
  resetSwarm(): void {
    this.claudeAgents.forEach(a => a.clearHistory());
    this.geminiAgents.forEach(a => a.startNewSession());
    this.agents.forEach(agent => {
      agent.status = 'idle';
      agent.currentTask = undefined;
      agent.tasksCompleted = 0;
    });
    this.tasks.clear();
    console.log('[Swarm] Reset complete');
  }
}

// ============================================================================
// Export
// ============================================================================

export default AISwarmCoordinator;
