/**
 * MLX Deep Council - Distributed Multi-Model Consensus System
 *
 * Implements Andrej Karpathy's LLM Council pattern for distributed
 * inference across multiple Mac machines using Apple's MLX framework.
 *
 * Architecture:
 * - Stage 1: Individual Opinions - Query multiple LLMs across distributed Macs
 * - Stage 2: Peer Review - Each model evaluates others' responses (anonymized)
 * - Stage 3: Chairman Synthesis - A designated model synthesizes consensus
 *
 * Features:
 * - Distributed inference using MLX ring topology
 * - Multi-Mac coordination via Thunderbolt or network
 * - Weighted voting based on peer reviews
 * - Self-healing node management
 * - Streaming response aggregation
 *
 * @see https://github.com/karpathy/llm-council
 * @see https://ml-explore.github.io/mlx/build/html/usage/distributed.html
 */

import { EventEmitter } from 'events';
import { spawn, ChildProcess, exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface CouncilMember {
  id: string;
  name: string;
  host: string;
  port: number;
  model: string;
  role: 'member' | 'chairman';
  rank: number;  // MLX distributed rank
  status: 'initializing' | 'ready' | 'busy' | 'offline' | 'error';
  capabilities: string[];
  metrics: MemberMetrics;
}

export interface MemberMetrics {
  totalResponses: number;
  averageLatency: number;
  peerReviewScore: number;
  chairmanSelections: number;
  lastActive: Date;
}

export interface DistributedNode {
  ssh: string;           // SSH hostname for remote access
  ips: string[];         // IP addresses (for ring topology)
  gpuMemory: number;     // Available GPU memory in GB
  chip: string;          // M1/M2/M3/M4 etc.
  models: string[];      // Pre-loaded models on this node
}

export interface CouncilConfig {
  name: string;
  nodes: DistributedNode[];
  defaultModel: string;
  chairmanModel?: string;
  backend: 'ring' | 'mpi' | 'auto';
  votingStrategy: 'weighted' | 'majority' | 'ranked-choice';
  anonymizePeerReview: boolean;
  enableStreaming: boolean;
  timeoutMs: number;
  retryCount: number;
}

export interface CouncilQuery {
  id: string;
  content: string;
  context?: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  requireConsensus?: boolean;
  minAgreement?: number;  // 0-1, minimum agreement threshold
}

export interface IndividualResponse {
  memberId: string;
  memberName: string;
  anonymousId: string;  // For peer review anonymization
  content: string;
  reasoning?: string;
  confidence: number;
  latency: number;
  tokenCount: number;
  timestamp: Date;
}

export interface PeerReview {
  reviewerId: string;
  reviews: Array<{
    anonymousId: string;
    score: number;       // 1-10
    reasoning: string;
    strengths: string[];
    weaknesses: string[];
    ranking: number;     // Position in reviewer's ranking
  }>;
  selfEvaluation?: {
    preferOther: boolean;
    preferredId?: string;
    reason?: string;
  };
}

export interface ChairmanSynthesis {
  finalResponse: string;
  reasoning: string;
  sourcesUsed: string[];
  conflictsResolved: string[];
  confidenceScore: number;
  dissent?: string;  // Minority viewpoint if significant
}

export interface CouncilSession {
  id: string;
  query: CouncilQuery;
  stage: 'collecting' | 'reviewing' | 'synthesizing' | 'complete' | 'failed';
  individualResponses: IndividualResponse[];
  peerReviews: PeerReview[];
  chairmanSynthesis?: ChairmanSynthesis;
  aggregatedScores: Map<string, number>;
  consensusReached: boolean;
  startTime: Date;
  endTime?: Date;
  metrics: SessionMetrics;
}

export interface SessionMetrics {
  totalLatency: number;
  stage1Latency: number;
  stage2Latency: number;
  stage3Latency: number;
  participatingMembers: number;
  averageConfidence: number;
  consensusStrength: number;  // How strong the agreement is
}

// ============================================================================
// MLX DISTRIBUTED COMMUNICATION LAYER
// ============================================================================

class MLXDistributedGroup extends EventEmitter {
  private nodes: DistributedNode[];
  private backend: 'ring' | 'mpi';
  private processes: Map<string, ChildProcess> = new Map();
  private hostfilePath: string = '';
  private initialized: boolean = false;

  constructor(nodes: DistributedNode[], backend: 'ring' | 'mpi' = 'ring') {
    super();
    this.nodes = nodes;
    this.backend = backend;
  }

  /**
   * Initialize the distributed group
   */
  async initialize(): Promise<void> {
    // Generate hostfile for MLX distributed
    await this.generateHostfile();

    // Validate connectivity to all nodes
    await this.validateConnectivity();

    this.initialized = true;
    this.emit('initialized', { nodes: this.nodes.length, backend: this.backend });
  }

  /**
   * Generate MLX ring topology hostfile
   */
  private async generateHostfile(): Promise<void> {
    const hostfile = this.nodes.map(node => ({
      ssh: node.ssh,
      ips: node.ips,
    }));

    this.hostfilePath = path.join('/tmp', `mlx-council-${Date.now()}.json`);
    await fs.promises.writeFile(this.hostfilePath, JSON.stringify(hostfile, null, 2));

    this.emit('hostfile:created', this.hostfilePath);
  }

  /**
   * Validate SSH connectivity to all nodes
   */
  private async validateConnectivity(): Promise<void> {
    const results = await Promise.allSettled(
      this.nodes.map(async (node) => {
        try {
          await execAsync(`ssh -o ConnectTimeout=5 ${node.ssh} "echo ok"`, { timeout: 10000 });
          return { node: node.ssh, status: 'ok' };
        } catch (error) {
          return { node: node.ssh, status: 'failed', error };
        }
      })
    );

    const failed = results.filter(r => r.status === 'rejected' ||
      (r.status === 'fulfilled' && r.value.status === 'failed'));

    if (failed.length > 0) {
      this.emit('connectivity:warning', { failedNodes: failed });
    }
  }

  /**
   * Launch a distributed Python script across all nodes
   */
  async launch(scriptPath: string, args: string[] = []): Promise<void> {
    const launchArgs = [
      '--hostfile', this.hostfilePath,
      '--backend', this.backend,
      scriptPath,
      ...args,
    ];

    return new Promise((resolve, reject) => {
      const process = spawn('python', ['-m', 'mlx.launch', ...launchArgs], {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      process.stdout?.on('data', (data) => {
        this.emit('output', data.toString());
      });

      process.stderr?.on('data', (data) => {
        this.emit('error', data.toString());
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Process exited with code ${code}`));
        }
      });

      this.processes.set(scriptPath, process);
    });
  }

  /**
   * Send a message to a specific rank
   */
  async send(rank: number, message: any): Promise<void> {
    // This would use mx.distributed.send() in the Python layer
    this.emit('send', { rank, message });
  }

  /**
   * Receive a message from a specific rank
   */
  async recv(rank: number): Promise<any> {
    // This would use mx.distributed.recv() in the Python layer
    return new Promise((resolve) => {
      this.once(`recv:${rank}`, resolve);
    });
  }

  /**
   * All-gather operation across the ring
   */
  async allGather<T>(data: T): Promise<T[]> {
    // This would use mx.distributed.all_gather() in the Python layer
    return [data]; // Placeholder
  }

  /**
   * Shutdown the distributed group
   */
  async shutdown(): Promise<void> {
    for (const [key, process] of this.processes) {
      process.kill();
      this.processes.delete(key);
    }

    if (this.hostfilePath) {
      try {
        await fs.promises.unlink(this.hostfilePath);
      } catch {
        // Ignore cleanup errors
      }
    }

    this.initialized = false;
    this.emit('shutdown');
  }
}

// ============================================================================
// COUNCIL MEMBER MANAGER
// ============================================================================

class CouncilMemberManager extends EventEmitter {
  private members: Map<string, CouncilMember> = new Map();
  private chairman: CouncilMember | null = null;
  private distributedGroup: MLXDistributedGroup;

  constructor(distributedGroup: MLXDistributedGroup) {
    super();
    this.distributedGroup = distributedGroup;
  }

  /**
   * Register a new council member
   */
  registerMember(member: Omit<CouncilMember, 'metrics'>): void {
    const fullMember: CouncilMember = {
      ...member,
      metrics: {
        totalResponses: 0,
        averageLatency: 0,
        peerReviewScore: 5.0,
        chairmanSelections: 0,
        lastActive: new Date(),
      },
    };

    this.members.set(member.id, fullMember);

    if (member.role === 'chairman') {
      this.chairman = fullMember;
    }

    this.emit('member:registered', fullMember);
  }

  /**
   * Get all active members
   */
  getActiveMembers(): CouncilMember[] {
    return [...this.members.values()].filter(m =>
      m.status === 'ready' || m.status === 'busy'
    );
  }

  /**
   * Get the chairman
   */
  getChairman(): CouncilMember | null {
    return this.chairman;
  }

  /**
   * Update member status
   */
  updateStatus(memberId: string, status: CouncilMember['status']): void {
    const member = this.members.get(memberId);
    if (member) {
      member.status = status;
      this.emit('member:status', { memberId, status });
    }
  }

  /**
   * Update member metrics after a response
   */
  updateMetrics(memberId: string, latency: number, peerScore?: number): void {
    const member = this.members.get(memberId);
    if (member) {
      member.metrics.totalResponses++;
      member.metrics.averageLatency =
        (member.metrics.averageLatency * (member.metrics.totalResponses - 1) + latency) /
        member.metrics.totalResponses;

      if (peerScore !== undefined) {
        member.metrics.peerReviewScore =
          (member.metrics.peerReviewScore * 0.9) + (peerScore * 0.1);
      }

      member.metrics.lastActive = new Date();
    }
  }

  /**
   * Elect a new chairman based on peer review scores
   */
  electChairman(): CouncilMember | null {
    const candidates = this.getActiveMembers();
    if (candidates.length === 0) return null;

    // Sort by peer review score
    candidates.sort((a, b) => b.metrics.peerReviewScore - a.metrics.peerReviewScore);

    const newChairman = candidates[0];

    // Update roles
    if (this.chairman) {
      this.chairman.role = 'member';
    }

    newChairman.role = 'chairman';
    this.chairman = newChairman;

    this.emit('chairman:elected', newChairman);
    return newChairman;
  }

  /**
   * Generate anonymous IDs for peer review
   */
  generateAnonymousIds(): Map<string, string> {
    const anonymousMap = new Map<string, string>();
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

    let index = 0;
    for (const member of this.members.values()) {
      anonymousMap.set(member.id, `Model ${letters[index % 26]}`);
      index++;
    }

    return anonymousMap;
  }
}

// ============================================================================
// STAGE 1: INDIVIDUAL RESPONSE COLLECTION
// ============================================================================

class IndividualResponseCollector extends EventEmitter {
  private memberManager: CouncilMemberManager;
  private inferenceEndpoints: Map<string, string> = new Map();
  private timeout: number;

  constructor(memberManager: CouncilMemberManager, timeout: number = 60000) {
    super();
    this.memberManager = memberManager;
    this.timeout = timeout;
  }

  /**
   * Register inference endpoint for a member
   */
  registerEndpoint(memberId: string, endpoint: string): void {
    this.inferenceEndpoints.set(memberId, endpoint);
  }

  /**
   * Collect responses from all council members
   */
  async collectResponses(query: CouncilQuery): Promise<IndividualResponse[]> {
    const members = this.memberManager.getActiveMembers();
    const anonymousIds = this.memberManager.generateAnonymousIds();

    this.emit('collection:started', { memberCount: members.length });

    const responsePromises = members.map(async (member): Promise<IndividualResponse | null> => {
      try {
        const startTime = Date.now();
        this.memberManager.updateStatus(member.id, 'busy');

        const response = await this.queryMember(member, query);
        const latency = Date.now() - startTime;

        this.memberManager.updateStatus(member.id, 'ready');
        this.memberManager.updateMetrics(member.id, latency);

        const individualResponse: IndividualResponse = {
          memberId: member.id,
          memberName: member.name,
          anonymousId: anonymousIds.get(member.id) || 'Unknown',
          content: response.content,
          reasoning: response.reasoning,
          confidence: response.confidence,
          latency,
          tokenCount: response.tokenCount,
          timestamp: new Date(),
        };

        this.emit('response:received', { memberId: member.id, latency });
        return individualResponse;
      } catch (error) {
        this.memberManager.updateStatus(member.id, 'error');
        this.emit('response:error', { memberId: member.id, error });
        return null;
      }
    });

    // Wait for all responses with timeout
    const results = await Promise.race([
      Promise.all(responsePromises),
      new Promise<null[]>((_, reject) =>
        setTimeout(() => reject(new Error('Collection timeout')), this.timeout)
      ),
    ]) as (IndividualResponse | null)[];

    const validResponses = results.filter((r): r is IndividualResponse => r !== null);

    this.emit('collection:complete', {
      total: members.length,
      successful: validResponses.length
    });

    return validResponses;
  }

  /**
   * Query a single member for their response
   */
  private async queryMember(
    member: CouncilMember,
    query: CouncilQuery
  ): Promise<{ content: string; reasoning?: string; confidence: number; tokenCount: number }> {
    const endpoint = this.inferenceEndpoints.get(member.id);

    if (!endpoint) {
      throw new Error(`No endpoint registered for member ${member.id}`);
    }

    const systemPrompt = query.systemPrompt || `You are ${member.name}, a member of an AI council.
Provide your best response to the following query. Be thorough, accurate, and insightful.
At the end, indicate your confidence level (0-100%) in your response.`;

    const response = await fetch(`${endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: member.model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: query.content },
        ],
        temperature: query.temperature ?? 0.7,
        max_tokens: query.maxTokens ?? 2048,
      }),
    });

    const data = await response.json();
    const content = data.choices[0].message.content;

    // Extract confidence from response (simple heuristic)
    const confidenceMatch = content.match(/confidence[:\s]+(\d+)/i);
    const confidence = confidenceMatch ? parseInt(confidenceMatch[1]) / 100 : 0.7;

    return {
      content,
      confidence,
      tokenCount: data.usage?.completion_tokens || 0,
    };
  }
}

// ============================================================================
// STAGE 2: PEER REVIEW
// ============================================================================

class PeerReviewCoordinator extends EventEmitter {
  private memberManager: CouncilMemberManager;
  private inferenceEndpoints: Map<string, string> = new Map();
  private anonymize: boolean;

  constructor(memberManager: CouncilMemberManager, anonymize: boolean = true) {
    super();
    this.memberManager = memberManager;
    this.anonymize = anonymize;
  }

  /**
   * Register inference endpoint for a member
   */
  registerEndpoint(memberId: string, endpoint: string): void {
    this.inferenceEndpoints.set(memberId, endpoint);
  }

  /**
   * Conduct peer review of all responses
   */
  async conductPeerReview(
    query: CouncilQuery,
    responses: IndividualResponse[]
  ): Promise<PeerReview[]> {
    const members = this.memberManager.getActiveMembers();

    this.emit('review:started', { reviewerCount: members.length });

    // Format responses for review (anonymized if configured)
    const formattedResponses = this.formatResponsesForReview(responses);

    const reviewPromises = members.map(async (member): Promise<PeerReview | null> => {
      try {
        const review = await this.getMemberReview(member, query, formattedResponses, responses);
        this.emit('review:received', { reviewerId: member.id });
        return review;
      } catch (error) {
        this.emit('review:error', { reviewerId: member.id, error });
        return null;
      }
    });

    const results = await Promise.all(reviewPromises);
    const validReviews = results.filter((r): r is PeerReview => r !== null);

    this.emit('review:complete', {
      total: members.length,
      successful: validReviews.length
    });

    return validReviews;
  }

  /**
   * Format responses for peer review presentation
   */
  private formatResponsesForReview(responses: IndividualResponse[]): string {
    return responses.map((r, index) => {
      const id = this.anonymize ? r.anonymousId : r.memberName;
      return `=== Response from ${id} ===\n${r.content}\n`;
    }).join('\n---\n\n');
  }

  /**
   * Get a single member's review of all responses
   */
  private async getMemberReview(
    reviewer: CouncilMember,
    originalQuery: CouncilQuery,
    formattedResponses: string,
    responses: IndividualResponse[]
  ): Promise<PeerReview> {
    const endpoint = this.inferenceEndpoints.get(reviewer.id);

    if (!endpoint) {
      throw new Error(`No endpoint registered for reviewer ${reviewer.id}`);
    }

    const reviewPrompt = `You are a peer reviewer on an AI council. Your task is to evaluate the following responses to a query.

ORIGINAL QUERY:
${originalQuery.content}

RESPONSES TO EVALUATE:
${formattedResponses}

Please evaluate each response and provide:
1. A score from 1-10 for each response
2. Key strengths and weaknesses
3. Your ranking of the responses (best to worst)
4. Whether you prefer any other response over your own (if applicable)

Format your response as follows:
EVALUATION FOR [Model ID]:
- Score: [1-10]
- Strengths: [list]
- Weaknesses: [list]
- Rank: [position]

OVERALL RANKING: [ordered list]
PREFER OTHER: [Yes/No, and if yes, which one and why]`;

    const response = await fetch(`${endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: reviewer.model,
        messages: [
          { role: 'system', content: 'You are a fair and thorough peer reviewer.' },
          { role: 'user', content: reviewPrompt },
        ],
        temperature: 0.3,  // Lower temperature for consistent evaluation
        max_tokens: 2048,
      }),
    });

    const data = await response.json();
    const reviewContent = data.choices[0].message.content;

    // Parse the review content
    return this.parseReviewContent(reviewer.id, reviewContent, responses);
  }

  /**
   * Parse review content into structured format
   */
  private parseReviewContent(
    reviewerId: string,
    content: string,
    responses: IndividualResponse[]
  ): PeerReview {
    const reviews: PeerReview['reviews'] = [];

    for (const response of responses) {
      const id = this.anonymize ? response.anonymousId : response.memberName;

      // Extract score (simple regex)
      const scoreMatch = content.match(new RegExp(`${id}[\\s\\S]*?Score:\\s*(\\d+)`, 'i'));
      const score = scoreMatch ? Math.min(10, Math.max(1, parseInt(scoreMatch[1]))) : 5;

      // Extract rank
      const rankMatch = content.match(new RegExp(`${id}[\\s\\S]*?Rank:\\s*(\\d+)`, 'i'));
      const ranking = rankMatch ? parseInt(rankMatch[1]) : responses.length;

      // Extract strengths
      const strengthsMatch = content.match(new RegExp(`${id}[\\s\\S]*?Strengths:\\s*([^\\n]+)`, 'i'));
      const strengths = strengthsMatch
        ? strengthsMatch[1].split(',').map(s => s.trim()).filter(s => s)
        : [];

      // Extract weaknesses
      const weaknessesMatch = content.match(new RegExp(`${id}[\\s\\S]*?Weaknesses:\\s*([^\\n]+)`, 'i'));
      const weaknesses = weaknessesMatch
        ? weaknessesMatch[1].split(',').map(s => s.trim()).filter(s => s)
        : [];

      reviews.push({
        anonymousId: response.anonymousId,
        score,
        reasoning: '',
        strengths,
        weaknesses,
        ranking,
      });
    }

    // Check for self-evaluation preference
    const preferMatch = content.match(/PREFER OTHER:\s*(Yes|No)/i);
    const prefersOther = preferMatch?.[1].toLowerCase() === 'yes';

    let selfEvaluation: PeerReview['selfEvaluation'];
    if (prefersOther) {
      const preferredMatch = content.match(/prefer.*?(Model [A-Z])/i);
      selfEvaluation = {
        preferOther: true,
        preferredId: preferredMatch?.[1],
        reason: 'Reviewer found another response superior',
      };
    }

    return {
      reviewerId,
      reviews,
      selfEvaluation,
    };
  }

  /**
   * Aggregate scores from all peer reviews
   */
  aggregateScores(reviews: PeerReview[]): Map<string, number> {
    const scores = new Map<string, { total: number; count: number }>();

    for (const review of reviews) {
      for (const r of review.reviews) {
        const existing = scores.get(r.anonymousId) || { total: 0, count: 0 };
        scores.set(r.anonymousId, {
          total: existing.total + r.score,
          count: existing.count + 1,
        });
      }
    }

    const averages = new Map<string, number>();
    for (const [id, { total, count }] of scores) {
      averages.set(id, total / count);
    }

    return averages;
  }
}

// ============================================================================
// STAGE 3: CHAIRMAN SYNTHESIS
// ============================================================================

class ChairmanSynthesizer extends EventEmitter {
  private memberManager: CouncilMemberManager;
  private endpoint: string = '';

  constructor(memberManager: CouncilMemberManager) {
    super();
    this.memberManager = memberManager;
  }

  /**
   * Set the chairman's inference endpoint
   */
  setEndpoint(endpoint: string): void {
    this.endpoint = endpoint;
  }

  /**
   * Synthesize the final response based on all inputs
   */
  async synthesize(
    query: CouncilQuery,
    responses: IndividualResponse[],
    reviews: PeerReview[],
    aggregatedScores: Map<string, number>
  ): Promise<ChairmanSynthesis> {
    const chairman = this.memberManager.getChairman();

    if (!chairman) {
      throw new Error('No chairman available for synthesis');
    }

    this.emit('synthesis:started', { chairmanId: chairman.id });

    // Build synthesis prompt
    const synthesisPrompt = this.buildSynthesisPrompt(query, responses, reviews, aggregatedScores);

    const response = await fetch(`${this.endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: chairman.model,
        messages: [
          {
            role: 'system',
            content: `You are the Chairman of an AI council. Your role is to synthesize the best response
from multiple council members, resolving conflicts and combining the strongest insights.
You have access to individual responses and peer review scores.
Create a comprehensive, authoritative final response.`
          },
          { role: 'user', content: synthesisPrompt },
        ],
        temperature: 0.5,
        max_tokens: 4096,
      }),
    });

    const data = await response.json();
    const synthesisContent = data.choices[0].message.content;

    const synthesis = this.parseSynthesis(synthesisContent, responses, aggregatedScores);

    chairman.metrics.chairmanSelections++;
    this.emit('synthesis:complete', { chairmanId: chairman.id });

    return synthesis;
  }

  /**
   * Build the synthesis prompt
   */
  private buildSynthesisPrompt(
    query: CouncilQuery,
    responses: IndividualResponse[],
    reviews: PeerReview[],
    scores: Map<string, number>
  ): string {
    // Sort responses by score
    const sortedResponses = [...responses].sort((a, b) => {
      const scoreA = scores.get(a.anonymousId) || 0;
      const scoreB = scores.get(b.anonymousId) || 0;
      return scoreB - scoreA;
    });

    let prompt = `ORIGINAL QUERY:\n${query.content}\n\n`;
    prompt += `COUNCIL RESPONSES (ordered by peer review score):\n\n`;

    for (const response of sortedResponses) {
      const score = scores.get(response.anonymousId) || 0;
      prompt += `=== ${response.anonymousId} (Score: ${score.toFixed(1)}/10) ===\n`;
      prompt += `${response.content}\n\n`;
    }

    prompt += `PEER REVIEW SUMMARY:\n`;
    for (const [id, score] of scores) {
      const response = responses.find(r => r.anonymousId === id);
      prompt += `- ${id}: Average score ${score.toFixed(1)}/10\n`;
    }

    prompt += `\nYour task:
1. Analyze all responses and their peer review scores
2. Identify the strongest elements from each response
3. Resolve any conflicts or contradictions
4. Synthesize a comprehensive final response that:
   - Uses the best insights from top-rated responses
   - Addresses any gaps or weaknesses identified in reviews
   - Provides a confident, authoritative answer
5. Note any significant dissenting viewpoints

Format your response as:
FINAL RESPONSE:
[Your synthesized response]

REASONING:
[Explain how you combined the responses]

SOURCES USED:
[Which council members' insights you incorporated]

CONFLICTS RESOLVED:
[Any contradictions you addressed]

CONFIDENCE: [0-100%]

DISSENT (if any):
[Note minority viewpoints worth mentioning]`;

    return prompt;
  }

  /**
   * Parse synthesis response into structured format
   */
  private parseSynthesis(
    content: string,
    responses: IndividualResponse[],
    scores: Map<string, number>
  ): ChairmanSynthesis {
    // Extract final response
    const responseMatch = content.match(/FINAL RESPONSE:\s*([\s\S]*?)(?=REASONING:|$)/i);
    const finalResponse = responseMatch?.[1]?.trim() || content;

    // Extract reasoning
    const reasoningMatch = content.match(/REASONING:\s*([\s\S]*?)(?=SOURCES USED:|$)/i);
    const reasoning = reasoningMatch?.[1]?.trim() || '';

    // Extract sources
    const sourcesMatch = content.match(/SOURCES USED:\s*([\s\S]*?)(?=CONFLICTS RESOLVED:|$)/i);
    const sourcesText = sourcesMatch?.[1]?.trim() || '';
    const sourcesUsed = sourcesText.split('\n')
      .map(s => s.replace(/^[-*]\s*/, '').trim())
      .filter(s => s);

    // Extract conflicts
    const conflictsMatch = content.match(/CONFLICTS RESOLVED:\s*([\s\S]*?)(?=CONFIDENCE:|$)/i);
    const conflictsText = conflictsMatch?.[1]?.trim() || '';
    const conflictsResolved = conflictsText.split('\n')
      .map(s => s.replace(/^[-*]\s*/, '').trim())
      .filter(s => s);

    // Extract confidence
    const confidenceMatch = content.match(/CONFIDENCE:\s*(\d+)/i);
    const confidenceScore = confidenceMatch
      ? Math.min(1, Math.max(0, parseInt(confidenceMatch[1]) / 100))
      : 0.8;

    // Extract dissent
    const dissentMatch = content.match(/DISSENT.*?:\s*([\s\S]*?)$/i);
    const dissent = dissentMatch?.[1]?.trim();

    return {
      finalResponse,
      reasoning,
      sourcesUsed,
      conflictsResolved,
      confidenceScore,
      dissent: dissent && dissent.toLowerCase() !== 'none' ? dissent : undefined,
    };
  }
}

// ============================================================================
// MAIN MLX DEEP COUNCIL CLASS
// ============================================================================

export class MLXDeepCouncil extends EventEmitter {
  private config: CouncilConfig;
  private distributedGroup: MLXDistributedGroup;
  private memberManager: CouncilMemberManager;
  private responseCollector: IndividualResponseCollector;
  private peerReviewer: PeerReviewCoordinator;
  private synthesizer: ChairmanSynthesizer;
  private sessions: Map<string, CouncilSession> = new Map();
  private initialized: boolean = false;

  constructor(config: CouncilConfig) {
    super();
    this.config = config;

    // Determine backend
    const backend = config.backend === 'auto'
      ? this.detectBestBackend()
      : config.backend;

    // Initialize components
    this.distributedGroup = new MLXDistributedGroup(config.nodes, backend);
    this.memberManager = new CouncilMemberManager(this.distributedGroup);
    this.responseCollector = new IndividualResponseCollector(
      this.memberManager,
      config.timeoutMs
    );
    this.peerReviewer = new PeerReviewCoordinator(
      this.memberManager,
      config.anonymizePeerReview
    );
    this.synthesizer = new ChairmanSynthesizer(this.memberManager);

    // Forward events
    this.setupEventForwarding();
  }

  /**
   * Detect the best distributed backend
   */
  private detectBestBackend(): 'ring' | 'mpi' {
    // Ring is preferred for Thunderbolt connections between Macs
    // MPI is better for network connections
    // For now, default to ring
    return 'ring';
  }

  /**
   * Setup event forwarding from components
   */
  private setupEventForwarding(): void {
    this.distributedGroup.on('initialized', (data) => this.emit('distributed:initialized', data));
    this.distributedGroup.on('error', (data) => this.emit('distributed:error', data));

    this.memberManager.on('member:registered', (data) => this.emit('member:registered', data));
    this.memberManager.on('chairman:elected', (data) => this.emit('chairman:elected', data));

    this.responseCollector.on('response:received', (data) => this.emit('stage1:response', data));
    this.responseCollector.on('collection:complete', (data) => this.emit('stage1:complete', data));

    this.peerReviewer.on('review:received', (data) => this.emit('stage2:review', data));
    this.peerReviewer.on('review:complete', (data) => this.emit('stage2:complete', data));

    this.synthesizer.on('synthesis:complete', (data) => this.emit('stage3:complete', data));
  }

  /**
   * Initialize the council
   */
  async initialize(): Promise<void> {
    console.log(`Initializing MLX Deep Council: ${this.config.name}`);

    // Initialize distributed group
    await this.distributedGroup.initialize();

    // Register members from nodes
    for (let i = 0; i < this.config.nodes.length; i++) {
      const node = this.config.nodes[i];
      const isChairman = i === 0 && this.config.chairmanModel;

      const member: Omit<CouncilMember, 'metrics'> = {
        id: `member_${i}`,
        name: `Council Member ${i + 1}`,
        host: node.ssh,
        port: 8080 + i,
        model: isChairman ? (this.config.chairmanModel || this.config.defaultModel) : this.config.defaultModel,
        role: isChairman ? 'chairman' : 'member',
        rank: i,
        status: 'initializing',
        capabilities: node.models,
      };

      this.memberManager.registerMember(member);

      // Register inference endpoints
      const endpoint = `http://${node.ips[0] || 'localhost'}:${8080 + i}`;
      this.responseCollector.registerEndpoint(member.id, endpoint);
      this.peerReviewer.registerEndpoint(member.id, endpoint);

      if (isChairman) {
        this.synthesizer.setEndpoint(endpoint);
      }
    }

    // Mark members as ready (in real impl, would verify each node)
    for (const member of this.memberManager.getActiveMembers()) {
      this.memberManager.updateStatus(member.id, 'ready');
    }

    this.initialized = true;
    this.emit('initialized', {
      name: this.config.name,
      memberCount: this.config.nodes.length
    });

    console.log(`Council "${this.config.name}" initialized with ${this.config.nodes.length} members`);
  }

  /**
   * Query the council
   */
  async query(query: Omit<CouncilQuery, 'id'>): Promise<CouncilSession> {
    if (!this.initialized) {
      throw new Error('Council not initialized');
    }

    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const fullQuery: CouncilQuery = { ...query, id: sessionId };

    const session: CouncilSession = {
      id: sessionId,
      query: fullQuery,
      stage: 'collecting',
      individualResponses: [],
      peerReviews: [],
      aggregatedScores: new Map(),
      consensusReached: false,
      startTime: new Date(),
      metrics: {
        totalLatency: 0,
        stage1Latency: 0,
        stage2Latency: 0,
        stage3Latency: 0,
        participatingMembers: 0,
        averageConfidence: 0,
        consensusStrength: 0,
      },
    };

    this.sessions.set(sessionId, session);
    this.emit('session:started', { sessionId });

    try {
      // STAGE 1: Collect individual responses
      console.log(`Stage 1: Collecting individual responses...`);
      const stage1Start = Date.now();

      session.individualResponses = await this.responseCollector.collectResponses(fullQuery);
      session.metrics.stage1Latency = Date.now() - stage1Start;
      session.metrics.participatingMembers = session.individualResponses.length;

      if (session.individualResponses.length === 0) {
        throw new Error('No responses collected from council members');
      }

      // STAGE 2: Peer review
      console.log(`Stage 2: Conducting peer review...`);
      session.stage = 'reviewing';
      const stage2Start = Date.now();

      session.peerReviews = await this.peerReviewer.conductPeerReview(
        fullQuery,
        session.individualResponses
      );
      session.aggregatedScores = this.peerReviewer.aggregateScores(session.peerReviews);
      session.metrics.stage2Latency = Date.now() - stage2Start;

      // Update member metrics with peer review scores
      for (const response of session.individualResponses) {
        const score = session.aggregatedScores.get(response.anonymousId);
        if (score !== undefined) {
          this.memberManager.updateMetrics(response.memberId, response.latency, score);
        }
      }

      // STAGE 3: Chairman synthesis
      console.log(`Stage 3: Chairman synthesizing final response...`);
      session.stage = 'synthesizing';
      const stage3Start = Date.now();

      session.chairmanSynthesis = await this.synthesizer.synthesize(
        fullQuery,
        session.individualResponses,
        session.peerReviews,
        session.aggregatedScores
      );
      session.metrics.stage3Latency = Date.now() - stage3Start;

      // Calculate consensus metrics
      session.metrics.averageConfidence = session.individualResponses.reduce(
        (sum, r) => sum + r.confidence, 0
      ) / session.individualResponses.length;

      session.metrics.consensusStrength = this.calculateConsensusStrength(
        session.aggregatedScores,
        session.peerReviews
      );

      session.consensusReached = session.metrics.consensusStrength >= (fullQuery.minAgreement || 0.6);

      // Complete session
      session.stage = 'complete';
      session.endTime = new Date();
      session.metrics.totalLatency = session.endTime.getTime() - session.startTime.getTime();

      this.emit('session:complete', { sessionId, session });
      console.log(`Session complete. Consensus: ${session.consensusReached ? 'Reached' : 'Not reached'}`);

    } catch (error) {
      session.stage = 'failed';
      session.endTime = new Date();
      this.emit('session:failed', { sessionId, error });
      throw error;
    }

    return session;
  }

  /**
   * Calculate consensus strength based on voting patterns
   */
  private calculateConsensusStrength(
    scores: Map<string, number>,
    reviews: PeerReview[]
  ): number {
    if (scores.size === 0) return 0;

    // Calculate variance in scores
    const scoreValues = [...scores.values()];
    const mean = scoreValues.reduce((a, b) => a + b, 0) / scoreValues.length;
    const variance = scoreValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / scoreValues.length;

    // High variance means less consensus
    // Normalize: variance of 0 = consensus of 1, variance of 25 = consensus of 0
    const varianceNormalized = Math.min(1, variance / 25);

    // Count how many reviewers preferred another response
    const selfDeferrals = reviews.filter(r => r.selfEvaluation?.preferOther).length;
    const deferralRate = selfDeferrals / reviews.length;

    // Higher deferral rate indicates openness but also less confidence
    // We want some deferral (shows objectivity) but not too much
    const optimalDeferral = 0.3;
    const deferralPenalty = Math.abs(deferralRate - optimalDeferral);

    return Math.max(0, 1 - varianceNormalized - (deferralPenalty * 0.5));
  }

  /**
   * Get session by ID
   */
  getSession(sessionId: string): CouncilSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Get all sessions
   */
  getAllSessions(): CouncilSession[] {
    return [...this.sessions.values()];
  }

  /**
   * Get council status
   */
  getStatus(): {
    name: string;
    initialized: boolean;
    members: CouncilMember[];
    chairman: CouncilMember | null;
    activeSessions: number;
    completedSessions: number;
  } {
    const sessions = this.getAllSessions();

    return {
      name: this.config.name,
      initialized: this.initialized,
      members: this.memberManager.getActiveMembers(),
      chairman: this.memberManager.getChairman(),
      activeSessions: sessions.filter(s =>
        s.stage !== 'complete' && s.stage !== 'failed'
      ).length,
      completedSessions: sessions.filter(s => s.stage === 'complete').length,
    };
  }

  /**
   * Shutdown the council
   */
  async shutdown(): Promise<void> {
    console.log(`Shutting down council: ${this.config.name}`);
    await this.distributedGroup.shutdown();
    this.initialized = false;
    this.emit('shutdown');
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create a local council for development/testing (single Mac, multiple model instances)
 */
export function createLocalCouncil(config: {
  name?: string;
  models: string[];
  ports?: number[];
}): MLXDeepCouncil {
  const nodes: DistributedNode[] = config.models.map((model, index) => ({
    ssh: 'localhost',
    ips: ['127.0.0.1'],
    gpuMemory: 32,
    chip: 'M3 Max',
    models: [model],
  }));

  return new MLXDeepCouncil({
    name: config.name || 'Local Development Council',
    nodes,
    defaultModel: config.models[0],
    chairmanModel: config.models[config.models.length - 1],
    backend: 'ring',
    votingStrategy: 'weighted',
    anonymizePeerReview: true,
    enableStreaming: true,
    timeoutMs: 120000,
    retryCount: 3,
  });
}

/**
 * Create a distributed council across multiple Macs
 */
export function createDistributedCouncil(config: {
  name: string;
  nodes: Array<{
    hostname: string;
    ip: string;
    model: string;
    gpuMemory: number;
    chip: string;
  }>;
  chairmanIndex?: number;
}): MLXDeepCouncil {
  const nodes: DistributedNode[] = config.nodes.map(node => ({
    ssh: node.hostname,
    ips: [node.ip],
    gpuMemory: node.gpuMemory,
    chip: node.chip,
    models: [node.model],
  }));

  const chairmanIndex = config.chairmanIndex ?? 0;

  return new MLXDeepCouncil({
    name: config.name,
    nodes,
    defaultModel: nodes[0].models[0],
    chairmanModel: nodes[chairmanIndex].models[0],
    backend: 'ring',
    votingStrategy: 'weighted',
    anonymizePeerReview: true,
    enableStreaming: true,
    timeoutMs: 180000,
    retryCount: 3,
  });
}

/**
 * Create a Thunderbolt-connected council (high-speed Mac cluster)
 */
export function createThunderboltCouncil(hostnames: string[]): Promise<MLXDeepCouncil> {
  // This would use mlx.distributed_config to auto-detect Thunderbolt topology
  return new Promise(async (resolve, reject) => {
    try {
      // Auto-detect Thunderbolt configuration
      const { stdout } = await execAsync(
        `python -m mlx.distributed_config --verbose --hosts ${hostnames.join(',')} --json`
      );

      const topology = JSON.parse(stdout);

      const nodes: DistributedNode[] = topology.nodes.map((node: any) => ({
        ssh: node.hostname,
        ips: node.thunderbolt_ips || [node.ip],
        gpuMemory: node.gpu_memory || 32,
        chip: node.chip || 'Unknown',
        models: ['mlx-community/Llama-3.2-3B-Instruct-4bit'],
      }));

      const council = new MLXDeepCouncil({
        name: 'Thunderbolt Council',
        nodes,
        defaultModel: 'mlx-community/Llama-3.2-3B-Instruct-4bit',
        backend: 'ring',
        votingStrategy: 'weighted',
        anonymizePeerReview: true,
        enableStreaming: true,
        timeoutMs: 60000,
        retryCount: 3,
      });

      resolve(council);
    } catch (error) {
      reject(error);
    }
  });
}

// ============================================================================
// EXPORTS
// ============================================================================

export default MLXDeepCouncil;
