/**
 * RuvVector + GNN + AgentDB Integration Example
 *
 * Demonstrates complete self-learning RAN optimization pipeline:
 * 1. Cell state -> RuvVector embedding
 * 2. GNN attention for interference prediction
 * 3. AgentDB reflexion memory for transfer learning
 * 4. PydanticAI validation before execution
 * 5. Feedback loop for continuous learning
 *
 * @module ml/integration-example
 * @version 7.0.0-alpha.1
 */

import { RuvectorGNN, RuvLLMClient, type OptimizationEpisode } from './ruvector-gnn.js';
import { GraphAttentionNetwork, type CellNode, type InterferenceEdge } from './attention-gnn.js';
import { AgentDBReflexion } from './agentdb-reflexion.js';
import { cmParametersValidator, recommendationValidator } from './pydantic-validation.js';
import type { PMCounters, CMParameters } from '../learning/self-learner.js';

// ============================================================================
// Complete Self-Learning Pipeline
// ============================================================================

/**
 * Integrated self-learning RAN system
 */
export class SelfLearningRANSystem {
  private ruvectorGNN: RuvectorGNN;
  private gat: GraphAttentionNetwork;
  private agentdb: AgentDBReflexion;
  private ruvllm: RuvLLMClient;

  constructor() {
    // Initialize components
    this.ruvectorGNN = new RuvectorGNN('./ruvector-spatial.db');
    this.gat = new GraphAttentionNetwork({
      numHeads: 8,
      hiddenDim: 64,
      nodeFeatureDim: 128,
      edgeFeatureDim: 32,
      outputDim: 128
    });
    this.agentdb = new AgentDBReflexion({
      dbPath: './titan-ran.db',
      maxMemorySize: 100000,
      embeddingDim: 768
    });
    this.ruvllm = new RuvLLMClient(this.ruvectorGNN);

    console.log('[SelfLearningRAN] Initialized complete pipeline');
  }

  /**
   * Initialize the system
   */
  async initialize(): Promise<void> {
    console.log('[SelfLearningRAN] Initializing components...');

    await this.ruvectorGNN.initialize();
    await this.agentdb.initialize();

    console.log('[SelfLearningRAN] System ready');
  }

  /**
   * Complete optimization workflow for a cell
   */
  async optimizeCell(cellId: string): Promise<{
    recommendation: any;
    propagationImpact: any;
    validation: any;
    executionPlan: any;
  }> {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`[SelfLearningRAN] Starting optimization for cell ${cellId}`);
    console.log('='.repeat(80));

    // Step 1: Get current cell state
    console.log('\n[Step 1] Retrieving cell state...');
    const cellState = await this.getCellState(cellId);
    console.log(`  SINR: ${cellState.pm.pmUlSinrMean?.toFixed(2)} dB`);
    console.log(`  CSSR: ${((cellState.pm.pmCssr || 0) * 100).toFixed(2)}%`);
    console.log(`  Drop Rate: ${((cellState.pm.pmCallDropRate || 0) * 100).toFixed(3)}%`);

    // Step 2: Find similar cells using RuvVector
    console.log('\n[Step 2] Finding similar cells via RuvVector HNSW...');
    const similarCells = await this.ruvectorGNN.findSimilarCells(cellId, 5);
    console.log(`  Found ${similarCells.length} similar cells:`);
    for (const cell of similarCells.slice(0, 3)) {
      console.log(`    - ${cell.cellId} (cluster: ${cell.metadata.cluster})`);
    }

    // Step 3: Query AgentDB for similar successful optimizations
    console.log('\n[Step 3] Querying AgentDB reflexion memory...');
    const cellEmbedding = await this.ruvectorGNN['cellEmbeddings'].get(cellId);
    if (!cellEmbedding) {
      throw new Error(`Cell ${cellId} not found`);
    }

    const transferLearning = await this.agentdb.queryForTransferLearning(
      cellState.pm,
      cellEmbedding.vector,
      5,
      { outcome: 'success', minReward: 0.3 }
    );
    console.log(`  Found ${transferLearning.length} similar successful optimizations`);
    if (transferLearning.length > 0) {
      const topMatch = transferLearning[0];
      console.log(`    Top match: ${topMatch.entry.id}`);
      console.log(`    Similarity: ${topMatch.similarity.toFixed(3)}`);
      console.log(`    Past reward: ${topMatch.entry.metadata.reward.toFixed(3)}`);
    }

    // Step 4: Generate recommendation using RuvLLM
    console.log('\n[Step 4] Generating recommendation with RuvLLM...');
    const recommendation = await this.ruvllm.recommendOptimization(cellId);
    console.log(`  Priority: ${recommendation.priority}`);
    console.log(`  Confidence: ${recommendation.confidence.toFixed(2)}`);
    console.log(`  Recommended action: ${JSON.stringify(recommendation.recommendedAction)}`);
    console.log(`  Expected SINR gain: ${recommendation.expectedGain.sinr?.toFixed(2)} dB`);
    console.log(`  Reasoning: ${recommendation.reasoning.substring(0, 100)}...`);

    // Step 5: Validate recommendation with PydanticAI-style validator
    console.log('\n[Step 5] Validating recommendation (3GPP + Physics)...');
    const validationResult = recommendationValidator.validate(recommendation, { coerce: true });

    if (!validationResult.valid) {
      console.log('  ❌ Validation FAILED:');
      for (const error of validationResult.errors) {
        console.log(`    - ${error.field}: ${error.message}`);
        console.log(`      Constraint: ${error.constraint}`);
      }
      throw new Error('Recommendation validation failed');
    }

    console.log('  ✓ Validation PASSED');
    if (validationResult.warnings && validationResult.warnings.length > 0) {
      console.log('  Warnings:');
      for (const warning of validationResult.warnings) {
        console.log(`    - ${warning}`);
      }
    }

    // Validate CM parameters specifically
    const cmValidation = cmParametersValidator.validate(
      recommendation.recommendedAction,
      { coerce: true }
    );

    if (!cmValidation.valid) {
      console.log('  ❌ CM Parameters validation FAILED:');
      for (const error of cmValidation.errors) {
        console.log(`    - ${error.field}: ${error.message}`);
      }
      throw new Error('CM parameters validation failed');
    }

    console.log('  ✓ CM Parameters validation PASSED');

    // Step 6: Predict network-wide impact using GAT
    console.log('\n[Step 6] Predicting propagation impact via Graph Attention Network...');
    const propagationResult = await this.gat.predictPropagation(
      cellId,
      recommendation.recommendedAction,
      2  // Depth 2 = direct neighbors + neighbors-of-neighbors
    );

    console.log(`  Affected cells: ${propagationResult.affectedCells.size}`);
    console.log(`  Total impact score: ${propagationResult.totalImpactScore.toFixed(3)}`);
    console.log(`  Propagation time: ${propagationResult.propagationTime.toFixed(2)}ms`);

    // Show top affected neighbors
    const topAffected = Array.from(propagationResult.affectedCells.entries())
      .sort((a, b) => b[1].impactScore - a[1].impactScore)
      .slice(0, 3);

    console.log('  Top affected neighbors:');
    for (const [neighborId, impact] of topAffected) {
      console.log(`    - ${neighborId}: impact=${impact.impactScore.toFixed(3)}, depth=${impact.propagationDepth}`);
    }

    // Step 7: Generate execution plan
    console.log('\n[Step 7] Generating execution plan...');
    const executionPlan = {
      cellId,
      action: cmValidation.data!,
      expectedOutcome: {
        sinrGain: recommendation.expectedGain.sinr,
        cssrGain: recommendation.expectedGain.cssr,
        dropRateChange: recommendation.expectedGain.dropRate
      },
      affectedCells: Array.from(propagationResult.affectedCells.keys()),
      risks: recommendation.risks,
      requiresApproval: recommendation.confidence < 0.7 || propagationResult.totalImpactScore > 2.0,
      estimatedExecutionTime: 30  // seconds
    };

    console.log(`  Requires human approval: ${executionPlan.requiresApproval ? 'YES' : 'NO'}`);
    console.log(`  Estimated execution time: ${executionPlan.estimatedExecutionTime}s`);

    console.log('\n' + '='.repeat(80));
    console.log('[SelfLearningRAN] Optimization plan complete');
    console.log('='.repeat(80) + '\n');

    return {
      recommendation,
      propagationImpact: propagationResult,
      validation: validationResult,
      executionPlan
    };
  }

  /**
   * Execute optimization and record results
   */
  async executeAndLearn(
    cellId: string,
    action: CMParameters,
    expectedOutcome: any
  ): Promise<OptimizationEpisode> {
    console.log(`\n[SelfLearningRAN] Executing optimization on ${cellId}...`);

    // Get state before
    const beforeState = await this.getCellState(cellId);

    // Execute CM parameter change (simulated)
    console.log('  Applying CM parameters...');
    await this.applyCMParameters(cellId, action);

    // Wait for PM counters to update
    console.log('  Waiting for PM counters (15min averaging period)...');
    await this.sleep(1000);  // In production: wait 15 minutes

    // Get state after
    const afterState = await this.getCellState(cellId);

    // Calculate actual outcome
    const sinrGain = (afterState.pm.pmUlSinrMean || 0) - (beforeState.pm.pmUlSinrMean || 0);
    const cssrGain = (afterState.pm.pmCssr || 0) - (beforeState.pm.pmCssr || 0);
    const dropRateChange = (afterState.pm.pmCallDropRate || 0) - (beforeState.pm.pmCallDropRate || 0);

    // Calculate reward
    const reward = this.calculateReward(sinrGain, cssrGain, dropRateChange);

    // Determine outcome
    const outcome: 'SUCCESS' | 'FAILURE' | 'NEUTRAL' = reward > 0.3 ? 'SUCCESS' :
      reward < -0.1 ? 'FAILURE' : 'NEUTRAL';

    console.log(`  Outcome: ${outcome}`);
    console.log(`  Reward: ${reward.toFixed(3)}`);
    console.log(`  SINR gain: ${sinrGain.toFixed(2)} dB`);

    // Create episode
    const episode: OptimizationEpisode = {
      id: `episode_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      cellId,
      timestamp: new Date(),
      pmBefore: beforeState.pm,
      cmBefore: beforeState.cm,
      fmAlarmsBefore: [],
      action,
      actionType: this.determineActionType(action),
      pmAfter: afterState.pm,
      cmAfter: { ...beforeState.cm, ...action },
      fmAlarmsAfter: [],
      reward,
      sinrGain,
      cssrGain,
      dropRateChange,
      outcome,
      stateEmbedding: beforeState.embedding,
      actionEmbedding: this.createActionEmbedding(action)
    };

    // Store in RuvVector GNN
    console.log('  Indexing in RuvVector...');
    await this.ruvectorGNN.indexOptimization(episode);

    // Store in AgentDB reflexion memory
    console.log('  Storing in AgentDB reflexion memory...');
    await this.agentdb.storeOptimization(episode);

    console.log(`[SelfLearningRAN] Learning complete. Episode ${episode.id} stored.\n`);

    return episode;
  }

  /**
   * Natural language query interface
   */
  async query(question: string): Promise<any> {
    console.log(`\n[RuvLLM Query] ${question}`);
    const insight = await this.ruvllm.queryRAN(question);

    console.log(`\n[Answer] ${insight.answer}`);
    console.log(`Confidence: ${insight.confidence.toFixed(2)}`);
    console.log(`Reasoning: ${insight.reasoning}\n`);

    return insight;
  }

  /**
   * Get system statistics
   */
  async getStatistics(): Promise<any> {
    const gnnStats = this.ruvectorGNN.getStats();
    const gatStats = this.gat.getStats();
    const reflexionStats = await this.agentdb.getReflexionStats();

    return {
      ruvectorGNN: gnnStats,
      graphAttentionNetwork: gatStats,
      reflexionMemory: reflexionStats
    };
  }

  // ============================================================================
  // Helper Methods (Simulated)
  // ============================================================================

  private async getCellState(cellId: string): Promise<{
    pm: PMCounters;
    cm: CMParameters;
    embedding: Float32Array;
  }> {
    // In production: fetch from ENM/OSS
    const pm: PMCounters = {
      pmUlSinrMean: 5.0 + Math.random() * 10,
      pmDlSinrMean: 8.0 + Math.random() * 10,
      pmUlBler: 0.01 + Math.random() * 0.05,
      pmDlBler: 0.01 + Math.random() * 0.05,
      pmCssr: 0.95 + Math.random() * 0.04,
      pmErabSuccessRate: 0.96 + Math.random() * 0.03,
      pmCallDropRate: 0.01 + Math.random() * 0.02,
      pmHoSuccessRate: 0.92 + Math.random() * 0.05,
      pmPuschPrbUsage: 30 + Math.random() * 40,
      pmPdschPrbUsage: 40 + Math.random() * 40
    };

    const cm: CMParameters = {
      p0NominalPUSCH: -103,
      alpha: 0.8,
      electricalTilt: 3.0,
      mechanicalTilt: 0,
      txPower: 40.0,
      crsGain: 0
    };

    // Create embedding (simplified)
    const embedding = new Float32Array(768);
    for (let i = 0; i < 768; i++) {
      embedding[i] = Math.random() - 0.5;
    }

    // Add to RuvVector if not exists
    const cellEmbedding = this.ruvectorGNN['cellEmbeddings'].get(cellId);
    if (!cellEmbedding) {
      await this.ruvectorGNN.addCell(cellId, pm, {
        cluster: 'cluster_01',
        site: 'site_123',
        sector: Math.floor(Math.random() * 3),
        lastOptimization: new Date(),
        neighborCells: [`neighbor_${cellId}_1`, `neighbor_${cellId}_2`],
        performanceClass: pm.pmUlSinrMean! > 10 ? 'good' : 'fair'
      });
    }

    return { pm, cm, embedding };
  }

  private async applyCMParameters(cellId: string, params: CMParameters): Promise<void> {
    // In production: send to ENM via NETCONF/REST API
    console.log(`  [ENM] Applying to ${cellId}:`, params);
  }

  private calculateReward(sinrGain: number, cssrGain: number, dropRateChange: number): number {
    return sinrGain * 0.4 + cssrGain * 100 * 0.3 - dropRateChange * 100 * 0.3;
  }

  private determineActionType(action: CMParameters): 'power' | 'tilt' | 'alpha' | 'beamweight' | 'combo' {
    if (action.p0NominalPUSCH !== undefined) return 'power';
    if (action.electricalTilt !== undefined) return 'tilt';
    if (action.alpha !== undefined) return 'alpha';
    if (action.beamWeights !== undefined) return 'beamweight';
    return 'combo';
  }

  private createActionEmbedding(action: CMParameters): Float32Array {
    const embedding = new Float32Array(768);
    // Simplified action encoding
    if (action.p0NominalPUSCH !== undefined) embedding[0] = action.p0NominalPUSCH / 100;
    if (action.electricalTilt !== undefined) embedding[1] = action.electricalTilt / 15;
    if (action.alpha !== undefined) embedding[2] = action.alpha;
    return embedding;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Example Usage
// ============================================================================

/**
 * Run complete self-learning example
 */
export async function runExample(): Promise<void> {
  console.log('\n' + '█'.repeat(80));
  console.log('█  TITAN RAN Self-Learning System - Complete Integration Example');
  console.log('█'.repeat(80) + '\n');

  const system = new SelfLearningRANSystem();
  await system.initialize();

  // Example 1: Optimize a cell
  console.log('\n' + '▓'.repeat(80));
  console.log('▓  Example 1: Optimize Cell ABC123');
  console.log('▓'.repeat(80));

  const result = await system.optimizeCell('ABC123');

  if (!result.executionPlan.requiresApproval) {
    console.log('\n[EXECUTION] Approval not required. Executing automatically...');

    const episode = await system.executeAndLearn(
      'ABC123',
      result.recommendation.recommendedAction,
      result.recommendation.expectedGain
    );

    console.log(`✓ Optimization complete. Reward: ${episode.reward.toFixed(3)}`);
  } else {
    console.log('\n[EXECUTION] Approval required. Awaiting operator confirmation...');
  }

  // Example 2: Natural language query
  console.log('\n' + '▓'.repeat(80));
  console.log('▓  Example 2: Natural Language Queries');
  console.log('▓'.repeat(80));

  await system.query('What cells have similar SINR patterns to ABC123?');
  await system.query('Explain why the last optimization was successful');

  // Example 3: Show statistics
  console.log('\n' + '▓'.repeat(80));
  console.log('▓  Example 3: System Statistics');
  console.log('▓'.repeat(80) + '\n');

  const stats = await system.getStatistics();

  console.log('RuvVector GNN:');
  console.log(`  Cells indexed: ${stats.ruvectorGNN.cellCount}`);
  console.log(`  Episodes indexed: ${stats.ruvectorGNN.episodeCount}`);
  console.log(`  Avg neighbors: ${stats.ruvectorGNN.avgCellNeighbors.toFixed(1)}`);

  console.log('\nGraph Attention Network:');
  console.log(`  Nodes: ${stats.graphAttentionNetwork.nodeCount}`);
  console.log(`  Edges: ${stats.graphAttentionNetwork.edgeCount}`);
  console.log(`  Avg degree: ${stats.graphAttentionNetwork.avgDegree.toFixed(1)}`);

  console.log('\nReflexion Memory:');
  console.log(`  Total episodes: ${stats.reflexionMemory.totalEpisodes}`);
  console.log(`  Success rate: ${(stats.reflexionMemory.successRate * 100).toFixed(1)}%`);
  console.log(`  Avg reward: ${stats.reflexionMemory.avgReward.toFixed(3)}`);
  console.log(`  Memory utilization: ${(stats.reflexionMemory.memoryUtilization * 100).toFixed(1)}%`);

  console.log('\n' + '█'.repeat(80));
  console.log('█  Example Complete');
  console.log('█'.repeat(80) + '\n');
}

// Export for use in other modules
// export { SelfLearningRANSystem };

// Run example if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runExample().catch(console.error);
}
