/**
 * Quantum Bridge - JavaScript integration between AgentCoordination and @qudag/napi-core
 *
 * This module provides the bridge between Rust AgentCoordination and QuantumDAG.
 * It handles:
 * - DAG vertex creation for agent operations
 * - Conflict detection using DAG structure
 * - Quantum-resistant cryptographic verification
 * - Tip tracking and coordination
 *
 * @module quantum_bridge
 */

const { QuantumDAG } = require('@qudag/napi-core');

/**
 * Quantum Bridge for AgentCoordination
 */
class QuantumBridge {
  /**
   * Create a new QuantumBridge instance
   * @param {Object} agentCoordination - Rust AgentCoordination instance
   */
  constructor(agentCoordination) {
    this.coordination = agentCoordination;
    this.dag = null;
    this.vertexMap = new Map(); // operation_id -> vertex_id
    this.initialized = false;
  }

  /**
   * Initialize QuantumDAG with default configuration
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.initialized) {
      return;
    }

    try {
      // Initialize QuantumDAG with quantum-resistant settings
      this.dag = new QuantumDAG({
        quantumResistant: true,
        hashAlgorithm: 'sha3-256', // Quantum-resistant hash
        conflictDetection: true,
        autoMerge: false, // Manual conflict resolution for agent coordination
      });

      // Enable quantum features in Rust coordination
      this.coordination.enableQuantum();

      this.initialized = true;
      console.log('[QuantumBridge] Initialized with quantum-resistant DAG');
    } catch (error) {
      console.error('[QuantumBridge] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Register an agent operation in the QuantumDAG
   * @param {string} operationId - Operation identifier
   * @param {Object} operation - Operation details
   * @param {string[]} affectedFiles - Files affected by operation
   * @returns {Promise<string>} Vertex ID in the DAG
   */
  async registerOperation(operationId, operation, affectedFiles) {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      // Get current DAG tips as parents
      const tips = await this.dag.getTips();
      const parents = tips.length > 0 ? tips : [];

      // Create vertex data
      const vertexData = {
        operationId,
        operationType: operation.operationType,
        agentId: operation.agentId,
        affectedFiles,
        timestamp: new Date().toISOString(),
        metadata: operation.metadata || {},
      };

      // Add vertex to DAG
      const vertexId = await this.dag.addVertex({
        data: vertexData,
        parents,
        quantumProof: true, // Enable quantum-resistant proof
      });

      // Store mapping
      this.vertexMap.set(operationId, vertexId);

      // Update Rust coordination with vertex mapping
      await this.coordination.registerDagVertex(operationId, vertexId);

      // Update tips cache
      await this.updateTipsCache();

      console.log(`[QuantumBridge] Registered operation ${operationId} as vertex ${vertexId}`);
      return vertexId;
    } catch (error) {
      console.error(`[QuantumBridge] Failed to register operation ${operationId}:`, error);
      throw error;
    }
  }

  /**
   * Check for conflicts using QuantumDAG structure
   * @param {string} operationId - Proposed operation ID
   * @param {string} operationType - Operation type
   * @param {string[]} affectedFiles - Files that will be affected
   * @returns {Promise<Object[]>} Array of conflicts with quantum verification
   */
  async checkConflicts(operationId, operationType, affectedFiles) {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      const conflicts = [];
      const tips = await this.dag.getTips();

      // Get recent vertices (last 100 operations)
      const recentVertices = await this.dag.getRecentVertices(100);

      for (const vertex of recentVertices) {
        const vertexData = await this.dag.getVertexData(vertex.id);

        // Skip if same operation
        if (vertexData.operationId === operationId) {
          continue;
        }

        // Check file conflicts
        const conflictingFiles = affectedFiles.filter(file =>
          vertexData.affectedFiles.includes(file)
        );

        if (conflictingFiles.length > 0) {
          // Use DAG structure to determine conflict severity
          const isAncestor = await this.dag.isAncestor(vertex.id, tips[0]);
          const distance = await this.dag.getDistance(vertex.id, tips[0]);

          // Analyze conflict with quantum verification
          const conflict = {
            operationA: operationId,
            operationB: vertexData.operationId,
            agents: [vertexData.agentId],
            conflictingFiles,
            severity: this.calculateSeverity(operationType, vertexData.operationType, distance),
            quantumVerified: true,
            dagDistance: distance,
            isAncestor,
            description: this.getConflictDescription(
              operationType,
              vertexData.operationType,
              conflictingFiles,
              distance
            ),
            resolutionStrategy: this.getResolutionStrategy(
              operationType,
              vertexData.operationType,
              isAncestor,
              distance
            ),
          };

          conflicts.push(conflict);
        }
      }

      console.log(`[QuantumBridge] Found ${conflicts.length} conflicts for operation ${operationId}`);
      return conflicts;
    } catch (error) {
      console.error(`[QuantumBridge] Conflict check failed for ${operationId}:`, error);
      throw error;
    }
  }

  /**
   * Calculate conflict severity based on DAG structure
   * @private
   */
  calculateSeverity(opA, opB, distance) {
    // Recent conflicts (small DAG distance) are more severe
    if (distance < 5) {
      return 3; // Severe
    } else if (distance < 20) {
      return 2; // Moderate
    } else {
      return 1; // Minor
    }
  }

  /**
   * Get conflict description
   * @private
   */
  getConflictDescription(opA, opB, files, distance) {
    return `Quantum-verified conflict: ${opA} and ${opB} operations on ${files.join(', ')} (DAG distance: ${distance})`;
  }

  /**
   * Get resolution strategy based on DAG structure
   * @private
   */
  getResolutionStrategy(opA, opB, isAncestor, distance) {
    if (isAncestor) {
      return 'auto_merge'; // Can auto-merge if ancestor
    } else if (distance < 5) {
      return 'manual_resolution'; // Recent conflicts need manual review
    } else {
      return 'sequential_execution'; // Coordinate execution order
    }
  }

  /**
   * Update tips cache in Rust coordination
   * @private
   */
  async updateTipsCache() {
    const tips = await this.dag.getTips();
    await this.coordination.updateDagTips(tips);
  }

  /**
   * Get DAG statistics
   * @returns {Promise<Object>} DAG statistics
   */
  async getStats() {
    if (!this.initialized) {
      return {
        initialized: false,
        vertices: 0,
        tips: 0,
        quantumEnabled: false,
      };
    }

    const stats = await this.dag.getStats();
    const tips = await this.dag.getTips();

    return {
      initialized: true,
      vertices: stats.vertexCount || 0,
      tips: tips.length,
      quantumEnabled: true,
      totalEdges: stats.edgeCount || 0,
      maxDepth: stats.maxDepth || 0,
    };
  }

  /**
   * Verify quantum-resistant proof for a vertex
   * @param {string} vertexId - Vertex ID to verify
   * @returns {Promise<boolean>} True if proof is valid
   */
  async verifyQuantumProof(vertexId) {
    if (!this.initialized) {
      throw new Error('QuantumBridge not initialized');
    }

    return await this.dag.verifyQuantumProof(vertexId);
  }

  /**
   * Get coordination tips (DAG tips)
   * @returns {Promise<string[]>} Array of tip vertex IDs
   */
  async getCoordinationTips() {
    if (!this.initialized) {
      return [];
    }

    return await this.dag.getTips();
  }

  /**
   * Clean up resources
   */
  async cleanup() {
    if (this.dag) {
      await this.dag.cleanup();
      this.dag = null;
    }
    this.vertexMap.clear();
    this.initialized = false;
  }
}

/**
 * Create a QuantumBridge instance for an AgentCoordination
 * @param {Object} agentCoordination - Rust AgentCoordination instance
 * @returns {QuantumBridge} Bridge instance
 */
function createQuantumBridge(agentCoordination) {
  return new QuantumBridge(agentCoordination);
}

module.exports = {
  QuantumBridge,
  createQuantumBridge,
};
