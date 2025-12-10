/**
 * TypeScript type definitions for Quantum Bridge
 *
 * Integration between AgentCoordination (Rust) and QuantumDAG (@qudag/napi-core)
 */

/**
 * Agent operation details
 */
export interface Operation {
  operationType: string;
  agentId: string;
  metadata?: Record<string, any>;
}

/**
 * Conflict details with quantum verification
 */
export interface QuantumConflict {
  operationA: string;
  operationB: string;
  agents: string[];
  conflictingFiles: string[];
  severity: number;
  quantumVerified: boolean;
  dagDistance: number;
  isAncestor: boolean;
  description: string;
  resolutionStrategy: string;
}

/**
 * DAG statistics
 */
export interface DagStats {
  initialized: boolean;
  vertices: number;
  tips: number;
  quantumEnabled: boolean;
  totalEdges?: number;
  maxDepth?: number;
}

/**
 * Quantum Bridge for AgentCoordination
 */
export class QuantumBridge {
  /**
   * Create a new QuantumBridge instance
   * @param agentCoordination - Rust AgentCoordination instance
   */
  constructor(agentCoordination: any);

  /**
   * Initialize QuantumDAG with default configuration
   */
  initialize(): Promise<void>;

  /**
   * Register an agent operation in the QuantumDAG
   * @param operationId - Operation identifier
   * @param operation - Operation details
   * @param affectedFiles - Files affected by operation
   * @returns Vertex ID in the DAG
   */
  registerOperation(
    operationId: string,
    operation: Operation,
    affectedFiles: string[]
  ): Promise<string>;

  /**
   * Check for conflicts using QuantumDAG structure
   * @param operationId - Proposed operation ID
   * @param operationType - Operation type
   * @param affectedFiles - Files that will be affected
   * @returns Array of conflicts with quantum verification
   */
  checkConflicts(
    operationId: string,
    operationType: string,
    affectedFiles: string[]
  ): Promise<QuantumConflict[]>;

  /**
   * Get DAG statistics
   * @returns DAG statistics
   */
  getStats(): Promise<DagStats>;

  /**
   * Verify quantum-resistant proof for a vertex
   * @param vertexId - Vertex ID to verify
   * @returns True if proof is valid
   */
  verifyQuantumProof(vertexId: string): Promise<boolean>;

  /**
   * Get coordination tips (DAG tips)
   * @returns Array of tip vertex IDs
   */
  getCoordinationTips(): Promise<string[]>;

  /**
   * Clean up resources
   */
  cleanup(): Promise<void>;
}

/**
 * Create a QuantumBridge instance for an AgentCoordination
 * @param agentCoordination - Rust AgentCoordination instance
 * @returns Bridge instance
 */
export function createQuantumBridge(agentCoordination: any): QuantumBridge;
