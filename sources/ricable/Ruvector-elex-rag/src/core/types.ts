/**
 * RuVector Core Types - Ericsson RAN Cognitive Automation Platform
 *
 * Type definitions for the self-learning RAG system handling:
 * - ELEX HTML documentation
 * - 3GPP MOM XML specifications
 * - Network topology graphs
 * - GNN-based optimization
 */

import { z } from 'zod';

// ============================================================================
// 3GPP Parameter Types (per TS 36.213 / TS 38.213)
// ============================================================================

export const AlphaValues = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] as const;
export type AlphaValue = typeof AlphaValues[number];

export interface PowerControlParams {
  /** Cell-specific P0 nominal PUSCH (-126 to 24 dBm) */
  pZeroNominalPusch: number;
  /** Fractional path loss compensation factor */
  alpha: AlphaValue;
  /** Cell-specific P0 nominal PUCCH */
  pZeroNominalPucch: number;
  /** Maximum configured UE output power (typically 23 dBm) */
  pCmax: number;
}

export interface CellConfiguration {
  /** E-UTRAN Cell Global Identifier */
  ecgi: string;
  /** NR Cell Global Identifier (for 5G) */
  ncgi?: string;
  /** Power control parameters */
  powerControl: PowerControlParams;
  /** Antenna tilt in degrees */
  antennaTilt: number;
  /** Azimuth in degrees */
  azimuth: number;
  /** Antenna height in meters */
  height: number;
  /** Maximum transmit power in dBm */
  maxTxPower: number;
  /** Cell bandwidth in MHz */
  bandwidth: number;
  /** Frequency band */
  band: number;
  /** Technology (LTE/NR) */
  technology: 'LTE' | 'NR' | 'LTE-A';
}

// ============================================================================
// Performance Management Types
// ============================================================================

export interface SINRDistribution {
  /** SINR bins (dB) */
  bins: number[];
  /** Count per bin */
  counts: number[];
  /** Mean SINR */
  mean: number;
  /** Standard deviation */
  stdDev: number;
  /** 5th percentile (cell edge) */
  p5: number;
  /** 50th percentile (median) */
  p50: number;
  /** 95th percentile */
  p95: number;
}

export interface PerformanceMetrics {
  /** Cell identifier */
  cellId: string;
  /** Timestamp of measurement */
  timestamp: Date;
  /** Roll-Out Period number */
  ropNumber: number;
  /** Uplink SINR distribution */
  pmPuschSinr: SINRDistribution;
  /** Total received wideband power */
  pmUlRssi: number;
  /** UL PRB utilization (0-100) */
  pmUlPrbUtilization: number;
  /** Channel Quality Indicator distribution */
  pmRadioUeRepCqi: number[];
  /** RRC connection success rate */
  pmRrcConnEstabSucc: number;
  /** Uplink throughput (Mbps) */
  pmUlThroughput: number;
  /** Number of active users */
  pmActiveUsers: number;
}

// ============================================================================
// Graph Topology Types
// ============================================================================

export interface CellNode {
  /** Unique node identifier */
  id: string;
  /** Cell configuration (CM data) */
  config: CellConfiguration;
  /** Current performance metrics (PM data) */
  metrics?: PerformanceMetrics;
  /** Geographic coordinates */
  coordinates: {
    latitude: number;
    longitude: number;
  };
  /** Node feature vector for GNN */
  features: Float32Array;
}

export interface InterferenceEdge {
  /** Source cell ID */
  source: string;
  /** Target cell ID */
  target: string;
  /** Edge weight (interference coupling strength) */
  weight: number;
  /** Physical distance in meters */
  distance: number;
  /** Handover count (indicator of coupling) */
  handoverCount: number;
  /** ANR (Automatic Neighbor Relation) derived */
  anrDerived: boolean;
}

export interface NetworkGraph {
  /** Graph identifier */
  id: string;
  /** Graph name/description */
  name: string;
  /** Cluster identifier */
  clusterId: string;
  /** All cell nodes */
  nodes: Map<string, CellNode>;
  /** Interference edges */
  edges: InterferenceEdge[];
  /** Adjacency list for efficient traversal */
  adjacencyList: Map<string, string[]>;
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt: Date;
}

// ============================================================================
// Document Types (ELEX & 3GPP)
// ============================================================================

export interface ELEXDocument {
  /** Document ID */
  id: string;
  /** Document title */
  title: string;
  /** Document version */
  version: string;
  /** Source file path */
  sourcePath: string;
  /** Document type/category */
  category: string;
  /** Raw HTML content */
  rawHtml: string;
  /** Extracted text content */
  textContent: string;
  /** Embedded images (base64) */
  images: {
    id: string;
    alt: string;
    data: string;
    mimeType: string;
  }[];
  /** Document sections */
  sections: DocumentSection[];
  /** Extracted metadata */
  metadata: Record<string, string>;
  /** Processing timestamp */
  processedAt: Date;
}

export interface DocumentSection {
  /** Section ID */
  id: string;
  /** Section title */
  title: string;
  /** Section level (1-6) */
  level: number;
  /** Section content */
  content: string;
  /** Parent section ID */
  parentId?: string;
  /** Child section IDs */
  childIds: string[];
  /** Tables in this section */
  tables: ExtractedTable[];
  /** Code blocks in this section */
  codeBlocks: string[];
}

export interface ExtractedTable {
  /** Table ID */
  id: string;
  /** Table caption/title */
  caption?: string;
  /** Column headers */
  headers: string[];
  /** Table rows */
  rows: string[][];
}

export interface MOMAttribute {
  /** Attribute name */
  name: string;
  /** Attribute type */
  type: string;
  /** Default value */
  defaultValue?: string;
  /** Valid range/values */
  range?: {
    min?: number;
    max?: number;
    enum?: string[];
  };
  /** Description */
  description: string;
  /** 3GPP reference */
  reference?: string;
  /** Read-only flag */
  readOnly: boolean;
  /** Mandatory flag */
  mandatory: boolean;
}

export interface MOMClass {
  /** Class name */
  name: string;
  /** Full qualified name */
  fqn: string;
  /** Parent class */
  parent?: string;
  /** Class description */
  description: string;
  /** Attributes */
  attributes: MOMAttribute[];
  /** Child classes */
  children: string[];
  /** 3GPP specification reference */
  specReference?: string;
}

export interface ThreeGPPMOM {
  /** MOM identifier */
  id: string;
  /** MOM name */
  name: string;
  /** Version */
  version: string;
  /** Technology (LTE/NR) */
  technology: 'LTE' | 'NR';
  /** All MOM classes */
  classes: Map<string, MOMClass>;
  /** Class hierarchy tree */
  hierarchy: MOMHierarchyNode;
  /** Source file */
  sourceFile: string;
  /** Processing timestamp */
  processedAt: Date;
}

export interface MOMHierarchyNode {
  /** Class name */
  className: string;
  /** Children in hierarchy */
  children: MOMHierarchyNode[];
}

// ============================================================================
// RAG Types
// ============================================================================

export interface DocumentChunk {
  /** Chunk ID */
  id: string;
  /** Source document ID */
  documentId: string;
  /** Document type */
  documentType: 'elex' | '3gpp' | 'config';
  /** Chunk text content */
  content: string;
  /** Embedding vector */
  embedding?: Float32Array;
  /** Chunk metadata */
  metadata: {
    title?: string;
    section?: string;
    pageNumber?: number;
    chunkIndex: number;
    totalChunks: number;
    sourceFile: string;
    technology?: string;
    parameterName?: string;
    momClass?: string;
  };
  /** Token count */
  tokenCount: number;
}

export interface RAGQuery {
  /** Query text */
  query: string;
  /** Number of results to retrieve */
  topK: number;
  /** Minimum similarity threshold */
  minSimilarity: number;
  /** Filter by document type */
  documentTypes?: ('elex' | '3gpp' | 'config')[];
  /** Filter by technology */
  technologies?: ('LTE' | 'NR')[];
  /** Filter by parameter names */
  parameterNames?: string[];
  /** Include metadata in response */
  includeMetadata: boolean;
}

export interface RAGResult {
  /** Retrieved chunks */
  chunks: DocumentChunk[];
  /** Similarity scores */
  scores: number[];
  /** Generated answer */
  answer?: string;
  /** Confidence score */
  confidence: number;
  /** Sources used */
  sources: string[];
  /** Processing time (ms) */
  processingTime: number;
}

// ============================================================================
// GNN Types
// ============================================================================

export interface GNNConfig {
  /** Input feature dimension */
  inputDim: number;
  /** Hidden layer dimension */
  hiddenDim: number;
  /** Output dimension */
  outputDim: number;
  /** Number of GNN layers (message passing rounds) */
  numLayers: number;
  /** Dropout rate */
  dropout: number;
  /** Learning rate */
  learningRate: number;
  /** Batch size */
  batchSize: number;
  /** Aggregation type */
  aggregation: 'sum' | 'mean' | 'max';
  /** Activation function */
  activation: 'relu' | 'gelu' | 'tanh';
}

export interface GNNPrediction {
  /** Predicted mean SINR improvement */
  sinrImprovement: number;
  /** Predicted spectral efficiency gain */
  spectralEfficiencyGain: number;
  /** Coverage impact score */
  coverageImpact: number;
  /** Uncertainty (epistemic) */
  uncertainty: number;
  /** Confidence interval (95%) */
  confidenceInterval: [number, number];
}

export interface BayesianPrediction extends GNNPrediction {
  /** Monte Carlo samples */
  mcSamples: number;
  /** Sample variance */
  variance: number;
  /** Aleatoric uncertainty */
  aleatoricUncertainty: number;
  /** Epistemic uncertainty */
  epistemicUncertainty: number;
}

// ============================================================================
// Agent Types
// ============================================================================

export type AgentRole = 'optimizer' | 'validator' | 'auditor' | 'coordinator';
export type AgentState = 'idle' | 'exploring' | 'optimizing' | 'validating' | 'waiting';

export interface Agent {
  /** Agent ID */
  id: string;
  /** Agent name */
  name: string;
  /** Agent role */
  role: AgentRole;
  /** Current state */
  state: AgentState;
  /** Assigned cluster */
  clusterId: string;
  /** Agent configuration */
  config: AgentConfig;
  /** Performance history */
  history: AgentAction[];
}

export interface AgentConfig {
  /** Exploration rate (epsilon for epsilon-greedy) */
  explorationRate: number;
  /** Learning rate */
  learningRate: number;
  /** Discount factor (gamma) */
  discountFactor: number;
  /** Maximum actions per cycle */
  maxActionsPerCycle: number;
  /** Risk tolerance (0-1) */
  riskTolerance: number;
  /** Minimum confidence for auto-approval */
  minAutoApprovalConfidence: number;
}

export interface AgentAction {
  /** Action ID */
  id: string;
  /** Agent ID */
  agentId: string;
  /** Action type */
  type: 'parameter_change' | 'simulation' | 'validation' | 'rollback';
  /** Target cells */
  targetCells: string[];
  /** Parameter changes */
  changes: ParameterChange[];
  /** Predicted outcome */
  prediction: BayesianPrediction;
  /** Actual outcome (after execution) */
  actualOutcome?: {
    sinrChange: number;
    spectralEfficiencyChange: number;
    coverageChange: number;
  };
  /** Action status */
  status: 'proposed' | 'approved' | 'executed' | 'rolled_back' | 'rejected';
  /** Timestamp */
  timestamp: Date;
}

export interface ParameterChange {
  /** Cell ID */
  cellId: string;
  /** Parameter name */
  parameter: keyof PowerControlParams;
  /** Old value */
  oldValue: number;
  /** New value */
  newValue: number;
}

// ============================================================================
// Optimization Types
// ============================================================================

export interface OptimizationCandidate {
  /** Candidate ID */
  id: string;
  /** Parameter configuration */
  config: Map<string, PowerControlParams>;
  /** Fitness score */
  fitness: number;
  /** Predicted SINR improvement */
  sinrImprovement: number;
  /** Predicted spectral efficiency */
  spectralEfficiency: number;
  /** Coverage penalty */
  coveragePenalty: number;
  /** Uncertainty penalty */
  uncertaintyPenalty: number;
  /** Generation number */
  generation: number;
}

export interface OptimizationResult {
  /** Best candidate */
  bestCandidate: OptimizationCandidate;
  /** All evaluated candidates */
  allCandidates: OptimizationCandidate[];
  /** Convergence history */
  convergenceHistory: number[];
  /** Total generations */
  totalGenerations: number;
  /** Total evaluations */
  totalEvaluations: number;
  /** Computation time (ms) */
  computationTime: number;
  /** Pareto frontier (for multi-objective) */
  paretoFrontier: OptimizationCandidate[];
}

// ============================================================================
// Validation Schemas
// ============================================================================

export const PowerControlParamsSchema = z.object({
  pZeroNominalPusch: z.number().min(-126).max(24),
  alpha: z.number().refine((v) => AlphaValues.includes(v as AlphaValue)),
  pZeroNominalPucch: z.number().min(-126).max(24),
  pCmax: z.number().min(10).max(30),
});

export const RAGQuerySchema = z.object({
  query: z.string().min(1).max(10000),
  topK: z.number().min(1).max(100).default(10),
  minSimilarity: z.number().min(0).max(1).default(0.5),
  documentTypes: z.array(z.enum(['elex', '3gpp', 'config'])).optional(),
  technologies: z.array(z.enum(['LTE', 'NR'])).optional(),
  parameterNames: z.array(z.string()).optional(),
  includeMetadata: z.boolean().default(true),
});

// ============================================================================
// Event Types
// ============================================================================

export type SystemEvent =
  | { type: 'DOCUMENT_INGESTED'; payload: { documentId: string; documentType: string } }
  | { type: 'CHUNK_EMBEDDED'; payload: { chunkId: string; documentId: string } }
  | { type: 'GRAPH_UPDATED'; payload: { graphId: string; nodesAdded: number; edgesAdded: number } }
  | { type: 'GNN_TRAINED'; payload: { epoch: number; loss: number; accuracy: number } }
  | { type: 'OPTIMIZATION_STARTED'; payload: { clusterId: string; agentId: string } }
  | { type: 'OPTIMIZATION_COMPLETED'; payload: { clusterId: string; result: OptimizationResult } }
  | { type: 'PARAMETER_CHANGED'; payload: { cellId: string; changes: ParameterChange[] } }
  | { type: 'ROLLBACK_TRIGGERED'; payload: { reason: string; affectedCells: string[] } }
  | { type: 'AGENT_STATE_CHANGED'; payload: { agentId: string; oldState: AgentState; newState: AgentState } };
