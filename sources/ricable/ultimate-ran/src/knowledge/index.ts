/**
 * Knowledge Graph Module - Unified Exports
 *
 * @module knowledge
 * @version 7.0.0-alpha.1
 */

// Exports from agentic-kg.js
export {
  // Main classes
  ThreeGPPKnowledgeAgent,
  KnowledgeEnhancedCouncil,

  // Types
  type ThreeGPPSpec,
  type SpecSection,
  type Parameter,
  type InformationElement,
  type IEField,
  type SPARCResult,
  type PRDDocument,
  type Requirement,
  type KGNode,
  type KGEdge,
  type KnowledgeGraph,
  type KnowledgeGraphAgent,
  type EnhancedProposal,

  // MCP Tools
  kgTools
} from './agentic-kg.js';

// Exports from kg-query.ts (NEW - Natural Language Query Interface)
export {
  KGQueryInterface,
  KGRuvLLMBridge,
  type GraphMLNode,
  type GraphMLEdge,
  type QueryResult,
  type CypherResult,
  type SPARQLResult,
  type GraphPath,
  type TraversalPattern,
  type ParsedQuery,
  createSampleKnowledgeGraph
} from './kg-query.js';

// Exports from graphml-parser.ts (3GPP Knowledge Graph GraphML Parser)
export {
  ThreeGPPKnowledgeGraph,
  GraphMLXMLParser,
  type NodeType,
  type EdgeType,
  type KnowledgeGraph as GraphMLKnowledgeGraph,
  type AdjacencyList,
  detect3GPPSeries,
  extractRelease,
  createKnowledgeGraph as createGraphMLKnowledgeGraph,
  loadKnowledgeGraph,
  loadKnowledgeGraphFromURL,
  extractASN1Definition,
  extractParameterRange,
  extractProcedureStates
} from './graphml-parser.js';
