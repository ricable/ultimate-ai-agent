/**
 * RuVector Telecom RAG Server
 *
 * REST API for the Ericsson RAN Cognitive Automation Platform.
 * Provides endpoints for:
 * - RAG queries on ELEX and 3GPP documentation
 * - Network optimization recommendations
 * - Agent swarm management
 * - GNN predictions
 */

import Fastify from 'fastify';
import cors from '@fastify/cors';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import { getConfig } from '../core/config.js';
import { RAGPipeline } from '../rag/rag-pipeline.js';
import { NetworkGraphBuilder } from '../gnn/network-graph.js';
import { GNNInferenceEngine } from '../gnn/gnn-engine.js';
import { SwarmController, OptimizerAgent, ValidatorAgent, AuditorAgent } from '../agents/agent-swarm.js';
import { logger } from '../utils/logger.js';
import type {
  RAGQuery,
  CellConfiguration,
  NetworkGraph,
  PowerControlParams,
} from '../core/types.js';

const config = getConfig();

// Initialize components
const ragPipeline = new RAGPipeline();
const graphBuilder = new NetworkGraphBuilder();
const gnnEngine = new GNNInferenceEngine();
const swarmController = new SwarmController();

// Store for network graphs (would be a database in production)
const graphs = new Map<string, NetworkGraph>();
const swarms = new Map<string, { optimizer: OptimizerAgent; validator: ValidatorAgent; auditor: AuditorAgent }>();

// Create Fastify instance
const fastify = Fastify({
  logger: {
    level: config.logging.level,
    transport: config.logging.format === 'pretty'
      ? { target: 'pino-pretty' }
      : undefined,
  },
});

// Register plugins
async function registerPlugins() {
  await fastify.register(cors, {
    origin: true,
    credentials: true,
  });

  if (config.server.enableSwagger) {
    await fastify.register(swagger, {
      openapi: {
        info: {
          title: 'RuVector Telecom RAG API',
          description: 'Cognitive Automation Platform for Ericsson RAN - Self-Learning RAG System',
          version: '1.0.0',
        },
        tags: [
          { name: 'RAG', description: 'Retrieval-Augmented Generation endpoints' },
          { name: 'Optimization', description: 'Network optimization endpoints' },
          { name: 'Agents', description: 'Agent swarm management' },
          { name: 'Ingestion', description: 'Document ingestion endpoints' },
        ],
      },
    });

    await fastify.register(swaggerUi, {
      routePrefix: '/docs',
    });
  }
}

// Request schemas
const RAGQuerySchema = z.object({
  query: z.string().min(1).max(10000),
  topK: z.number().min(1).max(100).optional().default(10),
  minSimilarity: z.number().min(0).max(1).optional().default(0.5),
  documentTypes: z.array(z.enum(['elex', '3gpp', 'config'])).optional(),
  technologies: z.array(z.enum(['LTE', 'NR'])).optional(),
  parameterNames: z.array(z.string()).optional(),
});

const OptimizationRequestSchema = z.object({
  graphId: z.string(),
  constraints: z.object({
    minEdgeSinr: z.number().optional(),
    maxUncertainty: z.number().optional(),
    targetSpectralEfficiency: z.number().optional(),
  }).optional(),
});

const CellConfigSchema = z.object({
  ecgi: z.string(),
  ncgi: z.string().optional(),
  powerControl: z.object({
    pZeroNominalPusch: z.number(),
    alpha: z.number().refine((v) => [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].includes(v)),
    pZeroNominalPucch: z.number(),
    pCmax: z.number(),
  }),
  antennaTilt: z.number(),
  azimuth: z.number(),
  height: z.number(),
  maxTxPower: z.number(),
  bandwidth: z.number(),
  band: z.number(),
  technology: z.enum(['LTE', 'NR', 'LTE-A']),
});

// Routes

// Health check
fastify.get('/health', async () => {
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    stats: ragPipeline.getStats(),
  };
});

// ============================================================================
// RAG Endpoints
// ============================================================================

// Query endpoint
fastify.post('/api/v1/rag/query', {
  schema: {
    tags: ['RAG'],
    summary: 'Query the RAG system',
    description: 'Submit a natural language query about Ericsson RAN documentation',
    body: {
      type: 'object',
      required: ['query'],
      properties: {
        query: { type: 'string' },
        topK: { type: 'number' },
        minSimilarity: { type: 'number' },
        documentTypes: { type: 'array', items: { type: 'string' } },
        technologies: { type: 'array', items: { type: 'string' } },
        parameterNames: { type: 'array', items: { type: 'string' } },
      },
    },
  },
}, async (request, reply) => {
  try {
    const body = RAGQuerySchema.parse(request.body);

    const result = await ragPipeline.query(body.query, {
      topK: body.topK,
      minSimilarity: body.minSimilarity,
      documentTypes: body.documentTypes,
      technologies: body.technologies,
      parameterNames: body.parameterNames,
      includeMetadata: true,
    });

    return {
      queryId: uuidv4(),
      answer: result.answer,
      confidence: result.confidence,
      sources: result.sources,
      chunks: result.chunks.map((c, i) => ({
        id: c.id,
        content: c.content.substring(0, 500) + (c.content.length > 500 ? '...' : ''),
        score: result.scores[i],
        metadata: c.metadata,
      })),
      processingTimeMs: result.processingTime,
    };
  } catch (error) {
    logger.error('RAG query failed', { error: (error as Error).message });
    reply.code(500).send({ error: 'Query failed', message: (error as Error).message });
  }
});

// Parameter lookup endpoint
fastify.get('/api/v1/rag/parameter/:name', {
  schema: {
    tags: ['RAG'],
    summary: 'Lookup a specific 3GPP parameter',
    params: {
      type: 'object',
      properties: {
        name: { type: 'string' },
      },
    },
  },
}, async (request, reply) => {
  const { name } = request.params as { name: string };

  // Query for the specific parameter
  const result = await ragPipeline.query(
    `What is the ${name} parameter? Include its type, range, default value, and 3GPP reference.`,
    {
      topK: 5,
      parameterNames: [name],
    }
  );

  return {
    parameter: name,
    answer: result.answer,
    sources: result.sources,
    confidence: result.confidence,
  };
});

// Feedback endpoint for self-learning
fastify.post('/api/v1/rag/feedback', {
  schema: {
    tags: ['RAG'],
    summary: 'Provide feedback on RAG results for self-learning',
    body: {
      type: 'object',
      required: ['queryId', 'chunkId', 'helpful'],
      properties: {
        queryId: { type: 'string' },
        chunkId: { type: 'string' },
        helpful: { type: 'boolean' },
      },
    },
  },
}, async (request, reply) => {
  const { queryId, chunkId, helpful } = request.body as {
    queryId: string;
    chunkId: string;
    helpful: boolean;
  };

  await ragPipeline.provideFeedback(queryId, chunkId, helpful);

  return { status: 'feedback recorded' };
});

// ============================================================================
// Ingestion Endpoints
// ============================================================================

// Ingest ELEX documentation
fastify.post('/api/v1/ingest/elex', {
  schema: {
    tags: ['Ingestion'],
    summary: 'Ingest ELEX documentation from ZIP files',
    body: {
      type: 'object',
      required: ['zipPaths'],
      properties: {
        zipPaths: { type: 'array', items: { type: 'string' } },
      },
    },
  },
}, async (request, reply) => {
  const { zipPaths } = request.body as { zipPaths: string[] };

  const chunksIngested = await ragPipeline.ingestELEX(zipPaths);

  return {
    status: 'completed',
    chunksIngested,
    stats: ragPipeline.getStats(),
  };
});

// Ingest 3GPP MOM
fastify.post('/api/v1/ingest/3gpp', {
  schema: {
    tags: ['Ingestion'],
    summary: 'Ingest 3GPP MOM XML files',
    body: {
      type: 'object',
      required: ['dirPath'],
      properties: {
        dirPath: { type: 'string' },
      },
    },
  },
}, async (request, reply) => {
  const { dirPath } = request.body as { dirPath: string };

  const chunksIngested = await ragPipeline.ingest3GPP(dirPath);

  return {
    status: 'completed',
    chunksIngested,
    stats: ragPipeline.getStats(),
  };
});

// ============================================================================
// Optimization Endpoints
// ============================================================================

// Create network graph
fastify.post('/api/v1/optimization/graph', {
  schema: {
    tags: ['Optimization'],
    summary: 'Create a network graph from cell configurations',
    body: {
      type: 'object',
      required: ['cells'],
      properties: {
        cells: { type: 'array', items: { type: 'object' } },
        neighborRelations: { type: 'object' },
      },
    },
  },
}, async (request, reply) => {
  try {
    const { cells, neighborRelations } = request.body as {
      cells: CellConfiguration[];
      neighborRelations?: Record<string, string[]>;
    };

    // Validate cells
    const validatedCells = cells.map((c) => CellConfigSchema.parse(c)) as CellConfiguration[];

    // Convert neighbor relations to Map
    const neighborMap = neighborRelations
      ? new Map(Object.entries(neighborRelations))
      : undefined;

    // Build graph
    const graph = graphBuilder.buildGraph(validatedCells, new Map(), neighborMap);

    // Normalize features
    graphBuilder.normalizeFeatures(graph);

    // Store graph
    graphs.set(graph.id, graph);

    // Create swarm for this graph
    const swarm = swarmController.createSwarm(graph.clusterId);
    swarms.set(graph.id, swarm);

    return {
      graphId: graph.id,
      clusterId: graph.clusterId,
      nodeCount: graph.nodes.size,
      edgeCount: graph.edges.length,
    };
  } catch (error) {
    logger.error('Failed to create graph', { error: (error as Error).message });
    reply.code(400).send({ error: 'Invalid input', message: (error as Error).message });
  }
});

// Run optimization
fastify.post('/api/v1/optimization/optimize', {
  schema: {
    tags: ['Optimization'],
    summary: 'Run optimization on a network graph',
    body: {
      type: 'object',
      required: ['graphId'],
      properties: {
        graphId: { type: 'string' },
        constraints: { type: 'object' },
      },
    },
  },
}, async (request, reply) => {
  try {
    const { graphId, constraints } = OptimizationRequestSchema.parse(request.body);

    const graph = graphs.get(graphId);
    if (!graph) {
      return reply.code(404).send({ error: 'Graph not found' });
    }

    const swarm = swarms.get(graphId);
    if (!swarm) {
      return reply.code(404).send({ error: 'Swarm not found for graph' });
    }

    // Run optimization cycle
    const action = await swarmController.runOptimizationCycle(graph, swarm);

    if (!action) {
      return {
        status: 'no_optimization_needed',
        message: 'No beneficial parameter changes found',
      };
    }

    return {
      status: action.status,
      actionId: action.id,
      prediction: {
        sinrImprovement: action.prediction.sinrImprovement,
        spectralEfficiencyGain: action.prediction.spectralEfficiencyGain,
        coverageImpact: action.prediction.coverageImpact,
        uncertainty: action.prediction.uncertainty,
        confidenceInterval: action.prediction.confidenceInterval,
      },
      changes: action.changes.map((c) => ({
        cellId: c.cellId,
        parameter: c.parameter,
        oldValue: c.oldValue,
        newValue: c.newValue,
      })),
      targetCellCount: action.targetCells.length,
    };
  } catch (error) {
    logger.error('Optimization failed', { error: (error as Error).message });
    reply.code(500).send({ error: 'Optimization failed', message: (error as Error).message });
  }
});

// Simulate parameter change
fastify.post('/api/v1/optimization/simulate', {
  schema: {
    tags: ['Optimization'],
    summary: 'Simulate parameter changes using GNN',
    body: {
      type: 'object',
      required: ['graphId', 'changes'],
      properties: {
        graphId: { type: 'string' },
        changes: {
          type: 'object',
          additionalProperties: {
            type: 'object',
            properties: {
              pZeroNominalPusch: { type: 'number' },
              alpha: { type: 'number' },
            },
          },
        },
      },
    },
  },
}, async (request, reply) => {
  const { graphId, changes } = request.body as {
    graphId: string;
    changes: Record<string, Partial<PowerControlParams>>;
  };

  const graph = graphs.get(graphId);
  if (!graph) {
    return reply.code(404).send({ error: 'Graph not found' });
  }

  // Convert to Map
  const changesMap = new Map(Object.entries(changes));

  // Run simulation
  const prediction = gnnEngine.predictSINRImprovement(graph, changesMap);

  return {
    prediction: {
      sinrImprovement: prediction.sinrImprovement,
      spectralEfficiencyGain: prediction.spectralEfficiencyGain,
      coverageImpact: prediction.coverageImpact,
      uncertainty: prediction.uncertainty,
      confidenceInterval: prediction.confidenceInterval,
      mcSamples: prediction.mcSamples,
      epistemicUncertainty: prediction.epistemicUncertainty,
      aleatoricUncertainty: prediction.aleatoricUncertainty,
    },
    meetsConfidenceThreshold: gnnEngine.meetsConfidenceThreshold(prediction),
  };
});

// ============================================================================
// Agent Endpoints
// ============================================================================

// List agents
fastify.get('/api/v1/agents', {
  schema: {
    tags: ['Agents'],
    summary: 'List all agents',
  },
}, async () => {
  const agents = swarmController.getAgents();
  return {
    agents: agents.map((a) => ({
      id: a.id,
      name: a.name,
      role: a.role,
      state: a.state,
      clusterId: a.clusterId,
      historyCount: a.history.length,
    })),
  };
});

// Get agent details
fastify.get('/api/v1/agents/:agentId', {
  schema: {
    tags: ['Agents'],
    summary: 'Get agent details',
    params: {
      type: 'object',
      properties: {
        agentId: { type: 'string' },
      },
    },
  },
}, async (request, reply) => {
  const { agentId } = request.params as { agentId: string };

  const agent = swarmController.getAgent(agentId);
  if (!agent) {
    return reply.code(404).send({ error: 'Agent not found' });
  }

  return {
    id: agent.id,
    name: agent.name,
    role: agent.role,
    state: agent.state,
    clusterId: agent.clusterId,
    config: agent.config,
    recentActions: agent.history.slice(-10).map((a) => ({
      id: a.id,
      type: a.type,
      status: a.status,
      timestamp: a.timestamp,
      targetCellCount: a.targetCells.length,
    })),
  };
});

// ============================================================================
// Start Server
// ============================================================================

export async function startServer() {
  try {
    // Initialize components
    await ragPipeline.initialize();

    // Register plugins
    await registerPlugins();

    // Start server
    await fastify.listen({
      port: config.server.port,
      host: config.server.host,
    });

    logger.info('Server started', {
      host: config.server.host,
      port: config.server.port,
      swagger: config.server.enableSwagger ? '/docs' : 'disabled',
    });
  } catch (error) {
    logger.error('Failed to start server', { error: (error as Error).message });
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  startServer();
}

export default fastify;
