/**
 * RuVector Configuration Management
 *
 * Centralized configuration for the Ericsson RAN Cognitive Automation Platform
 */

import { config as dotenvConfig } from 'dotenv';
import { z } from 'zod';
import { AlphaValues, type AlphaValue } from './types.js';

export { AlphaValues };

// Load environment variables
dotenvConfig();

const ConfigSchema = z.object({
  // LLM Configuration
  llm: z.object({
    openaiApiKey: z.string().optional(),
    anthropicApiKey: z.string().optional(),
    embeddingModel: z.string().default('text-embedding-3-large'),
    embeddingDimensions: z.number().default(3072),
    completionModel: z.string().default('gpt-4-turbo-preview'),
  }),

  // Vector Database
  vectorDb: z.object({
    storagePath: z.string().default('./data/vector_store'),
    indexType: z.enum(['hnsw', 'flat', 'ivf']).default('hnsw'),
    m: z.number().default(32),
    efConstruction: z.number().default(200),
    efSearch: z.number().default(100),
  }),

  // Graph Database
  graphDb: z.object({
    storagePath: z.string().default('./data/graph_store'),
    maxNodes: z.number().default(1000000),
    maxEdges: z.number().default(10000000),
  }),

  // GNN Configuration
  gnn: z.object({
    hiddenDim: z.number().default(256),
    numLayers: z.number().default(4),
    dropout: z.number().default(0.1),
    learningRate: z.number().default(0.001),
    batchSize: z.number().default(64),
  }),

  // Bayesian Settings
  bayesian: z.object({
    mcSamples: z.number().default(100),
    priorScale: z.number().default(1.0),
    uncertaintyThreshold: z.number().default(0.5),
  }),

  // Agent Swarm
  swarm: z.object({
    maxAgents: z.number().default(100),
    coordinationInterval: z.number().default(5000),
    explorationRate: z.number().default(0.1),
    learningRate: z.number().default(0.01),
  }),

  // Optimization
  optimization: z.object({
    populationSize: z.number().default(50),
    generations: z.number().default(100),
    mutationRate: z.number().default(0.1),
    crossoverRate: z.number().default(0.7),
  }),

  // Network Thresholds
  thresholds: z.object({
    minEdgeSinrDb: z.number().default(-5),
    maxInterferenceDb: z.number().default(-90),
    targetSpectralEfficiency: z.number().default(3.0),
    minCoveragePercentage: z.number().default(95),
  }),

  // 3GPP Parameter Ranges
  parameters: z.object({
    alphaMin: z.number().default(0.4),
    alphaMax: z.number().default(1.0),
    alphaValues: z.array(z.number()).default([...AlphaValues]),
    p0Min: z.number().default(-126),
    p0Max: z.number().default(24),
    defaultAlpha: z.number().default(1.0),
    defaultP0: z.number().default(-100),
  }),

  // Data Paths
  paths: z.object({
    elexData: z.string().default('./data/elex'),
    threeGppData: z.string().default('./data/3gpp'),
    processedData: z.string().default('./data/processed'),
  }),

  // Server
  server: z.object({
    host: z.string().default('0.0.0.0'),
    port: z.number().default(8080),
    enableSwagger: z.boolean().default(true),
  }),

  // Logging
  logging: z.object({
    level: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
    format: z.enum(['json', 'pretty']).default('json'),
  }),

  // ENM Integration
  enm: z.object({
    apiUrl: z.string().optional(),
    username: z.string().optional(),
    password: z.string().optional(),
    enabled: z.boolean().default(false),
  }),
});

type Config = z.infer<typeof ConfigSchema>;

function loadConfig(): Config {
  const rawConfig = {
    llm: {
      openaiApiKey: process.env.OPENAI_API_KEY,
      anthropicApiKey: process.env.ANTHROPIC_API_KEY,
      embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-3-large',
      embeddingDimensions: parseInt(process.env.EMBEDDING_DIMENSIONS || '3072', 10),
      completionModel: process.env.COMPLETION_MODEL || 'gpt-4-turbo-preview',
    },
    vectorDb: {
      storagePath: process.env.RUVECTOR_STORAGE_PATH || './data/vector_store',
      indexType: process.env.RUVECTOR_INDEX_TYPE || 'hnsw',
      m: parseInt(process.env.RUVECTOR_M || '32', 10),
      efConstruction: parseInt(process.env.RUVECTOR_EF_CONSTRUCTION || '200', 10),
      efSearch: parseInt(process.env.RUVECTOR_EF_SEARCH || '100', 10),
    },
    graphDb: {
      storagePath: process.env.GRAPH_STORAGE_PATH || './data/graph_store',
      maxNodes: parseInt(process.env.GRAPH_MAX_NODES || '1000000', 10),
      maxEdges: parseInt(process.env.GRAPH_MAX_EDGES || '10000000', 10),
    },
    gnn: {
      hiddenDim: parseInt(process.env.GNN_HIDDEN_DIM || '256', 10),
      numLayers: parseInt(process.env.GNN_NUM_LAYERS || '4', 10),
      dropout: parseFloat(process.env.GNN_DROPOUT || '0.1'),
      learningRate: parseFloat(process.env.GNN_LEARNING_RATE || '0.001'),
      batchSize: parseInt(process.env.GNN_BATCH_SIZE || '64', 10),
    },
    bayesian: {
      mcSamples: parseInt(process.env.BAYESIAN_MC_SAMPLES || '100', 10),
      priorScale: parseFloat(process.env.BAYESIAN_PRIOR_SCALE || '1.0'),
      uncertaintyThreshold: parseFloat(process.env.UNCERTAINTY_THRESHOLD || '0.5'),
    },
    swarm: {
      maxAgents: parseInt(process.env.SWARM_MAX_AGENTS || '100', 10),
      coordinationInterval: parseInt(process.env.SWARM_COORDINATION_INTERVAL || '5000', 10),
      explorationRate: parseFloat(process.env.AGENT_EXPLORATION_RATE || '0.1'),
      learningRate: parseFloat(process.env.AGENT_LEARNING_RATE || '0.01'),
    },
    optimization: {
      populationSize: parseInt(process.env.OPTIMIZATION_POPULATION_SIZE || '50', 10),
      generations: parseInt(process.env.OPTIMIZATION_GENERATIONS || '100', 10),
      mutationRate: parseFloat(process.env.MUTATION_RATE || '0.1'),
      crossoverRate: parseFloat(process.env.CROSSOVER_RATE || '0.7'),
    },
    thresholds: {
      minEdgeSinrDb: parseFloat(process.env.MIN_EDGE_SINR_DB || '-5'),
      maxInterferenceDb: parseFloat(process.env.MAX_INTERFERENCE_DB || '-90'),
      targetSpectralEfficiency: parseFloat(process.env.TARGET_SPECTRAL_EFFICIENCY || '3.0'),
      minCoveragePercentage: parseFloat(process.env.MIN_COVERAGE_PERCENTAGE || '95'),
    },
    parameters: {
      alphaMin: parseFloat(process.env.ALPHA_MIN || '0.4'),
      alphaMax: parseFloat(process.env.ALPHA_MAX || '1.0'),
      alphaValues: [...AlphaValues],
      p0Min: parseInt(process.env.P0_MIN || '-126', 10),
      p0Max: parseInt(process.env.P0_MAX || '24', 10),
      defaultAlpha: 1.0,
      defaultP0: -100,
    },
    paths: {
      elexData: process.env.ELEX_DATA_PATH || './data/elex',
      threeGppData: process.env.THREEGPP_DATA_PATH || './data/3gpp',
      processedData: process.env.PROCESSED_DATA_PATH || './data/processed',
    },
    server: {
      host: process.env.SERVER_HOST || '0.0.0.0',
      port: parseInt(process.env.SERVER_PORT || '8080', 10),
      enableSwagger: process.env.ENABLE_SWAGGER !== 'false',
    },
    logging: {
      level: process.env.LOG_LEVEL || 'info',
      format: process.env.LOG_FORMAT || 'json',
    },
    enm: {
      apiUrl: process.env.ENM_API_URL,
      username: process.env.ENM_USERNAME,
      password: process.env.ENM_PASSWORD,
      enabled: process.env.ENM_ENABLED === 'true',
    },
  };

  return ConfigSchema.parse(rawConfig);
}

// Singleton config instance
let configInstance: Config | null = null;

export function getConfig(): Config {
  if (!configInstance) {
    configInstance = loadConfig();
  }
  return configInstance;
}

export function resetConfig(): void {
  configInstance = null;
}

// Export type for use in other modules
export type { Config };
