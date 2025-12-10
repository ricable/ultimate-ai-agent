import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const ConfigSchema = z.object({
  // Anthropic API
  anthropicApiKey: z.string().min(1, 'ANTHROPIC_API_KEY is required'),
  claudeModel: z.string().default('claude-sonnet-4-5-20250929'),

  // Agent Configuration
  maxConcurrentAgents: z.number().default(10),
  agentTimeoutMs: z.number().default(300000),

  // Time Series Configuration
  tsGranularities: z.array(z.enum(['hourly', 'daily', 'weekly'])).default(['hourly', 'daily', 'weekly']),
  tsPredictionHorizon: z.number().default(168), // hours
  tsLookbackPeriod: z.number().default(720), // hours

  // RAN Configuration
  ranNodeTypes: z.array(z.string()).default(['gNB', 'eNB', '5G-SA', '4G-LTE']),
  cellAggregationLevels: z.array(z.string()).default(['cell', 'sector', 'cluster', 'region']),

  // Database
  dbPath: z.string().default('./data/agentdb'),
  timeseriesDbPath: z.string().default('./data/timeseries'),

  // Logging
  logLevel: z.string().default('info'),
  logPath: z.string().default('./logs'),
});

export type Config = z.infer<typeof ConfigSchema>;

function loadConfig(): Config {
  const rawConfig = {
    anthropicApiKey: process.env.ANTHROPIC_API_KEY || '',
    claudeModel: process.env.CLAUDE_MODEL || 'claude-sonnet-4-5-20250929',
    maxConcurrentAgents: parseInt(process.env.MAX_CONCURRENT_AGENTS || '10', 10),
    agentTimeoutMs: parseInt(process.env.AGENT_TIMEOUT_MS || '300000', 10),
    tsGranularities: (process.env.TS_GRANULARITIES || 'hourly,daily,weekly')
      .split(',')
      .map(g => g.trim()) as ('hourly' | 'daily' | 'weekly')[],
    tsPredictionHorizon: parseInt(process.env.TS_PREDICTION_HORIZON || '168', 10),
    tsLookbackPeriod: parseInt(process.env.TS_LOOKBACK_PERIOD || '720', 10),
    ranNodeTypes: (process.env.RAN_NODE_TYPES || 'gNB,eNB,5G-SA,4G-LTE')
      .split(',')
      .map(t => t.trim()),
    cellAggregationLevels: (process.env.CELL_AGGREGATION_LEVELS || 'cell,sector,cluster,region')
      .split(',')
      .map(l => l.trim()),
    dbPath: process.env.DB_PATH || './data/agentdb',
    timeseriesDbPath: process.env.TIMESERIES_DB_PATH || './data/timeseries',
    logLevel: process.env.LOG_LEVEL || 'info',
    logPath: process.env.LOG_PATH || './logs',
  };

  return ConfigSchema.parse(rawConfig);
}

export const config = loadConfig();

export default config;
