/**
 * =============================================================================
 * Ruvnet Ecosystem Integration
 * Unified interface for agentic-flow, claude-flow, agentdb, and ruvector
 * =============================================================================
 */

// Export all ruvnet integrations
export { AgenticFlowIntegration } from './agentic-flow.js';
export { ClaudeFlowIntegration } from './claude-flow.js';
export { AgentDBIntegration } from './agentdb.js';
export { RuVectorIntegration } from './ruvector.js';
export { RuvnetOrchestrator } from './orchestrator.js';

/**
 * Initialize the complete Ruvnet ecosystem
 * @param {Object} config - Configuration options
 * @returns {Promise<RuvnetOrchestrator>} Initialized orchestrator
 */
export async function initializeRuvnet(config = {}) {
    const { RuvnetOrchestrator } = await import('./orchestrator.js');
    const orchestrator = new RuvnetOrchestrator(config);
    await orchestrator.initialize();
    return orchestrator;
}

/**
 * Default configuration for Edge-Native AI deployment
 */
export const DEFAULT_CONFIG = {
    // Agentic Flow configuration
    agenticFlow: {
        booster: {
            enabled: true,
            wasmPath: '/opt/edge-ai/wasm/agent-booster.wasm',
            speedupTarget: 352
        },
        providers: ['local', 'openai', 'anthropic'],
        defaultProvider: 'local'
    },

    // Claude Flow configuration
    claudeFlow: {
        reasoningBank: {
            enabled: true,
            maxSize: 10000,
            persistPath: '/opt/edge-ai/data/reasoning-bank'
        },
        sparc: {
            enabled: true,
            methodology: 'sparc2'
        }
    },

    // AgentDB configuration
    agentdb: {
        local: {
            path: '/opt/edge-ai/data/agentdb.sqlite'
        },
        distributed: {
            redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
            syncInterval: 5000
        },
        encryption: {
            enabled: true,
            key: process.env.AGENTDB_ENCRYPTION_KEY
        }
    },

    // RuVector configuration
    ruvector: {
        backend: 'local', // 'local' | 'postgres' | 'remote'
        dimensions: 1536,
        indexType: 'hnsw',
        local: {
            path: '/opt/edge-ai/data/ruvector'
        },
        postgres: {
            connectionString: process.env.POSTGRES_URL
        }
    },

    // LiteLLM Gateway
    gateway: {
        url: process.env.LITELLM_URL || 'http://localhost:4000',
        apiKey: process.env.LITELLM_MASTER_KEY
    }
};
