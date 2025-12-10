"use strict";
/**
 * Production Configuration for Ericsson RAN Optimization SDK
 *
 * This configuration is optimized for production deployment with
 * maximum performance, reliability, and security.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateConfig = exports.getConfig = exports.ENVIRONMENT_CONFIGS = exports.PRODUCTION_TEST_CONFIG = exports.PRODUCTION_PERFORMANCE_CONFIG = exports.PRODUCTION_MCP_CONFIG = exports.PRODUCTION_RAN_CONFIG = void 0;
// Environment variables
const NODE_ENV = process.env.NODE_ENV || 'production';
const AGENTDB_PATH = process.env.AGENTDB_PATH || '/data/agentdb/ran-optimization.db';
const AGENTDB_QUIC_SYNC = process.env.AGENTDB_QUIC_SYNC === 'true';
const AGENTDB_QUIC_PEERS = (process.env.AGENTDB_QUIC_PEERS || '').split(',').filter(Boolean);
const CLAUDE_FLOW_TOPOLOGY = process.env.CLAUDE_FLOW_TOPOLOGY || 'hierarchical';
const FLOW_NEXUS_API_KEY = process.env.FLOW_NEXUS_API_KEY;
const FLOW_NEXUS_USER_ID = process.env.FLOW_NEXUS_USER_ID;
/**
 * Production RAN Optimization SDK Configuration
 */
exports.PRODUCTION_RAN_CONFIG = {
    claudeFlow: {
        topology: CLAUDE_FLOW_TOPOLOGY,
        maxAgents: 20,
        strategy: 'adaptive'
    },
    agentDB: {
        dbPath: AGENTDB_PATH,
        quantizationType: 'scalar',
        cacheSize: 2000,
        enableQUICSync: AGENTDB_QUIC_SYNC,
        syncPeers: AGENTDB_QUIC_PEERS
    },
    skillDiscovery: {
        maxContextSize: 6144,
        loadingStrategy: 'metadata-first',
        cacheEnabled: true
    },
    performance: {
        parallelExecution: true,
        cachingEnabled: true,
        benchmarkingEnabled: false,
        targetSpeedImprovement: 4.0
    },
    environment: NODE_ENV
};
/**
 * Production MCP Integration Configuration
 */
exports.PRODUCTION_MCP_CONFIG = {
    claudeFlow: {
        enabled: true,
        topology: CLAUDE_FLOW_TOPOLOGY,
        maxAgents: 20,
        strategy: 'adaptive'
    },
    flowNexus: {
        enabled: !!FLOW_NEXUS_API_KEY,
        autoAuth: true,
        creditManagement: {
            autoRefill: true,
            threshold: 100,
            amount: 50
        },
        sandbox: {
            template: 'claude-code',
            environment: {
                NODE_ENV: 'production',
                CLAUDE_FLOW_TOPOLOGY: CLAUDE_FLOW_TOPOLOGY,
                AGENTDB_PATH: AGENTDB_PATH,
                AGENTDB_QUIC_SYNC: AGENTDB_QUIC_SYNC.toString(),
                RAN_OPTIMIZATION_CYCLE: '15',
                LOG_LEVEL: 'info'
            },
            packages: [
                '@agentic-flow/agentdb',
                'claude-flow',
                '@ericsson/ran-optimization-sdk',
                'typescript'
            ]
        }
    },
    ruvSwarm: {
        enabled: true,
        topology: 'mesh',
        maxAgents: 10,
        strategy: 'specialized'
    },
    performance: {
        timeoutMs: 30000,
        retryAttempts: 3,
        batchSize: 5,
        parallelism: 8
    }
};
/**
 * Production Performance Configuration
 */
exports.PRODUCTION_PERFORMANCE_CONFIG = {
    caching: {
        enabled: true,
        strategy: 'lru',
        maxSize: 10000,
        ttlMs: 300000,
        compressionEnabled: true
    },
    vectorSearch: {
        hnswConfig: {
            M: 16,
            efConstruction: 100,
            efSearch: 50
        },
        quantization: 'scalar',
        mmrEnabled: true,
        mmrLambda: 0.5,
        targetSpeedup: 150
    },
    parallelism: {
        enabled: true,
        maxConcurrency: 20,
        batchSize: 5,
        loadBalancing: 'adaptive'
    },
    memory: {
        compressionEnabled: true,
        gcOptimization: true,
        poolSize: 1000,
        threshold: 0.8
    },
    monitoring: {
        enabled: true,
        metricsInterval: 30000,
        alertThresholds: {
            latency: 1000,
            errorRate: 0.05,
            memoryUsage: 0.9 // 90%
        }
    }
};
/**
 * Production Testing Configuration
 */
exports.PRODUCTION_TEST_CONFIG = {
    environment: 'production',
    coverage: {
        unitTests: false,
        integrationTests: true,
        performanceTests: true,
        securityTests: true,
        loadTests: false // Only run in staging
    },
    execution: {
        parallel: true,
        maxConcurrency: 5,
        timeoutMs: 60000,
        retryAttempts: 2,
        continueOnFailure: true
    },
    targets: {
        swetBenchSolveRate: 84.8,
        speedImprovement: 2.8,
        vectorSearchSpeedup: 150,
        cacheHitRate: 0.85,
        successRate: 0.95
    },
    monitoring: {
        enabled: true,
        detailedLogs: false,
        performanceMetrics: true,
        coverageReport: false
    }
};
/**
 * Environment-specific configurations
 */
exports.ENVIRONMENT_CONFIGS = {
    development: {
        claudeFlow: {
            topology: 'hierarchical',
            maxAgents: 5,
            strategy: 'balanced'
        },
        agentDB: {
            dbPath: '.agentdb/development.db',
            quantizationType: 'none',
            cacheSize: 500,
            enableQUICSync: false,
            syncPeers: []
        },
        skillDiscovery: {
            maxContextSize: 12288,
            loadingStrategy: 'eager',
            cacheEnabled: true
        },
        performance: {
            parallelExecution: true,
            cachingEnabled: true,
            benchmarkingEnabled: true,
            targetSpeedImprovement: 2.0
        },
        environment: 'development'
    },
    staging: {
        claudeFlow: {
            topology: 'hierarchical',
            maxAgents: 15,
            strategy: 'adaptive'
        },
        agentDB: {
            dbPath: '.agentdb/staging.db',
            quantizationType: 'scalar',
            cacheSize: 1500,
            enableQUICSync: true,
            syncPeers: ['staging-db-1:4433', 'staging-db-2:4433']
        },
        skillDiscovery: {
            maxContextSize: 6144,
            loadingStrategy: 'metadata-first',
            cacheEnabled: true
        },
        performance: {
            parallelExecution: true,
            cachingEnabled: true,
            benchmarkingEnabled: true,
            targetSpeedImprovement: 3.5
        },
        environment: 'staging'
    },
    production: exports.PRODUCTION_RAN_CONFIG
};
/**
 * Get configuration for current environment
 */
function getConfig(environment) {
    const env = environment || NODE_ENV;
    return exports.ENVIRONMENT_CONFIGS[env] || exports.PRODUCTION_RAN_CONFIG;
}
exports.getConfig = getConfig;
/**
 * Validate configuration
 */
function validateConfig(config) {
    const errors = [];
    // Validate AgentDB configuration
    if (!config.agentDB.dbPath) {
        errors.push('AgentDB path is required');
    }
    if (config.agentDB.cacheSize < 100 || config.agentDB.cacheSize > 10000) {
        errors.push('AgentDB cache size must be between 100 and 10000');
    }
    // Validate Claude Flow configuration
    if (config.claudeFlow.maxAgents < 1 || config.claudeFlow.maxAgents > 50) {
        errors.push('Claude Flow max agents must be between 1 and 50');
    }
    // Validate skill discovery configuration
    if (config.skillDiscovery.maxContextSize < 1024 || config.skillDiscovery.maxContextSize > 16384) {
        errors.push('Skill discovery max context size must be between 1KB and 16KB');
    }
    // Validate performance configuration
    if (config.performance.targetSpeedImprovement < 1.0 || config.performance.targetSpeedImprovement > 10.0) {
        errors.push('Target speed improvement must be between 1x and 10x');
    }
    return {
        valid: errors.length === 0,
        errors
    };
}
exports.validateConfig = validateConfig;
/**
 * Default export for production configuration
 */
exports.default = {
    ran: exports.PRODUCTION_RAN_CONFIG,
    mcp: exports.PRODUCTION_MCP_CONFIG,
    performance: exports.PRODUCTION_PERFORMANCE_CONFIG,
    testing: exports.PRODUCTION_TEST_CONFIG,
    environments: exports.ENVIRONMENT_CONFIGS,
    getConfig,
    validateConfig
};
//# sourceMappingURL=production-config.js.map