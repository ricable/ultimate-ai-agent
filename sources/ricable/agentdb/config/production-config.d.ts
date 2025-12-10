/**
 * Production Configuration for Ericsson RAN Optimization SDK
 *
 * This configuration is optimized for production deployment with
 * maximum performance, reliability, and security.
 */
import { RANOptimizationConfig } from '../src/sdk/ran-optimization-sdk';
import { MCPIntegrationConfig } from '../src/sdk/mcp-integration';
import { PerformanceConfig } from '../src/sdk/performance-optimizer';
import { TestConfig } from '../src/testing/integration-test-suite';
/**
 * Production RAN Optimization SDK Configuration
 */
export declare const PRODUCTION_RAN_CONFIG: RANOptimizationConfig;
/**
 * Production MCP Integration Configuration
 */
export declare const PRODUCTION_MCP_CONFIG: MCPIntegrationConfig;
/**
 * Production Performance Configuration
 */
export declare const PRODUCTION_PERFORMANCE_CONFIG: PerformanceConfig;
/**
 * Production Testing Configuration
 */
export declare const PRODUCTION_TEST_CONFIG: TestConfig;
/**
 * Environment-specific configurations
 */
export declare const ENVIRONMENT_CONFIGS: {
    development: {
        claudeFlow: {
            topology: "hierarchical";
            maxAgents: number;
            strategy: "balanced";
        };
        agentDB: {
            dbPath: string;
            quantizationType: "none";
            cacheSize: number;
            enableQUICSync: boolean;
            syncPeers: never[];
        };
        skillDiscovery: {
            maxContextSize: number;
            loadingStrategy: "eager";
            cacheEnabled: boolean;
        };
        performance: {
            parallelExecution: boolean;
            cachingEnabled: boolean;
            benchmarkingEnabled: boolean;
            targetSpeedImprovement: number;
        };
        environment: "development";
    };
    staging: {
        claudeFlow: {
            topology: "hierarchical";
            maxAgents: number;
            strategy: "adaptive";
        };
        agentDB: {
            dbPath: string;
            quantizationType: "scalar";
            cacheSize: number;
            enableQUICSync: boolean;
            syncPeers: string[];
        };
        skillDiscovery: {
            maxContextSize: number;
            loadingStrategy: "metadata-first";
            cacheEnabled: boolean;
        };
        performance: {
            parallelExecution: boolean;
            cachingEnabled: boolean;
            benchmarkingEnabled: boolean;
            targetSpeedImprovement: number;
        };
        environment: "staging";
    };
    production: RANOptimizationConfig;
};
/**
 * Get configuration for current environment
 */
export declare function getConfig(environment?: string): RANOptimizationConfig;
/**
 * Validate configuration
 */
export declare function validateConfig(config: RANOptimizationConfig): {
    valid: boolean;
    errors: string[];
};
/**
 * Default export for production configuration
 */
declare const _default: {
    ran: RANOptimizationConfig;
    mcp: MCPIntegrationConfig;
    performance: PerformanceConfig;
    testing: TestConfig;
    environments: {
        development: {
            claudeFlow: {
                topology: "hierarchical";
                maxAgents: number;
                strategy: "balanced";
            };
            agentDB: {
                dbPath: string;
                quantizationType: "none";
                cacheSize: number;
                enableQUICSync: boolean;
                syncPeers: never[];
            };
            skillDiscovery: {
                maxContextSize: number;
                loadingStrategy: "eager";
                cacheEnabled: boolean;
            };
            performance: {
                parallelExecution: boolean;
                cachingEnabled: boolean;
                benchmarkingEnabled: boolean;
                targetSpeedImprovement: number;
            };
            environment: "development";
        };
        staging: {
            claudeFlow: {
                topology: "hierarchical";
                maxAgents: number;
                strategy: "adaptive";
            };
            agentDB: {
                dbPath: string;
                quantizationType: "scalar";
                cacheSize: number;
                enableQUICSync: boolean;
                syncPeers: string[];
            };
            skillDiscovery: {
                maxContextSize: number;
                loadingStrategy: "metadata-first";
                cacheEnabled: boolean;
            };
            performance: {
                parallelExecution: boolean;
                cachingEnabled: boolean;
                benchmarkingEnabled: boolean;
                targetSpeedImprovement: number;
            };
            environment: "staging";
        };
        production: RANOptimizationConfig;
    };
    getConfig: typeof getConfig;
    validateConfig: typeof validateConfig;
};
export default _default;
//# sourceMappingURL=production-config.d.ts.map