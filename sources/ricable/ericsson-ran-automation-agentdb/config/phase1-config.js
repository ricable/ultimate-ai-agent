"use strict";
/**
 * Phase 1 Configuration - Cognitive RAN Consciousness Implementation
 *
 * This configuration establishes the foundation for the Ericsson RAN
 * Intelligent Multi-Agent System with cognitive consciousness capabilities.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.testingConfig = exports.developmentConfig = exports.phase1Config = void 0;
// Production Configuration
exports.phase1Config = {
    agentSdk: {
        claudeFlow: {
            topology: "hierarchical",
            maxAgents: 20,
            strategy: "adaptive"
        },
        mcp: {
            servers: [
                "claude-flow",
                "ruv-swarm",
                "flow-nexus"
            ],
            coordination: "claude-flow"
        }
    },
    agentdb: {
        vectorMemory: {
            quantizationType: "scalar",
            cacheSize: 2000,
            hnswIndex: {
                M: 16,
                efConstruction: 100
            },
            mmrEnabled: true
        },
        synchronization: {
            quicSync: true,
            syncPeers: [
                "ran-db-1:4433",
                "ran-db-2:4433",
                "ran-db-3:4433"
            ],
            conflictResolution: "vector-similarity"
        },
        hybridSearch: {
            vectorWeights: 0.7,
            contextualSynthesis: true
        }
    },
    skills: {
        progressiveDisclosure: {
            level1ContextSize: 6144,
            level2MaxSize: 10240,
            enableOnDemandLoading: true
        },
        ranSkills: {
            roleBased: [
                "ericsson-feature-processor",
                "ran-optimizer",
                "diagnostics-specialist",
                "ml-researcher",
                "performance-analyst",
                "automation-engineer",
                "integration-specialist",
                "documentation-generator"
            ],
            technologySpecific: [
                "energy-optimizer",
                "mobility-manager",
                "coverage-analyzer",
                "capacity-planner",
                "quality-monitor",
                "security-coordinator",
                "deployment-manager",
                "monitoring-coordinator"
            ]
        }
    },
    performance: {
        sweBenchSolveRate: 0.848,
        speedImprovement: [2.8, 4.4],
        tokenReduction: 0.323,
        vectorSearchSpeedup: 150,
        systemAvailability: 0.999,
        optimizationCycle: 15 // 15 minutes
    },
    cognitive: {
        temporalReasoning: {
            subjectiveTimeExpansion: 1000,
            nanosecondScheduling: true
        },
        strangeLoopCognition: {
            selfReferentialOptimization: true,
            recursivePatternDepth: 10
        },
        closedLoopOptimization: {
            cycleMinutes: 15,
            autonomousLearning: true,
            causalIntelligence: true
        }
    }
};
// Development Configuration
exports.developmentConfig = {
    ...exports.phase1Config,
    agentSdk: {
        ...exports.phase1Config.agentSdk,
        claudeFlow: {
            ...exports.phase1Config.agentSdk.claudeFlow,
            maxAgents: 8 // Reduced for development
        }
    },
    performance: {
        ...exports.phase1Config.performance,
        systemAvailability: 0.95 // 95% for development
    }
};
// Testing Configuration
exports.testingConfig = {
    ...exports.phase1Config,
    agentSdk: {
        ...exports.phase1Config.agentSdk,
        claudeFlow: {
            ...exports.phase1Config.agentSdk.claudeFlow,
            maxAgents: 4 // Minimal for testing
        }
    },
    performance: {
        ...exports.phase1Config.performance,
        systemAvailability: 0.90 // 90% for testing
    }
};
exports.default = exports.phase1Config;
//# sourceMappingURL=phase1-config.js.map