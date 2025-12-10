/**
 * Phase 1 Configuration - Cognitive RAN Consciousness Implementation
 *
 * This configuration establishes the foundation for the Ericsson RAN
 * Intelligent Multi-Agent System with cognitive consciousness capabilities.
 */
export interface Phase1Config {
    agentSdk: {
        claudeFlow: {
            topology: "hierarchical" | "mesh" | "ring" | "star";
            maxAgents: number;
            strategy: "balanced" | "specialized" | "adaptive";
        };
        mcp: {
            servers: string[];
            coordination: "claude-flow" | "mcp" | "hybrid";
        };
    };
    agentdb: {
        vectorMemory: {
            quantizationType: "binary" | "scalar" | "product";
            cacheSize: number;
            hnswIndex: {
                M: number;
                efConstruction: number;
            };
            mmrEnabled: boolean;
        };
        synchronization: {
            quicSync: boolean;
            syncPeers: string[];
            conflictResolution: "last-write-wins" | "vector-similarity";
        };
        hybridSearch: {
            vectorWeights: number;
            contextualSynthesis: boolean;
        };
    };
    skills: {
        progressiveDisclosure: {
            level1ContextSize: number;
            level2MaxSize: number;
            enableOnDemandLoading: boolean;
        };
        ranSkills: {
            roleBased: string[];
            technologySpecific: string[];
        };
    };
    performance: {
        sweBenchSolveRate: number;
        speedImprovement: [number, number];
        tokenReduction: number;
        vectorSearchSpeedup: number;
        systemAvailability: number;
        optimizationCycle: number;
    };
    cognitive: {
        temporalReasoning: {
            subjectiveTimeExpansion: number;
            nanosecondScheduling: boolean;
        };
        strangeLoopCognition: {
            selfReferentialOptimization: boolean;
            recursivePatternDepth: number;
        };
        closedLoopOptimization: {
            cycleMinutes: number;
            autonomousLearning: boolean;
            causalIntelligence: boolean;
        };
    };
}
export declare const phase1Config: Phase1Config;
export declare const developmentConfig: Phase1Config;
export declare const testingConfig: Phase1Config;
export default phase1Config;
//# sourceMappingURL=phase1-config.d.ts.map