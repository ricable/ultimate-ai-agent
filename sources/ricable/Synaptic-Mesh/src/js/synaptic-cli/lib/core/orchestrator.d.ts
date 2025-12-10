/**
 * Mesh Orchestrator - Core coordination engine
 */
export class MeshOrchestrator {
    constructor(config: any, options?: {});
    config: any;
    options: {};
    services: Map<any, any>;
    agents: Map<any, any>;
    running: boolean;
    initialize(): Promise<void>;
    startMCPServer(): Promise<void>;
    initializeMesh(): Promise<void>;
    spawnInitialAgents(): Promise<void>;
    startPeerDiscovery(): Promise<void>;
    shutdown(): Promise<void>;
    getStatus(): {
        running: boolean;
        services: any[];
        agents: number;
        config: any;
    };
}
//# sourceMappingURL=orchestrator.d.ts.map