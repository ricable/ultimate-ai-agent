export function createDefaultConfig(): {
    project: {
        name: string;
        template: string;
        version: string;
        created: string;
    };
    mesh: {
        topology: string;
        defaultAgents: number;
        coordinationPort: number;
        heartbeatInterval: number;
        nodeTimeout: number;
    };
    neural: {
        enabled: boolean;
        port: number;
        defaultModel: string;
        trainingEnabled: boolean;
        gpuAcceleration: boolean;
    };
    dag: {
        enabled: boolean;
        port: number;
        maxConcurrentWorkflows: number;
        workflowTimeout: number;
    };
    peer: {
        enabled: boolean;
        port: number;
        autoDiscovery: boolean;
        maxPeers: number;
        discoveryInterval: number;
        protocols: string[];
    };
    features: {
        mcp: boolean;
        mcpPort: number;
        webui: boolean;
        webuiPort: number;
        monitoring: boolean;
        logging: boolean;
        backup: boolean;
    };
    security: {
        encryption: boolean;
        authentication: boolean;
        certificates: {
            autoGenerate: boolean;
            keySize: number;
        };
    };
    storage: {
        provider: string;
        path: string;
        backup: {
            enabled: boolean;
            interval: number;
            retention: number;
        };
    };
    logging: {
        level: string;
        file: string;
        maxSize: string;
        maxFiles: number;
        console: boolean;
    };
    performance: {
        workerThreads: number;
        maxMemory: number;
        cacheSize: number;
        enableOptimizations: boolean;
    };
};
//# sourceMappingURL=default.d.ts.map