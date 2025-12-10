/**
 * Mesh Client - Interface to mesh coordination service
 */
export class MeshClient {
    constructor(host?: string, port?: number);
    host: string;
    port: number;
    baseUrl: string;
    getStatus(): Promise<{
        running: boolean;
        activeNodes: number;
        totalNodes: number;
        uptime: number;
        connections: number;
        neural: {
            running: boolean;
            connections: number;
        };
        dag: {
            running: boolean;
            connections: number;
        };
        p2p: {
            running: boolean;
            peers: number;
        };
        mcp: {
            running: boolean;
            connections: number;
        };
        metrics: {
            tasksProcessed: number;
            avgLatency: number;
            memoryUsage: number;
            cpuUsage: number;
            networkIO: number;
        };
        activity: never[];
    }>;
    getNodes(): Promise<never[]>;
    addNode(config: any): Promise<any>;
    removeNode(nodeId: any): Promise<boolean>;
    connectNodes(sourceId: any, targetId: any, options?: {}): Promise<boolean>;
    disconnectNodes(sourceId: any, targetId: any): Promise<boolean>;
    getTopology(): Promise<{
        type: string;
        nodes: never[];
        connections: never[];
    }>;
    optimizeTopology(strategy: any): Promise<{
        strategy: any;
        nodesAffected: number;
        connectionsChanged: number;
        performanceGain: number;
    }>;
}
//# sourceMappingURL=mesh-client.d.ts.map