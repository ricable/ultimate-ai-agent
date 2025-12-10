/**
 * Peer Client - Interface to P2P network service
 */
export class PeerClient {
    constructor(host?: string, port?: number);
    host: string;
    port: number;
    baseUrl: string;
    getPeers(status?: null): Promise<never[]>;
    connectToPeer(address: any, config?: {}): Promise<{
        id: string;
        address: any;
        protocol: string;
        latency: number;
        connectedAt: string;
    }>;
    disconnectFromPeer(peerId: any): Promise<boolean>;
    discoverPeers(config?: {}): Promise<never[]>;
    shareFile(filePath: any, config?: {}): Promise<{
        hash: string;
        size: number;
        replicas: never[];
    }>;
    downloadFile(hash: any, config?: {}): Promise<{
        path: any;
        size: number;
        sourcePeer: string;
        duration: number;
    }>;
    getNetworkStatus(): Promise<{
        nodeId: string;
        connectedPeers: number;
        networkSize: number;
        dataShared: number;
        dataReceived: number;
        uptime: number;
        protocols: {
            name: string;
            version: string;
        }[];
    }>;
}
//# sourceMappingURL=peer-client.d.ts.map