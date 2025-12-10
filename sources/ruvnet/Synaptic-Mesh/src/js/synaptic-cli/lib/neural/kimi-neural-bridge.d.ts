/**
 * Kimi Neural Bridge - Phase 4: Deep neural mesh integration
 *
 * Implements bidirectional AI-mesh communication:
 * - Inject Kimi-K2 thoughts into neural mesh
 * - Synchronize AI responses with mesh state
 * - Coordinate with DAA swarms
 * - Real-time thought synchronization
 */
import { EventEmitter } from 'events';
export interface NeuralThought {
    id: string;
    timestamp: number;
    source: 'kimi' | 'mesh' | 'user';
    content: string;
    confidence: number;
    context: any;
    relationships: string[];
}
export interface MeshState {
    nodes: Map<string, any>;
    connections: Map<string, any>;
    activeAgents: string[];
    consensus: any;
    lastUpdate: number;
}
export interface SyncProtocol {
    type: 'inject' | 'sync' | 'coordinate' | 'learn';
    payload: any;
    timestamp: number;
    priority: 'low' | 'medium' | 'high' | 'critical';
}
export declare class KimiNeuralBridge extends EventEmitter {
    private meshState;
    private thoughts;
    private syncQueue;
    private isActive;
    private learningHistory;
    private daaIntegration;
    private metrics;
    constructor();
    /**
     * Inject Kimi-K2 AI thoughts into the neural mesh
     */
    injectThought(content: string, context?: any, confidence?: number): Promise<string>;
    /**
     * Synchronize AI responses with mesh state
     */
    synchronizeWithMesh(): Promise<void>;
    /**
     * Coordinate with DAA swarms
     */
    coordinateWithSwarm(swarmId: string, coordinationType: string, payload: any): Promise<any>;
    /**
     * Real-time thought synchronization
     */
    startThoughtSync(interval?: number): Promise<void>;
    /**
     * Stop thought synchronization
     */
    stopThoughtSync(): Promise<void>;
    /**
     * Get bridge status and metrics
     */
    getStatus(): any;
    /**
     * Export neural bridge data for analysis
     */
    exportBridgeData(): any;
    private setupEventHandlers;
    private findRelatedThoughts;
    private injectIntoMesh;
    private getCurrentMeshState;
    private detectStateChanges;
    private processStateChanges;
    private updateThoughtsFromMesh;
    private processSyncQueue;
    private syncThoughtsToMesh;
    private executeCoordination;
    private executeProtocol;
    private updateLatencyMetric;
    private getBridgeId;
    private diffArrays;
    private setupDAAIntegration;
    private handleDAAAgentSpawned;
    private handleDAAConsensus;
    private handleDAAMessage;
    private handleDAAAgentFault;
    /**
     * Initialize a DAA swarm and integrate with neural mesh
     */
    initializeIntegratedSwarm(swarmConfig: any): Promise<string>;
    /**
     * Spawn DAA agent and sync with neural mesh
     */
    spawnIntegratedAgent(swarmId: string, agentConfig: any): Promise<string>;
    /**
     * Get integrated status (neural bridge + DAA)
     */
    getIntegratedStatus(): any;
}
export declare const neuralBridge: KimiNeuralBridge;
export declare function initializeNeuralBridge(): Promise<KimiNeuralBridge>;
export declare function getNeuralBridgeInstance(): KimiNeuralBridge;
//# sourceMappingURL=kimi-neural-bridge.d.ts.map