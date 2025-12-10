/**
 * DAA MCP Bridge - Dynamic Agent Architecture integration with MCP
 *
 * Implements DAA swarm coordination through MCP tools:
 * - Agent lifecycle management
 * - Consensus mechanisms
 * - Resource allocation
 * - Inter-agent communication
 * - Fault tolerance and recovery
 */
import { EventEmitter } from 'events';
export interface DAAAgent {
    id: string;
    type: 'coordinator' | 'worker' | 'specialist' | 'monitor';
    status: 'active' | 'busy' | 'idle' | 'failed' | 'terminated';
    capabilities: string[];
    resources: {
        cpu: number;
        memory: number;
        network: number;
    };
    performance: {
        tasksCompleted: number;
        successRate: number;
        averageLatency: number;
    };
    created: number;
    lastActivity: number;
}
export interface SwarmConfiguration {
    id: string;
    topology: 'mesh' | 'hierarchical' | 'ring' | 'star';
    maxAgents: number;
    strategy: 'parallel' | 'sequential' | 'adaptive' | 'balanced';
    consensus: {
        threshold: number;
        algorithm: 'raft' | 'pbft' | 'pos';
    };
    resources: {
        totalCpu: number;
        totalMemory: number;
        totalNetwork: number;
    };
}
export interface ConsensusProposal {
    id: string;
    proposer: string;
    content: any;
    votes: Map<string, 'approve' | 'reject' | 'abstain'>;
    threshold: number;
    timestamp: number;
    deadline: number;
    status: 'pending' | 'approved' | 'rejected' | 'expired';
}
export declare class DAAMCPBridge extends EventEmitter {
    private agents;
    private swarms;
    private proposals;
    private messageQueue;
    private isActive;
    private metrics;
    constructor();
    /**
     * Initialize a new DAA swarm
     */
    initializeSwarm(config: Partial<SwarmConfiguration>): Promise<string>;
    /**
     * Spawn a new DAA agent
     */
    spawnAgent(swarmId: string, agentConfig: Partial<DAAAgent>): Promise<string>;
    /**
     * Terminate an agent
     */
    terminateAgent(agentId: string): Promise<void>;
    /**
     * Send message between agents
     */
    sendMessage(fromAgent: string, toAgent: string, message: any): Promise<void>;
    /**
     * Initiate consensus mechanism
     */
    initiateConsensus(swarmId: string, proposerId: string, content: any): Promise<string>;
    /**
     * Cast vote for consensus proposal
     */
    castVote(proposalId: string, agentId: string, vote: 'approve' | 'reject' | 'abstain'): Promise<void>;
    /**
     * Allocate resources to agents
     */
    allocateResources(swarmId: string, allocations: Map<string, any>): Promise<void>;
    /**
     * Handle agent fault and recovery
     */
    handleFault(agentId: string, faultType: string): Promise<void>;
    /**
     * Get swarm status and metrics
     */
    getSwarmStatus(swarmId: string): any;
    /**
     * Get bridge metrics
     */
    getMetrics(): any;
    private setupEventHandlers;
    private getSwarmAgents;
    private checkConsensus;
    private calculateResourceUsage;
    private updateResponseTimeMetric;
}
export declare const daaBridge: DAAMCPBridge;
export declare function initializeDABridge(): Promise<DAAMCPBridge>;
export declare function getDAABridgeInstance(): DAAMCPBridge;
//# sourceMappingURL=daa-mcp-bridge.d.ts.map