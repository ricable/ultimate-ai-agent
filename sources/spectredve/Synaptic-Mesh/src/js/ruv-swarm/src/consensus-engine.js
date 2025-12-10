/**
 * Consensus Engine - Implements consensus-based decision making for swarm intelligence
 * 
 * Provides Byzantine fault-tolerant consensus algorithms for distributed decision making
 * in the Synaptic Neural Mesh swarm. Supports multiple consensus protocols and adaptive
 * selection based on network conditions.
 */

import EventEmitter from 'events';

/**
 * Consensus protocols supported by the engine
 */
export const CONSENSUS_PROTOCOLS = {
    PBFT: 'pbft',           // Practical Byzantine Fault Tolerance
    RAFT: 'raft',           // Raft consensus algorithm
    TENDERMINT: 'tendermint', // Tendermint BFT consensus
    SWARM_BFT: 'swarm_bft',  // Custom swarm-optimized BFT
    NEURAL_CONSENSUS: 'neural_consensus' // Neural network based consensus
};

/**
 * Message types for consensus protocol
 */
export const MESSAGE_TYPES = {
    PROPOSAL: 'proposal',
    PREPARE: 'prepare',
    COMMIT: 'commit',
    VOTE: 'vote',
    NEW_VIEW: 'new_view',
    HEARTBEAT: 'heartbeat',
    NEURAL_SIGNAL: 'neural_signal'
};

/**
 * Node states in consensus protocol
 */
export const NODE_STATES = {
    FOLLOWER: 'follower',
    CANDIDATE: 'candidate',
    LEADER: 'leader',
    FAULTY: 'faulty',
    OFFLINE: 'offline'
};

/**
 * Consensus Engine for distributed decision making
 */
export class ConsensusEngine extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            protocol: config.protocol || CONSENSUS_PROTOCOLS.SWARM_BFT,
            nodeId: config.nodeId || this.generateNodeId(),
            faultTolerance: config.faultTolerance || 0.33, // Up to 33% Byzantine faults
            timeout: config.timeout || 5000, // 5 second timeout
            heartbeatInterval: config.heartbeatInterval || 1000, // 1 second heartbeat
            batchSize: config.batchSize || 100, // Batch size for proposals
            adaptiveSelection: config.adaptiveSelection !== false,
            neuralWeight: config.neuralWeight || 0.3, // Weight for neural consensus
            ...config
        };

        // Consensus state
        this.currentView = 0;
        this.currentTerm = 0;
        this.currentLeader = null;
        this.nodeState = NODE_STATES.FOLLOWER;
        this.votedFor = null;
        this.commitIndex = 0;
        this.lastApplied = 0;

        // Network state
        this.nodes = new Map();
        this.connectedNodes = new Set();
        this.suspectedNodes = new Set();
        this.failedNodes = new Set();

        // Message handling
        this.messageQueue = [];
        this.pendingProposals = new Map();
        this.votes = new Map();
        this.prepareResponses = new Map();
        this.commitResponses = new Map();

        // Neural consensus
        this.neuralSignals = new Map();
        this.neuralWeights = new Map();
        this.learningRate = 0.01;

        // Performance metrics
        this.metrics = {
            consensusLatency: 0,
            throughput: 0,
            faultTolerance: 0,
            networkPartitions: 0,
            adaptations: 0,
            neuralAccuracy: 0
        };

        // Timers
        this.heartbeatTimer = null;
        this.timeoutTimer = null;
        this.leaderElectionTimer = null;

        this.initialized = false;
    }

    /**
     * Initialize the consensus engine
     */
    async initialize() {
        if (this.initialized) {
            return;
        }

        console.log(`🗳️ Initializing Consensus Engine (${this.config.protocol})...`);

        // Initialize protocol-specific state
        await this.initializeProtocol();

        // Start heartbeat
        this.startHeartbeat();

        // Start leader election if needed
        if (this.requiresLeaderElection()) {
            this.startLeaderElection();
        }

        this.initialized = true;
        this.emit('initialized');

        console.log(`✅ Consensus Engine initialized with protocol ${this.config.protocol}`);
    }

    /**
     * Initialize protocol-specific components
     */
    async initializeProtocol() {
        switch (this.config.protocol) {
            case CONSENSUS_PROTOCOLS.PBFT:
                await this.initializePBFT();
                break;
            case CONSENSUS_PROTOCOLS.RAFT:
                await this.initializeRaft();
                break;
            case CONSENSUS_PROTOCOLS.TENDERMINT:
                await this.initializeTendermint();
                break;
            case CONSENSUS_PROTOCOLS.SWARM_BFT:
                await this.initializeSwarmBFT();
                break;
            case CONSENSUS_PROTOCOLS.NEURAL_CONSENSUS:
                await this.initializeNeuralConsensus();
                break;
            default:
                throw new Error(`Unknown consensus protocol: ${this.config.protocol}`);
        }
    }

    /**
     * Initialize PBFT (Practical Byzantine Fault Tolerance)
     */
    async initializePBFT() {
        this.pbftState = {
            phase: 'prepare',
            sequenceNumber: 0,
            checkpoint: 0,
            lowWatermark: 0,
            highWatermark: 100
        };
    }

    /**
     * Initialize Raft consensus
     */
    async initializeRaft() {
        this.raftState = {
            log: [],
            nextIndex: new Map(),
            matchIndex: new Map(),
            electionTimeout: this.randomElectionTimeout(),
            lastLogIndex: 0,
            lastLogTerm: 0
        };
    }

    /**
     * Initialize Tendermint BFT
     */
    async initializeTendermint() {
        this.tendermintState = {
            height: 1,
            round: 0,
            step: 'propose',
            proposal: null,
            lockedValue: null,
            lockedRound: -1,
            validValue: null,
            validRound: -1
        };
    }

    /**
     * Initialize Swarm BFT (custom swarm-optimized protocol)
     */
    async initializeSwarmBFT() {
        this.swarmBFTState = {
            swarmId: this.config.swarmId || 'default',
            phase: 'listening',
            reputation: new Map(),
            trustScores: new Map(),
            adaptiveThreshold: 0.67, // 2/3 majority initially
            emergencyMode: false
        };

        // Initialize reputation for known nodes
        for (const nodeId of this.connectedNodes) {
            this.swarmBFTState.reputation.set(nodeId, 0.5);
            this.swarmBFTState.trustScores.set(nodeId, 0.5);
        }
    }

    /**
     * Initialize Neural Consensus (AI-driven consensus)
     */
    async initializeNeuralConsensus() {
        this.neuralConsensusState = {
            model: this.createNeuralModel(),
            trainingData: [],
            predictions: new Map(),
            confidence: new Map(),
            neuralVotes: new Map()
        };
    }

    /**
     * Add a node to the consensus network
     */
    addNode(nodeId, nodeInfo = {}) {
        this.nodes.set(nodeId, {
            id: nodeId,
            state: NODE_STATES.FOLLOWER,
            lastSeen: Date.now(),
            reputation: 0.5,
            trustScore: 0.5,
            performance: {
                latency: 100,
                availability: 1.0,
                accuracy: 0.5
            },
            ...nodeInfo
        });

        this.connectedNodes.add(nodeId);

        // Update protocol-specific state
        if (this.swarmBFTState) {
            this.swarmBFTState.reputation.set(nodeId, 0.5);
            this.swarmBFTState.trustScores.set(nodeId, 0.5);
        }

        if (this.raftState) {
            this.raftState.nextIndex.set(nodeId, this.raftState.log.length + 1);
            this.raftState.matchIndex.set(nodeId, 0);
        }

        this.emit('nodeAdded', { nodeId, nodeInfo });
    }

    /**
     * Remove a node from the consensus network
     */
    removeNode(nodeId) {
        this.nodes.delete(nodeId);
        this.connectedNodes.delete(nodeId);
        this.suspectedNodes.delete(nodeId);
        this.failedNodes.add(nodeId);

        // Clean up protocol-specific state
        if (this.swarmBFTState) {
            this.swarmBFTState.reputation.delete(nodeId);
            this.swarmBFTState.trustScores.delete(nodeId);
        }

        if (this.raftState) {
            this.raftState.nextIndex.delete(nodeId);
            this.raftState.matchIndex.delete(nodeId);
        }

        this.votes.delete(nodeId);
        this.neuralSignals.delete(nodeId);

        this.emit('nodeRemoved', { nodeId });

        // Check if we need to adapt consensus threshold
        if (this.config.adaptiveSelection) {
            this.adaptConsensusThreshold();
        }
    }

    /**
     * Propose a decision for consensus
     */
    async proposeDecision(decision, metadata = {}) {
        if (!this.canPropose()) {
            throw new Error(`Node cannot propose in current state: ${this.nodeState}`);
        }

        const proposal = {
            id: this.generateProposalId(),
            proposer: this.config.nodeId,
            decision,
            metadata,
            timestamp: Date.now(),
            view: this.currentView,
            term: this.currentTerm
        };

        console.log(`📋 Proposing decision: ${proposal.id}`);

        // Execute protocol-specific proposal logic
        switch (this.config.protocol) {
            case CONSENSUS_PROTOCOLS.PBFT:
                return await this.proposePBFT(proposal);
            case CONSENSUS_PROTOCOLS.RAFT:
                return await this.proposeRaft(proposal);
            case CONSENSUS_PROTOCOLS.TENDERMINT:
                return await this.proposeTendermint(proposal);
            case CONSENSUS_PROTOCOLS.SWARM_BFT:
                return await this.proposeSwarmBFT(proposal);
            case CONSENSUS_PROTOCOLS.NEURAL_CONSENSUS:
                return await this.proposeNeuralConsensus(proposal);
            default:
                throw new Error(`Proposal not implemented for protocol: ${this.config.protocol}`);
        }
    }

    /**
     * PBFT proposal implementation
     */
    async proposePBFT(proposal) {
        this.pbftState.sequenceNumber++;
        proposal.sequenceNumber = this.pbftState.sequenceNumber;

        this.pendingProposals.set(proposal.id, proposal);

        // Broadcast prepare message
        const prepareMessage = {
            type: MESSAGE_TYPES.PREPARE,
            proposal,
            sender: this.config.nodeId,
            view: this.currentView
        };

        await this.broadcastMessage(prepareMessage);

        // Start timeout for this proposal
        this.startProposalTimeout(proposal.id);

        return proposal.id;
    }

    /**
     * Raft proposal implementation
     */
    async proposeRaft(proposal) {
        if (this.nodeState !== NODE_STATES.LEADER) {
            throw new Error('Only leaders can propose in Raft');
        }

        // Add to log
        const logEntry = {
            term: this.currentTerm,
            index: this.raftState.log.length + 1,
            proposal,
            committed: false
        };

        this.raftState.log.push(logEntry);
        this.raftState.lastLogIndex = logEntry.index;
        this.raftState.lastLogTerm = logEntry.term;

        // Send append entries to followers
        await this.sendAppendEntries();

        return proposal.id;
    }

    /**
     * Swarm BFT proposal implementation
     */
    async proposeSwarmBFT(proposal) {
        // Calculate trust-weighted threshold
        const threshold = this.calculateAdaptiveThreshold();
        proposal.threshold = threshold;

        this.pendingProposals.set(proposal.id, proposal);

        // Include neural signals if available
        if (this.neuralConsensusState) {
            const neuralPrediction = await this.getNeuralPrediction(proposal);
            proposal.neuralPrediction = neuralPrediction;
        }

        // Broadcast proposal with trust scores
        const proposalMessage = {
            type: MESSAGE_TYPES.PROPOSAL,
            proposal,
            sender: this.config.nodeId,
            trustScore: this.getTrustScore(this.config.nodeId),
            reputation: this.getReputation(this.config.nodeId)
        };

        await this.broadcastMessage(proposalMessage);

        this.startProposalTimeout(proposal.id);

        return proposal.id;
    }

    /**
     * Neural consensus proposal implementation
     */
    async proposeNeuralConsensus(proposal) {
        // Generate neural prediction for the proposal
        const neuralInput = this.encodeProposalForNeural(proposal);
        const prediction = await this.neuralConsensusState.model.predict(neuralInput);
        
        proposal.neuralPrediction = prediction;
        proposal.confidence = prediction.confidence;

        this.pendingProposals.set(proposal.id, proposal);

        // Broadcast with neural signals
        const neuralMessage = {
            type: MESSAGE_TYPES.NEURAL_SIGNAL,
            proposal,
            neuralData: {
                prediction: prediction.value,
                confidence: prediction.confidence,
                features: neuralInput
            },
            sender: this.config.nodeId
        };

        await this.broadcastMessage(neuralMessage);

        return proposal.id;
    }

    /**
     * Process incoming consensus message
     */
    async processMessage(message) {
        if (!this.validateMessage(message)) {
            console.warn('⚠️ Invalid consensus message received');
            return;
        }

        // Update node's last seen time
        if (message.sender && this.nodes.has(message.sender)) {
            const node = this.nodes.get(message.sender);
            node.lastSeen = Date.now();
        }

        // Route to protocol-specific handler
        switch (this.config.protocol) {
            case CONSENSUS_PROTOCOLS.PBFT:
                await this.processPBFTMessage(message);
                break;
            case CONSENSUS_PROTOCOLS.RAFT:
                await this.processRaftMessage(message);
                break;
            case CONSENSUS_PROTOCOLS.SWARM_BFT:
                await this.processSwarmBFTMessage(message);
                break;
            case CONSENSUS_PROTOCOLS.NEURAL_CONSENSUS:
                await this.processNeuralConsensusMessage(message);
                break;
        }
    }

    /**
     * Process PBFT message
     */
    async processPBFTMessage(message) {
        switch (message.type) {
            case MESSAGE_TYPES.PREPARE:
                await this.handlePBFTPrepare(message);
                break;
            case MESSAGE_TYPES.COMMIT:
                await this.handlePBFTCommit(message);
                break;
        }
    }

    /**
     * Process Swarm BFT message
     */
    async processSwarmBFTMessage(message) {
        // Update reputation based on message validity
        this.updateReputation(message.sender, message);

        switch (message.type) {
            case MESSAGE_TYPES.PROPOSAL:
                await this.handleSwarmBFTProposal(message);
                break;
            case MESSAGE_TYPES.VOTE:
                await this.handleSwarmBFTVote(message);
                break;
        }
    }

    /**
     * Handle Swarm BFT proposal
     */
    async handleSwarmBFTProposal(message) {
        const { proposal } = message;
        
        // Validate proposal based on trust score
        const senderTrust = this.getTrustScore(message.sender);
        if (senderTrust < 0.3) {
            console.warn(`⚠️ Rejecting proposal from untrusted node: ${message.sender}`);
            return;
        }

        // Evaluate proposal with neural assistance if available
        let neuralScore = 0.5;
        if (proposal.neuralPrediction) {
            neuralScore = proposal.neuralPrediction.confidence || 0.5;
        }

        // Calculate weighted vote
        const vote = this.calculateSwarmBFTVote(proposal, senderTrust, neuralScore);

        // Send vote
        const voteMessage = {
            type: MESSAGE_TYPES.VOTE,
            proposalId: proposal.id,
            vote,
            sender: this.config.nodeId,
            trustScore: this.getTrustScore(this.config.nodeId),
            neuralScore
        };

        await this.broadcastMessage(voteMessage);
    }

    /**
     * Handle Swarm BFT vote
     */
    async handleSwarmBFTVote(message) {
        const { proposalId, vote, trustScore, neuralScore } = message;
        
        if (!this.votes.has(proposalId)) {
            this.votes.set(proposalId, new Map());
        }

        const proposalVotes = this.votes.get(proposalId);
        proposalVotes.set(message.sender, {
            vote,
            trustScore: trustScore || 0.5,
            neuralScore: neuralScore || 0.5,
            timestamp: Date.now()
        });

        // Check if we have enough votes for consensus
        await this.checkSwarmBFTConsensus(proposalId);
    }

    /**
     * Check Swarm BFT consensus
     */
    async checkSwarmBFTConsensus(proposalId) {
        const proposal = this.pendingProposals.get(proposalId);
        if (!proposal) {
            return;
        }

        const votes = this.votes.get(proposalId);
        if (!votes) {
            return;
        }

        // Calculate trust-weighted consensus
        let totalWeight = 0;
        let positiveWeight = 0;
        let neuralWeight = 0;
        let positiveNeuralWeight = 0;

        for (const [nodeId, voteData] of votes) {
            const weight = voteData.trustScore;
            const neuralContribution = voteData.neuralScore * this.config.neuralWeight;
            
            totalWeight += weight;
            neuralWeight += neuralContribution;
            
            if (voteData.vote > 0.5) {
                positiveWeight += weight;
                positiveNeuralWeight += neuralContribution;
            }
        }

        // Calculate consensus scores
        const trustConsensus = positiveWeight / totalWeight;
        const neuralConsensus = neuralWeight > 0 ? positiveNeuralWeight / neuralWeight : 0.5;
        
        // Combined consensus with neural weighting
        const combinedConsensus = (
            trustConsensus * (1 - this.config.neuralWeight) +
            neuralConsensus * this.config.neuralWeight
        );

        const threshold = proposal.threshold || this.swarmBFTState.adaptiveThreshold;

        if (combinedConsensus >= threshold && votes.size >= this.getMinimumVotes()) {
            await this.commitDecision(proposal, {
                trustConsensus,
                neuralConsensus,
                combinedConsensus,
                totalVotes: votes.size
            });
        }
    }

    /**
     * Commit a consensus decision
     */
    async commitDecision(proposal, consensusInfo = {}) {
        console.log(`✅ Consensus reached for proposal: ${proposal.id}`);

        // Remove from pending
        this.pendingProposals.delete(proposal.id);
        this.votes.delete(proposal.id);

        // Update metrics
        this.updateConsensusMetrics(proposal, consensusInfo);

        // Learn from the decision if neural consensus is enabled
        if (this.neuralConsensusState) {
            await this.learnFromDecision(proposal, consensusInfo);
        }

        // Update reputation of participating nodes
        this.updateParticipantReputations(proposal.id, true);

        // Emit consensus event
        this.emit('consensusReached', {
            proposal,
            consensusInfo,
            timestamp: Date.now()
        });

        return proposal;
    }

    /**
     * Calculate Swarm BFT vote based on multiple factors
     */
    calculateSwarmBFTVote(proposal, senderTrust, neuralScore) {
        // Base vote calculation
        let vote = 0.5;

        // Factor in sender trust
        vote += (senderTrust - 0.5) * 0.3;

        // Factor in neural prediction
        vote += (neuralScore - 0.5) * 0.2;

        // Factor in proposal metadata (simplified)
        if (proposal.metadata.priority === 'high') {
            vote += 0.1;
        }

        // Factor in current network conditions
        const networkHealth = this.calculateNetworkHealth();
        vote += (networkHealth - 0.5) * 0.1;

        // Apply emergency mode adjustments
        if (this.swarmBFTState.emergencyMode) {
            vote += 0.2; // More lenient in emergency
        }

        return Math.max(0, Math.min(1, vote));
    }

    /**
     * Calculate adaptive threshold based on network conditions
     */
    calculateAdaptiveThreshold() {
        const totalNodes = this.connectedNodes.size;
        const faultyNodes = this.failedNodes.size + this.suspectedNodes.size;
        const faultRate = totalNodes > 0 ? faultyNodes / totalNodes : 0;

        // Increase threshold if more faults detected
        let threshold = 0.67; // Base 2/3 majority
        
        if (faultRate > 0.2) {
            threshold = 0.75; // Require 3/4 majority
        } else if (faultRate > 0.1) {
            threshold = 0.70; // Require 70% majority
        }

        // Adjust for network size
        if (totalNodes < 4) {
            threshold = 1.0; // Require unanimity for small networks
        } else if (totalNodes > 100) {
            threshold = 0.60; // Allow 60% for large networks
        }

        return threshold;
    }

    /**
     * Adapt consensus threshold based on network conditions
     */
    adaptConsensusThreshold() {
        if (!this.swarmBFTState) {
            return;
        }

        const newThreshold = this.calculateAdaptiveThreshold();
        const oldThreshold = this.swarmBFTState.adaptiveThreshold;

        if (Math.abs(newThreshold - oldThreshold) > 0.05) {
            this.swarmBFTState.adaptiveThreshold = newThreshold;
            this.metrics.adaptations++;

            console.log(`🔄 Adapted consensus threshold: ${oldThreshold.toFixed(2)} → ${newThreshold.toFixed(2)}`);

            this.emit('thresholdAdapted', {
                oldThreshold,
                newThreshold,
                reason: 'network_conditions'
            });
        }
    }

    /**
     * Update reputation based on node behavior
     */
    updateReputation(nodeId, message) {
        if (!this.swarmBFTState || !this.nodes.has(nodeId)) {
            return;
        }

        const node = this.nodes.get(nodeId);
        let reputationDelta = 0;

        // Positive reputation for valid messages
        if (this.validateMessage(message)) {
            reputationDelta += 0.01;
        }

        // Negative reputation for suspicious behavior
        const messageAge = Date.now() - message.timestamp;
        if (messageAge > 30000) { // Message older than 30 seconds
            reputationDelta -= 0.05;
        }

        // Update reputation with decay
        const currentReputation = this.swarmBFTState.reputation.get(nodeId) || 0.5;
        const newReputation = Math.max(0, Math.min(1, currentReputation + reputationDelta));
        
        this.swarmBFTState.reputation.set(nodeId, newReputation);
        node.reputation = newReputation;

        // Update trust score (weighted average of reputation and performance)
        const performance = (node.performance.availability + node.performance.accuracy) / 2;
        const trustScore = (newReputation * 0.7 + performance * 0.3);
        
        this.swarmBFTState.trustScores.set(nodeId, trustScore);
        node.trustScore = trustScore;
    }

    /**
     * Create neural model for neural consensus
     */
    createNeuralModel() {
        // Simplified neural model for consensus decisions
        return {
            weights: Array(10).fill(0).map(() => Math.random() - 0.5),
            bias: Math.random() - 0.5,
            
            predict: async function(input) {
                let sum = this.bias;
                for (let i = 0; i < Math.min(input.length, this.weights.length); i++) {
                    sum += input[i] * this.weights[i];
                }
                
                const output = 1 / (1 + Math.exp(-sum)); // Sigmoid activation
                return {
                    value: output,
                    confidence: Math.abs(output - 0.5) * 2 // 0-1 confidence
                };
            },
            
            train: function(input, target, learningRate = 0.01) {
                const prediction = this.predict(input);
                const error = target - prediction.value;
                
                // Update weights
                for (let i = 0; i < Math.min(input.length, this.weights.length); i++) {
                    this.weights[i] += learningRate * error * input[i];
                }
                this.bias += learningRate * error;
            }
        };
    }

    /**
     * Encode proposal for neural processing
     */
    encodeProposalForNeural(proposal) {
        const features = [
            proposal.timestamp / 1000000000, // Normalized timestamp
            Object.keys(proposal.decision).length / 10, // Decision complexity
            proposal.metadata.priority === 'high' ? 1 : 0,
            proposal.metadata.urgency || 0.5,
            this.currentView / 100, // Normalized view
            this.connectedNodes.size / 100, // Network size
            this.calculateNetworkHealth(),
            this.metrics.consensusLatency / 10000, // Normalized latency
            this.metrics.throughput / 100, // Normalized throughput
            Math.random() // Random noise for exploration
        ];
        
        return features;
    }

    /**
     * Learn from consensus decision
     */
    async learnFromDecision(proposal, consensusInfo) {
        if (!this.neuralConsensusState) {
            return;
        }

        const input = this.encodeProposalForNeural(proposal);
        const target = consensusInfo.combinedConsensus || 0.5;

        // Train the neural model
        this.neuralConsensusState.model.train(input, target, this.learningRate);

        // Store training data
        this.neuralConsensusState.trainingData.push({
            input,
            target,
            timestamp: Date.now()
        });

        // Keep training data size manageable
        if (this.neuralConsensusState.trainingData.length > 1000) {
            this.neuralConsensusState.trainingData = 
                this.neuralConsensusState.trainingData.slice(-500);
        }

        // Update neural accuracy metric
        const prediction = await this.neuralConsensusState.model.predict(input);
        const accuracy = 1 - Math.abs(prediction.value - target);
        this.metrics.neuralAccuracy = (this.metrics.neuralAccuracy * 0.9 + accuracy * 0.1);
    }

    /**
     * Get neural prediction for proposal
     */
    async getNeuralPrediction(proposal) {
        if (!this.neuralConsensusState) {
            return { value: 0.5, confidence: 0 };
        }

        const input = this.encodeProposalForNeural(proposal);
        return await this.neuralConsensusState.model.predict(input);
    }

    /**
     * Helper methods
     */
    generateNodeId() {
        return `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    generateProposalId() {
        return `proposal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    canPropose() {
        return this.nodeState === NODE_STATES.LEADER || 
               this.config.protocol === CONSENSUS_PROTOCOLS.PBFT ||
               this.config.protocol === CONSENSUS_PROTOCOLS.SWARM_BFT;
    }

    requiresLeaderElection() {
        return this.config.protocol === CONSENSUS_PROTOCOLS.RAFT;
    }

    getTrustScore(nodeId) {
        if (this.swarmBFTState?.trustScores.has(nodeId)) {
            return this.swarmBFTState.trustScores.get(nodeId);
        }
        return 0.5;
    }

    getReputation(nodeId) {
        if (this.swarmBFTState?.reputation.has(nodeId)) {
            return this.swarmBFTState.reputation.get(nodeId);
        }
        return 0.5;
    }

    getMinimumVotes() {
        const totalNodes = this.connectedNodes.size;
        return Math.max(1, Math.floor(totalNodes * 0.5) + 1);
    }

    calculateNetworkHealth() {
        const totalNodes = this.nodes.size;
        const healthyNodes = totalNodes - this.failedNodes.size - this.suspectedNodes.size;
        return totalNodes > 0 ? healthyNodes / totalNodes : 1.0;
    }

    updateConsensusMetrics(proposal, consensusInfo) {
        const latency = Date.now() - proposal.timestamp;
        this.metrics.consensusLatency = this.metrics.consensusLatency * 0.9 + latency * 0.1;
        this.metrics.throughput = this.metrics.throughput * 0.9 + 1000 / latency * 0.1;
        this.metrics.faultTolerance = this.calculateNetworkHealth();
    }

    updateParticipantReputations(proposalId, success) {
        const votes = this.votes.get(proposalId);
        if (!votes) return;

        for (const [nodeId, voteData] of votes) {
            if (this.nodes.has(nodeId)) {
                const node = this.nodes.get(nodeId);
                const delta = success ? 0.02 : -0.01;
                node.reputation = Math.max(0, Math.min(1, node.reputation + delta));
            }
        }
    }

    validateMessage(message) {
        return message && 
               message.type && 
               message.sender && 
               message.timestamp &&
               (Date.now() - message.timestamp) < 60000; // Not older than 1 minute
    }

    startHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }

        this.heartbeatTimer = setInterval(async () => {
            const heartbeat = {
                type: MESSAGE_TYPES.HEARTBEAT,
                sender: this.config.nodeId,
                timestamp: Date.now(),
                state: this.nodeState,
                view: this.currentView,
                term: this.currentTerm
            };

            await this.broadcastMessage(heartbeat);
            
            // Check for failed nodes
            this.detectFailedNodes();
        }, this.config.heartbeatInterval);
    }

    startProposalTimeout(proposalId) {
        setTimeout(() => {
            if (this.pendingProposals.has(proposalId)) {
                console.warn(`⏰ Proposal timeout: ${proposalId}`);
                this.pendingProposals.delete(proposalId);
                this.votes.delete(proposalId);
                
                this.emit('proposalTimeout', { proposalId });
            }
        }, this.config.timeout);
    }

    detectFailedNodes() {
        const now = Date.now();
        const timeout = this.config.heartbeatInterval * 3;

        for (const [nodeId, node] of this.nodes) {
            if (now - node.lastSeen > timeout) {
                if (this.connectedNodes.has(nodeId)) {
                    this.connectedNodes.delete(nodeId);
                    this.suspectedNodes.add(nodeId);
                    console.warn(`⚠️ Node suspected failed: ${nodeId}`);
                    
                    this.emit('nodeSuspected', { nodeId });
                }
            }
        }
    }

    async broadcastMessage(message) {
        // This would be implemented by the transport layer
        this.emit('broadcastMessage', message);
    }

    startLeaderElection() {
        // Simplified leader election for Raft
        if (this.config.protocol === CONSENSUS_PROTOCOLS.RAFT) {
            this.leaderElectionTimer = setTimeout(() => {
                if (this.nodeState === NODE_STATES.FOLLOWER && !this.currentLeader) {
                    this.nodeState = NODE_STATES.CANDIDATE;
                    this.currentTerm++;
                    this.votedFor = this.config.nodeId;
                    
                    // Request votes (simplified)
                    this.emit('requestVotes', {
                        term: this.currentTerm,
                        candidateId: this.config.nodeId
                    });
                }
            }, this.randomElectionTimeout());
        }
    }

    randomElectionTimeout() {
        return 150 + Math.random() * 150; // 150-300ms
    }

    /**
     * Get current consensus metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            connectedNodes: this.connectedNodes.size,
            suspectedNodes: this.suspectedNodes.size,
            failedNodes: this.failedNodes.size,
            pendingProposals: this.pendingProposals.size,
            currentView: this.currentView,
            currentTerm: this.currentTerm,
            nodeState: this.nodeState,
            adaptiveThreshold: this.swarmBFTState?.adaptiveThreshold || 0.67
        };
    }

    /**
     * Stop the consensus engine
     */
    async stop() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }

        if (this.timeoutTimer) {
            clearTimeout(this.timeoutTimer);
            this.timeoutTimer = null;
        }

        if (this.leaderElectionTimer) {
            clearTimeout(this.leaderElectionTimer);
            this.leaderElectionTimer = null;
        }

        this.initialized = false;
        this.emit('stopped');
        
        console.log('🔻 Consensus Engine stopped');
    }
}

export default ConsensusEngine;