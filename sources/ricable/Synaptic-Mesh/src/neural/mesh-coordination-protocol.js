/**
 * Neural Mesh Coordination Protocol
 * Implements distributed neural decision protocols for agent coordination
 */

export class NeuralMeshCoordinator {
    constructor(options = {}) {
        this.agentId = options.agentId || 'coordinator';
        this.meshTopology = options.topology || 'hierarchical';
        this.coordinationMatrix = new Map();
        this.decisionThreshold = options.threshold || 0.7;
        this.consensusAlgorithm = options.consensus || 'weighted_voting';
        this.neuralWeights = new Map();
        this.adaptiveLearning = true;
        this.performanceHistory = [];
        
        // Neural coordination state
        this.synapticStrength = new Map();
        this.inhibitoryConnections = new Map();
        this.plasticityRate = 0.01;
        this.coordinationMemory = new WeakMap();
        
        // Performance tracking
        this.coordinationLatency = [];
        this.decisionAccuracy = [];
        this.networkCohesion = 0.0;
        
        console.log(`🧠 Neural Mesh Coordinator initialized for ${this.agentId}`);
    }
    
    /**
     * Register an agent in the neural mesh
     */
    async registerAgent(agentId, neuralNetwork, capabilities = []) {
        const agentProfile = {
            id: agentId,
            network: neuralNetwork,
            capabilities,
            timestamp: Date.now(),
            coordinationScore: 0.5,
            specializations: [],
            synapticConnections: new Set(),
            inhibitoryLevel: 0.1,
            plasticityFactor: 1.0,
            trust: 0.5,
            reliability: 0.5
        };
        
        this.coordinationMatrix.set(agentId, agentProfile);
        this.neuralWeights.set(agentId, this.initializeNeuralWeights(capabilities));
        this.synapticStrength.set(agentId, new Map());
        
        // Establish synaptic connections with existing agents
        await this.establishSynapticConnections(agentId);
        
        console.log(`🔗 Agent ${agentId} registered in neural mesh with ${capabilities.length} capabilities`);
        return agentProfile;
    }
    
    /**
     * Establish synaptic connections between agents
     */
    async establishSynapticConnections(newAgentId) {
        const newAgent = this.coordinationMatrix.get(newAgentId);
        if (!newAgent) return;
        
        for (const [existingAgentId, existingAgent] of this.coordinationMatrix.entries()) {
            if (existingAgentId === newAgentId) continue;
            
            // Calculate connection strength based on capability overlap
            const connectionStrength = this.calculateConnectionStrength(
                newAgent.capabilities,
                existingAgent.capabilities
            );
            
            if (connectionStrength > 0.3) {
                // Establish bidirectional synaptic connection
                this.synapticStrength.get(newAgentId).set(existingAgentId, connectionStrength);
                this.synapticStrength.get(existingAgentId).set(newAgentId, connectionStrength);
                
                newAgent.synapticConnections.add(existingAgentId);
                existingAgent.synapticConnections.add(newAgentId);
                
                console.log(`⚡ Synaptic connection established: ${newAgentId} ↔ ${existingAgentId} (strength: ${connectionStrength.toFixed(3)})`);
            }
        }
    }
    
    /**
     * Calculate synaptic connection strength between agents
     */
    calculateConnectionStrength(capabilities1, capabilities2) {
        if (capabilities1.length === 0 || capabilities2.length === 0) return 0.1;
        
        const intersection = capabilities1.filter(cap => capabilities2.includes(cap));
        const union = [...new Set([...capabilities1, ...capabilities2])];
        
        const jaccardSimilarity = intersection.length / union.length;
        const complementarity = this.calculateComplementarity(capabilities1, capabilities2);
        
        // Balanced strength considering both similarity and complementarity
        return (jaccardSimilarity * 0.6 + complementarity * 0.4);
    }
    
    /**
     * Calculate capability complementarity between agents
     */
    calculateComplementarity(capabilities1, capabilities2) {
        const uniqueToFirst = capabilities1.filter(cap => !capabilities2.includes(cap));
        const uniqueToSecond = capabilities2.filter(cap => !capabilities1.includes(cap));
        
        const totalUnique = uniqueToFirst.length + uniqueToSecond.length;
        const totalCapabilities = capabilities1.length + capabilities2.length;
        
        return totalUnique / Math.max(1, totalCapabilities);
    }
    
    /**
     * Initialize neural weights for an agent
     */
    initializeNeuralWeights(capabilities) {
        const weights = {
            decision: Math.random() * 0.5 + 0.5, // 0.5-1.0
            coordination: Math.random() * 0.5 + 0.5,
            specialization: capabilities.length > 0 ? Math.random() * 0.3 + 0.7 : 0.3,
            adaptability: Math.random() * 0.4 + 0.3,
            reliability: 0.5, // Starts neutral, adjusted based on performance
            innovation: Math.random() * 0.6 + 0.2
        };
        
        return weights;
    }
    
    /**
     * Coordinate decision making across the neural mesh
     */
    async coordinateDecision(task, requirements = {}, options = {}) {
        const startTime = Date.now();
        
        console.log(`🧠 Coordinating decision for task: ${task.id || 'unnamed'}`);
        
        // Phase 1: Neural activation propagation
        const activationMap = await this.propagateActivation(task, requirements);
        
        // Phase 2: Consensus building through synaptic communication
        const consensus = await this.buildConsensus(activationMap, requirements);
        
        // Phase 3: Decision synthesis with inhibitory control
        const decision = await this.synthesizeDecision(consensus, task, requirements);
        
        // Phase 4: Update synaptic strengths based on outcome
        if (options.updateWeights !== false) {
            await this.updateSynapticPlasticity(decision, task);
        }
        
        const coordinationTime = Date.now() - startTime;
        this.coordinationLatency.push(coordinationTime);
        
        console.log(`✅ Neural coordination completed in ${coordinationTime}ms with confidence ${decision.confidence.toFixed(3)}`);
        
        return {
            ...decision,
            coordinationTime,
            participatingAgents: consensus.participants,
            neuralActivity: activationMap
        };
    }
    
    /**
     * Propagate neural activation across the mesh
     */
    async propagateActivation(task, requirements) {
        const activationMap = new Map();
        const propagationWaves = [];
        
        // Initial activation based on task-agent matching
        for (const [agentId, agent] of this.coordinationMatrix.entries()) {
            const activation = await this.calculateInitialActivation(agent, task, requirements);
            activationMap.set(agentId, {
                initial: activation,
                current: activation,
                received: 0,
                propagated: 0
            });
        }
        
        // Propagate activation through synaptic connections
        for (let wave = 0; wave < 3; wave++) {
            const waveActivations = new Map();
            
            for (const [agentId, activation] of activationMap.entries()) {
                const synapticConnections = this.synapticStrength.get(agentId);
                if (!synapticConnections) continue;
                
                for (const [connectedAgentId, connectionStrength] of synapticConnections.entries()) {
                    const transmittedActivation = activation.current * connectionStrength * 0.3;
                    
                    if (!waveActivations.has(connectedAgentId)) {
                        waveActivations.set(connectedAgentId, 0);
                    }
                    waveActivations.set(connectedAgentId, 
                        waveActivations.get(connectedAgentId) + transmittedActivation);
                }
            }
            
            // Update activations with received signals
            for (const [agentId, receivedActivation] of waveActivations.entries()) {
                const current = activationMap.get(agentId);
                current.received += receivedActivation;
                current.current = Math.tanh(current.initial + current.received * 0.5);
                current.propagated = wave + 1;
            }
            
            propagationWaves.push(new Map(waveActivations));
        }
        
        console.log(`🌊 Neural activation propagated through ${propagationWaves.length} waves`);
        return { activations: activationMap, waves: propagationWaves };
    }
    
    /**
     * Calculate initial neural activation for an agent
     */
    async calculateInitialActivation(agent, task, requirements) {
        const weights = this.neuralWeights.get(agent.id);
        if (!weights) return 0.1;
        
        // Task-capability matching
        const capabilityMatch = this.calculateCapabilityMatch(agent.capabilities, requirements.requiredCapabilities || []);
        
        // Specialization relevance
        const specializationMatch = this.calculateSpecializationMatch(agent.specializations, task);
        
        // Historical performance factor
        const performanceFactor = this.getPerformanceFactor(agent.id, task.type);
        
        // Current load factor (inverse relationship)
        const loadFactor = 1.0 - Math.min(0.8, agent.currentLoad || 0);
        
        // Trust and reliability factors
        const trustFactor = weights.reliability * agent.trust;
        
        const activation = (
            capabilityMatch * weights.specialization * 0.4 +
            specializationMatch * weights.decision * 0.25 +
            performanceFactor * weights.reliability * 0.2 +
            loadFactor * weights.coordination * 0.1 +
            trustFactor * 0.05
        );
        
        return Math.max(0, Math.min(1, activation));
    }
    
    /**
     * Build consensus through synaptic communication
     */
    async buildConsensus(activationData, requirements) {
        const { activations } = activationData;
        const participants = [];
        const threshold = requirements.activationThreshold || this.decisionThreshold;
        
        // Identify highly activated agents
        for (const [agentId, activation] of activations.entries()) {
            if (activation.current > threshold) {
                const agent = this.coordinationMatrix.get(agentId);
                participants.push({
                    id: agentId,
                    activation: activation.current,
                    weight: this.calculateConsensusWeight(agent),
                    capabilities: agent.capabilities,
                    specializations: agent.specializations
                });
            }
        }
        
        // Sort by activation level
        participants.sort((a, b) => b.activation - a.activation);
        
        // Apply inhibitory control to prevent over-activation
        const inhibitedParticipants = this.applyInhibitoryControl(participants, requirements);
        
        // Calculate consensus strength
        const consensusStrength = this.calculateConsensusStrength(inhibitedParticipants);
        
        console.log(`🤝 Consensus built with ${inhibitedParticipants.length} agents (strength: ${consensusStrength.toFixed(3)})`);
        
        return {
            participants: inhibitedParticipants,
            strength: consensusStrength,
            activationThreshold: threshold,
            inhibitionApplied: participants.length !== inhibitedParticipants.length
        };
    }
    
    /**
     * Apply inhibitory control to prevent over-activation
     */
    applyInhibitoryControl(participants, requirements) {
        const maxParticipants = requirements.maxParticipants || Math.min(5, participants.length);
        const inhibitionThreshold = requirements.inhibitionThreshold || 0.8;
        
        // Apply global inhibition if too many agents are activated
        if (participants.length > maxParticipants) {
            // Keep only top N agents
            participants.splice(maxParticipants);
        }
        
        // Apply lateral inhibition between similar agents
        const inhibited = [];
        const used = new Set();
        
        for (const participant of participants) {
            if (used.has(participant.id)) continue;
            
            inhibited.push(participant);
            used.add(participant.id);
            
            // Inhibit similar agents with lower activation
            for (const other of participants) {
                if (used.has(other.id) || other.activation >= participant.activation) continue;
                
                const similarity = this.calculateAgentSimilarity(participant, other);
                if (similarity > inhibitionThreshold) {
                    used.add(other.id);
                    console.log(`🚫 Inhibiting ${other.id} due to similarity with ${participant.id} (${similarity.toFixed(3)})`);
                }
            }
        }
        
        return inhibited;
    }
    
    /**
     * Calculate similarity between two participating agents
     */
    calculateAgentSimilarity(agent1, agent2) {
        // Capability overlap
        const capOverlap = agent1.capabilities.filter(cap => 
            agent2.capabilities.includes(cap)
        ).length / Math.max(1, Math.max(agent1.capabilities.length, agent2.capabilities.length));
        
        // Specialization overlap
        const specOverlap = agent1.specializations.filter(spec => 
            agent2.specializations.includes(spec)
        ).length / Math.max(1, Math.max(agent1.specializations.length, agent2.specializations.length));
        
        return (capOverlap * 0.7 + specOverlap * 0.3);
    }
    
    /**
     * Synthesize final decision from consensus
     */
    async synthesizeDecision(consensus, task, requirements) {
        const { participants } = consensus;
        
        if (participants.length === 0) {
            return {
                decision: 'no_consensus',
                confidence: 0.0,
                reasoning: 'No agents met activation threshold',
                assignments: []
            };
        }
        
        // Weight votes by activation and reliability
        const weightedVotes = participants.map(p => {
            const agent = this.coordinationMatrix.get(p.id);
            const weights = this.neuralWeights.get(p.id);
            
            return {
                agentId: p.id,
                vote: this.generateAgentVote(agent, task, requirements),
                weight: p.activation * weights.reliability * p.weight,
                activation: p.activation
            };
        });
        
        // Apply consensus algorithm
        const decision = await this.applyConsensusAlgorithm(weightedVotes, consensus, requirements);
        
        // Calculate overall confidence
        const confidence = this.calculateDecisionConfidence(decision, consensus, weightedVotes);
        
        // Generate assignments
        const assignments = this.generateTaskAssignments(participants, task, decision);
        
        return {
            decision: decision.outcome,
            confidence,
            reasoning: decision.reasoning,
            assignments,
            consensusStrength: consensus.strength,
            participantCount: participants.length,
            weightedVotes: weightedVotes.map(v => ({ agentId: v.agentId, weight: v.weight, vote: v.vote }))
        };
    }
    
    /**
     * Generate agent vote for a task
     */
    generateAgentVote(agent, task, requirements) {
        const weights = this.neuralWeights.get(agent.id);
        const capabilities = agent.capabilities;
        
        // Capability-based confidence
        const capabilityConfidence = this.calculateCapabilityMatch(
            capabilities, 
            requirements.requiredCapabilities || []
        );
        
        // Load-based availability
        const availability = 1.0 - Math.min(0.9, agent.currentLoad || 0);
        
        // Historical success rate for similar tasks
        const successRate = this.getSuccessRate(agent.id, task.type) || 0.5;
        
        // Generate vote
        const voteStrength = (
            capabilityConfidence * weights.specialization * 0.5 +
            availability * weights.coordination * 0.3 +
            successRate * weights.reliability * 0.2
        );
        
        return {
            participate: voteStrength > 0.6,
            confidence: voteStrength,
            estimatedEffort: this.estimateEffort(agent, task),
            suggestedRole: this.suggestRole(agent, task, requirements)
        };
    }
    
    /**
     * Apply consensus algorithm to weighted votes
     */
    async applyConsensusAlgorithm(weightedVotes, consensus, requirements) {
        switch (this.consensusAlgorithm) {
            case 'weighted_voting':
                return this.weightedVotingConsensus(weightedVotes, requirements);
            case 'neural_majority':
                return this.neuralMajorityConsensus(weightedVotes, consensus);
            case 'capability_based':
                return this.capabilityBasedConsensus(weightedVotes, requirements);
            default:
                return this.weightedVotingConsensus(weightedVotes, requirements);
        }
    }
    
    /**
     * Weighted voting consensus algorithm
     */
    weightedVotingConsensus(weightedVotes, requirements) {
        let totalWeight = 0;
        let participationWeight = 0;
        let avgConfidence = 0;
        let avgEffort = 0;
        
        for (const vote of weightedVotes) {
            totalWeight += vote.weight;
            if (vote.vote.participate) {
                participationWeight += vote.weight;
            }
            avgConfidence += vote.vote.confidence * vote.weight;
            avgEffort += vote.vote.estimatedEffort * vote.weight;
        }
        
        avgConfidence /= totalWeight;
        avgEffort /= totalWeight;
        const participationRatio = participationWeight / totalWeight;
        
        const outcome = participationRatio > 0.5 ? 'proceed' : 'defer';
        const reasoning = `Weighted voting: ${(participationRatio * 100).toFixed(1)}% participation by weight`;
        
        return {
            outcome,
            reasoning,
            participationRatio,
            averageConfidence: avgConfidence,
            averageEffort: avgEffort
        };
    }
    
    /**
     * Update synaptic plasticity based on outcomes
     */
    async updateSynapticPlasticity(decision, task) {
        if (!this.adaptiveLearning) return;
        
        const outcome = decision.confidence > 0.7 ? 'success' : 'failure';
        const participants = decision.assignments.map(a => a.agentId);
        
        for (const agentId of participants) {
            const agent = this.coordinationMatrix.get(agentId);
            const weights = this.neuralWeights.get(agentId);
            
            if (outcome === 'success') {
                // Strengthen successful patterns
                weights.decision = Math.min(1.0, weights.decision + this.plasticityRate);
                weights.coordination = Math.min(1.0, weights.coordination + this.plasticityRate * 0.5);
                agent.coordinationScore = Math.min(1.0, agent.coordinationScore + 0.05);
                agent.trust = Math.min(1.0, agent.trust + 0.02);
                
                // Strengthen synaptic connections between successful participants
                this.strengthenConnections(participants, this.plasticityRate * 0.3);
            } else {
                // Weaken unsuccessful patterns
                weights.decision = Math.max(0.1, weights.decision - this.plasticityRate * 0.5);
                agent.coordinationScore = Math.max(0.0, agent.coordinationScore - 0.03);
                agent.trust = Math.max(0.1, agent.trust - 0.01);
                
                // Weaken synaptic connections between unsuccessful participants
                this.weakenConnections(participants, this.plasticityRate * 0.2);
            }
        }
        
        // Update performance history
        this.performanceHistory.push({
            timestamp: Date.now(),
            taskType: task.type,
            participants,
            outcome,
            confidence: decision.confidence,
            coordinationTime: decision.coordinationTime
        });
        
        // Trim history to last 100 entries
        if (this.performanceHistory.length > 100) {
            this.performanceHistory.shift();
        }
        
        console.log(`🧠 Neural plasticity updated for ${outcome} outcome (${participants.length} participants)`);
    }
    
    /**
     * Strengthen synaptic connections between agents
     */
    strengthenConnections(agentIds, strengthIncrease) {
        for (let i = 0; i < agentIds.length; i++) {
            for (let j = i + 1; j < agentIds.length; j++) {
                const agentA = agentIds[i];
                const agentB = agentIds[j];
                
                const connectionsA = this.synapticStrength.get(agentA);
                const connectionsB = this.synapticStrength.get(agentB);
                
                if (connectionsA && connectionsA.has(agentB)) {
                    const currentStrength = connectionsA.get(agentB);
                    const newStrength = Math.min(1.0, currentStrength + strengthIncrease);
                    connectionsA.set(agentB, newStrength);
                    connectionsB.set(agentA, newStrength);
                }
            }
        }
    }
    
    /**
     * Weaken synaptic connections between agents
     */
    weakenConnections(agentIds, strengthDecrease) {
        for (let i = 0; i < agentIds.length; i++) {
            for (let j = i + 1; j < agentIds.length; j++) {
                const agentA = agentIds[i];
                const agentB = agentIds[j];
                
                const connectionsA = this.synapticStrength.get(agentA);
                const connectionsB = this.synapticStrength.get(agentB);
                
                if (connectionsA && connectionsA.has(agentB)) {
                    const currentStrength = connectionsA.get(agentB);
                    const newStrength = Math.max(0.1, currentStrength - strengthDecrease);
                    connectionsA.set(agentB, newStrength);
                    connectionsB.set(agentA, newStrength);
                }
            }
        }
    }
    
    /**
     * Get neural mesh statistics
     */
    getStatistics() {
        const agentCount = this.coordinationMatrix.size;
        const connectionCount = Array.from(this.synapticStrength.values())
            .reduce((total, connections) => total + connections.size, 0) / 2;
        
        const avgLatency = this.coordinationLatency.length > 0 
            ? this.coordinationLatency.reduce((a, b) => a + b, 0) / this.coordinationLatency.length 
            : 0;
            
        const avgAccuracy = this.decisionAccuracy.length > 0
            ? this.decisionAccuracy.reduce((a, b) => a + b, 0) / this.decisionAccuracy.length
            : 0;
            
        const networkDensity = agentCount > 1 ? (2 * connectionCount) / (agentCount * (agentCount - 1)) : 0;
        
        return {
            agentCount,
            connectionCount,
            networkDensity,
            averageLatency: avgLatency,
            averageAccuracy: avgAccuracy,
            performanceHistorySize: this.performanceHistory.length,
            plasticityRate: this.plasticityRate,
            consensusAlgorithm: this.consensusAlgorithm,
            topPerformers: this.getTopPerformers(5)
        };
    }
    
    /**
     * Get top performing agents
     */
    getTopPerformers(count = 5) {
        const performers = Array.from(this.coordinationMatrix.entries())
            .map(([id, agent]) => ({
                id,
                score: agent.coordinationScore,
                trust: agent.trust,
                reliability: this.neuralWeights.get(id)?.reliability || 0.5
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, count);
            
        return performers;
    }
    
    // Helper methods for calculations
    calculateCapabilityMatch(agentCaps, requiredCaps) {
        if (requiredCaps.length === 0) return 0.5;
        const matches = agentCaps.filter(cap => requiredCaps.includes(cap)).length;
        return matches / requiredCaps.length;
    }
    
    calculateSpecializationMatch(specializations, task) {
        if (!specializations || specializations.length === 0) return 0.3;
        const taskType = task.type || task.category;
        return specializations.includes(taskType) ? 0.8 : 0.2;
    }
    
    getPerformanceFactor(agentId, taskType) {
        const relevant = this.performanceHistory.filter(p => 
            p.participants.includes(agentId) && p.taskType === taskType
        );
        if (relevant.length === 0) return 0.5;
        
        const successCount = relevant.filter(p => p.outcome === 'success').length;
        return successCount / relevant.length;
    }
    
    calculateConsensusWeight(agent) {
        const weights = this.neuralWeights.get(agent.id);
        return (weights.decision + weights.coordination + weights.reliability) / 3;
    }
    
    calculateConsensusStrength(participants) {
        if (participants.length === 0) return 0;
        
        const totalActivation = participants.reduce((sum, p) => sum + p.activation, 0);
        const avgActivation = totalActivation / participants.length;
        const diversity = this.calculateDiversity(participants);
        
        return avgActivation * (0.7 + diversity * 0.3);
    }
    
    calculateDiversity(participants) {
        if (participants.length < 2) return 0;
        
        const allCapabilities = new Set();
        participants.forEach(p => p.capabilities.forEach(cap => allCapabilities.add(cap)));
        
        return Math.min(1, allCapabilities.size / (participants.length * 3)); // Normalize by expected diversity
    }
    
    calculateDecisionConfidence(decision, consensus, votes) {
        const consensusStrength = consensus.strength;
        const voteConfidence = votes.reduce((sum, v) => sum + v.vote.confidence * v.weight, 0) / 
                              votes.reduce((sum, v) => sum + v.weight, 0);
        
        return (consensusStrength * 0.6 + voteConfidence * 0.4);
    }
    
    generateTaskAssignments(participants, task, decision) {
        return participants.map(p => ({
            agentId: p.id,
            role: p.vote?.suggestedRole || 'contributor',
            effort: p.vote?.estimatedEffort || 0.5,
            capabilities: p.capabilities,
            priority: p.activation
        }));
    }
    
    estimateEffort(agent, task) {
        // Simple effort estimation based on agent capabilities and task complexity
        const capabilityMatch = this.calculateCapabilityMatch(
            agent.capabilities, 
            task.requiredCapabilities || []
        );
        const complexity = task.complexity || 0.5;
        
        return Math.max(0.1, complexity * (1.2 - capabilityMatch));
    }
    
    suggestRole(agent, task, requirements) {
        if (agent.capabilities.includes('leadership')) return 'leader';
        if (agent.capabilities.includes('coordination')) return 'coordinator';
        if (agent.capabilities.includes('analysis')) return 'analyzer';
        if (agent.capabilities.includes('implementation')) return 'implementer';
        return 'contributor';
    }
    
    getSuccessRate(agentId, taskType) {
        const history = this.performanceHistory.filter(p => 
            p.participants.includes(agentId) && p.taskType === taskType
        );
        if (history.length === 0) return null;
        
        const successes = history.filter(p => p.outcome === 'success').length;
        return successes / history.length;
    }
    
    neuralMajorityConsensus(weightedVotes, consensus) {
        // Implementation for neural majority consensus
        const participationCount = weightedVotes.filter(v => v.vote.participate).length;
        const majority = participationCount > weightedVotes.length / 2;
        
        return {
            outcome: majority ? 'proceed' : 'defer',
            reasoning: `Neural majority: ${participationCount}/${weightedVotes.length} agents agree`,
            participationRatio: participationCount / weightedVotes.length
        };
    }
    
    capabilityBasedConsensus(weightedVotes, requirements) {
        // Implementation for capability-based consensus
        const requiredCaps = requirements.requiredCapabilities || [];
        let coverageScore = 0;
        
        for (const requiredCap of requiredCaps) {
            const hasCapability = weightedVotes.some(v => 
                v.vote.participate && this.coordinationMatrix.get(v.agentId).capabilities.includes(requiredCap)
            );
            if (hasCapability) coverageScore++;
        }
        
        const coverage = requiredCaps.length > 0 ? coverageScore / requiredCaps.length : 1;
        
        return {
            outcome: coverage > 0.8 ? 'proceed' : 'defer',
            reasoning: `Capability coverage: ${(coverage * 100).toFixed(1)}%`,
            participationRatio: coverage
        };
    }
}

export default NeuralMeshCoordinator;