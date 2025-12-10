/**
 * Swarm Intelligence Integration - Bridges Rust DAA with JavaScript coordination
 * 
 * This module provides the integration layer between the Rust-based DAA swarm intelligence
 * and the JavaScript ruv-swarm framework, enabling evolutionary behavior and self-organization.
 */

import { DAAService } from './daa-service.js';
import { NeuralAgent, NeuralAgentFactory } from './neural-agent.js';
import { WasmModuleLoader } from './wasm-loader.js';
import EventEmitter from 'events';

/**
 * Swarm Intelligence Coordinator
 * Manages evolutionary algorithms, self-organization, and adaptation
 */
export class SwarmIntelligenceCoordinator extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            topology: config.topology || 'adaptive',
            populationSize: config.populationSize || 50,
            evolutionInterval: config.evolutionInterval || 30000, // 30 seconds
            organizationInterval: config.organizationInterval || 60000, // 1 minute
            mutationRate: config.mutationRate || 0.1,
            crossoverRate: config.crossoverRate || 0.7,
            selectionPressure: config.selectionPressure || 2.0,
            elitismRate: config.elitismRate || 0.1,
            adaptationThreshold: config.adaptationThreshold || 0.1,
            selfHealingEnabled: config.selfHealingEnabled !== false,
            ...config
        };

        // Core components
        this.daaService = new DAAService();
        this.wasmLoader = new WasmModuleLoader();
        this.swarmIntelligence = null;
        this.evolutionaryMesh = null;
        this.selfOrganizing = null;
        
        // Agent management
        this.agents = new Map();
        this.agentPopulation = [];
        this.performanceHistory = [];
        
        // Evolution state
        this.generation = 0;
        this.isEvolutionActive = false;
        this.isOrganizationActive = false;
        
        // Metrics
        this.metrics = {
            averageFitness: 0,
            diversityIndex: 0,
            adaptationRate: 0,
            convergenceRate: 0,
            networkEfficiency: 0,
            faultTolerance: 0,
            lastEvolution: 0,
            lastOrganization: 0
        };

        // Timers
        this.evolutionTimer = null;
        this.organizationTimer = null;
        
        this.initialized = false;
    }

    /**
     * Initialize the swarm intelligence system
     */
    async initialize() {
        if (this.initialized) {
            return;
        }

        try {
            console.log('🧠 Initializing Swarm Intelligence...');

            // Initialize DAA service
            await this.daaService.initialize();

            // Initialize neural agent factory
            await NeuralAgentFactory.initializeFactory();

            // Load WASM modules for DAA integration
            await this.wasmLoader.initialize('progressive');
            
            // Try to get DAA modules
            try {
                const daaModule = await this.wasmLoader.loadModule('daa-swarm');
                if (daaModule?.exports) {
                    this.swarmIntelligence = new daaModule.exports.SwarmIntelligence('HybridAdaptive');
                    this.evolutionaryMesh = new daaModule.exports.EvolutionaryMesh('Adaptive', 'HybridAdaptive');
                    this.selfOrganizing = new daaModule.exports.SelfOrganizingSystem('Dynamic');
                }
            } catch (wasmError) {
                console.warn('⚠️ WASM DAA modules not available, using JavaScript fallback');
                this.initializeFallbackImplementations();
            }

            // Initialize population
            await this.initializePopulation();

            // Start evolution and organization cycles
            this.startEvolutionCycle();
            this.startOrganizationCycle();

            this.initialized = true;
            this.emit('initialized');

            console.log('✅ Swarm Intelligence initialized successfully');

        } catch (error) {
            console.error('❌ Failed to initialize Swarm Intelligence:', error);
            throw error;
        }
    }

    /**
     * Initialize fallback implementations when WASM is not available
     */
    initializeFallbackImplementations() {
        this.swarmIntelligence = new SwarmIntelligenceFallback(this.config);
        this.evolutionaryMesh = new EvolutionaryMeshFallback(this.config);
        this.selfOrganizing = new SelfOrganizingFallback(this.config);
    }

    /**
     * Initialize agent population
     */
    async initializePopulation() {
        console.log(`🔬 Initializing population of ${this.config.populationSize} agents...`);

        for (let i = 0; i < this.config.populationSize; i++) {
            const agentType = this.selectAgentType(i);
            const agent = await this.createEvolutionaryAgent(i, agentType);
            this.agentPopulation.push(agent);
            this.agents.set(agent.id, agent);
        }

        // Initialize mesh topology if available
        if (this.evolutionaryMesh?.initialize) {
            await this.evolutionaryMesh.initialize(this.config.populationSize);
        }

        // Initialize self-organizing rules if available
        if (this.selfOrganizing?.initialize_rules) {
            await this.selfOrganizing.initialize_rules();
        }

        console.log(`✅ Population initialized with ${this.agentPopulation.length} agents`);
    }

    /**
     * Select agent type based on diversity requirements
     */
    selectAgentType(index) {
        const types = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'];
        const typeDistribution = {
            researcher: 0.25,
            coder: 0.30,
            analyst: 0.20,
            optimizer: 0.15,
            coordinator: 0.10
        };

        // Ensure type diversity
        if (index < types.length) {
            return types[index];
        }

        // Random selection based on distribution
        const rand = Math.random();
        let cumulative = 0;
        for (const [type, probability] of Object.entries(typeDistribution)) {
            cumulative += probability;
            if (rand <= cumulative) {
                return type;
            }
        }

        return 'researcher'; // fallback
    }

    /**
     * Create an evolutionary agent with neural capabilities
     */
    async createEvolutionaryAgent(index, agentType) {
        // Create base agent
        const baseAgent = await this.daaService.createAgent({
            id: `agent_${index}`,
            capabilities: this.getAgentCapabilities(agentType),
            cognitivePattern: this.getCognitivePattern(agentType),
            learningRate: 0.001 + Math.random() * 0.009, // 0.001-0.01
            enableMemory: true,
            autonomousMode: true
        });

        // Enhance with neural capabilities
        const neuralAgent = NeuralAgentFactory.createNeuralAgent(baseAgent, agentType);

        // Add evolutionary traits
        neuralAgent.evolutionaryTraits = {
            fitness: 0.5,
            generation: 0,
            mutationRate: this.config.mutationRate,
            parentIds: [],
            offspringCount: 0,
            adaptationHistory: []
        };

        // Add performance tracking
        neuralAgent.performance = {
            tasksCompleted: 0,
            successRate: 0.5,
            averageResponseTime: 100,
            resourceEfficiency: 0.5,
            cooperationScore: 0.5,
            innovationScore: 0.5
        };

        return neuralAgent;
    }

    /**
     * Get agent capabilities based on type
     */
    getAgentCapabilities(agentType) {
        const capabilityMap = {
            researcher: ['analysis', 'exploration', 'pattern_recognition', 'knowledge_synthesis'],
            coder: ['code_generation', 'debugging', 'optimization', 'testing'],
            analyst: ['data_analysis', 'visualization', 'reporting', 'prediction'],
            optimizer: ['performance_tuning', 'resource_optimization', 'bottleneck_analysis'],
            coordinator: ['task_distribution', 'team_coordination', 'conflict_resolution']
        };

        return capabilityMap[agentType] || ['general_purpose'];
    }

    /**
     * Get cognitive pattern for agent type
     */
    getCognitivePattern(agentType) {
        const patternMap = {
            researcher: 'divergent',
            coder: 'convergent',
            analyst: 'critical',
            optimizer: 'systems',
            coordinator: 'systems'
        };

        return patternMap[agentType] || 'adaptive';
    }

    /**
     * Start evolution cycle
     */
    startEvolutionCycle() {
        if (this.evolutionTimer) {
            clearInterval(this.evolutionTimer);
        }

        this.evolutionTimer = setInterval(async () => {
            if (!this.isEvolutionActive) {
                await this.evolvePopulation();
            }
        }, this.config.evolutionInterval);
    }

    /**
     * Start organization cycle
     */
    startOrganizationCycle() {
        if (this.organizationTimer) {
            clearInterval(this.organizationTimer);
        }

        this.organizationTimer = setInterval(async () => {
            if (!this.isOrganizationActive) {
                await this.organizeSwarm();
            }
        }, this.config.organizationInterval);
    }

    /**
     * Evolve the agent population using genetic algorithms
     */
    async evolvePopulation() {
        if (this.isEvolutionActive) {
            return;
        }

        this.isEvolutionActive = true;
        const startTime = Date.now();

        try {
            console.log(`🧬 Starting evolution cycle ${this.generation + 1}...`);

            // Evaluate fitness for all agents
            await this.evaluateFitness();

            // Sort agents by fitness
            this.agentPopulation.sort((a, b) => b.evolutionaryTraits.fitness - a.evolutionaryTraits.fitness);

            // Create new generation
            const newGeneration = await this.createNewGeneration();

            // Replace population (keeping elites)
            const eliteCount = Math.floor(this.config.elitismRate * this.agentPopulation.length);
            const newPopulation = [
                ...this.agentPopulation.slice(0, eliteCount), // Keep elites
                ...newGeneration.slice(0, this.agentPopulation.length - eliteCount)
            ];

            // Update population
            this.agentPopulation = newPopulation;
            this.generation++;

            // Update mesh topology if available
            if (this.evolutionaryMesh?.evolve) {
                const performanceData = {};
                for (const agent of this.agentPopulation) {
                    performanceData[agent.id] = agent.evolutionaryTraits.fitness;
                }
                await this.evolutionaryMesh.evolve(performanceData);
            }

            // Update metrics
            await this.updateMetrics();

            const duration = Date.now() - startTime;
            console.log(`✅ Evolution cycle ${this.generation} completed in ${duration}ms`);

            this.emit('evolutionCompleted', {
                generation: this.generation,
                metrics: this.metrics,
                duration
            });

        } catch (error) {
            console.error('❌ Evolution cycle failed:', error);
            this.emit('evolutionError', error);
        } finally {
            this.isEvolutionActive = false;
            this.metrics.lastEvolution = Date.now();
        }
    }

    /**
     * Evaluate fitness for all agents
     */
    async evaluateFitness() {
        const evaluationPromises = this.agentPopulation.map(async (agent) => {
            const fitness = await this.calculateAgentFitness(agent);
            agent.evolutionaryTraits.fitness = fitness;
            return fitness;
        });

        await Promise.all(evaluationPromises);
    }

    /**
     * Calculate fitness for an agent
     */
    async calculateAgentFitness(agent) {
        const performance = agent.performance;
        const cognitiveState = agent.cognitiveState;

        // Multi-objective fitness function
        const fitness = (
            performance.successRate * 0.25 +
            (1.0 - performance.averageResponseTime / 10000) * 0.20 + // Normalize response time
            performance.resourceEfficiency * 0.15 +
            performance.cooperationScore * 0.15 +
            performance.innovationScore * 0.10 +
            (1.0 - cognitiveState.fatigue) * 0.10 +
            cognitiveState.confidence * 0.05
        );

        return Math.max(0, Math.min(1, fitness));
    }

    /**
     * Create new generation through selection, crossover, and mutation
     */
    async createNewGeneration() {
        const newGeneration = [];
        const populationSize = this.agentPopulation.length;

        while (newGeneration.length < populationSize) {
            // Selection
            const parent1 = this.tournamentSelection();
            const parent2 = this.tournamentSelection();

            if (parent1 && parent2 && parent1.id !== parent2.id) {
                // Crossover
                if (Math.random() < this.config.crossoverRate) {
                    const offspring = await this.crossover(parent1, parent2);
                    
                    // Mutation
                    if (Math.random() < this.config.mutationRate) {
                        await this.mutate(offspring);
                    }

                    offspring.evolutionaryTraits.generation = this.generation + 1;
                    offspring.evolutionaryTraits.parentIds = [parent1.id, parent2.id];
                    
                    newGeneration.push(offspring);
                } else {
                    // Direct copy with possible mutation
                    const offspring = await this.cloneAgent(parent1);
                    
                    if (Math.random() < this.config.mutationRate) {
                        await this.mutate(offspring);
                    }

                    offspring.evolutionaryTraits.generation = this.generation + 1;
                    offspring.evolutionaryTraits.parentIds = [parent1.id];
                    
                    newGeneration.push(offspring);
                }
            }
        }

        return newGeneration;
    }

    /**
     * Tournament selection for parent selection
     */
    tournamentSelection() {
        const tournamentSize = Math.max(2, Math.floor(this.config.selectionPressure));
        const tournament = [];

        for (let i = 0; i < tournamentSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.agentPopulation.length);
            tournament.push(this.agentPopulation[randomIndex]);
        }

        // Return the fittest from tournament
        return tournament.reduce((best, current) => 
            current.evolutionaryTraits.fitness > best.evolutionaryTraits.fitness ? current : best
        );
    }

    /**
     * Crossover two agents to create offspring
     */
    async crossover(parent1, parent2) {
        // Create new agent with blended characteristics
        const agentType = Math.random() < 0.5 ? parent1.agentType : parent2.agentType;
        const offspring = await this.createEvolutionaryAgent(
            this.getNextAgentId(),
            agentType
        );

        // Blend neural network weights
        this.blendNeuralNetworks(offspring, parent1, parent2);

        // Blend cognitive traits
        this.blendCognitiveTraits(offspring, parent1, parent2);

        // Blend performance characteristics
        this.blendPerformanceTraits(offspring, parent1, parent2);

        return offspring;
    }

    /**
     * Blend neural networks from two parents
     */
    blendNeuralNetworks(offspring, parent1, parent2) {
        const offspringNN = offspring.neuralNetwork;
        const parent1NN = parent1.neuralNetwork;
        const parent2NN = parent2.neuralNetwork;

        // Blend weights
        for (let i = 0; i < offspringNN.weights.length; i++) {
            for (let j = 0; j < offspringNN.weights[i].length; j++) {
                for (let k = 0; k < offspringNN.weights[i][j].length; k++) {
                    const alpha = Math.random();
                    offspringNN.weights[i][j][k] = 
                        alpha * parent1NN.weights[i][j][k] + 
                        (1 - alpha) * parent2NN.weights[i][j][k];
                }
            }
        }

        // Blend biases
        for (let i = 0; i < offspringNN.biases.length; i++) {
            for (let j = 0; j < offspringNN.biases[i].length; j++) {
                const alpha = Math.random();
                offspringNN.biases[i][j] = 
                    alpha * parent1NN.biases[i][j] + 
                    (1 - alpha) * parent2NN.biases[i][j];
            }
        }
    }

    /**
     * Blend cognitive traits from two parents
     */
    blendCognitiveTraits(offspring, parent1, parent2) {
        const p1State = parent1.cognitiveState;
        const p2State = parent2.cognitiveState;
        const offspringState = offspring.cognitiveState;

        offspringState.attention = (p1State.attention + p2State.attention) / 2;
        offspringState.confidence = (p1State.confidence + p2State.confidence) / 2;
        offspringState.exploration = (p1State.exploration + p2State.exploration) / 2;
    }

    /**
     * Blend performance traits from two parents
     */
    blendPerformanceTraits(offspring, parent1, parent2) {
        const p1Perf = parent1.performance;
        const p2Perf = parent2.performance;
        const offspringPerf = offspring.performance;

        offspringPerf.resourceEfficiency = (p1Perf.resourceEfficiency + p2Perf.resourceEfficiency) / 2;
        offspringPerf.cooperationScore = (p1Perf.cooperationScore + p2Perf.cooperationScore) / 2;
        offspringPerf.innovationScore = (p1Perf.innovationScore + p2Perf.innovationScore) / 2;
    }

    /**
     * Mutate an agent's characteristics
     */
    async mutate(agent) {
        const mutationStrength = 0.1;

        // Mutate neural network
        this.mutateNeuralNetwork(agent.neuralNetwork, mutationStrength);

        // Mutate cognitive traits
        this.mutateCognitiveTraits(agent, mutationStrength);

        // Record mutation
        agent.evolutionaryTraits.adaptationHistory.push({
            generation: this.generation,
            mutationType: 'genetic',
            timestamp: Date.now()
        });
    }

    /**
     * Mutate neural network weights
     */
    mutateNeuralNetwork(neuralNetwork, strength) {
        // Mutate weights
        for (let i = 0; i < neuralNetwork.weights.length; i++) {
            for (let j = 0; j < neuralNetwork.weights[i].length; j++) {
                for (let k = 0; k < neuralNetwork.weights[i][j].length; k++) {
                    if (Math.random() < this.config.mutationRate) {
                        const mutation = (Math.random() - 0.5) * strength;
                        neuralNetwork.weights[i][j][k] += mutation;
                    }
                }
            }
        }

        // Mutate biases
        for (let i = 0; i < neuralNetwork.biases.length; i++) {
            for (let j = 0; j < neuralNetwork.biases[i].length; j++) {
                if (Math.random() < this.config.mutationRate) {
                    const mutation = (Math.random() - 0.5) * strength;
                    neuralNetwork.biases[i][j] += mutation;
                }
            }
        }
    }

    /**
     * Mutate cognitive traits
     */
    mutateCognitiveTraits(agent, strength) {
        const traits = agent.cognitiveState;
        
        if (Math.random() < this.config.mutationRate) {
            traits.confidence += (Math.random() - 0.5) * strength;
            traits.confidence = Math.max(0, Math.min(1, traits.confidence));
        }

        if (Math.random() < this.config.mutationRate) {
            traits.exploration += (Math.random() - 0.5) * strength;
            traits.exploration = Math.max(0, Math.min(1, traits.exploration));
        }
    }

    /**
     * Clone an agent
     */
    async cloneAgent(original) {
        const clone = await this.createEvolutionaryAgent(
            this.getNextAgentId(),
            original.agentType
        );

        // Copy neural network state
        clone.neuralNetwork.load(original.neuralNetwork.save());

        // Copy cognitive state
        clone.cognitiveState = { ...original.cognitiveState };

        // Copy performance characteristics
        clone.performance = { ...original.performance };

        return clone;
    }

    /**
     * Organize swarm using self-organizing principles
     */
    async organizeSwarm() {
        if (this.isOrganizationActive) {
            return;
        }

        this.isOrganizationActive = true;
        const startTime = Date.now();

        try {
            console.log('🏗️ Starting swarm organization...');

            // Create node information for clustering
            const nodeInfos = this.agentPopulation.map(agent => ({
                id: agent.id,
                position: [Math.random(), Math.random()], // Simplified position
                capabilities: agent.getCapabilities ? agent.getCapabilities() : {},
                connections: [],
                performance: agent.evolutionaryTraits.fitness
            }));

            // Create organization context
            const context = {
                metrics: {
                    node_density: this.agentPopulation.length / 100,
                    performance_degradation: this.calculatePerformanceDegradation(),
                    clustering_quality: this.calculateClusteringQuality()
                },
                detected_patterns: {
                    leaderless_cluster: this.detectLeaderlessClusters()
                },
                event_counts: {},
                last_organization_time: this.metrics.lastOrganization,
                significant_change_detected: this.detectSignificantChanges(),
                desired_clusters: Math.max(3, Math.floor(this.agentPopulation.length / 10)),
                use_geographic_clustering: false
            };

            // Perform organization using self-organizing system
            if (this.selfOrganizing?.organize) {
                await this.selfOrganizing.organize(nodeInfos, context);
            } else {
                // Fallback organization
                await this.fallbackOrganization(nodeInfos, context);
            }

            // Apply self-healing if enabled
            if (this.config.selfHealingEnabled) {
                await this.performSelfHealing();
            }

            const duration = Date.now() - startTime;
            console.log(`✅ Swarm organization completed in ${duration}ms`);

            this.emit('organizationCompleted', {
                metrics: this.metrics,
                duration
            });

        } catch (error) {
            console.error('❌ Swarm organization failed:', error);
            this.emit('organizationError', error);
        } finally {
            this.isOrganizationActive = false;
            this.metrics.lastOrganization = Date.now();
        }
    }

    /**
     * Perform self-healing by identifying and replacing failed agents
     */
    async performSelfHealing() {
        const failedAgents = this.agentPopulation.filter(agent => 
            agent.evolutionaryTraits.fitness < 0.2 || 
            agent.cognitiveState.fatigue > 0.9
        );

        if (failedAgents.length === 0) {
            return;
        }

        console.log(`🔧 Self-healing: replacing ${failedAgents.length} failed agents...`);

        for (const failedAgent of failedAgents) {
            // Find the best performing agent as template
            const bestAgent = this.agentPopulation.reduce((best, current) => 
                current.evolutionaryTraits.fitness > best.evolutionaryTraits.fitness ? current : best
            );

            // Create replacement agent based on best performer
            const replacement = await this.cloneAgent(bestAgent);
            replacement.id = failedAgent.id; // Keep the same ID

            // Add some variation to avoid identical copies
            await this.mutate(replacement);

            // Replace in population
            const index = this.agentPopulation.indexOf(failedAgent);
            if (index !== -1) {
                this.agentPopulation[index] = replacement;
                this.agents.set(replacement.id, replacement);
            }

            // Clean up failed agent
            if (failedAgent.cleanup) {
                await failedAgent.cleanup();
            }
        }

        this.emit('selfHealingCompleted', {
            replacedCount: failedAgents.length
        });
    }

    /**
     * Update performance metrics
     */
    async updateMetrics() {
        const population = this.agentPopulation;
        
        // Average fitness
        this.metrics.averageFitness = population.reduce((sum, agent) => 
            sum + agent.evolutionaryTraits.fitness, 0) / population.length;

        // Diversity index
        this.metrics.diversityIndex = this.calculateDiversityIndex();

        // Adaptation rate
        this.metrics.adaptationRate = this.calculateAdaptationRate();

        // Convergence rate
        this.metrics.convergenceRate = 1.0 - this.metrics.diversityIndex;

        // Network efficiency
        this.metrics.networkEfficiency = this.calculateNetworkEfficiency();

        // Fault tolerance
        this.metrics.faultTolerance = this.calculateFaultTolerance();

        // Store performance history
        this.performanceHistory.push({
            generation: this.generation,
            metrics: { ...this.metrics },
            timestamp: Date.now()
        });

        // Keep history size manageable
        if (this.performanceHistory.length > 1000) {
            this.performanceHistory = this.performanceHistory.slice(-500);
        }
    }

    /**
     * Calculate diversity index
     */
    calculateDiversityIndex() {
        if (this.agentPopulation.length < 2) {
            return 1.0;
        }

        // Calculate fitness variance
        const mean = this.metrics.averageFitness;
        const variance = this.agentPopulation.reduce((sum, agent) => 
            sum + Math.pow(agent.evolutionaryTraits.fitness - mean, 2), 0) / this.agentPopulation.length;

        return Math.min(1.0, variance * 10); // Scale appropriately
    }

    /**
     * Calculate adaptation rate
     */
    calculateAdaptationRate() {
        if (this.performanceHistory.length < 2) {
            return 0.0;
        }

        const recent = this.performanceHistory.slice(-5);
        const improvement = recent[recent.length - 1].metrics.averageFitness - recent[0].metrics.averageFitness;
        
        return Math.max(0, Math.min(1, improvement * 10));
    }

    /**
     * Calculate network efficiency
     */
    calculateNetworkEfficiency() {
        // Simplified calculation based on average performance
        const totalPerformance = this.agentPopulation.reduce((sum, agent) => {
            return sum + (
                agent.performance.successRate * 0.4 +
                agent.performance.resourceEfficiency * 0.3 +
                agent.performance.cooperationScore * 0.3
            );
        }, 0);

        return totalPerformance / this.agentPopulation.length;
    }

    /**
     * Calculate fault tolerance
     */
    calculateFaultTolerance() {
        // Calculate percentage of agents that are performing well
        const healthyAgents = this.agentPopulation.filter(agent => 
            agent.evolutionaryTraits.fitness > 0.5 && 
            agent.cognitiveState.fatigue < 0.7
        ).length;

        return healthyAgents / this.agentPopulation.length;
    }

    /**
     * Helper methods for organization
     */
    calculatePerformanceDegradation() {
        if (this.performanceHistory.length < 2) {
            return 0.0;
        }

        const recent = this.performanceHistory[this.performanceHistory.length - 1];
        const previous = this.performanceHistory[this.performanceHistory.length - 2];
        
        return Math.max(0, previous.metrics.averageFitness - recent.metrics.averageFitness);
    }

    calculateClusteringQuality() {
        // Simplified clustering quality metric
        return this.metrics.networkEfficiency;
    }

    detectLeaderlessClusters() {
        // Simplified detection - look for low cooperation scores
        const lowCooperationAgents = this.agentPopulation.filter(agent => 
            agent.performance.cooperationScore < 0.3
        ).length;

        return lowCooperationAgents / this.agentPopulation.length;
    }

    detectSignificantChanges() {
        if (this.performanceHistory.length < 3) {
            return false;
        }

        const recent = this.performanceHistory.slice(-3);
        const variance = recent.reduce((sum, entry, index) => {
            if (index === 0) return 0;
            return sum + Math.abs(entry.metrics.averageFitness - recent[index - 1].metrics.averageFitness);
        }, 0) / (recent.length - 1);

        return variance > 0.1; // Threshold for significant change
    }

    async fallbackOrganization(nodeInfos, context) {
        // Simple fallback organization
        console.log('📝 Using fallback organization...');
        
        // Group agents by performance level
        const groups = {
            high: [],
            medium: [],
            low: []
        };

        for (const agent of this.agentPopulation) {
            if (agent.evolutionaryTraits.fitness > 0.7) {
                groups.high.push(agent);
            } else if (agent.evolutionaryTraits.fitness > 0.4) {
                groups.medium.push(agent);
            } else {
                groups.low.push(agent);
            }
        }

        // Assign cooperation patterns
        for (const agent of groups.high) {
            agent.performance.cooperationScore = Math.min(1.0, agent.performance.cooperationScore + 0.1);
        }

        for (const agent of groups.low) {
            // Low performing agents should cooperate more to improve
            agent.performance.cooperationScore = Math.min(1.0, agent.performance.cooperationScore + 0.2);
        }
    }

    /**
     * Get next available agent ID
     */
    getNextAgentId() {
        return `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get current swarm metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            generation: this.generation,
            populationSize: this.agentPopulation.length,
            initialized: this.initialized,
            isEvolutionActive: this.isEvolutionActive,
            isOrganizationActive: this.isOrganizationActive
        };
    }

    /**
     * Get agent by ID
     */
    getAgent(agentId) {
        return this.agents.get(agentId);
    }

    /**
     * Get all agents
     */
    getAllAgents() {
        return Array.from(this.agents.values());
    }

    /**
     * Get agents by type
     */
    getAgentsByType(agentType) {
        return this.agentPopulation.filter(agent => agent.agentType === agentType);
    }

    /**
     * Trigger manual evolution
     */
    async triggerEvolution() {
        if (!this.isEvolutionActive) {
            await this.evolvePopulation();
        }
    }

    /**
     * Trigger manual organization
     */
    async triggerOrganization() {
        if (!this.isOrganizationActive) {
            await this.organizeSwarm();
        }
    }

    /**
     * Stop the swarm intelligence system
     */
    async stop() {
        if (this.evolutionTimer) {
            clearInterval(this.evolutionTimer);
            this.evolutionTimer = null;
        }

        if (this.organizationTimer) {
            clearInterval(this.organizationTimer);
            this.organizationTimer = null;
        }

        // Clean up agents
        for (const agent of this.agentPopulation) {
            if (agent.cleanup) {
                await agent.cleanup();
            }
        }

        this.agents.clear();
        this.agentPopulation = [];
        this.initialized = false;

        this.emit('stopped');
        console.log('🔻 Swarm Intelligence stopped');
    }
}

/**
 * Fallback implementations for when WASM modules are not available
 */
class SwarmIntelligenceFallback {
    constructor(config) {
        this.config = config;
    }

    async evolve(context) {
        // Simplified evolution logic
        console.log('🔄 Running fallback evolution...');
    }
}

class EvolutionaryMeshFallback {
    constructor(config) {
        this.config = config;
    }

    async initialize(populationSize) {
        console.log(`🔗 Initializing fallback mesh with ${populationSize} nodes...`);
    }

    async evolve(performanceData) {
        console.log('🔄 Running fallback mesh evolution...');
    }
}

class SelfOrganizingFallback {
    constructor(config) {
        this.config = config;
    }

    async initialize_rules() {
        console.log('📋 Initializing fallback organization rules...');
    }

    async organize(nodeInfos, context) {
        console.log('🏗️ Running fallback organization...');
    }
}

export default SwarmIntelligenceCoordinator;