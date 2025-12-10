/**
 * Performance-Based Agent Selection System
 * 
 * Implements intelligent agent selection and task distribution based on performance
 * metrics, specialization, and swarm intelligence. Provides dynamic load balancing
 * and adaptive agent assignment for optimal swarm efficiency.
 */

import EventEmitter from 'events';

/**
 * Performance metrics tracked for each agent
 */
export const PERFORMANCE_METRICS = {
    THROUGHPUT: 'throughput',
    LATENCY: 'latency',
    ACCURACY: 'accuracy',
    RELIABILITY: 'reliability',
    EFFICIENCY: 'efficiency',
    ADAPTABILITY: 'adaptability',
    COOPERATION: 'cooperation',
    SPECIALIZATION: 'specialization'
};

/**
 * Selection strategies for agent assignment
 */
export const SELECTION_STRATEGIES = {
    PERFORMANCE_BASED: 'performance_based',
    ROUND_ROBIN: 'round_robin',
    LEAST_LOADED: 'least_loaded',
    SPECIALIZED: 'specialized',
    DIVERSE: 'diverse',
    HYBRID_ADAPTIVE: 'hybrid_adaptive',
    NEURAL_OPTIMIZED: 'neural_optimized'
};

/**
 * Task complexity levels
 */
export const TASK_COMPLEXITY = {
    SIMPLE: 1,
    MODERATE: 2,
    COMPLEX: 3,
    CRITICAL: 4
};

/**
 * Performance-based agent selector and task distributor
 */
export class PerformanceSelector extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            strategy: config.strategy || SELECTION_STRATEGIES.HYBRID_ADAPTIVE,
            performanceWindow: config.performanceWindow || 100, // Number of recent tasks to consider
            adaptationRate: config.adaptationRate || 0.1,
            diversityWeight: config.diversityWeight || 0.2,
            specializationBonus: config.specializationBonus || 0.3,
            cooperationWeight: config.cooperationWeight || 0.15,
            loadBalancingWeight: config.loadBalancingWeight || 0.25,
            neuralOptimization: config.neuralOptimization !== false,
            performanceDecay: config.performanceDecay || 0.95, // Decay factor for old performance
            minSelectionThreshold: config.minSelectionThreshold || 0.3,
            maxLoadImbalance: config.maxLoadImbalance || 2.0,
            ...config
        };

        // Agent performance tracking
        this.agentMetrics = new Map();
        this.taskHistory = new Map();
        this.performanceHistory = [];
        this.specializationProfiles = new Map();
        this.cooperationMatrix = new Map();
        
        // Load balancing
        this.agentLoads = new Map();
        this.taskQueue = [];
        this.assignmentHistory = [];
        
        // Neural optimization
        this.neuralOptimizer = null;
        this.optimizationModel = null;
        
        // Selection state
        this.lastSelection = new Map();
        this.selectionCount = new Map();
        this.diversityTracker = new Map();
        
        this.initialized = false;
    }

    /**
     * Initialize the performance selector
     */
    async initialize() {
        if (this.initialized) {
            return;
        }

        console.log('📊 Initializing Performance Selector...');

        // Initialize neural optimization if enabled
        if (this.config.neuralOptimization) {
            await this.initializeNeuralOptimization();
        }

        // Start performance decay timer
        this.startPerformanceDecay();

        this.initialized = true;
        this.emit('initialized');

        console.log('✅ Performance Selector initialized');
    }

    /**
     * Initialize neural optimization for agent selection
     */
    async initializeNeuralOptimization() {
        try {
            this.optimizationModel = {
                weights: {
                    performance: Array(8).fill(0).map(() => Math.random() - 0.5),
                    task: Array(6).fill(0).map(() => Math.random() - 0.5),
                    context: Array(4).fill(0).map(() => Math.random() - 0.5)
                },
                bias: Math.random() - 0.5,
                learningRate: 0.01,

                predict: function(performanceFeatures, taskFeatures, contextFeatures) {
                    let score = this.bias;
                    
                    // Performance features
                    for (let i = 0; i < Math.min(performanceFeatures.length, this.weights.performance.length); i++) {
                        score += performanceFeatures[i] * this.weights.performance[i];
                    }
                    
                    // Task features
                    for (let i = 0; i < Math.min(taskFeatures.length, this.weights.task.length); i++) {
                        score += taskFeatures[i] * this.weights.task[i];
                    }
                    
                    // Context features
                    for (let i = 0; i < Math.min(contextFeatures.length, this.weights.context.length); i++) {
                        score += contextFeatures[i] * this.weights.context[i];
                    }
                    
                    return 1 / (1 + Math.exp(-score)); // Sigmoid activation
                },

                train: function(performanceFeatures, taskFeatures, contextFeatures, target) {
                    const prediction = this.predict(performanceFeatures, taskFeatures, contextFeatures);
                    const error = target - prediction;
                    
                    // Update weights
                    for (let i = 0; i < Math.min(performanceFeatures.length, this.weights.performance.length); i++) {
                        this.weights.performance[i] += this.learningRate * error * performanceFeatures[i];
                    }
                    
                    for (let i = 0; i < Math.min(taskFeatures.length, this.weights.task.length); i++) {
                        this.weights.task[i] += this.learningRate * error * taskFeatures[i];
                    }
                    
                    for (let i = 0; i < Math.min(contextFeatures.length, this.weights.context.length); i++) {
                        this.weights.context[i] += this.learningRate * error * contextFeatures[i];
                    }
                    
                    this.bias += this.learningRate * error;
                }
            };

            console.log('🧠 Neural optimization model initialized');
        } catch (error) {
            console.warn('⚠️ Neural optimization initialization failed, using fallback');
            this.config.neuralOptimization = false;
        }
    }

    /**
     * Register an agent for performance tracking
     */
    registerAgent(agent) {
        const agentId = agent.id;
        
        // Initialize performance metrics
        this.agentMetrics.set(agentId, {
            [PERFORMANCE_METRICS.THROUGHPUT]: 0.5,
            [PERFORMANCE_METRICS.LATENCY]: 100, // ms
            [PERFORMANCE_METRICS.ACCURACY]: 0.5,
            [PERFORMANCE_METRICS.RELIABILITY]: 0.5,
            [PERFORMANCE_METRICS.EFFICIENCY]: 0.5,
            [PERFORMANCE_METRICS.ADAPTABILITY]: 0.5,
            [PERFORMANCE_METRICS.COOPERATION]: 0.5,
            [PERFORMANCE_METRICS.SPECIALIZATION]: 0.5,
            lastUpdated: Date.now(),
            sampleCount: 0
        });

        // Initialize task history
        this.taskHistory.set(agentId, []);
        
        // Initialize load tracking
        this.agentLoads.set(agentId, {
            currentTasks: 0,
            totalAssigned: 0,
            averageLoadTime: 0,
            lastAssigned: 0
        });

        // Initialize selection tracking
        this.selectionCount.set(agentId, 0);
        this.lastSelection.set(agentId, 0);

        // Initialize specialization profile
        this.initializeSpecializationProfile(agentId, agent);

        console.log(`📈 Registered agent ${agentId} for performance tracking`);
        this.emit('agentRegistered', { agentId, agent });
    }

    /**
     * Initialize specialization profile for an agent
     */
    initializeSpecializationProfile(agentId, agent) {
        const capabilities = agent.capabilities || [];
        const agentType = agent.agentType || 'general';
        
        // Create specialization scores based on agent type and capabilities
        const specializations = {
            computation: 0.5,
            analysis: 0.5,
            coordination: 0.5,
            optimization: 0.5,
            research: 0.5,
            testing: 0.5,
            documentation: 0.5,
            debugging: 0.5
        };

        // Adjust based on agent type
        switch (agentType) {
            case 'coder':
                specializations.computation = 0.8;
                specializations.debugging = 0.7;
                specializations.testing = 0.6;
                break;
            case 'analyst':
                specializations.analysis = 0.8;
                specializations.research = 0.7;
                specializations.documentation = 0.6;
                break;
            case 'coordinator':
                specializations.coordination = 0.8;
                specializations.optimization = 0.7;
                break;
            case 'researcher':
                specializations.research = 0.8;
                specializations.analysis = 0.7;
                break;
            case 'optimizer':
                specializations.optimization = 0.8;
                specializations.analysis = 0.6;
                break;
        }

        // Adjust based on capabilities
        for (const capability of capabilities) {
            if (capability.includes('analysis')) {
                specializations.analysis = Math.min(1.0, specializations.analysis + 0.2);
            }
            if (capability.includes('code') || capability.includes('programming')) {
                specializations.computation = Math.min(1.0, specializations.computation + 0.2);
            }
            if (capability.includes('test')) {
                specializations.testing = Math.min(1.0, specializations.testing + 0.2);
            }
            if (capability.includes('research')) {
                specializations.research = Math.min(1.0, specializations.research + 0.2);
            }
        }

        this.specializationProfiles.set(agentId, specializations);
    }

    /**
     * Select best agents for a given task
     */
    async selectAgents(task, options = {}) {
        const {
            count = 1,
            strategy = this.config.strategy,
            excludeAgents = [],
            requireSpecialization = null,
            urgency = 'normal',
            complexity = TASK_COMPLEXITY.MODERATE
        } = options;

        if (!this.initialized) {
            await this.initialize();
        }

        console.log(`🎯 Selecting ${count} agent(s) for task: ${task.id || 'unnamed'}`);

        // Get eligible agents
        const eligibleAgents = this.getEligibleAgents(excludeAgents);
        
        if (eligibleAgents.length === 0) {
            throw new Error('No eligible agents available for task assignment');
        }

        if (eligibleAgents.length <= count) {
            console.log(`📌 Assigning all ${eligibleAgents.length} available agents`);
            return eligibleAgents;
        }

        // Apply selection strategy
        let selectedAgents;
        switch (strategy) {
            case SELECTION_STRATEGIES.PERFORMANCE_BASED:
                selectedAgents = await this.selectByPerformance(task, eligibleAgents, count, complexity);
                break;
            case SELECTION_STRATEGIES.SPECIALIZED:
                selectedAgents = await this.selectBySpecialization(task, eligibleAgents, count, requireSpecialization);
                break;
            case SELECTION_STRATEGIES.LEAST_LOADED:
                selectedAgents = await this.selectByLoad(eligibleAgents, count);
                break;
            case SELECTION_STRATEGIES.DIVERSE:
                selectedAgents = await this.selectDiverse(eligibleAgents, count);
                break;
            case SELECTION_STRATEGIES.NEURAL_OPTIMIZED:
                selectedAgents = await this.selectNeuralOptimized(task, eligibleAgents, count, complexity);
                break;
            case SELECTION_STRATEGIES.HYBRID_ADAPTIVE:
                selectedAgents = await this.selectHybridAdaptive(task, eligibleAgents, count, complexity, urgency);
                break;
            default:
                selectedAgents = await this.selectRoundRobin(eligibleAgents, count);
        }

        // Update selection tracking
        for (const agentId of selectedAgents) {
            this.updateSelectionTracking(agentId);
        }

        // Update load tracking
        this.updateLoadAssignments(selectedAgents, task);

        console.log(`✅ Selected agents: ${selectedAgents.join(', ')}`);
        
        this.emit('agentsSelected', {
            task: task.id,
            selectedAgents,
            strategy,
            totalEligible: eligibleAgents.length
        });

        return selectedAgents;
    }

    /**
     * Select agents based on performance metrics
     */
    async selectByPerformance(task, eligibleAgents, count, complexity) {
        const scores = new Map();

        for (const agentId of eligibleAgents) {
            const metrics = this.agentMetrics.get(agentId);
            if (!metrics) continue;

            // Calculate composite performance score
            let score = (
                metrics[PERFORMANCE_METRICS.THROUGHPUT] * 0.25 +
                (1.0 - Math.min(1.0, metrics[PERFORMANCE_METRICS.LATENCY] / 1000)) * 0.20 +
                metrics[PERFORMANCE_METRICS.ACCURACY] * 0.25 +
                metrics[PERFORMANCE_METRICS.RELIABILITY] * 0.15 +
                metrics[PERFORMANCE_METRICS.EFFICIENCY] * 0.15
            );

            // Adjust for task complexity
            if (complexity >= TASK_COMPLEXITY.COMPLEX) {
                score *= metrics[PERFORMANCE_METRICS.ADAPTABILITY];
            }

            // Apply load balancing penalty
            const load = this.agentLoads.get(agentId);
            if (load && load.currentTasks > 0) {
                const loadPenalty = Math.min(0.5, load.currentTasks * 0.1);
                score *= (1.0 - loadPenalty);
            }

            scores.set(agentId, score);
        }

        // Sort by score and select top performers
        const sortedAgents = Array.from(scores.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([agentId]) => agentId);

        return sortedAgents.slice(0, count);
    }

    /**
     * Select agents based on specialization
     */
    async selectBySpecialization(task, eligibleAgents, count, requiredSpecialization) {
        const taskType = this.inferTaskType(task);
        const specialization = requiredSpecialization || taskType;

        if (!specialization) {
            return this.selectByPerformance(task, eligibleAgents, count, TASK_COMPLEXITY.MODERATE);
        }

        const scores = new Map();

        for (const agentId of eligibleAgents) {
            const profile = this.specializationProfiles.get(agentId);
            const metrics = this.agentMetrics.get(agentId);
            
            if (!profile || !metrics) continue;

            // Base specialization score
            let score = profile[specialization] || 0.5;

            // Boost with specialization performance metric
            score += metrics[PERFORMANCE_METRICS.SPECIALIZATION] * this.config.specializationBonus;

            // Factor in cooperation for team tasks
            if (count > 1) {
                score += metrics[PERFORMANCE_METRICS.COOPERATION] * this.config.cooperationWeight;
            }

            // Apply load balancing
            const load = this.agentLoads.get(agentId);
            if (load && load.currentTasks > 0) {
                score *= (1.0 - Math.min(0.3, load.currentTasks * 0.05));
            }

            scores.set(agentId, score);
        }

        const sortedAgents = Array.from(scores.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([agentId]) => agentId);

        return sortedAgents.slice(0, count);
    }

    /**
     * Select agents based on current load
     */
    async selectByLoad(eligibleAgents, count) {
        const loadScores = eligibleAgents.map(agentId => {
            const load = this.agentLoads.get(agentId);
            const currentLoad = load ? load.currentTasks : 0;
            const avgLoadTime = load ? load.averageLoadTime : 100;
            
            // Lower scores are better (less loaded)
            const loadScore = currentLoad + (avgLoadTime / 1000);
            
            return { agentId, loadScore };
        });

        // Sort by load (ascending) and select least loaded
        loadScores.sort((a, b) => a.loadScore - b.loadScore);
        
        return loadScores.slice(0, count).map(item => item.agentId);
    }

    /**
     * Select diverse agents to maximize team diversity
     */
    async selectDiverse(eligibleAgents, count) {
        if (count === 1) {
            return this.selectByPerformance({}, eligibleAgents, 1, TASK_COMPLEXITY.MODERATE);
        }

        const selected = [];
        const remaining = [...eligibleAgents];

        // Select first agent based on performance
        const firstAgent = await this.selectByPerformance({}, remaining, 1, TASK_COMPLEXITY.MODERATE);
        selected.push(firstAgent[0]);
        remaining.splice(remaining.indexOf(firstAgent[0]), 1);

        // Select remaining agents for maximum diversity
        while (selected.length < count && remaining.length > 0) {
            let bestAgent = null;
            let maxDiversityScore = -1;

            for (const agentId of remaining) {
                const diversityScore = this.calculateDiversityScore(agentId, selected);
                if (diversityScore > maxDiversityScore) {
                    maxDiversityScore = diversityScore;
                    bestAgent = agentId;
                }
            }

            if (bestAgent) {
                selected.push(bestAgent);
                remaining.splice(remaining.indexOf(bestAgent), 1);
            } else {
                break;
            }
        }

        return selected;
    }

    /**
     * Select agents using neural optimization
     */
    async selectNeuralOptimized(task, eligibleAgents, count, complexity) {
        if (!this.optimizationModel) {
            return this.selectByPerformance(task, eligibleAgents, count, complexity);
        }

        const scores = new Map();
        const taskFeatures = this.extractTaskFeatures(task, complexity);
        const contextFeatures = this.extractContextFeatures();

        for (const agentId of eligibleAgents) {
            const performanceFeatures = this.extractPerformanceFeatures(agentId);
            
            const score = this.optimizationModel.predict(
                performanceFeatures,
                taskFeatures,
                contextFeatures
            );

            scores.set(agentId, score);
        }

        const sortedAgents = Array.from(scores.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([agentId]) => agentId);

        return sortedAgents.slice(0, count);
    }

    /**
     * Hybrid adaptive selection combining multiple strategies
     */
    async selectHybridAdaptive(task, eligibleAgents, count, complexity, urgency) {
        const weights = this.calculateStrategyWeights(task, complexity, urgency);
        const combinedScores = new Map();

        // Initialize scores
        for (const agentId of eligibleAgents) {
            combinedScores.set(agentId, 0);
        }

        // Performance-based scores
        if (weights.performance > 0) {
            const perfAgents = await this.selectByPerformance(task, eligibleAgents, eligibleAgents.length, complexity);
            for (let i = 0; i < perfAgents.length; i++) {
                const score = (perfAgents.length - i) / perfAgents.length;
                combinedScores.set(perfAgents[i], combinedScores.get(perfAgents[i]) + score * weights.performance);
            }
        }

        // Specialization-based scores
        if (weights.specialization > 0) {
            const specAgents = await this.selectBySpecialization(task, eligibleAgents, eligibleAgents.length);
            for (let i = 0; i < specAgents.length; i++) {
                const score = (specAgents.length - i) / specAgents.length;
                combinedScores.set(specAgents[i], combinedScores.get(specAgents[i]) + score * weights.specialization);
            }
        }

        // Load-based scores
        if (weights.loadBalancing > 0) {
            const loadAgents = await this.selectByLoad(eligibleAgents, eligibleAgents.length);
            for (let i = 0; i < loadAgents.length; i++) {
                const score = (loadAgents.length - i) / loadAgents.length;
                combinedScores.set(loadAgents[i], combinedScores.get(loadAgents[i]) + score * weights.loadBalancing);
            }
        }

        // Neural optimization scores
        if (weights.neural > 0 && this.optimizationModel) {
            const neuralAgents = await this.selectNeuralOptimized(task, eligibleAgents, eligibleAgents.length, complexity);
            for (let i = 0; i < neuralAgents.length; i++) {
                const score = (neuralAgents.length - i) / neuralAgents.length;
                combinedScores.set(neuralAgents[i], combinedScores.get(neuralAgents[i]) + score * weights.neural);
            }
        }

        // Sort by combined scores
        const sortedAgents = Array.from(combinedScores.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([agentId]) => agentId);

        return sortedAgents.slice(0, count);
    }

    /**
     * Round-robin selection
     */
    async selectRoundRobin(eligibleAgents, count) {
        const selected = [];
        const sortedByLastSelection = eligibleAgents
            .map(agentId => ({ agentId, lastSelection: this.lastSelection.get(agentId) || 0 }))
            .sort((a, b) => a.lastSelection - b.lastSelection);

        for (let i = 0; i < count && i < sortedByLastSelection.length; i++) {
            selected.push(sortedByLastSelection[i].agentId);
        }

        return selected;
    }

    /**
     * Update agent performance metrics after task completion
     */
    updatePerformance(agentId, taskResult) {
        const metrics = this.agentMetrics.get(agentId);
        if (!metrics) {
            console.warn(`⚠️ No metrics found for agent: ${agentId}`);
            return;
        }

        const {
            duration,
            success,
            accuracy = 0.5,
            resourcesUsed = 0.5,
            cooperationScore = 0.5,
            innovationScore = 0.5,
            taskType = 'general'
        } = taskResult;

        // Update metrics with exponential moving average
        const alpha = this.config.adaptationRate;
        const beta = 1 - alpha;

        // Throughput (tasks per minute)
        const throughput = success ? 60000 / duration : 0;
        metrics[PERFORMANCE_METRICS.THROUGHPUT] = 
            beta * metrics[PERFORMANCE_METRICS.THROUGHPUT] + alpha * throughput;

        // Latency (response time)
        metrics[PERFORMANCE_METRICS.LATENCY] = 
            beta * metrics[PERFORMANCE_METRICS.LATENCY] + alpha * duration;

        // Accuracy
        metrics[PERFORMANCE_METRICS.ACCURACY] = 
            beta * metrics[PERFORMANCE_METRICS.ACCURACY] + alpha * accuracy;

        // Reliability (success rate)
        const reliabilityScore = success ? 1.0 : 0.0;
        metrics[PERFORMANCE_METRICS.RELIABILITY] = 
            beta * metrics[PERFORMANCE_METRICS.RELIABILITY] + alpha * reliabilityScore;

        // Efficiency (output per resource)
        const efficiency = success ? (1.0 - resourcesUsed) : 0.0;
        metrics[PERFORMANCE_METRICS.EFFICIENCY] = 
            beta * metrics[PERFORMANCE_METRICS.EFFICIENCY] + alpha * efficiency;

        // Cooperation
        metrics[PERFORMANCE_METRICS.COOPERATION] = 
            beta * metrics[PERFORMANCE_METRICS.COOPERATION] + alpha * cooperationScore;

        // Update specialization based on task type
        this.updateSpecialization(agentId, taskType, success, innovationScore);

        // Update adaptability based on task novelty
        const adaptabilityScore = this.calculateAdaptabilityScore(agentId, taskResult);
        metrics[PERFORMANCE_METRICS.ADAPTABILITY] = 
            beta * metrics[PERFORMANCE_METRICS.ADAPTABILITY] + alpha * adaptabilityScore;

        metrics.lastUpdated = Date.now();
        metrics.sampleCount++;

        // Store task history
        const taskHistory = this.taskHistory.get(agentId);
        taskHistory.push({
            timestamp: Date.now(),
            taskResult,
            metrics: { ...metrics }
        });

        // Keep history size manageable
        if (taskHistory.length > this.config.performanceWindow) {
            taskHistory.splice(0, taskHistory.length - this.config.performanceWindow);
        }

        // Update load tracking
        this.updateLoadCompletion(agentId, duration);

        // Learn from the assignment if neural optimization is enabled
        if (this.optimizationModel && this.assignmentHistory.length > 0) {
            this.learnFromAssignment(agentId, taskResult);
        }

        console.log(`📊 Updated performance for agent ${agentId}:`, {
            throughput: metrics[PERFORMANCE_METRICS.THROUGHPUT].toFixed(2),
            accuracy: metrics[PERFORMANCE_METRICS.ACCURACY].toFixed(2),
            reliability: metrics[PERFORMANCE_METRICS.RELIABILITY].toFixed(2)
        });

        this.emit('performanceUpdated', { agentId, metrics, taskResult });
    }

    /**
     * Update specialization profile based on task performance
     */
    updateSpecialization(agentId, taskType, success, innovationScore) {
        const profile = this.specializationProfiles.get(agentId);
        if (!profile) return;

        const specialization = this.mapTaskTypeToSpecialization(taskType);
        if (!specialization || !profile.hasOwnProperty(specialization)) return;

        const alpha = this.config.adaptationRate * 0.5; // Slower adaptation for specialization
        const performanceScore = success ? (0.5 + innovationScore * 0.5) : 0.2;

        profile[specialization] = (1 - alpha) * profile[specialization] + alpha * performanceScore;

        // Update the specialization performance metric
        const metrics = this.agentMetrics.get(agentId);
        if (metrics) {
            const avgSpecialization = Object.values(profile).reduce((sum, val) => sum + val, 0) / Object.keys(profile).length;
            metrics[PERFORMANCE_METRICS.SPECIALIZATION] = avgSpecialization;
        }
    }

    /**
     * Calculate adaptability score based on task novelty and performance
     */
    calculateAdaptabilityScore(agentId, taskResult) {
        const taskHistory = this.taskHistory.get(agentId);
        if (!taskHistory || taskHistory.length < 2) {
            return 0.5; // Default for new agents
        }

        // Check task similarity with recent history
        const recentTasks = taskHistory.slice(-10);
        const currentTaskType = taskResult.taskType || 'general';
        
        const similarTasks = recentTasks.filter(entry => 
            entry.taskResult.taskType === currentTaskType
        ).length;

        const noveltyScore = 1.0 - (similarTasks / recentTasks.length);
        const performanceScore = taskResult.success ? taskResult.accuracy || 0.5 : 0.0;

        // Higher adaptability if performed well on novel tasks
        return noveltyScore * performanceScore + (1 - noveltyScore) * 0.5;
    }

    /**
     * Calculate strategy weights for hybrid adaptive selection
     */
    calculateStrategyWeights(task, complexity, urgency) {
        const weights = {
            performance: 0.4,
            specialization: 0.3,
            loadBalancing: 0.2,
            neural: 0.1
        };

        // Adjust based on urgency
        if (urgency === 'high' || urgency === 'critical') {
            weights.performance += 0.2;
            weights.loadBalancing += 0.1;
            weights.specialization -= 0.2;
            weights.neural -= 0.1;
        }

        // Adjust based on complexity
        if (complexity >= TASK_COMPLEXITY.COMPLEX) {
            weights.specialization += 0.2;
            weights.neural += 0.1;
            weights.performance -= 0.1;
            weights.loadBalancing -= 0.2;
        }

        // Adjust based on load imbalance
        const loadImbalance = this.calculateLoadImbalance();
        if (loadImbalance > this.config.maxLoadImbalance) {
            weights.loadBalancing += 0.3;
            weights.performance -= 0.1;
            weights.specialization -= 0.1;
            weights.neural -= 0.1;
        }

        // Ensure weights sum to 1
        const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);
        for (const key in weights) {
            weights[key] /= totalWeight;
        }

        return weights;
    }

    /**
     * Calculate diversity score for an agent relative to selected agents
     */
    calculateDiversityScore(agentId, selectedAgents) {
        if (selectedAgents.length === 0) {
            return 1.0;
        }

        const profile = this.specializationProfiles.get(agentId);
        const metrics = this.agentMetrics.get(agentId);
        
        if (!profile || !metrics) {
            return 0.5;
        }

        let totalDifference = 0;
        let comparisons = 0;

        for (const selectedId of selectedAgents) {
            const selectedProfile = this.specializationProfiles.get(selectedId);
            const selectedMetrics = this.agentMetrics.get(selectedId);
            
            if (!selectedProfile || !selectedMetrics) continue;

            // Compare specialization profiles
            let specializationDiff = 0;
            for (const spec in profile) {
                specializationDiff += Math.abs(profile[spec] - (selectedProfile[spec] || 0.5));
            }
            specializationDiff /= Object.keys(profile).length;

            // Compare performance characteristics
            const performanceDiff = (
                Math.abs(metrics[PERFORMANCE_METRICS.THROUGHPUT] - selectedMetrics[PERFORMANCE_METRICS.THROUGHPUT]) +
                Math.abs(metrics[PERFORMANCE_METRICS.ACCURACY] - selectedMetrics[PERFORMANCE_METRICS.ACCURACY]) +
                Math.abs(metrics[PERFORMANCE_METRICS.COOPERATION] - selectedMetrics[PERFORMANCE_METRICS.COOPERATION])
            ) / 3;

            totalDifference += (specializationDiff + performanceDiff) / 2;
            comparisons++;
        }

        return comparisons > 0 ? totalDifference / comparisons : 1.0;
    }

    /**
     * Extract performance features for neural optimization
     */
    extractPerformanceFeatures(agentId) {
        const metrics = this.agentMetrics.get(agentId);
        const load = this.agentLoads.get(agentId);
        
        if (!metrics) {
            return Array(8).fill(0.5);
        }

        return [
            metrics[PERFORMANCE_METRICS.THROUGHPUT] / 100, // Normalized
            Math.min(1.0, 100 / metrics[PERFORMANCE_METRICS.LATENCY]), // Inverted and normalized
            metrics[PERFORMANCE_METRICS.ACCURACY],
            metrics[PERFORMANCE_METRICS.RELIABILITY],
            metrics[PERFORMANCE_METRICS.EFFICIENCY],
            metrics[PERFORMANCE_METRICS.ADAPTABILITY],
            metrics[PERFORMANCE_METRICS.COOPERATION],
            load ? Math.min(1.0, load.currentTasks / 10) : 0
        ];
    }

    /**
     * Extract task features for neural optimization
     */
    extractTaskFeatures(task, complexity) {
        return [
            complexity / 4, // Normalized complexity
            task.priority === 'high' ? 1 : (task.priority === 'medium' ? 0.5 : 0),
            task.estimatedDuration ? Math.min(1.0, task.estimatedDuration / 3600000) : 0.5, // Normalized to hours
            task.dependencies ? Math.min(1.0, task.dependencies.length / 10) : 0,
            task.resourceRequirement || 0.5,
            task.collaborationRequired ? 1 : 0
        ];
    }

    /**
     * Extract context features for neural optimization
     */
    extractContextFeatures() {
        const totalAgents = this.agentMetrics.size;
        const loadedAgents = Array.from(this.agentLoads.values())
            .filter(load => load.currentTasks > 0).length;
        
        return [
            totalAgents / 100, // Normalized agent count
            totalAgents > 0 ? loadedAgents / totalAgents : 0, // Load ratio
            this.calculateLoadImbalance() / 5, // Normalized load imbalance
            this.performanceHistory.length / 1000 // Normalized history length
        ];
    }

    /**
     * Learn from assignment outcomes for neural optimization
     */
    learnFromAssignment(agentId, taskResult) {
        if (!this.optimizationModel) return;

        // Find the corresponding assignment
        const assignment = this.assignmentHistory.find(a => 
            a.agentIds.includes(agentId) && 
            Math.abs(a.timestamp - (Date.now() - taskResult.duration)) < 60000
        );

        if (!assignment) return;

        const performanceFeatures = this.extractPerformanceFeatures(agentId);
        const taskFeatures = assignment.taskFeatures;
        const contextFeatures = assignment.contextFeatures;
        
        // Target is the normalized performance outcome
        const target = taskResult.success ? 
            (taskResult.accuracy * 0.7 + (1.0 - Math.min(1.0, taskResult.duration / 60000)) * 0.3) : 0.1;

        // Train the model
        this.optimizationModel.train(performanceFeatures, taskFeatures, contextFeatures, target);
    }

    /**
     * Helper methods
     */
    getEligibleAgents(excludeAgents = []) {
        const excludeSet = new Set(excludeAgents);
        return Array.from(this.agentMetrics.keys())
            .filter(agentId => !excludeSet.has(agentId));
    }

    inferTaskType(task) {
        const description = (task.description || '').toLowerCase();
        
        if (description.includes('code') || description.includes('program') || description.includes('implement')) {
            return 'computation';
        }
        if (description.includes('analyze') || description.includes('report') || description.includes('study')) {
            return 'analysis';
        }
        if (description.includes('coordinate') || description.includes('manage') || description.includes('organize')) {
            return 'coordination';
        }
        if (description.includes('optimize') || description.includes('improve') || description.includes('enhance')) {
            return 'optimization';
        }
        if (description.includes('research') || description.includes('investigate') || description.includes('explore')) {
            return 'research';
        }
        if (description.includes('test') || description.includes('verify') || description.includes('validate')) {
            return 'testing';
        }
        if (description.includes('document') || description.includes('write') || description.includes('explain')) {
            return 'documentation';
        }
        if (description.includes('debug') || description.includes('fix') || description.includes('troubleshoot')) {
            return 'debugging';
        }
        
        return 'general';
    }

    mapTaskTypeToSpecialization(taskType) {
        const mapping = {
            computation: 'computation',
            coding: 'computation',
            programming: 'computation',
            analysis: 'analysis',
            analytics: 'analysis',
            coordination: 'coordination',
            management: 'coordination',
            optimization: 'optimization',
            research: 'research',
            testing: 'testing',
            documentation: 'documentation',
            debugging: 'debugging'
        };
        
        return mapping[taskType.toLowerCase()] || null;
    }

    updateSelectionTracking(agentId) {
        this.selectionCount.set(agentId, (this.selectionCount.get(agentId) || 0) + 1);
        this.lastSelection.set(agentId, Date.now());
    }

    updateLoadAssignments(agentIds, task) {
        const estimatedDuration = task.estimatedDuration || 60000; // 1 minute default
        
        for (const agentId of agentIds) {
            const load = this.agentLoads.get(agentId);
            if (load) {
                load.currentTasks++;
                load.totalAssigned++;
                load.lastAssigned = Date.now();
            }
        }

        // Store assignment for learning
        this.assignmentHistory.push({
            timestamp: Date.now(),
            agentIds: [...agentIds],
            taskId: task.id,
            taskFeatures: this.extractTaskFeatures(task, task.complexity || TASK_COMPLEXITY.MODERATE),
            contextFeatures: this.extractContextFeatures(),
            estimatedDuration
        });

        // Keep history size manageable
        if (this.assignmentHistory.length > 1000) {
            this.assignmentHistory = this.assignmentHistory.slice(-500);
        }
    }

    updateLoadCompletion(agentId, actualDuration) {
        const load = this.agentLoads.get(agentId);
        if (load && load.currentTasks > 0) {
            load.currentTasks--;
            
            // Update average load time
            load.averageLoadTime = load.averageLoadTime * 0.9 + actualDuration * 0.1;
        }
    }

    calculateLoadImbalance() {
        const loads = Array.from(this.agentLoads.values()).map(load => load.currentTasks);
        if (loads.length === 0) return 0;

        const maxLoad = Math.max(...loads);
        const minLoad = Math.min(...loads);
        
        return minLoad === 0 ? maxLoad : maxLoad / minLoad;
    }

    startPerformanceDecay() {
        setInterval(() => {
            for (const [agentId, metrics] of this.agentMetrics) {
                const timeSinceUpdate = Date.now() - metrics.lastUpdated;
                
                // Apply decay if no recent updates (older than 5 minutes)
                if (timeSinceUpdate > 300000) {
                    const decayFactor = Math.pow(this.config.performanceDecay, timeSinceUpdate / 300000);
                    
                    // Decay performance metrics towards neutral values
                    for (const metric in metrics) {
                        if (typeof metrics[metric] === 'number' && metric !== 'lastUpdated' && metric !== 'sampleCount') {
                            const neutralValue = metric === PERFORMANCE_METRICS.LATENCY ? 100 : 0.5;
                            metrics[metric] = metrics[metric] * decayFactor + neutralValue * (1 - decayFactor);
                        }
                    }
                }
            }
        }, 60000); // Check every minute
    }

    /**
     * Get performance metrics for an agent
     */
    getAgentMetrics(agentId) {
        return this.agentMetrics.get(agentId);
    }

    /**
     * Get all agent metrics
     */
    getAllMetrics() {
        return Object.fromEntries(this.agentMetrics);
    }

    /**
     * Get performance statistics
     */
    getPerformanceStatistics() {
        const allMetrics = Array.from(this.agentMetrics.values());
        
        if (allMetrics.length === 0) {
            return {};
        }

        const stats = {};
        
        for (const metric of Object.values(PERFORMANCE_METRICS)) {
            const values = allMetrics.map(m => m[metric]).filter(v => typeof v === 'number');
            if (values.length > 0) {
                stats[metric] = {
                    mean: values.reduce((sum, v) => sum + v, 0) / values.length,
                    min: Math.min(...values),
                    max: Math.max(...values),
                    std: Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - stats[metric]?.mean || 0, 2), 0) / values.length)
                };
            }
        }

        return {
            agentCount: allMetrics.length,
            totalAssignments: Array.from(this.agentLoads.values()).reduce((sum, load) => sum + load.totalAssigned, 0),
            loadImbalance: this.calculateLoadImbalance(),
            metrics: stats
        };
    }

    /**
     * Stop the performance selector
     */
    async stop() {
        this.initialized = false;
        this.emit('stopped');
        console.log('🔻 Performance Selector stopped');
    }
}

export default PerformanceSelector;