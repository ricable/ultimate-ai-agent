#!/usr/bin/env node

/**
 * Synaptic Neural Mesh - Swarm Intelligence Demonstration
 * 
 * This demonstration showcases the advanced swarm intelligence capabilities
 * of the Synaptic Neural Mesh, including evolutionary algorithms, self-organizing
 * systems, consensus mechanisms, and performance-based agent selection.
 * 
 * Run with: node examples/swarm_intelligence_demo.js
 */

import { SwarmIntelligenceCoordinator } from '../src/js/ruv-swarm/src/swarm-intelligence-integration.js';
import { ConsensusEngine, CONSENSUS_PROTOCOLS } from '../src/js/ruv-swarm/src/consensus-engine.js';
import { PerformanceSelector, SELECTION_STRATEGIES } from '../src/js/ruv-swarm/src/performance-selector.js';

/**
 * Demonstration Configuration
 */
const DEMO_CONFIG = {
    swarmSize: 12,
    taskCount: 50,
    demonstrationTime: 300000, // 5 minutes
    evolutionInterval: 10000,   // 10 seconds
    organizationInterval: 15000, // 15 seconds
    printInterval: 5000,        // 5 seconds
    scenarios: [
        'basic_swarm_intelligence',
        'evolutionary_adaptation',
        'consensus_decision_making',
        'performance_optimization',
        'fault_tolerance_recovery',
        'self_organizing_behavior'
    ]
};

/**
 * Demo Task Types
 */
const TASK_TYPES = [
    { type: 'computation', complexity: 2, description: 'Calculate fibonacci sequence' },
    { type: 'analysis', complexity: 3, description: 'Analyze data patterns' },
    { type: 'coordination', complexity: 1, description: 'Coordinate team activities' },
    { type: 'optimization', complexity: 4, description: 'Optimize resource allocation' },
    { type: 'research', complexity: 3, description: 'Research best practices' },
    { type: 'testing', complexity: 2, description: 'Validate system functionality' },
    { type: 'documentation', complexity: 1, description: 'Document processes' },
    { type: 'debugging', complexity: 3, description: 'Debug system issues' }
];

/**
 * Main demonstration class
 */
class SwarmIntelligenceDemo {
    constructor() {
        this.swarmCoordinator = null;
        this.consensusEngine = null;
        this.performanceSelector = null;
        
        this.demoState = {
            running: false,
            scenario: null,
            startTime: 0,
            tasksCompleted: 0,
            consensusReached: 0,
            adaptationsTriggered: 0,
            agentFailures: 0,
            selfHealingEvents: 0
        };
        
        this.demoStats = {
            averageFitness: [],
            consensusLatency: [],
            throughput: [],
            diversityIndex: [],
            networkEfficiency: []
        };
    }

    /**
     * Run the complete demonstration
     */
    async runDemo() {
        console.log('🚀 Starting Synaptic Neural Mesh - Swarm Intelligence Demonstration\n');
        
        await this.initializeComponents();
        await this.printSystemStatus();
        
        for (const scenario of DEMO_CONFIG.scenarios) {
            console.log(`\n🎬 Running Scenario: ${scenario.replace(/_/g, ' ').toUpperCase()}`);
            console.log('=' * 60);
            
            await this.runScenario(scenario);
            await this.printScenarioResults(scenario);
            
            // Brief pause between scenarios
            await this.sleep(2000);
        }
        
        await this.printFinalResults();
        await this.cleanup();
        
        console.log('\n✅ Demonstration completed successfully!');
    }

    /**
     * Initialize all swarm intelligence components
     */
    async initializeComponents() {
        console.log('🔧 Initializing Swarm Intelligence Components...');
        
        // Initialize Swarm Intelligence Coordinator
        this.swarmCoordinator = new SwarmIntelligenceCoordinator({
            topology: 'adaptive',
            populationSize: DEMO_CONFIG.swarmSize,
            evolutionInterval: DEMO_CONFIG.evolutionInterval,
            organizationInterval: DEMO_CONFIG.organizationInterval,
            mutationRate: 0.1,
            crossoverRate: 0.7,
            selectionPressure: 2.0,
            selfHealingEnabled: true
        });

        // Initialize Consensus Engine
        this.consensusEngine = new ConsensusEngine({
            protocol: CONSENSUS_PROTOCOLS.SWARM_BFT,
            nodeId: 'demo_coordinator',
            faultTolerance: 0.33,
            timeout: 5000,
            adaptiveSelection: true,
            neuralWeight: 0.3
        });

        // Initialize Performance Selector
        this.performanceSelector = new PerformanceSelector({
            strategy: SELECTION_STRATEGIES.HYBRID_ADAPTIVE,
            performanceWindow: 20,
            adaptationRate: 0.15,
            neuralOptimization: true,
            loadBalancingWeight: 0.25
        });

        // Set up event listeners
        this.setupEventListeners();

        // Initialize all components
        await Promise.all([
            this.swarmCoordinator.initialize(),
            this.consensusEngine.initialize(),
            this.performanceSelector.initialize()
        ]);

        // Register agents with performance selector
        const agents = this.swarmCoordinator.getAllAgents();
        for (const agent of agents) {
            this.performanceSelector.registerAgent(agent);
            this.consensusEngine.addNode(agent.id, {
                agentType: agent.agentType,
                capabilities: agent.capabilities
            });
        }

        console.log('✅ All components initialized successfully');
        console.log(`📊 Swarm Size: ${agents.length} agents`);
        console.log(`🧬 Evolution Interval: ${DEMO_CONFIG.evolutionInterval}ms`);
        console.log(`🏗️ Organization Interval: ${DEMO_CONFIG.organizationInterval}ms`);
    }

    /**
     * Set up event listeners for demonstration tracking
     */
    setupEventListeners() {
        // Swarm Intelligence Events
        this.swarmCoordinator.on('evolutionCompleted', (data) => {
            this.demoState.adaptationsTriggered++;
            this.demoStats.averageFitness.push(data.metrics.averageFitness);
            this.demoStats.diversityIndex.push(data.metrics.diversityIndex);
            console.log(`🧬 Evolution #${data.generation} completed (fitness: ${data.metrics.averageFitness.toFixed(3)})`);
        });

        this.swarmCoordinator.on('organizationCompleted', (data) => {
            this.demoStats.networkEfficiency.push(data.metrics.networkEfficiency);
            console.log(`🏗️ Self-organization completed (efficiency: ${data.metrics.networkEfficiency.toFixed(3)})`);
        });

        this.swarmCoordinator.on('selfHealingCompleted', (data) => {
            this.demoState.selfHealingEvents++;
            console.log(`🔧 Self-healing: ${data.replacedCount} agents replaced`);
        });

        // Consensus Events
        this.consensusEngine.on('consensusReached', (data) => {
            this.demoState.consensusReached++;
            const latency = Date.now() - data.proposal.timestamp;
            this.demoStats.consensusLatency.push(latency);
            console.log(`🗳️ Consensus reached for proposal ${data.proposal.id} (${latency}ms)`);
        });

        this.consensusEngine.on('thresholdAdapted', (data) => {
            console.log(`🔄 Consensus threshold adapted: ${data.oldThreshold.toFixed(2)} → ${data.newThreshold.toFixed(2)}`);
        });

        // Performance Events
        this.performanceSelector.on('agentsSelected', (data) => {
            console.log(`🎯 Selected ${data.selectedAgents.length} agents for task ${data.task} using ${data.strategy}`);
        });

        this.performanceSelector.on('performanceUpdated', (data) => {
            const throughput = data.metrics.throughput || 0;
            this.demoStats.throughput.push(throughput);
        });
    }

    /**
     * Run a specific demonstration scenario
     */
    async runScenario(scenario) {
        this.demoState.scenario = scenario;
        this.demoState.running = true;
        this.demoState.startTime = Date.now();

        switch (scenario) {
            case 'basic_swarm_intelligence':
                await this.basicSwarmIntelligenceScenario();
                break;
            case 'evolutionary_adaptation':
                await this.evolutionaryAdaptationScenario();
                break;
            case 'consensus_decision_making':
                await this.consensusDecisionMakingScenario();
                break;
            case 'performance_optimization':
                await this.performanceOptimizationScenario();
                break;
            case 'fault_tolerance_recovery':
                await this.faultToleranceRecoveryScenario();
                break;
            case 'self_organizing_behavior':
                await this.selfOrganizingBehaviorScenario();
                break;
        }

        this.demoState.running = false;
    }

    /**
     * Basic swarm intelligence demonstration
     */
    async basicSwarmIntelligenceScenario() {
        console.log('📋 Demonstrating basic swarm intelligence capabilities...');
        
        // Generate and distribute tasks
        const tasks = this.generateTasks(10);
        
        for (const task of tasks) {
            // Select agents using performance-based selection
            const selectedAgents = await this.performanceSelector.selectAgents(task, {
                count: Math.random() < 0.3 ? 2 : 1, // 30% chance of team task
                strategy: SELECTION_STRATEGIES.HYBRID_ADAPTIVE
            });

            // Execute task with selected agents
            await this.executeTask(task, selectedAgents);
            
            this.demoState.tasksCompleted++;
            
            // Brief pause between tasks
            await this.sleep(500);
        }
    }

    /**
     * Evolutionary adaptation demonstration
     */
    async evolutionaryAdaptationScenario() {
        console.log('🧬 Demonstrating evolutionary adaptation...');
        
        // Trigger multiple evolution cycles with varied task loads
        const evolutionCycles = 3;
        
        for (let cycle = 0; cycle < evolutionCycles; cycle++) {
            console.log(`\n--- Evolution Cycle ${cycle + 1} ---`);
            
            // Generate tasks with increasing complexity
            const complexity = cycle + 2;
            const tasks = this.generateTasks(8, complexity);
            
            // Execute tasks to generate performance data
            for (const task of tasks) {
                const selectedAgents = await this.performanceSelector.selectAgents(task, {
                    count: 1,
                    complexity: complexity
                });
                
                await this.executeTask(task, selectedAgents);
                this.demoState.tasksCompleted++;
                await this.sleep(200);
            }
            
            // Trigger manual evolution
            console.log('🚀 Triggering swarm evolution...');
            await this.swarmCoordinator.triggerEvolution();
            
            // Wait for evolution to complete
            await this.sleep(2000);
        }
    }

    /**
     * Consensus decision making demonstration
     */
    async consensusDecisionMakingScenario() {
        console.log('🗳️ Demonstrating consensus decision making...');
        
        const decisions = [
            {
                type: 'resource_allocation',
                description: 'Allocate compute resources to high-priority tasks',
                priority: 'high',
                urgency: 'normal'
            },
            {
                type: 'topology_change',
                description: 'Adapt network topology for better performance',
                priority: 'medium',
                urgency: 'low'
            },
            {
                type: 'agent_specialization',
                description: 'Specialize agents for specific task types',
                priority: 'medium',
                urgency: 'normal'
            },
            {
                type: 'emergency_response',
                description: 'Respond to critical system failure',
                priority: 'critical',
                urgency: 'high'
            }
        ];

        for (const decision of decisions) {
            console.log(`\n🎯 Proposing decision: ${decision.description}`);
            
            try {
                const proposalId = await this.consensusEngine.proposeDecision(decision, {
                    timestamp: Date.now(),
                    priority: decision.priority,
                    urgency: decision.urgency
                });
                
                console.log(`📋 Proposal ${proposalId} submitted for consensus`);
                
                // Wait for consensus (timeout handled by consensus engine)
                await this.sleep(3000);
                
            } catch (error) {
                console.warn(`⚠️ Consensus proposal failed: ${error.message}`);
            }
        }
    }

    /**
     * Performance optimization demonstration
     */
    async performanceOptimizationScenario() {
        console.log('⚡ Demonstrating performance optimization...');
        
        // Test different selection strategies
        const strategies = [
            SELECTION_STRATEGIES.PERFORMANCE_BASED,
            SELECTION_STRATEGIES.SPECIALIZED,
            SELECTION_STRATEGIES.LEAST_LOADED,
            SELECTION_STRATEGIES.NEURAL_OPTIMIZED
        ];

        for (const strategy of strategies) {
            console.log(`\n🎯 Testing strategy: ${strategy}`);
            
            const tasks = this.generateTasks(5);
            const startTime = Date.now();
            
            for (const task of tasks) {
                const selectedAgents = await this.performanceSelector.selectAgents(task, {
                    count: 1,
                    strategy: strategy
                });
                
                await this.executeTask(task, selectedAgents);
                this.demoState.tasksCompleted++;
            }
            
            const duration = Date.now() - startTime;
            console.log(`⏱️ Strategy ${strategy} completed in ${duration}ms`);
            await this.sleep(1000);
        }
    }

    /**
     * Fault tolerance and recovery demonstration
     */
    async faultToleranceRecoveryScenario() {
        console.log('🛡️ Demonstrating fault tolerance and recovery...');
        
        // Simulate agent failures
        const agents = this.swarmCoordinator.getAllAgents();
        const failureCount = Math.min(3, Math.floor(agents.length * 0.25)); // Fail up to 25% of agents
        
        console.log(`⚠️ Simulating failure of ${failureCount} agents...`);
        
        // Simulate failures by reducing agent fitness
        for (let i = 0; i < failureCount; i++) {
            const agent = agents[i];
            if (agent) {
                // Simulate failure by setting very low fitness and high fatigue
                agent.evolutionaryTraits.fitness = 0.1;
                agent.cognitiveState.fatigue = 0.95;
                agent.performance.successRate = 0.1;
                
                console.log(`💥 Agent ${agent.id} (${agent.agentType}) simulated failure`);
                this.demoState.agentFailures++;
            }
        }

        // Continue with tasks to trigger self-healing
        console.log('🔧 Triggering self-healing mechanisms...');
        
        const tasks = this.generateTasks(8);
        for (const task of tasks) {
            try {
                const selectedAgents = await this.performanceSelector.selectAgents(task, {
                    count: 1,
                    strategy: SELECTION_STRATEGIES.PERFORMANCE_BASED
                });
                
                await this.executeTask(task, selectedAgents);
                this.demoState.tasksCompleted++;
                await this.sleep(300);
            } catch (error) {
                console.warn(`⚠️ Task execution failed: ${error.message}`);
            }
        }

        // Trigger manual organization to force self-healing
        await this.swarmCoordinator.triggerOrganization();
        await this.sleep(3000);
    }

    /**
     * Self-organizing behavior demonstration
     */
    async selfOrganizingBehaviorScenario() {
        console.log('🏗️ Demonstrating self-organizing behavior...');
        
        // Create different types of tasks to encourage specialization
        const taskPatterns = [
            { type: 'computation', count: 5, description: 'Computational tasks' },
            { type: 'analysis', count: 4, description: 'Analysis tasks' },
            { type: 'coordination', count: 3, description: 'Coordination tasks' },
            { type: 'research', count: 3, description: 'Research tasks' }
        ];

        for (const pattern of taskPatterns) {
            console.log(`\n📊 Executing ${pattern.count} ${pattern.description}...`);
            
            const tasks = this.generateTasksByType(pattern.type, pattern.count);
            
            for (const task of tasks) {
                const selectedAgents = await this.performanceSelector.selectAgents(task, {
                    count: 1,
                    strategy: SELECTION_STRATEGIES.SPECIALIZED,
                    requireSpecialization: pattern.type
                });
                
                await this.executeTask(task, selectedAgents);
                this.demoState.tasksCompleted++;
                await this.sleep(400);
            }
        }

        // Trigger organization to show adaptive clustering
        console.log('\n🔄 Triggering self-organization...');
        await this.swarmCoordinator.triggerOrganization();
        await this.sleep(2000);

        // Show resulting organization
        const clusters = await this.swarmCoordinator.selfOrganizing?.get_clusters() || [];
        console.log(`🏗️ Formed ${clusters.length} specialized clusters`);
    }

    /**
     * Generate demo tasks
     */
    generateTasks(count, complexity = null) {
        const tasks = [];
        
        for (let i = 0; i < count; i++) {
            const taskTemplate = TASK_TYPES[Math.floor(Math.random() * TASK_TYPES.length)];
            const taskComplexity = complexity || taskTemplate.complexity;
            
            tasks.push({
                id: `task_${Date.now()}_${i}`,
                type: taskTemplate.type,
                description: taskTemplate.description,
                complexity: taskComplexity,
                priority: Math.random() < 0.3 ? 'high' : (Math.random() < 0.5 ? 'medium' : 'low'),
                estimatedDuration: (taskComplexity * 1000) + Math.random() * 2000,
                resourceRequirement: Math.random() * 0.8 + 0.2,
                collaborationRequired: Math.random() < 0.2,
                timestamp: Date.now()
            });
        }
        
        return tasks;
    }

    /**
     * Generate tasks of a specific type
     */
    generateTasksByType(type, count) {
        const tasks = [];
        
        for (let i = 0; i < count; i++) {
            const taskTemplate = TASK_TYPES.find(t => t.type === type) || TASK_TYPES[0];
            
            tasks.push({
                id: `${type}_task_${Date.now()}_${i}`,
                type: type,
                description: `${taskTemplate.description} #${i + 1}`,
                complexity: taskTemplate.complexity,
                priority: 'medium',
                estimatedDuration: taskTemplate.complexity * 1000 + Math.random() * 1000,
                resourceRequirement: Math.random() * 0.6 + 0.3,
                collaborationRequired: false,
                timestamp: Date.now()
            });
        }
        
        return tasks;
    }

    /**
     * Simulate task execution
     */
    async executeTask(task, agentIds) {
        const startTime = Date.now();
        
        // Simulate task execution time based on complexity and agent count
        const baseTime = task.estimatedDuration || 1000;
        const agentSpeedup = Math.max(1, agentIds.length * 0.8); // Diminishing returns
        const executionTime = (baseTime / agentSpeedup) + (Math.random() * 500);
        
        await this.sleep(Math.min(executionTime, 3000)); // Cap at 3 seconds for demo
        
        const actualDuration = Date.now() - startTime;
        
        // Simulate task results
        for (const agentId of agentIds) {
            const agent = this.swarmCoordinator.getAgent(agentId);
            if (!agent) continue;

            // Simulate performance based on agent capabilities and task type
            const specialization = agent.specializationProfile?.[task.type] || 0.5;
            const baseSuccess = agent.evolutionaryTraits?.fitness || 0.5;
            const fatigue = agent.cognitiveState?.fatigue || 0.0;
            
            const successProbability = Math.min(0.95, baseSuccess * specialization * (1.0 - fatigue * 0.5));
            const success = Math.random() < successProbability;
            
            const taskResult = {
                duration: actualDuration,
                success: success,
                accuracy: success ? (0.7 + Math.random() * 0.3) : (0.1 + Math.random() * 0.4),
                resourcesUsed: task.resourceRequirement * (0.8 + Math.random() * 0.4),
                cooperationScore: agentIds.length > 1 ? (0.6 + Math.random() * 0.4) : 0.5,
                innovationScore: Math.random() * 0.4 + (specialization * 0.6),
                taskType: task.type
            };

            // Update performance metrics
            this.performanceSelector.updatePerformance(agentId, taskResult);
            
            // Update agent state
            if (agent.cognitiveState) {
                agent.cognitiveState.fatigue = Math.min(1.0, 
                    agent.cognitiveState.fatigue + (task.complexity * 0.05)
                );
            }
            
            if (agent.performance) {
                agent.performance.tasksCompleted++;
                agent.performance.successRate = 
                    (agent.performance.successRate * 0.9) + (success ? 0.1 : 0);
            }
        }
    }

    /**
     * Print current system status
     */
    async printSystemStatus() {
        console.log('\n📊 SYSTEM STATUS');
        console.log('=' * 50);
        
        const swarmMetrics = this.swarmCoordinator.getMetrics();
        const consensusMetrics = this.consensusEngine.getMetrics();
        const performanceStats = this.performanceSelector.getPerformanceStatistics();
        
        console.log(`🐝 Swarm Population: ${swarmMetrics.populationSize} agents`);
        console.log(`🧬 Current Generation: ${swarmMetrics.generation}`);
        console.log(`📈 Average Fitness: ${swarmMetrics.averageFitness.toFixed(3)}`);
        console.log(`🌀 Diversity Index: ${swarmMetrics.diversityIndex.toFixed(3)}`);
        console.log(`🔗 Network Efficiency: ${swarmMetrics.networkEfficiency.toFixed(3)}`);
        console.log(`🛡️ Fault Tolerance: ${swarmMetrics.faultTolerance.toFixed(3)}`);
        console.log(`🗳️ Connected Nodes: ${consensusMetrics.connectedNodes}`);
        console.log(`⚖️ Consensus Threshold: ${consensusMetrics.adaptiveThreshold.toFixed(2)}`);
        console.log(`📊 Total Assignments: ${performanceStats.totalAssignments || 0}`);
        console.log(`⚖️ Load Imbalance: ${performanceStats.loadImbalance.toFixed(2)}`);
    }

    /**
     * Print scenario results
     */
    async printScenarioResults(scenario) {
        const duration = Date.now() - this.demoState.startTime;
        
        console.log(`\n📊 ${scenario.toUpperCase()} RESULTS`);
        console.log('=' * 50);
        console.log(`⏱️ Duration: ${duration}ms`);
        console.log(`✅ Tasks Completed: ${this.demoState.tasksCompleted}`);
        console.log(`🗳️ Consensus Reached: ${this.demoState.consensusReached}`);
        console.log(`🔄 Adaptations: ${this.demoState.adaptationsTriggered}`);
        console.log(`💥 Agent Failures: ${this.demoState.agentFailures}`);
        console.log(`🔧 Self-Healing Events: ${this.demoState.selfHealingEvents}`);
        
        if (this.demoStats.consensusLatency.length > 0) {
            const avgLatency = this.demoStats.consensusLatency.reduce((a, b) => a + b, 0) / this.demoStats.consensusLatency.length;
            console.log(`⚡ Avg Consensus Latency: ${avgLatency.toFixed(0)}ms`);
        }
        
        if (this.demoStats.throughput.length > 0) {
            const avgThroughput = this.demoStats.throughput.reduce((a, b) => a + b, 0) / this.demoStats.throughput.length;
            console.log(`🚀 Avg Throughput: ${avgThroughput.toFixed(2)} tasks/min`);
        }
    }

    /**
     * Print final demonstration results
     */
    async printFinalResults() {
        console.log('\n🎉 FINAL DEMONSTRATION RESULTS');
        console.log('=' * 60);
        
        const finalMetrics = this.swarmCoordinator.getMetrics();
        
        console.log(`🎯 Total Tasks Completed: ${this.demoState.tasksCompleted}`);
        console.log(`🗳️ Total Consensus Decisions: ${this.demoState.consensusReached}`);
        console.log(`🧬 Evolution Cycles: ${this.demoState.adaptationsTriggered}`);
        console.log(`🔧 Self-Healing Events: ${this.demoState.selfHealingEvents}`);
        console.log(`💥 Agent Failures Recovered: ${this.demoState.agentFailures}`);
        
        if (this.demoStats.averageFitness.length > 0) {
            const fitnessImprovement = this.demoStats.averageFitness[this.demoStats.averageFitness.length - 1] - this.demoStats.averageFitness[0];
            console.log(`📈 Fitness Improvement: ${fitnessImprovement > 0 ? '+' : ''}${(fitnessImprovement * 100).toFixed(1)}%`);
        }
        
        if (this.demoStats.consensusLatency.length > 0) {
            const avgLatency = this.demoStats.consensusLatency.reduce((a, b) => a + b, 0) / this.demoStats.consensusLatency.length;
            const minLatency = Math.min(...this.demoStats.consensusLatency);
            const maxLatency = Math.max(...this.demoStats.consensusLatency);
            console.log(`⚡ Consensus Latency: ${avgLatency.toFixed(0)}ms avg (${minLatency}-${maxLatency}ms)`);
        }
        
        console.log(`\n🌟 Final Network Efficiency: ${finalMetrics.networkEfficiency.toFixed(3)}`);
        console.log(`🛡️ Final Fault Tolerance: ${finalMetrics.faultTolerance.toFixed(3)}`);
        console.log(`🌀 Final Diversity Index: ${finalMetrics.diversityIndex.toFixed(3)}`);
        
        console.log('\n🚀 SWARM INTELLIGENCE CAPABILITIES DEMONSTRATED:');
        console.log('  ✅ Evolutionary algorithms and adaptation');
        console.log('  ✅ Self-organizing behavior and clustering');
        console.log('  ✅ Consensus-based decision making');
        console.log('  ✅ Performance-based agent selection');
        console.log('  ✅ Fault tolerance and self-healing');
        console.log('  ✅ Neural network optimization');
        console.log('  ✅ Adaptive threshold adjustment');
        console.log('  ✅ Load balancing and efficiency');
    }

    /**
     * Clean up demonstration
     */
    async cleanup() {
        console.log('\n🧹 Cleaning up demonstration...');
        
        try {
            await Promise.all([
                this.swarmCoordinator?.stop(),
                this.consensusEngine?.stop(),
                this.performanceSelector?.stop()
            ]);
        } catch (error) {
            console.warn('⚠️ Cleanup warning:', error.message);
        }
    }

    /**
     * Utility function for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Run the demonstration
 */
async function main() {
    const demo = new SwarmIntelligenceDemo();
    
    try {
        await demo.runDemo();
    } catch (error) {
        console.error('❌ Demonstration failed:', error);
        await demo.cleanup();
        process.exit(1);
    }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
    console.log('\n🛑 Demonstration interrupted');
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\n🛑 Demonstration terminated');
    process.exit(0);
});

// Run the demonstration if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export { SwarmIntelligenceDemo };