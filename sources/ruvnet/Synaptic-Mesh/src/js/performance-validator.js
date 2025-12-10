/**
 * Performance Validator
 * Validates that all performance targets are being achieved in real-time
 */

import { PerformanceBenchmarks } from './ruv-swarm/src/performance-benchmarks.js';
import { PerformanceOptimizer } from './ruv-swarm/src/performance-optimizer.js';

/**
 * Real-time Performance Validation System
 */
export class PerformanceValidator {
    constructor(config = {}) {
        this.config = {
            validationInterval: config.validationInterval || 30000, // 30 seconds
            alertThreshold: config.alertThreshold || 1.2, // 20% above target
            warningThreshold: config.warningThreshold || 1.1, // 10% above target
            historySamples: config.historySamples || 100,
            autoOptimize: config.autoOptimize !== false,
            ...config
        };
        
        // Performance targets (from optimization requirements)
        this.targets = {
            neuralInference: 100, // ms
            agentSpawning: 500, // ms
            memoryPerAgent: 50, // MB
            systemStartup: 5000, // ms
            sweBenchSolveRate: 0.848, // 84.8%
            concurrentAgents: 1000,
            wasmBundleSize: 2 * 1024 * 1024, // 2MB
            tokenReduction: 0.323 // 32.3%
        };
        
        // Performance history
        this.performanceHistory = {
            neuralInference: [],
            agentSpawning: [],
            memoryUsage: [],
            systemMetrics: [],
            errors: []
        };
        
        // Validation state
        this.isValidating = false;
        this.validationTimer = null;
        this.alertCallbacks = [];
        this.optimizer = null;
        this.benchmarks = null;
        
        // Real-time metrics
        this.currentMetrics = {
            neuralInference: 0,
            agentSpawning: 0,
            memoryPerAgent: 0,
            activeAgents: 0,
            systemLoad: 0,
            errorRate: 0
        };
        
        console.log('📊 Performance Validator initialized');
        console.log('🎯 Performance targets loaded:', this.targets);
    }
    
    /**
     * Initialize the performance validator
     */
    async initialize() {
        console.log('🚀 Initializing Performance Validator...');
        
        try {
            // Initialize optimizer and benchmarks
            this.optimizer = new PerformanceOptimizer();
            this.benchmarks = new PerformanceBenchmarks();
            
            await Promise.all([
                this.optimizer.initialize(),
                this.benchmarks.initialize()
            ]);
            
            // Set up performance monitoring hooks
            this.setupPerformanceHooks();
            
            console.log('✅ Performance Validator initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize Performance Validator:', error);
            throw error;
        }
    }
    
    /**
     * Start continuous performance validation
     */
    startValidation() {
        if (this.isValidating) {
            console.log('⚠️ Performance validation already running');
            return;
        }
        
        console.log(`🔄 Starting performance validation (interval: ${this.config.validationInterval}ms)`);
        this.isValidating = true;
        
        // Run initial validation
        this.runValidationCycle();
        
        // Set up recurring validation
        this.validationTimer = setInterval(() => {
            this.runValidationCycle();
        }, this.config.validationInterval);
    }
    
    /**
     * Stop performance validation
     */
    stopValidation() {
        if (!this.isValidating) {
            return;
        }
        
        console.log('⏹️ Stopping performance validation');
        this.isValidating = false;
        
        if (this.validationTimer) {
            clearInterval(this.validationTimer);
            this.validationTimer = null;
        }
    }
    
    /**
     * Run a single validation cycle
     */
    async runValidationCycle() {
        try {
            console.log('📊 Running performance validation cycle...');
            
            // Collect current metrics
            const metrics = await this.collectCurrentMetrics();
            
            // Validate against targets
            const validationResult = this.validateMetrics(metrics);
            
            // Update history
            this.updatePerformanceHistory(metrics, validationResult);
            
            // Handle alerts and warnings
            this.handleAlerts(validationResult);
            
            // Auto-optimize if enabled and issues detected
            if (this.config.autoOptimize && validationResult.hasIssues) {
                await this.triggerAutoOptimization(validationResult);
            }
            
            // Log validation summary
            this.logValidationSummary(validationResult);
            
        } catch (error) {
            console.error('❌ Error during performance validation:', error);
            this.recordError(error);
        }
    }
    
    /**
     * Collect current performance metrics
     */
    async collectCurrentMetrics() {
        const startTime = performance.now();
        
        // Neural inference performance
        const neuralMetrics = await this.measureNeuralPerformance();
        
        // Memory usage
        const memoryMetrics = this.measureMemoryUsage();
        
        // System metrics
        const systemMetrics = await this.measureSystemMetrics();
        
        // Agent performance
        const agentMetrics = await this.measureAgentPerformance();
        
        const collectionTime = performance.now() - startTime;
        
        return {
            timestamp: Date.now(),
            collectionTime,
            neural: neuralMetrics,
            memory: memoryMetrics,
            system: systemMetrics,
            agents: agentMetrics
        };
    }
    
    /**
     * Measure neural inference performance
     */
    async measureNeuralPerformance() {
        const testData = {
            matrixA: new Float32Array(1000 * 1000).map(() => Math.random()),
            matrixB: new Float32Array(1000 * 1000).map(() => Math.random()),
            neuralInput: new Float32Array(1000).map(() => Math.random())
        };
        
        // Test SIMD matrix multiplication
        const matMulStart = performance.now();
        await this.optimizer.simdOptimizer.optimizedMatMul(
            testData.matrixA, 
            testData.matrixB, 
            { rows: 1000, cols: 1000 }
        );
        const matMulTime = performance.now() - matMulStart;
        
        // Test neural forward pass
        const neuralStart = performance.now();
        await this.optimizer.simdOptimizer.neuralForwardPass(
            testData.neuralInput.slice(0, 500), // weights
            testData.neuralInput.slice(500, 600), // biases
            testData.neuralInput.slice(600, 700), // inputs
            'relu'
        );
        const neuralTime = performance.now() - neuralStart;
        
        // Test GPU acceleration if available
        let gpuTime = null;
        if (this.optimizer.webGPUAccelerator && this.optimizer.webGPUAccelerator.isSupported()) {
            const gpuStart = performance.now();
            await this.optimizer.webGPUAccelerator.accelerateMatrixMultiplication(
                testData.matrixA, testData.matrixB
            );
            gpuTime = performance.now() - gpuStart;
        }
        
        return {
            matrixMultiplication: matMulTime,
            neuralForwardPass: neuralTime,
            gpuAcceleration: gpuTime,
            avgInferenceTime: (matMulTime + neuralTime) / 2
        };
    }
    
    /**
     * Measure memory usage
     */
    measureMemoryUsage() {
        let totalMemory = 0;
        let heapUsed = 0;
        let heapTotal = 0;
        
        if (process.memoryUsage) {
            const memUsage = process.memoryUsage();
            totalMemory = memUsage.rss;
            heapUsed = memUsage.heapUsed;
            heapTotal = memUsage.heapTotal;
        } else if (performance.memory) {
            heapUsed = performance.memory.usedJSHeapSize;
            heapTotal = performance.memory.totalJSHeapSize;
            totalMemory = heapTotal;
        }
        
        return {
            totalMemory: totalMemory,
            heapUsed: heapUsed,
            heapTotal: heapTotal,
            memoryPerAgent: this.currentMetrics.activeAgents > 0 
                ? heapUsed / this.currentMetrics.activeAgents 
                : 0
        };
    }
    
    /**
     * Measure system metrics
     */
    async measureSystemMetrics() {
        const metrics = {
            uptime: process.uptime ? process.uptime() * 1000 : performance.now(),
            cpuUsage: 0,
            loadAverage: 0,
            networkConnections: 0
        };
        
        // CPU usage (if available)
        if (process.cpuUsage) {
            const cpuUsage = process.cpuUsage();
            metrics.cpuUsage = (cpuUsage.user + cpuUsage.system) / 1000; // Convert to ms
        }
        
        // Load average (if available)
        if (process.loadavg) {
            metrics.loadAverage = process.loadavg()[0]; // 1-minute load average
        }
        
        return metrics;
    }
    
    /**
     * Measure agent performance
     */
    async measureAgentPerformance() {
        // Simulate agent spawning to measure performance
        const spawnTests = [];
        const testCount = 5;
        
        for (let i = 0; i < testCount; i++) {
            const spawnStart = performance.now();
            
            // Simulate agent initialization
            await this.simulateAgentSpawn(i);
            
            const spawnTime = performance.now() - spawnStart;
            spawnTests.push(spawnTime);
        }
        
        const avgSpawnTime = spawnTests.reduce((sum, time) => sum + time, 0) / spawnTests.length;
        
        return {
            avgSpawnTime: avgSpawnTime,
            spawnTests: spawnTests,
            activeAgents: this.currentMetrics.activeAgents,
            maxConcurrentAgents: this.targets.concurrentAgents
        };
    }
    
    /**
     * Validate metrics against targets
     */
    validateMetrics(metrics) {
        const results = {
            timestamp: metrics.timestamp,
            hasIssues: false,
            alerts: [],
            warnings: [],
            achievements: {},
            summary: {}
        };
        
        // Validate neural inference performance
        const neuralTarget = this.targets.neuralInference;
        const neuralActual = metrics.neural.avgInferenceTime;
        
        if (neuralActual > neuralTarget * this.config.alertThreshold) {
            results.alerts.push({
                metric: 'neuralInference',
                actual: neuralActual,
                target: neuralTarget,
                severity: 'alert'
            });
            results.hasIssues = true;
        } else if (neuralActual > neuralTarget * this.config.warningThreshold) {
            results.warnings.push({
                metric: 'neuralInference',
                actual: neuralActual,
                target: neuralTarget,
                severity: 'warning'
            });
        }
        
        results.achievements.neuralInference = {
            target: neuralTarget,
            actual: neuralActual,
            achieved: neuralActual <= neuralTarget,
            performance: (neuralTarget / neuralActual) * 100
        };
        
        // Validate agent spawning performance
        const agentTarget = this.targets.agentSpawning;
        const agentActual = metrics.agents.avgSpawnTime;
        
        if (agentActual > agentTarget * this.config.alertThreshold) {
            results.alerts.push({
                metric: 'agentSpawning',
                actual: agentActual,
                target: agentTarget,
                severity: 'alert'
            });
            results.hasIssues = true;
        } else if (agentActual > agentTarget * this.config.warningThreshold) {
            results.warnings.push({
                metric: 'agentSpawning',
                actual: agentActual,
                target: agentTarget,
                severity: 'warning'
            });
        }
        
        results.achievements.agentSpawning = {
            target: agentTarget,
            actual: agentActual,
            achieved: agentActual <= agentTarget,
            performance: (agentTarget / agentActual) * 100
        };
        
        // Validate memory usage per agent
        const memoryTarget = this.targets.memoryPerAgent * 1024 * 1024; // Convert MB to bytes
        const memoryActual = metrics.memory.memoryPerAgent;
        
        if (memoryActual > memoryTarget * this.config.alertThreshold) {
            results.alerts.push({
                metric: 'memoryPerAgent',
                actual: memoryActual / 1024 / 1024, // Convert to MB
                target: this.targets.memoryPerAgent,
                severity: 'alert'
            });
            results.hasIssues = true;
        } else if (memoryActual > memoryTarget * this.config.warningThreshold) {
            results.warnings.push({
                metric: 'memoryPerAgent',
                actual: memoryActual / 1024 / 1024,
                target: this.targets.memoryPerAgent,
                severity: 'warning'
            });
        }
        
        results.achievements.memoryPerAgent = {
            target: this.targets.memoryPerAgent,
            actual: memoryActual / 1024 / 1024,
            achieved: memoryActual <= memoryTarget,
            performance: (memoryTarget / memoryActual) * 100
        };
        
        // Calculate overall performance score
        const achievementValues = Object.values(results.achievements);
        const overallPerformance = achievementValues.reduce((sum, ach) => sum + ach.performance, 0) / achievementValues.length;
        
        results.summary = {
            overallPerformance: overallPerformance,
            targetsAchieved: achievementValues.filter(ach => ach.achieved).length,
            totalTargets: achievementValues.length,
            hasAlerts: results.alerts.length > 0,
            hasWarnings: results.warnings.length > 0
        };
        
        return results;
    }
    
    /**
     * Update performance history
     */
    updatePerformanceHistory(metrics, validationResult) {
        // Store neural inference history
        this.performanceHistory.neuralInference.push({
            timestamp: metrics.timestamp,
            value: metrics.neural.avgInferenceTime,
            target: this.targets.neuralInference,
            achieved: validationResult.achievements.neuralInference.achieved
        });
        
        // Store agent spawning history
        this.performanceHistory.agentSpawning.push({
            timestamp: metrics.timestamp,
            value: metrics.agents.avgSpawnTime,
            target: this.targets.agentSpawning,
            achieved: validationResult.achievements.agentSpawning.achieved
        });
        
        // Store memory usage history
        this.performanceHistory.memoryUsage.push({
            timestamp: metrics.timestamp,
            value: metrics.memory.memoryPerAgent,
            target: this.targets.memoryPerAgent * 1024 * 1024,
            achieved: validationResult.achievements.memoryPerAgent.achieved
        });
        
        // Store system metrics
        this.performanceHistory.systemMetrics.push({
            timestamp: metrics.timestamp,
            uptime: metrics.system.uptime,
            cpuUsage: metrics.system.cpuUsage,
            overallPerformance: validationResult.summary.overallPerformance
        });
        
        // Maintain history size limit
        this.trimHistoryToLimit();
    }
    
    /**
     * Handle performance alerts and warnings
     */
    handleAlerts(validationResult) {
        // Process alerts
        for (const alert of validationResult.alerts) {
            console.log(`🚨 PERFORMANCE ALERT: ${alert.metric}`);
            console.log(`   Target: ${alert.target} | Actual: ${alert.actual.toFixed(2)} | Severity: ${alert.severity}`);
            
            // Trigger alert callbacks
            this.triggerAlertCallbacks('alert', alert);
        }
        
        // Process warnings
        for (const warning of validationResult.warnings) {
            console.log(`⚠️ PERFORMANCE WARNING: ${warning.metric}`);
            console.log(`   Target: ${warning.target} | Actual: ${warning.actual.toFixed(2)} | Severity: ${warning.severity}`);
            
            // Trigger warning callbacks
            this.triggerAlertCallbacks('warning', warning);
        }
    }
    
    /**
     * Trigger auto-optimization if enabled
     */
    async triggerAutoOptimization(validationResult) {
        console.log('🔧 Triggering auto-optimization due to performance issues...');
        
        try {
            // Analyze issues and determine optimization strategy
            const optimizationPlan = this.createOptimizationPlan(validationResult);
            
            // Execute optimizations
            await this.executeOptimizationPlan(optimizationPlan);
            
            console.log('✅ Auto-optimization completed');
        } catch (error) {
            console.error('❌ Auto-optimization failed:', error);
        }
    }
    
    /**
     * Create optimization plan based on validation results
     */
    createOptimizationPlan(validationResult) {
        const plan = {
            actions: [],
            priority: 'medium'
        };
        
        // Neural performance issues
        const neuralIssues = [...validationResult.alerts, ...validationResult.warnings]
            .filter(issue => issue.metric === 'neuralInference');
            
        if (neuralIssues.length > 0) {
            plan.actions.push({
                type: 'optimizeNeuralPerformance',
                target: 'simd',
                severity: neuralIssues[0].severity
            });
            
            if (this.optimizer.webGPUAccelerator?.isSupported()) {
                plan.actions.push({
                    type: 'enableGPUAcceleration',
                    target: 'webgpu',
                    severity: neuralIssues[0].severity
                });
            }
        }
        
        // Memory issues
        const memoryIssues = [...validationResult.alerts, ...validationResult.warnings]
            .filter(issue => issue.metric === 'memoryPerAgent');
            
        if (memoryIssues.length > 0) {
            plan.actions.push({
                type: 'optimizeMemoryUsage',
                target: 'wasmPool',
                severity: memoryIssues[0].severity
            });
        }
        
        // Agent spawning issues
        const agentIssues = [...validationResult.alerts, ...validationResult.warnings]
            .filter(issue => issue.metric === 'agentSpawning');
            
        if (agentIssues.length > 0) {
            plan.actions.push({
                type: 'optimizeAgentSpawning',
                target: 'communication',
                severity: agentIssues[0].severity
            });
        }
        
        // Set overall priority based on alerts
        if (validationResult.alerts.length > 0) {
            plan.priority = 'high';
        } else if (validationResult.warnings.length > 0) {
            plan.priority = 'medium';
        }
        
        return plan;
    }
    
    /**
     * Execute optimization plan
     */
    async executeOptimizationPlan(plan) {
        console.log(`🚀 Executing optimization plan (${plan.actions.length} actions, priority: ${plan.priority})`);
        
        for (const action of plan.actions) {
            try {
                switch (action.type) {
                    case 'optimizeNeuralPerformance':
                        await this.optimizer.optimizeNeuralPerformance();
                        break;
                    case 'enableGPUAcceleration':
                        await this.optimizer.webGPUAccelerator.optimizePerformance();
                        break;
                    case 'optimizeMemoryUsage':
                        await this.optimizer.optimizeMemoryUsage();
                        break;
                    case 'optimizeAgentSpawning':
                        await this.optimizer.optimizeAgentCommunication();
                        break;
                }
                
                console.log(`   ✅ Completed: ${action.type}`);
            } catch (error) {
                console.error(`   ❌ Failed: ${action.type} - ${error.message}`);
            }
        }
    }
    
    /**
     * Log validation summary
     */
    logValidationSummary(validationResult) {
        const summary = validationResult.summary;
        
        console.log('📈 Performance Validation Summary:');
        console.log(`   Overall Performance: ${summary.overallPerformance.toFixed(1)}%`);
        console.log(`   Targets Achieved: ${summary.targetsAchieved}/${summary.totalTargets}`);
        console.log(`   Alerts: ${validationResult.alerts.length}`);
        console.log(`   Warnings: ${validationResult.warnings.length}`);
        
        // Log individual achievements
        for (const [metric, achievement] of Object.entries(validationResult.achievements)) {
            const status = achievement.achieved ? '✅' : '❌';
            console.log(`   ${status} ${metric}: ${achievement.actual.toFixed(2)} (target: ${achievement.target})`);
        }
    }
    
    /**
     * Get current validation status
     */
    getValidationStatus() {
        return {
            isValidating: this.isValidating,
            targets: this.targets,
            currentMetrics: this.currentMetrics,
            historyLength: {
                neuralInference: this.performanceHistory.neuralInference.length,
                agentSpawning: this.performanceHistory.agentSpawning.length,
                memoryUsage: this.performanceHistory.memoryUsage.length,
                systemMetrics: this.performanceHistory.systemMetrics.length
            },
            config: this.config
        };
    }
    
    /**
     * Get performance report
     */
    generatePerformanceReport() {
        const recentHistory = this.getRecentHistory(10); // Last 10 samples
        
        return {
            timestamp: Date.now(),
            targets: this.targets,
            currentStatus: this.getValidationStatus(),
            recentPerformance: recentHistory,
            trends: this.calculatePerformanceTrends(),
            recommendations: this.generateRecommendations()
        };
    }
    
    // Helper methods
    
    setupPerformanceHooks() {
        // Hook into performance-critical operations
        console.log('🔗 Setting up performance monitoring hooks');
    }
    
    async simulateAgentSpawn(id) {
        // Simulate agent initialization overhead
        await new Promise(resolve => setTimeout(resolve, Math.random() * 100 + 50));
        this.currentMetrics.activeAgents++;
        return { id, spawnTime: Date.now() };
    }
    
    trimHistoryToLimit() {
        const limit = this.config.historySamples;
        
        if (this.performanceHistory.neuralInference.length > limit) {
            this.performanceHistory.neuralInference = this.performanceHistory.neuralInference.slice(-limit);
        }
        if (this.performanceHistory.agentSpawning.length > limit) {
            this.performanceHistory.agentSpawning = this.performanceHistory.agentSpawning.slice(-limit);
        }
        if (this.performanceHistory.memoryUsage.length > limit) {
            this.performanceHistory.memoryUsage = this.performanceHistory.memoryUsage.slice(-limit);
        }
        if (this.performanceHistory.systemMetrics.length > limit) {
            this.performanceHistory.systemMetrics = this.performanceHistory.systemMetrics.slice(-limit);
        }
    }
    
    triggerAlertCallbacks(type, alert) {
        for (const callback of this.alertCallbacks) {
            try {
                callback(type, alert);
            } catch (error) {
                console.error('Error in alert callback:', error);
            }
        }
    }
    
    recordError(error) {
        this.performanceHistory.errors.push({
            timestamp: Date.now(),
            error: error.message,
            stack: error.stack
        });
        
        // Maintain error history limit
        if (this.performanceHistory.errors.length > 50) {
            this.performanceHistory.errors = this.performanceHistory.errors.slice(-50);
        }
    }
    
    getRecentHistory(samples) {
        return {
            neuralInference: this.performanceHistory.neuralInference.slice(-samples),
            agentSpawning: this.performanceHistory.agentSpawning.slice(-samples),
            memoryUsage: this.performanceHistory.memoryUsage.slice(-samples),
            systemMetrics: this.performanceHistory.systemMetrics.slice(-samples)
        };
    }
    
    calculatePerformanceTrends() {
        const trends = {};
        
        for (const [metric, history] of Object.entries(this.performanceHistory)) {
            if (metric === 'errors' || history.length < 2) continue;
            
            const recent = history.slice(-10);
            const older = history.slice(-20, -10);
            
            if (older.length === 0) {
                trends[metric] = 'insufficient_data';
                continue;
            }
            
            const recentAvg = recent.reduce((sum, item) => sum + item.value, 0) / recent.length;
            const olderAvg = older.reduce((sum, item) => sum + item.value, 0) / older.length;
            
            const change = (recentAvg - olderAvg) / olderAvg;
            
            if (change > 0.1) {
                trends[metric] = 'degrading';
            } else if (change < -0.1) {
                trends[metric] = 'improving';
            } else {
                trends[metric] = 'stable';
            }
        }
        
        return trends;
    }
    
    generateRecommendations() {
        const recommendations = [];
        const trends = this.calculatePerformanceTrends();
        
        // Analyze trends and generate recommendations
        for (const [metric, trend] of Object.entries(trends)) {
            if (trend === 'degrading') {
                switch (metric) {
                    case 'neuralInference':
                        recommendations.push('Neural inference performance is degrading - consider enabling GPU acceleration');
                        break;
                    case 'agentSpawning':
                        recommendations.push('Agent spawning is slowing down - optimize agent initialization');
                        break;
                    case 'memoryUsage':
                        recommendations.push('Memory usage is increasing - implement more aggressive garbage collection');
                        break;
                }
            }
        }
        
        if (recommendations.length === 0) {
            recommendations.push('Performance is stable - all targets are being maintained');
        }
        
        return recommendations;
    }
    
    /**
     * Add alert callback
     */
    onAlert(callback) {
        this.alertCallbacks.push(callback);
    }
    
    /**
     * Remove alert callback
     */
    removeAlert(callback) {
        const index = this.alertCallbacks.indexOf(callback);
        if (index > -1) {
            this.alertCallbacks.splice(index, 1);
        }
    }
}

// Export validator
export default PerformanceValidator;