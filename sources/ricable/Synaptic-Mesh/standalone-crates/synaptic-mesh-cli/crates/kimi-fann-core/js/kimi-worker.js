/**
 * Kimi-FANN Web Worker for async neural network processing
 * Handles compute-intensive tasks off the main thread
 */

import init, {
    MicroExpert,
    EnhancedExpertRouter,
    ExpertMemoryManager,
    ExpertDomain,
    AsyncProcessor
} from '../pkg/kimi_fann_core.js';

// Worker state
let wasmModule = null;
let router = null;
let memoryManager = null;
let experts = new Map();

// Initialize WASM module in worker
async function initializeWorker() {
    try {
        wasmModule = await init();
        memoryManager = new ExpertMemoryManager(512); // 512MB
        router = new EnhancedExpertRouter();
        
        console.log('[Worker] Kimi-FANN initialized successfully');
        return true;
    } catch (error) {
        console.error('[Worker] Failed to initialize:', error);
        return false;
    }
}

// Task handlers
const taskHandlers = {
    async expertInference(payload) {
        const { prompt, domain, expertConfig } = payload;
        
        try {
            // Get or create expert
            let expert = experts.get(domain);
            if (!expert) {
                expert = new MicroExpert(ExpertDomain[domain]);
                if (expertConfig) {
                    expert.update_config(expertConfig);
                }
                experts.set(domain, expert);
                memoryManager.store_expert(expert);
            }
            
            // Tokenize input (simplified)
            const inputTokens = prompt.split('').slice(0, 32)
                .map(char => char.charCodeAt(0) / 1000.0);
            
            // Run inference
            const startTime = performance.now();
            const output = expert.predict(inputTokens);
            const inferenceTime = performance.now() - startTime;
            
            return {
                domain,
                output: Array.from(output),
                confidence: expert.get_confidence(),
                inferenceTimeMs: inferenceTime,
                metrics: expert.get_metrics()
            };
        } catch (error) {
            throw new Error(`Inference failed: ${error.message}`);
        }
    },

    async modelTraining(payload) {
        const { domain, trainingData, epochs = 50 } = payload;
        
        try {
            let expert = experts.get(domain);
            if (!expert) {
                expert = new MicroExpert(ExpertDomain[domain]);
                experts.set(domain, expert);
            }
            
            const { inputs, outputs } = trainingData;
            const startTime = performance.now();
            const mse = expert.train(inputs, outputs, epochs);
            const trainingTime = performance.now() - startTime;
            
            // Store updated expert
            memoryManager.store_expert(expert);
            
            return {
                domain,
                finalMse: mse,
                epochs,
                trainingTimeMs: trainingTime,
                success: true
            };
        } catch (error) {
            throw new Error(`Training failed: ${error.message}`);
        }
    },

    async memoryCompression(payload) {
        const { algorithm = 'lz4', quantization = 'float32' } = payload;
        
        try {
            memoryManager.set_compression_algorithm(algorithm);
            memoryManager.set_quantization_method(quantization);
            
            // Trigger optimization
            memoryManager.optimize_cache();
            memoryManager.defragment_memory();
            
            const stats = memoryManager.get_compression_stats();
            
            return {
                algorithm,
                quantization,
                compressionStats: stats,
                success: true
            };
        } catch (error) {
            throw new Error(`Compression failed: ${error.message}`);
        }
    },

    async performanceAnalysis(payload) {
        const { analysisType = 'comprehensive' } = payload;
        
        try {
            const memoryStats = memoryManager.get_memory_stats();
            const routingAnalytics = router.get_enhanced_analytics();
            const performanceMetrics = memoryManager.get_performance_metrics();
            
            return {
                analysisType,
                memoryStats,
                routingAnalytics,
                performanceMetrics,
                timestamp: Date.now()
            };
        } catch (error) {
            throw new Error(`Analysis failed: ${error.message}`);
        }
    },

    async dataPreprocessing(payload) {
        const { data, preprocessingType = 'tokenization' } = payload;
        
        try {
            let result;
            
            switch (preprocessingType) {
                case 'tokenization':
                    result = tokenizeText(data);
                    break;
                case 'featureExtraction':
                    result = extractFeatures(data);
                    break;
                case 'normalization':
                    result = normalizeData(data);
                    break;
                default:
                    throw new Error(`Unknown preprocessing type: ${preprocessingType}`);
            }
            
            return {
                preprocessingType,
                originalSize: Array.isArray(data) ? data.length : data.length,
                processedSize: Array.isArray(result) ? result.length : result.length,
                result,
                success: true
            };
        } catch (error) {
            throw new Error(`Preprocessing failed: ${error.message}`);
        }
    },

    async batchInference(payload) {
        const { prompts, domain, batchSize = 10 } = payload;
        
        try {
            const results = [];
            const batchCount = Math.ceil(prompts.length / batchSize);
            
            for (let i = 0; i < batchCount; i++) {
                const batchStart = i * batchSize;
                const batchEnd = Math.min(batchStart + batchSize, prompts.length);
                const batch = prompts.slice(batchStart, batchEnd);
                
                const batchResults = await Promise.all(
                    batch.map(prompt => 
                        taskHandlers.expertInference({ prompt, domain })
                    )
                );
                
                results.push(...batchResults);
                
                // Send progress update
                self.postMessage({
                    type: 'progress',
                    taskId: payload.taskId,
                    progress: (i + 1) / batchCount,
                    completed: batchEnd,
                    total: prompts.length
                });
            }
            
            return {
                totalProcessed: prompts.length,
                results,
                batchSize,
                batchCount
            };
        } catch (error) {
            throw new Error(`Batch inference failed: ${error.message}`);
        }
    }
};

// Utility functions
function tokenizeText(text) {
    if (typeof text !== 'string') {
        throw new Error('Input must be a string');
    }
    
    return text.toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(token => token.length > 0)
        .map(token => ({
            token,
            length: token.length,
            numeric: /^\d+$/.test(token)
        }));
}

function extractFeatures(text) {
    if (typeof text !== 'string') {
        throw new Error('Input must be a string');
    }
    
    const words = text.split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    return {
        wordCount: words.length,
        sentenceCount: sentences.length,
        avgWordsPerSentence: words.length / Math.max(sentences.length, 1),
        avgWordLength: words.reduce((sum, word) => sum + word.length, 0) / Math.max(words.length, 1),
        uniqueWords: new Set(words.map(w => w.toLowerCase())).size,
        lexicalDiversity: new Set(words.map(w => w.toLowerCase())).size / Math.max(words.length, 1),
        characterCount: text.length,
        punctuationCount: (text.match(/[.!?,:;]/g) || []).length,
        numbersCount: (text.match(/\d+/g) || []).length,
        uppercaseRatio: text.replace(/[^A-Z]/g, '').length / Math.max(text.replace(/[^A-Za-z]/g, '').length, 1)
    };
}

function normalizeData(data) {
    if (!Array.isArray(data)) {
        throw new Error('Input must be an array');
    }
    
    const numbers = data.filter(x => typeof x === 'number' && !isNaN(x));
    if (numbers.length === 0) {
        throw new Error('No valid numbers found in data');
    }
    
    const mean = numbers.reduce((sum, x) => sum + x, 0) / numbers.length;
    const variance = numbers.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / numbers.length;
    const stdDev = Math.sqrt(variance);
    
    if (stdDev === 0) {
        return numbers.map(() => 0);
    }
    
    return numbers.map(x => (x - mean) / stdDev);
}

// Memory management
function cleanupMemory() {
    try {
        // Clean up old experts that haven't been used recently
        const threshold = Date.now() - (5 * 60 * 1000); // 5 minutes
        
        for (const [domain, expert] of experts.entries()) {
            // In a real implementation, we'd check last access time
            // For now, just limit the total number of experts
            if (experts.size > 10) {
                experts.delete(domain);
                console.log(`[Worker] Cleaned up expert: ${domain}`);
                break;
            }
        }
        
        // Trigger WASM memory cleanup if available
        if (memoryManager) {
            memoryManager.defragment_memory();
        }
        
        // Force garbage collection if available
        if (global.gc) {
            global.gc();
        }
    } catch (error) {
        console.warn('[Worker] Memory cleanup failed:', error);
    }
}

// Error handling
function handleTaskError(error, taskId) {
    console.error(`[Worker] Task ${taskId} failed:`, error);
    
    return {
        taskId,
        success: false,
        error: {
            message: error.message,
            stack: error.stack,
            name: error.name
        },
        timestamp: Date.now()
    };
}

// Performance monitoring
const performanceMonitor = {
    taskTimes: new Map(),
    memoryUsage: [],
    
    startTask(taskId) {
        this.taskTimes.set(taskId, performance.now());
    },
    
    endTask(taskId) {
        const startTime = this.taskTimes.get(taskId);
        if (startTime) {
            const duration = performance.now() - startTime;
            this.taskTimes.delete(taskId);
            return duration;
        }
        return 0;
    },
    
    recordMemoryUsage() {
        if (performance.memory) {
            this.memoryUsage.push({
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit,
                timestamp: Date.now()
            });
            
            // Keep only last 100 measurements
            if (this.memoryUsage.length > 100) {
                this.memoryUsage.shift();
            }
        }
    },
    
    getStats() {
        return {
            averageTaskTime: Array.from(this.taskTimes.values()).reduce((sum, time) => sum + time, 0) / this.taskTimes.size || 0,
            activeTasks: this.taskTimes.size,
            memoryUsage: this.memoryUsage.slice(-10), // Last 10 measurements
            timestamp: Date.now()
        };
    }
};

// Main message handler
self.onmessage = async function(event) {
    const { taskId, taskType, payload, priority = 5, timeout = 30000 } = event.data;
    
    try {
        // Initialize if not already done
        if (!wasmModule) {
            const initialized = await initializeWorker();
            if (!initialized) {
                throw new Error('Failed to initialize worker');
            }
        }
        
        // Start performance monitoring
        performanceMonitor.startTask(taskId);
        performanceMonitor.recordMemoryUsage();
        
        // Set up timeout
        const timeoutId = setTimeout(() => {
            throw new Error(`Task ${taskId} timed out after ${timeout}ms`);
        }, timeout);
        
        try {
            // Execute task
            const handler = taskHandlers[taskType];
            if (!handler) {
                throw new Error(`Unknown task type: ${taskType}`);
            }
            
            const result = await handler({ ...payload, taskId });
            const executionTime = performanceMonitor.endTask(taskId);
            
            clearTimeout(timeoutId);
            
            // Send success response
            self.postMessage({
                taskId,
                success: true,
                result,
                executionTimeMs: executionTime,
                memoryStats: memoryManager ? memoryManager.get_memory_stats() : null,
                timestamp: Date.now()
            });
            
        } catch (taskError) {
            clearTimeout(timeoutId);
            throw taskError;
        }
        
    } catch (error) {
        const errorResponse = handleTaskError(error, taskId);
        const executionTime = performanceMonitor.endTask(taskId);
        errorResponse.executionTimeMs = executionTime;
        
        self.postMessage(errorResponse);
    } finally {
        // Periodic cleanup
        if (Math.random() < 0.1) { // 10% chance
            cleanupMemory();
        }
    }
};

// Handle worker lifecycle events
self.addEventListener('error', (error) => {
    console.error('[Worker] Unhandled error:', error);
    
    self.postMessage({
        type: 'error',
        error: {
            message: error.message,
            filename: error.filename,
            lineno: error.lineno,
            colno: error.colno
        },
        timestamp: Date.now()
    });
});

self.addEventListener('unhandledrejection', (event) => {
    console.error('[Worker] Unhandled promise rejection:', event.reason);
    
    self.postMessage({
        type: 'unhandledRejection',
        reason: event.reason?.toString() || 'Unknown rejection',
        timestamp: Date.now()
    });
});

// Worker ready signal
self.postMessage({
    type: 'ready',
    capabilities: [
        'expertInference',
        'modelTraining', 
        'memoryCompression',
        'performanceAnalysis',
        'dataPreprocessing',
        'batchInference'
    ],
    timestamp: Date.now()
});

// Periodic health check
setInterval(() => {
    self.postMessage({
        type: 'healthCheck',
        stats: performanceMonitor.getStats(),
        expertsLoaded: experts.size,
        wasmInitialized: !!wasmModule,
        timestamp: Date.now()
    });
}, 30000); // Every 30 seconds