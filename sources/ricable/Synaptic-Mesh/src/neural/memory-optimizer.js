/**
 * Neural Memory Optimizer
 * Advanced memory management for large neural models with <50MB target
 */

export class NeuralMemoryOptimizer {
    constructor(options = {}) {
        this.targetMemoryMB = options.targetMemory || 50;
        this.maxAgents = options.maxAgents || 10;
        this.compressionLevel = options.compression || 'balanced';
        this.enableQuantization = options.quantization !== false;
        this.enablePruning = options.pruning !== false;
        this.enableGradientCheckpointing = options.gradientCheckpointing !== false;
        
        // Memory pools
        this.activationPool = new MemoryPool('activations', 8 * 1024 * 1024); // 8MB
        this.weightPool = new MemoryPool('weights', 16 * 1024 * 1024); // 16MB  
        this.gradientPool = new MemoryPool('gradients', 8 * 1024 * 1024); // 8MB
        this.temporaryPool = new MemoryPool('temporary', 4 * 1024 * 1024); // 4MB
        
        // Memory tracking
        this.memoryUsage = {
            activations: 0,
            weights: 0,
            gradients: 0,
            temporary: 0,
            total: 0
        };
        
        // Compression strategies
        this.compressionStrategies = {
            aggressive: {
                quantizationBits: 4,
                pruningThreshold: 0.1,
                activationPrecision: 'fp16',
                gradientAccumulation: 8,
                layerwise: true
            },
            balanced: {
                quantizationBits: 8,
                pruningThreshold: 0.05,
                activationPrecision: 'fp16',
                gradientAccumulation: 4,
                layerwise: false
            },
            conservative: {
                quantizationBits: 16,
                pruningThreshold: 0.01,
                activationPrecision: 'fp32',
                gradientAccumulation: 2,
                layerwise: false
            }
        };
        
        this.activeStrategy = this.compressionStrategies[this.compressionLevel];
        this.networkRegistry = new Map();
        this.memoryHistory = [];
        this.optimizationMetrics = {
            compressionRatio: 1.0,
            speedup: 1.0,
            accuracyPreservation: 1.0
        };
        
        console.log(`🧠 Neural Memory Optimizer initialized - Target: ${this.targetMemoryMB}MB for ${this.maxAgents} agents`);
    }
    
    /**
     * Register a neural network for memory optimization
     */
    async registerNetwork(networkId, network, config = {}) {
        const startTime = performance.now();
        
        // Analyze network memory requirements
        const analysis = await this.analyzeNetworkMemory(network, config);
        
        // Apply memory optimizations
        const optimizedNetwork = await this.optimizeNetworkMemory(network, analysis, config);
        
        // Register in memory system
        this.networkRegistry.set(networkId, {
            original: network,
            optimized: optimizedNetwork,
            analysis,
            config,
            timestamp: Date.now(),
            memoryFootprint: analysis.optimizedMemory,
            accessPattern: 'sequential'
        });
        
        const optimizationTime = performance.now() - startTime;
        console.log(`📊 Network ${networkId} optimized in ${optimizationTime.toFixed(2)}ms - Memory: ${(analysis.originalMemory / 1024 / 1024).toFixed(2)}MB → ${(analysis.optimizedMemory / 1024 / 1024).toFixed(2)}MB`);
        
        return optimizedNetwork;
    }
    
    /**
     * Analyze network memory requirements
     */
    async analyzeNetworkMemory(network, config) {
        const analysis = {
            layers: [],
            originalMemory: 0,
            optimizedMemory: 0,
            compressionPotential: 0,
            pruningPotential: 0,
            quantizationBenefit: 0,
            activationMemory: 0,
            weightMemory: 0,
            peakMemory: 0
        };
        
        // Analyze each layer
        for (let i = 0; i < network.layers.length; i++) {
            const layer = network.layers[i];
            const layerAnalysis = await this.analyzeLayer(layer, i, config);
            analysis.layers.push(layerAnalysis);
            
            analysis.originalMemory += layerAnalysis.originalSize;
            analysis.weightMemory += layerAnalysis.weightSize;
            analysis.activationMemory += layerAnalysis.activationSize;
        }
        
        // Calculate optimization potential
        analysis.compressionPotential = this.calculateCompressionPotential(analysis);
        analysis.pruningPotential = this.calculatePruningPotential(analysis);
        analysis.quantizationBenefit = this.calculateQuantizationBenefit(analysis);
        
        // Estimate optimized memory usage
        analysis.optimizedMemory = this.estimateOptimizedMemory(analysis);
        analysis.peakMemory = Math.max(analysis.originalMemory, analysis.optimizedMemory * 1.5);
        
        return analysis;
    }
    
    /**
     * Analyze individual layer memory characteristics
     */
    async analyzeLayer(layer, layerIndex, config) {
        const analysis = {
            index: layerIndex,
            type: layer.type || 'unknown',
            originalSize: 0,
            optimizedSize: 0,
            weightSize: 0,
            activationSize: 0,
            computeIntensity: 0,
            memoryAccessPattern: 'sequential',
            compressionRatio: 1.0,
            prunable: false,
            quantizable: true,
            activationPeak: 0
        };
        
        // Calculate weight memory
        if (layer.weights) {
            analysis.weightSize = layer.weights.length * 4; // fp32
            analysis.originalSize += analysis.weightSize;
        }
        
        if (layer.bias) {
            analysis.weightSize += layer.bias.length * 4;
            analysis.originalSize += layer.bias.length * 4;
        }
        
        // Estimate activation memory based on layer type
        analysis.activationSize = this.estimateActivationMemory(layer, config);
        analysis.originalSize += analysis.activationSize;
        
        // Analyze compute characteristics
        analysis.computeIntensity = this.calculateComputeIntensity(layer);
        analysis.memoryAccessPattern = this.determineAccessPattern(layer);
        
        // Determine optimization suitability
        analysis.prunable = this.isPrunable(layer);
        analysis.quantizable = this.isQuantizable(layer);
        
        // Calculate potential compression
        analysis.compressionRatio = this.estimateLayerCompressionRatio(layer, analysis);
        analysis.optimizedSize = analysis.originalSize * analysis.compressionRatio;
        
        return analysis;
    }
    
    /**
     * Optimize network memory usage
     */
    async optimizeNetworkMemory(network, analysis, config) {
        const optimizedNetwork = {
            ...network,
            layers: [],
            optimizations: {
                quantization: [],
                pruning: [],
                compression: [],
                checkpointing: []
            },
            memoryProfile: {
                baseline: analysis.originalMemory,
                optimized: analysis.optimizedMemory,
                reduction: (analysis.originalMemory - analysis.optimizedMemory) / analysis.originalMemory
            }
        };
        
        // Apply layer-wise optimizations
        for (let i = 0; i < network.layers.length; i++) {
            const layer = network.layers[i];
            const layerAnalysis = analysis.layers[i];
            
            let optimizedLayer = { ...layer };
            
            // Apply quantization
            if (this.enableQuantization && layerAnalysis.quantizable) {
                optimizedLayer = await this.quantizeLayer(optimizedLayer, layerAnalysis);
                optimizedNetwork.optimizations.quantization.push({
                    layer: i,
                    bits: this.activeStrategy.quantizationBits,
                    reduction: this.calculateQuantizationReduction(layerAnalysis)
                });
            }
            
            // Apply pruning
            if (this.enablePruning && layerAnalysis.prunable) {
                optimizedLayer = await this.pruneLayer(optimizedLayer, layerAnalysis);
                optimizedNetwork.optimizations.pruning.push({
                    layer: i,
                    threshold: this.activeStrategy.pruningThreshold,
                    sparsity: this.calculateSparsity(optimizedLayer)
                });
            }
            
            // Apply compression
            optimizedLayer = await this.compressLayer(optimizedLayer, layerAnalysis);
            optimizedNetwork.optimizations.compression.push({
                layer: i,
                method: 'lz4',
                ratio: layerAnalysis.compressionRatio
            });
            
            // Setup gradient checkpointing
            if (this.enableGradientCheckpointing && this.shouldCheckpoint(layerAnalysis)) {
                optimizedLayer.checkpoint = true;
                optimizedNetwork.optimizations.checkpointing.push(i);
            }
            
            optimizedNetwork.layers.push(optimizedLayer);
        }
        
        // Apply global optimizations
        await this.applyGlobalOptimizations(optimizedNetwork, analysis);
        
        return optimizedNetwork;
    }
    
    /**
     * Quantize layer weights and activations
     */
    async quantizeLayer(layer, analysis) {
        const quantizedLayer = { ...layer };
        const bits = this.activeStrategy.quantizationBits;
        
        if (layer.weights) {
            const quantizationResult = this.quantizeWeights(layer.weights, bits);
            quantizedLayer.weights = quantizationResult.quantized;
            quantizedLayer.quantization = {
                scale: quantizationResult.scale,
                offset: quantizationResult.offset,
                bits,
                originalType: 'fp32',
                quantizedType: `int${bits}`
            };
        }
        
        if (layer.bias) {
            const biasQuantization = this.quantizeWeights(layer.bias, Math.min(bits + 8, 32));
            quantizedLayer.bias = biasQuantization.quantized;
        }
        
        // Setup activation quantization
        quantizedLayer.activationQuantization = {
            enabled: true,
            precision: this.activeStrategy.activationPrecision,
            dynamicRange: this.estimateActivationRange(analysis)
        };
        
        return quantizedLayer;
    }
    
    /**
     * Quantize weights to specified bit width
     */
    quantizeWeights(weights, bits) {
        const maxValue = (1 << (bits - 1)) - 1;
        const minValue = -(1 << (bits - 1));
        
        // Find min/max values
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < weights.length; i++) {
            min = Math.min(min, weights[i]);
            max = Math.max(max, weights[i]);
        }
        
        // Calculate scale and offset
        const scale = (max - min) / (maxValue - minValue);
        const offset = min - minValue * scale;
        
        // Quantize weights
        const quantized = new Int8Array(weights.length);
        for (let i = 0; i < weights.length; i++) {
            const quantizedValue = Math.round((weights[i] - offset) / scale);
            quantized[i] = Math.max(minValue, Math.min(maxValue, quantizedValue));
        }
        
        return { quantized, scale, offset };
    }
    
    /**
     * Prune layer weights below threshold
     */
    async pruneLayer(layer, analysis) {
        const prunedLayer = { ...layer };
        const threshold = this.activeStrategy.pruningThreshold;
        
        if (layer.weights) {
            const { prunedWeights, mask, sparsity } = this.pruneWeights(layer.weights, threshold);
            prunedLayer.weights = prunedWeights;
            prunedLayer.pruning = {
                mask,
                sparsity,
                threshold,
                method: 'magnitude'
            };
        }
        
        return prunedLayer;
    }
    
    /**
     * Prune weights based on magnitude
     */
    pruneWeights(weights, threshold) {
        const mask = new Uint8Array(weights.length);
        const prunedWeights = new Float32Array(weights.length);
        let prunedCount = 0;
        
        for (let i = 0; i < weights.length; i++) {
            if (Math.abs(weights[i]) > threshold) {
                mask[i] = 1;
                prunedWeights[i] = weights[i];
            } else {
                mask[i] = 0;
                prunedWeights[i] = 0;
                prunedCount++;
            }
        }
        
        const sparsity = prunedCount / weights.length;
        return { prunedWeights, mask, sparsity };
    }
    
    /**
     * Compress layer data using efficient algorithms
     */
    async compressLayer(layer, analysis) {
        const compressedLayer = { ...layer };
        
        // Compress weights if they exist
        if (layer.weights) {
            const compressed = await this.compressArray(layer.weights, 'weights');
            compressedLayer.compressedWeights = compressed;
            compressedLayer.compression = {
                method: 'lz4',
                originalSize: layer.weights.length * 4,
                compressedSize: compressed.length,
                ratio: compressed.length / (layer.weights.length * 4)
            };
        }
        
        return compressedLayer;
    }
    
    /**
     * Compress array data using LZ4-like algorithm
     */
    async compressArray(data, type) {
        // Simplified compression algorithm
        // In practice, would use actual LZ4 or similar
        const compressed = [];
        let i = 0;
        
        while (i < data.length) {
            let matchLength = 0;
            let matchDistance = 0;
            
            // Find longest match in previous data
            for (let j = Math.max(0, i - 65536); j < i; j++) {
                let length = 0;
                while (i + length < data.length && 
                       j + length < i && 
                       Math.abs(data[i + length] - data[j + length]) < 0.001 && 
                       length < 255) {
                    length++;
                }
                
                if (length > matchLength) {
                    matchLength = length;
                    matchDistance = i - j;
                }
            }
            
            if (matchLength > 3) {
                // Store match
                compressed.push({ type: 'match', distance: matchDistance, length: matchLength });
                i += matchLength;
            } else {
                // Store literal
                compressed.push({ type: 'literal', value: data[i] });
                i++;
            }
        }
        
        return compressed;
    }
    
    /**
     * Apply global memory optimizations
     */
    async applyGlobalOptimizations(network, analysis) {
        // Implement gradient accumulation
        if (this.activeStrategy.gradientAccumulation > 1) {
            network.gradientAccumulation = {
                steps: this.activeStrategy.gradientAccumulation,
                enabled: true
            };
        }
        
        // Setup memory mapping for large networks
        if (analysis.originalMemory > 32 * 1024 * 1024) { // 32MB
            network.memoryMapping = {
                enabled: true,
                chunkSize: 4 * 1024 * 1024, // 4MB chunks
                swapThreshold: 0.8
            };
        }
        
        // Configure activation recomputation
        if (this.enableGradientCheckpointing) {
            network.activationRecomputation = {
                enabled: true,
                strategy: 'automatic',
                memoryBudget: this.targetMemoryMB * 1024 * 1024 * 0.6 // 60% of target
            };
        }
    }
    
    /**
     * Allocate memory from appropriate pool
     */
    allocateMemory(type, size, alignment = 16) {
        const pool = this.getMemoryPool(type);
        const alignedSize = Math.ceil(size / alignment) * alignment;
        
        const ptr = pool.allocate(alignedSize);
        if (ptr) {
            this.memoryUsage[type] += alignedSize;
            this.memoryUsage.total += alignedSize;
            this.updateMemoryHistory();
            return ptr;
        }
        
        // Try garbage collection and retry
        this.runGarbageCollection(type);
        return pool.allocate(alignedSize);
    }
    
    /**
     * Deallocate memory back to pool
     */
    deallocateMemory(type, ptr, size) {
        const pool = this.getMemoryPool(type);
        pool.deallocate(ptr);
        
        this.memoryUsage[type] -= size;
        this.memoryUsage.total -= size;
        this.updateMemoryHistory();
    }
    
    /**
     * Get appropriate memory pool for type
     */
    getMemoryPool(type) {
        switch (type) {
            case 'activations': return this.activationPool;
            case 'weights': return this.weightPool;
            case 'gradients': return this.gradientPool;
            case 'temporary': return this.temporaryPool;
            default: return this.temporaryPool;
        }
    }
    
    /**
     * Run garbage collection for specific memory type
     */
    runGarbageCollection(type) {
        const pool = this.getMemoryPool(type);
        const beforeSize = pool.getUsedMemory();
        
        // Mark and sweep unused memory
        pool.garbageCollect();
        
        const afterSize = pool.getUsedMemory();
        const recovered = beforeSize - afterSize;
        
        if (recovered > 0) {
            console.log(`🗑️ GC recovered ${(recovered / 1024 / 1024).toFixed(2)}MB in ${type} pool`);
        }
        
        return recovered;
    }
    
    /**
     * Update memory usage history
     */
    updateMemoryHistory() {
        this.memoryHistory.push({
            timestamp: Date.now(),
            ...this.memoryUsage
        });
        
        // Keep only last 100 entries
        if (this.memoryHistory.length > 100) {
            this.memoryHistory.shift();
        }
    }
    
    /**
     * Check if memory usage is within target
     */
    isWithinMemoryTarget() {
        const totalMB = this.memoryUsage.total / 1024 / 1024;
        return totalMB <= this.targetMemoryMB;
    }
    
    /**
     * Get memory optimization statistics
     */
    getOptimizationStats() {
        const totalNetworks = this.networkRegistry.size;
        let totalOriginalMemory = 0;
        let totalOptimizedMemory = 0;
        
        for (const [_, networkInfo] of this.networkRegistry.entries()) {
            totalOriginalMemory += networkInfo.analysis.originalMemory;
            totalOptimizedMemory += networkInfo.analysis.optimizedMemory;
        }
        
        const memoryReduction = totalOriginalMemory > 0 
            ? (totalOriginalMemory - totalOptimizedMemory) / totalOriginalMemory 
            : 0;
        
        const currentUsageMB = this.memoryUsage.total / 1024 / 1024;
        const targetUtilization = currentUsageMB / this.targetMemoryMB;
        
        return {
            totalNetworks,
            originalMemoryMB: totalOriginalMemory / 1024 / 1024,
            optimizedMemoryMB: totalOptimizedMemory / 1024 / 1024,
            memoryReduction: memoryReduction * 100,
            currentUsageMB,
            targetMemoryMB: this.targetMemoryMB,
            targetUtilization: targetUtilization * 100,
            isWithinTarget: this.isWithinMemoryTarget(),
            poolStats: {
                activations: this.activationPool.getStats(),
                weights: this.weightPool.getStats(),
                gradients: this.gradientPool.getStats(),
                temporary: this.temporaryPool.getStats()
            },
            compressionMetrics: this.optimizationMetrics
        };
    }
    
    // Helper methods for analysis
    
    estimateActivationMemory(layer, config) {
        if (!layer.outputShape) return 1024; // Default estimate
        
        const elements = layer.outputShape.reduce((a, b) => a * b, 1);
        const precision = this.activeStrategy.activationPrecision === 'fp16' ? 2 : 4;
        return elements * precision;
    }
    
    calculateComputeIntensity(layer) {
        if (layer.type === 'conv2d' && layer.kernelSize) {
            return layer.kernelSize * layer.kernelSize * (layer.filters || 1);
        }
        if (layer.type === 'dense' && layer.units) {
            return layer.units;
        }
        return 1;
    }
    
    determineAccessPattern(layer) {
        if (layer.type === 'conv2d') return 'spatial';
        if (layer.type === 'lstm' || layer.type === 'gru') return 'temporal';
        return 'sequential';
    }
    
    isPrunable(layer) {
        return layer.type === 'dense' || layer.type === 'conv2d';
    }
    
    isQuantizable(layer) {
        return layer.type !== 'batch_norm' && layer.type !== 'layer_norm';
    }
    
    shouldCheckpoint(analysis) {
        return analysis.activationSize > 1024 * 1024 && analysis.computeIntensity > 100;
    }
    
    calculateCompressionPotential(analysis) {
        return analysis.layers.reduce((total, layer) => {
            return total + layer.originalSize * (1 - layer.compressionRatio);
        }, 0);
    }
    
    calculatePruningPotential(analysis) {
        return analysis.layers.filter(l => l.prunable).reduce((total, layer) => {
            return total + layer.weightSize * 0.3; // Assume 30% prunable
        }, 0);
    }
    
    calculateQuantizationBenefit(analysis) {
        const bitsReduction = 32 - this.activeStrategy.quantizationBits;
        return analysis.weightMemory * (bitsReduction / 32);
    }
    
    estimateOptimizedMemory(analysis) {
        let optimized = 0;
        
        for (const layer of analysis.layers) {
            let layerMemory = layer.originalSize;
            
            // Apply quantization reduction
            if (layer.quantizable) {
                layerMemory *= this.activeStrategy.quantizationBits / 32;
            }
            
            // Apply pruning reduction
            if (layer.prunable) {
                layerMemory *= (1 - this.activeStrategy.pruningThreshold);
            }
            
            // Apply compression
            layerMemory *= layer.compressionRatio;
            
            optimized += layerMemory;
        }
        
        return optimized;
    }
    
    estimateLayerCompressionRatio(layer, analysis) {
        let ratio = 1.0;
        
        // Quantization effect
        if (analysis.quantizable) {
            ratio *= this.activeStrategy.quantizationBits / 32;
        }
        
        // Pruning effect
        if (analysis.prunable) {
            ratio *= (1 - this.activeStrategy.pruningThreshold * 0.5);
        }
        
        // Compression effect
        ratio *= 0.7; // Assume 30% compression
        
        return Math.max(0.1, ratio);
    }
    
    calculateQuantizationReduction(analysis) {
        return analysis.weightSize * (1 - this.activeStrategy.quantizationBits / 32);
    }
    
    calculateSparsity(layer) {
        if (!layer.pruning) return 0;
        return layer.pruning.sparsity;
    }
    
    estimateActivationRange(analysis) {
        // Estimate based on layer type and position
        if (analysis.type === 'relu') {
            return { min: 0, max: 6 };
        } else if (analysis.type === 'sigmoid') {
            return { min: 0, max: 1 };
        } else if (analysis.type === 'tanh') {
            return { min: -1, max: 1 };
        }
        return { min: -3, max: 3 }; // Default range
    }
}

/**
 * Memory Pool for efficient memory management
 */
class MemoryPool {
    constructor(name, maxSize) {
        this.name = name;
        this.maxSize = maxSize;
        this.blocks = [];
        this.freeBlocks = [];
        this.usedBlocks = new Map();
        this.totalAllocated = 0;
        this.peakUsage = 0;
        this.allocationCount = 0;
        this.deallocationCount = 0;
    }
    
    allocate(size) {
        // Try to find a suitable free block
        for (let i = 0; i < this.freeBlocks.length; i++) {
            const block = this.freeBlocks[i];
            if (block.size >= size) {
                this.freeBlocks.splice(i, 1);
                
                // Split block if too large
                if (block.size > size + 64) {
                    const remaining = {
                        ptr: block.ptr + size,
                        size: block.size - size
                    };
                    this.freeBlocks.push(remaining);
                }
                
                block.size = size;
                this.usedBlocks.set(block.ptr, block);
                this.allocationCount++;
                return block.ptr;
            }
        }
        
        // Allocate new block
        if (this.totalAllocated + size > this.maxSize) {
            this.garbageCollect();
            if (this.totalAllocated + size > this.maxSize) {
                return null; // Out of memory
            }
        }
        
        const ptr = this.totalAllocated;
        const block = { ptr, size };
        
        this.blocks.push(block);
        this.usedBlocks.set(ptr, block);
        this.totalAllocated += size;
        this.peakUsage = Math.max(this.peakUsage, this.totalAllocated);
        this.allocationCount++;
        
        return ptr;
    }
    
    deallocate(ptr) {
        const block = this.usedBlocks.get(ptr);
        if (block) {
            this.usedBlocks.delete(ptr);
            this.freeBlocks.push(block);
            this.deallocationCount++;
            
            // Try to merge adjacent free blocks
            this.mergeFreeBlocks();
        }
    }
    
    mergeFreeBlocks() {
        this.freeBlocks.sort((a, b) => a.ptr - b.ptr);
        
        for (let i = 0; i < this.freeBlocks.length - 1; i++) {
            const current = this.freeBlocks[i];
            const next = this.freeBlocks[i + 1];
            
            if (current.ptr + current.size === next.ptr) {
                current.size += next.size;
                this.freeBlocks.splice(i + 1, 1);
                i--; // Check again with merged block
            }
        }
    }
    
    garbageCollect() {
        // Compact memory by moving used blocks
        let writePtr = 0;
        const newUsedBlocks = new Map();
        
        for (const [oldPtr, block] of this.usedBlocks.entries()) {
            if (writePtr !== oldPtr) {
                // Move block data (simplified)
                block.ptr = writePtr;
                newUsedBlocks.set(writePtr, block);
            } else {
                newUsedBlocks.set(oldPtr, block);
            }
            writePtr += block.size;
        }
        
        this.usedBlocks = newUsedBlocks;
        this.totalAllocated = writePtr;
        this.freeBlocks = [];
        
        // Add remaining space as free block
        if (this.totalAllocated < this.maxSize) {
            this.freeBlocks.push({
                ptr: this.totalAllocated,
                size: this.maxSize - this.totalAllocated
            });
        }
    }
    
    getUsedMemory() {
        return Array.from(this.usedBlocks.values()).reduce((total, block) => total + block.size, 0);
    }
    
    getFreeMemory() {
        return this.freeBlocks.reduce((total, block) => total + block.size, 0);
    }
    
    getStats() {
        return {
            name: this.name,
            maxSize: this.maxSize,
            totalAllocated: this.totalAllocated,
            usedMemory: this.getUsedMemory(),
            freeMemory: this.getFreeMemory(),
            peakUsage: this.peakUsage,
            allocationCount: this.allocationCount,
            deallocationCount: this.deallocationCount,
            fragmentation: this.calculateFragmentation(),
            efficiency: this.calculateEfficiency()
        };
    }
    
    calculateFragmentation() {
        if (this.freeBlocks.length === 0) return 0;
        
        const totalFree = this.getFreeMemory();
        const largestFree = Math.max(...this.freeBlocks.map(b => b.size));
        
        return totalFree > 0 ? 1 - (largestFree / totalFree) : 0;
    }
    
    calculateEfficiency() {
        return this.maxSize > 0 ? this.getUsedMemory() / this.maxSize : 0;
    }
}

export { NeuralMemoryOptimizer, MemoryPool };