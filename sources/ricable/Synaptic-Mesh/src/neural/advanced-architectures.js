/**
 * Advanced Neural Architectures
 * LSTM, Transformer, CNN, and other modern neural network implementations
 */

/**
 * Long Short-Term Memory (LSTM) Network
 * Optimized for sequential data processing with memory retention
 */
export class LSTMNetwork {
    constructor(options = {}) {
        this.inputSize = options.inputSize || 128;
        this.hiddenSize = options.hiddenSize || 256;
        this.outputSize = options.outputSize || 64;
        this.numLayers = options.numLayers || 2;
        this.dropout = options.dropout || 0.1;
        this.bidirectional = options.bidirectional || false;
        
        // LSTM cell states
        this.hiddenStates = [];
        this.cellStates = [];
        
        // Weight matrices for each LSTM layer
        this.weights = this.initializeWeights();
        
        // Performance tracking
        this.sequenceLength = 0;
        this.processedSequences = 0;
        this.averageProcessingTime = 0;
        
        console.log(`🧠 LSTM Network initialized: ${this.inputSize}→${this.hiddenSize}x${this.numLayers}→${this.outputSize}`);
    }
    
    initializeWeights() {
        const layers = [];
        
        for (let layer = 0; layer < this.numLayers; layer++) {
            const inputDim = layer === 0 ? this.inputSize : this.hiddenSize;
            const directions = this.bidirectional ? 2 : 1;
            
            const layerWeights = {
                // Input gate weights
                Wi: this.createMatrix(this.hiddenSize, inputDim),
                Ui: this.createMatrix(this.hiddenSize, this.hiddenSize),
                bi: new Float32Array(this.hiddenSize),
                
                // Forget gate weights
                Wf: this.createMatrix(this.hiddenSize, inputDim),
                Uf: this.createMatrix(this.hiddenSize, this.hiddenSize),
                bf: new Float32Array(this.hiddenSize).fill(1.0), // Forget bias initialized to 1
                
                // Cell state weights
                Wc: this.createMatrix(this.hiddenSize, inputDim),
                Uc: this.createMatrix(this.hiddenSize, this.hiddenSize),
                bc: new Float32Array(this.hiddenSize),
                
                // Output gate weights
                Wo: this.createMatrix(this.hiddenSize, inputDim),
                Uo: this.createMatrix(this.hiddenSize, this.hiddenSize),
                bo: new Float32Array(this.hiddenSize),
                
                directions
            };
            
            // Initialize with Xavier/Glorot initialization
            this.applyXavierInitialization(layerWeights, inputDim, this.hiddenSize);
            
            layers.push(layerWeights);
        }
        
        // Output projection weights
        const outputWeights = {
            Wy: this.createMatrix(this.outputSize, this.hiddenSize * (this.bidirectional ? 2 : 1)),
            by: new Float32Array(this.outputSize)
        };
        
        layers.push(outputWeights);
        return layers;
    }
    
    createMatrix(rows, cols) {
        return new Float32Array(rows * cols);
    }
    
    applyXavierInitialization(weights, fanIn, fanOut) {
        const limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        for (const [key, matrix] of Object.entries(weights)) {
            if (matrix instanceof Float32Array && key !== 'directions') {
                for (let i = 0; i < matrix.length; i++) {
                    matrix[i] = (Math.random() * 2 - 1) * limit;
                }
            }
        }
    }
    
    /**
     * Forward pass through LSTM network
     */
    async forward(sequence, options = {}) {
        const startTime = performance.now();
        const returnSequences = options.returnSequences || false;
        const stateful = options.stateful || false;
        
        if (!stateful) {
            this.resetStates();
        }
        
        const seqLength = sequence.length;
        const outputs = [];
        
        // Process each timestep
        for (let t = 0; t < seqLength; t++) {
            const input = sequence[t];
            let layerInput = input;
            
            // Forward through each LSTM layer
            for (let layer = 0; layer < this.numLayers; layer++) {
                const { output: layerOutput, hidden, cell } = await this.forwardLSTMLayer(
                    layerInput, 
                    layer, 
                    this.hiddenStates[layer], 
                    this.cellStates[layer]
                );
                
                this.hiddenStates[layer] = hidden;
                this.cellStates[layer] = cell;
                layerInput = layerOutput;
            }
            
            // Apply output projection
            const finalOutput = this.applyOutputProjection(layerInput);
            
            if (returnSequences) {
                outputs.push(finalOutput);
            }
        }
        
        const processingTime = performance.now() - startTime;
        this.updatePerformanceMetrics(processingTime, seqLength);
        
        // Return either the last output or the full sequence
        return returnSequences ? outputs : outputs[outputs.length - 1] || this.applyOutputProjection(layerInput);
    }
    
    /**
     * Forward pass through a single LSTM layer
     */
    async forwardLSTMLayer(input, layerIndex, prevHidden, prevCell) {
        const weights = this.weights[layerIndex];
        const hiddenSize = this.hiddenSize;
        
        // Initialize states if not provided
        const h_prev = prevHidden || new Float32Array(hiddenSize);
        const c_prev = prevCell || new Float32Array(hiddenSize);
        
        // Compute gates
        const inputGate = this.computeGate(input, h_prev, weights.Wi, weights.Ui, weights.bi, 'sigmoid');
        const forgetGate = this.computeGate(input, h_prev, weights.Wf, weights.Uf, weights.bf, 'sigmoid');
        const cellInput = this.computeGate(input, h_prev, weights.Wc, weights.Uc, weights.bc, 'tanh');
        const outputGate = this.computeGate(input, h_prev, weights.Wo, weights.Uo, weights.bo, 'sigmoid');
        
        // Update cell state
        const c_new = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            c_new[i] = forgetGate[i] * c_prev[i] + inputGate[i] * cellInput[i];
        }
        
        // Compute hidden state
        const h_new = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            h_new[i] = outputGate[i] * Math.tanh(c_new[i]);
        }
        
        // Apply dropout if training
        if (this.dropout > 0 && Math.random() < this.dropout) {
            for (let i = 0; i < hiddenSize; i++) {
                if (Math.random() < this.dropout) {
                    h_new[i] = 0;
                }
            }
        }
        
        return {
            output: h_new,
            hidden: h_new,
            cell: c_new
        };
    }
    
    /**
     * Compute LSTM gate activation
     */
    computeGate(input, hidden, W, U, bias, activation) {
        const hiddenSize = this.hiddenSize;
        const gate = new Float32Array(hiddenSize);
        
        // W * input + U * hidden + bias
        for (let i = 0; i < hiddenSize; i++) {
            let sum = bias[i];
            
            // Add input contribution
            for (let j = 0; j < input.length; j++) {
                sum += W[i * input.length + j] * input[j];
            }
            
            // Add hidden contribution
            for (let j = 0; j < hiddenSize; j++) {
                sum += U[i * hiddenSize + j] * hidden[j];
            }
            
            // Apply activation function
            gate[i] = activation === 'sigmoid' ? this.sigmoid(sum) : Math.tanh(sum);
        }
        
        return gate;
    }
    
    applyOutputProjection(hiddenState) {
        const outputWeights = this.weights[this.weights.length - 1];
        const output = new Float32Array(this.outputSize);
        
        for (let i = 0; i < this.outputSize; i++) {
            let sum = outputWeights.by[i];
            for (let j = 0; j < hiddenState.length; j++) {
                sum += outputWeights.Wy[i * hiddenState.length + j] * hiddenState[j];
            }
            output[i] = sum;
        }
        
        return output;
    }
    
    resetStates() {
        this.hiddenStates = Array(this.numLayers).fill(null).map(() => new Float32Array(this.hiddenSize));
        this.cellStates = Array(this.numLayers).fill(null).map(() => new Float32Array(this.hiddenSize));
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
    }
    
    updatePerformanceMetrics(processingTime, seqLength) {
        this.processedSequences++;
        this.sequenceLength = seqLength;
        this.averageProcessingTime = (this.averageProcessingTime * (this.processedSequences - 1) + processingTime) / this.processedSequences;
    }
    
    getMetrics() {
        return {
            processedSequences: this.processedSequences,
            averageProcessingTime: this.averageProcessingTime,
            sequenceLength: this.sequenceLength,
            throughput: this.averageProcessingTime > 0 ? 1000 / this.averageProcessingTime : 0,
            parameterCount: this.getParameterCount()
        };
    }
    
    getParameterCount() {
        return this.weights.reduce((total, layer) => {
            return total + Object.values(layer).reduce((layerTotal, matrix) => {
                return layerTotal + (matrix instanceof Float32Array ? matrix.length : 0);
            }, 0);
        }, 0);
    }
}

/**
 * Transformer Architecture
 * Self-attention based model for sequence-to-sequence tasks
 */
export class TransformerNetwork {
    constructor(options = {}) {
        this.modelDim = options.modelDim || 512;
        this.numHeads = options.numHeads || 8;
        this.numLayers = options.numLayers || 6;
        this.feedforwardDim = options.feedforwardDim || 2048;
        this.maxSeqLength = options.maxSeqLength || 512;
        this.dropout = options.dropout || 0.1;
        this.vocabSize = options.vocabSize || 10000;
        
        this.headDim = this.modelDim / this.numHeads;
        
        // Initialize layers
        this.embeddings = this.createEmbeddingLayer();
        this.positionalEncoding = this.createPositionalEncoding();
        this.encoderLayers = this.createEncoderLayers();
        this.outputProjection = this.createOutputProjection();
        
        // Attention maps for visualization
        this.attentionMaps = [];
        
        console.log(`🔄 Transformer initialized: ${this.numLayers} layers, ${this.numHeads} heads, ${this.modelDim}d`);
    }
    
    createEmbeddingLayer() {
        const embeddings = new Float32Array(this.vocabSize * this.modelDim);
        
        // Initialize with small random values
        for (let i = 0; i < embeddings.length; i++) {
            embeddings[i] = (Math.random() - 0.5) * 0.1;
        }
        
        return embeddings;
    }
    
    createPositionalEncoding() {
        const pe = new Float32Array(this.maxSeqLength * this.modelDim);
        
        for (let pos = 0; pos < this.maxSeqLength; pos++) {
            for (let i = 0; i < this.modelDim; i += 2) {
                const angle = pos / Math.pow(10000, (2 * i) / this.modelDim);
                pe[pos * this.modelDim + i] = Math.sin(angle);
                if (i + 1 < this.modelDim) {
                    pe[pos * this.modelDim + i + 1] = Math.cos(angle);
                }
            }
        }
        
        return pe;
    }
    
    createEncoderLayers() {
        const layers = [];
        
        for (let i = 0; i < this.numLayers; i++) {
            layers.push({
                // Multi-head attention
                queryWeights: this.createMatrix(this.modelDim, this.modelDim),
                keyWeights: this.createMatrix(this.modelDim, this.modelDim),
                valueWeights: this.createMatrix(this.modelDim, this.modelDim),
                outputWeights: this.createMatrix(this.modelDim, this.modelDim),
                
                // Layer normalization
                layerNorm1Scale: new Float32Array(this.modelDim).fill(1.0),
                layerNorm1Bias: new Float32Array(this.modelDim),
                layerNorm2Scale: new Float32Array(this.modelDim).fill(1.0),
                layerNorm2Bias: new Float32Array(this.modelDim),
                
                // Feedforward network
                ff1Weights: this.createMatrix(this.feedforwardDim, this.modelDim),
                ff1Bias: new Float32Array(this.feedforwardDim),
                ff2Weights: this.createMatrix(this.modelDim, this.feedforwardDim),
                ff2Bias: new Float32Array(this.modelDim)
            });
        }
        
        return layers;
    }
    
    createOutputProjection() {
        return {
            weights: this.createMatrix(this.vocabSize, this.modelDim),
            bias: new Float32Array(this.vocabSize)
        };
    }
    
    createMatrix(rows, cols) {
        const matrix = new Float32Array(rows * cols);
        const limit = Math.sqrt(6.0 / (rows + cols));
        
        for (let i = 0; i < matrix.length; i++) {
            matrix[i] = (Math.random() * 2 - 1) * limit;
        }
        
        return matrix;
    }
    
    /**
     * Forward pass through transformer
     */
    async forward(inputIds, options = {}) {
        const startTime = performance.now();
        const mask = options.attentionMask;
        const returnAttentions = options.returnAttentions || false;
        
        this.attentionMaps = [];
        
        // Input embedding + positional encoding
        let hidden = this.embedInput(inputIds);
        
        // Pass through encoder layers
        for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
            const layer = this.encoderLayers[layerIdx];
            
            // Multi-head self-attention
            const { output: attentionOutput, attentions } = await this.multiHeadAttention(
                hidden, hidden, hidden, layer, mask
            );
            
            if (returnAttentions) {
                this.attentionMaps.push(attentions);
            }
            
            // Residual connection + layer norm
            hidden = this.layerNorm(this.addResidual(hidden, attentionOutput), 
                                   layer.layerNorm1Scale, layer.layerNorm1Bias);
            
            // Feedforward network
            const ffOutput = this.feedforward(hidden, layer);
            
            // Residual connection + layer norm
            hidden = this.layerNorm(this.addResidual(hidden, ffOutput), 
                                   layer.layerNorm2Scale, layer.layerNorm2Bias);
        }
        
        // Output projection
        const logits = this.projectToVocab(hidden);
        
        const processingTime = performance.now() - startTime;
        
        return {
            logits,
            hidden,
            attentions: returnAttentions ? this.attentionMaps : null,
            processingTime
        };
    }
    
    embedInput(inputIds) {
        const seqLength = inputIds.length;
        const embedded = new Float32Array(seqLength * this.modelDim);
        
        for (let pos = 0; pos < seqLength; pos++) {
            const tokenId = inputIds[pos];
            
            for (let dim = 0; dim < this.modelDim; dim++) {
                // Token embedding + positional encoding
                const tokenEmb = this.embeddings[tokenId * this.modelDim + dim];
                const posEmb = this.positionalEncoding[pos * this.modelDim + dim];
                embedded[pos * this.modelDim + dim] = tokenEmb + posEmb;
            }
        }
        
        return { data: embedded, seqLength };
    }
    
    /**
     * Multi-head self-attention mechanism
     */
    async multiHeadAttention(queries, keys, values, layer, mask = null) {
        const { data: qData, seqLength } = queries;
        const headOutputs = [];
        const attentionWeights = [];
        
        // Compute attention for each head
        for (let head = 0; head < this.numHeads; head++) {
            const headStart = head * this.headDim;
            const headEnd = headStart + this.headDim;
            
            // Extract head-specific queries, keys, values
            const headQueries = this.projectToHead(qData, layer.queryWeights, headStart, headEnd, seqLength);
            const headKeys = this.projectToHead(qData, layer.keyWeights, headStart, headEnd, seqLength);
            const headValues = this.projectToHead(qData, layer.valueWeights, headStart, headEnd, seqLength);
            
            // Compute scaled dot-product attention
            const { output, attention } = this.scaledDotProductAttention(
                headQueries, headKeys, headValues, mask, seqLength
            );
            
            headOutputs.push(output);
            attentionWeights.push(attention);
        }
        
        // Concatenate heads
        const concatenated = this.concatenateHeads(headOutputs, seqLength);
        
        // Output projection
        const output = this.matrixMultiply(concatenated, layer.outputWeights, seqLength, this.modelDim, this.modelDim);
        
        return {
            output: { data: output, seqLength },
            attentions: attentionWeights
        };
    }
    
    projectToHead(input, weights, headStart, headEnd, seqLength) {
        const projected = new Float32Array(seqLength * this.headDim);
        
        for (let pos = 0; pos < seqLength; pos++) {
            for (let dim = 0; dim < this.headDim; dim++) {
                let sum = 0;
                for (let inputDim = 0; inputDim < this.modelDim; inputDim++) {
                    const weightIdx = (headStart + dim) * this.modelDim + inputDim;
                    const inputIdx = pos * this.modelDim + inputDim;
                    sum += input[inputIdx] * weights[weightIdx];
                }
                projected[pos * this.headDim + dim] = sum;
            }
        }
        
        return projected;
    }
    
    scaledDotProductAttention(queries, keys, values, mask, seqLength) {
        // Compute attention scores
        const scores = new Float32Array(seqLength * seqLength);
        const scale = 1.0 / Math.sqrt(this.headDim);
        
        for (let i = 0; i < seqLength; i++) {
            for (let j = 0; j < seqLength; j++) {
                let score = 0;
                for (let d = 0; d < this.headDim; d++) {
                    score += queries[i * this.headDim + d] * keys[j * this.headDim + d];
                }
                scores[i * seqLength + j] = score * scale;
            }
        }
        
        // Apply mask if provided
        if (mask) {
            for (let i = 0; i < seqLength; i++) {
                for (let j = 0; j < seqLength; j++) {
                    if (!mask[i * seqLength + j]) {
                        scores[i * seqLength + j] = -Infinity;
                    }
                }
            }
        }
        
        // Apply softmax
        const attention = this.softmax2D(scores, seqLength);
        
        // Apply attention to values
        const output = new Float32Array(seqLength * this.headDim);
        for (let i = 0; i < seqLength; i++) {
            for (let d = 0; d < this.headDim; d++) {
                let sum = 0;
                for (let j = 0; j < seqLength; j++) {
                    sum += attention[i * seqLength + j] * values[j * this.headDim + d];
                }
                output[i * this.headDim + d] = sum;
            }
        }
        
        return { output, attention };
    }
    
    feedforward(hidden, layer) {
        const { data, seqLength } = hidden;
        
        // First linear layer + ReLU
        const intermediate = new Float32Array(seqLength * this.feedforwardDim);
        for (let pos = 0; pos < seqLength; pos++) {
            for (let dim = 0; dim < this.feedforwardDim; dim++) {
                let sum = layer.ff1Bias[dim];
                for (let inputDim = 0; inputDim < this.modelDim; inputDim++) {
                    sum += data[pos * this.modelDim + inputDim] * layer.ff1Weights[dim * this.modelDim + inputDim];
                }
                intermediate[pos * this.feedforwardDim + dim] = Math.max(0, sum); // ReLU
            }
        }
        
        // Second linear layer
        const output = new Float32Array(seqLength * this.modelDim);
        for (let pos = 0; pos < seqLength; pos++) {
            for (let dim = 0; dim < this.modelDim; dim++) {
                let sum = layer.ff2Bias[dim];
                for (let ffDim = 0; ffDim < this.feedforwardDim; ffDim++) {
                    sum += intermediate[pos * this.feedforwardDim + ffDim] * layer.ff2Weights[dim * this.feedforwardDim + ffDim];
                }
                output[pos * this.modelDim + dim] = sum;
            }
        }
        
        return { data: output, seqLength };
    }
    
    layerNorm(input, scale, bias) {
        const { data, seqLength } = input;
        const normalized = new Float32Array(data.length);
        
        for (let pos = 0; pos < seqLength; pos++) {
            const startIdx = pos * this.modelDim;
            const endIdx = startIdx + this.modelDim;
            
            // Calculate mean
            let mean = 0;
            for (let i = startIdx; i < endIdx; i++) {
                mean += data[i];
            }
            mean /= this.modelDim;
            
            // Calculate variance
            let variance = 0;
            for (let i = startIdx; i < endIdx; i++) {
                const diff = data[i] - mean;
                variance += diff * diff;
            }
            variance = variance / this.modelDim + 1e-6; // Add epsilon for numerical stability
            
            // Normalize
            const stddev = Math.sqrt(variance);
            for (let i = 0; i < this.modelDim; i++) {
                const idx = startIdx + i;
                normalized[idx] = ((data[idx] - mean) / stddev) * scale[i] + bias[i];
            }
        }
        
        return { data: normalized, seqLength };
    }
    
    addResidual(input1, input2) {
        const { data: data1, seqLength } = input1;
        const { data: data2 } = input2;
        const result = new Float32Array(data1.length);
        
        for (let i = 0; i < data1.length; i++) {
            result[i] = data1[i] + data2[i];
        }
        
        return { data: result, seqLength };
    }
    
    projectToVocab(hidden) {
        const { data, seqLength } = hidden;
        const logits = new Float32Array(seqLength * this.vocabSize);
        
        for (let pos = 0; pos < seqLength; pos++) {
            for (let vocab = 0; vocab < this.vocabSize; vocab++) {
                let sum = this.outputProjection.bias[vocab];
                for (let dim = 0; dim < this.modelDim; dim++) {
                    sum += data[pos * this.modelDim + dim] * this.outputProjection.weights[vocab * this.modelDim + dim];
                }
                logits[pos * this.vocabSize + vocab] = sum;
            }
        }
        
        return { data: logits, seqLength };
    }
    
    // Helper methods
    matrixMultiply(a, b, m, n, k) {
        const result = new Float32Array(m * k);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < k; j++) {
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += a[i * n + l] * b[j * n + l];
                }
                result[i * k + j] = sum;
            }
        }
        return result;
    }
    
    concatenateHeads(headOutputs, seqLength) {
        const concatenated = new Float32Array(seqLength * this.modelDim);
        
        for (let pos = 0; pos < seqLength; pos++) {
            for (let head = 0; head < this.numHeads; head++) {
                const headOutput = headOutputs[head];
                for (let dim = 0; dim < this.headDim; dim++) {
                    const srcIdx = pos * this.headDim + dim;
                    const dstIdx = pos * this.modelDim + head * this.headDim + dim;
                    concatenated[dstIdx] = headOutput[srcIdx];
                }
            }
        }
        
        return concatenated;
    }
    
    softmax2D(scores, seqLength) {
        const softmaxed = new Float32Array(scores.length);
        
        for (let i = 0; i < seqLength; i++) {
            const start = i * seqLength;
            const end = start + seqLength;
            
            // Find max for numerical stability
            let max = -Infinity;
            for (let j = start; j < end; j++) {
                max = Math.max(max, scores[j]);
            }
            
            // Compute exponentials and sum
            let sum = 0;
            for (let j = start; j < end; j++) {
                softmaxed[j] = Math.exp(scores[j] - max);
                sum += softmaxed[j];
            }
            
            // Normalize
            for (let j = start; j < end; j++) {
                softmaxed[j] /= sum;
            }
        }
        
        return softmaxed;
    }
    
    getMetrics() {
        return {
            modelDim: this.modelDim,
            numHeads: this.numHeads,
            numLayers: this.numLayers,
            parameterCount: this.getParameterCount(),
            attentionMaps: this.attentionMaps.length
        };
    }
    
    getParameterCount() {
        let count = this.embeddings.length;
        count += this.encoderLayers.length * (
            this.modelDim * this.modelDim * 4 + // Attention weights
            this.modelDim * 4 + // Layer norm parameters
            this.modelDim * this.feedforwardDim * 2 + // FF weights
            this.feedforwardDim + this.modelDim // FF biases
        );
        count += this.outputProjection.weights.length + this.outputProjection.bias.length;
        return count;
    }
}

/**
 * Convolutional Neural Network
 * Optimized for spatial data processing with multiple conv layers
 */
export class ConvolutionalNetwork {
    constructor(options = {}) {
        this.inputShape = options.inputShape || [32, 32, 3]; // [height, width, channels]
        this.numClasses = options.numClasses || 10;
        this.layers = this.buildLayers(options.architecture || 'resnet18');
        
        // Performance tracking
        this.processedImages = 0;
        this.averageInferenceTime = 0;
        
        console.log(`🖼️ CNN initialized: ${this.inputShape.join('x')} → ${this.numClasses} classes`);
    }
    
    buildLayers(architecture) {
        switch (architecture) {
            case 'simple':
                return this.buildSimpleCNN();
            case 'resnet18':
                return this.buildResNet18();
            case 'efficientnet':
                return this.buildEfficientNet();
            default:
                return this.buildSimpleCNN();
        }
    }
    
    buildSimpleCNN() {
        return [
            // Conv1: 32x32x3 -> 30x30x32
            { type: 'conv2d', filters: 32, kernelSize: 3, stride: 1, padding: 0, activation: 'relu' },
            { type: 'maxpool', kernelSize: 2, stride: 2 }, // 30x30x32 -> 15x15x32
            
            // Conv2: 15x15x32 -> 13x13x64
            { type: 'conv2d', filters: 64, kernelSize: 3, stride: 1, padding: 0, activation: 'relu' },
            { type: 'maxpool', kernelSize: 2, stride: 2 }, // 13x13x64 -> 6x6x64
            
            // Conv3: 6x6x64 -> 4x4x128
            { type: 'conv2d', filters: 128, kernelSize: 3, stride: 1, padding: 0, activation: 'relu' },
            { type: 'maxpool', kernelSize: 2, stride: 2 }, // 4x4x128 -> 2x2x128
            
            // Flatten and Dense
            { type: 'flatten' },
            { type: 'dense', units: 256, activation: 'relu' },
            { type: 'dropout', rate: 0.5 },
            { type: 'dense', units: this.numClasses, activation: 'softmax' }
        ];
    }
    
    buildResNet18() {
        // Simplified ResNet-18 architecture
        return [
            // Initial conv layer
            { type: 'conv2d', filters: 64, kernelSize: 7, stride: 2, padding: 3, activation: 'relu' },
            { type: 'maxpool', kernelSize: 3, stride: 2, padding: 1 },
            
            // ResNet blocks
            ...this.createResNetBlock(64, 64, 2),
            ...this.createResNetBlock(64, 128, 2, 2),
            ...this.createResNetBlock(128, 256, 2, 2),
            ...this.createResNetBlock(256, 512, 2, 2),
            
            // Global average pooling and classifier
            { type: 'global_avg_pool' },
            { type: 'dense', units: this.numClasses, activation: 'softmax' }
        ];
    }
    
    createResNetBlock(inChannels, outChannels, numBlocks, stride = 1) {
        const blocks = [];
        
        for (let i = 0; i < numBlocks; i++) {
            const blockStride = i === 0 ? stride : 1;
            blocks.push({
                type: 'residual_block',
                inChannels: i === 0 ? inChannels : outChannels,
                outChannels,
                stride: blockStride,
                downsample: i === 0 && (inChannels !== outChannels || stride !== 1)
            });
        }
        
        return blocks;
    }
    
    async forward(input) {
        const startTime = performance.now();
        let current = input;
        let currentShape = [...this.inputShape];
        
        for (const layer of this.layers) {
            const result = await this.forwardLayer(current, layer, currentShape);
            current = result.output;
            currentShape = result.shape;
        }
        
        const inferenceTime = performance.now() - startTime;
        this.updatePerformanceMetrics(inferenceTime);
        
        return {
            output: current,
            shape: currentShape,
            inferenceTime
        };
    }
    
    async forwardLayer(input, layer, inputShape) {
        switch (layer.type) {
            case 'conv2d':
                return this.conv2d(input, layer, inputShape);
            case 'maxpool':
                return this.maxPool(input, layer, inputShape);
            case 'global_avg_pool':
                return this.globalAvgPool(input, inputShape);
            case 'flatten':
                return this.flatten(input, inputShape);
            case 'dense':
                return this.dense(input, layer, inputShape);
            case 'dropout':
                return this.dropout(input, layer, inputShape);
            case 'residual_block':
                return this.residualBlock(input, layer, inputShape);
            default:
                throw new Error(`Unknown layer type: ${layer.type}`);
        }
    }
    
    conv2d(input, layer, inputShape) {
        const [height, width, inChannels] = inputShape;
        const { filters, kernelSize, stride, padding, activation } = layer;
        
        const outHeight = Math.floor((height + 2 * padding - kernelSize) / stride) + 1;
        const outWidth = Math.floor((width + 2 * padding - kernelSize) / stride) + 1;
        const outputShape = [outHeight, outWidth, filters];
        
        const output = new Float32Array(outHeight * outWidth * filters);
        
        // Initialize weights if not present
        if (!layer.weights) {
            layer.weights = this.initializeConvWeights(kernelSize, inChannels, filters);
            layer.bias = new Float32Array(filters);
        }
        
        // Perform convolution
        for (let f = 0; f < filters; f++) {
            for (let y = 0; y < outHeight; y++) {
                for (let x = 0; x < outWidth; x++) {
                    let sum = layer.bias[f];
                    
                    for (let ky = 0; ky < kernelSize; ky++) {
                        for (let kx = 0; kx < kernelSize; kx++) {
                            for (let c = 0; c < inChannels; c++) {
                                const inputY = y * stride - padding + ky;
                                const inputX = x * stride - padding + kx;
                                
                                if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
                                    const inputIdx = (inputY * width + inputX) * inChannels + c;
                                    const weightIdx = ((f * kernelSize + ky) * kernelSize + kx) * inChannels + c;
                                    sum += input[inputIdx] * layer.weights[weightIdx];
                                }
                            }
                        }
                    }
                    
                    const outputIdx = (y * outWidth + x) * filters + f;
                    output[outputIdx] = this.applyActivation(sum, activation);
                }
            }
        }
        
        return { output, shape: outputShape };
    }
    
    maxPool(input, layer, inputShape) {
        const [height, width, channels] = inputShape;
        const { kernelSize, stride } = layer;
        
        const outHeight = Math.floor((height - kernelSize) / stride) + 1;
        const outWidth = Math.floor((width - kernelSize) / stride) + 1;
        const outputShape = [outHeight, outWidth, channels];
        
        const output = new Float32Array(outHeight * outWidth * channels);
        
        for (let c = 0; c < channels; c++) {
            for (let y = 0; y < outHeight; y++) {
                for (let x = 0; x < outWidth; x++) {
                    let maxVal = -Infinity;
                    
                    for (let ky = 0; ky < kernelSize; ky++) {
                        for (let kx = 0; kx < kernelSize; kx++) {
                            const inputY = y * stride + ky;
                            const inputX = x * stride + kx;
                            const inputIdx = (inputY * width + inputX) * channels + c;
                            maxVal = Math.max(maxVal, input[inputIdx]);
                        }
                    }
                    
                    const outputIdx = (y * outWidth + x) * channels + c;
                    output[outputIdx] = maxVal;
                }
            }
        }
        
        return { output, shape: outputShape };
    }
    
    globalAvgPool(input, inputShape) {
        const [height, width, channels] = inputShape;
        const output = new Float32Array(channels);
        
        for (let c = 0; c < channels; c++) {
            let sum = 0;
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = (y * width + x) * channels + c;
                    sum += input[idx];
                }
            }
            output[c] = sum / (height * width);
        }
        
        return { output, shape: [channels] };
    }
    
    flatten(input, inputShape) {
        const size = inputShape.reduce((a, b) => a * b, 1);
        return { output: input, shape: [size] };
    }
    
    dense(input, layer, inputShape) {
        const inputSize = inputShape[0];
        const { units, activation } = layer;
        
        if (!layer.weights) {
            layer.weights = this.initializeDenseWeights(inputSize, units);
            layer.bias = new Float32Array(units);
        }
        
        const output = new Float32Array(units);
        
        for (let i = 0; i < units; i++) {
            let sum = layer.bias[i];
            for (let j = 0; j < inputSize; j++) {
                sum += input[j] * layer.weights[i * inputSize + j];
            }
            output[i] = this.applyActivation(sum, activation);
        }
        
        return { output, shape: [units] };
    }
    
    dropout(input, layer, inputShape) {
        const { rate } = layer;
        const output = new Float32Array(input.length);
        
        for (let i = 0; i < input.length; i++) {
            output[i] = Math.random() < rate ? 0 : input[i] / (1 - rate);
        }
        
        return { output, shape: inputShape };
    }
    
    residualBlock(input, layer, inputShape) {
        // Simplified residual block implementation
        const { inChannels, outChannels, stride, downsample } = layer;
        
        // First conv layer
        let residual = input;
        let { output, shape } = this.conv2d(input, {
            filters: outChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1,
            activation: 'relu'
        }, inputShape);
        
        // Second conv layer
        ({ output, shape } = this.conv2d(output, {
            filters: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: 'none'
        }, shape));
        
        // Downsample residual if needed
        if (downsample) {
            residual = this.conv2d(residual, {
                filters: outChannels,
                kernelSize: 1,
                stride: stride,
                padding: 0,
                activation: 'none'
            }, inputShape).output;
        }
        
        // Add residual connection
        for (let i = 0; i < output.length; i++) {
            output[i] = this.applyActivation(output[i] + residual[i], 'relu');
        }
        
        return { output, shape };
    }
    
    initializeConvWeights(kernelSize, inChannels, outChannels) {
        const fanIn = kernelSize * kernelSize * inChannels;
        const fanOut = kernelSize * kernelSize * outChannels;
        const limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        const weights = new Float32Array(kernelSize * kernelSize * inChannels * outChannels);
        for (let i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2 - 1) * limit;
        }
        
        return weights;
    }
    
    initializeDenseWeights(inputSize, outputSize) {
        const limit = Math.sqrt(6.0 / (inputSize + outputSize));
        const weights = new Float32Array(inputSize * outputSize);
        
        for (let i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2 - 1) * limit;
        }
        
        return weights;
    }
    
    applyActivation(x, activation) {
        switch (activation) {
            case 'relu':
                return Math.max(0, x);
            case 'sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'tanh':
                return Math.tanh(x);
            case 'softmax':
                return Math.exp(x); // Note: requires normalization
            case 'none':
            default:
                return x;
        }
    }
    
    updatePerformanceMetrics(inferenceTime) {
        this.processedImages++;
        this.averageInferenceTime = (this.averageInferenceTime * (this.processedImages - 1) + inferenceTime) / this.processedImages;
    }
    
    getMetrics() {
        return {
            processedImages: this.processedImages,
            averageInferenceTime: this.averageInferenceTime,
            throughput: this.averageInferenceTime > 0 ? 1000 / this.averageInferenceTime : 0,
            parameterCount: this.getParameterCount(),
            inputShape: this.inputShape,
            numClasses: this.numClasses
        };
    }
    
    getParameterCount() {
        return this.layers.reduce((total, layer) => {
            if (layer.weights) {
                return total + layer.weights.length + (layer.bias ? layer.bias.length : 0);
            }
            return total;
        }, 0);
    }
}

export { LSTMNetwork, TransformerNetwork, ConvolutionalNetwork };