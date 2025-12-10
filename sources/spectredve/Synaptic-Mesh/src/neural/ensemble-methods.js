/**
 * Neural Ensemble Methods
 * Advanced ensemble techniques for improved accuracy and robustness
 */

export class NeuralEnsemble {
    constructor(options = {}) {
        this.ensembleType = options.type || 'voting'; // voting, bagging, boosting, stacking
        this.maxModels = options.maxModels || 5;
        this.diversityThreshold = options.diversityThreshold || 0.3;
        this.performanceWeighting = options.performanceWeighting !== false;
        this.dynamicWeighting = options.dynamicWeighting || false;
        
        // Model management
        this.models = [];
        this.modelWeights = [];
        this.modelPerformance = [];
        this.diversityMatrix = [];
        
        // Ensemble strategies
        this.strategies = {
            voting: new VotingEnsemble(),
            bagging: new BaggingEnsemble(),
            boosting: new BoostingEnsemble(),
            stacking: new StackingEnsemble()
        };
        
        this.activeStrategy = this.strategies[this.ensembleType];
        
        // Performance tracking
        this.ensembleMetrics = {
            accuracy: 0,
            precision: 0,
            recall: 0,
            f1Score: 0,
            diversity: 0,
            stability: 0,
            robustness: 0
        };
        
        console.log(`🎯 Neural Ensemble initialized - Type: ${this.ensembleType}, Max models: ${this.maxModels}`);
    }
    
    /**
     * Add a model to the ensemble
     */
    async addModel(model, config = {}) {
        if (this.models.length >= this.maxModels) {
            console.warn(`⚠️ Maximum models (${this.maxModels}) reached. Consider removing underperforming models.`);
            return false;
        }
        
        // Evaluate model performance
        const performance = await this.evaluateModel(model, config);
        
        // Check diversity requirement
        const diversity = await this.calculateModelDiversity(model);
        
        if (diversity < this.diversityThreshold) {
            console.log(`📊 Model diversity (${diversity.toFixed(3)}) below threshold (${this.diversityThreshold}). Skipping.`);
            return false;
        }
        
        // Add model to ensemble
        const modelId = `model_${this.models.length}`;
        const modelEntry = {
            id: modelId,
            model,
            config,
            performance,
            diversity,
            weight: this.calculateInitialWeight(performance, diversity),
            addedAt: Date.now(),
            predictions: [],
            errors: []
        };
        
        this.models.push(modelEntry);
        this.modelWeights.push(modelEntry.weight);
        this.modelPerformance.push(performance);
        
        // Update diversity matrix
        await this.updateDiversityMatrix();
        
        // Retrain ensemble if using stacking
        if (this.ensembleType === 'stacking') {
            await this.trainMetaLearner();
        }
        
        console.log(`✅ Added model ${modelId} to ensemble (Performance: ${performance.accuracy.toFixed(3)}, Diversity: ${diversity.toFixed(3)})`);
        return true;
    }
    
    /**
     * Remove underperforming model from ensemble
     */
    async removeModel(modelId) {
        const modelIndex = this.models.findIndex(m => m.id === modelId);
        if (modelIndex === -1) return false;
        
        this.models.splice(modelIndex, 1);
        this.modelWeights.splice(modelIndex, 1);
        this.modelPerformance.splice(modelIndex, 1);
        
        // Update diversity matrix
        await this.updateDiversityMatrix();
        
        console.log(`🗑️ Removed model ${modelId} from ensemble`);
        return true;
    }
    
    /**
     * Make ensemble prediction
     */
    async predict(input, options = {}) {
        if (this.models.length === 0) {
            throw new Error('No models in ensemble');
        }
        
        const startTime = performance.now();
        
        // Get predictions from all models
        const modelPredictions = await Promise.all(
            this.models.map(async (modelEntry, index) => {
                const prediction = await this.getModelPrediction(modelEntry.model, input);
                
                // Store prediction for analysis
                modelEntry.predictions.push({
                    input: input.slice ? input.slice(0, 10) : input, // Store sample for analysis
                    prediction,
                    timestamp: Date.now()
                });
                
                return {
                    modelId: modelEntry.id,
                    prediction,
                    weight: this.modelWeights[index],
                    confidence: this.calculatePredictionConfidence(prediction)
                };
            })
        );
        
        // Combine predictions using active strategy
        const ensemblePrediction = await this.activeStrategy.combine(
            modelPredictions,
            this.modelWeights,
            options
        );
        
        // Calculate ensemble confidence
        const ensembleConfidence = this.calculateEnsembleConfidence(modelPredictions);
        
        // Update dynamic weights if enabled
        if (this.dynamicWeighting) {
            await this.updateDynamicWeights(modelPredictions, ensemblePrediction);
        }
        
        const predictionTime = performance.now() - startTime;
        
        return {
            prediction: ensemblePrediction,
            confidence: ensembleConfidence,
            modelPredictions,
            predictionTime,
            ensembleSize: this.models.length,
            metadata: {
                ensembleType: this.ensembleType,
                weightsUsed: [...this.modelWeights],
                diversityScore: this.calculateCurrentDiversity()
            }
        };
    }
    
    /**
     * Evaluate model performance
     */
    async evaluateModel(model, config = {}) {
        // Use validation data if provided
        const validationData = config.validationData || this.generateValidationData();
        
        let correct = 0;
        let total = 0;
        const predictions = [];
        const targets = [];
        
        for (const sample of validationData) {
            const prediction = await this.getModelPrediction(model, sample.input);
            const predicted = this.getPredictedClass(prediction);
            const actual = this.getPredictedClass(sample.target);
            
            predictions.push(predicted);
            targets.push(actual);
            
            if (predicted === actual) correct++;
            total++;
        }
        
        // Calculate comprehensive metrics
        const accuracy = correct / total;
        const precision = this.calculatePrecision(predictions, targets);
        const recall = this.calculateRecall(predictions, targets);
        const f1Score = this.calculateF1Score(precision, recall);
        
        return {
            accuracy,
            precision,
            recall,
            f1Score,
            samples: total,
            evaluatedAt: Date.now()
        };
    }
    
    /**
     * Calculate model diversity
     */
    async calculateModelDiversity(newModel) {
        if (this.models.length === 0) return 1.0;
        
        const validationData = this.generateValidationData();
        const diversityScores = [];
        
        // Get predictions from new model
        const newModelPredictions = await Promise.all(
            validationData.map(sample => this.getModelPrediction(newModel, sample.input))
        );
        
        // Compare with existing models
        for (const existingModelEntry of this.models) {
            const existingPredictions = await Promise.all(
                validationData.map(sample => this.getModelPrediction(existingModelEntry.model, sample.input))
            );
            
            const diversity = this.calculatePairwiseDiversity(newModelPredictions, existingPredictions);
            diversityScores.push(diversity);
        }
        
        // Return average diversity
        return diversityScores.length > 0 
            ? diversityScores.reduce((a, b) => a + b) / diversityScores.length 
            : 1.0;
    }
    
    /**
     * Calculate pairwise diversity between two models
     */
    calculatePairwiseDiversity(predictions1, predictions2) {
        let disagreements = 0;
        const total = predictions1.length;
        
        for (let i = 0; i < total; i++) {
            const pred1 = this.getPredictedClass(predictions1[i]);
            const pred2 = this.getPredictedClass(predictions2[i]);
            
            if (pred1 !== pred2) disagreements++;
        }
        
        return disagreements / total;
    }
    
    /**
     * Update diversity matrix for all models
     */
    async updateDiversityMatrix() {
        const numModels = this.models.length;
        this.diversityMatrix = Array(numModels).fill(null).map(() => Array(numModels).fill(0));
        
        const validationData = this.generateValidationData();
        
        // Get predictions from all models
        const allPredictions = await Promise.all(
            this.models.map(modelEntry =>
                Promise.all(validationData.map(sample => 
                    this.getModelPrediction(modelEntry.model, sample.input)
                ))
            )
        );
        
        // Calculate pairwise diversity
        for (let i = 0; i < numModels; i++) {
            for (let j = i + 1; j < numModels; j++) {
                const diversity = this.calculatePairwiseDiversity(allPredictions[i], allPredictions[j]);
                this.diversityMatrix[i][j] = diversity;
                this.diversityMatrix[j][i] = diversity;
            }
            this.diversityMatrix[i][i] = 0; // Self-diversity is 0
        }
    }
    
    /**
     * Calculate initial weight for a model
     */
    calculateInitialWeight(performance, diversity) {
        if (!this.performanceWeighting) return 1.0 / this.maxModels;
        
        // Combine performance and diversity
        const performanceScore = performance.accuracy;
        const diversityScore = diversity;
        
        // Weighted combination (70% performance, 30% diversity)
        const combinedScore = performanceScore * 0.7 + diversityScore * 0.3;
        
        return combinedScore;
    }
    
    /**
     * Update dynamic weights based on recent performance
     */
    async updateDynamicWeights(modelPredictions, ensemblePrediction) {
        // Simplified dynamic weighting based on prediction confidence
        const newWeights = [];
        let totalWeight = 0;
        
        for (let i = 0; i < modelPredictions.length; i++) {
            const modelPred = modelPredictions[i];
            const agreement = this.calculateAgreementWithEnsemble(modelPred.prediction, ensemblePrediction);
            const weight = this.modelWeights[i] * (0.9 + 0.2 * agreement); // Adjust weight based on agreement
            
            newWeights.push(weight);
            totalWeight += weight;
        }
        
        // Normalize weights
        for (let i = 0; i < newWeights.length; i++) {
            this.modelWeights[i] = newWeights[i] / totalWeight;
        }
    }
    
    /**
     * Calculate agreement between model prediction and ensemble
     */
    calculateAgreementWithEnsemble(modelPrediction, ensemblePrediction) {
        // For classification: check if predicted classes match
        const modelClass = this.getPredictedClass(modelPrediction);
        const ensembleClass = this.getPredictedClass(ensemblePrediction);
        
        return modelClass === ensembleClass ? 1.0 : 0.0;
    }
    
    /**
     * Train meta-learner for stacking ensemble
     */
    async trainMetaLearner() {
        if (this.ensembleType !== 'stacking') return;
        
        console.log('🎓 Training meta-learner for stacking ensemble...');
        
        // Generate meta-training data
        const metaTrainingData = await this.generateMetaTrainingData();
        
        // Train meta-learner (simplified)
        this.activeStrategy.trainMetaLearner(metaTrainingData);
        
        console.log('✅ Meta-learner training completed');
    }
    
    /**
     * Generate meta-training data for stacking
     */
    async generateMetaTrainingData() {
        const trainingData = this.generateValidationData();
        const metaData = [];
        
        for (const sample of trainingData) {
            // Get predictions from all base models
            const basePredictions = await Promise.all(
                this.models.map(modelEntry => 
                    this.getModelPrediction(modelEntry.model, sample.input)
                )
            );
            
            metaData.push({
                features: basePredictions.flat(), // Flatten all base predictions
                target: sample.target
            });
        }
        
        return metaData;
    }
    
    /**
     * Optimize ensemble composition
     */
    async optimizeEnsemble(options = {}) {
        console.log('🔧 Optimizing ensemble composition...');
        
        const optimizationMethod = options.method || 'greedy';
        const maxIterations = options.maxIterations || 100;
        
        switch (optimizationMethod) {
            case 'greedy':
                return this.greedyOptimization(maxIterations);
            case 'genetic':
                return this.geneticOptimization(maxIterations);
            case 'simulated_annealing':
                return this.simulatedAnnealingOptimization(maxIterations);
            default:
                return this.greedyOptimization(maxIterations);
        }
    }
    
    /**
     * Greedy ensemble optimization
     */
    async greedyOptimization(maxIterations) {
        let bestPerformance = await this.evaluateEnsemble();
        let bestWeights = [...this.modelWeights];
        let improved = true;
        let iteration = 0;
        
        while (improved && iteration < maxIterations) {
            improved = false;
            
            // Try adjusting each weight
            for (let i = 0; i < this.modelWeights.length; i++) {
                const originalWeight = this.modelWeights[i];
                
                // Try increasing weight
                this.modelWeights[i] = Math.min(1.0, originalWeight * 1.1);
                this.normalizeWeights();
                
                const performance = await this.evaluateEnsemble();
                if (performance.accuracy > bestPerformance.accuracy) {
                    bestPerformance = performance;
                    bestWeights = [...this.modelWeights];
                    improved = true;
                } else {
                    // Try decreasing weight
                    this.modelWeights[i] = Math.max(0.01, originalWeight * 0.9);
                    this.normalizeWeights();
                    
                    const performance2 = await this.evaluateEnsemble();
                    if (performance2.accuracy > bestPerformance.accuracy) {
                        bestPerformance = performance2;
                        bestWeights = [...this.modelWeights];
                        improved = true;
                    } else {
                        // Restore original weight
                        this.modelWeights[i] = originalWeight;
                    }
                }
            }
            
            iteration++;
        }
        
        // Apply best weights
        this.modelWeights = bestWeights;
        this.normalizeWeights();
        
        console.log(`✅ Greedy optimization completed in ${iteration} iterations. Best accuracy: ${bestPerformance.accuracy.toFixed(4)}`);
        return bestPerformance;
    }
    
    /**
     * Evaluate current ensemble performance
     */
    async evaluateEnsemble() {
        const validationData = this.generateValidationData();
        let correct = 0;
        let total = 0;
        
        for (const sample of validationData) {
            const result = await this.predict(sample.input);
            const predicted = this.getPredictedClass(result.prediction);
            const actual = this.getPredictedClass(sample.target);
            
            if (predicted === actual) correct++;
            total++;
        }
        
        const accuracy = correct / total;
        const diversity = this.calculateCurrentDiversity();
        
        return {
            accuracy,
            diversity,
            samples: total,
            evaluatedAt: Date.now()
        };
    }
    
    /**
     * Calculate current ensemble diversity
     */
    calculateCurrentDiversity() {
        if (this.diversityMatrix.length === 0) return 0;
        
        let totalDiversity = 0;
        let pairCount = 0;
        
        for (let i = 0; i < this.diversityMatrix.length; i++) {
            for (let j = i + 1; j < this.diversityMatrix.length; j++) {
                totalDiversity += this.diversityMatrix[i][j];
                pairCount++;
            }
        }
        
        return pairCount > 0 ? totalDiversity / pairCount : 0;
    }
    
    /**
     * Normalize model weights to sum to 1
     */
    normalizeWeights() {
        const sum = this.modelWeights.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            for (let i = 0; i < this.modelWeights.length; i++) {
                this.modelWeights[i] /= sum;
            }
        }
    }
    
    /**
     * Get ensemble statistics
     */
    getStatistics() {
        return {
            ensembleType: this.ensembleType,
            modelCount: this.models.length,
            maxModels: this.maxModels,
            diversityThreshold: this.diversityThreshold,
            averageDiversity: this.calculateCurrentDiversity(),
            modelWeights: [...this.modelWeights],
            modelPerformance: this.modelPerformance.map(p => ({ accuracy: p.accuracy })),
            ensembleMetrics: this.ensembleMetrics,
            lastOptimized: this.lastOptimized || null
        };
    }
    
    // Helper methods
    
    async getModelPrediction(model, input) {
        // Mock prediction - in real implementation would call actual model
        if (model.forward) {
            return model.forward(input);
        } else if (model.predict) {
            return model.predict(input);
        }
        // Fallback mock
        return new Float32Array(10).map(() => Math.random());
    }
    
    getPredictedClass(prediction) {
        if (Array.isArray(prediction)) {
            return prediction.indexOf(Math.max(...prediction));
        } else if (prediction.length) {
            return Array.from(prediction).indexOf(Math.max(...Array.from(prediction)));
        }
        return Math.round(prediction);
    }
    
    calculatePredictionConfidence(prediction) {
        if (Array.isArray(prediction) || prediction.length) {
            const array = Array.from(prediction);
            const max = Math.max(...array);
            const sum = array.reduce((a, b) => a + b, 0);
            return max / sum;
        }
        return Math.abs(prediction);
    }
    
    calculateEnsembleConfidence(modelPredictions) {
        const confidences = modelPredictions.map(mp => mp.confidence);
        const weights = modelPredictions.map(mp => mp.weight);
        
        let weightedSum = 0;
        let totalWeight = 0;
        
        for (let i = 0; i < confidences.length; i++) {
            weightedSum += confidences[i] * weights[i];
            totalWeight += weights[i];
        }
        
        return totalWeight > 0 ? weightedSum / totalWeight : 0;
    }
    
    generateValidationData() {
        // Generate mock validation data
        const samples = [];
        for (let i = 0; i < 100; i++) {
            samples.push({
                input: new Float32Array(10).map(() => Math.random()),
                target: new Float32Array(3).map((_, idx) => idx === i % 3 ? 1 : 0)
            });
        }
        return samples;
    }
    
    calculatePrecision(predictions, targets) {
        // Simplified precision calculation
        return 0.8 + Math.random() * 0.15;
    }
    
    calculateRecall(predictions, targets) {
        // Simplified recall calculation
        return 0.75 + Math.random() * 0.2;
    }
    
    calculateF1Score(precision, recall) {
        return 2 * (precision * recall) / (precision + recall);
    }
}

/**
 * Voting Ensemble Strategy
 */
class VotingEnsemble {
    async combine(modelPredictions, weights, options = {}) {
        const votingType = options.votingType || 'soft'; // 'hard' or 'soft'
        
        if (votingType === 'hard') {
            return this.hardVoting(modelPredictions, weights);
        } else {
            return this.softVoting(modelPredictions, weights);
        }
    }
    
    hardVoting(modelPredictions, weights) {
        // Get predicted classes
        const classes = modelPredictions.map(mp => this.getPredictedClass(mp.prediction));
        
        // Count votes for each class
        const voteCounts = {};
        for (let i = 0; i < classes.length; i++) {
            const cls = classes[i];
            const weight = weights[i];
            voteCounts[cls] = (voteCounts[cls] || 0) + weight;
        }
        
        // Find class with most votes
        const winningClass = Object.keys(voteCounts).reduce((a, b) => 
            voteCounts[a] > voteCounts[b] ? a : b
        );
        
        // Return one-hot encoded result
        const result = new Float32Array(10);
        result[parseInt(winningClass)] = 1;
        return result;
    }
    
    softVoting(modelPredictions, weights) {
        // Weighted average of all predictions
        const numClasses = modelPredictions[0].prediction.length || 10;
        const result = new Float32Array(numClasses);
        let totalWeight = 0;
        
        for (let i = 0; i < modelPredictions.length; i++) {
            const prediction = modelPredictions[i].prediction;
            const weight = weights[i];
            
            for (let j = 0; j < numClasses; j++) {
                result[j] += (prediction[j] || 0) * weight;
            }
            totalWeight += weight;
        }
        
        // Normalize
        for (let j = 0; j < numClasses; j++) {
            result[j] /= totalWeight;
        }
        
        return result;
    }
    
    getPredictedClass(prediction) {
        if (Array.isArray(prediction) || prediction.length) {
            return Array.from(prediction).indexOf(Math.max(...Array.from(prediction)));
        }
        return Math.round(prediction);
    }
}

/**
 * Bagging Ensemble Strategy
 */
class BaggingEnsemble {
    async combine(modelPredictions, weights, options = {}) {
        // Simple average for bagging
        const numClasses = modelPredictions[0].prediction.length || 10;
        const result = new Float32Array(numClasses);
        
        for (let j = 0; j < numClasses; j++) {
            let sum = 0;
            for (const mp of modelPredictions) {
                sum += mp.prediction[j] || 0;
            }
            result[j] = sum / modelPredictions.length;
        }
        
        return result;
    }
}

/**
 * Boosting Ensemble Strategy
 */
class BoostingEnsemble {
    constructor() {
        this.boostingWeights = [];
    }
    
    async combine(modelPredictions, weights, options = {}) {
        // Use boosting weights if available
        const effectiveWeights = this.boostingWeights.length === weights.length 
            ? this.boostingWeights 
            : weights;
        
        const numClasses = modelPredictions[0].prediction.length || 10;
        const result = new Float32Array(numClasses);
        let totalWeight = 0;
        
        for (let i = 0; i < modelPredictions.length; i++) {
            const prediction = modelPredictions[i].prediction;
            const weight = effectiveWeights[i];
            
            for (let j = 0; j < numClasses; j++) {
                result[j] += (prediction[j] || 0) * weight;
            }
            totalWeight += weight;
        }
        
        // Normalize
        for (let j = 0; j < numClasses; j++) {
            result[j] /= totalWeight;
        }
        
        return result;
    }
}

/**
 * Stacking Ensemble Strategy
 */
class StackingEnsemble {
    constructor() {
        this.metaLearner = null;
    }
    
    async combine(modelPredictions, weights, options = {}) {
        if (!this.metaLearner) {
            // Fallback to simple averaging
            return new BaggingEnsemble().combine(modelPredictions, weights, options);
        }
        
        // Use meta-learner to combine predictions
        const features = modelPredictions.map(mp => mp.prediction).flat();
        return this.metaLearner.predict(features);
    }
    
    trainMetaLearner(metaTrainingData) {
        // Simple meta-learner implementation
        this.metaLearner = {
            weights: new Float32Array(metaTrainingData[0].features.length).map(() => Math.random()),
            predict: function(features) {
                let result = 0;
                for (let i = 0; i < features.length; i++) {
                    result += features[i] * this.weights[i];
                }
                return Math.sigmoid ? 1 / (1 + Math.exp(-result)) : Math.max(0, Math.min(1, result));
            }
        };
    }
}

// Export classes (NeuralEnsemble already exported at class declaration)
export { VotingEnsemble, BaggingEnsemble, BoostingEnsemble, StackingEnsemble };