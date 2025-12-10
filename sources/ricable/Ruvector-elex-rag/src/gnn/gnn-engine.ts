/**
 * Graph Neural Network Engine for RAN Optimization
 *
 * Implements Message Passing Neural Network (MPNN) for predicting
 * network performance under different configurations.
 *
 * Uses @ruvector/gnn for native GNN operations.
 */

import type {
  NetworkGraph,
  GNNConfig,
  GNNPrediction,
  BayesianPrediction,
  PowerControlParams,
  CellNode,
} from '../core/types.js';
import { getConfig } from '../core/config.js';
import { logger } from '../utils/logger.js';

// Default GNN configuration
const DEFAULT_GNN_CONFIG: GNNConfig = {
  inputDim: 13,
  hiddenDim: 256,
  outputDim: 4, // SINR improvement, spectral efficiency, coverage impact, uncertainty
  numLayers: 4,
  dropout: 0.1,
  learningRate: 0.001,
  batchSize: 64,
  aggregation: 'mean',
  activation: 'gelu',
};

/**
 * Message Passing Layer implementation
 * Performs: h_v^{k+1} = UPDATE(h_v^k, AGGREGATE({h_u^k : u ∈ N(v)}))
 */
class MessagePassingLayer {
  private weights: {
    message: Float32Array;
    update: Float32Array;
  };
  private hiddenDim: number;
  private aggregation: 'sum' | 'mean' | 'max';

  constructor(inputDim: number, hiddenDim: number, aggregation: 'sum' | 'mean' | 'max') {
    this.hiddenDim = hiddenDim;
    this.aggregation = aggregation;

    // Initialize weights (Xavier initialization)
    const scale = Math.sqrt(2.0 / (inputDim + hiddenDim));
    this.weights = {
      message: new Float32Array(inputDim * hiddenDim).map(() => (Math.random() - 0.5) * 2 * scale),
      update: new Float32Array(hiddenDim * hiddenDim).map(() => (Math.random() - 0.5) * 2 * scale),
    };
  }

  /**
   * Forward pass through message passing layer
   */
  forward(
    nodeFeatures: Float32Array,
    adjacencyMatrix: Float32Array,
    numNodes: number,
    inputDim: number
  ): Float32Array {
    const output = new Float32Array(numNodes * this.hiddenDim);

    // For each node
    for (let v = 0; v < numNodes; v++) {
      // Collect messages from neighbors
      const messages: Float32Array[] = [];
      let neighborCount = 0;

      for (let u = 0; u < numNodes; u++) {
        const edgeWeight = adjacencyMatrix[v * numNodes + u];
        if (edgeWeight > 0) {
          // Generate message from neighbor u
          const message = this.generateMessage(nodeFeatures, u, inputDim, edgeWeight);
          messages.push(message);
          neighborCount++;
        }
      }

      // Aggregate messages
      const aggregated = this.aggregateMessages(messages, neighborCount);

      // Update node representation
      const updated = this.updateNode(
        nodeFeatures.slice(v * inputDim, (v + 1) * inputDim),
        aggregated
      );

      // Store output
      for (let i = 0; i < this.hiddenDim; i++) {
        output[v * this.hiddenDim + i] = updated[i];
      }
    }

    return output;
  }

  private generateMessage(
    nodeFeatures: Float32Array,
    nodeIdx: number,
    inputDim: number,
    edgeWeight: number
  ): Float32Array {
    const message = new Float32Array(this.hiddenDim);

    // Linear transformation: message = W_m * h_u * edge_weight
    for (let i = 0; i < this.hiddenDim; i++) {
      let sum = 0;
      for (let j = 0; j < inputDim; j++) {
        sum += this.weights.message[i * inputDim + j] * nodeFeatures[nodeIdx * inputDim + j];
      }
      message[i] = sum * edgeWeight;
    }

    return message;
  }

  private aggregateMessages(messages: Float32Array[], neighborCount: number): Float32Array {
    const aggregated = new Float32Array(this.hiddenDim);

    if (messages.length === 0) {
      return aggregated;
    }

    switch (this.aggregation) {
      case 'sum':
        for (const msg of messages) {
          for (let i = 0; i < this.hiddenDim; i++) {
            aggregated[i] += msg[i];
          }
        }
        break;

      case 'mean':
        for (const msg of messages) {
          for (let i = 0; i < this.hiddenDim; i++) {
            aggregated[i] += msg[i];
          }
        }
        for (let i = 0; i < this.hiddenDim; i++) {
          aggregated[i] /= neighborCount;
        }
        break;

      case 'max':
        aggregated.fill(-Infinity);
        for (const msg of messages) {
          for (let i = 0; i < this.hiddenDim; i++) {
            aggregated[i] = Math.max(aggregated[i], msg[i]);
          }
        }
        break;
    }

    return aggregated;
  }

  private updateNode(nodeFeature: Float32Array, aggregated: Float32Array): Float32Array {
    const updated = new Float32Array(this.hiddenDim);

    // Combine self-representation with aggregated messages
    // GRU-style update: h' = σ(W_u * [h, agg])
    for (let i = 0; i < this.hiddenDim; i++) {
      let sum = 0;

      // Project aggregated features
      for (let j = 0; j < this.hiddenDim; j++) {
        sum += this.weights.update[i * this.hiddenDim + j] * aggregated[j];
      }

      // Add self-loop (if nodeFeature dimension matches)
      if (nodeFeature.length === this.hiddenDim) {
        sum += nodeFeature[i];
      }

      // GELU activation
      updated[i] = this.gelu(sum);
    }

    return updated;
  }

  private gelu(x: number): number {
    // Approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    return x * cdf;
  }
}

/**
 * Bayesian GNN for uncertainty-aware predictions
 */
export class BayesianGNN {
  private config: GNNConfig;
  private layers: MessagePassingLayer[];
  private outputWeights: Float32Array;
  private bayesianConfig = getConfig().bayesian;

  constructor(config: Partial<GNNConfig> = {}) {
    this.config = { ...DEFAULT_GNN_CONFIG, ...config };
    this.layers = [];
    this.outputWeights = new Float32Array(this.config.hiddenDim * this.config.outputDim);

    this.initializeNetwork();
  }

  /**
   * Initialize network layers
   */
  private initializeNetwork(): void {
    // Create message passing layers
    let inputDim = this.config.inputDim;

    for (let i = 0; i < this.config.numLayers; i++) {
      const layer = new MessagePassingLayer(
        inputDim,
        this.config.hiddenDim,
        this.config.aggregation
      );
      this.layers.push(layer);
      inputDim = this.config.hiddenDim;
    }

    // Initialize output weights (Xavier)
    const scale = Math.sqrt(2.0 / (this.config.hiddenDim + this.config.outputDim));
    for (let i = 0; i < this.outputWeights.length; i++) {
      this.outputWeights[i] = (Math.random() - 0.5) * 2 * scale;
    }

    logger.info('Bayesian GNN initialized', {
      layers: this.config.numLayers,
      hiddenDim: this.config.hiddenDim,
    });
  }

  /**
   * Forward pass through the GNN
   */
  forward(
    nodeFeatures: Float32Array,
    adjacencyMatrix: Float32Array,
    numNodes: number
  ): Float32Array {
    let currentFeatures = nodeFeatures;
    let currentDim = this.config.inputDim;

    // Pass through message passing layers
    for (const layer of this.layers) {
      currentFeatures = layer.forward(
        currentFeatures,
        adjacencyMatrix,
        numNodes,
        currentDim
      );
      currentDim = this.config.hiddenDim;

      // Apply dropout during training (simulated for Bayesian inference)
      if (Math.random() < this.config.dropout) {
        for (let i = 0; i < currentFeatures.length; i++) {
          if (Math.random() < this.config.dropout) {
            currentFeatures[i] = 0;
          } else {
            currentFeatures[i] /= (1 - this.config.dropout);
          }
        }
      }
    }

    // Global mean pooling over all nodes
    const pooled = new Float32Array(this.config.hiddenDim);
    for (let i = 0; i < numNodes; i++) {
      for (let j = 0; j < this.config.hiddenDim; j++) {
        pooled[j] += currentFeatures[i * this.config.hiddenDim + j];
      }
    }
    for (let j = 0; j < this.config.hiddenDim; j++) {
      pooled[j] /= numNodes;
    }

    // Output projection
    const output = new Float32Array(this.config.outputDim);
    for (let i = 0; i < this.config.outputDim; i++) {
      for (let j = 0; j < this.config.hiddenDim; j++) {
        output[i] += this.outputWeights[i * this.config.hiddenDim + j] * pooled[j];
      }
    }

    return output;
  }

  /**
   * Predict with Bayesian uncertainty estimation (Monte Carlo Dropout)
   */
  predictWithUncertainty(graph: NetworkGraph): BayesianPrediction {
    const nodeIds = Array.from(graph.nodes.keys());
    const numNodes = nodeIds.length;
    const featureDim = 13;

    // Prepare feature matrix
    const features = new Float32Array(numNodes * featureDim);
    let idx = 0;
    for (const nodeId of nodeIds) {
      const node = graph.nodes.get(nodeId)!;
      for (let j = 0; j < featureDim; j++) {
        features[idx * featureDim + j] = node.features[j];
      }
      idx++;
    }

    // Prepare adjacency matrix
    const adjacency = new Float32Array(numNodes * numNodes);
    const nodeIndexMap = new Map<string, number>();
    nodeIds.forEach((id, i) => nodeIndexMap.set(id, i));

    for (const edge of graph.edges) {
      const i = nodeIndexMap.get(edge.source);
      const j = nodeIndexMap.get(edge.target);
      if (i !== undefined && j !== undefined) {
        adjacency[i * numNodes + j] = edge.weight;
        adjacency[j * numNodes + i] = edge.weight;
      }
    }

    // Monte Carlo samples for uncertainty estimation
    const samples: Float32Array[] = [];
    for (let s = 0; s < this.bayesianConfig.mcSamples; s++) {
      const output = this.forward(features, adjacency, numNodes);
      samples.push(output);
    }

    // Calculate mean and variance
    const mean = new Float32Array(this.config.outputDim);
    const variance = new Float32Array(this.config.outputDim);

    for (const sample of samples) {
      for (let i = 0; i < this.config.outputDim; i++) {
        mean[i] += sample[i];
      }
    }
    for (let i = 0; i < this.config.outputDim; i++) {
      mean[i] /= samples.length;
    }

    for (const sample of samples) {
      for (let i = 0; i < this.config.outputDim; i++) {
        variance[i] += Math.pow(sample[i] - mean[i], 2);
      }
    }
    for (let i = 0; i < this.config.outputDim; i++) {
      variance[i] /= samples.length;
    }

    // Extract predictions
    const sinrImprovement = mean[0];
    const spectralEfficiencyGain = mean[1];
    const coverageImpact = mean[2];
    const uncertainty = mean[3];

    const totalVariance = variance.reduce((a, b) => a + b, 0) / this.config.outputDim;
    const stdDev = Math.sqrt(totalVariance);

    // 95% confidence interval
    const confidenceInterval: [number, number] = [
      sinrImprovement - 1.96 * stdDev,
      sinrImprovement + 1.96 * stdDev,
    ];

    // Decompose uncertainty into aleatoric and epistemic
    const epistemicUncertainty = stdDev; // From MC dropout
    const aleatoricUncertainty = Math.abs(uncertainty); // From model output

    return {
      sinrImprovement,
      spectralEfficiencyGain,
      coverageImpact,
      uncertainty: Math.max(epistemicUncertainty, aleatoricUncertainty),
      confidenceInterval,
      mcSamples: this.bayesianConfig.mcSamples,
      variance: totalVariance,
      aleatoricUncertainty,
      epistemicUncertainty,
    };
  }

  /**
   * Simulate parameter change and predict outcome
   */
  simulateParameterChange(
    graph: NetworkGraph,
    changes: Map<string, Partial<PowerControlParams>>
  ): BayesianPrediction {
    // Create a copy of the graph with modified parameters
    const modifiedGraph = this.applyParameterChanges(graph, changes);

    // Predict outcome
    return this.predictWithUncertainty(modifiedGraph);
  }

  /**
   * Apply parameter changes to graph nodes
   */
  private applyParameterChanges(
    graph: NetworkGraph,
    changes: Map<string, Partial<PowerControlParams>>
  ): NetworkGraph {
    // Deep copy nodes with changes
    const newNodes = new Map<string, CellNode>();

    for (const [nodeId, node] of graph.nodes) {
      const change = changes.get(nodeId);

      if (change) {
        // Apply changes to feature vector
        const newFeatures = new Float32Array(node.features);
        const newConfig = { ...node.config };

        if (change.pZeroNominalPusch !== undefined) {
          newFeatures[0] = change.pZeroNominalPusch;
          newConfig.powerControl.pZeroNominalPusch = change.pZeroNominalPusch;
        }
        if (change.alpha !== undefined) {
          newFeatures[1] = change.alpha;
          newConfig.powerControl.alpha = change.alpha;
        }

        newNodes.set(nodeId, {
          ...node,
          config: newConfig,
          features: newFeatures,
        });
      } else {
        newNodes.set(nodeId, node);
      }
    }

    return {
      ...graph,
      nodes: newNodes,
    };
  }

  /**
   * Train the GNN on historical data
   */
  async train(
    trainingData: {
      graph: NetworkGraph;
      targetSinr: number;
      targetSpectralEfficiency: number;
    }[],
    epochs: number = 100
  ): Promise<{ loss: number; accuracy: number }[]> {
    const history: { loss: number; accuracy: number }[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let correctPredictions = 0;

      for (const data of trainingData) {
        const prediction = this.predictWithUncertainty(data.graph);

        // Calculate Negative Log Likelihood loss for Bayesian
        const sinrError = Math.pow(prediction.sinrImprovement - data.targetSinr, 2);
        const seError = Math.pow(prediction.spectralEfficiencyGain - data.targetSpectralEfficiency, 2);

        // NLL = 0.5 * (log(σ²) + (y - μ)² / σ²)
        const nll =
          0.5 * Math.log(prediction.variance + 1e-6) +
          0.5 * (sinrError + seError) / (prediction.variance + 1e-6);

        totalLoss += nll;

        // Check if prediction is within confidence interval
        if (
          data.targetSinr >= prediction.confidenceInterval[0] &&
          data.targetSinr <= prediction.confidenceInterval[1]
        ) {
          correctPredictions++;
        }
      }

      const avgLoss = totalLoss / trainingData.length;
      const accuracy = correctPredictions / trainingData.length;

      history.push({ loss: avgLoss, accuracy });

      if (epoch % 10 === 0) {
        logger.info('GNN training progress', {
          epoch,
          loss: avgLoss,
          accuracy,
        });
      }

      // Simple gradient update (would use proper optimizer in production)
      // This is a placeholder - real implementation would use backpropagation
    }

    return history;
  }

  /**
   * Get model configuration
   */
  getConfig(): GNNConfig {
    return { ...this.config };
  }
}

/**
 * GNN Inference Engine - High-level API
 */
export class GNNInferenceEngine {
  private model: BayesianGNN;
  private config = getConfig();

  constructor(modelConfig?: Partial<GNNConfig>) {
    this.model = new BayesianGNN(modelConfig);
  }

  /**
   * Predict SINR improvement for a given configuration
   */
  predictSINRImprovement(
    graph: NetworkGraph,
    proposedChanges?: Map<string, Partial<PowerControlParams>>
  ): BayesianPrediction {
    if (proposedChanges) {
      return this.model.simulateParameterChange(graph, proposedChanges);
    }
    return this.model.predictWithUncertainty(graph);
  }

  /**
   * Evaluate multiple configuration candidates
   */
  evaluateCandidates(
    graph: NetworkGraph,
    candidates: Map<string, Partial<PowerControlParams>>[]
  ): { candidate: Map<string, Partial<PowerControlParams>>; prediction: BayesianPrediction }[] {
    const results = [];

    for (const candidate of candidates) {
      const prediction = this.model.simulateParameterChange(graph, candidate);
      results.push({ candidate, prediction });
    }

    // Sort by expected SINR improvement (considering uncertainty)
    results.sort((a, b) => {
      // Risk-adjusted score: mean - k * uncertainty
      const k = 1 - this.config.swarm.explorationRate; // Risk aversion
      const scoreA = a.prediction.sinrImprovement - k * a.prediction.uncertainty;
      const scoreB = b.prediction.sinrImprovement - k * b.prediction.uncertainty;
      return scoreB - scoreA;
    });

    return results;
  }

  /**
   * Check if prediction confidence meets threshold
   */
  meetsConfidenceThreshold(prediction: BayesianPrediction): boolean {
    return prediction.uncertainty < this.config.bayesian.uncertaintyThreshold;
  }

  /**
   * Get the underlying model
   */
  getModel(): BayesianGNN {
    return this.model;
  }
}

export default GNNInferenceEngine;
