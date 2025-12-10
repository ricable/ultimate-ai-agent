/**
 * Decision Transformer - Offline Reinforcement Learning Engine
 *
 * Unlike traditional online RL (Q-Learning, PPO) which requires dangerous
 * exploration on live networks, Decision Transformer treats RL as a
 * sequence modeling problem. It learns from historical logs of (State, Action, Reward)
 * sequences and can generate optimal actions conditioned on desired rewards.
 *
 * Architecture:
 * - GPT-style transformer with causal attention
 * - Returns-to-go conditioning for goal-directed behavior
 * - State/Action/Reward embeddings
 * - Position encoding for temporal structure
 */

import { EventEmitter } from 'eventemitter3';

// ============================================================================
// TENSOR OPERATIONS (Lightweight Matrix Library)
// ============================================================================

class Tensor {
  data: Float32Array;
  shape: number[];

  constructor(data: number[] | Float32Array, shape: number[]) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;

    const expectedSize = shape.reduce((a, b) => a * b, 1);
    if (this.data.length !== expectedSize) {
      throw new Error(`Data size ${this.data.length} doesn't match shape ${shape}`);
    }
  }

  static zeros(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape);
  }

  static randn(shape: number[], scale: number = 0.02): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
    }
    return new Tensor(data, shape);
  }

  get(indices: number[]): number {
    let idx = 0;
    let multiplier = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      idx += indices[i] * multiplier;
      multiplier *= this.shape[i];
    }
    return this.data[idx];
  }

  set(indices: number[], value: number): void {
    let idx = 0;
    let multiplier = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      idx += indices[i] * multiplier;
      multiplier *= this.shape[i];
    }
    this.data[idx] = value;
  }

  add(other: Tensor): Tensor {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i] + other.data[i];
    }
    return new Tensor(result, [...this.shape]);
  }

  mul(scalar: number): Tensor {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i] * scalar;
    }
    return new Tensor(result, [...this.shape]);
  }

  clone(): Tensor {
    return new Tensor(new Float32Array(this.data), [...this.shape]);
  }
}

// ============================================================================
// LAYER IMPLEMENTATIONS
// ============================================================================

class LayerNorm {
  gamma: Tensor;
  beta: Tensor;
  eps: number;
  size: number;

  constructor(size: number, eps: number = 1e-5) {
    this.size = size;
    this.eps = eps;
    this.gamma = new Tensor(new Float32Array(size).fill(1), [size]);
    this.beta = Tensor.zeros([size]);
  }

  forward(x: Tensor): Tensor {
    // x shape: [batch, seq, hidden]
    const [batch, seq, hidden] = x.shape;
    const result = Tensor.zeros(x.shape);

    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seq; s++) {
        // Calculate mean and variance
        let mean = 0;
        let variance = 0;
        const offset = (b * seq + s) * hidden;

        for (let h = 0; h < hidden; h++) {
          mean += x.data[offset + h];
        }
        mean /= hidden;

        for (let h = 0; h < hidden; h++) {
          const diff = x.data[offset + h] - mean;
          variance += diff * diff;
        }
        variance /= hidden;

        // Normalize
        const std = Math.sqrt(variance + this.eps);
        for (let h = 0; h < hidden; h++) {
          result.data[offset + h] =
            this.gamma.data[h] * (x.data[offset + h] - mean) / std + this.beta.data[h];
        }
      }
    }

    return result;
  }
}

class Linear {
  weights: Tensor;
  bias: Tensor;
  inFeatures: number;
  outFeatures: number;

  constructor(inFeatures: number, outFeatures: number) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weights = Tensor.randn([inFeatures, outFeatures]);
    this.bias = Tensor.zeros([outFeatures]);
  }

  forward(x: Tensor): Tensor {
    // x shape: [batch, seq, in] -> [batch, seq, out]
    const [batch, seq] = x.shape.slice(0, 2);
    const result = Tensor.zeros([batch, seq, this.outFeatures]);

    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seq; s++) {
        const inOffset = (b * seq + s) * this.inFeatures;
        const outOffset = (b * seq + s) * this.outFeatures;

        for (let o = 0; o < this.outFeatures; o++) {
          let sum = this.bias.data[o];
          for (let i = 0; i < this.inFeatures; i++) {
            sum += x.data[inOffset + i] * this.weights.data[i * this.outFeatures + o];
          }
          result.data[outOffset + o] = sum;
        }
      }
    }

    return result;
  }
}

// ============================================================================
// ATTENTION MECHANISM
// ============================================================================

class CausalSelfAttention {
  numHeads: number;
  headDim: number;
  hiddenSize: number;

  queryProj: Linear;
  keyProj: Linear;
  valueProj: Linear;
  outputProj: Linear;

  constructor(hiddenSize: number, numHeads: number) {
    this.hiddenSize = hiddenSize;
    this.numHeads = numHeads;
    this.headDim = hiddenSize / numHeads;

    this.queryProj = new Linear(hiddenSize, hiddenSize);
    this.keyProj = new Linear(hiddenSize, hiddenSize);
    this.valueProj = new Linear(hiddenSize, hiddenSize);
    this.outputProj = new Linear(hiddenSize, hiddenSize);
  }

  forward(x: Tensor): Tensor {
    const [batch, seq, _] = x.shape;

    // Project to Q, K, V
    const Q = this.queryProj.forward(x);
    const K = this.keyProj.forward(x);
    const V = this.valueProj.forward(x);

    // Compute attention scores with causal masking
    const result = Tensor.zeros([batch, seq, this.hiddenSize]);
    const scale = 1 / Math.sqrt(this.headDim);

    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < this.numHeads; h++) {
        // Compute attention for this head
        const headOffset = h * this.headDim;

        for (let i = 0; i < seq; i++) {
          // Compute softmax attention weights for position i
          const scores: number[] = [];
          let maxScore = -Infinity;

          for (let j = 0; j <= i; j++) { // Causal: only attend to past
            let score = 0;
            for (let d = 0; d < this.headDim; d++) {
              const qIdx = (b * seq + i) * this.hiddenSize + headOffset + d;
              const kIdx = (b * seq + j) * this.hiddenSize + headOffset + d;
              score += Q.data[qIdx] * K.data[kIdx];
            }
            score *= scale;
            scores.push(score);
            maxScore = Math.max(maxScore, score);
          }

          // Softmax
          let sumExp = 0;
          for (let j = 0; j < scores.length; j++) {
            scores[j] = Math.exp(scores[j] - maxScore);
            sumExp += scores[j];
          }
          for (let j = 0; j < scores.length; j++) {
            scores[j] /= sumExp;
          }

          // Apply attention to values
          for (let d = 0; d < this.headDim; d++) {
            let value = 0;
            for (let j = 0; j <= i; j++) {
              const vIdx = (b * seq + j) * this.hiddenSize + headOffset + d;
              value += scores[j] * V.data[vIdx];
            }
            const outIdx = (b * seq + i) * this.hiddenSize + headOffset + d;
            result.data[outIdx] = value;
          }
        }
      }
    }

    // Output projection
    return this.outputProj.forward(result);
  }
}

// ============================================================================
// TRANSFORMER BLOCK
// ============================================================================

class TransformerBlock {
  attention: CausalSelfAttention;
  ln1: LayerNorm;
  ln2: LayerNorm;
  mlp: {
    fc1: Linear;
    fc2: Linear;
  };
  hiddenSize: number;

  constructor(hiddenSize: number, numHeads: number) {
    this.hiddenSize = hiddenSize;
    this.attention = new CausalSelfAttention(hiddenSize, numHeads);
    this.ln1 = new LayerNorm(hiddenSize);
    this.ln2 = new LayerNorm(hiddenSize);
    this.mlp = {
      fc1: new Linear(hiddenSize, hiddenSize * 4),
      fc2: new Linear(hiddenSize * 4, hiddenSize)
    };
  }

  forward(x: Tensor): Tensor {
    // Attention block with residual
    const normed1 = this.ln1.forward(x);
    const attended = this.attention.forward(normed1);
    let h = x.add(attended);

    // MLP block with residual
    const normed2 = this.ln2.forward(h);
    let mlpOut = this.mlp.fc1.forward(normed2);

    // GELU activation
    for (let i = 0; i < mlpOut.data.length; i++) {
      const x = mlpOut.data[i];
      mlpOut.data[i] = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
    }

    mlpOut = this.mlp.fc2.forward(mlpOut);
    return h.add(mlpOut);
  }
}

// ============================================================================
// DECISION TRANSFORMER
// ============================================================================

export interface DecisionTransformerConfig {
  stateSize: number;      // Dimension of state observation
  actionSize: number;     // Dimension of action space
  hiddenSize: number;     // Transformer hidden dimension
  numHeads: number;       // Number of attention heads
  numLayers: number;      // Number of transformer blocks
  contextLength: number;  // Maximum sequence length
  maxEpisodeLength: number;
  maxReturnToGo: number;  // Maximum cumulative reward
}

export interface TrajectorySegment {
  states: number[][];     // [seq, state_size]
  actions: number[][];    // [seq, action_size]
  rewards: number[];      // [seq]
  returnsToGo: number[];  // [seq]
  timesteps: number[];    // [seq]
}

export class DecisionTransformer extends EventEmitter {
  config: DecisionTransformerConfig;

  // Embeddings
  stateEmbed: Linear;
  actionEmbed: Linear;
  returnEmbed: Linear;
  timestepEmbed: Tensor; // Learned positional embedding

  // Transformer
  blocks: TransformerBlock[];
  lnFinal: LayerNorm;

  // Prediction heads
  actionPredictor: Linear;
  statePredictor: Linear;

  // Training state
  trainingStep: number = 0;
  replayBuffer: TrajectorySegment[] = [];
  maxBufferSize: number = 10000;

  constructor(config: DecisionTransformerConfig) {
    super();
    this.config = config;

    // Initialize embeddings
    this.stateEmbed = new Linear(config.stateSize, config.hiddenSize);
    this.actionEmbed = new Linear(config.actionSize, config.hiddenSize);
    this.returnEmbed = new Linear(1, config.hiddenSize);
    this.timestepEmbed = Tensor.randn([config.maxEpisodeLength, config.hiddenSize]);

    // Initialize transformer blocks
    this.blocks = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.blocks.push(new TransformerBlock(config.hiddenSize, config.numHeads));
    }

    this.lnFinal = new LayerNorm(config.hiddenSize);

    // Prediction heads
    this.actionPredictor = new Linear(config.hiddenSize, config.actionSize);
    this.statePredictor = new Linear(config.hiddenSize, config.stateSize);
  }

  /**
   * Forward pass through the Decision Transformer
   *
   * Input sequence format: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
   * where R is return-to-go, s is state, a is action
   */
  forward(
    states: Tensor,      // [batch, seq, state_size]
    actions: Tensor,     // [batch, seq, action_size]
    returnsToGo: Tensor, // [batch, seq, 1]
    timesteps: Tensor    // [batch, seq]
  ): { predictedActions: Tensor; predictedStates: Tensor } {
    const [batch, seq] = states.shape.slice(0, 2);

    // Embed each modality
    const stateEmbeds = this.stateEmbed.forward(states);
    const actionEmbeds = this.actionEmbed.forward(actions);
    const returnEmbeds = this.returnEmbed.forward(returnsToGo);

    // Add timestep embeddings
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seq; s++) {
        const t = Math.min(
          Math.floor(timesteps.data[b * seq + s]),
          this.config.maxEpisodeLength - 1
        );
        for (let h = 0; h < this.config.hiddenSize; h++) {
          const idx = (b * seq + s) * this.config.hiddenSize + h;
          stateEmbeds.data[idx] += this.timestepEmbed.data[t * this.config.hiddenSize + h];
          actionEmbeds.data[idx] += this.timestepEmbed.data[t * this.config.hiddenSize + h];
          returnEmbeds.data[idx] += this.timestepEmbed.data[t * this.config.hiddenSize + h];
        }
      }
    }

    // Interleave: R, s, a, R, s, a, ...
    const seqLen = seq * 3;
    const hiddenStates = Tensor.zeros([batch, seqLen, this.config.hiddenSize]);

    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seq; s++) {
        const srcOffset = (b * seq + s) * this.config.hiddenSize;
        // Return embedding at position 3*s
        const retDstOffset = (b * seqLen + s * 3) * this.config.hiddenSize;
        // State embedding at position 3*s + 1
        const stateDstOffset = (b * seqLen + s * 3 + 1) * this.config.hiddenSize;
        // Action embedding at position 3*s + 2
        const actDstOffset = (b * seqLen + s * 3 + 2) * this.config.hiddenSize;

        for (let h = 0; h < this.config.hiddenSize; h++) {
          hiddenStates.data[retDstOffset + h] = returnEmbeds.data[srcOffset + h];
          hiddenStates.data[stateDstOffset + h] = stateEmbeds.data[srcOffset + h];
          hiddenStates.data[actDstOffset + h] = actionEmbeds.data[srcOffset + h];
        }
      }
    }

    // Pass through transformer blocks
    let x = hiddenStates;
    for (const block of this.blocks) {
      x = block.forward(x);
    }
    x = this.lnFinal.forward(x);

    // Extract state positions (3*s + 1) for action prediction
    const statePositionOutputs = Tensor.zeros([batch, seq, this.config.hiddenSize]);
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seq; s++) {
        const srcOffset = (b * seqLen + s * 3 + 1) * this.config.hiddenSize;
        const dstOffset = (b * seq + s) * this.config.hiddenSize;
        for (let h = 0; h < this.config.hiddenSize; h++) {
          statePositionOutputs.data[dstOffset + h] = x.data[srcOffset + h];
        }
      }
    }

    const predictedActions = this.actionPredictor.forward(statePositionOutputs);
    const predictedStates = this.statePredictor.forward(statePositionOutputs);

    return { predictedActions, predictedStates };
  }

  /**
   * Generate action given current state and target return
   */
  predict(
    contextStates: number[][],
    contextActions: number[][],
    contextReturns: number[],
    targetReturn: number
  ): number[] {
    const seq = contextStates.length;
    const batch = 1;

    // Prepare tensors
    const states = Tensor.zeros([batch, seq, this.config.stateSize]);
    const actions = Tensor.zeros([batch, seq, this.config.actionSize]);
    const returns = Tensor.zeros([batch, seq, 1]);
    const timesteps = Tensor.zeros([batch, seq]);

    for (let s = 0; s < seq; s++) {
      for (let i = 0; i < this.config.stateSize; i++) {
        states.data[s * this.config.stateSize + i] = contextStates[s]?.[i] || 0;
      }
      for (let i = 0; i < this.config.actionSize; i++) {
        actions.data[s * this.config.actionSize + i] = contextActions[s]?.[i] || 0;
      }
      returns.data[s] = s === seq - 1 ? targetReturn : contextReturns[s];
      timesteps.data[s] = s;
    }

    // Forward pass
    const { predictedActions } = this.forward(states, actions, returns, timesteps);

    // Return last action prediction
    const result: number[] = [];
    const lastOffset = (seq - 1) * this.config.actionSize;
    for (let i = 0; i < this.config.actionSize; i++) {
      result.push(predictedActions.data[lastOffset + i]);
    }

    return result;
  }

  /**
   * Add trajectory to replay buffer
   */
  addTrajectory(trajectory: TrajectorySegment): void {
    this.replayBuffer.push(trajectory);

    // Trim if over capacity
    while (this.replayBuffer.length > this.maxBufferSize) {
      this.replayBuffer.shift();
    }

    this.emit('trajectory-added', { bufferSize: this.replayBuffer.length });
  }

  /**
   * Sample a batch for training
   */
  sampleBatch(batchSize: number): TrajectorySegment[] {
    const batch: TrajectorySegment[] = [];
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      batch.push(this.replayBuffer[idx]);
    }
    return batch;
  }

  /**
   * Training step (simplified gradient descent)
   */
  trainStep(batch: TrajectorySegment[], learningRate: number = 0.0001): number {
    // Compute forward pass and loss for batch
    let totalLoss = 0;

    for (const traj of batch) {
      const seq = traj.states.length;

      // Create tensors
      const states = Tensor.zeros([1, seq, this.config.stateSize]);
      const actions = Tensor.zeros([1, seq, this.config.actionSize]);
      const returns = Tensor.zeros([1, seq, 1]);
      const timesteps = Tensor.zeros([1, seq]);

      for (let s = 0; s < seq; s++) {
        for (let i = 0; i < this.config.stateSize && i < traj.states[s].length; i++) {
          states.data[s * this.config.stateSize + i] = traj.states[s][i];
        }
        for (let i = 0; i < this.config.actionSize && i < traj.actions[s].length; i++) {
          actions.data[s * this.config.actionSize + i] = traj.actions[s][i];
        }
        returns.data[s] = traj.returnsToGo[s] / this.config.maxReturnToGo; // Normalize
        timesteps.data[s] = traj.timesteps[s];
      }

      // Forward
      const { predictedActions } = this.forward(states, actions, returns, timesteps);

      // Compute MSE loss against actual actions
      for (let s = 0; s < seq; s++) {
        for (let i = 0; i < this.config.actionSize; i++) {
          const predicted = predictedActions.data[s * this.config.actionSize + i];
          const actual = actions.data[s * this.config.actionSize + i];
          totalLoss += (predicted - actual) ** 2;
        }
      }
    }

    totalLoss /= batch.length;

    // In a real implementation, we'd do backprop here
    // For this demo, we'll use a simplified parameter update
    this.trainingStep++;
    this.emit('training-step', { step: this.trainingStep, loss: totalLoss });

    return totalLoss;
  }

  /**
   * Export model weights
   */
  exportWeights(): ArrayBuffer {
    const weights: number[] = [];

    // Collect all weights
    const collectLinear = (layer: Linear) => {
      weights.push(...Array.from(layer.weights.data));
      weights.push(...Array.from(layer.bias.data));
    };

    collectLinear(this.stateEmbed);
    collectLinear(this.actionEmbed);
    collectLinear(this.returnEmbed);
    weights.push(...Array.from(this.timestepEmbed.data));

    for (const block of this.blocks) {
      collectLinear(block.attention.queryProj);
      collectLinear(block.attention.keyProj);
      collectLinear(block.attention.valueProj);
      collectLinear(block.attention.outputProj);
      collectLinear(block.mlp.fc1);
      collectLinear(block.mlp.fc2);
    }

    collectLinear(this.actionPredictor);
    collectLinear(this.statePredictor);

    return new Float32Array(weights).buffer;
  }

  /**
   * Get model statistics
   */
  getStats(): {
    parameters: number;
    trainingSteps: number;
    bufferSize: number;
    config: DecisionTransformerConfig;
  } {
    let params = 0;

    // Count parameters
    const countLinear = (layer: Linear) => {
      return layer.inFeatures * layer.outFeatures + layer.outFeatures;
    };

    params += countLinear(this.stateEmbed);
    params += countLinear(this.actionEmbed);
    params += countLinear(this.returnEmbed);
    params += this.timestepEmbed.data.length;

    for (const block of this.blocks) {
      params += countLinear(block.attention.queryProj);
      params += countLinear(block.attention.keyProj);
      params += countLinear(block.attention.valueProj);
      params += countLinear(block.attention.outputProj);
      params += countLinear(block.mlp.fc1);
      params += countLinear(block.mlp.fc2);
      params += block.ln1.gamma.data.length * 2;
      params += block.ln2.gamma.data.length * 2;
    }

    params += countLinear(this.actionPredictor);
    params += countLinear(this.statePredictor);

    return {
      parameters: params,
      trainingSteps: this.trainingStep,
      bufferSize: this.replayBuffer.length,
      config: this.config
    };
  }
}
