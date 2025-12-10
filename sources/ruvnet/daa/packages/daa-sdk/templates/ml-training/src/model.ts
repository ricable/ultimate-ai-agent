/**
 * Model Definitions
 *
 * Define machine learning model architectures for federated learning
 */

export interface ModelConfig {
  name: string;
  architecture: string;
  parameters: ModelParameters;
  training: TrainingConfig;
}

export interface ModelParameters {
  layers: number;
  hiddenSize: number;
  attentionHeads?: number;
  vocabularySize?: number;
  maxSequenceLength?: number;
  dropoutRate?: number;
}

export interface TrainingConfig {
  batchSize: number;
  learningRate: number;
  epochs: number;
  optimizer: 'adam' | 'sgd' | 'adamw';
  scheduler?: 'cosine' | 'linear' | 'exponential';
  warmupSteps?: number;
}

/**
 * GPT-Mini: Small transformer model for text generation
 */
export const GPT_MINI: ModelConfig = {
  name: 'gpt-mini',
  architecture: 'transformer',
  parameters: {
    layers: 6,
    hiddenSize: 512,
    attentionHeads: 8,
    vocabularySize: 50000,
    maxSequenceLength: 1024,
    dropoutRate: 0.1,
  },
  training: {
    batchSize: 32,
    learningRate: 0.001,
    epochs: 100,
    optimizer: 'adamw',
    scheduler: 'cosine',
    warmupSteps: 1000,
  },
};

/**
 * BERT-Tiny: Small BERT model for classification
 */
export const BERT_TINY: ModelConfig = {
  name: 'bert-tiny',
  architecture: 'transformer',
  parameters: {
    layers: 4,
    hiddenSize: 256,
    attentionHeads: 4,
    vocabularySize: 30000,
    maxSequenceLength: 512,
    dropoutRate: 0.1,
  },
  training: {
    batchSize: 64,
    learningRate: 0.0005,
    epochs: 50,
    optimizer: 'adam',
    scheduler: 'linear',
    warmupSteps: 500,
  },
};

/**
 * ResNet-18: Convolutional neural network for image classification
 */
export const RESNET_18: ModelConfig = {
  name: 'resnet-18',
  architecture: 'cnn',
  parameters: {
    layers: 18,
    hiddenSize: 512, // Final layer dimension
  },
  training: {
    batchSize: 128,
    learningRate: 0.1,
    epochs: 200,
    optimizer: 'sgd',
    scheduler: 'cosine',
  },
};

/**
 * Model Registry
 */
export const MODEL_REGISTRY: Record<string, ModelConfig> = {
  'gpt-mini': GPT_MINI,
  'bert-tiny': BERT_TINY,
  'resnet-18': RESNET_18,
};

/**
 * Get model configuration by name
 */
export function getModelConfig(name: string): ModelConfig {
  const config = MODEL_REGISTRY[name];
  if (!config) {
    throw new Error(`Model '${name}' not found in registry`);
  }
  return config;
}

/**
 * Calculate total model parameters
 */
export function calculateTotalParameters(config: ModelConfig): number {
  const { architecture, parameters } = config;

  if (architecture === 'transformer') {
    const {
      layers,
      hiddenSize,
      attentionHeads,
      vocabularySize,
      maxSequenceLength,
    } = parameters;

    // Simplified calculation for transformer models
    const attentionParams = layers! * (3 * hiddenSize! * hiddenSize! + hiddenSize!);
    const ffnParams = layers! * (4 * hiddenSize! * hiddenSize! + hiddenSize!);
    const embeddingParams = vocabularySize! * hiddenSize!;
    const positionParams = maxSequenceLength! * hiddenSize!;

    return attentionParams + ffnParams + embeddingParams + positionParams;
  } else if (architecture === 'cnn') {
    // Simplified calculation for ResNet-18
    // Actual: ~11.7M parameters
    return 11700000;
  }

  return 0;
}

/**
 * Get model memory requirements (in GB)
 */
export function getMemoryRequirements(config: ModelConfig): {
  training: number;
  inference: number;
} {
  const totalParams = calculateTotalParameters(config);
  const { batchSize } = config.training;

  // Rough estimates:
  // - 4 bytes per parameter (fp32)
  // - Training requires ~4x parameter memory (weights + gradients + optimizer states)
  // - Inference requires ~2x parameter memory (weights + activations)

  const paramMemory = (totalParams * 4) / (1024 ** 3); // Convert to GB

  return {
    training: paramMemory * 4 * (batchSize / 32), // Scale with batch size
    inference: paramMemory * 2,
  };
}

/**
 * Display model information
 */
export function displayModelInfo(config: ModelConfig): void {
  console.log('Model Configuration:');
  console.log('  Name:', config.name);
  console.log('  Architecture:', config.architecture);
  console.log();

  console.log('Model Parameters:');
  Object.entries(config.parameters).forEach(([key, value]) => {
    if (value !== undefined) {
      console.log(`  ${key}:`, value);
    }
  });
  console.log();

  const totalParams = calculateTotalParameters(config);
  console.log('Total Parameters:', (totalParams / 1000000).toFixed(2), 'M');
  console.log();

  console.log('Training Configuration:');
  console.log('  Batch size:', config.training.batchSize);
  console.log('  Learning rate:', config.training.learningRate);
  console.log('  Epochs:', config.training.epochs);
  console.log('  Optimizer:', config.training.optimizer);
  if (config.training.scheduler) {
    console.log('  Scheduler:', config.training.scheduler);
  }
  if (config.training.warmupSteps) {
    console.log('  Warmup steps:', config.training.warmupSteps);
  }
  console.log();

  const memory = getMemoryRequirements(config);
  console.log('Memory Requirements:');
  console.log('  Training:', memory.training.toFixed(2), 'GB');
  console.log('  Inference:', memory.inference.toFixed(2), 'GB');
  console.log();
}

// Example usage
if (require.main === module) {
  console.log('🧠 Model Definitions\n');
  console.log('='.repeat(50));
  console.log();

  // Display all models
  Object.values(MODEL_REGISTRY).forEach((config) => {
    displayModelInfo(config);
    console.log('='.repeat(50));
    console.log();
  });
}
