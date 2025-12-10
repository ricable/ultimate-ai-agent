/**
 * Data Loading Utilities
 *
 * Utilities for loading and preprocessing training data
 */

export interface DataLoaderConfig {
  dataPath: string;
  batchSize: number;
  shuffle: boolean;
  numWorkers: number;
  prefetchFactor: number;
}

export interface Dataset {
  samples: any[];
  labels?: any[];
  metadata?: Record<string, any>;
}

export class DataLoader {
  private config: DataLoaderConfig;
  private dataset: Dataset | null = null;
  private currentIndex: number = 0;

  constructor(config: DataLoaderConfig) {
    this.config = config;
  }

  /**
   * Load dataset from disk
   */
  async loadDataset(): Promise<void> {
    console.log('📚 Loading dataset...');
    console.log('  Path:', this.config.dataPath);
    console.log('  Batch size:', this.config.batchSize);
    console.log('  Shuffle:', this.config.shuffle ? 'Yes' : 'No');
    console.log();

    // In real implementation, load actual data
    // For demo, create synthetic dataset
    this.dataset = this.createSyntheticDataset(10000);

    console.log('✅ Dataset loaded');
    console.log('  Total samples:', this.dataset.samples.length.toLocaleString());
    if (this.dataset.labels) {
      console.log('  Labels:', this.dataset.labels.length.toLocaleString());
    }
    console.log();
  }

  /**
   * Create synthetic dataset for demonstration
   */
  private createSyntheticDataset(numSamples: number): Dataset {
    const samples = Array.from({ length: numSamples }, (_, i) => ({
      id: i,
      features: Array.from({ length: 128 }, () => Math.random()),
    }));

    const labels = Array.from({ length: numSamples }, (_, i) => i % 10);

    return {
      samples,
      labels,
      metadata: {
        numSamples,
        numFeatures: 128,
        numClasses: 10,
      },
    };
  }

  /**
   * Get next batch of data
   */
  getNextBatch(): { samples: any[]; labels?: any[]; hasMore: boolean } {
    if (!this.dataset) {
      throw new Error('Dataset not loaded. Call loadDataset() first.');
    }

    const { samples, labels } = this.dataset;
    const { batchSize } = this.config;

    const startIdx = this.currentIndex;
    const endIdx = Math.min(startIdx + batchSize, samples.length);

    const batchSamples = samples.slice(startIdx, endIdx);
    const batchLabels = labels ? labels.slice(startIdx, endIdx) : undefined;

    this.currentIndex = endIdx;
    const hasMore = this.currentIndex < samples.length;

    if (!hasMore && this.config.shuffle) {
      this.shuffleDataset();
      this.currentIndex = 0;
    }

    return {
      samples: batchSamples,
      labels: batchLabels,
      hasMore,
    };
  }

  /**
   * Shuffle dataset
   */
  private shuffleDataset(): void {
    if (!this.dataset) return;

    const { samples, labels } = this.dataset;

    // Fisher-Yates shuffle
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
      if (labels) {
        [labels[i], labels[j]] = [labels[j], labels[i]];
      }
    }
  }

  /**
   * Reset data loader to beginning
   */
  reset(): void {
    this.currentIndex = 0;
    if (this.config.shuffle) {
      this.shuffleDataset();
    }
  }

  /**
   * Get dataset statistics
   */
  getStatistics(): Record<string, any> {
    if (!this.dataset) {
      throw new Error('Dataset not loaded');
    }

    const { samples, labels, metadata } = this.dataset;

    const stats: Record<string, any> = {
      totalSamples: samples.length,
      ...metadata,
    };

    if (labels) {
      // Calculate class distribution
      const classCounts: Record<number, number> = {};
      labels.forEach((label) => {
        classCounts[label] = (classCounts[label] || 0) + 1;
      });

      stats.classDistribution = classCounts;
      stats.isBalanced = this.checkBalance(classCounts);
    }

    return stats;
  }

  /**
   * Check if dataset is balanced
   */
  private checkBalance(classCounts: Record<number, number>): boolean {
    const counts = Object.values(classCounts);
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
    const maxDeviation = Math.max(...counts.map((c) => Math.abs(c - mean)));
    return maxDeviation / mean < 0.1; // Within 10% of mean
  }
}

/**
 * Data augmentation utilities
 */
export class DataAugmentation {
  /**
   * Apply random augmentation to image data
   */
  static augmentImage(image: any): any {
    // In real implementation, apply actual augmentation:
    // - Random crop
    // - Random flip
    // - Color jitter
    // - Rotation
    // - Normalization

    return image; // Placeholder
  }

  /**
   * Apply random augmentation to text data
   */
  static augmentText(text: string): string {
    // In real implementation, apply actual augmentation:
    // - Synonym replacement
    // - Random insertion
    // - Random deletion
    // - Random swap

    return text; // Placeholder
  }
}

/**
 * Data preprocessing utilities
 */
export class DataPreprocessor {
  /**
   * Normalize features to [0, 1] range
   */
  static normalize(features: number[]): number[] {
    const min = Math.min(...features);
    const max = Math.max(...features);
    const range = max - min;

    return features.map((f) => (f - min) / range);
  }

  /**
   * Standardize features to zero mean and unit variance
   */
  static standardize(features: number[]): number[] {
    const mean = features.reduce((a, b) => a + b, 0) / features.length;
    const variance =
      features.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / features.length;
    const std = Math.sqrt(variance);

    return features.map((f) => (f - mean) / std);
  }

  /**
   * Tokenize text for NLP models
   */
  static tokenize(text: string, vocabularySize: number): number[] {
    // In real implementation, use actual tokenizer (BPE, WordPiece, etc.)
    // For demo, create simple token IDs
    const words = text.toLowerCase().split(/\s+/);
    return words.map((word) => {
      // Simple hash to token ID
      let hash = 0;
      for (let i = 0; i < word.length; i++) {
        hash = (hash << 5) - hash + word.charCodeAt(i);
        hash = hash & hash;
      }
      return Math.abs(hash) % vocabularySize;
    });
  }
}

// Example usage
async function main() {
  console.log('📊 Data Loading Example\n');

  const config: DataLoaderConfig = {
    dataPath: './data/training',
    batchSize: 32,
    shuffle: true,
    numWorkers: 4,
    prefetchFactor: 2,
  };

  const loader = new DataLoader(config);

  // Load dataset
  await loader.loadDataset();

  // Get statistics
  const stats = loader.getStatistics();
  console.log('Dataset Statistics:');
  console.log('  Total samples:', stats.totalSamples.toLocaleString());
  console.log('  Features:', stats.numFeatures);
  console.log('  Classes:', stats.numClasses);
  console.log('  Balanced:', stats.isBalanced ? 'Yes' : 'No');
  console.log();

  // Process a few batches
  console.log('Processing batches:');
  for (let i = 0; i < 3; i++) {
    const batch = loader.getNextBatch();
    console.log(`  Batch ${i + 1}:`);
    console.log(`    Samples:`, batch.samples.length);
    console.log(`    Labels:`, batch.labels?.length || 'N/A');
    console.log(`    Has more:`, batch.hasMore ? 'Yes' : 'No');
  }
  console.log();

  console.log('🎉 Data loading example completed!');
}

// Run if executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error('❌ Error:', error);
    process.exit(1);
  });
}
