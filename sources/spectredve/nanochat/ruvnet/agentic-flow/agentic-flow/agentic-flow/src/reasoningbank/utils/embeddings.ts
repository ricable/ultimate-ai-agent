import crypto from 'crypto';

export interface EmbeddingProvider {
  generate(text: string): Promise<number[]>;
  dimensions: number;
}

class HashEmbedding implements EmbeddingProvider {
  dimensions = 384;

  async generate(text: string): Promise<number[]> {
    const hash = crypto.createHash('md5').update(text).digest();
    const embedding = new Array(this.dimensions).fill(0);

    for (let i = 0; i < this.dimensions; i++) {
      const byteIndex = i % hash.length;
      embedding[i] = (hash[byteIndex] / 255) * 2 - 1;
    }

    return embedding;
  }
}

export function createEmbeddingProvider(
  provider: 'openai' | 'anthropic' | 'hash' = 'hash',
  options?: { apiKey?: string; model?: string }
): EmbeddingProvider {
  return new HashEmbedding();
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same dimensions');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}
