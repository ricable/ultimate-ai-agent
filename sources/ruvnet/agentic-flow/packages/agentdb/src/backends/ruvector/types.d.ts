// Type declarations for optional @ruvector dependencies
// These allow TypeScript compilation without installing the packages

declare module '@ruvector/core' {
  export class VectorDB {
    constructor(dimension: number, config?: {
      metric?: 'cosine' | 'l2' | 'ip';
      maxElements?: number;
      efConstruction?: number;
      M?: number;
    });
    insert(id: string, embedding: number[]): void;
    search(query: number[], k: number): Array<{ id: string; distance: number }>;
    remove(id: string): boolean;
    count(): number;
    setEfSearch(ef: number): void;
    save(path: string): void;
    load(path: string): void;
    memoryUsage?(): number;
  }

  export function isNative(): boolean;
}

declare module '@ruvector/gnn' {
  export class GNNLayer {
    constructor(inputDim: number, outputDim: number, heads: number);
    forward(
      query: number[],
      neighbors: number[][],
      weights: number[]
    ): number[];
    train(
      samples: Array<{ embedding: number[]; label: number }>,
      options: {
        epochs: number;
        learningRate: number;
        batchSize: number;
      }
    ): Promise<{ epochs: number; finalLoss: number }>;
    save(path: string): void;
    load(path: string): void;
  }
}

declare module '@ruvector/graph-node' {
  export interface GraphNode {
    id: string;
    labels: string[];
    properties: Record<string, any>;
  }

  export interface QueryResult {
    nodes: GraphNode[];
    relationships: any[];
  }

  export class GraphDB {
    execute(cypher: string, params?: Record<string, any>): Promise<QueryResult>;
    createNode(labels: string[], properties: Record<string, any>): Promise<string>;
    getNode(id: string): Promise<GraphNode | null>;
    deleteNode(id: string): Promise<boolean>;
  }
}
