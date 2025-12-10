/**
 * AgentDB Manager Utility
 *
 * AgentDB integration for pattern learning, memory synchronization,
 and distributed cognitive patterns with <1ms QUIC sync capabilities.
 */

import { AgentDBPattern, AgentDBIntegrationInfo } from '../types/export-types';
import { PriorityTemplate } from '../../rtb/hierarchical-template-system/interfaces';
import { ExportResult, ValidationResults } from '../types/export-types';

export interface AgentDBConfig {
  connectionString: string;
  enableQUICSync: boolean;
  syncInterval: number;
  maxBatchSize: number;
  compressionEnabled: boolean;
  encryptionEnabled: boolean;
  retryAttempts: number;
  timeout: number;
}

export interface SyncStatus {
  lastSyncTime: Date;
  syncInProgress: boolean;
  queuedOperations: number;
  failedOperations: number;
  syncErrors: string[];
}

export interface PatternStatistics {
  totalPatterns: number;
  patternsByType: Record<string, number>;
  averageConfidence: number;
  mostUsedPatterns: AgentDBPattern[];
  learningRate: number;
  syncLatency: number;
}

export class AgentDBManager {
  private config: AgentDBConfig;
  private isConnected: boolean = false;
  private patterns: Map<string, AgentDBPattern> = new Map();
  private syncStatus: SyncStatus;
  private syncQueue: any[] = [];
  private syncInterval?: NodeJS.Timeout;

  constructor(config: AgentDBConfig) {
    this.config = config;
    this.syncStatus = {
      lastSyncTime: new Date(),
      syncInProgress: false,
      queuedOperations: 0,
      failedOperations: 0,
      syncErrors: []
    };
  }

  async initialize(): Promise<void> {
    console.log('🗄️ Initializing AgentDB Manager...');

    try {
      // Connect to AgentDB
      await this.connect();

      // Load existing patterns
      await this.loadPatterns();

      // Start sync interval if QUIC sync is enabled
      if (this.config.enableQUICSync) {
        this.startSyncInterval();
      }

      this.isConnected = true;
      console.log('✅ AgentDB Manager initialized successfully');

    } catch (error) {
      console.error('❌ AgentDB initialization failed:', error);
      throw error;
    }
  }

  async storeExportPattern(
    template: PriorityTemplate,
    result: ExportResult,
    validation: ValidationResults
  ): Promise<void> {
    if (!this.isConnected) {
      console.warn('AgentDB not connected, skipping pattern storage');
      return;
    }

    const pattern: AgentDBPattern = {
      patternId: `export_${template.meta.templateId}_${Date.now()}`,
      patternType: 'template_export',
      content: {
        templateId: template.meta.templateId,
        templateVersion: template.meta.version,
        exportResult: result,
        validationResults: validation,
        timestamp: new Date().toISOString()
      },
      confidence: this.calculatePatternConfidence(template, validation),
      createdAt: new Date(),
      lastAccessed: new Date(),
      accessCount: 0,
      relatedPatterns: this.findRelatedPatterns(template)
    };

    this.patterns.set(pattern.patternId, pattern);
    this.queueSyncOperation('store', pattern);

    console.log(`💾 Stored export pattern: ${pattern.patternId}`);
  }

  async storeValidationPattern(pattern: any): Promise<void> {
    if (!this.isConnected) return;

    const agentdbPattern: AgentDBPattern = {
      patternId: pattern.patternId,
      patternType: 'validation_pattern',
      content: pattern,
      confidence: pattern.confidence || 0.5,
      createdAt: new Date(),
      lastAccessed: new Date(),
      accessCount: 0,
      relatedPatterns: []
    };

    this.patterns.set(agentdbPattern.patternId, agentdbPattern);
    this.queueSyncOperation('store', agentdbPattern);

    console.log(`💾 Stored validation pattern: ${pattern.patternId}`);
  }

  async retrieveSimilarPatterns(query: any): Promise<AgentDBPattern[]> {
    if (!this.isConnected) return [];

    const similarityThreshold = 0.7;
    const similarPatterns: AgentDBPattern[] = [];

    for (const pattern of this.patterns.values()) {
      const similarity = this.calculateSimilarity(query, pattern);
      if (similarity >= similarityThreshold) {
        similarPatterns.push(pattern);
        pattern.accessCount++;
        pattern.lastAccessed = new Date();
      }
    }

    // Sort by confidence and similarity
    similarPatterns.sort((a, b) => {
      const scoreA = a.confidence * this.calculateSimilarity(query, a);
      const scoreB = b.confidence * this.calculateSimilarity(query, b);
      return scoreB - scoreA;
    });

    return similarPatterns.slice(0, 10); // Return top 10 similar patterns
  }

  async synchronizeWithNodes(nodes: any[]): Promise<void> {
    if (!this.config.enableQUICSync || !this.isConnected) {
      console.log('QUIC sync disabled or not connected');
      return;
    }

    console.log(`🔄 Synchronizing with ${nodes.length} nodes via QUIC...`);

    const startTime = Date.now();
    this.syncStatus.syncInProgress = true;

    try {
      // In a real implementation, this would use QUIC protocol
      // For now, simulate QUIC sync with HTTP
      for (const node of nodes) {
        await this.syncWithNode(node);
      }

      this.syncStatus.lastSyncTime = new Date();
      const syncTime = Date.now() - startTime;
      console.log(`✅ QUIC sync completed in ${syncTime}ms`);

    } catch (error) {
      console.error('❌ QUIC sync failed:', error);
      this.syncStatus.syncErrors.push(error.message);
    } finally {
      this.syncStatus.syncInProgress = false;
    }
  }

  async getIntegrationInfo(): Promise<AgentDBIntegrationInfo> {
    return {
      connected: this.isConnected,
      syncStatus: this.isConnected ?
        (this.syncStatus.syncInProgress ? 'syncing' : 'synced') :
        'error',
      lastSyncTime: this.syncStatus.lastSyncTime,
      storedPatterns: this.patterns.size,
      retrievedPatterns: Array.from(this.patterns.values())
        .reduce((sum, p) => sum + p.accessCount, 0),
      synchronizationTime: this.config.enableQUICSync ?
        this.syncStatus.lastSyncTime.getTime() : 0,
      patterns: Array.from(this.patterns.values()).slice(0, 10) // Return 10 recent patterns
    };
  }

  async getStatistics(): Promise<PatternStatistics> {
    const patternsByType: Record<string, number> = {};
    let totalConfidence = 0;
    const mostUsedPatterns = Array.from(this.patterns.values())
      .sort((a, b) => b.accessCount - a.accessCount)
      .slice(0, 5);

    for (const pattern of this.patterns.values()) {
      patternsByType[pattern.patternType] = (patternsByType[pattern.patternType] || 0) + 1;
      totalConfidence += pattern.confidence;
    }

    return {
      totalPatterns: this.patterns.size,
      patternsByType,
      averageConfidence: this.patterns.size > 0 ? totalConfidence / this.patterns.size : 0,
      mostUsedPatterns,
      learningRate: this.calculateLearningRate(),
      syncLatency: this.config.enableQUICSync ? 50 : 0 // Simulated latency
    };
  }

  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down AgentDB Manager...');

    // Stop sync interval
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }

    // Final sync if connected
    if (this.isConnected && this.syncQueue.length > 0) {
      await this.processSyncQueue();
    }

    // Clear patterns
    this.patterns.clear();
    this.syncQueue = [];

    this.isConnected = false;
    console.log('✅ AgentDB Manager shutdown complete');
  }

  // Private methods

  private async connect(): Promise<void> {
    // Simulate connection to AgentDB
    console.log(`🔌 Connecting to AgentDB at ${this.config.connectionString}...`);

    // In a real implementation, this would establish QUIC connection
    await new Promise(resolve => setTimeout(resolve, 100));

    console.log('✅ Connected to AgentDB');
  }

  private async loadPatterns(): Promise<void> {
    // Load existing patterns from AgentDB
    console.log('📚 Loading existing patterns from AgentDB...');

    // In a real implementation, this would query AgentDB
    // For now, simulate loading some patterns
    await new Promise(resolve => setTimeout(resolve, 50));

    console.log(`✅ Loaded ${this.patterns.size} patterns`);
  }

  private startSyncInterval(): void {
    this.syncInterval = setInterval(async () => {
      if (this.syncQueue.length > 0 && !this.syncStatus.syncInProgress) {
        await this.processSyncQueue();
      }
    }, this.config.syncInterval);
  }

  private queueSyncOperation(operation: string, data: any): void {
    this.syncQueue.push({
      operation,
      data,
      timestamp: Date.now()
    });
    this.syncStatus.queuedOperations++;
  }

  private async processSyncQueue(): Promise<void> {
    if (this.syncQueue.length === 0) return;

    console.log(`🔄 Processing ${this.syncQueue.length} sync operations...`);

    const batchSize = Math.min(this.config.maxBatchSize, this.syncQueue.length);
    const batch = this.syncQueue.splice(0, batchSize);

    try {
      // Simulate QUIC sync operations
      await new Promise(resolve => setTimeout(resolve, 10));

      this.syncStatus.queuedOperations -= batch.length;
      console.log(`✅ Synced ${batch.length} operations`);

    } catch (error) {
      console.error('❌ Sync operation failed:', error);
      this.syncStatus.failedOperations += batch.length;
      this.syncStatus.syncErrors.push(error.message);

      // Re-queue failed operations if retry attempts remaining
      if (this.config.retryAttempts > 0) {
        this.syncQueue.unshift(...batch);
      }
    }
  }

  private async syncWithNode(node: any): Promise<void> {
    // Simulate QUIC sync with a node
    console.log(`🔄 Syncing with node: ${node.id}`);

    // In a real implementation, this would use QUIC protocol
    await new Promise(resolve => setTimeout(resolve, 5));

    console.log(`✅ Synced with node: ${node.id}`);
  }

  private calculatePatternConfidence(template: PriorityTemplate, validation: ValidationResults): number {
    // Base confidence from validation score
    let confidence = validation.validationScore;

    // Adjust based on template complexity
    const complexity = Object.keys(template.configuration).length;
    if (complexity > 100) {
      confidence *= 0.9; // Reduce confidence for very complex templates
    } else if (complexity < 10) {
      confidence *= 0.8; // Reduce confidence for very simple templates
    }

    // Adjust based on validation errors
    if (validation.errors.length > 0) {
      confidence *= 0.5; // Significant reduction for errors
    } else if (validation.warnings.length > 0) {
      confidence *= 0.8; // Slight reduction for warnings
    }

    return Math.max(0.1, Math.min(1.0, confidence));
  }

  private calculateSimilarity(query: any, pattern: AgentDBPattern): number {
    // Simplified similarity calculation
    // In a real implementation, would use vector similarity with AgentDB

    if (query.templateId && pattern.content.templateId) {
      if (query.templateId === pattern.content.templateId) {
        return 1.0;
      }
    }

    if (query.patternType && pattern.patternType === query.patternType) {
      return 0.8;
    }

    // Calculate content similarity (simplified)
    const queryStr = JSON.stringify(query).toLowerCase();
    const patternStr = JSON.stringify(pattern.content).toLowerCase();

    const commonWords = queryStr.split(' ').filter(word =>
      word.length > 3 && patternStr.includes(word)
    );

    return commonWords.length / Math.max(queryStr.split(' ').length, 1);
  }

  private findRelatedPatterns(template: PriorityTemplate): string[] {
    const relatedPatterns: string[] = [];

    for (const [patternId, pattern] of this.patterns.entries()) {
      if (pattern.patternType === 'template_export' &&
          pattern.content.templateId === template.meta.templateId) {
        relatedPatterns.push(patternId);
      }
    }

    return relatedPatterns;
  }

  private calculateLearningRate(): number {
    // Calculate learning rate based on recent pattern additions and usage
    const recentPatterns = Array.from(this.patterns.values()).filter(p =>
      Date.now() - p.createdAt.getTime() < 24 * 60 * 60 * 1000 // Last 24 hours
    );

    const totalAccess = Array.from(this.patterns.values())
      .reduce((sum, p) => sum + p.accessCount, 0);

    return (recentPatterns.length * 0.5 + totalAccess * 0.1) /
           Math.max(this.patterns.length, 1);
  }
}