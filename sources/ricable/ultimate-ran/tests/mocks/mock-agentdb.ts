/**
 * Mock AgentDB Client - London School TDD
 *
 * Mocks agentdb@alpha for testing Reflexion memory and episode storage
 */

import { vi } from 'vitest';
import type { Mock } from 'vitest';
import { createMockLearningEpisode, generateEmbedding } from '../setup';

export interface MockAgentDBConfig {
  episodes?: any[];
  searchLatency?: number;
  failureRate?: number;
}

export class MockAgentDB {
  private episodes: Map<string, any>;
  private reflexions: Map<string, any>;
  private searchLatency: number;
  private failureRate: number;

  // Mock methods
  public storeEpisode: Mock;
  public queryEpisodes: Mock;
  public getReflexion: Mock;
  public storeReflexion: Mock;
  public searchSimilar: Mock;
  public getFailedProposals: Mock;

  constructor(config: MockAgentDBConfig = {}) {
    this.episodes = new Map();
    this.reflexions = new Map();
    this.searchLatency = config.searchLatency ?? 5;
    this.failureRate = config.failureRate ?? 0;

    // Initialize with mock episodes
    if (config.episodes) {
      config.episodes.forEach(ep => this.episodes.set(ep.id, ep));
    }

    // Create mock methods
    this.storeEpisode = vi.fn(async (episode) => {
      if (Math.random() < this.failureRate) {
        throw new Error('AgentDB storage failed');
      }
      this.episodes.set(episode.id, episode);
      return { success: true, id: episode.id };
    });

    this.queryEpisodes = vi.fn(async (filter) => {
      await this._simulateLatency();

      let results = Array.from(this.episodes.values());

      if (filter.cellId) {
        results = results.filter(ep => ep.cellId === filter.cellId);
      }
      if (filter.outcome) {
        results = results.filter(ep => ep.outcome === filter.outcome);
      }
      if (filter.minReward !== undefined) {
        results = results.filter(ep => ep.reward >= filter.minReward);
      }

      return results.slice(0, filter.limit ?? 10);
    });

    this.getReflexion = vi.fn(async (episodeId) => {
      await this._simulateLatency();
      return this.reflexions.get(episodeId) ?? null;
    });

    this.storeReflexion = vi.fn(async (episodeId, reflexion) => {
      this.reflexions.set(episodeId, {
        episodeId,
        timestamp: Date.now(),
        critique: reflexion.critique,
        learnings: reflexion.learnings,
        improvements: reflexion.improvements,
      });
      return { success: true };
    });

    this.searchSimilar = vi.fn(async (embedding, options = {}) => {
      await this._simulateLatency();

      const k = options.k ?? 5;
      const threshold = options.threshold ?? 0.7;

      // Simple mock similarity (in reality would use vector search)
      const results = Array.from(this.episodes.values())
        .map(ep => ({
          episode: ep,
          similarity: Math.random(), // Mock similarity score
        }))
        .filter(r => r.similarity >= threshold)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, k);

      return results;
    });

    this.getFailedProposals = vi.fn(async (options = {}) => {
      await this._simulateLatency();

      const failures = Array.from(this.episodes.values())
        .filter(ep => ep.outcome === 'FAILURE')
        .slice(0, options.limit ?? 20);

      return failures.map(ep => ({
        id: ep.id,
        cellId: ep.cellId,
        parameters: ep.cmChange,
        reason: ep.failureReason ?? 'Unknown',
        embedding: ep.embedding,
      }));
    });
  }

  private async _simulateLatency(): Promise<void> {
    if (this.searchLatency > 0) {
      await new Promise(resolve => setTimeout(resolve, this.searchLatency));
    }
  }

  /**
   * Test helper: Add mock episode
   */
  addEpisode(episode: any): void {
    this.episodes.set(episode.id, episode);
  }

  /**
   * Test helper: Clear all episodes
   */
  clear(): void {
    this.episodes.clear();
    this.reflexions.clear();
  }

  /**
   * Test helper: Get call count for method
   */
  getCallCount(method: keyof MockAgentDB): number {
    const mockMethod = this[method];
    if (typeof mockMethod === 'function' && 'mock' in mockMethod) {
      return (mockMethod as Mock).mock.calls.length;
    }
    return 0;
  }
}

/**
 * Factory function for creating mock AgentDB
 */
export function createMockAgentDB(config?: MockAgentDBConfig): MockAgentDB {
  return new MockAgentDB(config);
}

/**
 * Default mock for tests
 */
export const mockAgentDB = createMockAgentDB({
  episodes: [
    createMockLearningEpisode({ id: 'ep-1', outcome: 'SUCCESS', reward: 0.9 }),
    createMockLearningEpisode({ id: 'ep-2', outcome: 'SUCCESS', reward: 0.7 }),
    createMockLearningEpisode({ id: 'ep-3', outcome: 'FAILURE', reward: -0.5 }),
  ],
  searchLatency: 5,
});
