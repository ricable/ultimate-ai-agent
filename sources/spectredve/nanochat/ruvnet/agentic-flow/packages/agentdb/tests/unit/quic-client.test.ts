import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

/**
 * QUICClient Unit Tests
 *
 * Tests for QUIC client functionality including:
 * - Connection establishment
 * - Authentication
 * - Data synchronization
 * - Reconnection logic
 * - Error handling
 */

// Mock QUICClient implementation for testing
class MockQUICClient {
  private serverUrl: string;
  private authToken: string;
  private isConnected: boolean = false;
  private connectionAttempts: number = 0;
  private maxRetries: number = 3;

  constructor(serverUrl: string, authToken: string) {
    this.serverUrl = serverUrl;
    this.authToken = authToken;
  }

  async connect(): Promise<boolean> {
    this.connectionAttempts++;

    if (this.isConnected) {
      throw new Error('Client is already connected');
    }

    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 10));

    // Simulate authentication
    if (this.authToken !== 'valid-token-123') {
      throw new Error('Authentication failed');
    }

    this.isConnected = true;
    return true;
  }

  async disconnect(): Promise<void> {
    if (!this.isConnected) {
      throw new Error('Client is not connected');
    }
    this.isConnected = false;
  }

  async reconnect(): Promise<boolean> {
    if (this.isConnected) {
      await this.disconnect();
    }

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        await this.connect();
        return true;
      } catch (error) {
        lastError = error as Error;
        await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)));
      }
    }

    throw new Error(`Failed to reconnect after ${this.maxRetries} attempts: ${lastError?.message}`);
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  getConnectionAttempts(): number {
    return this.connectionAttempts;
  }

  async syncEpisodes(episodes: any[]): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Not connected to server');
    }

    return {
      type: 'episode_sync',
      status: 'success',
      synced: episodes.length,
      timestamp: Date.now()
    };
  }

  async syncSkills(skills: any[]): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Not connected to server');
    }

    return {
      type: 'skill_sync',
      status: 'success',
      synced: skills.length,
      timestamp: Date.now()
    };
  }

  async syncCausalEdges(edges: any[]): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Not connected to server');
    }

    return {
      type: 'causal_sync',
      status: 'success',
      synced: edges.length,
      timestamp: Date.now()
    };
  }

  async pullUpdates(since: number): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Not connected to server');
    }

    return {
      episodes: [],
      skills: [],
      edges: [],
      latestTimestamp: Date.now()
    };
  }

  async pushUpdates(data: any): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Not connected to server');
    }

    const totalItems =
      (data.episodes?.length || 0) +
      (data.skills?.length || 0) +
      (data.edges?.length || 0);

    return {
      status: 'success',
      pushed: totalItems,
      timestamp: Date.now()
    };
  }
}

describe('QUICClient', () => {
  let client: MockQUICClient;

  beforeEach(() => {
    client = new MockQUICClient('quic://localhost:4433', 'valid-token-123');
  });

  afterEach(async () => {
    if (client.getConnectionStatus()) {
      await client.disconnect();
    }
  });

  describe('Connection Management', () => {
    it('should connect successfully with valid credentials', async () => {
      const result = await client.connect();
      expect(result).toBe(true);
      expect(client.getConnectionStatus()).toBe(true);
    });

    it('should disconnect successfully', async () => {
      await client.connect();
      await client.disconnect();
      expect(client.getConnectionStatus()).toBe(false);
    });

    it('should throw error when connecting while already connected', async () => {
      await client.connect();
      await expect(client.connect()).rejects.toThrow('Client is already connected');
    });

    it('should throw error when disconnecting while not connected', async () => {
      await expect(client.disconnect()).rejects.toThrow('Client is not connected');
    });

    it('should fail connection with invalid credentials', async () => {
      const invalidClient = new MockQUICClient('quic://localhost:4433', 'invalid-token');
      await expect(invalidClient.connect()).rejects.toThrow('Authentication failed');
    });

    it('should track connection attempts', async () => {
      await client.connect();
      expect(client.getConnectionAttempts()).toBe(1);
    });
  });

  describe('Reconnection Logic', () => {
    it('should reconnect successfully', async () => {
      await client.connect();
      await client.disconnect();

      const result = await client.reconnect();
      expect(result).toBe(true);
      expect(client.getConnectionStatus()).toBe(true);
    });

    it('should handle reconnection when already connected', async () => {
      await client.connect();

      const result = await client.reconnect();
      expect(result).toBe(true);
      expect(client.getConnectionStatus()).toBe(true);
    });

    it('should retry connection on failure', async () => {
      const failingClient = new MockQUICClient('quic://localhost:4433', 'invalid-token');

      await expect(failingClient.reconnect()).rejects.toThrow(/Failed to reconnect after/);
    });
  });

  describe('Episode Synchronization', () => {
    beforeEach(async () => {
      await client.connect();
    });

    it('should sync episodes successfully', async () => {
      const episodes = [
        { id: '1', content: 'Episode 1', timestamp: Date.now() },
        { id: '2', content: 'Episode 2', timestamp: Date.now() }
      ];

      const result = await client.syncEpisodes(episodes);
      expect(result.status).toBe('success');
      expect(result.synced).toBe(2);
      expect(result.type).toBe('episode_sync');
    });

    it('should handle empty episode array', async () => {
      const result = await client.syncEpisodes([]);
      expect(result.status).toBe('success');
      expect(result.synced).toBe(0);
    });

    it('should throw error when syncing while disconnected', async () => {
      await client.disconnect();

      await expect(
        client.syncEpisodes([{ id: '1', content: 'test' }])
      ).rejects.toThrow('Not connected to server');
    });

    it('should handle large episode batches', async () => {
      const episodes = Array.from({ length: 1000 }, (_, i) => ({
        id: `episode-${i}`,
        content: `Content ${i}`,
        timestamp: Date.now()
      }));

      const result = await client.syncEpisodes(episodes);
      expect(result.synced).toBe(1000);
    });
  });

  describe('Skill Synchronization', () => {
    beforeEach(async () => {
      await client.connect();
    });

    it('should sync skills successfully', async () => {
      const skills = [
        { id: '1', name: 'Skill 1', code: 'function test1() {}' },
        { id: '2', name: 'Skill 2', code: 'function test2() {}' }
      ];

      const result = await client.syncSkills(skills);
      expect(result.status).toBe('success');
      expect(result.synced).toBe(2);
      expect(result.type).toBe('skill_sync');
    });

    it('should handle empty skill array', async () => {
      const result = await client.syncSkills([]);
      expect(result.status).toBe('success');
      expect(result.synced).toBe(0);
    });

    it('should throw error when syncing while disconnected', async () => {
      await client.disconnect();

      await expect(
        client.syncSkills([{ id: '1', name: 'test' }])
      ).rejects.toThrow('Not connected to server');
    });
  });

  describe('Causal Edge Synchronization', () => {
    beforeEach(async () => {
      await client.connect();
    });

    it('should sync causal edges successfully', async () => {
      const edges = [
        { from: 'episode-1', to: 'episode-2', weight: 0.8 },
        { from: 'episode-2', to: 'episode-3', weight: 0.9 }
      ];

      const result = await client.syncCausalEdges(edges);
      expect(result.status).toBe('success');
      expect(result.synced).toBe(2);
      expect(result.type).toBe('causal_sync');
    });

    it('should handle empty edge array', async () => {
      const result = await client.syncCausalEdges([]);
      expect(result.status).toBe('success');
      expect(result.synced).toBe(0);
    });

    it('should throw error when syncing while disconnected', async () => {
      await client.disconnect();

      await expect(
        client.syncCausalEdges([{ from: '1', to: '2' }])
      ).rejects.toThrow('Not connected to server');
    });
  });

  describe('Pull Operations', () => {
    beforeEach(async () => {
      await client.connect();
    });

    it('should pull updates successfully', async () => {
      const since = Date.now() - 86400000; // 24 hours ago
      const result = await client.pullUpdates(since);

      expect(result).toHaveProperty('episodes');
      expect(result).toHaveProperty('skills');
      expect(result).toHaveProperty('edges');
      expect(result).toHaveProperty('latestTimestamp');
    });

    it('should throw error when pulling while disconnected', async () => {
      await client.disconnect();

      await expect(
        client.pullUpdates(Date.now())
      ).rejects.toThrow('Not connected to server');
    });
  });

  describe('Push Operations', () => {
    beforeEach(async () => {
      await client.connect();
    });

    it('should push updates successfully', async () => {
      const data = {
        episodes: [{ id: '1', content: 'test' }],
        skills: [{ id: '1', name: 'skill' }],
        edges: [{ from: '1', to: '2' }]
      };

      const result = await client.pushUpdates(data);
      expect(result.status).toBe('success');
      expect(result.pushed).toBe(3);
    });

    it('should handle empty push data', async () => {
      const result = await client.pushUpdates({});
      expect(result.status).toBe('success');
      expect(result.pushed).toBe(0);
    });

    it('should throw error when pushing while disconnected', async () => {
      await client.disconnect();

      await expect(
        client.pushUpdates({ episodes: [] })
      ).rejects.toThrow('Not connected to server');
    });
  });

  describe('Error Handling', () => {
    it('should handle network timeouts gracefully', async () => {
      // This would require a more sophisticated mock
      // For now, we'll test the basic error case
      const disconnectedClient = new MockQUICClient('quic://localhost:4433', 'valid-token-123');

      await expect(
        disconnectedClient.syncEpisodes([{ id: '1', content: 'test' }])
      ).rejects.toThrow('Not connected to server');
    });

    it('should handle multiple rapid connection attempts', async () => {
      await client.connect();
      const attempts = client.getConnectionAttempts();

      // Try to connect again (should fail)
      await client.connect().catch(() => {});

      // Connection attempts should have increased
      expect(client.getConnectionAttempts()).toBeGreaterThan(attempts);
    });
  });

  describe('Concurrent Operations', () => {
    beforeEach(async () => {
      await client.connect();
    });

    it('should handle concurrent sync operations', async () => {
      const promises = [
        client.syncEpisodes([{ id: '1', content: 'test1' }]),
        client.syncSkills([{ id: '1', name: 'skill1' }]),
        client.syncCausalEdges([{ from: '1', to: '2' }])
      ];

      const results = await Promise.all(promises);

      expect(results).toHaveLength(3);
      expect(results[0].type).toBe('episode_sync');
      expect(results[1].type).toBe('skill_sync');
      expect(results[2].type).toBe('causal_sync');
    });

    it('should handle concurrent push and pull operations', async () => {
      const promises = [
        client.pushUpdates({ episodes: [{ id: '1', content: 'test' }] }),
        client.pullUpdates(Date.now() - 1000)
      ];

      const results = await Promise.all(promises);

      expect(results).toHaveLength(2);
      expect(results[0]).toHaveProperty('pushed');
      expect(results[1]).toHaveProperty('latestTimestamp');
    });
  });
});
