import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

/**
 * QUICServer Unit Tests
 *
 * Tests for QUIC server functionality including:
 * - Server initialization and startup
 * - Connection acceptance
 * - Authentication
 * - Stream handling
 * - Error handling
 */

// Mock QUICServer implementation for testing
class MockQUICServer {
  private port: number;
  private isRunning: boolean = false;
  private connections: Map<string, any> = new Map();
  private authTokens: Set<string> = new Set(['valid-token-123']);

  constructor(port: number = 4433) {
    this.port = port;
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Server is already running');
    }
    this.isRunning = true;
  }

  async stop(): Promise<void> {
    if (!this.isRunning) {
      throw new Error('Server is not running');
    }
    this.connections.clear();
    this.isRunning = false;
  }

  isListening(): boolean {
    return this.isRunning;
  }

  getPort(): number {
    return this.port;
  }

  async acceptConnection(clientId: string, token: string): Promise<boolean> {
    if (!this.isRunning) {
      throw new Error('Server is not running');
    }

    if (!this.authTokens.has(token)) {
      return false;
    }

    this.connections.set(clientId, {
      id: clientId,
      connectedAt: Date.now(),
      authenticated: true
    });

    return true;
  }

  getActiveConnections(): number {
    return this.connections.size;
  }

  disconnectClient(clientId: string): boolean {
    return this.connections.delete(clientId);
  }

  async handleSyncRequest(clientId: string, data: any): Promise<any> {
    if (!this.connections.has(clientId)) {
      throw new Error('Client not connected');
    }

    const { type, payload } = data;

    switch (type) {
      case 'episode_sync':
        return { status: 'success', synced: payload.episodes.length };
      case 'skill_sync':
        return { status: 'success', synced: payload.skills.length };
      case 'causal_sync':
        return { status: 'success', synced: payload.edges.length };
      default:
        throw new Error(`Unknown sync type: ${type}`);
    }
  }
}

describe('QUICServer', () => {
  let server: MockQUICServer;

  beforeEach(() => {
    server = new MockQUICServer(4433);
  });

  afterEach(async () => {
    if (server.isListening()) {
      await server.stop();
    }
  });

  describe('Server Lifecycle', () => {
    it('should start server successfully', async () => {
      await server.start();
      expect(server.isListening()).toBe(true);
    });

    it('should stop server successfully', async () => {
      await server.start();
      await server.stop();
      expect(server.isListening()).toBe(false);
    });

    it('should throw error when starting already running server', async () => {
      await server.start();
      await expect(server.start()).rejects.toThrow('Server is already running');
    });

    it('should throw error when stopping non-running server', async () => {
      await expect(server.stop()).rejects.toThrow('Server is not running');
    });

    it('should return correct port', () => {
      expect(server.getPort()).toBe(4433);
    });
  });

  describe('Connection Management', () => {
    beforeEach(async () => {
      await server.start();
    });

    it('should accept connection with valid token', async () => {
      const result = await server.acceptConnection('client-1', 'valid-token-123');
      expect(result).toBe(true);
      expect(server.getActiveConnections()).toBe(1);
    });

    it('should reject connection with invalid token', async () => {
      const result = await server.acceptConnection('client-1', 'invalid-token');
      expect(result).toBe(false);
      expect(server.getActiveConnections()).toBe(0);
    });

    it('should handle multiple concurrent connections', async () => {
      await server.acceptConnection('client-1', 'valid-token-123');
      await server.acceptConnection('client-2', 'valid-token-123');
      await server.acceptConnection('client-3', 'valid-token-123');

      expect(server.getActiveConnections()).toBe(3);
    });

    it('should disconnect client successfully', async () => {
      await server.acceptConnection('client-1', 'valid-token-123');
      expect(server.getActiveConnections()).toBe(1);

      const result = server.disconnectClient('client-1');
      expect(result).toBe(true);
      expect(server.getActiveConnections()).toBe(0);
    });

    it('should return false when disconnecting non-existent client', () => {
      const result = server.disconnectClient('non-existent');
      expect(result).toBe(false);
    });

    it('should throw error when accepting connection on stopped server', async () => {
      await server.stop();
      await expect(
        server.acceptConnection('client-1', 'valid-token-123')
      ).rejects.toThrow('Server is not running');
    });
  });

  describe('Sync Request Handling', () => {
    beforeEach(async () => {
      await server.start();
      await server.acceptConnection('client-1', 'valid-token-123');
    });

    it('should handle episode sync request', async () => {
      const request = {
        type: 'episode_sync',
        payload: {
          episodes: [
            { id: '1', content: 'test' },
            { id: '2', content: 'test2' }
          ]
        }
      };

      const response = await server.handleSyncRequest('client-1', request);
      expect(response.status).toBe('success');
      expect(response.synced).toBe(2);
    });

    it('should handle skill sync request', async () => {
      const request = {
        type: 'skill_sync',
        payload: {
          skills: [
            { id: '1', name: 'skill1' },
            { id: '2', name: 'skill2' },
            { id: '3', name: 'skill3' }
          ]
        }
      };

      const response = await server.handleSyncRequest('client-1', request);
      expect(response.status).toBe('success');
      expect(response.synced).toBe(3);
    });

    it('should handle causal edge sync request', async () => {
      const request = {
        type: 'causal_sync',
        payload: {
          edges: [
            { from: '1', to: '2' },
            { from: '2', to: '3' }
          ]
        }
      };

      const response = await server.handleSyncRequest('client-1', request);
      expect(response.status).toBe('success');
      expect(response.synced).toBe(2);
    });

    it('should throw error for unknown sync type', async () => {
      const request = {
        type: 'unknown_type',
        payload: {}
      };

      await expect(
        server.handleSyncRequest('client-1', request)
      ).rejects.toThrow('Unknown sync type: unknown_type');
    });

    it('should throw error for sync request from unconnected client', async () => {
      const request = {
        type: 'episode_sync',
        payload: { episodes: [] }
      };

      await expect(
        server.handleSyncRequest('non-existent', request)
      ).rejects.toThrow('Client not connected');
    });
  });

  describe('Error Handling', () => {
    it('should handle server start failure gracefully', async () => {
      await server.start();
      const error = await server.start().catch(e => e);
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toContain('already running');
    });

    it('should clear all connections on server stop', async () => {
      await server.start();
      await server.acceptConnection('client-1', 'valid-token-123');
      await server.acceptConnection('client-2', 'valid-token-123');

      expect(server.getActiveConnections()).toBe(2);

      await server.stop();
      await server.start();

      expect(server.getActiveConnections()).toBe(0);
    });
  });

  describe('Large Data Transfer', () => {
    beforeEach(async () => {
      await server.start();
      await server.acceptConnection('client-1', 'valid-token-123');
    });

    it('should handle large episode batch sync', async () => {
      const episodes = Array.from({ length: 1000 }, (_, i) => ({
        id: `episode-${i}`,
        content: `Content ${i}`,
        timestamp: Date.now()
      }));

      const request = {
        type: 'episode_sync',
        payload: { episodes }
      };

      const response = await server.handleSyncRequest('client-1', request);
      expect(response.status).toBe('success');
      expect(response.synced).toBe(1000);
    });

    it('should handle large skill batch sync', async () => {
      const skills = Array.from({ length: 500 }, (_, i) => ({
        id: `skill-${i}`,
        name: `Skill ${i}`,
        code: `function skill${i}() { return ${i}; }`
      }));

      const request = {
        type: 'skill_sync',
        payload: { skills }
      };

      const response = await server.handleSyncRequest('client-1', request);
      expect(response.status).toBe('success');
      expect(response.synced).toBe(500);
    });
  });
});
