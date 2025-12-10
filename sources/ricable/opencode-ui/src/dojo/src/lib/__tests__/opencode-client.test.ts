/**
 * OpenCode API Client Tests
 * 
 * Comprehensive test suite for the OpenCode API client functionality
 */

import { describe, it, expect, beforeEach, afterEach, vi, type Mock } from 'vitest';
import { 
  OpenCodeClient, 
  OpenCodeAPIError, 
  isValidSession, 
  isValidProvider 
} from '../opencode-client';

// Mock fetch globally
const mockFetch = vi.fn() as Mock;
global.fetch = mockFetch;

// Mock WebSocket
class MockWebSocket {
  public onopen: ((event: Event) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  
  public readyState = WebSocket.CONNECTING;
  
  constructor(public url: string) {}
  
  close(code?: number, reason?: string) {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code, reason }));
    }
  }
  
  send(data: string) {
    // Mock implementation
  }
}

global.WebSocket = MockWebSocket as any;

describe('OpenCodeClient', () => {
  let client: OpenCodeClient;
  
  beforeEach(() => {
    client = new OpenCodeClient('http://localhost:8080');
    mockFetch.mockClear();
  });
  
  afterEach(() => {
    client.disconnect();
  });

  describe('Constructor and Basic Setup', () => {
    it('should initialize with default base URL', () => {
      const defaultClient = new OpenCodeClient();
      expect(defaultClient).toBeDefined();
    });

    it('should initialize with custom base URL', () => {
      const customClient = new OpenCodeClient('http://custom:3000');
      expect(customClient).toBeDefined();
    });
  });

  describe('HTTP Client Methods', () => {
    it('should make successful GET request', async () => {
      const mockResponse = { status: 'ok', data: 'test' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client['request']('/test');
      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/test',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('should handle HTTP errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: () => Promise.resolve('Not found'),
      });

      await expect(client['request']('/test')).rejects.toThrow(OpenCodeAPIError);
    });
  });

  describe('Provider Management', () => {
    it('should get providers list', async () => {
      const providers = await client.getProviders();
      
      expect(Array.isArray(providers)).toBe(true);
      expect(providers.length).toBeGreaterThan(0);
      
      // Check first provider structure
      const provider = providers[0];
      expect(provider).toHaveProperty('id');
      expect(provider).toHaveProperty('name');
      expect(provider).toHaveProperty('type');
      expect(provider).toHaveProperty('models');
      expect(Array.isArray(provider.models)).toBe(true);
    });

    it('should authenticate provider', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, provider_id: 'anthropic' }),
      });

      const result = await client.authenticateProvider('anthropic', { apiKey: 'test' });
      expect(result.success).toBe(true);
    });

    it('should get provider metrics', async () => {
      const metrics = await client.getProviderMetrics();
      
      expect(Array.isArray(metrics)).toBe(true);
      expect(metrics.length).toBeGreaterThan(0);
      
      // Check metrics structure
      const metric = metrics[0];
      expect(metric).toHaveProperty('provider_id');
      expect(metric).toHaveProperty('requests');
      expect(metric).toHaveProperty('avg_response_time');
      expect(metric).toHaveProperty('total_cost');
    });
  });

  describe('Session Management', () => {
    it('should create session', async () => {
      const sessionConfig = {
        provider: 'anthropic',
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 8000,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'test-session',
          name: 'Test Session',
          ...sessionConfig,
          created_at: Date.now(),
          updated_at: Date.now(),
          status: 'active',
          message_count: 0,
          total_cost: 0,
          config: sessionConfig,
        }),
      });

      const session = await client.createSession(sessionConfig);
      expect(session.id).toBe('test-session');
      expect(session.provider).toBe('anthropic');
    });

    it('should get sessions list', async () => {
      const sessions = await client.getSessions();
      
      expect(Array.isArray(sessions)).toBe(true);
      
      if (sessions.length > 0) {
        const session = sessions[0];
        expect(session).toHaveProperty('id');
        expect(session).toHaveProperty('name');
        expect(session).toHaveProperty('provider');
        expect(session).toHaveProperty('model');
      }
    });

    it('should send message to session', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ messageId: 'msg-123' }),
      });

      const result = await client.sendMessage('session-1', 'Hello, world!');
      expect(result.messageId).toBe('msg-123');
    });

    it('should get session messages', async () => {
      const messages = await client.getMessages('session-1');
      
      expect(Array.isArray(messages)).toBe(true);
      
      if (messages.length > 0) {
        const message = messages[0];
        expect(message).toHaveProperty('id');
        expect(message).toHaveProperty('role');
        expect(message).toHaveProperty('content');
        expect(message).toHaveProperty('timestamp');
      }
    });

    it('should share session', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          url: 'https://share.opencode.ai/abc123',
          expires_at: Date.now() + 86400000,
          password_protected: false,
          view_count: 0,
        }),
      });

      const shareLink = await client.shareSession('session-1');
      expect(shareLink.url).toContain('share.opencode.ai');
    });
  });

  describe('WebSocket Functionality', () => {
    it('should subscribe to session updates', () => {
      const mockCallback = vi.fn();
      const unsubscribe = client.subscribeToSession('session-1', mockCallback);
      
      expect(typeof unsubscribe).toBe('function');
      
      // Clean up
      unsubscribe();
    });

    it('should handle WebSocket connection', () => {
      const mockCallback = vi.fn();
      const unsubscribe = client.subscribeToSession('session-1', mockCallback);
      
      // Simulate WebSocket connection
      const wsConnections = client['websockets'];
      expect(wsConnections.has('session-1')).toBe(true);
      
      unsubscribe();
      expect(wsConnections.has('session-1')).toBe(false);
    });
  });

  describe('Tool System', () => {
    it('should get available tools', async () => {
      const tools = await client.getTools();
      
      expect(Array.isArray(tools)).toBe(true);
      expect(tools.length).toBeGreaterThan(0);
      
      const tool = tools[0];
      expect(tool).toHaveProperty('id');
      expect(tool).toHaveProperty('name');
      expect(tool).toHaveProperty('description');
      expect(tool).toHaveProperty('category');
    });

    it('should execute tool', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          result: 'Tool executed successfully',
          execution_time: 150,
          tool_id: 'test_tool',
        }),
      });

      const result = await client.executeTool('test_tool', { param: 'value' });
      expect(result.success).toBe(true);
      expect(result.tool_id).toBe('test_tool');
    });

    it('should get tool executions', async () => {
      const executions = await client.getToolExecutions();
      
      expect(Array.isArray(executions)).toBe(true);
      
      if (executions.length > 0) {
        const execution = executions[0];
        expect(execution).toHaveProperty('id');
        expect(execution).toHaveProperty('tool_id');
        expect(execution).toHaveProperty('status');
      }
    });
  });

  describe('Configuration Management', () => {
    it('should get configuration', async () => {
      const config = await client.getConfig();
      
      expect(config).toHaveProperty('theme');
      expect(config).toHaveProperty('model');
      expect(config).toHaveProperty('providers');
      expect(config).toHaveProperty('agents');
    });

    it('should update configuration', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await expect(client.updateConfig({ theme: 'dark' })).resolves.not.toThrow();
    });

    it('should validate configuration', async () => {
      const mockConfig = {
        theme: 'opencode',
        model: 'anthropic/claude-3-5-sonnet-20241022',
        autoshare: false,
        autoupdate: true,
        providers: {},
        agents: {},
        mcp: {},
        lsp: {},
        keybinds: {},
        shell: { path: '/bin/bash', args: ['-l'] },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          valid: true,
          errors: [],
          warnings: [],
        }),
      });

      const validation = await client.validateConfig(mockConfig);
      expect(validation.valid).toBe(true);
      expect(Array.isArray(validation.errors)).toBe(true);
    });
  });

  describe('Health and Connection', () => {
    it('should check health', async () => {
      const health = await client.healthCheck();
      
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('version');
    });

    it('should test connection', async () => {
      const connectionTest = await client.testConnection();
      
      expect(connectionTest).toHaveProperty('success');
      expect(connectionTest).toHaveProperty('latency');
      expect(typeof connectionTest.latency).toBe('number');
    });

    it('should get connection status', () => {
      const status = client.getConnectionStatus();
      
      expect(status).toHaveProperty('status');
      expect(status).toHaveProperty('activeWebSockets');
      expect(status).toHaveProperty('reconnectAttempts');
    });
  });

  describe('Event System', () => {
    it('should register and emit events', () => {
      const mockCallback = vi.fn();
      const unsubscribe = client.on('test_event', mockCallback);
      
      client.emit('test_event', { data: 'test' });
      
      expect(mockCallback).toHaveBeenCalledWith({ data: 'test' });
      expect(client.listenerCount('test_event')).toBe(1);
      
      unsubscribe();
      expect(client.listenerCount('test_event')).toBe(0);
    });

    it('should handle once events', () => {
      const mockCallback = vi.fn();
      client.once('test_once', mockCallback);
      
      client.emit('test_once', { data: 'first' });
      client.emit('test_once', { data: 'second' });
      
      expect(mockCallback).toHaveBeenCalledTimes(1);
      expect(mockCallback).toHaveBeenCalledWith({ data: 'first' });
    });
  });

  describe('Enhanced Features', () => {
    it('should get LSP servers', async () => {
      const lspServers = await client.getLSPServers();
      
      expect(Array.isArray(lspServers)).toBe(true);
      
      if (lspServers.length > 0) {
        const server = lspServers[0];
        expect(server).toHaveProperty('id');
        expect(server).toHaveProperty('name');
        expect(server).toHaveProperty('command');
        expect(server).toHaveProperty('status');
      }
    });

    it('should get diagnostics', async () => {
      const diagnostics = await client.getDiagnostics();
      
      expect(Array.isArray(diagnostics)).toBe(true);
      
      if (diagnostics.length > 0) {
        const diagnostic = diagnostics[0];
        expect(diagnostic).toHaveProperty('file_path');
        expect(diagnostic).toHaveProperty('line');
        expect(diagnostic).toHaveProperty('column');
        expect(diagnostic).toHaveProperty('severity');
        expect(diagnostic).toHaveProperty('message');
      }
    });

    it('should get custom commands', async () => {
      const commands = await client.getCustomCommands();
      
      expect(Array.isArray(commands)).toBe(true);
      
      if (commands.length > 0) {
        const command = commands[0];
        expect(command).toHaveProperty('id');
        expect(command).toHaveProperty('name');
        expect(command).toHaveProperty('command');
      }
    });

    it('should get usage statistics', async () => {
      const stats = await client.getUsageStats();
      
      expect(stats).toHaveProperty('total_sessions');
      expect(stats).toHaveProperty('total_messages');
      expect(stats).toHaveProperty('total_cost');
      expect(stats).toHaveProperty('today');
      expect(stats).toHaveProperty('this_week');
      expect(stats).toHaveProperty('this_month');
    });
  });

  describe('Type Guards', () => {
    it('should validate session objects', () => {
      const validSession = {
        id: 'test-session',
        name: 'Test Session',
        provider: 'anthropic',
        model: 'claude-3-5-sonnet-20241022',
        created_at: Date.now(),
        updated_at: Date.now(),
        status: 'active',
        message_count: 0,
        total_cost: 0,
        config: {},
      };

      const invalidSession = {
        id: 'test-session',
        // missing required fields
      };

      expect(isValidSession(validSession)).toBe(true);
      expect(isValidSession(invalidSession)).toBe(false);
      expect(isValidSession(null)).toBe(false);
    });

    it('should validate provider objects', () => {
      const validProvider = {
        id: 'anthropic',
        name: 'Anthropic',
        type: 'anthropic',
        models: ['claude-3-5-sonnet-20241022'],
        authenticated: true,
        status: 'online',
        cost_per_1k_tokens: 0.003,
        avg_response_time: 850,
        description: 'Claude AI models',
      };

      const invalidProvider = {
        id: 'anthropic',
        name: 'Anthropic',
        // missing required fields
      };

      expect(isValidProvider(validProvider)).toBe(true);
      expect(isValidProvider(invalidProvider)).toBe(false);
      expect(isValidProvider(null)).toBe(false);
    });
  });
});