/**
 * Unit Tests for Kimi Client
 * 
 * Tests the core functionality of the KimiClient class:
 * - Configuration validation
 * - Error handling
 * - Method behavior
 * - Event emission
 */

const { KimiClient } = require('../../lib/core/kimi-client');
const EventEmitter = require('events');

// Mock fetch for testing
global.fetch = jest.fn();

describe('KimiClient Unit Tests', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  describe('Configuration Validation', () => {
    it('should validate required API key', () => {
      expect(() => {
        new KimiClient({
          provider: 'moonshot',
          apiKey: ''
        });
      }).toThrow('API key is required');
    });

    it('should validate provider', () => {
      expect(() => {
        new KimiClient({
          provider: 'invalid-provider',
          apiKey: 'test-key'
        });
      }).toThrow('Provider must be either "moonshot" or "openrouter"');
    });

    it('should set default values for optional parameters', () => {
      const client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      const status = client.getStatus();
      expect(status.model).toBe('moonshot-v1-128k');
      expect(status.features.multiModal).toBe(true);
      expect(status.features.codeGeneration).toBe(true);
    });

    it('should use custom configuration values', () => {
      const client = new KimiClient({
        provider: 'openrouter',
        apiKey: 'test-key',
        modelVersion: 'custom-model',
        maxTokens: 50000,
        temperature: 0.5,
        features: {
          multiModal: false,
          codeGeneration: true
        }
      });

      const status = client.getStatus();
      expect(status.provider).toBe('openrouter');
      expect(status.model).toBe('custom-model');
      expect(status.features.multiModal).toBe(false);
      expect(status.features.codeGeneration).toBe(true);
    });
  });

  describe('Connection Management', () => {
    let client;

    beforeEach(() => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });
    });

    afterEach(() => {
      client.disconnect();
    });

    it('should start disconnected', () => {
      const status = client.getStatus();
      expect(status.connected).toBe(false);
      expect(status.sessionId).toBeUndefined();
    });

    it('should emit connected event on successful connection', async () => {
      // Mock successful API response
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Test response' } }],
          usage: { total_tokens: 10 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const connectedSpy = jest.fn();
      client.on('connected', connectedSpy);

      await client.connect();

      expect(connectedSpy).toHaveBeenCalled();
      expect(client.getStatus().connected).toBe(true);
      expect(client.getStatus().sessionId).toBeDefined();
    });

    it('should emit error event on connection failure', async () => {
      // Mock failed API response
      fetch.mockResolvedValueOnce({
        ok: false,
        text: async () => 'API Error'
      });

      const errorSpy = jest.fn();
      client.on('error', errorSpy);

      await expect(client.connect()).rejects.toThrow();
      expect(errorSpy).toHaveBeenCalled();
    });

    it('should disconnect properly', async () => {
      // Mock successful connection first
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Test response' } }],
          usage: { total_tokens: 10 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      await client.connect();
      expect(client.getStatus().connected).toBe(true);

      client.disconnect();
      expect(client.getStatus().connected).toBe(false);
      expect(client.getStatus().sessionId).toBeUndefined();
    });
  });

  describe('Chat Functionality', () => {
    let client;

    beforeEach(async () => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      // Mock connection
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Connection test' } }],
          usage: { total_tokens: 5 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      await client.connect();
    });

    afterEach(() => {
      client.disconnect();
    });

    it('should send chat messages successfully', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ 
            message: { 
              content: 'Hello! How can I help you?' 
            }
          }],
          usage: { 
            prompt_tokens: 10,
            completion_tokens: 8,
            total_tokens: 18
          },
          model: 'moonshot-v1-128k'
        }),
        headers: new Map()
      });

      const response = await client.chat('Hello');
      
      expect(response).toBe('Hello! How can I help you?');
      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-key',
            'Content-Type': 'application/json'
          })
        })
      );
    });

    it('should require connection before chatting', async () => {
      const disconnectedClient = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      await expect(disconnectedClient.chat('Hello')).rejects.toThrow(
        'Not connected to Kimi-K2. Run connect() first.'
      );
    });

    it('should maintain conversation history', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Response' } }],
          usage: { total_tokens: 10 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      await client.chat('First message');
      await client.chat('Second message');

      const status = client.getStatus();
      expect(status.conversationLength).toBe(4); // 2 user + 2 assistant messages
    });

    it('should limit conversation history', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Response' } }],
          usage: { total_tokens: 10 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      // Send many messages to test history limiting
      for (let i = 0; i < 15; i++) {
        await client.chat(`Message ${i}`);
      }

      const status = client.getStatus();
      expect(status.conversationLength).toBeLessThanOrEqual(20);
    });

    it('should emit response event', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Test response' } }],
          usage: { total_tokens: 15 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const responseSpy = jest.fn();
      client.on('response', responseSpy);

      await client.chat('Test message');

      expect(responseSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          content: 'Test response',
          usage: expect.objectContaining({
            total_tokens: 15
          })
        })
      );
    });
  });

  describe('Code Generation', () => {
    let client;

    beforeEach(async () => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      // Mock connection
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Connection test' } }],
          usage: { total_tokens: 5 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      await client.connect();
    });

    afterEach(() => {
      client.disconnect();
    });

    it('should generate code successfully', async () => {
      const codeResponse = JSON.stringify({
        code: 'function add(a, b) { return a + b; }',
        explanation: 'This function adds two numbers',
        tests: 'assert(add(2, 3) === 5);'
      });

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: codeResponse } }],
          usage: { total_tokens: 50 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const result = await client.generateCode('Create an add function', 'javascript');

      expect(result.code).toContain('function add');
      expect(result.explanation).toBeDefined();
      expect(result.tests).toBeDefined();
    });

    it('should handle non-JSON responses gracefully', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'function test() { return true; }' } }],
          usage: { total_tokens: 20 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const result = await client.generateCode('Create a test function');

      expect(result.code).toBe('function test() { return true; }');
      expect(result.explanation).toBe('Code generated successfully');
    });
  });

  describe('File Analysis', () => {
    let client;

    beforeEach(async () => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      // Mock connection
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Connection test' } }],
          usage: { total_tokens: 5 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      await client.connect();
    });

    afterEach(() => {
      client.disconnect();
    });

    it('should analyze files successfully', async () => {
      const analysisResponse = JSON.stringify({
        summary: 'Simple function with good structure',
        complexity: 3,
        maintainabilityIndex: 85,
        suggestions: ['Add error handling'],
        issues: [],
        metrics: {
          linesOfCode: 10,
          cyclomaticComplexity: 2,
          cognitiveComplexity: 1,
          technicalDebt: 'Low'
        }
      });

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: analysisResponse } }],
          usage: { total_tokens: 80 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const result = await client.analyzeFile(
        'test.js', 
        'function test() { return true; }',
        'quality'
      );

      expect(result.summary).toBeDefined();
      expect(result.complexity).toBeDefined();
      expect(result.suggestions).toBeInstanceOf(Array);
      expect(result.metrics).toBeDefined();
    });

    it('should provide fallback analysis for invalid responses', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Invalid JSON response' } }],
          usage: { total_tokens: 20 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const result = await client.analyzeFile('test.js', 'code content');

      expect(result.summary).toBe('Invalid JSON response');
      expect(result.complexity).toBeDefined();
      expect(result.suggestions).toBeInstanceOf(Array);
    });
  });

  describe('Error Handling', () => {
    let client;

    beforeEach(() => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key',
        retryAttempts: 2
      });
    });

    it('should retry failed requests', async () => {
      // First attempt fails, second succeeds
      fetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            choices: [{ message: { content: 'Success' } }],
            usage: { total_tokens: 10 },
            model: 'test-model'
          }),
          headers: new Map()
        });

      await client.connect();
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    it('should fail after max retries', async () => {
      fetch.mockRejectedValue(new Error('Persistent error'));

      await expect(client.connect()).rejects.toThrow();
      expect(fetch).toHaveBeenCalledTimes(2); // Initial + 1 retry
    });

    it('should emit error events', async () => {
      fetch.mockRejectedValue(new Error('Test error'));

      const errorSpy = jest.fn();
      client.on('error', errorSpy);

      await expect(client.connect()).rejects.toThrow();
      expect(errorSpy).toHaveBeenCalled();
    });
  });

  describe('Header Building', () => {
    it('should build correct headers for Moonshot', () => {
      const client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-moonshot-key'
      });

      // Access private method through connection attempt
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'test' } }],
          usage: { total_tokens: 5 },
          model: 'test'
        }),
        headers: new Map()
      });

      client.connect().catch(() => {}); // Ignore errors, just check headers

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-moonshot-key',
            'Content-Type': 'application/json'
          })
        })
      );
    });

    it('should build correct headers for OpenRouter', () => {
      const client = new KimiClient({
        provider: 'openrouter',
        apiKey: 'test-openrouter-key'
      });

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'test' } }],
          usage: { total_tokens: 5 },
          model: 'test'
        }),
        headers: new Map()
      });

      client.connect().catch(() => {});

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-openrouter-key',
            'HTTP-Referer': 'https://github.com/ruvnet/Synaptic-Neural-Mesh',
            'X-Title': 'Synaptic Neural Mesh'
          })
        })
      );
    });
  });

  describe('Memory Management', () => {
    let client;

    beforeEach(async () => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      // Mock connection
      fetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'Response' } }],
          usage: { total_tokens: 10 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      await client.connect();
    });

    afterEach(() => {
      client.disconnect();
    });

    it('should clear conversation history', async () => {
      await client.chat('Test message');
      expect(client.getStatus().conversationLength).toBeGreaterThan(0);

      const historyClearedSpy = jest.fn();
      client.on('history_cleared', historyClearedSpy);

      client.clearHistory();
      
      expect(client.getStatus().conversationLength).toBe(0);
      expect(historyClearedSpy).toHaveBeenCalled();
    });
  });

  describe('Event Emission', () => {
    it('should extend EventEmitter', () => {
      const client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      expect(client).toBeInstanceOf(EventEmitter);
      expect(typeof client.on).toBe('function');
      expect(typeof client.emit).toBe('function');
    });

    it('should emit api_call events', async () => {
      const client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: 'test' } }],
          usage: { total_tokens: 15 },
          model: 'test-model'
        }),
        headers: new Map()
      });

      const apiCallSpy = jest.fn();
      client.on('api_call', apiCallSpy);

      await client.connect();

      expect(apiCallSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          success: true,
          tokens: 15,
          model: 'test-model'
        })
      );
    });
  });
});