/**
 * Kimi-K2 Integration Test Suite
 * Tests the MCP tools and API client functionality
 */

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { SynapticMCPServer } from '../synaptic-mcp-server.js';
import { KimiClient, KimiMultiProvider } from '../../js/synaptic-cli/lib/kimi-client.js';

// Mock fetch for testing
global.fetch = jest.fn();

describe('Kimi-K2 Integration', () => {
  let mcpServer: SynapticMCPServer;
  let mockFetch: jest.MockedFunction<typeof fetch>;

  beforeEach(() => {
    mcpServer = new SynapticMCPServer();
    mockFetch = fetch as jest.MockedFunction<typeof fetch>;
    mockFetch.mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('MCP Tool Integration', () => {
    it('should test provider connections', async () => {
      // Mock successful API responses
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            id: 'test-response',
            object: 'chat.completion',
            created: Date.now(),
            model: 'moonshot-v1-128k',
            choices: [{
              index: 0,
              message: { role: 'assistant', content: 'Hello' },
              finish_reason: 'stop'
            }],
            usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
          })
        } as Response)
      );

      const result = await mcpServer.executeTool('kimi_provider_test', {
        providers: ['moonshot', 'openrouter'],
        timeout: 5000
      });

      expect(result.success).toBe(true);
      expect(result.tested_providers).toBe(2);
      expect(result.results).toHaveProperty('moonshot');
      expect(result.results).toHaveProperty('openrouter');
    });

    it('should handle chat completion requests', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            id: 'chat-test',
            object: 'chat.completion',
            created: Date.now(),
            model: 'moonshot-v1-128k',
            choices: [{
              index: 0,
              message: { 
                role: 'assistant', 
                content: 'A synaptic neural mesh is a distributed network of interconnected artificial neurons...' 
              },
              finish_reason: 'stop'
            }],
            usage: { prompt_tokens: 50, completion_tokens: 100, total_tokens: 150 }
          })
        } as Response)
      );

      const result = await mcpServer.executeTool('kimi_chat_completion', {
        provider: 'moonshot',
        messages: [
          { role: 'user', content: 'What is a synaptic neural mesh?' }
        ],
        temperature: 0.7,
        max_tokens: 500
      });

      expect(result.success).toBe(true);
      expect(result.provider).toBe('moonshot');
      expect(result.response).toHaveProperty('content');
      expect(result.usage).toHaveProperty('total_tokens');
    });

    it('should handle tool calling responses', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            id: 'tool-test',
            object: 'chat.completion',
            created: Date.now(),
            model: 'moonshot-v1-128k',
            choices: [{
              index: 0,
              message: { 
                role: 'assistant',
                content: null,
                tool_calls: [{
                  id: 'call_123',
                  type: 'function',
                  function: {
                    name: 'mesh_status',
                    arguments: '{"meshId": "test-mesh"}'
                  }
                }]
              },
              finish_reason: 'tool_calls'
            }],
            usage: { prompt_tokens: 30, completion_tokens: 20, total_tokens: 50 }
          })
        } as Response)
      );

      const chatResult = await mcpServer.executeTool('kimi_chat_completion', {
        provider: 'moonshot',
        messages: [
          { role: 'user', content: 'Get the mesh status' }
        ],
        tools: [{
          type: 'function',
          function: {
            name: 'mesh_status',
            description: 'Get mesh status',
            parameters: {
              type: 'object',
              properties: {
                meshId: { type: 'string' }
              }
            }
          }
        }],
        tool_choice: 'auto'
      });

      expect(chatResult.success).toBe(true);
      expect(chatResult.tool_calls).toHaveLength(1);
      expect(chatResult.tool_calls[0].function.name).toBe('mesh_status');

      // Execute the tool calls
      const toolResult = await mcpServer.executeTool('kimi_tool_execution', {
        tool_calls: chatResult.tool_calls
      });

      expect(toolResult.success).toBe(true);
      expect(toolResult.executed_tools).toBe(1);
      expect(toolResult.results).toHaveLength(1);
    });

    it('should manage conversation context', async () => {
      const longConversation = Array(100).fill(null).map((_, i) => ({
        role: i % 2 === 0 ? 'user' : 'assistant',
        content: `Message ${i + 1}: This is a test message with some content.`
      }));

      const result = await mcpServer.executeTool('kimi_context_management', {
        messages: longConversation,
        context_window: 128000,
        strategy: 'sliding_window'
      });

      expect(result.success).toBe(true);
      expect(result.original_messages).toBe(100);
      expect(result.managed_messages).toBeLessThanOrEqual(100);
      expect(result.tokens_saved).toBeGreaterThanOrEqual(0);
    });

    it('should list available models', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            data: [
              { id: 'moonshot-v1-128k', object: 'model', created: Date.now() },
              { id: 'moonshot-v1-32k', object: 'model', created: Date.now() }
            ]
          })
        } as Response)
      );

      const result = await mcpServer.executeTool('kimi_model_list', {
        provider: 'moonshot'
      });

      expect(result.success).toBe(true);
      expect(result.models.moonshot).toHaveLength(2);
      expect(result.models.moonshot[0].id).toBe('moonshot-v1-128k');
    });
  });

  describe('KimiClient Direct Usage', () => {
    let client: KimiClient;

    beforeEach(() => {
      client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key',
        model: 'moonshot-v1-128k'
      });
    });

    it('should initialize with correct configuration', () => {
      expect(client).toBeInstanceOf(KimiClient);
    });

    it('should handle chat completion requests', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            id: 'test',
            object: 'chat.completion',
            created: Date.now(),
            model: 'moonshot-v1-128k',
            choices: [{
              index: 0,
              message: { role: 'assistant', content: 'Test response' },
              finish_reason: 'stop'
            }],
            usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
          })
        } as Response)
      );

      const response = await client.chatCompletion({
        model: 'moonshot-v1-128k',
        messages: [{ role: 'user', content: 'Hello' }]
      });

      expect(response.choices).toHaveLength(1);
      expect(response.choices[0].message.content).toBe('Test response');
    });

    it('should manage context correctly', () => {
      const messages = Array(50).fill(null).map((_, i) => ({
        role: i % 2 === 0 ? 'user' : 'assistant' as const,
        content: `Message ${i}: This is a longer message to test token estimation.`
      }));

      const managedMessages = client.manageContext(messages);
      expect(managedMessages.length).toBeLessThanOrEqual(messages.length);
    });

    it('should estimate tokens correctly', () => {
      const messages = [
        { role: 'user' as const, content: 'Short message' },
        { role: 'assistant' as const, content: 'This is a longer response with more tokens' }
      ];

      const tokens = (client as any).estimateTokens(messages);
      expect(tokens).toBeGreaterThan(0);
      expect(typeof tokens).toBe('number');
    });

    it('should handle API errors gracefully', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: false,
          status: 401,
          statusText: 'Unauthorized'
        } as Response)
      );

      await expect(client.chatCompletion({
        model: 'moonshot-v1-128k',
        messages: [{ role: 'user', content: 'Hello' }]
      })).rejects.toThrow('API request failed: 401 Unauthorized');
    });
  });

  describe('KimiMultiProvider', () => {
    let multiProvider: KimiMultiProvider;

    beforeEach(() => {
      multiProvider = new KimiMultiProvider();
      multiProvider.addProvider('moonshot', {
        provider: 'moonshot',
        apiKey: 'test-key-1',
        model: 'moonshot-v1-128k'
      });
      multiProvider.addProvider('openrouter', {
        provider: 'openrouter',
        apiKey: 'test-key-2',
        model: 'anthropic/claude-3.5-sonnet'
      });
    });

    it('should add and retrieve providers', () => {
      const moonshotClient = multiProvider.getProvider('moonshot');
      const openrouterClient = multiProvider.getProvider('openrouter');

      expect(moonshotClient).toBeInstanceOf(KimiClient);
      expect(openrouterClient).toBeInstanceOf(KimiClient);
      expect(multiProvider.getProvider('nonexistent')).toBeUndefined();
    });

    it('should test all providers', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            id: 'test',
            object: 'chat.completion',
            created: Date.now(),
            model: 'test-model',
            choices: [{ index: 0, message: { role: 'assistant', content: 'Test' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 5, completion_tokens: 5, total_tokens: 10 }
          })
        } as Response)
      );

      const results = await multiProvider.testAllProviders();
      expect(results).toHaveProperty('moonshot');
      expect(results).toHaveProperty('openrouter');
    });

    it('should find the best provider', async () => {
      mockFetch
        .mockImplementationOnce(() => Promise.reject(new Error('Failed')))
        .mockImplementationOnce(() =>
          Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              id: 'test',
              object: 'chat.completion',
              created: Date.now(),
              model: 'test-model',
              choices: [{ index: 0, message: { role: 'assistant', content: 'Test' }, finish_reason: 'stop' }],
              usage: { prompt_tokens: 5, completion_tokens: 5, total_tokens: 10 }
            })
          } as Response)
        );

      const bestProvider = await multiProvider.getBestProvider();
      expect(bestProvider).not.toBeNull();
      expect(bestProvider?.name).toBe('openrouter');
    });
  });

  describe('Error Handling', () => {
    it('should handle network timeouts', async () => {
      mockFetch.mockImplementation(() =>
        new Promise(() => {}) // Never resolves - simulates timeout
      );

      const client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key',
        timeout: 100 // Very short timeout
      });

      await expect(client.chatCompletion({
        model: 'moonshot-v1-128k',
        messages: [{ role: 'user', content: 'Hello' }]
      })).rejects.toThrow('Request timeout after 100ms');
    });

    it('should handle malformed responses', async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ invalid: 'response' })
        } as Response)
      );

      const client = new KimiClient({
        provider: 'moonshot',
        apiKey: 'test-key'
      });

      await expect(client.chatCompletion({
        model: 'moonshot-v1-128k',
        messages: [{ role: 'user', content: 'Hello' }]
      })).rejects.toThrow('Invalid response: no choices returned');
    });
  });
});