// Anthropic provider implementation
import Anthropic from '@anthropic-ai/sdk';
import {
  LLMProvider,
  ChatParams,
  ChatResponse,
  StreamChunk,
  ProviderConfig,
  ProviderError,
  ContentBlock
} from '../types.js';

export class AnthropicProvider implements LLMProvider {
  name = 'anthropic';
  type = 'anthropic' as const;
  supportsStreaming = true;
  supportsTools = true;
  supportsMCP = true; // Native support

  private client: Anthropic;
  private config: ProviderConfig;

  constructor(config: ProviderConfig) {
    this.config = config;

    if (!config.apiKey) {
      throw new Error('Anthropic API key is required');
    }

    this.client = new Anthropic({
      apiKey: config.apiKey,
      baseURL: config.baseUrl,
      timeout: config.timeout || 120000,
      maxRetries: config.maxRetries || 3
    });
  }

  validateCapabilities(features: string[]): boolean {
    const supported = ['chat', 'streaming', 'tools', 'mcp'];
    return features.every(f => supported.includes(f));
  }

  async chat(params: ChatParams): Promise<ChatResponse> {
    try {
      // Extract system message if present (Anthropic requires it as top-level parameter)
      const systemMessage = params.messages.find(m => m.role === 'system');
      const nonSystemMessages = params.messages.filter(m => m.role !== 'system');

      const response = await this.client.messages.create({
        model: params.model,
        messages: nonSystemMessages as any,
        system: systemMessage ? (typeof systemMessage.content === 'string' ? systemMessage.content : JSON.stringify(systemMessage.content)) : undefined,
        temperature: params.temperature,
        max_tokens: params.maxTokens || 4096,
        tools: params.tools as any,
        tool_choice: params.toolChoice as any
      });

      return {
        id: response.id,
        model: response.model,
        content: response.content as ContentBlock[],
        stopReason: response.stop_reason as any,
        usage: {
          inputTokens: response.usage.input_tokens,
          outputTokens: response.usage.output_tokens
        },
        metadata: {
          provider: 'anthropic',
          cost: this.calculateCost(response.usage),
          latency: 0 // Will be set by router
        }
      };
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  async *stream(params: ChatParams): AsyncGenerator<StreamChunk> {
    try {
      // Extract system message if present (Anthropic requires it as top-level parameter)
      const systemMessage = params.messages.find(m => m.role === 'system');
      const nonSystemMessages = params.messages.filter(m => m.role !== 'system');

      const stream = await this.client.messages.create({
        model: params.model,
        messages: nonSystemMessages as any,
        system: systemMessage ? (typeof systemMessage.content === 'string' ? systemMessage.content : JSON.stringify(systemMessage.content)) : undefined,
        temperature: params.temperature,
        max_tokens: params.maxTokens || 4096,
        tools: params.tools as any,
        tool_choice: params.toolChoice as any,
        stream: true
      });

      for await (const event of stream) {
        yield event as StreamChunk;
      }
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  private calculateCost(usage: { input_tokens: number; output_tokens: number }): number {
    // Claude 3.5 Sonnet pricing: $3/MTok input, $15/MTok output
    const inputCost = (usage.input_tokens / 1_000_000) * 3;
    const outputCost = (usage.output_tokens / 1_000_000) * 15;
    return inputCost + outputCost;
  }

  private handleError(error: any): ProviderError {
    const providerError = new Error(
      error.message || 'Anthropic request failed'
    ) as ProviderError;

    providerError.provider = 'anthropic';
    providerError.statusCode = error.status;
    providerError.retryable = error.status >= 500 || error.status === 429;

    return providerError;
  }
}
