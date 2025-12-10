// Google Gemini provider implementation
import { GoogleGenAI } from '@google/genai';
import {
  LLMProvider,
  ChatParams,
  ChatResponse,
  StreamChunk,
  ProviderConfig,
  ProviderError,
  ContentBlock
} from '../types.js';

export class GeminiProvider implements LLMProvider {
  name = 'gemini';
  type = 'gemini' as const;
  supportsStreaming = true;
  supportsTools = false;
  supportsMCP = false;

  private client: GoogleGenAI;
  private config: ProviderConfig;

  constructor(config: ProviderConfig) {
    this.config = config;

    if (!config.apiKey) {
      throw new Error('Google Gemini API key is required');
    }

    this.client = new GoogleGenAI({ apiKey: config.apiKey });
  }

  validateCapabilities(features: string[]): boolean {
    const supported = ['chat', 'streaming'];
    return features.every(f => supported.includes(f));
  }

  async chat(params: ChatParams): Promise<ChatResponse> {
    try {
      // Convert messages to Gemini format
      const contents = params.messages.map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content) }]
      }));

      const response = await this.client.models.generateContent({
        model: params.model || 'gemini-2.0-flash-exp',
        contents,
        config: {
          temperature: params.temperature,
          maxOutputTokens: params.maxTokens || 8192
        }
      });

      const text = response.text || '';

      return {
        id: crypto.randomUUID(),
        model: params.model || 'gemini-2.0-flash-exp',
        content: [{ type: 'text', text }] as ContentBlock[],
        stopReason: 'end_turn',
        usage: {
          inputTokens: response.usageMetadata?.promptTokenCount || 0,
          outputTokens: response.usageMetadata?.candidatesTokenCount || 0
        },
        metadata: {
          provider: 'gemini',
          cost: this.calculateCost(response.usageMetadata || {}),
          latency: 0
        }
      };
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  async *stream(params: ChatParams): AsyncGenerator<StreamChunk> {
    try {
      const contents = params.messages.map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content) }]
      }));

      const stream = await this.client.models.generateContentStream({
        model: params.model || 'gemini-2.0-flash-exp',
        contents,
        config: {
          temperature: params.temperature,
          maxOutputTokens: params.maxTokens || 8192
        }
      });

      for await (const chunk of stream) {
        const text = chunk.text || '';
        if (text) {
          yield {
            type: 'content_block_delta',
            delta: { type: 'text_delta', text }
          } as StreamChunk;
        }
      }
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  private calculateCost(usage: any): number {
    // Gemini pricing varies by model
    const inputTokens = usage.promptTokenCount || 0;
    const outputTokens = usage.candidatesTokenCount || 0;

    // Flash pricing: Free tier or low cost
    const inputCost = (inputTokens / 1_000_000) * 0.075;
    const outputCost = (outputTokens / 1_000_000) * 0.3;
    return inputCost + outputCost;
  }

  private handleError(error: any): ProviderError {
    const providerError = new Error(
      error.message || 'Gemini request failed'
    ) as ProviderError;

    providerError.provider = 'gemini';
    providerError.statusCode = error.status || 500;
    providerError.retryable = error.status >= 500 || error.status === 429;

    return providerError;
  }
}
