// OpenRouter provider implementation
import axios, { AxiosInstance } from 'axios';
import {
  LLMProvider,
  ChatParams,
  ChatResponse,
  StreamChunk,
  ProviderConfig,
  ProviderError,
  ContentBlock,
  Tool
} from '../types.js';
import { mapModelId } from '../model-mapping.js';

export class OpenRouterProvider implements LLMProvider {
  name = 'openrouter';
  type = 'openrouter' as const;
  supportsStreaming = true;
  supportsTools = true;
  supportsMCP = false; // Requires translation

  private client: AxiosInstance;
  private config: ProviderConfig;

  constructor(config: ProviderConfig) {
    this.config = config;

    if (!config.apiKey) {
      throw new Error('OpenRouter API key is required');
    }

    this.client = axios.create({
      baseURL: config.baseUrl || 'https://openrouter.ai/api/v1',
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'HTTP-Referer': 'https://github.com/ruvnet/agentic-flow',
        'X-Title': 'Agentic Flow Multi-Model Router',
        'Content-Type': 'application/json'
      },
      timeout: config.timeout || 180000
    });
  }

  validateCapabilities(features: string[]): boolean {
    const supported = ['chat', 'streaming', 'tools'];
    return features.every(f => supported.includes(f));
  }

  async chat(params: ChatParams): Promise<ChatResponse> {
    try {
      const requestBody = this.formatRequest(params);

      const response = await this.client.post('/chat/completions', requestBody);

      return this.formatResponse(response.data, params.model);
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  async *stream(params: ChatParams): AsyncGenerator<StreamChunk> {
    try {
      const requestBody = this.formatRequest(params, true);

      const response = await this.client.post('/chat/completions', requestBody, {
        responseType: 'stream'
      });

      for await (const chunk of response.data) {
        const lines = chunk.toString().split('\n').filter((line: string) => line.trim() !== '');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            if (data === '[DONE]') {
              yield { type: 'message_stop' };
              return;
            }

            try {
              const parsed = JSON.parse(data);
              const streamChunk = this.formatStreamChunk(parsed);
              if (streamChunk) {
                yield streamChunk;
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  private formatRequest(params: ChatParams, stream = false): any {
    const messages = params.messages.map(msg => ({
      role: msg.role,
      content: typeof msg.content === 'string'
        ? msg.content
        : msg.content.map(block => {
            if (block.type === 'text') return { type: 'text', text: block.text };
            if (block.type === 'tool_use') return {
              type: 'function',
              function: {
                name: block.name,
                arguments: JSON.stringify(block.input)
              }
            };
            if (block.type === 'tool_result') return {
              type: 'function_result',
              content: block.content,
              is_error: block.is_error
            };
            return block;
          })
    }));

    // Map model ID to OpenRouter format
    const openrouterModel = mapModelId(params.model, 'openrouter');

    const body: any = {
      model: openrouterModel,
      messages,
      temperature: params.temperature ?? 0.7,
      max_tokens: params.maxTokens,
      stream
    };

    // Add tools if provided
    if (params.tools && params.tools.length > 0) {
      body.tools = params.tools.map(tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.input_schema
        }
      }));

      if (params.toolChoice) {
        if (params.toolChoice === 'auto' || params.toolChoice === 'none') {
          body.tool_choice = params.toolChoice;
        } else if (typeof params.toolChoice === 'object') {
          body.tool_choice = {
            type: 'function',
            function: { name: params.toolChoice.name }
          };
        }
      }
    }

    // Add OpenRouter-specific preferences
    if (this.config.preferences) {
      if (this.config.preferences.requireParameters) {
        body.require_parameters = true;
      }
      if (this.config.preferences.dataCollection) {
        body['X-Data-Collection'] = this.config.preferences.dataCollection;
      }
      if (this.config.preferences.order) {
        body.provider = {
          order: this.config.preferences.order
        };
      }
    }

    return body;
  }

  private formatResponse(data: any, model: string): ChatResponse {
    const choice = data.choices[0];
    const message = choice.message;

    const content: ContentBlock[] = [];

    // Handle text content
    if (message.content) {
      content.push({
        type: 'text',
        text: message.content
      });
    }

    // Handle tool calls
    if (message.tool_calls) {
      for (const toolCall of message.tool_calls) {
        content.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.function.name,
          input: JSON.parse(toolCall.function.arguments)
        });
      }
    }

    const stopReason = this.mapFinishReason(choice.finish_reason);

    return {
      id: data.id,
      model: data.model || model,
      content,
      stopReason,
      usage: {
        inputTokens: data.usage?.prompt_tokens || 0,
        outputTokens: data.usage?.completion_tokens || 0
      },
      metadata: {
        provider: 'openrouter',
        cost: this.calculateCost(data.usage),
        latency: 0 // Will be set by router
      }
    };
  }

  private formatStreamChunk(data: any): StreamChunk | null {
    const choice = data.choices?.[0];
    if (!choice) return null;

    const delta = choice.delta;

    if (delta?.content) {
      return {
        type: 'content_block_delta',
        delta: {
          type: 'text_delta',
          text: delta.content
        }
      };
    }

    if (delta?.tool_calls) {
      const toolCall = delta.tool_calls[0];
      return {
        type: 'content_block_delta',
        delta: {
          type: 'input_json_delta',
          partial_json: toolCall.function?.arguments || ''
        }
      };
    }

    if (choice.finish_reason) {
      return {
        type: 'message_stop',
        usage: data.usage ? {
          inputTokens: data.usage.prompt_tokens || 0,
          outputTokens: data.usage.completion_tokens || 0
        } : undefined
      };
    }

    return null;
  }

  private mapFinishReason(reason: string): ChatResponse['stopReason'] {
    switch (reason) {
      case 'stop': return 'end_turn';
      case 'length': return 'max_tokens';
      case 'tool_calls': return 'tool_use';
      default: return 'end_turn';
    }
  }

  private calculateCost(usage?: { prompt_tokens?: number; completion_tokens?: number }): number {
    if (!usage) return 0;

    // OpenRouter pricing varies by model
    // Using average pricing: $0.01/1K input, $0.03/1K output
    const inputCost = (usage.prompt_tokens || 0) * 0.00001;
    const outputCost = (usage.completion_tokens || 0) * 0.00003;

    return inputCost + outputCost;
  }

  private handleError(error: any): ProviderError {
    const providerError = new Error(
      error.response?.data?.error?.message || error.message || 'OpenRouter request failed'
    ) as ProviderError;

    providerError.provider = 'openrouter';
    providerError.statusCode = error.response?.status;
    providerError.retryable = error.response?.status >= 500 || error.code === 'ECONNABORTED';

    return providerError;
  }
}
