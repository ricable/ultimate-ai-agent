// Base agent class for all specialized RAN agents
// Uses claude-agent-sdk patterns for agent implementation

import Anthropic from '@anthropic-ai/sdk';
import { config } from '../core/config.js';
import { logger } from '../core/logger.js';
import { z } from 'zod';

export interface AgentConfig {
  name: string;
  role: string;
  systemPrompt: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface AgentMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AgentResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  reasoning?: string;
  metadata?: Record<string, unknown>;
}

export abstract class BaseAgent<TInput = unknown, TOutput = unknown> {
  protected client: Anthropic;
  protected conversationHistory: AgentMessage[] = [];
  protected config: AgentConfig;

  constructor(agentConfig: AgentConfig) {
    this.client = new Anthropic({
      apiKey: config.anthropicApiKey,
    });
    this.config = agentConfig;
  }

  /**
   * Main execution method - must be implemented by subclasses
   */
  abstract execute(input: TInput): Promise<AgentResult<TOutput>>;

  /**
   * Validate input using Zod schema
   */
  protected async validateInput<T>(input: unknown, schema: z.ZodSchema<T>): Promise<T> {
    try {
      return schema.parse(input);
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw new Error(`Input validation failed: ${JSON.stringify(error.errors)}`);
      }
      throw error;
    }
  }

  /**
   * Send a message to Claude and get a response
   */
  protected async sendMessage(
    message: string,
    systemPrompt?: string
  ): Promise<string> {
    try {
      this.conversationHistory.push({
        role: 'user',
        content: message,
      });

      const response = await this.client.messages.create({
        model: this.config.model || config.claudeModel,
        max_tokens: this.config.maxTokens || 4096,
        temperature: this.config.temperature || 0.7,
        system: systemPrompt || this.config.systemPrompt,
        messages: this.conversationHistory.map(msg => ({
          role: msg.role,
          content: msg.content,
        })),
      });

      const assistantMessage = response.content[0];
      const responseText = assistantMessage.type === 'text' ? assistantMessage.text : '';

      this.conversationHistory.push({
        role: 'assistant',
        content: responseText,
      });

      logger.debug(`Agent ${this.config.name} received response`, {
        model: response.model,
        stopReason: response.stop_reason,
        usage: response.usage,
      });

      return responseText;
    } catch (error) {
      logger.error(`Error in agent ${this.config.name}:`, error);
      throw error;
    }
  }

  /**
   * Reset conversation history
   */
  protected resetConversation(): void {
    this.conversationHistory = [];
  }

  /**
   * Get agent name
   */
  getName(): string {
    return this.config.name;
  }

  /**
   * Get agent role
   */
  getRole(): string {
    return this.config.role;
  }

  /**
   * Log agent activity
   */
  protected log(level: 'info' | 'warn' | 'error', message: string, metadata?: Record<string, unknown>): void {
    logger[level](`[${this.config.name}] ${message}`, metadata);
  }
}

export default BaseAgent;
