/**
 * LLM Router for Council
 * Routes requests to appropriate LLM providers with abstraction
 */

export interface LLMProvider {
  name: string;
  model: string;
  generateResponse(prompt: string, context?: any): Promise<any>;
  simulateTimeout?: boolean;
  simulateError?: boolean;
}

export interface LLMRouterConfig {
  deepseek?: LLMProvider;
  gemini?: LLMProvider;
  claude?: LLMProvider;
}

/**
 * LLM Router
 * Abstracts provider-specific implementations
 */
export class LLMRouter {
  private providers: Map<string, LLMProvider>;

  constructor(config: LLMRouterConfig) {
    this.providers = new Map();

    if (config.deepseek) {
      this.providers.set('deepseek', config.deepseek);
    }
    if (config.gemini) {
      this.providers.set('gemini', config.gemini);
    }
    if (config.claude) {
      this.providers.set('claude', config.claude);
    }
  }

  /**
   * Route request to appropriate provider
   */
  async route(
    provider: 'deepseek' | 'gemini' | 'claude',
    prompt: string,
    context?: any,
    timeoutMs: number = 30000
  ): Promise<any> {
    const llmProvider = this.providers.get(provider);

    if (!llmProvider) {
      throw new Error(`Provider ${provider} not configured`);
    }

    // Implement timeout
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('LLM request timeout')), timeoutMs);
    });

    try {
      const response = await Promise.race([
        llmProvider.generateResponse(prompt, context),
        timeoutPromise
      ]);

      return response;
    } catch (error) {
      if (error instanceof Error && error.message === 'LLM request timeout') {
        throw error;
      }
      throw new Error(`Provider ${provider} failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get provider by name
   */
  getProvider(name: string): LLMProvider | undefined {
    return this.providers.get(name);
  }

  /**
   * Check if provider is available
   */
  hasProvider(name: string): boolean {
    return this.providers.has(name);
  }
}
