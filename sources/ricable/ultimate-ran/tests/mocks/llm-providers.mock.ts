/**
 * LLM Provider Mocks for Council Testing
 * Simulates DeepSeek, Gemini, and Claude responses
 */

export interface LLMResponse {
  model: string;
  content: string;
  timestamp: number;
  confidence?: number;
}

export interface MockLLMProvider {
  name: string;
  model: string;
  generateResponse: (prompt: string, context?: any) => Promise<LLMResponse>;
  simulateTimeout?: boolean;
  simulateError?: boolean;
}

/**
 * Mock DeepSeek R1 Distill (Analyst role)
 * Focus: Data-driven analysis, pattern recognition
 */
export const mockDeepSeekProvider: MockLLMProvider = {
  name: 'deepseek',
  model: 'deepseek-r1-distill',
  generateResponse: async (prompt: string, context?: any): Promise<LLMResponse> => {
    if (mockDeepSeekProvider.simulateTimeout) {
      await new Promise(resolve => setTimeout(resolve, 35000)); // Exceed 30s timeout
    }

    if (mockDeepSeekProvider.simulateError) {
      throw new Error('DeepSeek API error');
    }

    // Simulate analytical response based on prompt type
    let content = '';
    if (prompt.includes('optimize') || prompt.includes('parameter')) {
      content = `Analytical assessment: Based on PM counter patterns, recommend parameter adjustment with 85% confidence. Historical data shows correlation between proposed change and performance improvement. Vote: APPROVE`;
    } else if (prompt.includes('critique')) {
      content = `Critical analysis: The proposal has merit but lacks consideration of edge cases. Recommend additional validation for high-load scenarios. Vote: APPROVE`;
    } else {
      content = `Data-driven analysis: Current metrics indicate stable baseline. Proposed changes should be tested incrementally. Vote: APPROVE`;
    }

    return {
      model: 'deepseek-r1-distill',
      content,
      timestamp: Date.now(),
      confidence: 0.85
    };
  }
};

/**
 * Mock Gemini 1.5 Pro (Historian role)
 * Focus: Historical context, trend analysis
 */
export const mockGeminiProvider: MockLLMProvider = {
  name: 'gemini',
  model: 'gemini-1.5-pro',
  generateResponse: async (prompt: string, context?: any): Promise<LLMResponse> => {
    if (mockGeminiProvider.simulateTimeout) {
      await new Promise(resolve => setTimeout(resolve, 35000));
    }

    if (mockGeminiProvider.simulateError) {
      throw new Error('Gemini API error');
    }

    let content = '';
    if (prompt.includes('optimize') || prompt.includes('parameter')) {
      content = `Historical perspective: Similar parameter changes in past 6 months showed 73% success rate. Network exhibited stable behavior during previous optimizations of this type. Vote: APPROVE`;
    } else if (prompt.includes('critique')) {
      content = `Historical validation: Past implementations of similar proposals encountered thermal throttling issues. Recommend monitoring cell temperature metrics. Vote: APPROVE`;
    } else {
      content = `Trend analysis: Network performance has been improving steadily over past 30 days. Current trajectory is positive. Vote: APPROVE`;
    }

    return {
      model: 'gemini-1.5-pro',
      content,
      timestamp: Date.now(),
      confidence: 0.78
    };
  }
};

/**
 * Mock Claude 3.7 Sonnet (Strategist role)
 * Focus: Strategic planning, risk assessment
 */
export const mockClaudeProvider: MockLLMProvider = {
  name: 'claude',
  model: 'claude-3-7-sonnet',
  generateResponse: async (prompt: string, context?: any): Promise<LLMResponse> => {
    if (mockClaudeProvider.simulateTimeout) {
      await new Promise(resolve => setTimeout(resolve, 35000));
    }

    if (mockClaudeProvider.simulateError) {
      throw new Error('Claude API error');
    }

    let content = '';
    if (prompt.includes('optimize') || prompt.includes('parameter')) {
      content = `Strategic recommendation: Proposal aligns with long-term network optimization goals. Recommend phased rollout with 3-ROP monitoring to mitigate risks. Consider impact on neighboring cells. Vote: APPROVE`;
    } else if (prompt.includes('critique')) {
      content = `Strategic critique: While technically sound, the proposal may conflict with current capacity expansion plans. Suggest coordination with cluster orchestrator before deployment. Vote: APPROVE`;
    } else {
      content = `Strategic overview: Network is operating within acceptable parameters. Focus should shift to proactive optimization opportunities. Vote: APPROVE`;
    }

    return {
      model: 'claude-3-7-sonnet',
      content,
      timestamp: Date.now(),
      confidence: 0.82
    };
  }
};

/**
 * Mock factory for creating provider instances
 */
export function createMockProvider(name: 'deepseek' | 'gemini' | 'claude'): MockLLMProvider {
  switch (name) {
    case 'deepseek':
      return { ...mockDeepSeekProvider, simulateTimeout: false, simulateError: false };
    case 'gemini':
      return { ...mockGeminiProvider, simulateTimeout: false, simulateError: false };
    case 'claude':
      return { ...mockClaudeProvider, simulateTimeout: false, simulateError: false };
  }
}

/**
 * Reset all mock configurations
 */
export function resetMocks(): void {
  mockDeepSeekProvider.simulateTimeout = false;
  mockDeepSeekProvider.simulateError = false;
  mockGeminiProvider.simulateTimeout = false;
  mockGeminiProvider.simulateError = false;
  mockClaudeProvider.simulateTimeout = false;
  mockClaudeProvider.simulateError = false;
}
