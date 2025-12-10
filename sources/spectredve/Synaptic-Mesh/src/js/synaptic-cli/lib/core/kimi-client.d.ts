/**
 * Real Kimi-K2 API Client with 128k context window support
 *
 * Implements:
 * - Moonshot AI API integration
 * - OpenRouter API integration
 * - Real-time streaming responses
 * - Tool calling functionality
 * - Rate limiting and retry logic
 * - Error handling and monitoring
 */
import { EventEmitter } from 'events';
export interface KimiConfig {
    provider: 'moonshot' | 'openrouter';
    apiKey: string;
    modelVersion?: string;
    endpoint?: string;
    maxTokens?: number;
    temperature?: number;
    timeout?: number;
    retryAttempts?: number;
    rateLimitDelay?: number;
    features?: {
        multiModal?: boolean;
        codeGeneration?: boolean;
        documentAnalysis?: boolean;
        imageProcessing?: boolean;
        streaming?: boolean;
        toolCalling?: boolean;
    };
}
export interface KimiResponse {
    id: string;
    choices: Array<{
        message: {
            role: string;
            content: string;
            tool_calls?: Array<{
                id: string;
                type: string;
                function: {
                    name: string;
                    arguments: string;
                };
            }>;
        };
        finish_reason: string;
    }>;
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
    model: string;
}
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
    tool_call_id?: string;
    name?: string;
}
export interface ToolDefinition {
    type: 'function';
    function: {
        name: string;
        description: string;
        parameters: {
            type: string;
            properties: Record<string, any>;
            required?: string[];
        };
    };
}
/**
 * Real Kimi-K2 API Client
 */
export declare class KimiClient extends EventEmitter {
    private config;
    private connected;
    private sessionId?;
    private conversationHistory;
    private rateLimitInfo;
    constructor(config: KimiConfig);
    private validateConfig;
    private getDefaultEndpoint;
    connect(): Promise<void>;
    chat(message: string, options?: {
        systemPrompt?: string;
        tools?: ToolDefinition[];
        temperature?: number;
        maxTokens?: number;
        streaming?: boolean;
    }): Promise<string>;
    generateCode(prompt: string, language?: string, options?: {
        optimize?: boolean;
        includeTests?: boolean;
        includeComments?: boolean;
    }): Promise<{
        code: string;
        explanation: string;
        tests?: string;
    }>;
    analyzeFile(filePath: string, fileContent: string, analysisType?: 'quality' | 'security' | 'performance'): Promise<{
        summary: string;
        complexity: number;
        maintainabilityIndex: number;
        suggestions: string[];
        issues: Array<{
            severity: 'error' | 'warning' | 'info';
            line?: number;
            message: string;
            rule?: string;
        }>;
        metrics: {
            linesOfCode: number;
            cyclomaticComplexity: number;
            cognitiveComplexity: number;
            technicalDebt: string;
        };
    }>;
    processDocument(content: string, task: 'summarize' | 'extract' | 'translate' | 'analyze', options?: any): Promise<string>;
    private makeRequest;
    private buildHeaders;
    private updateRateLimitInfo;
    private sleep;
    getStatus(): {
        connected: boolean;
        sessionId?: string;
        provider: string;
        model: string;
        endpoint: string;
        features: any;
        rateLimitInfo: any;
        conversationLength: number;
    };
    clearHistory(): void;
    disconnect(): void;
    callTool(toolName: string, parameters: any): Promise<any>;
    streamChat(message: string, onChunk: (chunk: string) => void, options?: any): Promise<void>;
}
export default KimiClient;
//# sourceMappingURL=kimi-client.d.ts.map