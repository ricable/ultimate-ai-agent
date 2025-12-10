/**
 * =============================================================================
 * Edge-Native AI Gateway
 * Unified API gateway with LiteLLM proxy and intelligent routing
 * =============================================================================
 */

import EventEmitter from 'events';

/**
 * AIGateway - Intelligent routing gateway for AI requests
 */
export class AIGateway extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            litellmUrl: process.env.LITELLM_URL || 'http://localhost:4000',
            masterKey: process.env.LITELLM_MASTER_KEY,
            routing: {
                strategy: 'local-first', // 'local-first' | 'cost-optimized' | 'performance'
                localLatencyThreshold: 2000, // ms
                maxRetries: 3,
                timeout: 120000
            },
            semantic_cache: {
                enabled: true,
                similarity_threshold: 0.95,
                ttl: 3600
            },
            ...config
        };

        this.modelStatus = new Map();
        this.requestMetrics = {
            total: 0,
            local: 0,
            cloud: 0,
            cached: 0,
            failed: 0,
            totalLatency: 0,
            totalCost: 0
        };
    }

    /**
     * Initialize the gateway
     */
    async initialize() {
        // Check LiteLLM connectivity
        try {
            const response = await fetch(`${this.config.litellmUrl}/health/liveliness`);
            if (!response.ok) {
                throw new Error('LiteLLM not reachable');
            }
        } catch (error) {
            console.warn('LiteLLM not available:', error.message);
        }

        // Start health checks
        this.startHealthChecks();

        this.emit('initialized');
        return true;
    }

    /**
     * Start periodic health checks
     */
    startHealthChecks() {
        setInterval(async () => {
            await this.checkModelHealth();
        }, 30000);
    }

    /**
     * Check health of all models
     */
    async checkModelHealth() {
        try {
            const response = await fetch(`${this.config.litellmUrl}/model/info`, {
                headers: this.getHeaders()
            });

            if (response.ok) {
                const data = await response.json();
                for (const model of data.data || []) {
                    this.modelStatus.set(model.model_name, {
                        healthy: true,
                        tier: model.model_info?.tier || 'unknown',
                        lastCheck: new Date().toISOString()
                    });
                }
            }
        } catch (error) {
            // Silent fail for health checks
        }

        this.emit('health-check-completed', { models: this.modelStatus.size });
    }

    /**
     * Get request headers
     */
    getHeaders() {
        return {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.masterKey}`
        };
    }

    /**
     * Route and execute chat completion request
     * @param {Object} request - Chat completion request
     * @returns {Promise<Object>} Completion response
     */
    async chatCompletion(request) {
        const startTime = Date.now();
        this.requestMetrics.total++;

        try {
            // Determine best model based on routing strategy
            const model = await this.selectModel(request);

            // Execute request
            const response = await this.executeRequest('/v1/chat/completions', {
                ...request,
                model
            });

            // Update metrics
            const duration = Date.now() - startTime;
            this.updateMetrics(model, duration, response);

            this.emit('completion', {
                model,
                duration,
                tokens: response.usage?.total_tokens
            });

            return response;
        } catch (error) {
            this.requestMetrics.failed++;
            this.emit('error', { error, request });
            throw error;
        }
    }

    /**
     * Execute embedding request
     */
    async createEmbedding(request) {
        return this.executeRequest('/v1/embeddings', {
            ...request,
            model: request.model || 'local-embedding'
        });
    }

    /**
     * Select best model based on routing strategy
     */
    async selectModel(request) {
        const requestedModel = request.model;

        // If specific model requested, use it
        if (requestedModel && !['coder', 'fast', 'best'].includes(requestedModel)) {
            return requestedModel;
        }

        // Map aliases to model groups
        const modelGroup = requestedModel || 'coder';

        switch (this.config.routing.strategy) {
            case 'local-first':
                return this.selectLocalFirst(modelGroup);
            case 'cost-optimized':
                return this.selectCostOptimized(modelGroup);
            case 'performance':
                return this.selectPerformance(modelGroup);
            default:
                return this.selectLocalFirst(modelGroup);
        }
    }

    /**
     * Select local model first, fallback to cloud
     */
    selectLocalFirst(modelGroup) {
        const localModels = {
            coder: 'qwen-coder',
            fast: 'local-general',
            best: 'qwen-coder-14b',
            embedding: 'local-embedding'
        };

        const localModel = localModels[modelGroup];
        const status = this.modelStatus.get(localModel);

        if (status?.healthy) {
            return localModel;
        }

        // Fallback to cloud
        const cloudFallbacks = {
            coder: 'gpt-4o',
            fast: 'gpt-4o-mini',
            best: 'claude-3.5-sonnet',
            embedding: 'text-embedding-ada-002'
        };

        return cloudFallbacks[modelGroup];
    }

    /**
     * Select cheapest available model
     */
    selectCostOptimized(modelGroup) {
        // Cost order: local (free) -> gaianet (cheap) -> cloud
        const costOrder = {
            coder: ['qwen-coder', 'gaianet-coder', 'gpt-4o-mini', 'gpt-4o'],
            fast: ['local-general', 'gpt-4o-mini', 'claude-3-haiku'],
            best: ['qwen-coder-14b', 'gpt-4o-mini', 'gpt-4o'],
            embedding: ['local-embedding', 'text-embedding-ada-002']
        };

        const models = costOrder[modelGroup] || costOrder.coder;

        for (const model of models) {
            const status = this.modelStatus.get(model);
            if (!status || status.healthy) {
                return model;
            }
        }

        return models[models.length - 1];
    }

    /**
     * Select fastest available model
     */
    selectPerformance(modelGroup) {
        // Performance order based on typical latency
        const perfOrder = {
            coder: ['gpt-4o-mini', 'qwen-coder', 'gpt-4o'],
            fast: ['gpt-4o-mini', 'claude-3-haiku', 'local-general'],
            best: ['gpt-4o', 'claude-3.5-sonnet'],
            embedding: ['text-embedding-ada-002', 'local-embedding']
        };

        const models = perfOrder[modelGroup] || perfOrder.coder;

        for (const model of models) {
            const status = this.modelStatus.get(model);
            if (!status || status.healthy) {
                return model;
            }
        }

        return models[0];
    }

    /**
     * Execute HTTP request to LiteLLM
     */
    async executeRequest(path, body) {
        const response = await fetch(`${this.config.litellmUrl}${path}`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(this.config.routing.timeout)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error?.message || `Request failed: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Execute streaming request
     */
    async *streamCompletion(request) {
        const model = await this.selectModel(request);

        const response = await fetch(`${this.config.litellmUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                ...request,
                model,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`Stream request failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') return;

                    try {
                        yield JSON.parse(data);
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
        }
    }

    /**
     * Update metrics
     */
    updateMetrics(model, duration, response) {
        const status = this.modelStatus.get(model);

        if (status?.tier === 'local') {
            this.requestMetrics.local++;
        } else {
            this.requestMetrics.cloud++;
        }

        this.requestMetrics.totalLatency += duration;

        // Estimate cost based on tokens
        const tokens = response.usage?.total_tokens || 0;
        const costPerToken = status?.tier === 'local' ? 0 : 0.00001;
        this.requestMetrics.totalCost += tokens * costPerToken;
    }

    /**
     * Get gateway metrics
     */
    getMetrics() {
        return {
            ...this.requestMetrics,
            averageLatency: this.requestMetrics.total > 0
                ? this.requestMetrics.totalLatency / this.requestMetrics.total
                : 0,
            localPercentage: this.requestMetrics.total > 0
                ? (this.requestMetrics.local / this.requestMetrics.total) * 100
                : 0,
            models: Array.from(this.modelStatus.entries()).map(([name, status]) => ({
                name,
                ...status
            }))
        };
    }

    /**
     * Shutdown
     */
    async shutdown() {
        this.emit('shutdown');
    }
}

export default AIGateway;
