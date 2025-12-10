/**
 * =============================================================================
 * Agent-to-Agent (A2A) Protocol Implementation
 * Google A2A protocol for cross-agent communication and discovery
 * =============================================================================
 */

import EventEmitter from 'events';
import crypto from 'crypto';

/**
 * A2AProtocol - Implementation of Google's Agent-to-Agent protocol
 */
export class A2AProtocol extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            agentId: config.agentId || this.generateAgentId(),
            name: config.name || 'Edge-AI-Agent',
            description: config.description || 'Edge-Native AI Agent',
            version: config.version || '1.0.0',
            baseUrl: config.baseUrl || 'http://localhost:8080',
            capabilities: config.capabilities || [],
            authentication: {
                type: config.authentication?.type || 'none', // 'none' | 'api_key' | 'oauth2'
                ...config.authentication
            },
            discovery: {
                method: config.discovery?.method || 'kubernetes', // 'kubernetes' | 'dns' | 'static'
                refreshInterval: config.discovery?.refreshInterval || 30000,
                ...config.discovery
            },
            ...config
        };

        this.knownAgents = new Map();
        this.pendingTasks = new Map();
        this.taskHandlers = new Map();
        this.discoveryTimer = null;
    }

    /**
     * Generate unique agent ID
     */
    generateAgentId() {
        return `agent-${crypto.randomUUID()}`;
    }

    /**
     * Initialize A2A protocol
     */
    async initialize() {
        // Start agent discovery
        if (this.config.discovery.method !== 'none') {
            await this.startDiscovery();
        }

        this.emit('initialized', { agentId: this.config.agentId });
        return true;
    }

    /**
     * Get Agent Card (/.well-known/agent.json)
     * Standard A2A agent discovery document
     */
    getAgentCard() {
        return {
            // Required fields
            name: this.config.name,
            description: this.config.description,
            url: this.config.baseUrl,
            version: this.config.version,

            // Capabilities
            capabilities: {
                streaming: true,
                pushNotifications: false,
                stateTransitionHistory: true
            },

            // Authentication
            authentication: this.config.authentication.type !== 'none' ? {
                schemes: [this.config.authentication.type]
            } : undefined,

            // Default input/output modes
            defaultInputModes: ['text'],
            defaultOutputModes: ['text'],

            // Skills/Tasks this agent can perform
            skills: this.config.capabilities.map(cap => ({
                id: cap.id || cap.name.toLowerCase().replace(/\s+/g, '-'),
                name: cap.name,
                description: cap.description,
                inputSchema: cap.inputSchema || { type: 'object' },
                outputSchema: cap.outputSchema || { type: 'object' }
            })),

            // Provider information
            provider: {
                organization: 'Edge-Native-AI',
                url: 'https://edge-ai.local'
            }
        };
    }

    /**
     * Register a task handler
     * @param {string} taskType - Type of task to handle
     * @param {Function} handler - Handler function
     */
    registerTaskHandler(taskType, handler) {
        this.taskHandlers.set(taskType, handler);
        this.emit('handler-registered', { taskType });
    }

    /**
     * Send task to another agent
     * @param {string} agentUrl - Target agent URL
     * @param {Object} task - Task to send
     */
    async sendTask(agentUrl, task) {
        const taskId = task.id || `task-${crypto.randomUUID()}`;

        const request = {
            jsonrpc: '2.0',
            method: 'tasks/send',
            id: taskId,
            params: {
                id: taskId,
                message: {
                    role: 'user',
                    parts: [
                        {
                            type: 'text',
                            text: typeof task.input === 'string' ? task.input : JSON.stringify(task.input)
                        }
                    ]
                },
                metadata: task.metadata || {}
            }
        };

        try {
            const response = await this.makeRequest(agentUrl, '/a2a', request);

            this.pendingTasks.set(taskId, {
                id: taskId,
                agentUrl,
                status: 'pending',
                createdAt: new Date().toISOString()
            });

            this.emit('task-sent', { taskId, agentUrl });
            return { taskId, response };
        } catch (error) {
            this.emit('task-send-failed', { taskId, agentUrl, error });
            throw error;
        }
    }

    /**
     * Handle incoming A2A request
     * @param {Object} request - JSON-RPC request
     */
    async handleRequest(request) {
        if (request.jsonrpc !== '2.0') {
            return this.createErrorResponse(request.id, -32600, 'Invalid JSON-RPC version');
        }

        switch (request.method) {
            case 'tasks/send':
                return this.handleTaskSend(request);
            case 'tasks/get':
                return this.handleTaskGet(request);
            case 'tasks/cancel':
                return this.handleTaskCancel(request);
            case 'tasks/sendSubscribe':
                return this.handleTaskSubscribe(request);
            default:
                return this.createErrorResponse(request.id, -32601, `Unknown method: ${request.method}`);
        }
    }

    /**
     * Handle tasks/send request
     */
    async handleTaskSend(request) {
        const { id, message, metadata } = request.params;

        this.emit('task-received', { taskId: id, message });

        try {
            // Extract text from message parts
            const textParts = message.parts
                .filter(p => p.type === 'text')
                .map(p => p.text)
                .join('\n');

            // Find appropriate handler
            let result;
            const taskType = metadata?.taskType || 'default';
            const handler = this.taskHandlers.get(taskType) || this.taskHandlers.get('default');

            if (handler) {
                result = await handler({
                    taskId: id,
                    input: textParts,
                    metadata
                });
            } else {
                return this.createErrorResponse(request.id, -32603, 'No handler for task type');
            }

            // Create response
            return {
                jsonrpc: '2.0',
                id: request.id,
                result: {
                    id,
                    status: {
                        state: 'completed',
                        timestamp: new Date().toISOString()
                    },
                    artifacts: [
                        {
                            parts: [
                                {
                                    type: 'text',
                                    text: typeof result === 'string' ? result : JSON.stringify(result)
                                }
                            ]
                        }
                    ]
                }
            };
        } catch (error) {
            return this.createErrorResponse(request.id, -32603, error.message);
        }
    }

    /**
     * Handle tasks/get request
     */
    async handleTaskGet(request) {
        const { id } = request.params;
        const task = this.pendingTasks.get(id);

        if (!task) {
            return this.createErrorResponse(request.id, -32602, 'Task not found');
        }

        return {
            jsonrpc: '2.0',
            id: request.id,
            result: task
        };
    }

    /**
     * Handle tasks/cancel request
     */
    async handleTaskCancel(request) {
        const { id } = request.params;

        if (this.pendingTasks.has(id)) {
            this.pendingTasks.delete(id);
            this.emit('task-cancelled', { taskId: id });
        }

        return {
            jsonrpc: '2.0',
            id: request.id,
            result: { success: true }
        };
    }

    /**
     * Handle tasks/sendSubscribe (streaming)
     */
    async handleTaskSubscribe(request) {
        // Return SSE stream setup info
        return {
            jsonrpc: '2.0',
            id: request.id,
            result: {
                streamUrl: `${this.config.baseUrl}/a2a/stream/${request.params.id}`,
                protocol: 'sse'
            }
        };
    }

    /**
     * Create JSON-RPC error response
     */
    createErrorResponse(id, code, message) {
        return {
            jsonrpc: '2.0',
            id,
            error: {
                code,
                message
            }
        };
    }

    /**
     * Make HTTP request to another agent
     */
    async makeRequest(agentUrl, path, body) {
        const url = `${agentUrl}${path}`;

        const headers = {
            'Content-Type': 'application/json'
        };

        // Add authentication if configured
        if (this.config.authentication.type === 'api_key') {
            headers['Authorization'] = `Bearer ${this.config.authentication.apiKey}`;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            throw new Error(`A2A request failed: ${response.status}`);
        }

        return response.json();
    }

    // =========================================================================
    // AGENT DISCOVERY
    // =========================================================================

    /**
     * Start agent discovery
     */
    async startDiscovery() {
        await this.discoverAgents();

        this.discoveryTimer = setInterval(
            () => this.discoverAgents(),
            this.config.discovery.refreshInterval
        );
    }

    /**
     * Discover agents based on configured method
     */
    async discoverAgents() {
        switch (this.config.discovery.method) {
            case 'kubernetes':
                await this.discoverKubernetesAgents();
                break;
            case 'dns':
                await this.discoverDnsAgents();
                break;
            case 'static':
                await this.discoverStaticAgents();
                break;
        }

        this.emit('discovery-completed', { agentCount: this.knownAgents.size });
    }

    /**
     * Discover agents via Kubernetes service discovery
     */
    async discoverKubernetesAgents() {
        try {
            // Use Kubernetes DNS to find agents in the cluster
            const namespace = process.env.POD_NAMESPACE || 'edge-ai-agents';
            const labelSelector = 'edge-ai.io/type=agent';

            // In K8s, we'd query the API or use DNS
            // For now, use environment variable or config
            const endpoints = this.config.discovery.endpoints || [];

            for (const endpoint of endpoints) {
                await this.probeAgent(endpoint);
            }
        } catch (error) {
            console.warn('Kubernetes discovery failed:', error.message);
        }
    }

    /**
     * Discover agents via DNS SRV records
     */
    async discoverDnsAgents() {
        // DNS-based discovery for edge environments
        const domain = this.config.discovery.domain || 'edge-ai.local';

        try {
            const dns = await import('dns').then(m => m.promises);
            const records = await dns.resolveSrv(`_a2a._tcp.${domain}`);

            for (const record of records) {
                const url = `http://${record.name}:${record.port}`;
                await this.probeAgent(url);
            }
        } catch (error) {
            console.warn('DNS discovery failed:', error.message);
        }
    }

    /**
     * Use static agent list
     */
    async discoverStaticAgents() {
        const agents = this.config.discovery.agents || [];

        for (const agentUrl of agents) {
            await this.probeAgent(agentUrl);
        }
    }

    /**
     * Probe an agent to get its capabilities
     */
    async probeAgent(agentUrl) {
        try {
            const response = await fetch(`${agentUrl}/.well-known/agent.json`);

            if (!response.ok) {
                throw new Error(`Agent probe failed: ${response.status}`);
            }

            const agentCard = await response.json();

            this.knownAgents.set(agentUrl, {
                url: agentUrl,
                card: agentCard,
                lastSeen: new Date().toISOString(),
                healthy: true
            });

            this.emit('agent-discovered', { url: agentUrl, card: agentCard });
        } catch (error) {
            // Mark as unhealthy if previously known
            if (this.knownAgents.has(agentUrl)) {
                const agent = this.knownAgents.get(agentUrl);
                agent.healthy = false;
                agent.lastError = error.message;
            }
        }
    }

    /**
     * Find agents by capability
     */
    findAgentsByCapability(capability) {
        const results = [];

        for (const [url, agent] of this.knownAgents.entries()) {
            if (!agent.healthy) continue;

            const hasCapability = agent.card.skills?.some(skill =>
                skill.name.toLowerCase().includes(capability.toLowerCase()) ||
                skill.description?.toLowerCase().includes(capability.toLowerCase())
            );

            if (hasCapability) {
                results.push({
                    url,
                    card: agent.card,
                    lastSeen: agent.lastSeen
                });
            }
        }

        return results;
    }

    /**
     * Get all known agents
     */
    getKnownAgents() {
        return Array.from(this.knownAgents.entries()).map(([url, agent]) => ({
            url,
            name: agent.card.name,
            healthy: agent.healthy,
            lastSeen: agent.lastSeen,
            capabilities: agent.card.skills?.map(s => s.name) || []
        }));
    }

    /**
     * Shutdown
     */
    async shutdown() {
        if (this.discoveryTimer) {
            clearInterval(this.discoveryTimer);
        }

        this.knownAgents.clear();
        this.pendingTasks.clear();
        this.emit('shutdown');
    }
}

/**
 * A2AServer - Express middleware for A2A protocol
 */
export class A2AServer {
    constructor(protocol) {
        this.protocol = protocol;
    }

    /**
     * Get Express router for A2A endpoints
     */
    getRouter() {
        const express = require('express');
        const router = express.Router();

        // Agent Card endpoint
        router.get('/.well-known/agent.json', (req, res) => {
            res.json(this.protocol.getAgentCard());
        });

        // A2A JSON-RPC endpoint
        router.post('/a2a', async (req, res) => {
            try {
                const response = await this.protocol.handleRequest(req.body);
                res.json(response);
            } catch (error) {
                res.status(500).json({
                    jsonrpc: '2.0',
                    id: req.body?.id,
                    error: {
                        code: -32603,
                        message: error.message
                    }
                });
            }
        });

        // SSE streaming endpoint
        router.get('/a2a/stream/:taskId', (req, res) => {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            const taskId = req.params.taskId;

            // Send events as task progresses
            const sendEvent = (event, data) => {
                res.write(`event: ${event}\n`);
                res.write(`data: ${JSON.stringify(data)}\n\n`);
            };

            // Initial acknowledgment
            sendEvent('connected', { taskId });

            // Cleanup on close
            req.on('close', () => {
                this.protocol.emit('stream-closed', { taskId });
            });
        });

        return router;
    }
}

export default A2AProtocol;
