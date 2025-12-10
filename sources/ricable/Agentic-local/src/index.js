/**
 * =============================================================================
 * Edge-Native AI - Main Entry Point
 * Decentralized AI Platform with Kairos, SpinKube, K3s, and Ruvnet Ecosystem
 * =============================================================================
 */

import { initializeRuvnet, DEFAULT_CONFIG } from './ruvnet/index.js';
import { AIGateway } from './gateway/index.js';
import { A2AProtocol, A2AServer } from './a2a/protocol.js';
import { HybridSandbox } from './e2b/sandbox.js';
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';

/**
 * Main Edge-Native AI Application
 */
class EdgeNativeAI {
    constructor() {
        this.app = express();
        this.orchestrator = null;
        this.gateway = null;
        this.a2a = null;
        this.sandbox = null;
    }

    /**
     * Initialize all components
     */
    async initialize() {
        console.log('🚀 Initializing Edge-Native AI Platform...\n');

        // Initialize Ruvnet Orchestrator
        console.log('📦 Loading Ruvnet ecosystem (agentic-flow, claude-flow, agentdb, ruvector)...');
        try {
            this.orchestrator = await initializeRuvnet({
                ...DEFAULT_CONFIG,
                gateway: {
                    url: process.env.LITELLM_URL || 'http://localhost:4000',
                    apiKey: process.env.LITELLM_MASTER_KEY
                }
            });
            console.log('   ✓ Ruvnet Orchestrator initialized\n');
        } catch (error) {
            console.warn('   ⚠ Ruvnet initialization partial:', error.message, '\n');
        }

        // Initialize AI Gateway
        console.log('🌐 Initializing AI Gateway (LiteLLM)...');
        try {
            this.gateway = new AIGateway({
                litellmUrl: process.env.LITELLM_URL || 'http://localhost:4000',
                masterKey: process.env.LITELLM_MASTER_KEY,
                routing: { strategy: 'local-first' }
            });
            await this.gateway.initialize();
            console.log('   ✓ AI Gateway initialized\n');
        } catch (error) {
            console.warn('   ⚠ AI Gateway not available:', error.message, '\n');
        }

        // Initialize A2A Protocol
        console.log('🔗 Initializing A2A Protocol...');
        this.a2a = new A2AProtocol({
            agentId: process.env.AGENT_ID || 'edge-ai-main',
            name: 'Edge-Native-AI',
            description: 'Decentralized AI agent platform with local-first inference',
            baseUrl: process.env.BASE_URL || `http://localhost:${PORT}`,
            capabilities: [
                { name: 'Code Generation', description: 'Generate code using local Qwen Coder models' },
                { name: 'Code Execution', description: 'Execute code securely via E2B or local sandbox' },
                { name: 'Semantic Search', description: 'Search knowledge base using vector embeddings' },
                { name: 'Workflow Orchestration', description: 'Execute SPARC methodology workflows' }
            ],
            discovery: {
                method: process.env.A2A_DISCOVERY || 'static',
                agents: (process.env.A2A_AGENTS || '').split(',').filter(Boolean)
            }
        });

        // Register default task handler
        this.a2a.registerTaskHandler('default', async (task) => {
            if (this.orchestrator) {
                const result = await this.orchestrator.executeTask({
                    type: 'chat',
                    input: task.input,
                    metadata: task.metadata
                });
                return result.result;
            }
            return `Processed: ${task.input}`;
        });

        await this.a2a.initialize();
        console.log('   ✓ A2A Protocol initialized\n');

        // Initialize Hybrid Sandbox
        console.log('🔒 Initializing Hybrid Sandbox (E2B + Local)...');
        try {
            this.sandbox = new HybridSandbox({
                preferLocal: !process.env.E2B_API_KEY,
                cloudForDataScience: !!process.env.E2B_API_KEY,
                e2b: { apiKey: process.env.E2B_API_KEY }
            });
            await this.sandbox.initialize();
            console.log('   ✓ Hybrid Sandbox initialized\n');
        } catch (error) {
            console.warn('   ⚠ Sandbox not available:', error.message, '\n');
        }

        // Setup Express middleware
        this.app.use(cors());
        this.app.use(express.json());

        // Setup routes
        this.setupRoutes();

        console.log('✅ Edge-Native AI Platform initialized successfully!\n');
    }

    /**
     * Setup Express routes
     */
    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                version: '2.0.0',
                components: {
                    orchestrator: !!this.orchestrator,
                    gateway: !!this.gateway,
                    a2a: !!this.a2a,
                    sandbox: !!this.sandbox
                },
                timestamp: new Date().toISOString()
            });
        });

        // A2A Agent Card
        this.app.get('/.well-known/agent.json', (req, res) => {
            res.json(this.a2a.getAgentCard());
        });

        // A2A JSON-RPC endpoint
        this.app.post('/a2a', async (req, res) => {
            try {
                const response = await this.a2a.handleRequest(req.body);
                res.json(response);
            } catch (error) {
                res.status(500).json({
                    jsonrpc: '2.0',
                    id: req.body?.id,
                    error: { code: -32603, message: error.message }
                });
            }
        });

        // Chat endpoint
        this.app.post('/api/chat', async (req, res) => {
            try {
                const { message, conversationId, model } = req.body;

                if (this.gateway) {
                    const response = await this.gateway.chatCompletion({
                        model: model || 'coder',
                        messages: [{ role: 'user', content: message }]
                    });
                    res.json(response);
                } else {
                    res.json({ message: 'Gateway not available', input: message });
                }
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Execute code endpoint
        this.app.post('/api/execute', async (req, res) => {
            try {
                const { code, language } = req.body;

                if (this.sandbox) {
                    const result = await this.sandbox.executeCode(code, { language });
                    res.json(result);
                } else {
                    res.status(503).json({ error: 'Sandbox not available' });
                }
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Agents endpoint
        this.app.get('/api/agents', async (req, res) => {
            if (this.orchestrator) {
                const agents = Array.from(this.orchestrator.agents.values());
                res.json({ agents, total: agents.length });
            } else {
                res.json({ agents: [], total: 0 });
            }
        });

        // Create agent endpoint
        this.app.post('/api/agents', async (req, res) => {
            try {
                if (this.orchestrator) {
                    const agent = await this.orchestrator.createAgent(req.body);
                    res.json(agent);
                } else {
                    res.status(503).json({ error: 'Orchestrator not available' });
                }
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Gateway metrics
        this.app.get('/api/metrics', (req, res) => {
            if (this.gateway) {
                res.json(this.gateway.getMetrics());
            } else {
                res.json({ error: 'Gateway not available' });
            }
        });

        // Root endpoint
        this.app.get('/', (req, res) => {
            res.json({
                name: 'Edge-Native AI',
                version: '2.0.0',
                description: 'Decentralized AI Platform with Kairos, SpinKube, K3s, and Ruvnet Ecosystem',
                endpoints: {
                    health: '/health',
                    agentCard: '/.well-known/agent.json',
                    a2a: '/a2a',
                    chat: '/api/chat',
                    execute: '/api/execute',
                    agents: '/api/agents',
                    metrics: '/api/metrics'
                },
                documentation: 'https://github.com/edge-native-ai/agentic-local'
            });
        });
    }

    /**
     * Start the server
     */
    async start() {
        await this.initialize();

        this.app.listen(PORT, HOST, () => {
            console.log('═══════════════════════════════════════════════════════════════');
            console.log('');
            console.log('   🌟 Edge-Native AI Platform');
            console.log('   ══════════════════════════');
            console.log('');
            console.log(`   📡 Server:      http://${HOST}:${PORT}`);
            console.log(`   📋 Agent Card:  http://${HOST}:${PORT}/.well-known/agent.json`);
            console.log(`   📖 API Docs:    http://${HOST}:${PORT}/`);
            console.log('');
            console.log('   Components:');
            console.log(`     • Ruvnet Orchestrator: ${this.orchestrator ? '✓' : '✗'}`);
            console.log(`     • AI Gateway:         ${this.gateway ? '✓' : '✗'}`);
            console.log(`     • A2A Protocol:       ${this.a2a ? '✓' : '✗'}`);
            console.log(`     • Hybrid Sandbox:     ${this.sandbox ? '✓' : '✗'}`);
            console.log('');
            console.log('═══════════════════════════════════════════════════════════════');
        });
    }

    /**
     * Graceful shutdown
     */
    async shutdown() {
        console.log('\n🛑 Shutting down Edge-Native AI...');

        if (this.orchestrator) await this.orchestrator.shutdown();
        if (this.gateway) await this.gateway.shutdown();
        if (this.a2a) await this.a2a.shutdown();
        if (this.sandbox) await this.sandbox.shutdown();

        console.log('👋 Goodbye!\n');
        process.exit(0);
    }
}

// Create and start application
const app = new EdgeNativeAI();

// Handle graceful shutdown
process.on('SIGINT', () => app.shutdown());
process.on('SIGTERM', () => app.shutdown());

// Start the application
app.start().catch(error => {
    console.error('Failed to start Edge-Native AI:', error);
    process.exit(1);
});

export default EdgeNativeAI;
