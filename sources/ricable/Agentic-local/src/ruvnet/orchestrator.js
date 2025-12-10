/**
 * =============================================================================
 * Ruvnet Orchestrator
 * Unified orchestration layer for the entire Ruvnet ecosystem
 * =============================================================================
 */

import EventEmitter from 'events';
import { AgenticFlowIntegration } from './agentic-flow.js';
import { ClaudeFlowIntegration } from './claude-flow.js';
import { AgentDBIntegration } from './agentdb.js';
import { RuVectorIntegration } from './ruvector.js';

/**
 * RuvnetOrchestrator - Master orchestrator for all Ruvnet components
 */
export class RuvnetOrchestrator extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            agenticFlow: config.agenticFlow || {},
            claudeFlow: config.claudeFlow || {},
            agentdb: config.agentdb || {},
            ruvector: config.ruvector || {},
            gateway: {
                url: process.env.LITELLM_URL || 'http://localhost:4000',
                apiKey: process.env.LITELLM_MASTER_KEY,
                ...config.gateway
            },
            a2a: {
                enabled: true,
                discoveryMethod: 'kubernetes',
                ...config.a2a
            },
            ...config
        };

        // Component instances
        this.agenticFlow = null;
        this.claudeFlow = null;
        this.agentdb = null;
        this.ruvector = null;

        // Orchestration state
        this.initialized = false;
        this.agents = new Map();
        this.workflows = new Map();
    }

    /**
     * Initialize all Ruvnet components
     */
    async initialize() {
        if (this.initialized) {
            return true;
        }

        this.emit('initializing');

        try {
            // Initialize components in parallel where possible
            const initPromises = [];

            // Initialize AgentDB first (dependency for others)
            this.agentdb = new AgentDBIntegration(this.config.agentdb);
            await this.agentdb.initialize();
            this.emit('component-initialized', { component: 'agentdb' });

            // Initialize remaining components in parallel
            this.agenticFlow = new AgenticFlowIntegration(this.config.agenticFlow);
            this.claudeFlow = new ClaudeFlowIntegration(this.config.claudeFlow);
            this.ruvector = new RuVectorIntegration(this.config.ruvector);

            initPromises.push(
                this.agenticFlow.initialize().then(() =>
                    this.emit('component-initialized', { component: 'agenticFlow' })
                ),
                this.claudeFlow.initialize().then(() =>
                    this.emit('component-initialized', { component: 'claudeFlow' })
                ),
                this.ruvector.initialize().then(() =>
                    this.emit('component-initialized', { component: 'ruvector' })
                )
            );

            await Promise.all(initPromises);

            // Wire up event handlers
            this.wireEventHandlers();

            this.initialized = true;
            this.emit('initialized', { components: ['agentdb', 'agenticFlow', 'claudeFlow', 'ruvector'] });

            return true;
        } catch (error) {
            this.emit('initialization-error', { error });
            throw error;
        }
    }

    /**
     * Wire up event handlers between components
     */
    wireEventHandlers() {
        // AgenticFlow events
        this.agenticFlow.on('agent-created', async (data) => {
            // Persist agent to AgentDB
            await this.agentdb.createAgent({
                id: data.agentId,
                name: data.agent.name,
                type: data.agent.type,
                state: data.agent,
                metadata: data.agent.metadata
            });
        });

        this.agenticFlow.on('task-completed', async (data) => {
            // Store task result
            const task = await this.agentdb.createTask({
                agentId: data.agentId,
                type: 'execution',
                input: data.task
            });
            await this.agentdb.updateTaskStatus(task.id, 'completed', data.result);

            // Store in ReasoningBank
            await this.claudeFlow.storeReasoning(`task-${task.id}`, {
                type: 'task-execution',
                agentId: data.agentId,
                task: data.task,
                result: data.result,
                duration: data.duration
            });
        });

        // ClaudeFlow events
        this.claudeFlow.on('workflow-completed', async (data) => {
            // Index workflow outputs in RuVector for retrieval
            for (const [key, output] of Object.entries(data.workflow.outputs)) {
                if (typeof output === 'string') {
                    await this.ruvector.add(
                        `workflow-${data.workflowId}-${key}`,
                        output,
                        {
                            type: 'workflow-output',
                            workflowId: data.workflowId,
                            outputKey: key
                        }
                    );
                }
            }
        });

        // RuVector events
        this.ruvector.on('vector-added', (data) => {
            this.emit('knowledge-indexed', data);
        });
    }

    // =========================================================================
    // UNIFIED AGENT MANAGEMENT
    // =========================================================================

    /**
     * Create an intelligent agent with full Ruvnet integration
     * @param {Object} spec - Agent specification
     */
    async createAgent(spec) {
        await this.ensureInitialized();

        // Create agent in AgenticFlow
        const agent = await this.agenticFlow.createAgent(spec);

        // Index agent capabilities in RuVector for discovery
        if (spec.capabilities && spec.capabilities.length > 0) {
            await this.ruvector.add(
                `agent-${agent.id}-capabilities`,
                spec.capabilities.join('. '),
                {
                    type: 'agent-capability',
                    agentId: agent.id,
                    capabilities: spec.capabilities
                }
            );
        }

        this.agents.set(agent.id, agent);
        this.emit('agent-registered', { agent });

        return agent;
    }

    /**
     * Find agents by capability using semantic search
     * @param {string} capability - Capability to search for
     */
    async findAgentsByCapability(capability) {
        await this.ensureInitialized();

        const results = await this.ruvector.search(capability, {
            k: 10,
            filter: { type: 'agent-capability' }
        });

        return results.map(r => ({
            agentId: r.metadata.agentId,
            capabilities: r.metadata.capabilities,
            relevance: r.score
        }));
    }

    // =========================================================================
    // UNIFIED WORKFLOW MANAGEMENT
    // =========================================================================

    /**
     * Create and execute a SPARC workflow
     * @param {string} templateId - Workflow template
     * @param {Object} context - Workflow context
     */
    async executeWorkflow(templateId, context) {
        await this.ensureInitialized();

        const workflow = await this.claudeFlow.createWorkflow(templateId, context);
        this.workflows.set(workflow.id, workflow);

        // Execute and return results
        const result = await this.claudeFlow.executeWorkflow(workflow.id);

        return result;
    }

    // =========================================================================
    // UNIFIED TASK EXECUTION
    // =========================================================================

    /**
     * Execute a task using the best available agent
     * @param {Object} task - Task specification
     */
    async executeTask(task) {
        await this.ensureInitialized();

        // Find best agent for the task
        let agentId = task.agentId;

        if (!agentId && task.requiredCapabilities) {
            const candidates = await this.findAgentsByCapability(
                task.requiredCapabilities.join(' ')
            );
            if (candidates.length > 0) {
                agentId = candidates[0].agentId;
            }
        }

        if (!agentId) {
            // Create a temporary agent
            const agent = await this.createAgent({
                type: 'task-executor',
                capabilities: task.requiredCapabilities || ['general']
            });
            agentId = agent.id;
        }

        // Execute task
        const result = await this.agenticFlow.executeTask(agentId, task);

        // Store result for retrieval
        if (result.success && task.storeResult !== false) {
            await this.ruvector.add(
                `task-result-${Date.now()}`,
                JSON.stringify(result.result),
                {
                    type: 'task-result',
                    taskType: task.type,
                    agentId
                }
            );
        }

        return result;
    }

    /**
     * Execute a swarm task for collaborative problem solving
     * @param {Object} swarmSpec - Swarm specification
     * @param {Object} task - Task to execute
     */
    async executeSwarmTask(swarmSpec, task) {
        await this.ensureInitialized();

        // Create swarm if not exists
        let swarm = swarmSpec.id ? this.agenticFlow.getSwarm(swarmSpec.id) : null;

        if (!swarm) {
            swarm = await this.agenticFlow.createSwarm(swarmSpec);
        }

        // Execute collaborative task
        return await this.agenticFlow.executeSwarmTask(swarm.id, task);
    }

    // =========================================================================
    // KNOWLEDGE MANAGEMENT
    // =========================================================================

    /**
     * Add knowledge to the vector store
     * @param {string} content - Content to store
     * @param {Object} metadata - Associated metadata
     */
    async addKnowledge(content, metadata = {}) {
        await this.ensureInitialized();

        const id = `knowledge-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        await this.ruvector.add(id, content, {
            type: 'knowledge',
            ...metadata
        });

        return { id };
    }

    /**
     * Search knowledge base
     * @param {string} query - Search query
     * @param {Object} options - Search options
     */
    async searchKnowledge(query, options = {}) {
        await this.ensureInitialized();

        return await this.ruvector.search(query, {
            ...options,
            filter: { type: 'knowledge', ...options.filter }
        });
    }

    /**
     * Retrieve relevant context for a query using RAG
     * @param {string} query - Query text
     * @param {Object} options - RAG options
     */
    async retrieveContext(query, options = {}) {
        await this.ensureInitialized();

        const { k = 5, minScore = 0.7 } = options;

        // Search across all knowledge types
        const results = await this.ruvector.search(query, { k: k * 2 });

        // Filter by score and deduplicate
        const filtered = results
            .filter(r => r.score >= minScore)
            .slice(0, k);

        return {
            context: filtered.map(r => r.metadata.originalText || JSON.stringify(r.metadata)).join('\n\n'),
            sources: filtered.map(r => ({
                id: r.id,
                score: r.score,
                type: r.metadata.type
            }))
        };
    }

    // =========================================================================
    // MEMORY AND STATE MANAGEMENT
    // =========================================================================

    /**
     * Add memory for an agent
     * @param {string} agentId - Agent ID
     * @param {Object} memory - Memory to add
     */
    async addAgentMemory(agentId, memory) {
        await this.ensureInitialized();

        // Add to AgentDB
        const memoryId = await this.agentdb.addMemory(agentId, memory);

        // Index in RuVector for retrieval
        if (memory.content) {
            const content = typeof memory.content === 'string'
                ? memory.content
                : JSON.stringify(memory.content);

            await this.ruvector.add(
                `memory-${agentId}-${memoryId}`,
                content,
                {
                    type: 'agent-memory',
                    agentId,
                    memoryType: memory.type
                }
            );
        }

        return memoryId;
    }

    /**
     * Retrieve relevant memories for an agent
     * @param {string} agentId - Agent ID
     * @param {string} query - Query for relevance
     */
    async retrieveAgentMemories(agentId, query, options = {}) {
        await this.ensureInitialized();

        const { k = 10 } = options;

        // Search vector store for relevant memories
        const results = await this.ruvector.search(query, {
            k,
            filter: { type: 'agent-memory', agentId }
        });

        return results;
    }

    // =========================================================================
    // CONVERSATION MANAGEMENT
    // =========================================================================

    /**
     * Create a conversation with an agent
     * @param {string} agentId - Agent ID
     * @param {string} userId - User ID
     */
    async createConversation(agentId, userId = null) {
        await this.ensureInitialized();
        return await this.agentdb.createConversation(agentId, userId);
    }

    /**
     * Send message in conversation and get response
     * @param {string} conversationId - Conversation ID
     * @param {string} message - User message
     */
    async chat(conversationId, message) {
        await this.ensureInitialized();

        const conversation = await this.agentdb.getConversation(conversationId);
        if (!conversation) {
            throw new Error(`Conversation not found: ${conversationId}`);
        }

        // Add user message
        await this.agentdb.addMessage(conversationId, {
            role: 'user',
            content: message
        });

        // Get agent
        const agent = this.agents.get(conversation.agent_id) ||
            await this.agentdb.getAgent(conversation.agent_id);

        // Retrieve relevant context
        const context = await this.retrieveContext(message, { k: 3 });

        // Execute task with context
        const result = await this.agenticFlow.executeTask(conversation.agent_id, {
            type: 'chat',
            message,
            context: context.context,
            conversationHistory: conversation.messages.slice(-10)
        });

        // Add assistant response
        await this.agentdb.addMessage(conversationId, {
            role: 'assistant',
            content: result.result.content || result.result
        });

        return {
            response: result.result.content || result.result,
            sources: context.sources
        };
    }

    // =========================================================================
    // SYSTEM OPERATIONS
    // =========================================================================

    /**
     * Get system health and metrics
     */
    async getHealth() {
        await this.ensureInitialized();

        const [agenticMetrics, vectorStats] = await Promise.all([
            this.agenticFlow.getMetrics(),
            this.ruvector.stats()
        ]);

        return {
            status: 'healthy',
            components: {
                agenticFlow: {
                    status: 'healthy',
                    metrics: agenticMetrics
                },
                claudeFlow: {
                    status: 'healthy',
                    reasoningBankSize: this.claudeFlow.reasoningBank?.entries.size || 0
                },
                agentdb: {
                    status: 'healthy'
                },
                ruvector: {
                    status: 'healthy',
                    stats: vectorStats
                }
            },
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Ensure orchestrator is initialized
     */
    async ensureInitialized() {
        if (!this.initialized) {
            await this.initialize();
        }
    }

    /**
     * Shutdown all components
     */
    async shutdown() {
        this.emit('shutting-down');

        const shutdownPromises = [];

        if (this.agenticFlow) {
            shutdownPromises.push(this.agenticFlow.shutdown());
        }
        if (this.claudeFlow) {
            shutdownPromises.push(this.claudeFlow.shutdown());
        }
        if (this.agentdb) {
            shutdownPromises.push(this.agentdb.shutdown());
        }
        if (this.ruvector) {
            shutdownPromises.push(this.ruvector.shutdown());
        }

        await Promise.all(shutdownPromises);

        this.initialized = false;
        this.emit('shutdown');
    }
}

export default RuvnetOrchestrator;
