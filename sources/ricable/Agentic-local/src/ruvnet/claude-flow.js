/**
 * =============================================================================
 * Claude Flow Integration
 * Enterprise workflow orchestration with ReasoningBank and SPARC methodology
 * =============================================================================
 */

import EventEmitter from 'events';
import fs from 'fs/promises';
import path from 'path';

/**
 * ClaudeFlowIntegration - Advanced workflow orchestration
 */
export class ClaudeFlowIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            reasoningBank: {
                enabled: true,
                maxSize: 10000,
                persistPath: './data/reasoning-bank',
                compressionEnabled: true
            },
            sparc: {
                enabled: true,
                methodology: 'sparc2',
                phases: ['specification', 'pseudocode', 'architecture', 'refinement', 'completion']
            },
            workflows: {
                maxConcurrent: 5,
                timeout: 300000,
                retryAttempts: 3
            },
            ...config
        };

        this.claudeFlow = null;
        this.reasoningBank = null;
        this.workflows = new Map();
        this.templates = new Map();
    }

    /**
     * Initialize Claude Flow integration
     */
    async initialize() {
        try {
            // Dynamic import of claude-flow
            const claudeFlowModule = await import('claude-flow');
            this.claudeFlow = claudeFlowModule.default || claudeFlowModule;

            // Initialize ReasoningBank
            if (this.config.reasoningBank.enabled) {
                await this.initializeReasoningBank();
            }

            // Load workflow templates
            await this.loadWorkflowTemplates();

            this.emit('initialized', { config: this.config });
            return true;
        } catch (error) {
            this.emit('error', { phase: 'initialization', error });
            throw error;
        }
    }

    /**
     * Initialize the ReasoningBank for persistent reasoning storage
     */
    async initializeReasoningBank() {
        this.reasoningBank = {
            entries: new Map(),
            metadata: {
                created: new Date().toISOString(),
                version: '1.0',
                totalEntries: 0
            }
        };

        // Load persisted reasoning if exists
        try {
            const persistPath = this.config.reasoningBank.persistPath;
            await fs.mkdir(persistPath, { recursive: true });

            const indexPath = path.join(persistPath, 'index.json');
            const indexExists = await fs.access(indexPath).then(() => true).catch(() => false);

            if (indexExists) {
                const indexData = await fs.readFile(indexPath, 'utf-8');
                const index = JSON.parse(indexData);
                this.reasoningBank.metadata = index.metadata;

                // Load entries lazily
                for (const entryId of index.entries) {
                    this.reasoningBank.entries.set(entryId, null); // Lazy load
                }
            }
        } catch (error) {
            console.warn('Could not load ReasoningBank:', error.message);
        }

        this.emit('reasoning-bank-initialized', { metadata: this.reasoningBank.metadata });
    }

    /**
     * Store reasoning in the ReasoningBank
     * @param {string} key - Unique key for the reasoning
     * @param {Object} reasoning - Reasoning data to store
     */
    async storeReasoning(key, reasoning) {
        const entry = {
            id: key,
            timestamp: new Date().toISOString(),
            reasoning: reasoning,
            metadata: {
                type: reasoning.type || 'general',
                confidence: reasoning.confidence || 1.0,
                source: reasoning.source || 'unknown',
                tags: reasoning.tags || []
            }
        };

        this.reasoningBank.entries.set(key, entry);
        this.reasoningBank.metadata.totalEntries++;

        // Persist if enabled
        if (this.config.reasoningBank.persistPath) {
            await this.persistReasoningEntry(key, entry);
        }

        // Enforce max size
        if (this.reasoningBank.entries.size > this.config.reasoningBank.maxSize) {
            await this.pruneReasoningBank();
        }

        this.emit('reasoning-stored', { key, entry });
        return entry;
    }

    /**
     * Retrieve reasoning from the ReasoningBank
     * @param {string} key - Key to retrieve
     */
    async getReasoning(key) {
        let entry = this.reasoningBank.entries.get(key);

        // Lazy load from disk if needed
        if (entry === null && this.config.reasoningBank.persistPath) {
            entry = await this.loadReasoningEntry(key);
            this.reasoningBank.entries.set(key, entry);
        }

        return entry;
    }

    /**
     * Search reasoning by tags or content
     * @param {Object} query - Search query
     */
    async searchReasoning(query) {
        const results = [];

        for (const [key, entry] of this.reasoningBank.entries) {
            if (entry === null) continue;

            let matches = true;

            if (query.tags && query.tags.length > 0) {
                matches = query.tags.some(tag =>
                    entry.metadata.tags.includes(tag)
                );
            }

            if (matches && query.type) {
                matches = entry.metadata.type === query.type;
            }

            if (matches && query.minConfidence) {
                matches = entry.metadata.confidence >= query.minConfidence;
            }

            if (matches) {
                results.push(entry);
            }
        }

        return results.sort((a, b) =>
            new Date(b.timestamp) - new Date(a.timestamp)
        );
    }

    /**
     * Persist reasoning entry to disk
     */
    async persistReasoningEntry(key, entry) {
        const persistPath = this.config.reasoningBank.persistPath;
        const entryPath = path.join(persistPath, `${key}.json`);

        await fs.writeFile(entryPath, JSON.stringify(entry, null, 2));

        // Update index
        const indexPath = path.join(persistPath, 'index.json');
        const index = {
            metadata: this.reasoningBank.metadata,
            entries: Array.from(this.reasoningBank.entries.keys())
        };
        await fs.writeFile(indexPath, JSON.stringify(index, null, 2));
    }

    /**
     * Load reasoning entry from disk
     */
    async loadReasoningEntry(key) {
        const persistPath = this.config.reasoningBank.persistPath;
        const entryPath = path.join(persistPath, `${key}.json`);

        try {
            const data = await fs.readFile(entryPath, 'utf-8');
            return JSON.parse(data);
        } catch (error) {
            return null;
        }
    }

    /**
     * Prune old entries from ReasoningBank
     */
    async pruneReasoningBank() {
        const entries = Array.from(this.reasoningBank.entries.entries())
            .filter(([_, entry]) => entry !== null)
            .sort((a, b) => new Date(a[1].timestamp) - new Date(b[1].timestamp));

        const toRemove = entries.slice(0, entries.length - this.config.reasoningBank.maxSize);

        for (const [key, _] of toRemove) {
            this.reasoningBank.entries.delete(key);
            // Remove from disk
            if (this.config.reasoningBank.persistPath) {
                const entryPath = path.join(this.config.reasoningBank.persistPath, `${key}.json`);
                await fs.unlink(entryPath).catch(() => { });
            }
        }

        this.emit('reasoning-bank-pruned', { removed: toRemove.length });
    }

    /**
     * Load workflow templates
     */
    async loadWorkflowTemplates() {
        // Built-in SPARC workflow templates
        this.templates.set('sparc-development', {
            name: 'SPARC Development Workflow',
            phases: [
                {
                    name: 'specification',
                    prompt: 'Analyze the requirements and create a detailed specification.',
                    outputs: ['requirements', 'constraints', 'acceptance_criteria']
                },
                {
                    name: 'pseudocode',
                    prompt: 'Design the solution using pseudocode and flowcharts.',
                    outputs: ['pseudocode', 'data_structures', 'algorithms']
                },
                {
                    name: 'architecture',
                    prompt: 'Define the system architecture and component design.',
                    outputs: ['architecture_diagram', 'components', 'interfaces']
                },
                {
                    name: 'refinement',
                    prompt: 'Implement the solution with iterative refinement.',
                    outputs: ['code', 'tests', 'documentation']
                },
                {
                    name: 'completion',
                    prompt: 'Finalize, review, and validate the implementation.',
                    outputs: ['final_code', 'test_results', 'deployment_ready']
                }
            ]
        });

        this.templates.set('code-review', {
            name: 'Code Review Workflow',
            phases: [
                { name: 'analysis', prompt: 'Analyze the code for patterns and issues.' },
                { name: 'security', prompt: 'Check for security vulnerabilities.' },
                { name: 'performance', prompt: 'Evaluate performance implications.' },
                { name: 'suggestions', prompt: 'Provide improvement suggestions.' }
            ]
        });

        this.templates.set('research', {
            name: 'Research Workflow',
            phases: [
                { name: 'gather', prompt: 'Gather relevant information and sources.' },
                { name: 'analyze', prompt: 'Analyze and synthesize findings.' },
                { name: 'conclude', prompt: 'Draw conclusions and recommendations.' }
            ]
        });
    }

    /**
     * Create a new workflow instance
     * @param {string} templateId - Template to use
     * @param {Object} context - Workflow context
     */
    async createWorkflow(templateId, context = {}) {
        const template = this.templates.get(templateId);
        if (!template) {
            throw new Error(`Unknown workflow template: ${templateId}`);
        }

        const workflowId = `workflow-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        const workflow = {
            id: workflowId,
            template: templateId,
            name: template.name,
            context: context,
            phases: template.phases.map(phase => ({
                ...phase,
                status: 'pending',
                result: null,
                startedAt: null,
                completedAt: null
            })),
            currentPhase: 0,
            status: 'created',
            createdAt: new Date().toISOString(),
            completedAt: null,
            outputs: {}
        };

        this.workflows.set(workflowId, workflow);
        this.emit('workflow-created', { workflowId, workflow });

        return workflow;
    }

    /**
     * Execute a workflow
     * @param {string} workflowId - Workflow to execute
     */
    async executeWorkflow(workflowId) {
        const workflow = this.workflows.get(workflowId);
        if (!workflow) {
            throw new Error(`Workflow not found: ${workflowId}`);
        }

        workflow.status = 'running';
        this.emit('workflow-started', { workflowId });

        try {
            for (let i = workflow.currentPhase; i < workflow.phases.length; i++) {
                const phase = workflow.phases[i];
                workflow.currentPhase = i;

                await this.executePhase(workflow, phase);

                // Store reasoning for this phase
                await this.storeReasoning(`${workflowId}-${phase.name}`, {
                    type: 'workflow-phase',
                    workflowId,
                    phase: phase.name,
                    result: phase.result,
                    context: workflow.context
                });
            }

            workflow.status = 'completed';
            workflow.completedAt = new Date().toISOString();

            this.emit('workflow-completed', { workflowId, workflow });
            return workflow;
        } catch (error) {
            workflow.status = 'failed';
            this.emit('workflow-failed', { workflowId, error });
            throw error;
        }
    }

    /**
     * Execute a single workflow phase
     */
    async executePhase(workflow, phase) {
        phase.status = 'running';
        phase.startedAt = new Date().toISOString();

        this.emit('phase-started', { workflowId: workflow.id, phase: phase.name });

        try {
            // Build context from previous phases
            const phaseContext = {
                workflow: workflow.context,
                previousOutputs: workflow.outputs,
                currentPhase: phase.name
            };

            // Execute through LLM gateway
            const result = await this.callGateway({
                model: workflow.context.model || 'qwen-coder',
                messages: [
                    {
                        role: 'system',
                        content: `You are executing the "${phase.name}" phase of the ${workflow.name}. ${phase.prompt}`
                    },
                    {
                        role: 'user',
                        content: JSON.stringify(phaseContext)
                    }
                ]
            });

            phase.result = result;
            phase.status = 'completed';
            phase.completedAt = new Date().toISOString();

            // Store outputs
            if (phase.outputs) {
                for (const output of phase.outputs) {
                    workflow.outputs[output] = result[output] || result;
                }
            }

            this.emit('phase-completed', { workflowId: workflow.id, phase: phase.name, result });
        } catch (error) {
            phase.status = 'failed';
            phase.error = error.message;
            throw error;
        }
    }

    /**
     * Call the LiteLLM gateway
     */
    async callGateway(request) {
        const gatewayUrl = process.env.LITELLM_URL || 'http://localhost:4000';

        const response = await fetch(`${gatewayUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.LITELLM_MASTER_KEY || ''}`
            },
            body: JSON.stringify(request)
        });

        if (!response.ok) {
            throw new Error(`Gateway error: ${response.status}`);
        }

        const data = await response.json();
        return data.choices[0].message.content;
    }

    /**
     * Get workflow status
     */
    getWorkflow(workflowId) {
        return this.workflows.get(workflowId);
    }

    /**
     * List all workflows
     */
    listWorkflows(filter = {}) {
        let workflows = Array.from(this.workflows.values());

        if (filter.status) {
            workflows = workflows.filter(w => w.status === filter.status);
        }

        if (filter.template) {
            workflows = workflows.filter(w => w.template === filter.template);
        }

        return workflows;
    }

    /**
     * Get available templates
     */
    getTemplates() {
        return Array.from(this.templates.entries()).map(([id, template]) => ({
            id,
            name: template.name,
            phases: template.phases.length
        }));
    }

    /**
     * Shutdown
     */
    async shutdown() {
        // Persist any unsaved reasoning
        if (this.reasoningBank && this.config.reasoningBank.persistPath) {
            const indexPath = path.join(this.config.reasoningBank.persistPath, 'index.json');
            const index = {
                metadata: this.reasoningBank.metadata,
                entries: Array.from(this.reasoningBank.entries.keys())
            };
            await fs.writeFile(indexPath, JSON.stringify(index, null, 2));
        }

        this.workflows.clear();
        this.emit('shutdown');
    }
}

export default ClaudeFlowIntegration;
