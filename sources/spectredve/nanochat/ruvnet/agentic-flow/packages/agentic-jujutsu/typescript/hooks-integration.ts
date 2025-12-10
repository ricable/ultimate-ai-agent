/**
 * TypeScript integration for agentic-flow hooks system
 *
 * This module provides type-safe integration between agentic-jujutsu
 * and the agentic-flow hooks system for Node.js and browser environments.
 */

import { JJWrapper, JJOperation, JJConfig } from './index';

/**
 * Hook context information
 */
export interface HookContext {
    /** Unique identifier for the agent */
    agentId: string;
    /** Session identifier for grouping related operations */
    sessionId: string;
    /** Human-readable task description */
    taskDescription: string;
    /** Unix timestamp in seconds */
    timestamp: number;
    /** Optional metadata for additional context */
    metadata?: Record<string, any>;
}

/**
 * Types of hook events
 */
export enum HookEventType {
    PreTask = 'PreTask',
    PostEdit = 'PostEdit',
    PostTask = 'PostTask',
    ConflictDetected = 'ConflictDetected',
    OperationLogged = 'OperationLogged',
    SessionInit = 'SessionInit',
    SessionEnd = 'SessionEnd',
}

/**
 * Hook event data structure
 */
export interface JJHookEvent {
    /** Type of hook event */
    eventType: HookEventType;
    /** Associated jj operation (if any) */
    operation?: JJOperation;
    /** Hook execution context */
    context: HookContext;
    /** Additional event metadata */
    metadata?: Record<string, any>;
}

/**
 * AgentDB episode data structure
 */
export interface AgentDBEpisode {
    sessionId: string;
    task: string;
    agentId: string;
    input?: string;
    output?: string;
    critique?: string;
    success: boolean;
    reward: number;
    latencyMs?: number;
    tokensUsed?: number;
    operation?: JJOperation;
    timestamp: number;
}

/**
 * Integration layer for agentic-flow hooks
 */
export class JJHooksIntegration {
    private wrapper: JJWrapper;
    private sessionId: string;
    private agentId: string;
    private agentdbEnabled: boolean;
    private currentSession?: HookContext;
    private operationLog: JJOperation[] = [];

    /**
     * Create a new hooks integration instance
     *
     * @param wrapper - JJWrapper instance
     * @param sessionId - Session identifier
     * @param agentId - Agent identifier
     * @param agentdbEnabled - Whether to sync with AgentDB
     */
    constructor(
        wrapper: JJWrapper,
        sessionId: string,
        agentId: string,
        agentdbEnabled: boolean = false
    ) {
        this.wrapper = wrapper;
        this.sessionId = sessionId;
        this.agentId = agentId;
        this.agentdbEnabled = agentdbEnabled;
    }

    /**
     * Execute pre-task hook
     */
    async onPreTask(description: string, metadata?: Record<string, any>): Promise<JJHookEvent> {
        const context: HookContext = {
            agentId: this.agentId,
            sessionId: this.sessionId,
            taskDescription: description,
            timestamp: Math.floor(Date.now() / 1000),
            metadata,
        };

        this.currentSession = context;

        const event: JJHookEvent = {
            eventType: HookEventType.PreTask,
            context,
            metadata: {
                action: 'session_init',
                description: `[pre-task] Agent: ${this.agentId} | Session: ${this.sessionId} | Task: ${description}`,
            },
        };

        console.log(`[jj] 🚀 Pre-task: ${description}`);

        // Sync to AgentDB if enabled
        if (this.agentdbEnabled) {
            await this.syncToAgentDB(event);
        }

        return event;
    }

    /**
     * Execute post-edit hook
     */
    async onPostEdit(file: string, changeDescription?: string): Promise<JJOperation> {
        const context: HookContext = {
            agentId: this.agentId,
            sessionId: this.sessionId,
            taskDescription: changeDescription || `Edit ${file}`,
            timestamp: Math.floor(Date.now() / 1000),
        };

        const description = `[post-edit] Agent: ${this.agentId} | File: ${file} | Session: ${this.sessionId}`;

        // Create operation record
        const operation: JJOperation = {
            id: this.generateOperationId(),
            operationType: 'Describe',
            description,
            timestamp: context.timestamp,
            user: this.agentId,
            args: [],
            metadata: {
                file,
                sessionId: this.sessionId,
                agentId: this.agentId,
                hook: 'post-edit',
            },
        };

        // Add to operation log
        this.operationLog.push(operation);

        console.log(`[jj] ✏️  Post-edit: ${file}`);

        // Sync to AgentDB if enabled
        if (this.agentdbEnabled) {
            await this.syncOperationToAgentDB(operation);
        }

        return operation;
    }

    /**
     * Execute post-task hook
     */
    async onPostTask(): Promise<JJOperation[]> {
        if (!this.currentSession) {
            throw new Error('No active session. Call onPreTask first.');
        }

        const operations = [...this.operationLog];

        const context: HookContext = {
            ...this.currentSession,
            timestamp: Math.floor(Date.now() / 1000),
        };

        const event: JJHookEvent = {
            eventType: HookEventType.PostTask,
            context,
            metadata: {
                sessionId: this.sessionId,
                agentId: this.agentId,
                operationsCount: operations.length,
            },
        };

        console.log(`[jj] ✅ Post-task: ${operations.length} operations logged`);

        // Sync to AgentDB if enabled
        if (this.agentdbEnabled) {
            await this.syncToAgentDB(event);
        }

        // Clear session
        this.currentSession = undefined;
        this.operationLog = [];

        return operations;
    }

    /**
     * Handle conflict detection
     */
    async onConflictDetected(conflicts: string[]): Promise<JJHookEvent> {
        if (!this.currentSession) {
            throw new Error('No active session. Call onPreTask first.');
        }

        const event: JJHookEvent = {
            eventType: HookEventType.ConflictDetected,
            context: this.currentSession,
            metadata: {
                conflicts,
                requiresResolution: true,
            },
        };

        console.log(`[jj] ⚠️  Conflicts detected: ${conflicts.join(', ')}`);

        // Sync to AgentDB for learning
        if (this.agentdbEnabled) {
            await this.syncToAgentDB(event);
        }

        return event;
    }

    /**
     * Get operations for current session
     */
    getSessionOperations(): JJOperation[] {
        return [...this.operationLog];
    }

    /**
     * Sync event to AgentDB
     */
    private async syncToAgentDB(event: JJHookEvent): Promise<void> {
        if (!this.agentdbEnabled) {
            return;
        }

        const episode: AgentDBEpisode = {
            sessionId: event.context.sessionId,
            task: event.context.taskDescription,
            agentId: event.context.agentId,
            success: true,
            reward: 1.0,
            operation: event.operation,
            timestamp: event.context.timestamp,
        };

        // TODO: Implement actual AgentDB sync via MCP
        console.log('[jj-agentdb] Would sync event:', JSON.stringify(episode, null, 2));

        // For now, could write to file or call MCP via subprocess
        if (process.env.AGENTDB_SYNC_FILE) {
            const fs = await import('fs');
            fs.appendFileSync(
                process.env.AGENTDB_SYNC_FILE,
                JSON.stringify(episode) + '\n'
            );
        }
    }

    /**
     * Sync operation to AgentDB
     */
    private async syncOperationToAgentDB(operation: JJOperation): Promise<void> {
        if (!this.agentdbEnabled) {
            return;
        }

        const episode: AgentDBEpisode = {
            sessionId: this.sessionId,
            task: operation.description,
            agentId: this.agentId,
            success: true,
            reward: 1.0,
            operation,
            timestamp: operation.timestamp,
        };

        console.log('[jj-agentdb] Would sync operation:', operation.id);

        // TODO: Implement actual AgentDB sync
        if (process.env.AGENTDB_SYNC_FILE) {
            const fs = await import('fs');
            fs.appendFileSync(
                process.env.AGENTDB_SYNC_FILE,
                JSON.stringify(episode) + '\n'
            );
        }
    }

    /**
     * Generate unique operation ID
     */
    private generateOperationId(): string {
        return `op-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Check if AgentDB sync is enabled
     */
    isAgentDBEnabled(): boolean {
        return this.agentdbEnabled;
    }

    /**
     * Get current session context
     */
    getCurrentSession(): HookContext | undefined {
        return this.currentSession;
    }
}

/**
 * Factory function to create hooks integration
 */
export async function createHooksIntegration(
    config: JJConfig,
    sessionId: string,
    agentId: string,
    agentdbEnabled: boolean = false
): Promise<JJHooksIntegration> {
    const wrapper = await JJWrapper.new(config);
    return new JJHooksIntegration(wrapper, sessionId, agentId, agentdbEnabled);
}

/**
 * Helper to execute a full task lifecycle with hooks
 */
export async function withHooks<T>(
    integration: JJHooksIntegration,
    taskDescription: string,
    task: (integration: JJHooksIntegration) => Promise<T>
): Promise<{ result: T; operations: JJOperation[] }> {
    // Pre-task
    await integration.onPreTask(taskDescription);

    try {
        // Execute task
        const result = await task(integration);

        // Post-task
        const operations = await integration.onPostTask();

        return { result, operations };
    } catch (error) {
        // Even on error, try to capture operations
        const operations = await integration.onPostTask();
        throw error;
    }
}
