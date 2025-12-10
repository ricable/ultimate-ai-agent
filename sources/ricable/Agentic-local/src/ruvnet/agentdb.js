/**
 * =============================================================================
 * AgentDB Integration
 * Distributed agent state management with SQLite local + Redis sync
 * =============================================================================
 */

import EventEmitter from 'events';
import Database from 'better-sqlite3';
import Redis from 'ioredis';
import crypto from 'crypto';

/**
 * AgentDBIntegration - Manages distributed agent state
 */
export class AgentDBIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            local: {
                path: './data/agentdb.sqlite',
                wal: true,
                verbose: false
            },
            distributed: {
                enabled: true,
                redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
                syncInterval: 5000,
                namespace: 'agentdb'
            },
            encryption: {
                enabled: false,
                key: process.env.AGENTDB_ENCRYPTION_KEY,
                algorithm: 'aes-256-gcm'
            },
            ...config
        };

        this.db = null;
        this.redis = null;
        this.syncTimer = null;
        this.pendingSync = new Map();
    }

    /**
     * Initialize AgentDB
     */
    async initialize() {
        try {
            // Initialize SQLite database
            await this.initializeLocalDB();

            // Initialize Redis for distributed sync
            if (this.config.distributed.enabled) {
                await this.initializeRedis();
                this.startSyncLoop();
            }

            this.emit('initialized', { config: this.config });
            return true;
        } catch (error) {
            this.emit('error', { phase: 'initialization', error });
            throw error;
        }
    }

    /**
     * Initialize local SQLite database
     */
    async initializeLocalDB() {
        const dbPath = this.config.local.path;

        // Ensure directory exists
        const fs = await import('fs/promises');
        const path = await import('path');
        await fs.mkdir(path.dirname(dbPath), { recursive: true });

        this.db = new Database(dbPath, {
            verbose: this.config.local.verbose ? console.log : null
        });

        // Enable WAL mode for better concurrency
        if (this.config.local.wal) {
            this.db.pragma('journal_mode = WAL');
        }

        // Create tables
        this.db.exec(`
            -- Agents table
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'general',
                state TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                version INTEGER DEFAULT 1
            );

            -- Agent memory table
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );

            -- Conversations table
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                messages TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );

            -- Tasks table
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                input TEXT DEFAULT '{}',
                output TEXT,
                error TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL
            );

            -- Sync log for distributed coordination
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                data TEXT NOT NULL,
                synced INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_agent_memory_agent ON agent_memory(agent_id);
            CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory(type);
            CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations(agent_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_sync_log_synced ON sync_log(synced);
        `);

        this.emit('local-db-initialized');
    }

    /**
     * Initialize Redis for distributed sync
     */
    async initializeRedis() {
        this.redis = new Redis(this.config.distributed.redisUrl, {
            retryStrategy: (times) => Math.min(times * 50, 2000),
            maxRetriesPerRequest: 3
        });

        this.redis.on('connect', () => {
            this.emit('redis-connected');
        });

        this.redis.on('error', (error) => {
            this.emit('redis-error', { error });
        });

        // Subscribe to sync channel
        const subscriber = this.redis.duplicate();
        const channel = `${this.config.distributed.namespace}:sync`;

        await subscriber.subscribe(channel);
        subscriber.on('message', (ch, message) => {
            if (ch === channel) {
                this.handleSyncMessage(JSON.parse(message));
            }
        });
    }

    /**
     * Start the sync loop for distributed updates
     */
    startSyncLoop() {
        this.syncTimer = setInterval(async () => {
            await this.syncToRedis();
        }, this.config.distributed.syncInterval);
    }

    /**
     * Sync pending changes to Redis
     */
    async syncToRedis() {
        if (!this.redis) return;

        const unsyncedStmt = this.db.prepare(`
            SELECT * FROM sync_log WHERE synced = 0 ORDER BY created_at ASC LIMIT 100
        `);

        const unsynced = unsyncedStmt.all();
        if (unsynced.length === 0) return;

        const pipeline = this.redis.pipeline();
        const channel = `${this.config.distributed.namespace}:sync`;

        for (const record of unsynced) {
            const key = `${this.config.distributed.namespace}:${record.table_name}:${record.record_id}`;

            if (record.operation === 'DELETE') {
                pipeline.del(key);
            } else {
                const data = this.maybeEncrypt(record.data);
                pipeline.set(key, data);
            }

            // Publish sync event
            pipeline.publish(channel, JSON.stringify({
                table: record.table_name,
                id: record.record_id,
                operation: record.operation,
                timestamp: record.created_at
            }));
        }

        await pipeline.exec();

        // Mark as synced
        const markSyncedStmt = this.db.prepare(`
            UPDATE sync_log SET synced = 1 WHERE id = ?
        `);

        for (const record of unsynced) {
            markSyncedStmt.run(record.id);
        }

        this.emit('sync-completed', { count: unsynced.length });
    }

    /**
     * Handle incoming sync messages from other nodes
     */
    async handleSyncMessage(message) {
        // Skip messages from self (implement node ID check)
        this.emit('sync-received', message);

        // Fetch latest from Redis and update local
        const key = `${this.config.distributed.namespace}:${message.table}:${message.id}`;
        const data = await this.redis.get(key);

        if (data) {
            const decrypted = this.maybeDecrypt(data);
            // Update local without triggering sync
            this.updateLocalFromRemote(message.table, message.id, JSON.parse(decrypted));
        }
    }

    /**
     * Update local DB from remote sync
     */
    updateLocalFromRemote(table, id, data) {
        // Implementation depends on table structure
        // Skip sync_log entry to prevent infinite loop
    }

    /**
     * Encrypt data if encryption is enabled
     */
    maybeEncrypt(data) {
        if (!this.config.encryption.enabled || !this.config.encryption.key) {
            return data;
        }

        const iv = crypto.randomBytes(16);
        const key = Buffer.from(this.config.encryption.key, 'hex');
        const cipher = crypto.createCipheriv(this.config.encryption.algorithm, key, iv);

        let encrypted = cipher.update(data, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        const authTag = cipher.getAuthTag();

        return JSON.stringify({
            iv: iv.toString('hex'),
            data: encrypted,
            tag: authTag.toString('hex')
        });
    }

    /**
     * Decrypt data if encryption is enabled
     */
    maybeDecrypt(data) {
        if (!this.config.encryption.enabled || !this.config.encryption.key) {
            return data;
        }

        const parsed = JSON.parse(data);
        const key = Buffer.from(this.config.encryption.key, 'hex');
        const iv = Buffer.from(parsed.iv, 'hex');
        const authTag = Buffer.from(parsed.tag, 'hex');

        const decipher = crypto.createDecipheriv(this.config.encryption.algorithm, key, iv);
        decipher.setAuthTag(authTag);

        let decrypted = decipher.update(parsed.data, 'hex', 'utf8');
        decrypted += decipher.final('utf8');

        return decrypted;
    }

    /**
     * Log operation for sync
     */
    logSync(table, id, operation, data) {
        const stmt = this.db.prepare(`
            INSERT INTO sync_log (table_name, record_id, operation, data)
            VALUES (?, ?, ?, ?)
        `);
        stmt.run(table, id, operation, JSON.stringify(data));
    }

    // =========================================================================
    // AGENT OPERATIONS
    // =========================================================================

    /**
     * Create a new agent
     */
    createAgent(agent) {
        const id = agent.id || `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        const stmt = this.db.prepare(`
            INSERT INTO agents (id, name, type, state, metadata)
            VALUES (?, ?, ?, ?, ?)
        `);

        stmt.run(
            id,
            agent.name || id,
            agent.type || 'general',
            JSON.stringify(agent.state || {}),
            JSON.stringify(agent.metadata || {})
        );

        this.logSync('agents', id, 'INSERT', agent);
        this.emit('agent-created', { id, agent });

        return { id, ...agent };
    }

    /**
     * Get agent by ID
     */
    getAgent(id) {
        const stmt = this.db.prepare(`
            SELECT * FROM agents WHERE id = ?
        `);

        const row = stmt.get(id);
        if (!row) return null;

        return {
            ...row,
            state: JSON.parse(row.state),
            metadata: JSON.parse(row.metadata)
        };
    }

    /**
     * Update agent
     */
    updateAgent(id, updates) {
        const agent = this.getAgent(id);
        if (!agent) {
            throw new Error(`Agent not found: ${id}`);
        }

        const newState = { ...agent.state, ...updates.state };
        const newMetadata = { ...agent.metadata, ...updates.metadata };

        const stmt = this.db.prepare(`
            UPDATE agents
            SET name = COALESCE(?, name),
                type = COALESCE(?, type),
                state = ?,
                metadata = ?,
                updated_at = datetime('now'),
                version = version + 1
            WHERE id = ?
        `);

        stmt.run(
            updates.name || null,
            updates.type || null,
            JSON.stringify(newState),
            JSON.stringify(newMetadata),
            id
        );

        this.logSync('agents', id, 'UPDATE', { ...updates, state: newState, metadata: newMetadata });
        this.emit('agent-updated', { id, updates });

        return this.getAgent(id);
    }

    /**
     * Delete agent
     */
    deleteAgent(id) {
        const stmt = this.db.prepare(`
            DELETE FROM agents WHERE id = ?
        `);

        stmt.run(id);
        this.logSync('agents', id, 'DELETE', { id });
        this.emit('agent-deleted', { id });
    }

    /**
     * List agents
     */
    listAgents(filter = {}) {
        let sql = 'SELECT * FROM agents WHERE 1=1';
        const params = [];

        if (filter.type) {
            sql += ' AND type = ?';
            params.push(filter.type);
        }

        if (filter.limit) {
            sql += ' LIMIT ?';
            params.push(filter.limit);
        }

        const stmt = this.db.prepare(sql);
        const rows = stmt.all(...params);

        return rows.map(row => ({
            ...row,
            state: JSON.parse(row.state),
            metadata: JSON.parse(row.metadata)
        }));
    }

    // =========================================================================
    // MEMORY OPERATIONS
    // =========================================================================

    /**
     * Add memory to agent
     */
    addMemory(agentId, memory) {
        const stmt = this.db.prepare(`
            INSERT INTO agent_memory (agent_id, type, content, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
        `);

        const result = stmt.run(
            agentId,
            memory.type || 'general',
            JSON.stringify(memory.content),
            memory.embedding ? Buffer.from(new Float32Array(memory.embedding).buffer) : null,
            JSON.stringify(memory.metadata || {})
        );

        this.emit('memory-added', { agentId, memoryId: result.lastInsertRowid });
        return result.lastInsertRowid;
    }

    /**
     * Get agent memories
     */
    getMemories(agentId, options = {}) {
        let sql = 'SELECT * FROM agent_memory WHERE agent_id = ?';
        const params = [agentId];

        if (options.type) {
            sql += ' AND type = ?';
            params.push(options.type);
        }

        sql += ' ORDER BY created_at DESC';

        if (options.limit) {
            sql += ' LIMIT ?';
            params.push(options.limit);
        }

        const stmt = this.db.prepare(sql);
        return stmt.all(...params).map(row => ({
            ...row,
            content: JSON.parse(row.content),
            embedding: row.embedding ? Array.from(new Float32Array(row.embedding.buffer)) : null,
            metadata: JSON.parse(row.metadata)
        }));
    }

    // =========================================================================
    // CONVERSATION OPERATIONS
    // =========================================================================

    /**
     * Create conversation
     */
    createConversation(agentId, userId = null) {
        const id = `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        const stmt = this.db.prepare(`
            INSERT INTO conversations (id, agent_id, user_id)
            VALUES (?, ?, ?)
        `);

        stmt.run(id, agentId, userId);
        return { id, agentId, userId, messages: [] };
    }

    /**
     * Add message to conversation
     */
    addMessage(conversationId, message) {
        const getStmt = this.db.prepare(`
            SELECT messages FROM conversations WHERE id = ?
        `);

        const row = getStmt.get(conversationId);
        if (!row) {
            throw new Error(`Conversation not found: ${conversationId}`);
        }

        const messages = JSON.parse(row.messages);
        messages.push({
            ...message,
            timestamp: new Date().toISOString()
        });

        const updateStmt = this.db.prepare(`
            UPDATE conversations
            SET messages = ?, updated_at = datetime('now')
            WHERE id = ?
        `);

        updateStmt.run(JSON.stringify(messages), conversationId);
        this.emit('message-added', { conversationId, message });

        return messages;
    }

    /**
     * Get conversation
     */
    getConversation(id) {
        const stmt = this.db.prepare(`
            SELECT * FROM conversations WHERE id = ?
        `);

        const row = stmt.get(id);
        if (!row) return null;

        return {
            ...row,
            messages: JSON.parse(row.messages),
            metadata: JSON.parse(row.metadata)
        };
    }

    // =========================================================================
    // TASK OPERATIONS
    // =========================================================================

    /**
     * Create task
     */
    createTask(task) {
        const id = task.id || `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        const stmt = this.db.prepare(`
            INSERT INTO tasks (id, agent_id, type, input)
            VALUES (?, ?, ?, ?)
        `);

        stmt.run(id, task.agentId || null, task.type, JSON.stringify(task.input || {}));
        this.emit('task-created', { id, task });

        return { id, ...task, status: 'pending' };
    }

    /**
     * Update task status
     */
    updateTaskStatus(id, status, output = null, error = null) {
        const stmt = this.db.prepare(`
            UPDATE tasks
            SET status = ?,
                output = ?,
                error = ?,
                started_at = CASE WHEN ? = 'running' AND started_at IS NULL THEN datetime('now') ELSE started_at END,
                completed_at = CASE WHEN ? IN ('completed', 'failed') THEN datetime('now') ELSE completed_at END
            WHERE id = ?
        `);

        stmt.run(
            status,
            output ? JSON.stringify(output) : null,
            error,
            status,
            status,
            id
        );

        this.emit('task-updated', { id, status, output, error });
    }

    /**
     * Get task
     */
    getTask(id) {
        const stmt = this.db.prepare(`
            SELECT * FROM tasks WHERE id = ?
        `);

        const row = stmt.get(id);
        if (!row) return null;

        return {
            ...row,
            input: JSON.parse(row.input),
            output: row.output ? JSON.parse(row.output) : null
        };
    }

    /**
     * Shutdown
     */
    async shutdown() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
        }

        // Final sync
        if (this.redis) {
            await this.syncToRedis();
            this.redis.disconnect();
        }

        if (this.db) {
            this.db.close();
        }

        this.emit('shutdown');
    }
}

export default AgentDBIntegration;
