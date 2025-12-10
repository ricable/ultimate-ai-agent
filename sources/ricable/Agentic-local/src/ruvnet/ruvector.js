/**
 * =============================================================================
 * RuVector Integration
 * High-performance vector database for semantic search and embeddings
 * =============================================================================
 */

import EventEmitter from 'events';

/**
 * RuVectorIntegration - Vector database for AI agents
 */
export class RuVectorIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            backend: 'local', // 'local' | 'postgres' | 'remote'
            dimensions: 1536, // OpenAI ada-002 dimensions
            indexType: 'hnsw', // 'hnsw' | 'flat' | 'ivf'
            similarity: 'cosine', // 'cosine' | 'euclidean' | 'dot'
            local: {
                path: './data/ruvector',
                maxElements: 1000000,
                efConstruction: 200,
                M: 16
            },
            postgres: {
                connectionString: process.env.POSTGRES_URL,
                tableName: 'vectors'
            },
            embedding: {
                provider: 'local', // 'local' | 'openai'
                model: 'all-MiniLM-L6-v2',
                batchSize: 32
            },
            ...config
        };

        this.index = null;
        this.vectors = new Map();
        this.metadata = new Map();
        this.embeddingModel = null;
    }

    /**
     * Initialize RuVector
     */
    async initialize() {
        try {
            // Initialize based on backend
            switch (this.config.backend) {
                case 'local':
                    await this.initializeLocalIndex();
                    break;
                case 'postgres':
                    await this.initializePostgres();
                    break;
                case 'remote':
                    await this.initializeRemote();
                    break;
            }

            // Initialize embedding model
            await this.initializeEmbedding();

            this.emit('initialized', { config: this.config });
            return true;
        } catch (error) {
            this.emit('error', { phase: 'initialization', error });
            throw error;
        }
    }

    /**
     * Initialize local HNSW index
     */
    async initializeLocalIndex() {
        const fs = await import('fs/promises');
        const path = await import('path');

        await fs.mkdir(this.config.local.path, { recursive: true });

        // Initialize in-memory HNSW index
        this.index = {
            type: 'hnsw',
            dimensions: this.config.dimensions,
            efConstruction: this.config.local.efConstruction,
            M: this.config.local.M,
            elements: new Map(),
            graph: new Map()
        };

        // Load existing index if available
        const indexPath = path.join(this.config.local.path, 'index.json');
        try {
            const data = await fs.readFile(indexPath, 'utf-8');
            const saved = JSON.parse(data);
            this.index.elements = new Map(saved.elements);
            this.vectors = new Map(saved.vectors);
            this.metadata = new Map(saved.metadata);
            this.emit('index-loaded', { count: this.index.elements.size });
        } catch (error) {
            // No existing index
        }
    }

    /**
     * Initialize PostgreSQL with pgvector
     */
    async initializePostgres() {
        const pg = await import('pg');
        this.pgClient = new pg.Client(this.config.postgres.connectionString);
        await this.pgClient.connect();

        // Create table with vector extension
        await this.pgClient.query(`
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS ${this.config.postgres.tableName} (
                id TEXT PRIMARY KEY,
                embedding vector(${this.config.dimensions}),
                content TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS ${this.config.postgres.tableName}_embedding_idx
            ON ${this.config.postgres.tableName}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        `);

        this.emit('postgres-initialized');
    }

    /**
     * Initialize remote vector service
     */
    async initializeRemote() {
        // Placeholder for remote vector DB (e.g., Pinecone, Qdrant, Weaviate)
        this.remoteClient = {
            url: this.config.remote?.url,
            apiKey: this.config.remote?.apiKey
        };
    }

    /**
     * Initialize embedding model
     */
    async initializeEmbedding() {
        if (this.config.embedding.provider === 'local') {
            // Use local embedding through LiteLLM
            this.embeddingModel = {
                type: 'local',
                model: this.config.embedding.model
            };
        } else {
            this.embeddingModel = {
                type: 'openai',
                model: 'text-embedding-ada-002'
            };
        }
    }

    /**
     * Generate embeddings for text
     * @param {string|string[]} texts - Text(s) to embed
     * @returns {Promise<number[][]>} Embedding vectors
     */
    async embed(texts) {
        const textArray = Array.isArray(texts) ? texts : [texts];
        const embeddings = [];

        // Batch processing
        for (let i = 0; i < textArray.length; i += this.config.embedding.batchSize) {
            const batch = textArray.slice(i, i + this.config.embedding.batchSize);
            const batchEmbeddings = await this.embedBatch(batch);
            embeddings.push(...batchEmbeddings);
        }

        return Array.isArray(texts) ? embeddings : embeddings[0];
    }

    /**
     * Embed a batch of texts
     */
    async embedBatch(texts) {
        const gatewayUrl = process.env.LITELLM_URL || 'http://localhost:4000';

        const response = await fetch(`${gatewayUrl}/v1/embeddings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.LITELLM_MASTER_KEY || ''}`
            },
            body: JSON.stringify({
                model: this.embeddingModel.model,
                input: texts
            })
        });

        if (!response.ok) {
            throw new Error(`Embedding error: ${response.status}`);
        }

        const data = await response.json();
        return data.data.map(d => d.embedding);
    }

    /**
     * Add vector to index
     * @param {string} id - Unique ID
     * @param {number[]|string} vectorOrText - Vector or text to embed
     * @param {Object} metadata - Associated metadata
     */
    async add(id, vectorOrText, metadata = {}) {
        let vector;

        // Check if input is text or vector
        if (typeof vectorOrText === 'string') {
            vector = await this.embed(vectorOrText);
            metadata.originalText = vectorOrText;
        } else {
            vector = vectorOrText;
        }

        // Validate dimensions
        if (vector.length !== this.config.dimensions) {
            throw new Error(`Vector dimensions mismatch: expected ${this.config.dimensions}, got ${vector.length}`);
        }

        switch (this.config.backend) {
            case 'local':
                await this.addLocal(id, vector, metadata);
                break;
            case 'postgres':
                await this.addPostgres(id, vector, metadata);
                break;
            case 'remote':
                await this.addRemote(id, vector, metadata);
                break;
        }

        this.emit('vector-added', { id, dimensions: vector.length });
        return { id, dimensions: vector.length };
    }

    /**
     * Add vector to local index
     */
    async addLocal(id, vector, metadata) {
        this.vectors.set(id, vector);
        this.metadata.set(id, {
            ...metadata,
            addedAt: new Date().toISOString()
        });
        this.index.elements.set(id, vector);
    }

    /**
     * Add vector to PostgreSQL
     */
    async addPostgres(id, vector, metadata) {
        const vectorStr = `[${vector.join(',')}]`;

        await this.pgClient.query(`
            INSERT INTO ${this.config.postgres.tableName} (id, embedding, content, metadata)
            VALUES ($1, $2::vector, $3, $4)
            ON CONFLICT (id) DO UPDATE
            SET embedding = $2::vector, content = $3, metadata = $4
        `, [id, vectorStr, metadata.originalText || '', JSON.stringify(metadata)]);
    }

    /**
     * Add vector to remote service
     */
    async addRemote(id, vector, metadata) {
        // Implement based on remote service API
    }

    /**
     * Search for similar vectors
     * @param {number[]|string} queryOrText - Query vector or text
     * @param {Object} options - Search options
     * @returns {Promise<Object[]>} Search results
     */
    async search(queryOrText, options = {}) {
        const {
            k = 10,
            threshold = 0,
            filter = null
        } = options;

        let queryVector;

        if (typeof queryOrText === 'string') {
            queryVector = await this.embed(queryOrText);
        } else {
            queryVector = queryOrText;
        }

        let results;

        switch (this.config.backend) {
            case 'local':
                results = await this.searchLocal(queryVector, k, filter);
                break;
            case 'postgres':
                results = await this.searchPostgres(queryVector, k, filter);
                break;
            case 'remote':
                results = await this.searchRemote(queryVector, k, filter);
                break;
        }

        // Apply threshold filter
        if (threshold > 0) {
            results = results.filter(r => r.score >= threshold);
        }

        this.emit('search-completed', { k, resultsCount: results.length });
        return results;
    }

    /**
     * Search local index
     */
    async searchLocal(queryVector, k, filter) {
        const results = [];

        for (const [id, vector] of this.vectors.entries()) {
            const metadata = this.metadata.get(id);

            // Apply filter if provided
            if (filter && !this.matchesFilter(metadata, filter)) {
                continue;
            }

            const score = this.cosineSimilarity(queryVector, vector);
            results.push({ id, score, metadata });
        }

        // Sort by score descending
        results.sort((a, b) => b.score - a.score);

        return results.slice(0, k);
    }

    /**
     * Search PostgreSQL
     */
    async searchPostgres(queryVector, k, filter) {
        const vectorStr = `[${queryVector.join(',')}]`;

        let whereClause = '';
        const params = [vectorStr, k];

        if (filter) {
            const conditions = Object.entries(filter)
                .map(([key, value], i) => `metadata->>'${key}' = $${i + 3}`)
                .join(' AND ');
            whereClause = `WHERE ${conditions}`;
            params.push(...Object.values(filter));
        }

        const result = await this.pgClient.query(`
            SELECT id, content, metadata,
                   1 - (embedding <=> $1::vector) as score
            FROM ${this.config.postgres.tableName}
            ${whereClause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        `, params);

        return result.rows.map(row => ({
            id: row.id,
            score: parseFloat(row.score),
            metadata: {
                ...row.metadata,
                originalText: row.content
            }
        }));
    }

    /**
     * Search remote service
     */
    async searchRemote(queryVector, k, filter) {
        // Implement based on remote service API
        return [];
    }

    /**
     * Check if metadata matches filter
     */
    matchesFilter(metadata, filter) {
        for (const [key, value] of Object.entries(filter)) {
            if (metadata[key] !== value) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculate cosine similarity
     */
    cosineSimilarity(a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Delete vector by ID
     */
    async delete(id) {
        switch (this.config.backend) {
            case 'local':
                this.vectors.delete(id);
                this.metadata.delete(id);
                this.index.elements.delete(id);
                break;
            case 'postgres':
                await this.pgClient.query(`
                    DELETE FROM ${this.config.postgres.tableName} WHERE id = $1
                `, [id]);
                break;
        }

        this.emit('vector-deleted', { id });
    }

    /**
     * Get vector by ID
     */
    async get(id) {
        switch (this.config.backend) {
            case 'local':
                return {
                    id,
                    vector: this.vectors.get(id),
                    metadata: this.metadata.get(id)
                };
            case 'postgres':
                const result = await this.pgClient.query(`
                    SELECT id, embedding, content, metadata
                    FROM ${this.config.postgres.tableName}
                    WHERE id = $1
                `, [id]);
                if (result.rows.length === 0) return null;
                return {
                    id: result.rows[0].id,
                    vector: result.rows[0].embedding,
                    metadata: result.rows[0].metadata
                };
        }
    }

    /**
     * Get index statistics
     */
    async stats() {
        let count;

        switch (this.config.backend) {
            case 'local':
                count = this.vectors.size;
                break;
            case 'postgres':
                const result = await this.pgClient.query(`
                    SELECT COUNT(*) as count FROM ${this.config.postgres.tableName}
                `);
                count = parseInt(result.rows[0].count);
                break;
        }

        return {
            backend: this.config.backend,
            dimensions: this.config.dimensions,
            indexType: this.config.indexType,
            vectorCount: count,
            embeddingModel: this.embeddingModel.model
        };
    }

    /**
     * Persist local index to disk
     */
    async persist() {
        if (this.config.backend !== 'local') return;

        const fs = await import('fs/promises');
        const path = await import('path');

        const indexPath = path.join(this.config.local.path, 'index.json');
        const data = {
            elements: Array.from(this.index.elements.entries()),
            vectors: Array.from(this.vectors.entries()),
            metadata: Array.from(this.metadata.entries()),
            savedAt: new Date().toISOString()
        };

        await fs.writeFile(indexPath, JSON.stringify(data));
        this.emit('index-persisted', { count: this.vectors.size });
    }

    /**
     * Shutdown
     */
    async shutdown() {
        // Persist local index
        if (this.config.backend === 'local') {
            await this.persist();
        }

        // Close PostgreSQL connection
        if (this.pgClient) {
            await this.pgClient.end();
        }

        this.emit('shutdown');
    }
}

export default RuVectorIntegration;
