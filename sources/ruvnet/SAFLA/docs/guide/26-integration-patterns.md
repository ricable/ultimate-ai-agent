# Integration Patterns

## Overview

SAFLA provides flexible integration patterns for connecting with external systems, developing custom MCP servers, and implementing third-party service integrations. This guide covers common integration patterns, best practices, and detailed examples for extending SAFLA's capabilities.

## Table of Contents

- [MCP Server Development](#mcp-server-development)
- [Third-Party Service Integration](#third-party-service-integration)
- [Custom Memory Providers](#custom-memory-providers)
- [Agent Extension Patterns](#agent-extension-patterns)
- [API Integration Strategies](#api-integration-strategies)
- [Event-Driven Integrations](#event-driven-integrations)
- [Security Considerations](#security-considerations)
- [Testing Integration Points](#testing-integration-points)

## MCP Server Development

### Creating Custom MCP Servers

MCP (Model Context Protocol) servers extend SAFLA's capabilities by providing specialized tools and resources.

#### Basic MCP Server Structure

```typescript
// mcp-server-template.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';

class CustomMCPServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'custom-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupResourceHandlers();
  }

  private setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'custom_analysis',
            description: 'Perform custom data analysis',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'string',
                  description: 'Data to analyze',
                },
                analysisType: {
                  type: 'string',
                  enum: ['statistical', 'pattern', 'trend'],
                  description: 'Type of analysis to perform',
                },
              },
              required: ['data', 'analysisType'],
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'custom_analysis':
          return await this.performCustomAnalysis(args);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  private async performCustomAnalysis(args: any) {
    const { data, analysisType } = args;
    
    // Implement your custom analysis logic
    const result = await this.analyzeData(data, analysisType);
    
    return {
      content: [
        {
          type: 'text',
          text: `Analysis complete: ${JSON.stringify(result)}`,
        },
      ],
    };
  }

  private async analyzeData(data: string, type: string) {
    // Custom analysis implementation
    switch (type) {
      case 'statistical':
        return this.performStatisticalAnalysis(data);
      case 'pattern':
        return this.performPatternAnalysis(data);
      case 'trend':
        return this.performTrendAnalysis(data);
      default:
        throw new Error(`Unsupported analysis type: ${type}`);
    }
  }

  private performStatisticalAnalysis(data: string) {
    // Implementation for statistical analysis
    return {
      mean: 0,
      median: 0,
      standardDeviation: 0,
      confidence: 0.95,
    };
  }

  private performPatternAnalysis(data: string) {
    // Implementation for pattern analysis
    return {
      patterns: [],
      frequency: {},
      anomalies: [],
    };
  }

  private performTrendAnalysis(data: string) {
    // Implementation for trend analysis
    return {
      trend: 'increasing',
      slope: 0.1,
      correlation: 0.8,
      forecast: [],
    };
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}

// Start the server
const server = new CustomMCPServer();
server.start().catch(console.error);
```

#### Advanced MCP Server Features

```typescript
// advanced-mcp-server.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  Resource,
} from '@modelcontextprotocol/sdk/types.js';

class AdvancedMCPServer extends CustomMCPServer {
  private resources: Map<string, any> = new Map();
  private cache: Map<string, any> = new Map();

  constructor() {
    super();
    this.setupAdvancedFeatures();
  }

  private setupAdvancedFeatures() {
    this.setupResourceHandlers();
    this.setupCaching();
    this.setupErrorHandling();
    this.setupLogging();
  }

  private setupResourceHandlers() {
    // List available resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: Array.from(this.resources.entries()).map(([uri, resource]) => ({
          uri,
          name: resource.name,
          description: resource.description,
          mimeType: resource.mimeType,
        })),
      };
    });

    // Read resource content
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;
      const resource = this.resources.get(uri);
      
      if (!resource) {
        throw new Error(`Resource not found: ${uri}`);
      }

      return {
        contents: [
          {
            uri,
            mimeType: resource.mimeType,
            text: resource.content,
          },
        ],
      };
    });
  }

  private setupCaching() {
    // Implement caching for expensive operations
    setInterval(() => {
      this.cleanupCache();
    }, 300000); // Clean cache every 5 minutes
  }

  private cleanupCache() {
    const now = Date.now();
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > 600000) { // 10 minutes TTL
        this.cache.delete(key);
      }
    }
  }

  private setupErrorHandling() {
    process.on('uncaughtException', (error) => {
      console.error('Uncaught exception:', error);
      // Implement graceful shutdown
    });

    process.on('unhandledRejection', (reason, promise) => {
      console.error('Unhandled rejection at:', promise, 'reason:', reason);
    });
  }

  private setupLogging() {
    // Implement structured logging
    const originalLog = console.log;
    console.log = (...args) => {
      const timestamp = new Date().toISOString();
      originalLog(`[${timestamp}]`, ...args);
    };
  }

  // Add resource dynamically
  addResource(uri: string, resource: any) {
    this.resources.set(uri, {
      ...resource,
      timestamp: Date.now(),
    });
  }

  // Remove resource
  removeResource(uri: string) {
    this.resources.delete(uri);
  }
}
```

### MCP Server Configuration

```yaml
# mcp-server-config.yaml
servers:
  custom-analysis:
    command: "node"
    args: ["dist/mcp-server-template.js"]
    env:
      LOG_LEVEL: "info"
      CACHE_SIZE: "1000"
    
  advanced-features:
    command: "node"
    args: ["dist/advanced-mcp-server.js"]
    env:
      LOG_LEVEL: "debug"
      ENABLE_CACHING: "true"
      CACHE_TTL: "600"

  external-api:
    command: "python"
    args: ["-m", "external_api_server"]
    env:
      API_KEY: "${EXTERNAL_API_KEY}"
      RATE_LIMIT: "100"
```

## Third-Party Service Integration

### REST API Integration

```typescript
// api-integration.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { RateLimiter } from 'limiter';

export class APIIntegration {
  private client: AxiosInstance;
  private rateLimiter: RateLimiter;
  private cache: Map<string, any> = new Map();

  constructor(
    baseURL: string,
    apiKey: string,
    rateLimit: number = 100
  ) {
    this.client = axios.create({
      baseURL,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    this.rateLimiter = new RateLimiter({
      tokensPerInterval: rateLimit,
      interval: 'hour',
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor for rate limiting
    this.client.interceptors.request.use(async (config) => {
      await this.rateLimiter.removeTokens(1);
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        throw error;
      }
    );
  }

  async get(endpoint: string, params?: any, useCache: boolean = true) {
    const cacheKey = `GET:${endpoint}:${JSON.stringify(params)}`;
    
    if (useCache && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < 300000) { // 5 minutes
        return cached.data;
      }
    }

    const response = await this.client.get(endpoint, { params });
    
    if (useCache) {
      this.cache.set(cacheKey, {
        data: response.data,
        timestamp: Date.now(),
      });
    }

    return response.data;
  }

  async post(endpoint: string, data: any) {
    const response = await this.client.post(endpoint, data);
    return response.data;
  }

  async put(endpoint: string, data: any) {
    const response = await this.client.put(endpoint, data);
    return response.data;
  }

  async delete(endpoint: string) {
    const response = await this.client.delete(endpoint);
    return response.data;
  }
}

// Usage example
const apiIntegration = new APIIntegration(
  'https://api.example.com/v1',
  process.env.API_KEY!,
  50 // 50 requests per hour
);

// Use in MCP server
export class APIIntegrationMCPServer {
  private api: APIIntegration;

  constructor() {
    this.api = new APIIntegration(
      process.env.API_BASE_URL!,
      process.env.API_KEY!
    );
  }

  async handleAPICall(endpoint: string, method: string, data?: any) {
    try {
      switch (method.toUpperCase()) {
        case 'GET':
          return await this.api.get(endpoint);
        case 'POST':
          return await this.api.post(endpoint, data);
        case 'PUT':
          return await this.api.put(endpoint, data);
        case 'DELETE':
          return await this.api.delete(endpoint);
        default:
          throw new Error(`Unsupported method: ${method}`);
      }
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  }
}
```

### Database Integration

```typescript
// database-integration.ts
import { Pool, PoolClient } from 'pg';
import { MongoClient, Db } from 'mongodb';

export class DatabaseIntegration {
  private pgPool?: Pool;
  private mongoClient?: MongoClient;
  private mongoDB?: Db;

  async initializePostgreSQL(connectionString: string) {
    this.pgPool = new Pool({
      connectionString,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    // Test connection
    const client = await this.pgPool.connect();
    await client.query('SELECT NOW()');
    client.release();
  }

  async initializeMongoDB(connectionString: string, dbName: string) {
    this.mongoClient = new MongoClient(connectionString);
    await this.mongoClient.connect();
    this.mongoDB = this.mongoClient.db(dbName);
  }

  async executePostgreSQLQuery(query: string, params?: any[]) {
    if (!this.pgPool) {
      throw new Error('PostgreSQL not initialized');
    }

    const client = await this.pgPool.connect();
    try {
      const result = await client.query(query, params);
      return result.rows;
    } finally {
      client.release();
    }
  }

  async executeMongoDBOperation(collection: string, operation: string, query: any) {
    if (!this.mongoDB) {
      throw new Error('MongoDB not initialized');
    }

    const coll = this.mongoDB.collection(collection);
    
    switch (operation) {
      case 'find':
        return await coll.find(query).toArray();
      case 'findOne':
        return await coll.findOne(query);
      case 'insertOne':
        return await coll.insertOne(query);
      case 'updateOne':
        return await coll.updateOne(query.filter, query.update);
      case 'deleteOne':
        return await coll.deleteOne(query);
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }

  async close() {
    if (this.pgPool) {
      await this.pgPool.end();
    }
    if (this.mongoClient) {
      await this.mongoClient.close();
    }
  }
}
```

## Custom Memory Providers

### Vector Memory Provider

```typescript
// custom-vector-provider.ts
import { VectorMemoryProvider } from '../memory/vector-memory';
import { Embedding } from '../types/memory';

export class CustomVectorProvider implements VectorMemoryProvider {
  private vectors: Map<string, Embedding> = new Map();
  private index: any; // Your custom index implementation

  async store(id: string, vector: number[], metadata?: any): Promise<void> {
    const embedding: Embedding = {
      id,
      vector,
      metadata,
      timestamp: Date.now(),
    };

    this.vectors.set(id, embedding);
    await this.updateIndex(embedding);
  }

  async retrieve(id: string): Promise<Embedding | null> {
    return this.vectors.get(id) || null;
  }

  async search(
    queryVector: number[],
    limit: number = 10,
    threshold: number = 0.7
  ): Promise<Embedding[]> {
    const results: Array<{ embedding: Embedding; similarity: number }> = [];

    for (const embedding of this.vectors.values()) {
      const similarity = this.calculateSimilarity(queryVector, embedding.vector);
      if (similarity >= threshold) {
        results.push({ embedding, similarity });
      }
    }

    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map(r => r.embedding);
  }

  async delete(id: string): Promise<boolean> {
    const deleted = this.vectors.delete(id);
    if (deleted) {
      await this.removeFromIndex(id);
    }
    return deleted;
  }

  private calculateSimilarity(a: number[], b: number[]): number {
    // Cosine similarity implementation
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

  private async updateIndex(embedding: Embedding): Promise<void> {
    // Update your custom index
    // Implementation depends on your indexing strategy
  }

  private async removeFromIndex(id: string): Promise<void> {
    // Remove from your custom index
    // Implementation depends on your indexing strategy
  }
}
```

## Agent Extension Patterns

### Custom Agent Implementation

```typescript
// custom-agent.ts
import { Agent, AgentCapabilities } from '../agents/base-agent';
import { Task, TaskResult } from '../types/task';

export class CustomAnalysisAgent extends Agent {
  constructor() {
    super('custom-analysis-agent', {
      capabilities: [
        AgentCapabilities.DATA_ANALYSIS,
        AgentCapabilities.PATTERN_RECOGNITION,
        AgentCapabilities.REPORTING,
      ],
      maxConcurrentTasks: 5,
      priority: 1,
    });
  }

  async canHandle(task: Task): Promise<boolean> {
    return task.type === 'data-analysis' || task.type === 'pattern-recognition';
  }

  async execute(task: Task): Promise<TaskResult> {
    try {
      this.updateStatus('processing');
      
      const result = await this.performAnalysis(task);
      
      this.updateStatus('completed');
      return {
        success: true,
        data: result,
        metadata: {
          processingTime: Date.now() - task.createdAt,
          agent: this.id,
        },
      };
    } catch (error) {
      this.updateStatus('failed');
      return {
        success: false,
        error: error.message,
        metadata: {
          agent: this.id,
          failureReason: error.name,
        },
      };
    }
  }

  private async performAnalysis(task: Task): Promise<any> {
    const { data, analysisType } = task.parameters;
    
    switch (analysisType) {
      case 'statistical':
        return await this.performStatisticalAnalysis(data);
      case 'pattern':
        return await this.performPatternAnalysis(data);
      case 'trend':
        return await this.performTrendAnalysis(data);
      default:
        throw new Error(`Unsupported analysis type: ${analysisType}`);
    }
  }

  private async performStatisticalAnalysis(data: any): Promise<any> {
    // Custom statistical analysis implementation
    return {
      mean: this.calculateMean(data),
      median: this.calculateMedian(data),
      standardDeviation: this.calculateStandardDeviation(data),
      outliers: this.detectOutliers(data),
    };
  }

  private async performPatternAnalysis(data: any): Promise<any> {
    // Custom pattern analysis implementation
    return {
      patterns: this.identifyPatterns(data),
      frequency: this.calculateFrequency(data),
      correlations: this.findCorrelations(data),
    };
  }

  private async performTrendAnalysis(data: any): Promise<any> {
    // Custom trend analysis implementation
    return {
      trend: this.identifyTrend(data),
      seasonality: this.detectSeasonality(data),
      forecast: this.generateForecast(data),
    };
  }

  // Helper methods for analysis
  private calculateMean(data: number[]): number {
    return data.reduce((sum, value) => sum + value, 0) / data.length;
  }

  private calculateMedian(data: number[]): number {
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  private calculateStandardDeviation(data: number[]): number {
    const mean = this.calculateMean(data);
    const variance = data.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / data.length;
    return Math.sqrt(variance);
  }

  private detectOutliers(data: number[]): number[] {
    const mean = this.calculateMean(data);
    const stdDev = this.calculateStandardDeviation(data);
    const threshold = 2 * stdDev;
    
    return data.filter(value => Math.abs(value - mean) > threshold);
  }

  private identifyPatterns(data: any): any[] {
    // Pattern identification logic
    return [];
  }

  private calculateFrequency(data: any): Record<string, number> {
    // Frequency calculation logic
    return {};
  }

  private findCorrelations(data: any): any[] {
    // Correlation analysis logic
    return [];
  }

  private identifyTrend(data: any): string {
    // Trend identification logic
    return 'stable';
  }

  private detectSeasonality(data: any): any {
    // Seasonality detection logic
    return null;
  }

  private generateForecast(data: any): any[] {
    // Forecasting logic
    return [];
  }
}
```

## API Integration Strategies

### GraphQL Integration

```typescript
// graphql-integration.ts
import { GraphQLClient } from 'graphql-request';
import { gql } from 'graphql-tag';

export class GraphQLIntegration {
  private client: GraphQLClient;

  constructor(endpoint: string, headers?: Record<string, string>) {
    this.client = new GraphQLClient(endpoint, {
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
    });
  }

  async query(query: string, variables?: any) {
    try {
      return await this.client.request(query, variables);
    } catch (error) {
      console.error('GraphQL query failed:', error);
      throw error;
    }
  }

  async mutation(mutation: string, variables?: any) {
    try {
      return await this.client.request(mutation, variables);
    } catch (error) {
      console.error('GraphQL mutation failed:', error);
      throw error;
    }
  }

  // Example queries
  async getUserData(userId: string) {
    const query = gql`
      query GetUser($userId: ID!) {
        user(id: $userId) {
          id
          name
          email
          profile {
            bio
            avatar
          }
        }
      }
    `;

    return await this.query(query, { userId });
  }

  async createUser(userData: any) {
    const mutation = gql`
      mutation CreateUser($input: UserInput!) {
        createUser(input: $input) {
          id
          name
          email
        }
      }
    `;

    return await this.mutation(mutation, { input: userData });
  }
}
```

### WebSocket Integration

```typescript
// websocket-integration.ts
import WebSocket from 'ws';
import { EventEmitter } from 'events';

export class WebSocketIntegration extends EventEmitter {
  private ws?: WebSocket;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(private url: string, private options?: WebSocket.ClientOptions) {
    super();
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url, this.options);

      this.ws.on('open', () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected');
        resolve();
      });

      this.ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.emit('message', message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      });

      this.ws.on('close', () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected');
        this.attemptReconnect();
      });

      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
        reject(error);
      });
    });
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket not connected');
    }
  }

  close(): void {
    if (this.ws) {
      this.ws.close();
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect().catch(console.error);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }
}

// Usage in MCP server
export class WebSocketMCPServer {
  private wsIntegration: WebSocketIntegration;

  constructor(wsUrl: string) {
    this.wsIntegration = new WebSocketIntegration(wsUrl);
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.wsIntegration.on('message', (message) => {
      this.handleWebSocketMessage(message);
    });

    this.wsIntegration.on('connected', () => {
      console.log('WebSocket integration connected');
    });

    this.wsIntegration.on('disconnected', () => {
      console.log('WebSocket integration disconnected');
    });
  }

  private handleWebSocketMessage(message: any): void {
    // Process incoming WebSocket messages
    console.log('Received WebSocket message:', message);
    
    // Forward to SAFLA system if needed
    this.forwardToSAFLA(message);
  }

  private forwardToSAFLA(message: any): void {
    // Implementation to forward messages to SAFLA
  }

  async start(): Promise<void> {
    await this.wsIntegration.connect();
  }

  stop(): void {
    this.wsIntegration.close();
  }
}
```

## Event-Driven Integrations

### Event Bus Integration

```typescript
// event-bus-integration.ts
import { EventEmitter } from 'events';
import { Redis } from 'ioredis';

export class EventBusIntegration extends EventEmitter {
  private redis: Redis;
  private subscriber: Redis;

  constructor(redisUrl: string) {
    super();
    this.redis = new Redis(redisUrl);
    this.subscriber = new Redis(redisUrl);
    this.setupSubscriber();
  }

  private setupSubscriber(): void {
    this.subscriber.on('message', (channel, message) => {
      try {
        const event = JSON.parse(message);
        this.emit('event', { channel, event });
      } catch (error) {
        console.error('Failed to parse event message:', error);
      }
    });
  }

  async subscribe(channels: string[]): Promise<void> {
    await this.subscriber.subscribe(...channels);
  }

  async publish(channel: string, event: any): Promise<void> {
    await this.redis.publish(channel, JSON.stringify(event));
  }

  async unsubscribe(channels: string[]): Promise<void> {
    await this.subscriber.unsubscribe(...channels);
  }

  async close(): Promise<void> {
    await this.redis.quit();
    await this.subscriber.quit();
  }
}

// Usage example
const eventBus = new EventBusIntegration('redis://localhost:6379');

eventBus.on('event', ({ channel, event }) => {
  console.log(`Received event on ${channel}:`, event);
  
  // Process event based on channel
  switch (channel) {
    case 'safla.memory.update':
      handleMemoryUpdate(event);
      break;
    case 'safla.agent.task':
      handleAgentTask(event);
      break;
    default:
      console.log('Unknown event channel:', channel);
  }
});

function handleMemoryUpdate(event: any): void {
  // Handle memory update events
}

function handleAgentTask(event: any): void {
  // Handle agent task events
}
```

## Security Considerations

### Authentication and Authorization

```typescript
// security-integration.ts
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { RateLimiterRedis } from 'rate-limiter-flexible';

export class SecurityIntegration {
  private jwtSecret: string;
  private rateLimiter: RateLimiterRedis;

  constructor(jwtSecret: string, redisClient: any) {
    this.jwtSecret = jwtSecret;
    this.rateLimiter = new RateLimiterRedis({
      storeClient: redisClient,
      keyPrefix: 'safla_rate_limit',
      points: 100, // Number of requests
      duration: 3600, // Per hour
    });
  }

  async authenticateToken(token: string): Promise<any> {
    try {
      return jwt.verify(token, this.jwtSecret);
    } catch (error) {
      throw new Error('Invalid token');
    }
  }

  async generateToken(payload: any, expiresIn: string = '1h'): Promise<string> {
    return jwt.sign(payload, this.jwtSecret, { expiresIn });
  }

  async hashPassword(password: string): Promise<string> {
    const saltRounds = 12;
    return await bcrypt.hash(password, saltRounds);
  }

  async verifyPassword(password: string, hash: string): Promise<boolean> {
    return await bcrypt.compare(password, hash);
  }

  async checkRateLimit(identifier: string): Promise<void> {
    try {
      await this.rateLimiter.consume(identifier);
    } catch (rejRes) {
      throw new Error('Rate limit exceeded');
    }
  }

  validateInput(input: any, schema: any): boolean {
    // Input validation logic
    // Use libraries like Joi or Yup for comprehensive validation
    return true;
  }

  sanitizeInput(input: string): string {
    // Input sanitization logic
    return input.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
  }
}
```

### Encryption and Data Protection

```typescript
// encryption-integration.ts
import crypto from 'crypto';

export class EncryptionIntegration {
  private algorithm = 'aes-256-gcm';
  private keyLength = 32;
  private ivLength = 16;
  private tagLength = 16;

  constructor(private masterKey: string) {}

  encrypt(data: string): { encrypted: string; iv: string; tag: string } {
    const iv = crypto.randomBytes(this.ivLength);
    const key = crypto.scryptSync(this.masterKey, 'salt', this.keyLength);
    const cipher = crypto.createCipherGCM(this.algorithm, key, iv);
    
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const tag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      tag: tag.toString('hex'),
    };
  }

  decrypt(encrypted: string, iv: string, tag: string): string {
    const key = crypto.scryptSync(this.masterKey, 'salt', this.keyLength);
    const decipher = crypto.createDecipherGCM(
      this.algorithm,
      key,
      Buffer.from(iv, 'hex')
    );
    
    decipher.setAuthTag(Buffer.from(tag, 'hex'));
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }

  generateHash(data: string): string {
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  generateSecureRandom(length: number): string {
    return crypto.randomBytes(length).toString('hex');
  }
}
```

## Testing Integration Points

### Integration Test Framework

```typescript
// integration-test-framework.ts
import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { TestContainer } from './test-container';

export class IntegrationTestFramework {
  private testContainer: TestContainer;

  constructor() {
    this.testContainer = new TestContainer();
  }

  async setup(): Promise<void> {
    await this.testContainer.start();
  }

  async teardown(): Promise<void> {
    await this.testContainer.stop();
  }

  createMCPServerTest(serverName: string, serverPath: string) {
    return describe(`MCP Server: ${serverName}`, () => {
      let server: any;

      beforeAll(async () => {
        server = await this.testContainer.startMCPServer(serverPath);
      });

      afterAll(async () => {
        await this.testContainer.stopMCPServer(server);
      });

      it('should list available tools', async () => {
        const tools = await server.listTools();
        expect(tools).toBeDefined();
        expect(Array.isArray(tools.tools)).toBe(true);
      });

      it('should handle tool calls', async () => {
        const tools = await server.listTools();
        if (tools.tools.length > 0) {
          const tool = tools.tools[0];
          const result = await server.callTool(tool.name, {});
          expect(result).toBeDefined();
        }
      });

      it('should list available resources', async () => {
        const resources = await server.listResources();
        expect(resources).toBeDefined();
      });
    });
  }

  createAPIIntegrationTest(apiName: string, apiConfig: any) {
    return describe(`API Integration: ${apiName}`, () => {
      let api: any;

      beforeAll(async () => {
        api = await this.testContainer.createAPIIntegration(apiConfig);
      });

      it('should authenticate successfully', async () => {
        const result = await api.authenticate();
        expect(result.success).toBe(true);
      });

      it('should handle GET requests', async () => {
        const result = await api.get('/test');
        expect(result).toBeDefined();
      });

      it('should handle POST requests', async () => {
        const result = await api.post('/test', { data: 'test' });
        expect(result).toBeDefined();
      });

      it('should handle rate limiting', async () => {
        // Test rate limiting behavior
        const promises = Array(10).fill(null).map(() => api.get('/test'));
        const results = await Promise.allSettled(promises);
        
        // Some requests should succeed, others might be rate limited
        const successful = results.filter(r => r.status === 'fulfilled').length;
        expect(successful).toBeGreaterThan(0);
      });
    });
  }

  createDatabaseIntegrationTest(dbName: string, dbConfig: any) {
    return describe(`Database Integration: ${dbName}`, () => {
      let db: any;

      beforeAll(async () => {
        db = await this.testContainer.createDatabaseIntegration(dbConfig);
        await db.migrate();
      });

      afterAll(async () => {
        await db.cleanup();
      });

      it('should connect successfully', async () => {
        const isConnected = await db.isConnected();
        expect(isConnected).toBe(true);
      });

      it('should perform CRUD operations', async () => {
        // Create
        const created = await db.create('test_table', { name: 'test' });
        expect(created.id).toBeDefined();

        // Read
        const read = await db.findById('test_table', created.id);
        expect(read.name).toBe('test');

        // Update
        const updated = await db.update('test_table', created.id, { name: 'updated' });
        expect(updated.name).toBe('updated');

        // Delete
        const deleted = await db.delete('test_table', created.id);
        expect(deleted).toBe(true);
      });
    });
  }
}

// Usage
const testFramework = new IntegrationTestFramework();

beforeAll(async () => {
  await testFramework.setup();
});

afterAll(async () => {
  await testFramework.teardown();
});

// Create tests for different integrations
testFramework.createMCPServerTest('Custom Analysis', './dist/custom-mcp-server.js');
testFramework.createAPIIntegrationTest('External API', {
  baseURL: 'https://api.example.com',
  apiKey: 'test-key',
});
testFramework.createDatabaseIntegrationTest('PostgreSQL', {
  type: 'postgresql',
  connectionString: 'postgresql://test:test@localhost:5432/test',
});
```

## Best Practices

### 1. Error Handling
- Implement comprehensive error handling with proper error types
- Use circuit breakers for external service calls
- Implement retry logic with exponential backoff
- Log errors with sufficient context for debugging

### 2. Performance Optimization
- Implement caching strategies for frequently accessed data
- Use connection pooling for database connections
- Implement rate limiting to prevent abuse
- Monitor performance metrics and set up alerts

### 3. Security
- Always validate and sanitize input data
- Use secure authentication and authorization mechanisms
- Encrypt sensitive data at rest and in transit
- Implement proper access controls and audit logging

### 4. Monitoring and Observability
- Implement structured logging with correlation IDs
- Set up health checks for all integration points
- Monitor key performance indicators (KPIs)
- Implement distributed tracing for complex workflows

### 5. Testing
- Write comprehensive integration tests
- Use test containers for isolated testing environments
- Implement contract testing for API integrations
- Perform load testing for performance validation

### 6. Documentation
- Document all integration points and their configurations
- Provide clear examples and usage patterns
- Maintain up-to-date API documentation
- Document troubleshooting procedures

---

This comprehensive integration patterns guide provides the foundation for extending SAFLA with custom integrations, MCP servers, and third-party services while maintaining security, performance, and reliability standards.