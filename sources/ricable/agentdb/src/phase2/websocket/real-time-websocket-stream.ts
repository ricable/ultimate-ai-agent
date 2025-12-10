/**
 * WebSocket-based Real-time Streaming with Sub-second Latency
 * Phase 2: High-Performance Real-time Communication for Multi-Agent ML Workflows
 */

import { StreamProcessor, StreamContext, StreamType } from '../stream-chain-core';
import { AgentDB } from '../../agentdb/agentdb-core';
import { TemporalReasoningCore } from '../../temporal/temporal-core';
import { WebSocket, WebSocketServer } from 'ws';
import { EventEmitter } from 'events';

// WebSocket Streaming Interfaces
export interface WebSocketStreamConfig {
  server: WebSocketServerConfig;
  connection: WebSocketConnectionConfig;
  messageHandling: MessageHandlingConfig;
  performance: WebSocketPerformanceConfig;
  security: WebSocketSecurityConfig;
  monitoring: WebSocketMonitoringConfig;
}

export interface WebSocketServerConfig {
  port: number;
  host: string;
  maxConnections: number;
  backlog: number;
  perMessageDeflate: boolean;
  compression: CompressionConfig;
  ssl: SSLConfig;
}

export interface CompressionConfig {
  enabled: boolean;
  threshold: number;
  level: CompressionLevel;
  strategy: CompressionStrategy;
}

export enum CompressionLevel {
  NONE = 0,
  LOW = 1,
  MEDIUM = 3,
  HIGH = 6,
  MAXIMUM = 9
}

export enum CompressionStrategy {
  SHARED_COMPRESSOR = 'shared_compressor',
  DEDICATED_COMPRESSOR = 'dedicated_compressor',
  ADAPTIVE = 'adaptive'
}

export interface SSLConfig {
  enabled: boolean;
  certFile: string;
  keyFile: string;
  caFile?: string;
  passphrase?: string;
  rejectUnauthorized: boolean;
}

export interface WebSocketConnectionConfig {
  heartbeat: HeartbeatConfig;
  timeout: TimeoutConfig;
  reconnect: ReconnectConfig;
  rateLimiting: RateLimitingConfig;
  loadBalancing: LoadBalancingConfig;
}

export interface HeartbeatConfig {
  enabled: boolean;
  interval: number;
  timeout: number;
  message: string;
}

export interface TimeoutConfig {
  connect: number;
  idle: number;
  send: number;
  receive: number;
}

export interface ReconnectConfig {
  enabled: boolean;
  maxAttempts: number;
  backoffStrategy: BackoffStrategy;
  initialDelay: number;
  maxDelay: number;
  factor: number;
}

export enum BackoffStrategy {
  LINEAR = 'linear',
  EXPONENTIAL = 'exponential',
  FIXED = 'fixed',
  ADAPTIVE = 'adaptive'
}

export interface RateLimitingConfig {
  enabled: boolean;
  windowMs: number;
  maxMessages: number;
  strategy: RateLimitingStrategy;
}

export enum RateLimitingStrategy {
  FIXED_WINDOW = 'fixed_window',
  SLIDING_WINDOW = 'sliding_window',
  TOKEN_BUCKET = 'token_bucket',
  LEAKY_BUCKET = 'leaky_bucket'
}

export interface LoadBalancingConfig {
  enabled: boolean;
  algorithm: LoadBalancingAlgorithm;
  healthChecks: boolean;
  stickySessions: boolean;
}

export enum LoadBalancingAlgorithm {
  ROUND_ROBIN = 'round_robin',
  LEAST_CONNECTIONS = 'least_connections',
  WEIGHTED_ROUND_ROBIN = 'weighted_round_robin',
  HASH_BASED = 'hash_based',
  RANDOM = 'random'
}

export interface MessageHandlingConfig {
  serialization: SerializationConfig;
  validation: ValidationConfig;
  routing: MessageRoutingConfig;
  batching: MessageBatchingConfig;
  ordering: MessageOrderingConfig;
}

export interface SerializationConfig {
  format: SerializationFormat;
  compression: boolean;
  encryption: boolean;
  schemaValidation: boolean;
}

export enum SerializationFormat {
  JSON = 'json',
  MSGPACK = 'msgpack',
  PROTOBUF = 'protobuf',
  AVRO = 'avro',
  FLATBUFFERS = 'flatbuffers'
}

export interface ValidationConfig {
  enabled: boolean;
  schema: any;
  strictMode: boolean;
  customValidators: CustomValidator[];
}

export interface CustomValidator {
  name: string;
  validator: (message: any) => boolean;
  errorMessage: string;
}

export interface MessageRoutingConfig {
  strategy: RoutingStrategy;
  topics: TopicConfig[];
  subscriptions: SubscriptionConfig[];
}

export enum RoutingStrategy {
  BROADCAST = 'broadcast',
  TOPIC_BASED = 'topic_based',
  DIRECT = 'direct',
  FAN_OUT = 'fan_out',
  REQUEST_REPLY = 'request_reply'
}

export interface TopicConfig {
  name: string;
  pattern: string;
  persistent: boolean;
  maxSubscribers: number;
  retention: RetentionPolicy;
}

export interface RetentionPolicy {
  enabled: boolean;
  size: number;
  time: number;
  strategy: RetentionStrategy;
}

export enum RetentionStrategy {
  DELETE_OLDEST = 'delete_oldest',
  DELETE_ALL = 'delete_all',
  COMPRESS = 'compress'
}

export interface SubscriptionConfig {
  topic: string;
  subscriberId: string;
  filter: MessageFilter;
  qos: QualityOfService;
}

export interface MessageFilter {
  type: FilterType;
  expression: string;
  parameters: any;
}

export enum FilterType {
  JAVASCRIPT = 'javascript',
  JSON_PATH = 'json_path',
  SQL = 'sql',
  CUSTOM = 'custom'
}

export interface QualityOfService {
  level: QoSLevel;
  maxRetries: number;
  timeout: number;
}

export enum QoSLevel {
  AT_MOST_ONCE = 'at_most_once',
  AT_LEAST_ONCE = 'at_least_once',
  EXACTLY_ONCE = 'exactly_once'
}

export interface MessageBatchingConfig {
  enabled: boolean;
  maxBatchSize: number;
  maxWaitTime: number;
  aggregation: AggregationStrategy;
}

export enum AggregationStrategy {
  TIME_BASED = 'time_based',
  SIZE_BASED = 'size_based',
  HYBRID = 'hybrid'
}

export interface MessageOrderingConfig {
  enabled: boolean;
  strategy: OrderingStrategy;
  maxOutOfOrder: number;
}

export enum OrderingStrategy {
  FIFO = 'fifo',
  PRIORITY = 'priority',
  SEQUENTIAL = 'sequential',
  CAUSAL = 'causal'
}

export interface WebSocketPerformanceConfig {
  buffering: BufferingConfig;
  multiplexing: MultiplexingConfig;
  pipelining: PipeliningConfig;
  connectionPooling: ConnectionPoolingConfig;
  optimization: OptimizationConfig;
}

export interface BufferingConfig {
  enabled: boolean;
  strategy: BufferingStrategy;
  maxSize: number;
  flushInterval: number;
  highWaterMark: number;
  lowWaterMark: number;
}

export enum BufferingStrategy {
  NO_BUFFERING = 'no_buffering',
  MESSAGE_BUFFERING = 'message_buffering',
  STREAM_BUFFERING = 'stream_buffering',
  ADAPTIVE_BUFFERING = 'adaptive_buffering'
}

export interface MultiplexingConfig {
  enabled: boolean;
  maxStreams: number;
  streamTimeout: number;
  priority: boolean;
  loadBalancing: boolean;
}

export interface PipeliningConfig {
  enabled: boolean;
  maxInFlight: number;
  batchSize: number;
  timeout: number;
  retryPolicy: RetryPolicy;
}

export interface RetryPolicy {
  maxAttempts: number;
  backoffMs: number;
  retryableErrors: string[];
}

export interface ConnectionPoolingConfig {
  enabled: boolean;
  maxConnections: number;
  minConnections: number;
  maxIdleTime: number;
  validationQuery: string;
}

export interface OptimizationConfig {
  latencyOptimization: LatencyOptimizationConfig;
  throughputOptimization: ThroughputOptimizationConfig;
  memoryOptimization: MemoryOptimizationConfig;
  cpuOptimization: CPUOptimizationConfig;
}

export interface LatencyOptimizationConfig {
  targetLatency: number;
  enableNagle: boolean;
  enableQuickAck: boolean;
  bufferSize: number;
  tcpNoDelay: boolean;
}

export interface ThroughputOptimizationConfig {
  targetThroughput: number;
  enableCompression: boolean;
  maxMessageSize: number;
  windowSize: number;
}

export interface MemoryOptimizationConfig {
  maxMemoryUsage: number;
  enableGarbageCollection: boolean;
  gcThreshold: number;
  objectPooling: boolean;
}

export interface CPUOptimizationConfig {
  enableSIMD: boolean;
  enableVectorization: boolean;
  maxWorkers: number;
  threadAffinity: boolean;
}

export interface WebSocketSecurityConfig {
  authentication: AuthenticationConfig;
  authorization: AuthorizationConfig;
  encryption: EncryptionConfig;
  rateLimit: SecurityRateLimitConfig;
  cors: CORSConfig;
}

export interface AuthenticationConfig {
  enabled: boolean;
  methods: AuthenticationMethod[];
  tokenValidation: TokenValidationConfig;
  sessionManagement: SessionManagementConfig;
}

export enum AuthenticationMethod {
  JWT = 'jwt',
  API_KEY = 'api_key',
  OAUTH = 'oauth',
  BASIC_AUTH = 'basic_auth',
  CERTIFICATE = 'certificate'
}

export interface TokenValidationConfig {
  algorithm: string;
  issuer: string;
  audience: string;
  clockTolerance: number;
}

export interface SessionManagementConfig {
  enabled: boolean;
  store: SessionStore;
  ttl: number;
  refreshEnabled: boolean;
}

export enum SessionStore {
  MEMORY = 'memory',
  REDIS = 'redis',
  DATABASE = 'database',
  FILE = 'file'
}

export interface AuthorizationConfig {
  enabled: boolean;
  rbac: RBACConfig;
  resourceBased: ResourceBasedConfig;
  policyEngine: PolicyEngineConfig;
}

export interface RBACConfig {
  enabled: boolean;
  roles: RoleConfig[];
  permissions: PermissionConfig[];
}

export interface RoleConfig {
  name: string;
  permissions: string[];
  inherits?: string[];
}

export interface PermissionConfig {
  name: string;
  resource: string;
  action: string;
  conditions?: any;
}

export interface ResourceBasedConfig {
  enabled: boolean;
  policies: PolicyConfig[];
}

export interface PolicyConfig {
  name: string;
  effect: PolicyEffect;
  resources: string[];
  actions: string[];
  conditions?: any;
}

export enum PolicyEffect {
  ALLOW = 'allow',
  DENY = 'deny'
}

export interface PolicyEngineConfig {
  engine: PolicyEngine;
  decisionCache: boolean;
  auditLogging: boolean;
}

export enum PolicyEngine {
  OPEN_POLICY_AGENT = 'open_policy_agent',
  CASBIN = 'casbin',
  OPA = 'opa',
  CUSTOM = 'custom'
}

export interface EncryptionConfig {
  enabled: boolean;
  algorithm: EncryptionAlgorithm;
  keyRotation: KeyRotationConfig;
  messageEncryption: boolean;
}

export enum EncryptionAlgorithm {
  AES_256_GCM = 'aes_256_gcm',
  CHACHA20_POLY1305 = 'chacha20_poly1305',
  AES_128_CBC = 'aes_128_cbc'
}

export interface KeyRotationConfig {
  enabled: boolean;
  interval: number;
  keyDerivation: boolean;
}

export interface SecurityRateLimitConfig {
  enabled: boolean;
  windowMs: number;
  maxRequests: number;
  skipSuccessfulRequests: boolean;
  skipFailedRequests: boolean;
}

export interface CORSConfig {
  enabled: boolean;
  origin: string[];
  methods: string[];
  allowedHeaders: string[];
  credentials: boolean;
  maxAge: number;
}

export interface WebSocketMonitoringConfig {
  metrics: MetricsConfig;
  logging: LoggingConfig;
  tracing: TracingConfig;
  alerting: AlertingConfig;
  healthChecks: HealthCheckConfig;
}

export interface MetricsConfig {
  enabled: boolean;
  collectInterval: number;
  exportFormat: MetricsExportFormat;
  labels: MetricsLabelsConfig;
}

export enum MetricsExportFormat {
  PROMETHEUS = 'prometheus',
  INFLUXDB = 'influxdb',
  DATADOG = 'datadog',
  STATSD = 'statsd'
}

export interface MetricsLabelsConfig {
  includeConnectionId: boolean;
  includeClientId: boolean;
  includeTopic: boolean;
  includeMessageType: boolean;
  customLabels: { [key: string]: string };
}

export interface LoggingConfig {
  enabled: boolean;
  level: LogLevel;
  format: LogFormat;
  includePayloads: boolean;
  maxPayloadSize: number;
}

export interface TracingConfig {
  enabled: boolean;
  samplingRate: number;
  exportFormat: TracingExportFormat;
  includePayloads: boolean;
}

export enum TracingExportFormat {
  JAEGER = 'jaeger',
  ZIPKIN = 'zipkin',
  OPENTELEMETRY = 'opentelemetry',
  DATADOG = 'datadog'
}

export interface AlertingConfig {
  enabled: boolean;
  channels: AlertChannel[];
  rules: AlertRule[];
  thresholds: AlertThresholds;
}

export interface AlertChannel {
  type: ChannelType;
  configuration: ChannelConfiguration;
  enabled: boolean;
}

export interface AlertRule {
  name: string;
  condition: string;
  threshold: number;
  duration: number;
  severity: AlertSeverity;
  enabled: boolean;
}

export interface AlertThresholds {
  connectionCount: number;
  messageRate: number;
  latency: number;
  errorRate: number;
  memoryUsage: number;
  cpuUsage: number;
}

export interface HealthCheckConfig {
  enabled: boolean;
  interval: number;
  timeout: number;
  endpoints: HealthCheckEndpoint[];
}

export interface HealthCheckEndpoint {
  path: string;
  method: string;
  expectedStatus: number;
  timeout: number;
}

// WebSocket Real-time Streaming Implementation
export class RealTimeWebSocketStream extends EventEmitter {
  private config: WebSocketStreamConfig;
  private agentDB: AgentDB;
  private temporalCore: TemporalReasoningCore;
  private server: WebSocketServer;
  private connections: Map<string, WebSocketConnection> = new Map();
  private messageHandler: MessageHandler;
  private connectionManager: ConnectionManager;
  private performanceMonitor: WebSocketPerformanceMonitor;
  private securityManager: WebSocketSecurityManager;

  constructor(
    agentDB: AgentDB,
    temporalCore: TemporalReasoningCore,
    config: Partial<WebSocketStreamConfig> = {}
  ) {
    super();
    this.agentDB = agentDB;
    this.temporalCore = temporalCore;
    this.config = this.mergeWithDefaults(config);

    this.messageHandler = new MessageHandler(this.config.messageHandling);
    this.connectionManager = new ConnectionManager(this.config.connection);
    this.performanceMonitor = new WebSocketPerformanceMonitor(this.config.monitoring);
    this.securityManager = new WebSocketSecurityManager(this.config.security);
  }

  // Initialize WebSocket server
  async initialize(): Promise<void> {
    console.log('Initializing Real-time WebSocket Stream...');

    try {
      // Initialize WebSocket server
      await this.initializeServer();

      // Initialize components
      await this.messageHandler.initialize();
      await this.connectionManager.initialize();
      await this.performanceMonitor.initialize();
      await this.securityManager.initialize();

      // Setup event handlers
      this.setupEventHandlers();

      // Start performance monitoring
      await this.performanceMonitor.start();

      console.log(`WebSocket server initialized on ${this.config.server.host}:${this.config.server.port}`);

    } catch (error) {
      console.error('Failed to initialize Real-time WebSocket Stream:', error);
      throw error;
    }
  }

  // Create WebSocket stream processor
  createWebSocketStreamProcessor(): StreamProcessor {
    return {
      process: async (data: any, context: StreamContext): Promise<any> => {
        const startTime = Date.now();
        const messageId = this.generateMessageId();

        try {
          // Create WebSocket message
          const message: WebSocketMessage = {
            id: messageId,
            type: MessageType.DATA,
            timestamp: new Date(),
            payload: data,
            metadata: {
              correlationId: context.correlationId,
              agentId: context.agentId,
              pipelineId: context.pipelineId,
              priority: MessagePriority.NORMAL,
              ttl: 30000 // 30 seconds
            }
          };

          // Serialize message
          const serializedMessage = await this.messageHandler.serialize(message);

          // Broadcast to all connected clients
          const broadcastResult = await this.broadcastMessage(serializedMessage, message.metadata);

          // Store in AgentDB for persistence
          await this.storeMessage(message, context);

          const processingTime = Date.now() - startTime;

          // Record metrics
          await this.performanceMonitor.recordMessage(message, processingTime, true);

          return {
            messageId,
            broadcastResult,
            processingTime,
            success: true
          };

        } catch (error) {
          const processingTime = Date.now() - startTime;

          // Record error metrics
          await this.performanceMonitor.recordError(messageId, error, processingTime);

          throw error;
        }
      }
    };
  }

  // Create real-time subscription handler
  createSubscriptionHandler(): SubscriptionHandler {
    return {
      async subscribe(topic: string, filter?: MessageFilter): Promise<string> {
        const subscriptionId = this.generateSubscriptionId();

        const subscription: Subscription = {
          id: subscriptionId,
          topic,
          filter,
          createdAt: new Date(),
          active: true
        };

        // Register subscription
        await this.messageHandler.addSubscription(subscription);

        console.log(`Subscription created: ${subscriptionId} for topic: ${topic}`);

        return subscriptionId;
      },

      async unsubscribe(subscriptionId: string): Promise<void> {
        await this.messageHandler.removeSubscription(subscriptionId);
        console.log(`Subscription removed: ${subscriptionId}`);
      },

      async publish(topic: string, message: any, options?: PublishOptions): Promise<PublishResult> {
        const startTime = Date.now();

        try {
          const wsMessage: WebSocketMessage = {
            id: this.generateMessageId(),
            type: MessageType.DATA,
            timestamp: new Date(),
            payload: message,
            metadata: {
              topic,
              correlationId: options?.correlationId,
              agentId: options?.agentId,
              priority: options?.priority || MessagePriority.NORMAL,
              ttl: options?.ttl || 30000
            }
          };

          // Serialize message
          const serializedMessage = await this.messageHandler.serialize(wsMessage);

          // Send to subscribed clients
          const result = await this.sendToTopic(topic, serializedMessage, wsMessage.metadata);

          const processingTime = Date.now() - startTime;

          await this.performanceMonitor.recordMessage(wsMessage, processingTime, true);

          return {
            messageId: wsMessage.id,
            subscribersReached: result.subscriberCount,
            processingTime,
            success: true
          };

        } catch (error) {
          const processingTime = Date.now() - startTime;

          await this.performanceMonitor.recordError(
            options?.correlationId || 'unknown',
            error,
            processingTime
          );

          throw error;
        }
      }
    };
  }

  // Create real-time bi-directional handler
  createBidirectionalHandler(): BidirectionalHandler {
    return {
      async handleRequest(request: any, options?: RequestOptions): Promise<any> {
        const requestId = this.generateRequestId();
        const startTime = Date.now();

        try {
          // Create request message
          const requestMessage: WebSocketMessage = {
            id: requestId,
            type: MessageType.REQUEST,
            timestamp: new Date(),
            payload: request,
            metadata: {
              requestId,
              correlationId: options?.correlationId,
              agentId: options?.agentId,
              expectsReply: true,
              timeout: options?.timeout || 30000
            }
          };

          // Serialize and send request
          const serializedMessage = await this.messageHandler.serialize(requestMessage);

          // Send to appropriate handler
          const response = await this.sendAndWaitForResponse(
            serializedMessage,
            requestMessage.metadata
          );

          const processingTime = Date.now() - startTime;

          await this.performanceMonitor.recordMessage(requestMessage, processingTime, true);

          return response;

        } catch (error) {
          const processingTime = Date.now() - startTime;

          await this.performanceMonitor.recordError(requestId, error, processingTime);

          throw error;
        }
      },

      async onResponse(response: any, requestId: string): Promise<void> {
        // Handle response message
        const responseMessage: WebSocketMessage = {
          id: this.generateMessageId(),
          type: MessageType.RESPONSE,
          timestamp: new Date(),
          payload: response,
          metadata: {
            requestId,
            correlationId: requestId,
            isResponse: true
          }
        };

        const serializedMessage = await this.messageHandler.serialize(responseMessage);
        await this.sendResponse(serializedMessage, responseMessage.metadata);
      }
    };
  }

  // Private helper methods
  private async initializeServer(): Promise<void> {
    const options: any = {
      port: this.config.server.port,
      host: this.config.server.host,
      backlog: this.config.server.backlog,
      perMessageDeflate: this.config.server.perMessageDeflate
    };

    // Add SSL configuration if enabled
    if (this.config.server.ssl.enabled) {
      options.server = {
        cert: await this.readFile(this.config.server.ssl.certFile),
        key: await this.readFile(this.config.server.ssl.keyFile),
        ca: this.config.server.ssl.caFile ? await this.readFile(this.config.server.ssl.caFile) : undefined,
        passphrase: this.config.server.ssl.passphrase,
        rejectUnauthorized: this.config.server.ssl.rejectUnauthorized
      };
    }

    this.server = new WebSocketServer(options);
  }

  private setupEventHandlers(): void {
    this.server.on('connection', (ws: WebSocket, req: any) => {
      this.handleConnection(ws, req);
    });

    this.server.on('error', (error: Error) => {
      console.error('WebSocket server error:', error);
      this.emit('serverError', error);
    });

    this.server.on('listening', () => {
      console.log(`WebSocket server listening on ${this.config.server.host}:${this.config.server.port}`);
      this.emit('serverListening');
    });
  }

  private async handleConnection(ws: WebSocket, req: any): Promise<void> {
    const connectionId = this.generateConnectionId();
    const clientIP = req.socket.remoteAddress;

    console.log(`New WebSocket connection: ${connectionId} from ${clientIP}`);

    try {
      // Authenticate connection
      const authResult = await this.securityManager.authenticate(ws, req);
      if (!authResult.success) {
        ws.close(1008, 'Authentication failed');
        return;
      }

      // Create connection object
      const connection: WebSocketConnection = {
        id: connectionId,
        ws,
        ip: clientIP,
        userAgent: req.headers['user-agent'],
        connectedAt: new Date(),
        lastActivity: new Date(),
        authenticated: authResult.authenticated,
        userId: authResult.userId,
        subscriptions: new Set(),
        messageCount: 0,
        bytesReceived: 0,
        bytesSent: 0
      };

      this.connections.set(connectionId, connection);

      // Setup connection event handlers
      this.setupConnectionHandlers(connection);

      // Start heartbeat if enabled
      if (this.config.connection.heartbeat.enabled) {
        this.startHeartbeat(connection);
      }

      // Accept connection
      this.emit('connection', connection);

      // Send welcome message
      const welcomeMessage: WebSocketMessage = {
        id: this.generateMessageId(),
        type: MessageType.SYSTEM,
        timestamp: new Date(),
        payload: {
          type: 'welcome',
          connectionId,
          serverTime: new Date(),
          capabilities: this.getServerCapabilities()
        },
        metadata: {
          priority: MessagePriority.HIGH
        }
      };

      const serializedWelcome = await this.messageHandler.serialize(welcomeMessage);
      ws.send(serializedWelcome);

    } catch (error) {
      console.error(`Failed to handle connection ${connectionId}:`, error);
      ws.close(1011, 'Internal server error');
    }
  }

  private setupConnectionHandlers(connection: WebSocketConnection): void {
    const ws = connection.ws;

    ws.on('message', async (data: Buffer) => {
      await this.handleMessage(connection, data);
    });

    ws.on('close', (code: number, reason: string) => {
      this.handleDisconnection(connection, code, reason);
    });

    ws.on('error', (error: Error) => {
      console.error(`WebSocket error for connection ${connection.id}:`, error);
      this.handleConnectionError(connection, error);
    });

    ws.on('pong', () => {
      connection.lastActivity = new Date();
    });
  }

  private async handleMessage(connection: WebSocketConnection, data: Buffer): Promise<void> {
    const startTime = Date.now();

    try {
      connection.lastActivity = new Date();
      connection.messageCount++;
      connection.bytesReceived += data.length;

      // Deserialize message
      const message = await this.messageHandler.deserialize(data);

      // Validate message
      const validationResult = await this.messageHandler.validate(message);
      if (!validationResult.valid) {
        await this.sendErrorMessage(connection, 'Invalid message format', validationResult.errors);
        return;
      }

      // Route message based on type
      switch (message.type) {
        case MessageType.SUBSCRIBE:
          await this.handleSubscribe(connection, message);
          break;

        case MessageType.UNSUBSCRIBE:
          await this.handleUnsubscribe(connection, message);
          break;

        case MessageType.REQUEST:
          await this.handleRequest(connection, message);
          break;

        case MessageType.RESPONSE:
          await this.handleResponse(connection, message);
          break;

        case MessageType.DATA:
          await this.handleDataMessage(connection, message);
          break;

        case MessageType.HEARTBEAT:
          await this.handleHeartbeat(connection, message);
          break;

        default:
          console.warn(`Unknown message type: ${message.type}`);
      }

      const processingTime = Date.now() - startTime;
      await this.performanceMonitor.recordMessage(message, processingTime, true);

    } catch (error) {
      const processingTime = Date.now() - startTime;
      await this.performanceMonitor.recordError(connection.id, error, processingTime);
      await this.sendErrorMessage(connection, 'Message processing failed', [error.message]);
    }
  }

  private async handleSubscribe(connection: WebSocketConnection, message: WebSocketMessage): Promise<void> {
    const { topic, filter } = message.payload;

    // Check authorization
    const authResult = await this.securityManager.authorizeSubscription(
      connection.userId,
      topic,
      'subscribe'
    );

    if (!authResult.authorized) {
      await this.sendErrorMessage(connection, 'Subscription not authorized', [authResult.reason]);
      return;
    }

    // Add subscription
    const subscriptionId = await this.messageHandler.addSubscription({
      id: this.generateSubscriptionId(),
      topic,
      filter,
      connectionId: connection.id,
      createdAt: new Date(),
      active: true
    });

    connection.subscriptions.add(subscriptionId);

    // Send confirmation
    const responseMessage: WebSocketMessage = {
      id: this.generateMessageId(),
      type: MessageType.RESPONSE,
      timestamp: new Date(),
      payload: {
        type: 'subscription_confirmed',
        subscriptionId,
        topic
      },
      metadata: {
        correlationId: message.metadata.correlationId
      }
    };

    const serializedResponse = await this.messageHandler.serialize(responseMessage);
    connection.ws.send(serializedResponse);

    console.log(`Connection ${connection.id} subscribed to topic: ${topic}`);
  }

  private async handleUnsubscribe(connection: WebSocketConnection, message: WebSocketMessage): Promise<void> {
    const { subscriptionId } = message.payload;

    // Remove subscription
    await this.messageHandler.removeSubscription(subscriptionId);
    connection.subscriptions.delete(subscriptionId);

    // Send confirmation
    const responseMessage: WebSocketMessage = {
      id: this.generateMessageId(),
      type: MessageType.RESPONSE,
      timestamp: new Date(),
      payload: {
        type: 'unsubscription_confirmed',
        subscriptionId
      },
      metadata: {
        correlationId: message.metadata.correlationId
      }
    };

    const serializedResponse = await this.messageHandler.serialize(responseMessage);
    connection.ws.send(serializedResponse);

    console.log(`Connection ${connection.id} unsubscribed from subscription: ${subscriptionId}`);
  }

  private async handleRequest(connection: WebSocketConnection, message: WebSocketMessage): Promise<void> {
    // Handle request message
    // This would typically involve processing the request and sending a response
    console.log(`Request received from connection ${connection.id}:`, message.payload);
  }

  private async handleResponse(connection: WebSocketConnection, message: WebSocketMessage): Promise<void> {
    // Handle response message
    // This would typically involve correlating with a pending request
    console.log(`Response received from connection ${connection.id}:`, message.payload);
  }

  private async handleDataMessage(connection: WebSocketConnection, message: WebSocketMessage): Promise<void> {
    // Handle data message
    console.log(`Data message received from connection ${connection.id}:`, message.payload);

    // Store in AgentDB
    await this.storeMessage(message, {
      pipelineId: 'websocket-stream',
      agentId: connection.userId,
      timestamp: new Date(),
      correlationId: message.metadata.correlationId,
      metadata: new Map()
    });
  }

  private async handleHeartbeat(connection: WebSocketConnection, message: WebSocketMessage): Promise<void> {
    // Respond with pong
    const pongMessage: WebSocketMessage = {
      id: this.generateMessageId(),
      type: MessageType.HEARTBEAT,
      timestamp: new Date(),
      payload: {
        type: 'pong',
        timestamp: new Date()
      },
      metadata: {
        correlationId: message.metadata.correlationId
      }
    };

    const serializedPong = await this.messageHandler.serialize(pongMessage);
    connection.ws.send(serializedPong);
  }

  private handleDisconnection(connection: WebSocketConnection, code: number, reason: string): Promise<void> {
    console.log(`Connection ${connection.id} disconnected: ${code} - ${reason}`);

    // Clean up subscriptions
    for (const subscriptionId of connection.subscriptions) {
      this.messageHandler.removeSubscription(subscriptionId);
    }

    // Remove connection
    this.connections.delete(connection.id);

    // Emit disconnection event
    this.emit('disconnection', { connection, code, reason });
  }

  private handleConnectionError(connection: WebSocketConnection, error: Error): void {
    console.error(`Connection error for ${connection.id}:`, error);
    this.emit('connectionError', { connection, error });
  }

  private startHeartbeat(connection: WebSocketConnection): void {
    const interval = setInterval(() => {
      if (connection.ws.readyState === WebSocket.OPEN) {
        connection.ws.ping();
      } else {
        clearInterval(interval);
      }
    }, this.config.connection.heartbeat.interval);
  }

  private async broadcastMessage(message: Buffer, metadata: any): Promise<BroadcastResult> {
    let successCount = 0;
    let failureCount = 0;
    const errors: string[] = [];

    for (const connection of this.connections.values()) {
      if (connection.ws.readyState === WebSocket.OPEN) {
        try {
          connection.ws.send(message);
          connection.bytesSent += message.length;
          successCount++;
        } catch (error) {
          failureCount++;
          errors.push(`Connection ${connection.id}: ${error.message}`);
        }
      }
    }

    return {
      totalConnections: this.connections.size,
      successCount,
      failureCount,
      errors
    };
  }

  private async sendToTopic(topic: string, message: Buffer, metadata: any): Promise<TopicSendResult> {
    const subscriptions = await this.messageHandler.getSubscriptionsForTopic(topic);
    const targetConnections = new Set<string>();

    for (const subscription of subscriptions) {
      if (subscription.active) {
        targetConnections.add(subscription.connectionId);
      }
    }

    let successCount = 0;
    let failureCount = 0;

    for (const connectionId of targetConnections) {
      const connection = this.connections.get(connectionId);
      if (connection && connection.ws.readyState === WebSocket.OPEN) {
        try {
          connection.ws.send(message);
          connection.bytesSent += message.length;
          successCount++;
        } catch (error) {
          failureCount++;
          console.error(`Failed to send to connection ${connectionId}:`, error);
        }
      }
    }

    return {
      topic,
      subscriberCount: targetConnections.size,
      successCount,
      failureCount
    };
  }

  private async sendAndWaitForResponse(
    message: Buffer,
    metadata: any
  ): Promise<any> {
    // Implementation for request-response pattern
    // This would involve waiting for a response with matching correlation ID
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, metadata.timeout || 30000);

      // Set up response handler
      this.once(`response:${metadata.correlationId}`, (response) => {
        clearTimeout(timeout);
        resolve(response);
      });

      // Send message
      // This would be sent to the appropriate target(s)
    });
  }

  private async sendResponse(message: Buffer, metadata: any): Promise<void> {
    // Send response message
    this.emit(`response:${metadata.correlationId}`, message);
  }

  private async sendErrorMessage(
    connection: WebSocketConnection,
    message: string,
    errors: string[]
  ): Promise<void> {
    const errorMessage: WebSocketMessage = {
      id: this.generateMessageId(),
      type: MessageType.ERROR,
      timestamp: new Date(),
      payload: {
        message,
        errors,
        code: 400
      },
      metadata: {
        priority: MessagePriority.HIGH
      }
    };

    try {
      const serializedError = await this.messageHandler.serialize(errorMessage);
      connection.ws.send(serializedError);
    } catch (error) {
      console.error(`Failed to send error message to connection ${connection.id}:`, error);
    }
  }

  private async storeMessage(message: WebSocketMessage, context: StreamContext): Promise<void> {
    const key = `websocket:message:${message.id}`;
    await this.agentDB.store(key, {
      message,
      context: {
        pipelineId: context.pipelineId,
        agentId: context.agentId,
        timestamp: context.timestamp
      },
      timestamp: new Date()
    });
  }

  private getServerCapabilities(): any {
    return {
      supportedMessageTypes: Object.values(MessageType),
      serializationFormats: [SerializationFormat.JSON, SerializationFormat.MSGPACK],
      compression: this.config.server.perMessageDeflate,
      authentication: this.config.security.authentication.enabled,
      maxMessageSize: 1024 * 1024, // 1MB
      heartbeat: this.config.connection.heartbeat.enabled
    };
  }

  private async readFile(filePath: string): Promise<string> {
    // Implementation to read file content
    // In a real implementation, you would use fs.readFile
    return 'placeholder';
  }

  private mergeWithDefaults(config: Partial<WebSocketStreamConfig>): WebSocketStreamConfig {
    return {
      server: {
        port: 8080,
        host: '0.0.0.0',
        maxConnections: 10000,
        backlog: 511,
        perMessageDeflate: true,
        compression: {
          enabled: true,
          threshold: 1024,
          level: CompressionLevel.MEDIUM,
          strategy: CompressionStrategy.SHARED_COMPRESSOR
        },
        ssl: {
          enabled: false,
          certFile: '',
          keyFile: '',
          rejectUnauthorized: true
        },
        ...config.server
      },
      connection: {
        heartbeat: {
          enabled: true,
          interval: 30000, // 30 seconds
          timeout: 5000, // 5 seconds
          message: 'ping'
        },
        timeout: {
          connect: 10000, // 10 seconds
          idle: 300000, // 5 minutes
          send: 5000, // 5 seconds
          receive: 30000 // 30 seconds
        },
        reconnect: {
          enabled: false, // Server-side doesn't reconnect
          maxAttempts: 5,
          backoffStrategy: BackoffStrategy.EXPONENTIAL,
          initialDelay: 1000,
          maxDelay: 30000,
          factor: 2
        },
        rateLimiting: {
          enabled: true,
          windowMs: 60000, // 1 minute
          maxMessages: 1000,
          strategy: RateLimitingStrategy.SLIDING_WINDOW
        },
        loadBalancing: {
          enabled: true,
          algorithm: LoadBalancingAlgorithm.LEAST_CONNECTIONS,
          healthChecks: true,
          stickySessions: false
        },
        ...config.connection
      },
      messageHandling: {
        serialization: {
          format: SerializationFormat.JSON,
          compression: true,
          encryption: false,
          schemaValidation: true
        },
        validation: {
          enabled: true,
          schema: null,
          strictMode: false,
          customValidators: []
        },
        routing: {
          strategy: RoutingStrategy.TOPIC_BASED,
          topics: [],
          subscriptions: []
        },
        batching: {
          enabled: false,
          maxBatchSize: 100,
          maxWaitTime: 100,
          aggregation: AggregationStrategy.HYBRID
        },
        ordering: {
          enabled: false,
          strategy: OrderingStrategy.FIFO,
          maxOutOfOrder: 10
        },
        ...config.messageHandling
      },
      performance: {
        buffering: {
          enabled: true,
          strategy: BufferingStrategy.MESSAGE_BUFFERING,
          maxSize: 1000,
          flushInterval: 100,
          highWaterMark: 800,
          lowWaterMark: 200
        },
        multiplexing: {
          enabled: true,
          maxStreams: 100,
          streamTimeout: 30000,
          priority: true,
          loadBalancing: false
        },
        pipelining: {
          enabled: true,
          maxInFlight: 10,
          batchSize: 5,
          timeout: 5000,
          retryPolicy: {
            maxAttempts: 3,
            backoffMs: 1000,
            retryableErrors: ['ECONNRESET', 'ETIMEDOUT']
          }
        },
        connectionPooling: {
          enabled: false, // Not applicable for WebSocket server
          maxConnections: 100,
          minConnections: 10,
          maxIdleTime: 300000,
          validationQuery: 'ping'
        },
        optimization: {
          latencyOptimization: {
            targetLatency: 100, // 100ms
            enableNagle: false,
            enableQuickAck: true,
            bufferSize: 64 * 1024, // 64KB
            tcpNoDelay: true
          },
          throughputOptimization: {
            targetThroughput: 10000, // 10K messages/sec
            enableCompression: true,
            maxMessageSize: 1024 * 1024, // 1MB
            windowSize: 1024 * 1024 // 1MB
          },
          memoryOptimization: {
            maxMemoryUsage: 512 * 1024 * 1024, // 512MB
            enableGarbageCollection: true,
            gcThreshold: 0.8,
            objectPooling: true
          },
          cpuOptimization: {
            enableSIMD: true,
            enableVectorization: true,
            maxWorkers: 4,
            threadAffinity: false
          }
        },
        ...config.performance
      },
      security: {
        authentication: {
          enabled: true,
          methods: [AuthenticationMethod.JWT],
          tokenValidation: {
            algorithm: 'HS256',
            issuer: 'ran-automation-server',
            audience: 'ran-automation-clients',
            clockTolerance: 30
          },
          sessionManagement: {
            enabled: true,
            store: SessionStore.MEMORY,
            ttl: 3600000, // 1 hour
            refreshEnabled: true
          }
        },
        authorization: {
          enabled: true,
          rbac: {
            enabled: true,
            roles: [],
            permissions: []
          },
          resourceBased: {
            enabled: false,
            policies: []
          },
          policyEngine: {
            engine: PolicyEngine.CUSTOM,
            decisionCache: true,
            auditLogging: true
          }
        },
        encryption: {
          enabled: false,
          algorithm: EncryptionAlgorithm.AES_256_GCM,
          keyRotation: {
            enabled: false,
            interval: 86400000, // 24 hours
            keyDerivation: true
          },
          messageEncryption: false
        },
        rateLimit: {
          enabled: true,
          windowMs: 60000, // 1 minute
          maxRequests: 1000,
          skipSuccessfulRequests: false,
          skipFailedRequests: false
        },
        cors: {
          enabled: true,
          origin: ['*'],
          methods: ['GET', 'POST'],
          allowedHeaders: ['Content-Type', 'Authorization'],
          credentials: true,
          maxAge: 86400 // 24 hours
        },
        ...config.security
      },
      monitoring: {
        metrics: {
          enabled: true,
          collectInterval: 10000, // 10 seconds
          exportFormat: MetricsExportFormat.PROMETHEUS,
          labels: {
            includeConnectionId: true,
            includeClientId: true,
            includeTopic: true,
            includeMessageType: true,
            customLabels: {}
          }
        },
        logging: {
          enabled: true,
          level: LogLevel.INFO,
          format: LogFormat.JSON,
          includePayloads: false,
          maxPayloadSize: 1024
        },
        tracing: {
          enabled: true,
          samplingRate: 0.1,
          exportFormat: TracingExportFormat.OPENTELEMETRY,
          includePayloads: false
        },
        alerting: {
          enabled: true,
          channels: [],
          rules: [],
          thresholds: {
            connectionCount: 1000,
            messageRate: 10000,
            latency: 1000,
            errorRate: 0.05,
            memoryUsage: 0.8,
            cpuUsage: 0.8
          }
        },
        healthChecks: {
          enabled: true,
          interval: 30000, // 30 seconds
          timeout: 5000,
          endpoints: [
            {
              path: '/health',
              method: 'GET',
              expectedStatus: 200,
              timeout: 3000
            }
          ]
        },
        ...config.monitoring
      }
    };
  }

  // Utility methods
  private generateConnectionId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateSubscriptionId(): string {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Public API methods
  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server.listen(() => {
        resolve();
      });

      this.server.on('error', reject);
    });
  }

  async stop(): Promise<void> {
    return new Promise((resolve) => {
      // Close all connections
      for (const connection of this.connections.values()) {
        connection.ws.close(1001, 'Server shutting down');
      }

      // Close server
      this.server.close(() => {
        resolve();
      });
    });
  }

  getConnections(): WebSocketConnection[] {
    return Array.from(this.connections.values());
  }

  getConnectionCount(): number {
    return this.connections.size;
  }

  async getMetrics(): Promise<WebSocketMetrics> {
    return await this.performanceMonitor.getMetrics();
  }
}

// Supporting Interfaces
export enum MessageType {
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  DATA = 'data',
  REQUEST = 'request',
  RESPONSE = 'response',
  ERROR = 'error',
  SYSTEM = 'system',
  HEARTBEAT = 'heartbeat'
}

export enum MessagePriority {
  LOW = 1,
  NORMAL = 2,
  HIGH = 3,
  CRITICAL = 4
}

export interface WebSocketMessage {
  id: string;
  type: MessageType;
  timestamp: Date;
  payload: any;
  metadata: MessageMetadata;
}

export interface MessageMetadata {
  correlationId?: string;
  agentId?: string;
  pipelineId?: string;
  topic?: string;
  priority: MessagePriority;
  ttl?: number;
  requestId?: string;
  expectsReply?: boolean;
  timeout?: number;
  isResponse?: boolean;
  userId?: string;
}

export interface WebSocketConnection {
  id: string;
  ws: WebSocket;
  ip: string;
  userAgent?: string;
  connectedAt: Date;
  lastActivity: Date;
  authenticated: boolean;
  userId?: string;
  subscriptions: Set<string>;
  messageCount: number;
  bytesReceived: number;
  bytesSent: number;
}

export interface BroadcastResult {
  totalConnections: number;
  successCount: number;
  failureCount: number;
  errors: string[];
}

export interface TopicSendResult {
  topic: string;
  subscriberCount: number;
  successCount: number;
  failureCount: number;
}

export interface Subscription {
  id: string;
  topic: string;
  filter?: MessageFilter;
  connectionId?: string;
  createdAt: Date;
  active: boolean;
}

export interface PublishOptions {
  correlationId?: string;
  agentId?: string;
  priority?: MessagePriority;
  ttl?: number;
}

export interface PublishResult {
  messageId: string;
  subscribersReached: number;
  processingTime: number;
  success: boolean;
}

export interface RequestOptions {
  correlationId?: string;
  agentId?: string;
  timeout?: number;
}

export interface SubscriptionHandler {
  subscribe(topic: string, filter?: MessageFilter): Promise<string>;
  unsubscribe(subscriptionId: string): Promise<void>;
  publish(topic: string, message: any, options?: PublishOptions): Promise<PublishResult>;
}

export interface BidirectionalHandler {
  handleRequest(request: any, options?: RequestOptions): Promise<any>;
  onResponse(response: any, requestId: string): Promise<void>;
}

export interface WebSocketMetrics {
  connections: number;
  messagesPerSecond: number;
  averageLatency: number;
  errorRate: number;
  bytesReceived: number;
  bytesSent: number;
  timestamp: Date;
}

// Supporting Classes
class MessageHandler {
  private config: MessageHandlingConfig;
  private subscriptions: Map<string, Subscription> = new Map();

  constructor(config: MessageHandlingConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log('Initializing Message Handler...');
  }

  async serialize(message: WebSocketMessage): Promise<Buffer> {
    const serialized = JSON.stringify(message);
    return Buffer.from(serialized, 'utf8');
  }

  async deserialize(data: Buffer): Promise<WebSocketMessage> {
    const str = data.toString('utf8');
    return JSON.parse(str) as WebSocketMessage;
  }

  async validate(message: WebSocketMessage): Promise<ValidationResult> {
    return {
      valid: true,
      errors: []
    };
  }

  async addSubscription(subscription: Subscription): Promise<string> {
    this.subscriptions.set(subscription.id, subscription);
    return subscription.id;
  }

  async removeSubscription(subscriptionId: string): Promise<void> {
    this.subscriptions.delete(subscriptionId);
  }

  async getSubscriptionsForTopic(topic: string): Promise<Subscription[]> {
    return Array.from(this.subscriptions.values()).filter(
      sub => sub.topic === topic && sub.active
    );
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }

  async shutdown(): Promise<void> {
    this.subscriptions.clear();
  }
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
}

class ConnectionManager {
  private config: WebSocketConnectionConfig;

  constructor(config: WebSocketConnectionConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log('Initializing Connection Manager...');
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }

  async shutdown(): Promise<void> {
  }
}

class WebSocketPerformanceMonitor {
  private config: WebSocketMonitoringConfig;
  private metrics: WebSocketMetrics;

  constructor(config: WebSocketMonitoringConfig) {
    this.config = config;
    this.metrics = {
      connections: 0,
      messagesPerSecond: 0,
      averageLatency: 0,
      errorRate: 0,
      bytesReceived: 0,
      bytesSent: 0,
      timestamp: new Date()
    };
  }

  async initialize(): Promise<void> {
    console.log('Initializing WebSocket Performance Monitor...');
  }

  async start(): Promise<void> {
    // Start metrics collection
    setInterval(() => {
      this.updateMetrics();
    }, this.config.metrics.collectInterval);
  }

  async recordMessage(message: WebSocketMessage, processingTime: number, success: boolean): Promise<void> {
    // Record message metrics
  }

  async recordError(messageId: string, error: Error, processingTime: number): Promise<void> {
    // Record error metrics
  }

  async getMetrics(): Promise<WebSocketMetrics> {
    return this.metrics;
  }

  private updateMetrics(): void {
    // Update metrics
    this.metrics.timestamp = new Date();
  }

  async shutdown(): Promise<void> {
  }
}

class WebSocketSecurityManager {
  private config: WebSocketSecurityConfig;

  constructor(config: WebSocketSecurityConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log('Initializing WebSocket Security Manager...');
  }

  async authenticate(ws: WebSocket, req: any): Promise<AuthenticationResult> {
    return {
      success: true,
      authenticated: true,
      userId: 'user123'
    };
  }

  async authorizeSubscription(userId: string, topic: string, action: string): Promise<AuthorizationResult> {
    return {
      authorized: true,
      reason: ''
    };
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }

  async shutdown(): Promise<void> {
  }
}

interface AuthenticationResult {
  success: boolean;
  authenticated: boolean;
  userId?: string;
}

interface AuthorizationResult {
  authorized: boolean;
  reason: string;
}

export default RealTimeWebSocketStream;