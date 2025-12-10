/**
 * OpenCode Library Exports
 * 
 * Centralized exports for all OpenCode client functionality
 */

// Main client exports
export {
  OpenCodeClient,
  OpenCodeAPIError,
  openCodeClient,
  createOpenCodeClient,
  isOpenCodeAvailable,
  isValidSession,
  isValidProvider,
} from './opencode-client';

// Session store exports
export {
  useSessionStore,
  useActiveSession,
  useActiveSessionMessages,
  useProviderByStatus,
  useSessionsByProvider,
  type AppState,
  type SessionTemplate,
} from './session-store';

// Core types
export type {
  Provider,
  Session,
  SessionConfig,
  Message,
  Project,
  Tool,
  MCPServer,
  OpenCodeConfig,
  ProviderConfig,
  AgentConfig,
  LSPConfig,
  ShellConfig,
  ProviderMetrics,
  ProviderHealth,
  ToolExecution,
  ValidationResult,
  
  // Extended types
  SessionUpdate,
  ShareLink,
  AuthResult,
  ToolResult,
  ToolExecutionRequest,
  LSPServer,
  LSPDiagnostic,
  CustomCommand,
  UsageStats,
  ServerInfo,
} from './opencode-client';

// Extended types
export type {
  WebSocketManager,
  EventEmitter,
  HTTPClientConfig,
  HTTPClient,
  ModelConfig,
  ProviderEndpoint,
  APIError,
  StreamChunk,
  StreamOptions,
  CostBreakdown,
  ConnectionStatus,
  ProjectStats,
  ProjectTemplate,
  FileOperationResult,
  DirectoryListing,
} from '../types/opencode';

// Utility functions
export * as utils from './utils';