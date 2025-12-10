// AG-UI Protocol Type Definitions
// Unified Agentic Platform - Real-time communication protocol

export type AGUIEventType =
  | 'connection_open'
  | 'connection_close'
  | 'connection_error'
  | 'user_message'
  | 'text_message_content'
  | 'tool_call_start'
  | 'tool_call_end'
  | 'state_delta'
  | 'agent_thinking'
  | 'error';

export interface BaseAGUIEvent {
  type: AGUIEventType;
  timestamp?: number;
  requestId?: string;
}

export interface UserMessageEvent extends BaseAGUIEvent {
  type: 'user_message';
  content: string;
  metadata?: {
    framework?: 'auto' | 'copilot' | 'agno' | 'mastra';
    agentId?: string;
    sessionId?: string;
    [key: string]: any;
  };
}

export interface TextMessageContentEvent extends BaseAGUIEvent {
  type: 'text_message_content';
  content: string;
  metadata?: {
    framework?: string;
    agentId?: string;
    partial?: boolean;
    [key: string]: any;
  };
}

export interface ToolCallStartEvent extends BaseAGUIEvent {
  type: 'tool_call_start';
  content?: string;
  metadata: {
    toolName: string;
    toolId: string;
    parameters?: Record<string, any>;
    [key: string]: any;
  };
}

export interface ToolCallEndEvent extends BaseAGUIEvent {
  type: 'tool_call_end';
  content?: string;
  metadata: {
    toolName: string;
    toolId: string;
    result?: any;
    error?: string;
    duration?: number;
    [key: string]: any;
  };
}

export interface StateDeltaEvent extends BaseAGUIEvent {
  type: 'state_delta';
  content?: string;
  metadata: {
    stateName: string;
    oldValue?: any;
    newValue?: any;
    [key: string]: any;
  };
}

export interface AgentThinkingEvent extends BaseAGUIEvent {
  type: 'agent_thinking';
  content: string;
  metadata?: {
    framework?: string;
    agentId?: string;
    reasoning?: string;
    [key: string]: any;
  };
}

export interface ConnectionEvent extends BaseAGUIEvent {
  type: 'connection_open' | 'connection_close' | 'connection_error';
  content?: string;
  metadata?: {
    reason?: string;
    code?: number;
    [key: string]: any;
  };
}

export interface ErrorEvent extends BaseAGUIEvent {
  type: 'error';
  content: string;
  metadata: {
    errorType: string;
    errorCode?: string;
    framework?: string;
    agentId?: string;
    [key: string]: any;
  };
}

export type AGUIEvent =
  | UserMessageEvent
  | TextMessageContentEvent
  | ToolCallStartEvent
  | ToolCallEndEvent
  | StateDeltaEvent
  | AgentThinkingEvent
  | ConnectionEvent
  | ErrorEvent;

export interface AGUIConnectionConfig {
  endpoint: string;
  transport?: 'websocket';
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  maxReconnectInterval?: number;
  timeout?: number;
  protocols?: string[];
}

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error';

export interface ConnectionStatus {
  state: ConnectionState;
  lastConnected?: number;
  reconnectAttempts: number;
  error?: string;
}

export type AGUIEventHandler = (event: AGUIEvent) => void;