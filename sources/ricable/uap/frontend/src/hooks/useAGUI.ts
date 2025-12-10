// File: frontend/src/hooks/useAGUI.ts
// AG-UI Protocol Hook - Native WebSocket Implementation with Authentication
import { useEffect, useState, useCallback, useRef } from 'react';
import {
  AGUIEvent,
  AGUIConnectionConfig,
  ConnectionState,
  ConnectionStatus,
  UserMessageEvent,
  AGUIEventHandler
} from '../types/ag-ui';
import { useAuth } from '../auth/AuthContext';

interface UseAGUIReturn {
  messages: AGUIEvent[];
  connectionStatus: ConnectionStatus;
  isConnected: boolean;
  sendMessage: (content: string, framework?: 'auto' | 'copilot' | 'agno' | 'mastra') => Promise<void>;
  clearMessages: () => void;
  reconnect: () => void;
}

export function useAGUI(
  agentId: string,
  config: Partial<AGUIConnectionConfig> = {}
): UseAGUIReturn {
  // Authentication
  // TODO: Enable auth when ready
  // const { token, user, isAuthenticated } = useAuth();
  
  // State management
  const [messages, setMessages] = useState<AGUIEvent[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    state: 'disconnected',
    reconnectAttempts: 0
  });
  
  // Refs for stable references
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const agentIdRef = useRef(agentId);
  const eventHandlersRef = useRef<Map<string, Set<AGUIEventHandler>>>(new Map());
  
  // Update agentId ref when it changes
  useEffect(() => {
    agentIdRef.current = agentId;
  }, [agentId]);

  // Configuration with defaults
  const connectionConfig: AGUIConnectionConfig = {
    endpoint: '',
    transport: 'websocket',
    reconnect: true,
    maxReconnectAttempts: 5,
    reconnectInterval: 1000,
    maxReconnectInterval: 30000,
    timeout: 30000,
    protocols: [],
    ...config
  };

  // Generate WebSocket URL with authentication
  const getWebSocketUrl = useCallback(() => {
    if (connectionConfig.endpoint) {
      return connectionConfig.endpoint;
    }
    
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const baseUrl = `${wsProtocol}//${window.location.host}/ws/agents/${agentIdRef.current}`;
    
    // TODO: Add token as query parameter when auth is enabled
    // if (token) {
    //   const separator = baseUrl.includes('?') ? '&' : '?';
    //   return `${baseUrl}${separator}token=${encodeURIComponent(token)}`;
    // }
    
    return baseUrl;
  }, [connectionConfig.endpoint]);

  // Event handler management
  const addEventListener = useCallback((eventType: string, handler: AGUIEventHandler) => {
    if (!eventHandlersRef.current.has(eventType)) {
      eventHandlersRef.current.set(eventType, new Set());
    }
    eventHandlersRef.current.get(eventType)!.add(handler);
  }, []);

  const removeEventListener = useCallback((eventType: string, handler: AGUIEventHandler) => {
    const handlers = eventHandlersRef.current.get(eventType);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        eventHandlersRef.current.delete(eventType);
      }
    }
  }, []);

  const emitEvent = useCallback((event: AGUIEvent) => {
    // Add to messages state for all relevant event types
    if (['user_message', 'text_message_content', 'tool_call_start', 'tool_call_end', 'state_delta', 'agent_thinking', 'error'].includes(event.type)) {
      setMessages(prev => [...prev, event]);
    }

    // Emit to registered handlers
    const handlers = eventHandlersRef.current.get(event.type);
    if (handlers) {
      handlers.forEach(handler => handler(event));
    }

    // Emit to wildcard handlers
    const wildcardHandlers = eventHandlersRef.current.get('*');
    if (wildcardHandlers) {
      wildcardHandlers.forEach(handler => handler(event));
    }
  }, []);

  // Calculate next reconnection delay with exponential backoff
  const getReconnectDelay = useCallback((attempt: number): number => {
    const delay = connectionConfig.reconnectInterval! * Math.pow(2, attempt);
    return Math.min(delay, connectionConfig.maxReconnectInterval!);
  }, [connectionConfig.reconnectInterval, connectionConfig.maxReconnectInterval]);

  // Clear any existing reconnection timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    // TODO: Enable auth check when ready
    // if (!isAuthenticated || !token) {
    //   setConnectionStatus({
    //     state: 'disconnected',
    //     reconnectAttempts: 0,
    //     error: 'Authentication required'
    //   });
    //   return;
    // }

    // Close existing connection if any
    if (websocketRef.current) {
      websocketRef.current.close();
    }

    const wsUrl = getWebSocketUrl();
    
    setConnectionStatus(prev => ({
      ...prev,
      state: prev.reconnectAttempts > 0 ? 'reconnecting' : 'connecting'
    }));

    try {
      const ws = new WebSocket(wsUrl, connectionConfig.protocols);
      websocketRef.current = ws;

      // Connection timeout
      const timeoutId = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.close();
          setConnectionStatus(prev => ({
            ...prev,
            state: 'error',
            error: 'Connection timeout'
          }));
        }
      }, connectionConfig.timeout);

      ws.onopen = () => {
        clearTimeout(timeoutId);
        setConnectionStatus(prev => ({
          ...prev,
          state: 'connected',
          lastConnected: Date.now(),
          reconnectAttempts: 0,
          error: undefined
        }));

        emitEvent({
          type: 'connection_open',
          timestamp: Date.now(),
          metadata: { agentId: agentIdRef.current }
        });
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const aguiEvent: AGUIEvent = {
            ...data,
            timestamp: data.timestamp || Date.now()
          };
          emitEvent(aguiEvent);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          emitEvent({
            type: 'error',
            content: 'Failed to parse message from server',
            timestamp: Date.now(),
            metadata: {
              errorType: 'parse_error',
              originalMessage: event.data
            }
          });
        }
      };

      ws.onclose = (event) => {
        clearTimeout(timeoutId);
        
        const shouldReconnect = connectionConfig.reconnect &&
          connectionStatus.reconnectAttempts < connectionConfig.maxReconnectAttempts! &&
          !event.wasClean;

        setConnectionStatus(prev => ({
          ...prev,
          state: shouldReconnect ? 'reconnecting' : 'disconnected',
          error: event.reason || undefined
        }));

        emitEvent({
          type: 'connection_close',
          timestamp: Date.now(),
          metadata: {
            agentId: agentIdRef.current,
            reason: event.reason,
            code: event.code,
            wasClean: event.wasClean
          }
        });

        if (shouldReconnect) {
          const delay = getReconnectDelay(connectionStatus.reconnectAttempts);
          reconnectTimeoutRef.current = setTimeout(() => {
            setConnectionStatus(prev => ({
              ...prev,
              reconnectAttempts: prev.reconnectAttempts + 1
            }));
            connect();
          }, delay);
        }
      };

      ws.onerror = () => {
        clearTimeout(timeoutId);
        setConnectionStatus(prev => ({
          ...prev,
          state: 'error',
          error: 'WebSocket connection error'
        }));

        emitEvent({
          type: 'connection_error',
          content: 'WebSocket connection error',
          timestamp: Date.now(),
          metadata: { agentId: agentIdRef.current }
        });
      };

    } catch (error) {
      setConnectionStatus(prev => ({
        ...prev,
        state: 'error',
        error: error instanceof Error ? error.message : 'Unknown connection error'
      }));
    }
  }, [getWebSocketUrl, connectionConfig, connectionStatus.reconnectAttempts, emitEvent, getReconnectDelay]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    clearReconnectTimeout();
    if (websocketRef.current) {
      websocketRef.current.close(1000, 'User initiated disconnect');
      websocketRef.current = null;
    }
    setConnectionStatus({
      state: 'disconnected',
      reconnectAttempts: 0
    });
  }, [clearReconnectTimeout]);

  // Reconnect manually
  const reconnect = useCallback(() => {
    disconnect();
    setConnectionStatus(prev => ({ ...prev, reconnectAttempts: 0 }));
    connect();
  }, [disconnect, connect]);

  // Send message to agent
  const sendMessage = useCallback(async (
    content: string,
    framework: 'auto' | 'copilot' | 'agno' | 'mastra' = 'auto'
  ): Promise<void> => {
    // TODO: Enable auth check when ready
    // if (!isAuthenticated || !token) {
    //   throw new Error('Authentication required');
    // }

    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected');
    }

    const message: UserMessageEvent = {
      type: 'user_message',
      content,
      timestamp: Date.now(),
      requestId: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      metadata: {
        framework,
        agentId: agentIdRef.current,
        sessionId: `session_${agentIdRef.current}`,
        // TODO: Add user context when auth is ready
        // user_context: user ? {
        //   user_id: user.id,
        //   username: user.username,
        //   roles: user.roles
        // } : undefined
      }
    };

    try {
      // Add user message to local messages immediately
      emitEvent(message);
      
      // Send message to backend
      websocketRef.current.send(JSON.stringify(message));
    } catch (error) {
      emitEvent({
        type: 'error',
        content: 'Failed to send message',
        timestamp: Date.now(),
        metadata: {
          errorType: 'send_error',
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      });
      throw error;
    }
  }, [emitEvent]);

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  // Connection lifecycle management
  useEffect(() => {
    // TODO: Enable auth check when ready
    // if (isAuthenticated && token) {
      connect();
    // } else {
    //   disconnect();
    // }
    return () => {
      disconnect();
    };
  }, [agentId]); // Reconnect when agentId changes

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearReconnectTimeout();
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, [clearReconnectTimeout]);

  return {
    messages,
    connectionStatus,
    isConnected: connectionStatus.state === 'connected',
    sendMessage,
    clearMessages,
    reconnect
  };
}