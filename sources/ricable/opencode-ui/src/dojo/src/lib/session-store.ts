/**
 * Session Store - Zustand-based state management for OpenCode sessions
 * Handles multi-provider sessions, real-time updates, and persistent state
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { 
  openCodeClient, 
  type Session, 
  type Message, 
  type Provider, 
  type ProviderHealth,
  type ProviderMetrics,
  type Tool,
  type ToolExecution,
  type OpenCodeConfig,
  type SessionConfig,
  type ValidationResult
} from './opencode-client';

export interface SessionTemplate {
  id: string;
  name: string;
  description: string;
  provider: string;
  model: string;
  initial_prompt?: string;
  tools: string[];
  configuration: Partial<OpenCodeConfig>;
}

export interface AppState {
  // Connection Status
  serverStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  serverVersion: string | null;
  lastConnectionAttempt: number | null;
  connectionLatency: number | null;
  
  // Session Management
  sessions: Session[];
  activeSessionId: string | null;
  sessionMessages: Record<string, Message[]>;
  sessionTemplates: SessionTemplate[];
  isLoadingSessions: boolean;
  sessionSubscriptions: Set<string>;
  
  // Multi-Provider Management
  providers: Provider[];
  authenticatedProviders: string[];
  activeProvider: string | null;
  providerMetrics: ProviderMetrics[];
  providerHealth: ProviderHealth[];
  isLoadingProviders: boolean;
  providerAuthStatus: Record<string, { authenticated: boolean; expires_at?: number; error?: string }>;
  
  // Tool System
  availableTools: Tool[];
  toolExecutions: ToolExecution[];
  pendingApprovals: ToolExecution[];
  isLoadingTools: boolean;
  toolAuditLog: Array<{
    tool_id: string;
    executed_at: number;
    success: boolean;
    session_id?: string;
  }>;
  
  // Configuration
  config: OpenCodeConfig | null;
  configProfiles: Array<{
    id: string;
    name: string;
    config: OpenCodeConfig;
  }>;
  configValidation: ValidationResult | null;
  
  // UI State
  currentView: 'projects' | 'session' | 'settings' | 'providers' | 'tools';
  showTimeline: boolean;
  showProviderDashboard: boolean;
  showToolDashboard: boolean;
  sidebarCollapsed: boolean;
  
  // Real-time Data
  isStreaming: boolean;
  streamingSessionId: string | null;
  connectionErrors: string[];
  lastActivity: number;
  
  // Actions
  actions: {
    // Connection Management
    connect: () => Promise<void>;
    disconnect: () => void;
    checkHealth: () => Promise<void>;
    
    // Session Management
    loadSessions: () => Promise<void>;
    createSession: (config: SessionConfig) => Promise<Session>;
    deleteSession: (id: string) => Promise<void>;
    setActiveSession: (id: string | null) => void;
    loadSessionMessages: (sessionId: string) => Promise<void>;
    sendMessage: (sessionId: string, content: string) => Promise<void>;
    shareSession: (id: string) => Promise<string>;
    
    // Provider Management
    loadProviders: () => Promise<void>;
    authenticateProvider: (provider: string, credentials: any) => Promise<boolean>;
    setActiveProvider: (provider: string) => void;
    loadProviderMetrics: () => Promise<void>;
    loadProviderHealth: () => Promise<void>;
    
    // Tool Management
    loadTools: () => Promise<void>;
    executeTools: (toolId: string, input: any, sessionId?: string) => Promise<ToolExecution>;
    approveToolExecution: (executionId: string) => Promise<void>;
    cancelToolExecution: (executionId: string) => Promise<void>;
    loadToolExecutions: (sessionId?: string) => Promise<void>;
    
    // Configuration Management
    loadConfig: () => Promise<void>;
    updateConfig: (config: Partial<OpenCodeConfig>) => Promise<void>;
    validateConfig: (config: OpenCodeConfig) => Promise<ValidationResult>;
    
    // UI Actions
    setCurrentView: (view: AppState['currentView']) => void;
    toggleTimeline: () => void;
    toggleProviderDashboard: () => void;
    toggleToolDashboard: () => void;
    toggleSidebar: () => void;
    
    // Error Handling
    addConnectionError: (error: string) => void;
    clearConnectionErrors: () => void;
  };
}

export const useSessionStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial State
      serverStatus: 'disconnected',
      serverVersion: null,
      lastConnectionAttempt: null,
      connectionLatency: null,
      sessions: [],
      activeSessionId: null,
      sessionMessages: {},
      sessionTemplates: [],
      isLoadingSessions: false,
      sessionSubscriptions: new Set(),
      providers: [],
      authenticatedProviders: [],
      activeProvider: null,
      providerMetrics: [],
      providerHealth: [],
      isLoadingProviders: false,
      providerAuthStatus: {},
      availableTools: [],
      toolExecutions: [],
      pendingApprovals: [],
      isLoadingTools: false,
      toolAuditLog: [],
      config: null,
      configProfiles: [],
      configValidation: null,
      currentView: 'projects',
      showTimeline: false,
      showProviderDashboard: false,
      showToolDashboard: false,
      sidebarCollapsed: false,
      isStreaming: false,
      streamingSessionId: null,
      connectionErrors: [],
      lastActivity: Date.now(),

      // Actions
      actions: {
        // Connection Management
        connect: async () => {
          const startTime = Date.now();
          set({ 
            serverStatus: 'connecting',
            lastConnectionAttempt: startTime
          });
          
          try {
            const health = await openCodeClient.healthCheck();
            const latency = Date.now() - startTime;
            
            set({ 
              serverStatus: 'connected', 
              serverVersion: health.version,
              connectionLatency: latency,
              lastActivity: Date.now()
            });
            
            // Set up real-time subscriptions
            openCodeClient.subscribeToProviderUpdates((update) => {
              set((state) => {
                if (update.type === 'auth') {
                  return {
                    providerAuthStatus: {
                      ...state.providerAuthStatus,
                      [update.providerId]: {
                        authenticated: update.data.authenticated,
                        expires_at: update.data.expires_at,
                        error: update.data.error
                      }
                    }
                  };
                }
                return state;
              });
            });
            
            openCodeClient.subscribeToToolExecutions((update) => {
              if (update.type === 'execution') {
                set((state) => ({
                  toolExecutions: [...state.toolExecutions.filter(t => t.id !== update.data.id), update.data],
                  pendingApprovals: update.data.status === 'pending' 
                    ? [...state.pendingApprovals.filter(t => t.id !== update.data.id), update.data]
                    : state.pendingApprovals.filter(t => t.id !== update.data.id)
                }));
              }
            });
            
            // Enhanced event listeners
            openCodeClient.on('session_update', (data: any) => {
              const { sessionId, update } = data;
              set((state) => {
                const messages = state.sessionMessages[sessionId] || [];
                if (update.type === 'message') {
                  return {
                    sessionMessages: {
                      ...state.sessionMessages,
                      [sessionId]: [...messages, update.data]
                    },
                    lastActivity: Date.now()
                  };
                }
                return { lastActivity: Date.now() };
              });
            });
            
            openCodeClient.on('provider_update', (data: any) => {
              set((state) => ({
                providerHealth: state.providerHealth.map(p => 
                  p.provider_id === data.providerId ? { ...p, ...data.data } : p
                ),
                lastActivity: Date.now()
              }));
            });
            
            openCodeClient.on('tool_execution_logged', (data: any) => {
              set((state) => ({
                toolAuditLog: [{
                  tool_id: data.tool_id,
                  executed_at: data.timestamp,
                  success: data.result.success,
                  session_id: data.session_id
                }, ...state.toolAuditLog].slice(0, 100), // Keep last 100 entries
                lastActivity: Date.now()
              }));
            });
            
            openCodeClient.on('tool_approval_required', (data: any) => {
              // Handle tool approval UI prompts
              console.log('Tool approval required:', data);
              // In a real implementation, this would trigger a UI modal
            });
            
            openCodeClient.on('websocket_error', (data: any) => {
              console.error('WebSocket error:', data);
              set((state) => ({
                connectionErrors: [...state.connectionErrors, `WebSocket error for ${data.sessionId}`]
              }));
            });

            openCodeClient.on('websocket_reconnect_failed', (data: any) => {
              console.warn('WebSocket reconnection failed:', data);
              set((state) => ({
                connectionErrors: [...state.connectionErrors, `Failed to reconnect to session ${data.sessionId}`]
              }));
            });
            
            // Load initial data with better error handling
            try {
              await Promise.allSettled([
                get().actions.loadSessions(),
                // Skip provider loading until endpoint is available
                // get().actions.loadProviders(),
                // Skip tool loading until endpoint is available  
                // get().actions.loadTools(),
                get().actions.loadConfig()
              ]);
            } catch (error) {
              console.error('Failed to load initial data:', error);
              // Continue despite initialization errors
            }
            
          } catch (error) {
            console.error('Failed to connect to OpenCode server:', error);
            set({ 
              serverStatus: 'error',
              connectionErrors: [...get().connectionErrors, error instanceof Error ? error.message : 'Connection failed']
            });
          }
        },

        disconnect: () => {
          openCodeClient.disconnect();
          set({ 
            serverStatus: 'disconnected',
            isStreaming: false,
            streamingSessionId: null
          });
        },

        checkHealth: async () => {
          try {
            const health = await openCodeClient.healthCheck();
            set({ 
              serverStatus: health.status === 'ok' ? 'connected' : 'error',
              serverVersion: health.version 
            });
          } catch (error) {
            set({ serverStatus: 'error' });
          }
        },

        // Session Management
        loadSessions: async () => {
          set({ isLoadingSessions: true });
          try {
            const sessions = await openCodeClient.getSessions();
            set({ sessions, isLoadingSessions: false, serverStatus: 'connected' });
          } catch (error) {
            console.error('Failed to load sessions:', error);
            set({ 
              isLoadingSessions: false, 
              serverStatus: 'error',
              sessions: [], // Clear sessions when server is not available
              connectionErrors: [...get().connectionErrors, error instanceof Error ? error.message : 'Failed to load sessions']
            });
          }
        },

        createSession: async (config: SessionConfig) => {
          try {
            const session = await openCodeClient.createSession(config);
            set((state) => ({
              sessions: [...state.sessions, session],
              activeSessionId: session.id
            }));
            
            // Subscribe to the new session
            await openCodeClient.subscribeToSession(session.id);
            
            return session;
          } catch (error) {
            console.error('Failed to create session:', error);
            throw error;
          }
        },

        deleteSession: async (id: string) => {
          try {
            await openCodeClient.deleteSession(id);
            set((state) => ({
              sessions: state.sessions.filter(s => s.id !== id),
              activeSessionId: state.activeSessionId === id ? null : state.activeSessionId,
              sessionMessages: Object.fromEntries(
                Object.entries(state.sessionMessages).filter(([sessionId]) => sessionId !== id)
              )
            }));
          } catch (error) {
            console.error('Failed to delete session:', error);
            throw error;
          }
        },

        setActiveSession: (id: string | null) => {
          set({ activeSessionId: id });
          if (id) {
            get().actions.loadSessionMessages(id);
          }
        },

        loadSessionMessages: async (sessionId: string) => {
          try {
            const messages = await openCodeClient.getMessages(sessionId);
            set((state) => ({
              sessionMessages: {
                ...state.sessionMessages,
                [sessionId]: messages
              }
            }));
          } catch (error) {
            console.error('Failed to load session messages:', error);
          }
        },

        sendMessage: async (sessionId: string, content: string) => {
          try {
            set({ isStreaming: true, streamingSessionId: sessionId });
            await openCodeClient.sendMessage(sessionId, content);
            // Real-time updates will be handled by WebSocket listeners
          } catch (error) {
            console.error('Failed to send message:', error);
            set({ isStreaming: false, streamingSessionId: null });
            throw error;
          }
        },

        shareSession: async (id: string) => {
          try {
            const shareLink = await openCodeClient.shareSession(id);
            // Update session with share status
            set((state) => ({
              sessions: state.sessions.map(s => 
                s.id === id ? { ...s, shared: true, share_url: shareLink.url } : s
              )
            }));
            return shareLink.url;
          } catch (error) {
            console.error('Failed to share session:', error);
            throw error;
          }
        },

        // Provider Management
        loadProviders: async () => {
          set({ isLoadingProviders: true });
          try {
            const [providers, health, metrics] = await Promise.all([
              openCodeClient.getProviders(),
              openCodeClient.getProviderHealth(),
              openCodeClient.getProviderMetrics()
            ]);
            
            // Build authentication status from local storage
            const authStatus: Record<string, { authenticated: boolean; expires_at?: number; error?: string }> = {};
            providers.forEach(provider => {
              if (typeof localStorage !== 'undefined') {
                try {
                  const authData = localStorage.getItem(`auth_${provider.id}`);
                  if (authData) {
                    const parsed = JSON.parse(authData);
                    authStatus[provider.id] = {
                      authenticated: parsed.authenticated,
                      expires_at: parsed.expires_at,
                      error: parsed.error
                    };
                  }
                } catch (error) {
                  console.warn(`Failed to parse auth data for ${provider.id}:`, error);
                }
              }
            });
            
            set({ 
              providers, 
              providerHealth: health,
              providerMetrics: metrics,
              authenticatedProviders: providers.filter(p => p.authenticated).map(p => p.id),
              providerAuthStatus: authStatus,
              isLoadingProviders: false,
              lastActivity: Date.now()
            });
          } catch (error) {
            console.error('Failed to load providers:', error);
            set({ 
              isLoadingProviders: false,
              serverStatus: 'error',
              providers: [], // Clear providers when server is not available
              providerHealth: [],
              providerMetrics: [],
              connectionErrors: [...get().connectionErrors, error instanceof Error ? error.message : 'Failed to load providers']
            });
          }
        },

        authenticateProvider: async (provider: string, credentials: any) => {
          try {
            const result = await openCodeClient.authenticateProvider(provider, credentials);
            if (result.success) {
              set((state) => ({
                authenticatedProviders: [...state.authenticatedProviders, provider],
                providers: state.providers.map(p => 
                  p.id === provider ? { ...p, authenticated: true } : p
                )
              }));
            }
            return result.success;
          } catch (error) {
            console.error('Failed to authenticate provider:', error);
            return false;
          }
        },

        setActiveProvider: (provider: string) => {
          set({ activeProvider: provider });
        },

        loadProviderMetrics: async () => {
          try {
            // Mock metrics data for development
            const metrics = [
              { provider_id: "anthropic", requests: 125, avg_response_time: 850, total_cost: 2.45, error_rate: 0.02, last_24h: { requests: 45, cost: 0.89, avg_response_time: 920 }},
              { provider_id: "openai", requests: 89, avg_response_time: 750, total_cost: 1.89, error_rate: 0.01, last_24h: { requests: 32, cost: 0.67, avg_response_time: 780 }}
            ];
            set({ providerMetrics: metrics });
          } catch (error) {
            console.error('Failed to load provider metrics:', error);
          }
        },

        loadProviderHealth: async () => {
          try {
            // Mock health data for development
            const health = [
              { provider_id: "anthropic", status: "online" as const, response_time: 850, last_check: Date.now(), uptime: 99.9, region: "us-east-1" },
              { provider_id: "openai", status: "online" as const, response_time: 750, last_check: Date.now(), uptime: 98.5, region: "us-west-2" }
            ];
            set({ providerHealth: health });
          } catch (error) {
            console.error('Failed to load provider health:', error);
          }
        },

        // Tool Management
        loadTools: async () => {
          set({ isLoadingTools: true });
          try {
            const [tools, executions] = await Promise.all([
              openCodeClient.getTools(),
              openCodeClient.getToolExecutions()
            ]);
            
            set({ 
              availableTools: tools,
              toolExecutions: executions,
              pendingApprovals: executions.filter(e => e.status === 'pending'),
              isLoadingTools: false,
              lastActivity: Date.now()
            });
          } catch (error) {
            console.error('Failed to load tools:', error);
            set({ 
              isLoadingTools: false,
              serverStatus: 'error',
              availableTools: [], // Clear tools when server is not available
              toolExecutions: [],
              pendingApprovals: [],
              connectionErrors: [...get().connectionErrors, error instanceof Error ? error.message : 'Failed to load tools']
            });
          }
        },

        executeTools: async (toolId: string, input: any, sessionId?: string) => {
          try {
            const result = await openCodeClient.executeTool(toolId, input, sessionId);
            
            // Convert ToolResult to ToolExecution for state management
            const execution: ToolExecution = {
              id: `exec-${Date.now()}`,
              tool_id: toolId,
              session_id: sessionId || '',
              status: result.success ? 'completed' : 'failed',
              params: input,
              result: result.result,
              error: result.error,
              created_at: Date.now(),
              completed_at: result.success ? Date.now() : undefined
            };
            
            set((state) => ({
              toolExecutions: [...state.toolExecutions, execution],
              pendingApprovals: execution.status === 'pending' 
                ? [...state.pendingApprovals, execution]
                : state.pendingApprovals
            }));
            return execution;
          } catch (error) {
            console.error('Failed to execute tool:', error);
            throw error;
          }
        },

        approveToolExecution: async (executionId: string) => {
          try {
            await openCodeClient.approveToolExecution(executionId);
            set((state) => ({
              pendingApprovals: state.pendingApprovals.filter(e => e.id !== executionId),
              toolExecutions: state.toolExecutions.map(e => 
                e.id === executionId ? { ...e, status: 'completed' as const } : e
              )
            }));
          } catch (error) {
            console.error('Failed to approve tool execution:', error);
            throw error;
          }
        },

        cancelToolExecution: async (executionId: string) => {
          try {
            await openCodeClient.cancelToolExecution(executionId);
            set((state) => ({
              pendingApprovals: state.pendingApprovals.filter(e => e.id !== executionId),
              toolExecutions: state.toolExecutions.map(e => 
                e.id === executionId ? { ...e, status: 'failed' as const } : e
              )
            }));
          } catch (error) {
            console.error('Failed to cancel tool execution:', error);
            throw error;
          }
        },

        loadToolExecutions: async (sessionId?: string) => {
          try {
            const executions = await openCodeClient.getToolExecutions(sessionId);
            set({ 
              toolExecutions: executions,
              pendingApprovals: executions.filter(e => e.status === 'pending')
            });
          } catch (error) {
            console.error('Failed to load tool executions:', error);
          }
        },

        // Configuration Management
        loadConfig: async () => {
          try {
            const config = await openCodeClient.getConfig();
            set({ config, serverStatus: 'connected' });
          } catch (error) {
            console.error('Failed to load config:', error);
            set({ 
              config: null, // Clear config when server is not available
              serverStatus: 'error',
              connectionErrors: [...get().connectionErrors, error instanceof Error ? error.message : 'Failed to load config']
            });
          }
        },

        updateConfig: async (config: Partial<OpenCodeConfig>) => {
          try {
            await openCodeClient.updateConfig(config);
            set((state) => ({
              config: state.config ? { ...state.config, ...config } : null
            }));
          } catch (error) {
            console.error('Failed to update config:', error);
            throw error;
          }
        },

        validateConfig: async (config: OpenCodeConfig) => {
          try {
            const validation = await openCodeClient.validateConfig(config);
            set({ configValidation: validation });
            return validation;
          } catch (error) {
            console.error('Failed to validate config:', error);
            throw error;
          }
        },

        // UI Actions
        setCurrentView: (view: AppState['currentView']) => {
          set({ currentView: view });
        },

        toggleTimeline: () => {
          set((state) => ({ showTimeline: !state.showTimeline }));
        },

        toggleProviderDashboard: () => {
          set((state) => ({ showProviderDashboard: !state.showProviderDashboard }));
        },

        toggleToolDashboard: () => {
          set((state) => ({ showToolDashboard: !state.showToolDashboard }));
        },

        toggleSidebar: () => {
          set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
        },

        // Error Handling
        addConnectionError: (error: string) => {
          set((state) => ({
            connectionErrors: [...state.connectionErrors, error]
          }));
        },

        clearConnectionErrors: () => {
          set({ connectionErrors: [] });
        }
      }
    }),
    {
      name: 'opencode-session-store',
      // Only persist UI preferences and authentication status
      partialize: (state) => ({
        currentView: state.currentView,
        sidebarCollapsed: state.sidebarCollapsed,
        authenticatedProviders: state.authenticatedProviders,
        activeProvider: state.activeProvider,
        configProfiles: state.configProfiles
      })
    }
  )
);

// Selectors for easy access to specific state slices
export const useActiveSession = () => {
  const { sessions, activeSessionId } = useSessionStore();
  return sessions.find(s => s.id === activeSessionId) || null;
};

export const useActiveSessionMessages = () => {
  const { sessionMessages, activeSessionId } = useSessionStore();
  return activeSessionId ? sessionMessages[activeSessionId] || [] : [];
};

export const useProviderByStatus = (status: 'online' | 'offline' | 'error') => {
  const { providers, providerHealth } = useSessionStore();
  const healthMap = new Map(providerHealth.map(h => [h.provider_id, h.status]));
  return providers.filter(p => healthMap.get(p.id) === status);
};

export const useSessionsByProvider = (providerId: string) => {
  const { sessions } = useSessionStore();
  return sessions.filter(s => s.provider === providerId);
};