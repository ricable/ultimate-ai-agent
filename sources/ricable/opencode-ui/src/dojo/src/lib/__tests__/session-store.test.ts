import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useSessionStore } from '../session-store'
import { openCodeClient } from '../opencode-client'
import { mockSessions, mockProviders, mockTools, mockConfig } from '@/test/mocks/data'

// Mock the OpenCode client
vi.mock('../opencode-client', () => ({
  openCodeClient: {
    healthCheck: vi.fn(),
    getSessions: vi.fn(),
    createSession: vi.fn(),
    deleteSession: vi.fn(),
    getProviders: vi.fn(),
    authenticateProvider: vi.fn(),
    getTools: vi.fn(),
    getToolExecutions: vi.fn(),
    getConfig: vi.fn(),
    updateConfig: vi.fn(),
    sendMessage: vi.fn(),
    shareSession: vi.fn(),
    getMessages: vi.fn(),
    subscribeToProviderUpdates: vi.fn(),
    subscribeToToolExecutions: vi.fn(),
    subscribeToSession: vi.fn(),
    disconnect: vi.fn(),
    on: vi.fn(),
    emit: vi.fn(),
    executeTool: vi.fn(),
    approveToolExecution: vi.fn(),
    cancelToolExecution: vi.fn(),
  }
}))

describe('useSessionStore', () => {
  const mockClient = openCodeClient as any

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks()
    
    // Setup default mock implementations
    mockClient.healthCheck.mockResolvedValue({ status: 'ok', version: '2.1.0' })
    mockClient.getSessions.mockResolvedValue(mockSessions)
    mockClient.getProviders.mockResolvedValue(mockProviders)
    mockClient.getTools.mockResolvedValue(mockTools)
    mockClient.getToolExecutions.mockResolvedValue([])
    mockClient.getConfig.mockResolvedValue(mockConfig)
    mockClient.subscribeToProviderUpdates.mockResolvedValue(undefined)
    mockClient.subscribeToToolExecutions.mockResolvedValue(undefined)
    mockClient.subscribeToSession.mockResolvedValue(undefined)
    mockClient.on.mockReturnValue(() => {})
  })

  afterEach(() => {
    // Reset store state
    const { result } = renderHook(() => useSessionStore())
    act(() => {
      result.current.actions.disconnect()
    })
  })

  describe('Connection Management', () => {
    it('should connect to OpenCode server successfully', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      expect(result.current.serverStatus).toBe('disconnected')
      
      await act(async () => {
        await result.current.actions.connect()
      })
      
      await waitFor(() => {
        expect(result.current.serverStatus).toBe('connected')
        expect(result.current.serverVersion).toBe('2.1.0')
      })
      
      expect(mockClient.healthCheck).toHaveBeenCalled()
      expect(mockClient.getSessions).toHaveBeenCalled()
      expect(mockClient.getProviders).toHaveBeenCalled()
      expect(mockClient.getTools).toHaveBeenCalled()
    })

    it('should handle connection failure', async () => {
      mockClient.healthCheck.mockRejectedValue(new Error('Connection failed'))
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.connect()
      })
      
      await waitFor(() => {
        expect(result.current.serverStatus).toBe('error')
        expect(result.current.connectionErrors.length).toBeGreaterThan(0)
      })
    })

    it('should disconnect from server', () => {
      const { result } = renderHook(() => useSessionStore())
      
      act(() => {
        result.current.actions.disconnect()
      })
      
      expect(result.current.serverStatus).toBe('disconnected')
      expect(result.current.isStreaming).toBe(false)
      expect(mockClient.disconnect).toHaveBeenCalled()
    })

    it('should check health status', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.checkHealth()
      })
      
      expect(mockClient.healthCheck).toHaveBeenCalled()
      expect(result.current.serverStatus).toBe('connected')
    })
  })

  describe('Session Management', () => {
    it('should load sessions', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.loadSessions()
      })
      
      await waitFor(() => {
        expect(result.current.sessions).toEqual(mockSessions)
        expect(result.current.isLoadingSessions).toBe(false)
      })
    })

    it('should create new session', async () => {
      const newSession = {
        id: 'new-session',
        name: 'New Session',
        provider: 'anthropic',
        model: 'claude-3-5-sonnet-20241022',
        created_at: Date.now(),
        updated_at: Date.now(),
        status: 'active' as const,
        message_count: 0,
        total_cost: 0,
        config: {}
      }
      
      mockClient.createSession.mockResolvedValue(newSession)
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        const session = await result.current.actions.createSession({
          provider: 'anthropic',
          model: 'claude-3-5-sonnet-20241022',
          name: 'New Session'
        })
        expect(session).toEqual(newSession)
      })
      
      await waitFor(() => {
        expect(result.current.sessions).toContainEqual(newSession)
        expect(result.current.activeSessionId).toBe('new-session')
      })
    })

    it('should delete session', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      // First load sessions
      await act(async () => {
        await result.current.actions.loadSessions()
      })
      
      const sessionToDelete = 'session-1'
      
      await act(async () => {
        await result.current.actions.deleteSession(sessionToDelete)
      })
      
      await waitFor(() => {
        expect(result.current.sessions.find(s => s.id === sessionToDelete)).toBeUndefined()
        expect(mockClient.deleteSession).toHaveBeenCalledWith(sessionToDelete)
      })
    })

    it('should set active session', async () => {
      mockClient.getMessages.mockResolvedValue([])
      
      const { result } = renderHook(() => useSessionStore())
      
      act(() => {
        result.current.actions.setActiveSession('session-1')
      })
      
      expect(result.current.activeSessionId).toBe('session-1')
      
      await waitFor(() => {
        expect(mockClient.getMessages).toHaveBeenCalledWith('session-1')
      })
    })

    it('should send message to session', async () => {
      mockClient.sendMessage.mockResolvedValue({ messageId: 'msg-123' })
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.sendMessage('session-1', 'Hello, world!')
      })
      
      expect(mockClient.sendMessage).toHaveBeenCalledWith('session-1', 'Hello, world!')
      expect(result.current.isStreaming).toBe(false)
    })

    it('should share session', async () => {
      mockClient.shareSession.mockResolvedValue({
        url: 'https://share.opencode.ai/session-1',
        expires_at: Date.now() + 86400000,
        password_protected: false,
        view_count: 0
      })
      
      const { result } = renderHook(() => useSessionStore())
      
      // First load sessions
      await act(async () => {
        await result.current.actions.loadSessions()
      })
      
      let shareUrl: string = ''
      await act(async () => {
        shareUrl = await result.current.actions.shareSession('session-1')
      })
      
      expect(shareUrl).toBe('https://share.opencode.ai/session-1')
      expect(mockClient.shareSession).toHaveBeenCalledWith('session-1')
    })
  })

  describe('Provider Management', () => {
    it('should load providers', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.loadProviders()
      })
      
      await waitFor(() => {
        expect(result.current.providers).toEqual(mockProviders)
        expect(result.current.isLoadingProviders).toBe(false)
      })
    })

    it('should authenticate provider', async () => {
      mockClient.authenticateProvider.mockResolvedValue({ success: true })
      
      const { result } = renderHook(() => useSessionStore())
      
      // First load providers
      await act(async () => {
        await result.current.actions.loadProviders()
      })
      
      let success: boolean = false
      await act(async () => {
        success = await result.current.actions.authenticateProvider('groq', { apiKey: 'test-key' })
      })
      
      expect(success).toBe(true)
      expect(mockClient.authenticateProvider).toHaveBeenCalledWith('groq', { apiKey: 'test-key' })
      
      await waitFor(() => {
        expect(result.current.authenticatedProviders).toContain('groq')
      })
    })

    it('should set active provider', () => {
      const { result } = renderHook(() => useSessionStore())
      
      act(() => {
        result.current.actions.setActiveProvider('anthropic')
      })
      
      expect(result.current.activeProvider).toBe('anthropic')
    })
  })

  describe('Tool Management', () => {
    it('should load tools and executions', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.loadTools()
      })
      
      await waitFor(() => {
        expect(result.current.availableTools).toEqual(mockTools)
        expect(result.current.isLoadingTools).toBe(false)
      })
    })

    it('should execute tool', async () => {
      const toolResult = {
        success: true,
        result: 'Tool executed successfully',
        execution_time: 150,
        tool_id: 'file_edit'
      }
      
      mockClient.executeTool.mockResolvedValue(toolResult)
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        const execution = await result.current.actions.executeTools('file_edit', { file: 'test.txt' })
        expect(execution.tool_id).toBe('file_edit')
        expect(execution.status).toBe('completed')
      })
      
      expect(mockClient.executeTool).toHaveBeenCalledWith('file_edit', { file: 'test.txt' }, undefined)
    })

    it('should approve tool execution', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.approveToolExecution('exec-1')
      })
      
      expect(mockClient.approveToolExecution).toHaveBeenCalledWith('exec-1')
    })

    it('should cancel tool execution', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.cancelToolExecution('exec-1')
      })
      
      expect(mockClient.cancelToolExecution).toHaveBeenCalledWith('exec-1')
    })
  })

  describe('Configuration Management', () => {
    it('should load configuration', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.loadConfig()
      })
      
      expect(result.current.config).toEqual(mockConfig)
    })

    it('should update configuration', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      // First load config
      await act(async () => {
        await result.current.actions.loadConfig()
      })
      
      const configUpdate = { theme: 'dark' as const }
      
      await act(async () => {
        await result.current.actions.updateConfig(configUpdate)
      })
      
      expect(mockClient.updateConfig).toHaveBeenCalledWith(configUpdate)
      expect(result.current.config?.theme).toBe('dark')
    })

    it('should validate configuration', async () => {
      mockClient.validateConfig.mockResolvedValue({
        valid: true,
        errors: [],
        warnings: []
      })
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        const validation = await result.current.actions.validateConfig(mockConfig)
        expect(validation.valid).toBe(true)
      })
      
      expect(mockClient.validateConfig).toHaveBeenCalledWith(mockConfig)
    })
  })

  describe('UI State Management', () => {
    it('should set current view', () => {
      const { result } = renderHook(() => useSessionStore())
      
      act(() => {
        result.current.actions.setCurrentView('providers')
      })
      
      expect(result.current.currentView).toBe('providers')
    })

    it('should toggle sidebar', () => {
      const { result } = renderHook(() => useSessionStore())
      
      const initialState = result.current.sidebarCollapsed
      
      act(() => {
        result.current.actions.toggleSidebar()
      })
      
      expect(result.current.sidebarCollapsed).toBe(!initialState)
    })

    it('should toggle timeline', () => {
      const { result } = renderHook(() => useSessionStore())
      
      const initialState = result.current.showTimeline
      
      act(() => {
        result.current.actions.toggleTimeline()
      })
      
      expect(result.current.showTimeline).toBe(!initialState)
    })
  })

  describe('Error Handling', () => {
    it('should handle session loading errors', async () => {
      mockClient.getSessions.mockRejectedValue(new Error('Failed to load sessions'))
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.loadSessions()
      })
      
      expect(result.current.isLoadingSessions).toBe(false)
      expect(result.current.sessions).toEqual([])
    })

    it('should handle provider loading errors', async () => {
      mockClient.getProviders.mockRejectedValue(new Error('Failed to load providers'))
      
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.loadProviders()
      })
      
      expect(result.current.isLoadingProviders).toBe(false)
      expect(result.current.providers).toEqual([])
    })

    it('should add and clear connection errors', () => {
      const { result } = renderHook(() => useSessionStore())
      
      act(() => {
        result.current.actions.addConnectionError('Test error')
      })
      
      expect(result.current.connectionErrors).toContain('Test error')
      
      act(() => {
        result.current.actions.clearConnectionErrors()
      })
      
      expect(result.current.connectionErrors).toEqual([])
    })
  })

  describe('Real-time Updates', () => {
    it('should handle session update events', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      // Simulate connecting which sets up event listeners
      await act(async () => {
        await result.current.actions.connect()
      })
      
      // Get the event handler that was registered
      const onCall = mockClient.on.mock.calls.find(call => call[0] === 'session_update')
      expect(onCall).toBeDefined()
      
      const eventHandler = onCall[1]
      
      // Simulate a session update event
      act(() => {
        eventHandler({
          sessionId: 'session-1',
          update: {
            type: 'message',
            data: {
              id: 'new-msg',
              role: 'assistant',
              content: 'New message',
              timestamp: Date.now(),
              session_id: 'session-1'
            }
          }
        })
      })
      
      await waitFor(() => {
        expect(result.current.sessionMessages['session-1']).toBeDefined()
      })
    })

    it('should handle provider health events', async () => {
      const { result } = renderHook(() => useSessionStore())
      
      await act(async () => {
        await result.current.actions.connect()
      })
      
      const onCall = mockClient.on.mock.calls.find(call => call[0] === 'provider_health')
      expect(onCall).toBeDefined()
      
      const eventHandler = onCall[1]
      
      act(() => {
        eventHandler({
          provider_id: 'anthropic',
          status: 'degraded',
          response_time: 1200,
          last_check: Date.now(),
          uptime: 95.0,
          region: 'us-east-1'
        })
      })
      
      await waitFor(() => {
        const healthData = result.current.providerHealth.find(p => p.provider_id === 'anthropic')
        expect(healthData?.status).toBe('degraded')
      })
    })
  })
})