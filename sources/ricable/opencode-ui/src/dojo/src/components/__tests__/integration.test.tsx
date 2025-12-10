import React from 'react'
import { render, screen, waitFor } from '@/test/utils'
import userEvent from '@testing-library/user-event'
import { OpenCodeLayout } from '../opencode/opencode-layout'
import { useSessionStore } from '@/lib/session-store'
import { mockProviders, mockSessions, mockTools } from '@/test/mocks/data'

vi.mock('@/lib/session-store')

const mockUseSessionStore = useSessionStore as vi.MockedFunction<typeof useSessionStore>

describe('OpenCode Integration Tests', () => {
  const mockStore = {
    serverStatus: 'connected' as const,
    serverVersion: '2.1.0',
    sessions: mockSessions,
    activeSessionId: null,
    providers: mockProviders,
    availableTools: mockTools,
    currentView: 'projects' as const,
    sidebarCollapsed: false,
    isLoadingSessions: false,
    isLoadingProviders: false,
    isLoadingTools: false,
    sessionMessages: {},
    connectionErrors: [],
    actions: {
      connect: vi.fn(),
      disconnect: vi.fn(),
      loadSessions: vi.fn(),
      createSession: vi.fn(),
      deleteSession: vi.fn(),
      setActiveSession: vi.fn(),
      loadProviders: vi.fn(),
      authenticateProvider: vi.fn(),
      loadTools: vi.fn(),
      setCurrentView: vi.fn(),
      toggleSidebar: vi.fn(),
      sendMessage: vi.fn(),
      shareSession: vi.fn(),
    }
  }

  beforeEach(() => {
    mockUseSessionStore.mockReturnValue(mockStore as any)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Complete Session Workflow', () => {
    it('allows user to create and interact with a session', async () => {
      const mockCreateSession = vi.fn().mockResolvedValue({
        id: 'new-session',
        name: 'Test Session',
        provider: 'anthropic',
        model: 'claude-3-5-sonnet-20241022',
        status: 'active',
        created_at: Date.now(),
        updated_at: Date.now(),
        message_count: 0,
        total_cost: 0,
        config: {}
      })

      const mockSendMessage = vi.fn().mockResolvedValue({ messageId: 'msg-1' })

      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        actions: {
          ...mockStore.actions,
          createSession: mockCreateSession,
          sendMessage: mockSendMessage,
        }
      } as any)

      render(<OpenCodeLayout />)

      // 1. Start at projects view
      expect(screen.getByText('Projects')).toBeInTheDocument()
      expect(screen.getByText('React Component Development')).toBeInTheDocument()

      // 2. Create a new session
      const newSessionButton = screen.getByRole('button', { name: /new session/i })
      await userEvent.click(newSessionButton)

      // Fill out session creation form
      await userEvent.type(screen.getByLabelText(/session name/i), 'Test Session')
      
      const providerSelect = screen.getByLabelText(/provider/i)
      await userEvent.click(providerSelect)
      await userEvent.click(screen.getByText('Anthropic'))

      const createButton = screen.getByRole('button', { name: /create session/i })
      await userEvent.click(createButton)

      await waitFor(() => {
        expect(mockCreateSession).toHaveBeenCalledWith({
          name: 'Test Session',
          provider: 'anthropic',
          model: expect.any(String)
        })
      })

      // 3. Session should be created and active
      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        activeSessionId: 'new-session',
        currentView: 'session',
        actions: {
          ...mockStore.actions,
          createSession: mockCreateSession,
          sendMessage: mockSendMessage,
        }
      } as any)

      // Rerender to reflect new state
      render(<OpenCodeLayout />)

      // 4. Send a message in the session
      const messageInput = screen.getByPlaceholderText(/type your message/i)
      await userEvent.type(messageInput, 'Hello, can you help me with React?')

      const sendButton = screen.getByRole('button', { name: /send/i })
      await userEvent.click(sendButton)

      await waitFor(() => {
        expect(mockSendMessage).toHaveBeenCalledWith('new-session', 'Hello, can you help me with React?')
      })
    })

    it('handles multi-provider authentication flow', async () => {
      const mockAuthenticateProvider = vi.fn()
        .mockResolvedValueOnce(true) // First provider succeeds
        .mockResolvedValueOnce(false) // Second provider fails

      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        currentView: 'providers',
        providers: mockProviders.map(p => ({ ...p, authenticated: false })),
        actions: {
          ...mockStore.actions,
          authenticateProvider: mockAuthenticateProvider,
        }
      } as any)

      render(<OpenCodeLayout />)

      // Navigate to providers view
      const providersTab = screen.getByRole('tab', { name: /providers/i })
      await userEvent.click(providersTab)

      // Authenticate first provider (should succeed)
      const anthropicAuth = screen.getAllByText(/authenticate/i)[0]
      await userEvent.click(anthropicAuth)

      // Fill in API key
      const apiKeyInput = screen.getByLabelText(/api key/i)
      await userEvent.type(apiKeyInput, 'test-api-key-1')

      const saveButton = screen.getByRole('button', { name: /save/i })
      await userEvent.click(saveButton)

      await waitFor(() => {
        expect(mockAuthenticateProvider).toHaveBeenCalledWith('anthropic', {
          apiKey: 'test-api-key-1'
        })
      })

      // Should show success message
      expect(screen.getByText(/successfully authenticated/i)).toBeInTheDocument()

      // Try to authenticate second provider (should fail)
      const openaiAuth = screen.getAllByText(/authenticate/i)[0]
      await userEvent.click(openaiAuth)

      const apiKeyInput2 = screen.getByLabelText(/api key/i)
      await userEvent.type(apiKeyInput2, 'invalid-key')

      const saveButton2 = screen.getByRole('button', { name: /save/i })
      await userEvent.click(saveButton2)

      await waitFor(() => {
        expect(mockAuthenticateProvider).toHaveBeenCalledWith('openai', {
          apiKey: 'invalid-key'
        })
      })

      // Should show error message
      expect(screen.getByText(/authentication failed/i)).toBeInTheDocument()
    })

    it('demonstrates tool execution workflow', async () => {
      const mockExecuteTool = vi.fn().mockResolvedValue({
        id: 'exec-1',
        tool_id: 'file_edit',
        status: 'completed',
        result: 'File created successfully',
        params: { file: 'test.txt', content: 'Hello World' }
      })

      const mockApproveToolExecution = vi.fn()

      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        currentView: 'tools',
        pendingApprovals: [
          {
            id: 'exec-pending',
            tool_id: 'bash',
            status: 'pending',
            params: { command: 'npm install' },
            created_at: Date.now()
          }
        ],
        actions: {
          ...mockStore.actions,
          executeTools: mockExecuteTool,
          approveToolExecution: mockApproveToolExecution,
        }
      } as any)

      render(<OpenCodeLayout />)

      // Navigate to tools view
      const toolsTab = screen.getByRole('tab', { name: /tools/i })
      await userEvent.click(toolsTab)

      // Should see available tools
      expect(screen.getByText('File Editor')).toBeInTheDocument()
      expect(screen.getByText('Shell Command')).toBeInTheDocument()

      // Execute a tool
      const fileEditTool = screen.getByText('File Editor').closest('div')
      const executeButton = fileEditTool?.querySelector('button[aria-label*="execute"]')
      
      if (executeButton) {
        await userEvent.click(executeButton)

        // Fill in tool parameters
        await userEvent.type(screen.getByLabelText(/file path/i), 'test.txt')
        await userEvent.type(screen.getByLabelText(/content/i), 'Hello World')

        const runButton = screen.getByRole('button', { name: /run tool/i })
        await userEvent.click(runButton)

        await waitFor(() => {
          expect(mockExecuteTool).toHaveBeenCalledWith('file_edit', {
            file: 'test.txt',
            content: 'Hello World'
          })
        })
      }

      // Handle pending approval
      expect(screen.getByText(/pending approval/i)).toBeInTheDocument()
      expect(screen.getByText('npm install')).toBeInTheDocument()

      const approveButton = screen.getByRole('button', { name: /approve/i })
      await userEvent.click(approveButton)

      await waitFor(() => {
        expect(mockApproveToolExecution).toHaveBeenCalledWith('exec-pending')
      })
    })

    it('handles error states gracefully', async () => {
      // Mock connection error
      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        serverStatus: 'error',
        connectionErrors: ['Connection timeout', 'Server unreachable'],
      } as any)

      render(<OpenCodeLayout />)

      // Should show error state
      expect(screen.getByText('Error')).toBeInTheDocument()
      expect(screen.getByText('Connection timeout')).toBeInTheDocument()
      expect(screen.getByText('Server unreachable')).toBeInTheDocument()

      // Should have retry button
      const retryButton = screen.getByRole('button', { name: /retry/i })
      expect(retryButton).toBeInTheDocument()

      // Mock loading states
      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        serverStatus: 'connected',
        isLoadingSessions: true,
        isLoadingProviders: true,
        isLoadingTools: true,
      } as any)

      render(<OpenCodeLayout />)

      // Should show loading indicators
      expect(screen.getAllByText(/loading/i).length).toBeGreaterThan(0)
    })

    it('supports keyboard navigation throughout the interface', async () => {
      render(<OpenCodeLayout />)

      // Tab through main navigation
      await userEvent.tab()
      expect(screen.getByRole('tab', { name: /projects/i })).toHaveFocus()

      await userEvent.tab()
      expect(screen.getByRole('tab', { name: /providers/i })).toHaveFocus()

      await userEvent.tab()
      expect(screen.getByRole('tab', { name: /tools/i })).toHaveFocus()

      // Use arrow keys to navigate tabs
      await userEvent.keyboard('{ArrowLeft}')
      expect(screen.getByRole('tab', { name: /providers/i })).toHaveFocus()

      await userEvent.keyboard('{ArrowLeft}')
      expect(screen.getByRole('tab', { name: /projects/i })).toHaveFocus()

      // Activate tab with Enter or Space
      await userEvent.keyboard('{Enter}')
      expect(mockStore.actions.setCurrentView).toHaveBeenCalledWith('projects')
    })
  })

  describe('Responsive Behavior', () => {
    it('adapts to different screen sizes', () => {
      // Mock different viewport sizes
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768, // Tablet size
      })

      render(<OpenCodeLayout />)

      // Sidebar should be collapsible on smaller screens
      const sidebarToggle = screen.getByRole('button', { name: /toggle sidebar/i })
      expect(sidebarToggle).toBeInTheDocument()
    })

    it('handles touch interactions', async () => {
      render(<OpenCodeLayout />)

      // Test touch interactions on mobile-like interface
      const sessionCard = screen.getByText('React Component Development').closest('div')
      
      if (sessionCard) {
        // Simulate touch events
        await userEvent.pointer([
          { target: sessionCard, keys: '[TouchA>]' },
          { target: sessionCard, keys: '[/TouchA]' }
        ])

        expect(mockStore.actions.setActiveSession).toHaveBeenCalled()
      }
    })
  })

  describe('Real-time Updates', () => {
    it('displays real-time session updates', async () => {
      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        isStreaming: true,
        streamingSessionId: 'session-1',
        activeSessionId: 'session-1',
        currentView: 'session',
      } as any)

      render(<OpenCodeLayout />)

      // Should show streaming indicator
      expect(screen.getByText(/ai is typing/i)).toBeInTheDocument()

      // Input should be disabled during streaming
      const messageInput = screen.getByPlaceholderText(/type your message/i)
      expect(messageInput).toBeDisabled()

      const sendButton = screen.getByRole('button', { name: /send/i })
      expect(sendButton).toBeDisabled()
    })

    it('handles provider health updates', () => {
      mockUseSessionStore.mockReturnValue({
        ...mockStore,
        currentView: 'providers',
        providerHealth: [
          { provider_id: 'anthropic', status: 'degraded', response_time: 1200, last_check: Date.now(), uptime: 95.0, region: 'us-east-1' },
          { provider_id: 'openai', status: 'online', response_time: 750, last_check: Date.now(), uptime: 99.5, region: 'us-west-2' }
        ]
      } as any)

      render(<OpenCodeLayout />)

      // Should show provider health status
      expect(screen.getByText(/degraded/i)).toBeInTheDocument()
      expect(screen.getByText(/online/i)).toBeInTheDocument()
      expect(screen.getByText(/1200ms/)).toBeInTheDocument()
      expect(screen.getByText(/750ms/)).toBeInTheDocument()
    })
  })
})