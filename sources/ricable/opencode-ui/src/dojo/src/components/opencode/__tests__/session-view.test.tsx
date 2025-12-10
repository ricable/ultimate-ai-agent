import React from 'react'
import { render, screen, waitFor } from '@/test/utils'
import userEvent from '@testing-library/user-event'
import { SessionView } from '../session-view'
import { useSessionStore } from '@/lib/session-store'
import { mockSessions, mockMessages } from '@/test/mocks/data'

vi.mock('@/lib/session-store')

const mockUseSessionStore = useSessionStore as vi.MockedFunction<typeof useSessionStore>

describe('SessionView', () => {
  const mockSession = mockSessions[0]
  const mockSessionMessages = mockMessages['session-1']

  const mockStore = {
    sessions: mockSessions,
    activeSessionId: 'session-1',
    sessionMessages: { 'session-1': mockSessionMessages },
    isStreaming: false,
    streamingSessionId: null,
    actions: {
      sendMessage: vi.fn(),
      loadSessionMessages: vi.fn(),
      shareSession: vi.fn(),
      deleteSession: vi.fn(),
      setActiveSession: vi.fn(),
    }
  }

  beforeEach(() => {
    mockUseSessionStore.mockReturnValue(mockStore as any)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders session view with messages', async () => {
    render(<SessionView sessionId="session-1" />)
    
    expect(screen.getByText('React Component Development')).toBeInTheDocument()
    
    await waitFor(() => {
      expect(screen.getByText('Help me create a React component for displaying user profiles.')).toBeInTheDocument()
      expect(screen.getByText(/I'll help you create a React component/)).toBeInTheDocument()
    })
  })

  it('shows session metadata', () => {
    render(<SessionView sessionId="session-1" />)
    
    expect(screen.getByText('anthropic')).toBeInTheDocument()
    expect(screen.getByText('claude-3-5-sonnet-20241022')).toBeInTheDocument()
    expect(screen.getByText('12 messages')).toBeInTheDocument()
    expect(screen.getByText('$0.45')).toBeInTheDocument()
  })

  it('handles sending new messages', async () => {
    const mockSendMessage = vi.fn().mockResolvedValue(undefined)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        sendMessage: mockSendMessage
      }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    const messageInput = screen.getByPlaceholderText(/type your message/i)
    const sendButton = screen.getByRole('button', { name: /send/i })
    
    await userEvent.type(messageInput, 'This is a test message')
    await userEvent.click(sendButton)
    
    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledWith('session-1', 'This is a test message')
    })
    
    // Message input should be cleared after sending
    expect((messageInput as HTMLInputElement).value).toBe('')
  })

  it('prevents sending empty messages', async () => {
    const mockSendMessage = vi.fn()
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        sendMessage: mockSendMessage
      }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    const sendButton = screen.getByRole('button', { name: /send/i })
    
    // Try to send empty message
    await userEvent.click(sendButton)
    
    expect(mockSendMessage).not.toHaveBeenCalled()
  })

  it('shows streaming indicator when streaming', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      isStreaming: true,
      streamingSessionId: 'session-1'
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    expect(screen.getByText(/ai is typing/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
  })

  it('handles session sharing', async () => {
    const mockShareSession = vi.fn().mockResolvedValue('https://share.opencode.ai/session-1')
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        shareSession: mockShareSession
      }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    const shareButton = screen.getByRole('button', { name: /share/i })
    await userEvent.click(shareButton)
    
    await waitFor(() => {
      expect(mockShareSession).toHaveBeenCalledWith('session-1')
      expect(screen.getByText(/session shared/i)).toBeInTheDocument()
    })
  })

  it('displays tool calls in messages', async () => {
    render(<SessionView sessionId="session-1" />)
    
    await waitFor(() => {
      // Check for tool call display
      expect(screen.getByText(/file_edit/i)).toBeInTheDocument()
      expect(screen.getByText(/UserProfile\.tsx/)).toBeInTheDocument()
    })
  })

  it('handles message retry on failure', async () => {
    const mockSendMessage = vi.fn()
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(undefined)

    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        sendMessage: mockSendMessage
      }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    const messageInput = screen.getByPlaceholderText(/type your message/i)
    const sendButton = screen.getByRole('button', { name: /send/i })
    
    await userEvent.type(messageInput, 'Test message')
    await userEvent.click(sendButton)
    
    // Should show error message
    await waitFor(() => {
      expect(screen.getByText(/failed to send message/i)).toBeInTheDocument()
    })
    
    // Should show retry button
    const retryButton = screen.getByRole('button', { name: /retry/i })
    await userEvent.click(retryButton)
    
    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledTimes(2)
    })
  })

  it('shows empty state when no messages', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      sessionMessages: { 'session-1': [] }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    expect(screen.getByText(/no messages yet/i)).toBeInTheDocument()
    expect(screen.getByText(/start a conversation/i)).toBeInTheDocument()
  })

  it('handles keyboard shortcuts', async () => {
    const mockSendMessage = vi.fn().mockResolvedValue(undefined)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        sendMessage: mockSendMessage
      }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    const messageInput = screen.getByPlaceholderText(/type your message/i)
    await userEvent.type(messageInput, 'Test message')
    
    // Send with Ctrl+Enter
    await userEvent.keyboard('{Control>}{Enter}{/Control}')
    
    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledWith('session-1', 'Test message')
    })
  })

  it('auto-scrolls to latest message', async () => {
    const scrollIntoViewMock = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoViewMock

    render(<SessionView sessionId="session-1" />)
    
    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled()
    })
  })

  it('shows session status and actions', () => {
    render(<SessionView sessionId="session-1" />)
    
    expect(screen.getByText('active')).toBeInTheDocument()
    
    const actionsButton = screen.getByRole('button', { name: /more actions/i })
    expect(actionsButton).toBeInTheDocument()
  })

  it('handles session deletion with confirmation', async () => {
    const mockDeleteSession = vi.fn().mockResolvedValue(undefined)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        deleteSession: mockDeleteSession
      }
    } as any)

    render(<SessionView sessionId="session-1" />)
    
    const actionsButton = screen.getByRole('button', { name: /more actions/i })
    await userEvent.click(actionsButton)
    
    const deleteButton = screen.getByRole('menuitem', { name: /delete session/i })
    await userEvent.click(deleteButton)
    
    // Should show confirmation dialog
    await waitFor(() => {
      expect(screen.getByText(/are you sure/i)).toBeInTheDocument()
    })
    
    const confirmButton = screen.getByRole('button', { name: /delete/i })
    await userEvent.click(confirmButton)
    
    await waitFor(() => {
      expect(mockDeleteSession).toHaveBeenCalledWith('session-1')
    })
  })

  it('is accessible', async () => {
    const { container } = render(<SessionView sessionId="session-1" />)
    
    // Check for proper ARIA structure
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: /message input/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
    
    // Check message list accessibility
    const messageList = screen.getByRole('log')
    expect(messageList).toBeInTheDocument()
    expect(messageList).toHaveAttribute('aria-live', 'polite')
    
    // Run accessibility check
    const { checkAccessibility } = await import('@/test/utils')
    const results = await checkAccessibility(container)
    expect(results).toHaveNoViolations()
  })

  it('handles session not found', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      sessions: [],
      activeSessionId: null
    } as any)

    render(<SessionView sessionId="nonexistent-session" />)
    
    expect(screen.getByText(/session not found/i)).toBeInTheDocument()
  })
})