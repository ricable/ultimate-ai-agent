import React from 'react'
import { render, screen, waitFor } from '@/test/utils'
import userEvent from '@testing-library/user-event'
import { ConnectionStatus } from '../connection-status'
import { useSessionStore } from '@/lib/session-store'

vi.mock('@/lib/session-store')

const mockUseSessionStore = useSessionStore as vi.MockedFunction<typeof useSessionStore>

describe('ConnectionStatus', () => {
  const mockStore = {
    serverStatus: 'connected' as const,
    serverVersion: '2.1.0',
    connectionErrors: [],
    actions: {
      connect: vi.fn(),
      disconnect: vi.fn(),
      checkHealth: vi.fn(),
      clearConnectionErrors: vi.fn(),
    }
  }

  beforeEach(() => {
    mockUseSessionStore.mockReturnValue(mockStore as any)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders connected status', () => {
    render(<ConnectionStatus />)
    
    expect(screen.getByText('Connected')).toBeInTheDocument()
    expect(screen.getByText('v2.1.0')).toBeInTheDocument()
    expect(screen.getByText(/connected to opencode server/i)).toBeInTheDocument()
  })

  it('renders disconnected status', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'disconnected',
      serverVersion: null,
    } as any)

    render(<ConnectionStatus />)
    
    expect(screen.getByText('Disconnected')).toBeInTheDocument()
    expect(screen.getByText(/not connected to server/i)).toBeInTheDocument()
    
    const connectButton = screen.getByRole('button', { name: /connect/i })
    expect(connectButton).toBeInTheDocument()
  })

  it('renders connecting status', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'connecting',
      serverVersion: null,
    } as any)

    render(<ConnectionStatus />)
    
    expect(screen.getByText('Connecting')).toBeInTheDocument()
    expect(screen.getByText(/connecting to server/i)).toBeInTheDocument()
  })

  it('renders error status with error messages', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'error',
      serverVersion: null,
      connectionErrors: ['Connection timeout', 'Server unreachable'],
    } as any)

    render(<ConnectionStatus />)
    
    expect(screen.getByText('Error')).toBeInTheDocument()
    expect(screen.getByText('Connection timeout')).toBeInTheDocument()
    expect(screen.getByText('Server unreachable')).toBeInTheDocument()
    
    const retryButton = screen.getByRole('button', { name: /retry/i })
    expect(retryButton).toBeInTheDocument()
  })

  it('handles connect action', async () => {
    const mockConnect = vi.fn().mockResolvedValue(undefined)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'disconnected',
      actions: {
        ...mockStore.actions,
        connect: mockConnect,
      }
    } as any)

    render(<ConnectionStatus />)
    
    const connectButton = screen.getByRole('button', { name: /connect/i })
    await userEvent.click(connectButton)
    
    expect(mockConnect).toHaveBeenCalled()
  })

  it('handles disconnect action', async () => {
    const mockDisconnect = vi.fn()
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        disconnect: mockDisconnect,
      }
    } as any)

    render(<ConnectionStatus />)
    
    const actionsButton = screen.getByRole('button', { name: /more options/i })
    await userEvent.click(actionsButton)
    
    const disconnectButton = screen.getByRole('menuitem', { name: /disconnect/i })
    await userEvent.click(disconnectButton)
    
    expect(mockDisconnect).toHaveBeenCalled()
  })

  it('handles health check action', async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue(undefined)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        checkHealth: mockCheckHealth,
      }
    } as any)

    render(<ConnectionStatus />)
    
    const actionsButton = screen.getByRole('button', { name: /more options/i })
    await userEvent.click(actionsButton)
    
    const healthCheckButton = screen.getByRole('menuitem', { name: /check health/i })
    await userEvent.click(healthCheckButton)
    
    expect(mockCheckHealth).toHaveBeenCalled()
  })

  it('clears errors when requested', async () => {
    const mockClearErrors = vi.fn()
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'error',
      connectionErrors: ['Test error'],
      actions: {
        ...mockStore.actions,
        clearConnectionErrors: mockClearErrors,
      }
    } as any)

    render(<ConnectionStatus />)
    
    const clearButton = screen.getByRole('button', { name: /clear errors/i })
    await userEvent.click(clearButton)
    
    expect(mockClearErrors).toHaveBeenCalled()
  })

  it('shows loading state during connection attempt', async () => {
    const mockConnect = vi.fn().mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 100))
    )
    
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'disconnected',
      actions: {
        ...mockStore.actions,
        connect: mockConnect,
      }
    } as any)

    render(<ConnectionStatus />)
    
    const connectButton = screen.getByRole('button', { name: /connect/i })
    await userEvent.click(connectButton)
    
    expect(screen.getByText(/connecting/i)).toBeInTheDocument()
    expect(connectButton).toBeDisabled()
  })

  it('displays connection latency and uptime', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'connected',
      serverVersion: '2.1.0',
      connectionStats: {
        latency: 45,
        uptime: 99.9,
        lastPing: Date.now() - 5000
      }
    } as any)

    render(<ConnectionStatus />)
    
    // Check for connection statistics
    expect(screen.getByText(/45ms/)).toBeInTheDocument()
    expect(screen.getByText(/99\.9%/)).toBeInTheDocument()
  })

  it('has proper ARIA attributes', () => {
    render(<ConnectionStatus />)
    
    const statusElement = screen.getByRole('status')
    expect(statusElement).toBeInTheDocument()
    expect(statusElement).toHaveAttribute('aria-live', 'polite')
    
    // Check that status has proper label
    expect(screen.getByLabelText(/connection status/i)).toBeInTheDocument()
  })

  it('shows visual indicators for different states', () => {
    // Test connected state
    render(<ConnectionStatus />)
    expect(screen.getByTestId('status-indicator')).toHaveClass('connected')
    
    // Test disconnected state  
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'disconnected'
    } as any)
    
    render(<ConnectionStatus />)
    expect(screen.getByTestId('status-indicator')).toHaveClass('disconnected')
    
    // Test error state
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      serverStatus: 'error'
    } as any)
    
    render(<ConnectionStatus />)
    expect(screen.getByTestId('status-indicator')).toHaveClass('error')
  })

  it('auto-refreshes connection status', async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue(undefined)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        checkHealth: mockCheckHealth,
      }
    } as any)

    // Mock timers to test auto-refresh
    vi.useFakeTimers()
    
    render(<ConnectionStatus />)
    
    // Fast-forward time to trigger auto-refresh
    vi.advanceTimersByTime(30000) // 30 seconds
    
    await waitFor(() => {
      expect(mockCheckHealth).toHaveBeenCalled()
    })
    
    vi.useRealTimers()
  })
})