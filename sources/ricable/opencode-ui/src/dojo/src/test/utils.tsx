import * as React from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { ThemeProvider } from 'next-themes'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Create a simple test wrapper
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}

// Custom render function that includes providers
const customRender = (
  ui: React.ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

// Re-export everything from RTL
export * from '@testing-library/react'
export { customRender as render }

// Common test utilities
export const mockSessionStore = {
  sessions: [],
  activeSessionId: null,
  providers: [],
  serverStatus: 'connected' as const,
  actions: {
    connect: vi.fn(),
    disconnect: vi.fn(),
    loadSessions: vi.fn(),
    createSession: vi.fn(),
    deleteSession: vi.fn(),
    setActiveSession: vi.fn(),
    loadProviders: vi.fn(),
    authenticateProvider: vi.fn(),
    setCurrentView: vi.fn(),
    toggleSidebar: vi.fn(),
  }
}

// Helper to create mock session data
export const createMockSession = (overrides = {}) => ({
  id: `session-${Date.now()}`,
  name: 'Test Session',
  provider: 'anthropic',
  model: 'claude-3-5-sonnet-20241022',
  created_at: Date.now(),
  updated_at: Date.now(),
  status: 'active' as const,
  message_count: 0,
  total_cost: 0,
  config: {},
  ...overrides
})

// Helper to create mock provider data
export const createMockProvider = (overrides = {}) => ({
  id: 'test-provider',
  name: 'Test Provider',
  type: 'test',
  models: ['test-model-1', 'test-model-2'],
  authenticated: true,
  status: 'online' as const,
  cost_per_1k_tokens: 0.001,
  avg_response_time: 500,
  description: 'Test provider for testing',
  ...overrides
})

// Helper to wait for async state updates
export const waitForStoreUpdate = (timeout = 1000) => 
  new Promise(resolve => setTimeout(resolve, timeout))

// Mock user event helpers
export const createMockUserEvent = () => ({
  click: vi.fn(),
  type: vi.fn(),
  keyboard: vi.fn(),
  hover: vi.fn(),
  unhover: vi.fn(),
  selectOptions: vi.fn(),
  deselectOptions: vi.fn(),
  upload: vi.fn(),
  clear: vi.fn(),
  tab: vi.fn(),
  dblClick: vi.fn(),
})

// Accessibility testing helpers
export const checkAccessibility = async (container: HTMLElement) => {
  const { axe } = await import('jest-axe')
  const results = await axe(container)
  return results
}