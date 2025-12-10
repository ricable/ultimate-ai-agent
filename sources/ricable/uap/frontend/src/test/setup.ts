/**
 * Vitest setup file for UAP Frontend Testing
 * Configures test environment, mocks, and global utilities
 */

import { expect, afterEach, vi, beforeAll, afterAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'

// Global test setup
beforeAll(() => {
  // Mock auth context for testing
  vi.mock('../auth/AuthContext', () => ({
    useAuth: () => ({
      token: 'mock-token',
      user: { id: 'test-user', email: 'test@example.com' },
      isAuthenticated: true,
      login: vi.fn(),
      logout: vi.fn(),
      register: vi.fn(),
    }),
    AuthProvider: ({ children }: any) => children,
  }))

  // Mock CopilotKit components
  vi.mock('@copilotkit/react-core', () => ({
    CopilotProvider: ({ children }: any) => children,
    useCopilotChat: () => ({
      messages: [],
      sendMessage: vi.fn(),
      isLoading: false,
    }),
  }))

  vi.mock('@copilotkit/react-ui', () => ({
    CopilotSidebar: () => {
      const React = require('react')
      return React.createElement('div', { 'data-testid': 'copilot-sidebar' }, 'CopilotSidebar')
    },
    CopilotPopup: () => {
      const React = require('react')
      return React.createElement('div', { 'data-testid': 'copilot-popup' }, 'CopilotPopup')
    },
  }))

  // Mock WebSocket for testing
  global.WebSocket = vi.fn(() => ({
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    send: vi.fn(),
    close: vi.fn(),
    readyState: WebSocket.OPEN,
  })) as any

  // Mock window.location for WebSocket URL construction
  Object.defineProperty(window, 'location', {
    value: {
      protocol: 'http:',
      host: 'localhost:3000',
      href: 'http://localhost:3000',
    },
    writable: true,
  })

  // Performance measurement utilities for UI load time testing
  global.performance = {
    ...global.performance,
    mark: vi.fn(),
    measure: vi.fn(),
    getEntriesByName: vi.fn(() => []),
    getEntriesByType: vi.fn(() => []),
    now: vi.fn(() => Date.now()),
  }

  // Mock IntersectionObserver for component visibility testing
  global.IntersectionObserver = vi.fn(() => ({
    observe: vi.fn(),
    disconnect: vi.fn(),
    unobserve: vi.fn(),
  })) as any

  // Mock ResizeObserver for responsive component testing
  global.ResizeObserver = vi.fn(() => ({
    observe: vi.fn(),
    disconnect: vi.fn(),
    unobserve: vi.fn(),
  })) as any
})

// Cleanup after each test
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// Global cleanup
afterAll(() => {
  vi.resetAllMocks()
})

// Custom matchers for UAP-specific testing
expect.extend({
  toBeWithinResponseTime(received: number, maxTime: number) {
    const pass = received <= maxTime
    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received}ms to exceed ${maxTime}ms`
          : `Expected ${received}ms to be within ${maxTime}ms response time requirement`,
    }
  },
  toHaveAGUICompliantStructure(received: any) {
    const hasType = typeof received.type === 'string'
    const hasContent = 'content' in received
    const hasMetadata = 'metadata' in received && typeof received.metadata === 'object'
    
    const pass = hasType && hasContent && hasMetadata
    return {
      pass,
      message: () =>
        pass
          ? `Expected object not to have AG-UI compliant structure`
          : `Expected object to have AG-UI compliant structure (type, content, metadata)`,
    }
  },
})

// Global test utilities
export const testUtils = {
  // Mock AG-UI event factory
  createMockAGUIEvent: (type: string, content: string, metadata: any = {}) => ({
    type,
    content,
    metadata,
  }),

  // Performance measurement utilities
  measureComponentLoadTime: async (componentRenderer: () => Promise<any>) => {
    const startTime = performance.now()
    await componentRenderer()
    const endTime = performance.now()
    return endTime - startTime
  },

  // Wait for component to be interactive
  waitForInteractivity: async (element: Element, timeout = 1000) => {
    return new Promise((resolve, reject) => {
      const startTime = Date.now()
      
      const check = () => {
        const isInteractive = !element.hasAttribute('disabled') && 
                            !element.classList.contains('loading')
        
        if (isInteractive) {
          resolve(Date.now() - startTime)
        } else if (Date.now() - startTime > timeout) {
          reject(new Error(`Element not interactive within ${timeout}ms`))
        } else {
          setTimeout(check, 10)
        }
      }
      
      check()
    })
  },

  // Mock network delay for testing
  mockNetworkDelay: (delay: number) => {
    return new Promise(resolve => setTimeout(resolve, delay))
  },
}

// Export types for tests
export type MockAGUIEvent = {
  type: string
  content: string
  metadata: Record<string, any>
}

export type PerformanceMetrics = {
  loadTime: number
  timeToInteractive: number
  firstContentfulPaint?: number
}

// Declare extended matchers for TypeScript
declare module 'vitest' {
  interface Assertion<T = any> {
    toBeWithinResponseTime(maxTime: number): T
    toHaveAGUICompliantStructure(): T
  }
}