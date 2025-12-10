/**
 * Enhanced test utilities for UAP Frontend Testing
 * Provides React testing utilities, mocks, and performance helpers
 */

import React from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { vi } from 'vitest'

// Mock providers for components that need context
const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <div data-testid="mock-providers">{children}</div>
}

// Custom render function that includes providers
const customRender = (
  ui: React.ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: MockProviders, ...options })

// Mock WebSocket for testing
export const createMockWebSocket = () => {
  const mockWS = {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    send: vi.fn(),
    close: vi.fn(),
    readyState: WebSocket.OPEN,
    onopen: null,
    onmessage: null,
    onclose: null,
    onerror: null,
  }
  
  // Simulate connection events
  setTimeout(() => {
    if (mockWS.onopen) mockWS.onopen({} as Event)
  }, 10)
  
  return mockWS
}

// Mock AG-UI event factory
export const createMockAGUIEvent = (type: string, content?: string, metadata: any = {}) => ({
  type,
  content: content || `Mock ${type} content`,
  timestamp: Date.now(),
  metadata,
})

// Performance testing utilities
export const measureRenderTime = async (renderFn: () => void) => {
  const startTime = performance.now()
  renderFn()
  return performance.now() - startTime
}

// Wait for elements to be interactive
export const waitForInteractivity = async (element: Element, timeout = 1000) => {
  return new Promise<number>((resolve, reject) => {
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
}

// Mock network delay
export const mockNetworkDelay = (delay: number) => {
  return new Promise(resolve => setTimeout(resolve, delay))
}

// Enhanced matchers for AG-UI compliance
export const expectAGUICompliant = (event: any) => {
  expect(event).toHaveProperty('type')
  expect(event).toHaveProperty('metadata')
  expect(typeof event.type).toBe('string')
  expect(typeof event.metadata).toBe('object')
}

// Mock disabled components to enable testing
export const enableDisabledElements = (container: HTMLElement) => {
  const disabledElements = container.querySelectorAll('[disabled]')
  disabledElements.forEach(el => {
    el.removeAttribute('disabled')
  })
}

// Export everything including re-exports from testing-library
export * from '@testing-library/react'
export { default as userEvent } from '@testing-library/user-event'
export { customRender as render }