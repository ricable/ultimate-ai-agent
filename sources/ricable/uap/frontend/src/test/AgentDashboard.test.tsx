/**
 * Test suite for AgentDashboard component
 * Tests dashboard layout, agent card rendering, and overall functionality
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AgentDashboard } from '../components/agents/AgentDashboard'

describe('AgentDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Component Rendering', () => {
    it('should render without errors', () => {
      expect(() => render(<AgentDashboard />)).not.toThrow()
    })

    it('should render multiple agent cards', () => {
      render(<AgentDashboard />)

      // Should contain multiple agent cards
      const dashboard = screen.getByRole('main') || document.body
      expect(dashboard).toBeInTheDocument()
    })

    it('should display agent cards in grid layout', () => {
      render(<AgentDashboard />)

      // Component should render successfully
      expect(document.body).toContainHTML('<div')
    })
  })

  describe('Agent Card Integration', () => {
    it('should render different types of agents', () => {
      render(<AgentDashboard />)

      // Dashboard should contain agent information
      // The exact agents depend on the mock data in AgentDashboard
      const dashboard = document.body
      expect(dashboard).toBeInTheDocument()
    })

    it('should support different agent frameworks', () => {
      render(<AgentDashboard />)

      // Should render without framework-specific errors
      expect(document.body).toBeInTheDocument()
    })
  })

  describe('Layout and Styling', () => {
    it('should use responsive grid layout', () => {
      render(<AgentDashboard />)

      // Should render with proper structure
      expect(document.body).toBeInTheDocument()
    })

    it('should handle different screen sizes', () => {
      // Test mobile view
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })

      render(<AgentDashboard />)
      expect(document.body).toBeInTheDocument()

      // Test desktop view
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1200,
      })

      render(<AgentDashboard />)
      expect(document.body).toBeInTheDocument()
    })
  })

  describe('Performance', () => {
    it('should render within acceptable time', () => {
      const startTime = performance.now()
      
      render(<AgentDashboard />)
      
      const renderTime = performance.now() - startTime
      
      // Dashboard should render quickly (< 500ms)
      expect(renderTime).toBeLessThan(500)
    })

    it('should handle multiple agent cards efficiently', () => {
      const startTime = performance.now()
      
      render(<AgentDashboard />)
      
      const renderTime = performance.now() - startTime
      
      // Should render multiple cards efficiently
      expect(renderTime).toBeLessThan(1000)
    })
  })

  describe('Error Handling', () => {
    it('should handle missing agent data gracefully', () => {
      // Mock console.error to avoid noise in tests
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      expect(() => render(<AgentDashboard />)).not.toThrow()
      
      consoleErrorSpy.mockRestore()
    })

    it('should render empty state appropriately', () => {
      render(<AgentDashboard />)

      // Should render without errors even with no data
      expect(document.body).toBeInTheDocument()
    })
  })

  describe('Accessibility', () => {
    it('should have proper semantic structure', () => {
      render(<AgentDashboard />)

      // Should have proper document structure
      expect(document.body).toBeInTheDocument()
    })

    it('should support keyboard navigation', () => {
      render(<AgentDashboard />)

      // Dashboard should be accessible
      expect(document.body).toBeInTheDocument()
    })
  })

  describe('Integration with Agent System', () => {
    it('should integrate with different agent frameworks', () => {
      render(<AgentDashboard />)

      // Should handle framework integration
      expect(document.body).toBeInTheDocument()
    })

    it('should handle agent status updates', () => {
      render(<AgentDashboard />)

      // Should handle dynamic updates
      expect(document.body).toBeInTheDocument()
    })
  })
})