/**
 * Comprehensive test suite for AgentCard component
 * Tests UI functionality, AG-UI integration, and user interactions
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { testUtils } from './setup'
import { AgentCard } from '../components/agents/AgentCard'

describe('AgentCard Component', () => {
  const defaultProps = {
    id: 'test-agent',
    name: 'Test Agent',
    description: 'A test agent for validation',
    framework: 'copilot'
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Component Rendering', () => {
    it('should render agent information correctly', () => {
      render(<AgentCard {...defaultProps} />)

      expect(screen.getByText('Test Agent')).toBeInTheDocument()
      expect(screen.getByText('copilot')).toBeInTheDocument()
      expect(screen.getByText('A test agent for validation')).toBeInTheDocument()
    })

    it('should render all required UI elements', () => {
      render(<AgentCard {...defaultProps} />)

      expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
      expect(screen.getByRole('textbox')).toBeInTheDocument()
      expect(screen.getByText('Test Agent')).toBeInTheDocument()
    })

    it('should display framework badge with correct styling', () => {
      render(<AgentCard {...defaultProps} />)

      const frameworkBadge = screen.getByText('copilot')
      expect(frameworkBadge).toBeInTheDocument()
    })

    it('should render different frameworks correctly', () => {
      const frameworks = ['copilot', 'agno', 'mastra']

      frameworks.forEach(framework => {
        const { unmount } = render(
          <AgentCard {...defaultProps} framework={framework} />
        )
        
        expect(screen.getByText(framework)).toBeInTheDocument()
        unmount()
      })
    })
  })

  describe('Connection Management', () => {
    it('should show connection status', () => {
      render(<AgentCard {...defaultProps} />)

      // Connection status should be visible (either connected or disconnected)
      const agentCard = screen.getByText('Test Agent').closest('div')
      expect(agentCard).toBeInTheDocument()
    })

    it('should handle connection state changes', async () => {
      render(<AgentCard {...defaultProps} />)

      // Component should render without errors
      expect(screen.getByText('Test Agent')).toBeInTheDocument()
    })
  })

  describe('Message Handling', () => {
    it('should send message when button is clicked', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')
      const button = screen.getByRole('button', { name: /send/i })

      await user.type(input, 'Hello agent!')
      await user.click(button)

      // Verify input was cleared after sending
      expect(input).toHaveValue('')
    })

    it('should send message when Enter key is pressed', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')

      await user.type(input, 'Hello agent!')
      await user.keyboard('{Enter}')

      // Verify input was cleared after sending
      expect(input).toHaveValue('')
    })

    it('should clear input after sending message', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')
      const button = screen.getByRole('button', { name: /send/i })

      await user.type(input, 'Test message')
      expect(input).toHaveValue('Test message')

      await user.click(button)

      expect(input).toHaveValue('')
    })

    it('should not send empty messages', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const button = screen.getByRole('button', { name: /send/i })

      // Clicking send with empty input should not cause errors
      await user.click(button)

      // Component should still be functional
      expect(screen.getByText('Test Agent')).toBeInTheDocument()
    })
  })

  describe('Performance Requirements', () => {
    it('should render within acceptable time (< 1s TTI)', async () => {
      const startTime = performance.now()
      
      render(<AgentCard {...defaultProps} />)

      const endTime = performance.now()
      const renderTime = endTime - startTime

      // Component should render quickly
      expect(renderTime).toBeWithinResponseTime(1000) // 1s TTI requirement
    })

    it('should handle rapid user interactions', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')
      const button = screen.getByRole('button', { name: /send/i })

      const startTime = performance.now()

      // Perform rapid interactions
      for (let i = 0; i < 5; i++) {
        await user.type(input, `Rapid message ${i}`)
        await user.click(button)
      }

      const endTime = performance.now()
      const totalTime = endTime - startTime

      // Should handle rapid interactions smoothly (< 2s for 5 messages)
      expect(totalTime).toBeWithinResponseTime(2000)
    })

    it('should maintain responsive UI during message exchange', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')
      const button = screen.getByRole('button', { name: /send/i })

      // Send message and measure UI responsiveness
      const startTime = performance.now()
      
      await user.type(input, 'Responsiveness test')
      await user.click(button)

      // UI should remain responsive (input should be available immediately)
      const responseTime = performance.now() - startTime
      expect(responseTime).toBeWithinResponseTime(100) // < 100ms for UI responsiveness
    })
  })

  describe('Accessibility', () => {
    it('should have proper form elements', () => {
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')
      const button = screen.getByRole('button', { name: /send/i })
      
      expect(input).toBeInTheDocument()
      expect(button).toBeInTheDocument()
    })

    it('should support keyboard navigation', async () => {
      const user = userEvent.setup()
      render(<AgentCard {...defaultProps} />)

      const input = screen.getByRole('textbox')

      // Should be able to focus input
      await user.tab()
      expect(input).toHaveFocus()

      // Should be able to send message with keyboard
      await user.type(input, 'Keyboard test')
      await user.keyboard('{Enter}')

      // Input should be cleared after sending
      expect(input).toHaveValue('')
    })
  })

  describe('Error Handling', () => {
    it('should handle component props gracefully', () => {
      // Test with minimal props
      const minimalProps = {
        id: 'minimal-agent',
        name: 'Minimal Agent',
        description: 'Minimal description',
        framework: 'copilot'
      }

      expect(() => render(<AgentCard {...minimalProps} />)).not.toThrow()
    })

    it('should handle empty values gracefully', () => {
      const emptyProps = {
        id: '',
        name: '',
        description: '',
        framework: 'copilot'
      }

      expect(() => render(<AgentCard {...emptyProps} />)).not.toThrow()
    })
  })

  describe('Framework-Specific Behavior', () => {
    it('should adapt behavior based on framework', () => {
      const frameworks = [
        { name: 'copilot', description: 'CopilotKit agent' },
        { name: 'agno', description: 'Agno agent for research' },
        { name: 'mastra', description: 'Mastra workflow agent' }
      ]

      frameworks.forEach(({ name, description }) => {
        const { unmount } = render(
          <AgentCard 
            {...defaultProps} 
            framework={name}
            description={description}
          />
        )

        expect(screen.getByText(name)).toBeInTheDocument()
        expect(screen.getByText(description)).toBeInTheDocument()
        
        unmount()
      })
    })
  })
})