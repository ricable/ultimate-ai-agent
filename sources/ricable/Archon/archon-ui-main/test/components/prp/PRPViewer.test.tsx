import { render, screen, fireEvent } from '@testing-library/react'
import { describe, test, expect, vi } from 'vitest'
import React from 'react'
import { PRPViewer } from '../../../src/components/prp/PRPViewer'
import type { PRPContent } from '../../../src/components/prp/types/prp.types'

describe('PRPViewer', () => {
  const mockContent: PRPContent = {
    title: 'Test PRP',
    version: '1.0',
    author: 'Test Author',
    date: '2025-07-30',
    status: 'draft',
    goal: 'Test goal with [Image #1] placeholder',
    why: 'Test reason with [Image #2] reference',
    what: {
      description: 'Test description with [Image #3] and [Image #4]',
      success_criteria: ['Criterion 1', 'Criterion 2 with [Image #5]']
    },
    context: {
      background: 'Background with [Image #6]',
      objectives: ['Objective 1', 'Objective 2']
    }
  }

  test('renders without [Image #N] placeholders', () => {
    render(<PRPViewer content={mockContent} />)

    // Check that [Image #N] placeholders are replaced
    expect(screen.queryByText(/\[Image #\d+\]/)).not.toBeInTheDocument()
    
    // Check that content is present
    expect(screen.getByText(/Test goal/)).toBeInTheDocument()
    expect(screen.getByText(/Test reason/)).toBeInTheDocument()
    expect(screen.getByText(/Test description/)).toBeInTheDocument()
  })

  test('processes nested content with image placeholders', () => {
    const { container } = render(<PRPViewer content={mockContent} />)
    
    // Check that the content has been processed
    const htmlContent = container.innerHTML
    
    // Should not contain raw [Image #N] text
    expect(htmlContent).not.toMatch(/\[Image #\d+\]/)
    
    // Should contain processed markdown image syntax
    expect(htmlContent).toContain('Image 1')
    expect(htmlContent).toContain('Image 2')
  })

  test('renders metadata section correctly', () => {
    render(<PRPViewer content={mockContent} />)

    expect(screen.getByText('Test PRP')).toBeInTheDocument()
    expect(screen.getByText('1.0')).toBeInTheDocument()
    expect(screen.getByText('Test Author')).toBeInTheDocument()
    expect(screen.getByText('draft')).toBeInTheDocument()
  })

  test('handles empty content gracefully', () => {
    render(<PRPViewer content={{} as PRPContent} />)
    
    // Should render without errors
    expect(screen.getByText(/Metadata/)).toBeInTheDocument()
  })

  test('handles null content', () => {
    render(<PRPViewer content={null as any} />)
    
    expect(screen.getByText('No PRP content available')).toBeInTheDocument()
  })

  test('handles string content in objects', () => {
    const stringContent = {
      title: 'String Test',
      description: 'This has [Image #1] in it'
    }
    
    render(<PRPViewer content={stringContent as any} />)
    
    // Should process the image placeholder
    expect(screen.queryByText(/\[Image #1\]/)).not.toBeInTheDocument()
    expect(screen.getByText(/This has/)).toBeInTheDocument()
  })

  test('handles array content with image placeholders', () => {
    const arrayContent = {
      title: 'Array Test',
      items: [
        'Item 1 with [Image #1]',
        'Item 2 with [Image #2]',
        { nested: 'Nested with [Image #3]' }
      ]
    }
    
    render(<PRPViewer content={arrayContent as any} />)
    
    // Should process all image placeholders
    expect(screen.queryByText(/\[Image #\d+\]/)).not.toBeInTheDocument()
  })

  test('renders collapsible sections', () => {
    render(<PRPViewer content={mockContent} />)
    
    // Find collapsible sections
    const contextSection = screen.getByText('Context').closest('div')
    expect(contextSection).toBeInTheDocument()
    
    // Should have chevron icon for collapsible sections
    const chevrons = screen.getAllByTestId('chevron-icon')
    expect(chevrons.length).toBeGreaterThan(0)
  })

  test('toggles section visibility', () => {
    render(<PRPViewer content={mockContent} />)
    
    // Find a collapsible section header
    const contextHeader = screen.getByText('Context').closest('button')
    
    // The section should be visible initially (defaultOpen for first 5 sections)
    expect(screen.getByText(/Background with/)).toBeInTheDocument()
    
    // Click to collapse
    fireEvent.click(contextHeader!)
    
    // Content should be hidden
    expect(screen.queryByText(/Background with/)).not.toBeInTheDocument()
    
    // Click to expand
    fireEvent.click(contextHeader!)
    
    // Content should be visible again
    expect(screen.getByText(/Background with/)).toBeInTheDocument()
  })

  test('applies dark mode styles', () => {
    const { container } = render(<PRPViewer content={mockContent} isDarkMode={true} />)
    
    const viewer = container.querySelector('.prp-viewer')
    expect(viewer?.className).toContain('dark')
  })

  test('uses section overrides when provided', () => {
    const CustomSection = ({ data, title }: any) => (
      <div data-testid="custom-section">
        <h3>{title}</h3>
        <p>Custom rendering of: {JSON.stringify(data)}</p>
      </div>
    )
    
    const overrides = {
      context: CustomSection
    }
    
    render(<PRPViewer content={mockContent} sectionOverrides={overrides} />)
    
    expect(screen.getByTestId('custom-section')).toBeInTheDocument()
    expect(screen.getByText(/Custom rendering of/)).toBeInTheDocument()
  })

  test('sorts sections by group', () => {
    const complexContent = {
      title: 'Complex PRP',
      // These should be sorted in a specific order
      validation_gates: { test: 'validation' },
      user_personas: { test: 'personas' },
      context: { test: 'context' },
      user_flows: { test: 'flows' },
      success_metrics: { test: 'metrics' }
    }
    
    const { container } = render(<PRPViewer content={complexContent as any} />)
    
    // Get all section titles in order
    const sectionTitles = Array.from(
      container.querySelectorAll('h3')
    ).map(el => el.textContent)
    
    // Context should come before personas
    const contextIndex = sectionTitles.findIndex(t => t?.includes('Context'))
    const personasIndex = sectionTitles.findIndex(t => t?.includes('Personas'))
    
    expect(contextIndex).toBeLessThan(personasIndex)
  })
})