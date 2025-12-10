import { render, screen, fireEvent } from '@testing-library/react'
import { describe, test, expect, vi } from 'vitest'
import React from 'react'
import { DocumentCard, NewDocumentCard } from '../../../src/components/project-tasks/DocumentCard'
import type { ProjectDoc } from '../../../src/components/project-tasks/DocumentCard'

describe('DocumentCard', () => {
  const mockDocument: ProjectDoc = {
    id: 'doc-1',
    title: 'Test Document',
    content: { test: 'content' },
    document_type: 'prp',
    updated_at: '2025-07-30T12:00:00Z',
  }

  const mockHandlers = {
    onSelect: vi.fn(),
    onDelete: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders document card with correct content', () => {
    render(
      <DocumentCard
        document={mockDocument}
        isActive={false}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    expect(screen.getByText('Test Document')).toBeInTheDocument()
    expect(screen.getByText('prp')).toBeInTheDocument()
    expect(screen.getByText('7/30/2025')).toBeInTheDocument()
  })

  test('shows correct icon and color for different document types', () => {
    const documentTypes = [
      { type: 'prp', expectedClass: 'text-blue-600' },
      { type: 'technical', expectedClass: 'text-green-600' },
      { type: 'business', expectedClass: 'text-purple-600' },
      { type: 'meeting_notes', expectedClass: 'text-orange-600' },
    ]

    documentTypes.forEach(({ type, expectedClass }) => {
      const { container, rerender } = render(
        <DocumentCard
          document={{ ...mockDocument, document_type: type }}
          isActive={false}
          onSelect={mockHandlers.onSelect}
          onDelete={mockHandlers.onDelete}
          isDarkMode={false}
        />
      )

      const badge = container.querySelector(`.${expectedClass}`)
      expect(badge).toBeInTheDocument()
    })
  })

  test('applies active styles when selected', () => {
    const { container } = render(
      <DocumentCard
        document={mockDocument}
        isActive={true}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    const card = container.firstChild as HTMLElement
    expect(card.className).toContain('border-blue-500')
    expect(card.className).toContain('scale-105')
  })

  test('calls onSelect when clicked', () => {
    render(
      <DocumentCard
        document={mockDocument}
        isActive={false}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    const card = screen.getByText('Test Document').closest('div')
    fireEvent.click(card!)

    expect(mockHandlers.onSelect).toHaveBeenCalledWith(mockDocument)
  })

  test('shows delete button on hover', () => {
    const { container } = render(
      <DocumentCard
        document={mockDocument}
        isActive={false}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    const card = container.firstChild as HTMLElement
    
    // Delete button should not be visible initially
    expect(screen.queryByLabelText('Delete Test Document')).not.toBeInTheDocument()

    // Hover over the card
    fireEvent.mouseEnter(card)
    
    // Delete button should now be visible
    expect(screen.getByLabelText('Delete Test Document')).toBeInTheDocument()

    // Mouse leave
    fireEvent.mouseLeave(card)
    
    // Delete button should be hidden again
    expect(screen.queryByLabelText('Delete Test Document')).not.toBeInTheDocument()
  })

  test('does not show delete button on active card', () => {
    const { container } = render(
      <DocumentCard
        document={mockDocument}
        isActive={true}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    const card = container.firstChild as HTMLElement
    fireEvent.mouseEnter(card)
    
    expect(screen.queryByLabelText('Delete Test Document')).not.toBeInTheDocument()
  })

  test('confirms before deleting', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    const { container } = render(
      <DocumentCard
        document={mockDocument}
        isActive={false}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    const card = container.firstChild as HTMLElement
    fireEvent.mouseEnter(card)
    
    const deleteButton = screen.getByLabelText('Delete Test Document')
    fireEvent.click(deleteButton)

    expect(confirmSpy).toHaveBeenCalledWith('Delete "Test Document"?')
    expect(mockHandlers.onDelete).toHaveBeenCalledWith('doc-1')
    
    confirmSpy.mockRestore()
  })

  test('cancels delete when user declines', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    
    const { container } = render(
      <DocumentCard
        document={mockDocument}
        isActive={false}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={false}
      />
    )

    const card = container.firstChild as HTMLElement
    fireEvent.mouseEnter(card)
    
    const deleteButton = screen.getByLabelText('Delete Test Document')
    fireEvent.click(deleteButton)

    expect(confirmSpy).toHaveBeenCalled()
    expect(mockHandlers.onDelete).not.toHaveBeenCalled()
    
    confirmSpy.mockRestore()
  })

  test('applies dark mode styles correctly', () => {
    const { container } = render(
      <DocumentCard
        document={mockDocument}
        isActive={false}
        onSelect={mockHandlers.onSelect}
        onDelete={mockHandlers.onDelete}
        isDarkMode={true}
      />
    )

    const card = container.firstChild as HTMLElement
    expect(card.className).toContain('dark:')
  })
})

describe('NewDocumentCard', () => {
  test('renders new document card', () => {
    const onClick = vi.fn()
    render(<NewDocumentCard onClick={onClick} />)

    expect(screen.getByText('New Document')).toBeInTheDocument()
  })

  test('calls onClick when clicked', () => {
    const onClick = vi.fn()
    render(<NewDocumentCard onClick={onClick} />)

    const card = screen.getByText('New Document').closest('div')
    fireEvent.click(card!)

    expect(onClick).toHaveBeenCalledTimes(1)
  })
})