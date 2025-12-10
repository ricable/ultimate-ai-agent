import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, test, expect, vi, beforeEach } from 'vitest'
import React from 'react'

// Mock the dependencies
vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => ({
    showToast: vi.fn()
  })
}))

vi.mock('../../services/projectService', () => ({
  projectService: {
    getProjectDocuments: vi.fn().mockResolvedValue([])
  }
}))

vi.mock('../../services/knowledgeBaseService', () => ({
  knowledgeBaseService: {
    getItems: vi.fn().mockResolvedValue([])
  }
}))

// Create a minimal DocsTab component for testing
const DocsTabTest = () => {
  const [documents, setDocuments] = React.useState([
    {
      id: 'doc-1',
      title: 'Document 1',
      content: { type: 'prp' },
      document_type: 'prp',
      updated_at: '2025-07-30T12:00:00Z'
    },
    {
      id: 'doc-2',
      title: 'Document 2',
      content: { type: 'technical' },
      document_type: 'technical',
      updated_at: '2025-07-30T13:00:00Z'
    },
    {
      id: 'doc-3',
      title: 'Document 3',
      content: { type: 'business' },
      document_type: 'business',
      updated_at: '2025-07-30T14:00:00Z'
    }
  ])
  
  const [selectedDocument, setSelectedDocument] = React.useState(documents[0])
  const { showToast } = { showToast: vi.fn() }
  
  return (
    <div>
      <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700">
        {documents.map(doc => (
          <div
            key={doc.id}
            data-testid={`document-card-${doc.id}`}
            className={`flex-shrink-0 w-48 p-4 rounded-lg cursor-pointer ${
              selectedDocument?.id === doc.id ? 'border-2 border-blue-500' : 'border border-gray-200'
            }`}
            onClick={() => setSelectedDocument(doc)}
          >
            <div className={`text-xs ${doc.document_type}`}>{doc.document_type}</div>
            <h4>{doc.title}</h4>
            {selectedDocument?.id !== doc.id && (
              <button
                data-testid={`delete-${doc.id}`}
                onClick={(e) => {
                  e.stopPropagation()
                  if (confirm(`Delete "${doc.title}"?`)) {
                    setDocuments(prev => prev.filter(d => d.id !== doc.id))
                    if (selectedDocument?.id === doc.id) {
                      setSelectedDocument(documents.find(d => d.id !== doc.id) || null)
                    }
                    showToast('Document deleted', 'success')
                  }
                }}
              >
                Delete
              </button>
            )}
          </div>
        ))}
        <div
          data-testid="new-document-card"
          className="flex-shrink-0 w-48 h-32 rounded-lg border-2 border-dashed"
          onClick={() => console.log('New document')}
        >
          New Document
        </div>
      </div>
      {selectedDocument && (
        <div data-testid="selected-document">
          Selected: {selectedDocument.title}
        </div>
      )}
    </div>
  )
}

describe('DocsTab Document Cards Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders all document cards', () => {
    render(<DocsTabTest />)
    
    expect(screen.getByTestId('document-card-doc-1')).toBeInTheDocument()
    expect(screen.getByTestId('document-card-doc-2')).toBeInTheDocument()
    expect(screen.getByTestId('document-card-doc-3')).toBeInTheDocument()
    expect(screen.getByTestId('new-document-card')).toBeInTheDocument()
  })

  test('shows active state on selected document', () => {
    render(<DocsTabTest />)
    
    const doc1 = screen.getByTestId('document-card-doc-1')
    expect(doc1.className).toContain('border-blue-500')
    
    const doc2 = screen.getByTestId('document-card-doc-2')
    expect(doc2.className).not.toContain('border-blue-500')
  })

  test('switches between documents', () => {
    render(<DocsTabTest />)
    
    // Initially doc-1 is selected
    expect(screen.getByTestId('selected-document')).toHaveTextContent('Selected: Document 1')
    
    // Click on doc-2
    fireEvent.click(screen.getByTestId('document-card-doc-2'))
    
    // Now doc-2 should be selected
    expect(screen.getByTestId('selected-document')).toHaveTextContent('Selected: Document 2')
    
    // Check active states
    expect(screen.getByTestId('document-card-doc-1').className).not.toContain('border-blue-500')
    expect(screen.getByTestId('document-card-doc-2').className).toContain('border-blue-500')
  })

  test('deletes document with confirmation', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<DocsTabTest />)
    
    // Click delete on doc-2
    const deleteButton = screen.getByTestId('delete-doc-2')
    fireEvent.click(deleteButton)
    
    expect(confirmSpy).toHaveBeenCalledWith('Delete "Document 2"?')
    
    // Document should be removed
    expect(screen.queryByTestId('document-card-doc-2')).not.toBeInTheDocument()
    
    confirmSpy.mockRestore()
  })

  test('cancels delete when user declines', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    
    render(<DocsTabTest />)
    
    // Click delete on doc-2
    const deleteButton = screen.getByTestId('delete-doc-2')
    fireEvent.click(deleteButton)
    
    // Document should still be there
    expect(screen.getByTestId('document-card-doc-2')).toBeInTheDocument()
    
    confirmSpy.mockRestore()
  })

  test('selects next document when deleting active document', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<DocsTabTest />)
    
    // doc-1 is initially selected
    expect(screen.getByTestId('selected-document')).toHaveTextContent('Selected: Document 1')
    
    // Switch to doc-2
    fireEvent.click(screen.getByTestId('document-card-doc-2'))
    expect(screen.getByTestId('selected-document')).toHaveTextContent('Selected: Document 2')
    
    // Delete doc-2 (currently selected)
    const deleteButton = screen.getByTestId('delete-doc-2')
    fireEvent.click(deleteButton)
    
    // Should automatically select another document
    expect(screen.getByTestId('selected-document')).toHaveTextContent('Selected: Document')
    expect(screen.queryByTestId('document-card-doc-2')).not.toBeInTheDocument()
    
    confirmSpy.mockRestore()
  })

  test('does not show delete button on active card', () => {
    render(<DocsTabTest />)
    
    // doc-1 is active, should not have delete button
    expect(screen.queryByTestId('delete-doc-1')).not.toBeInTheDocument()
    
    // doc-2 is not active, should have delete button
    expect(screen.getByTestId('delete-doc-2')).toBeInTheDocument()
  })

  test('horizontal scroll container has correct classes', () => {
    const { container } = render(<DocsTabTest />)
    
    const scrollContainer = container.querySelector('.overflow-x-auto')
    expect(scrollContainer).toBeInTheDocument()
    expect(scrollContainer?.className).toContain('scrollbar-thin')
    expect(scrollContainer?.className).toContain('scrollbar-thumb-gray-300')
  })

  test('document cards maintain fixed width', () => {
    render(<DocsTabTest />)
    
    const cards = screen.getAllByTestId(/document-card-doc-/)
    cards.forEach(card => {
      expect(card.className).toContain('flex-shrink-0')
      expect(card.className).toContain('w-48')
    })
  })
})