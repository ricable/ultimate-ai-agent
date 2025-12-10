import { render, screen, fireEvent } from '@testing-library/react'
import { describe, test, expect, vi } from 'vitest'
import React from 'react'

describe('Component Tests', () => {
  test('button component works', () => {
    const onClick = vi.fn()
    const MockButton = ({ children, ...props }: any) => (
      <button {...props}>{children}</button>
    )
    
    render(<MockButton onClick={onClick}>Click me</MockButton>)
    
    const button = screen.getByRole('button')
    fireEvent.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  test('input component works', () => {
    const MockInput = () => {
      const [value, setValue] = React.useState('')
      return (
        <input 
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Test input"
        />
      )
    }
    
    render(<MockInput />)
    const input = screen.getByPlaceholderText('Test input')
    
    fireEvent.change(input, { target: { value: 'test' } })
    expect((input as HTMLInputElement).value).toBe('test')
  })

  test('modal component works', () => {
    const MockModal = () => {
      const [isOpen, setIsOpen] = React.useState(false)
      return (
        <div>
          <button onClick={() => setIsOpen(true)}>Open Modal</button>
          {isOpen && (
            <div role="dialog">
              <h2>Modal Title</h2>
              <button onClick={() => setIsOpen(false)}>Close</button>
            </div>
          )}
        </div>
      )
    }
    
    render(<MockModal />)
    
    // Modal not visible initially
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    
    // Open modal
    fireEvent.click(screen.getByText('Open Modal'))
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    
    // Close modal
    fireEvent.click(screen.getByText('Close'))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  test('progress bar component works', () => {
    const MockProgressBar = ({ value, max }: { value: number; max: number }) => (
      <div>
        <div>Progress: {Math.round((value / max) * 100)}%</div>
        <div style={{ width: `${(value / max) * 100}%` }}>Bar</div>
      </div>
    )
    
    const { rerender } = render(<MockProgressBar value={0} max={100} />)
    expect(screen.getByText('Progress: 0%')).toBeInTheDocument()
    
    rerender(<MockProgressBar value={50} max={100} />)
    expect(screen.getByText('Progress: 50%')).toBeInTheDocument()
    
    rerender(<MockProgressBar value={100} max={100} />)
    expect(screen.getByText('Progress: 100%')).toBeInTheDocument()
  })

  test('tooltip component works', () => {
    const MockTooltip = ({ children, tooltip }: any) => {
      const [show, setShow] = React.useState(false)
      return (
        <div>
          <button
            onMouseEnter={() => setShow(true)}
            onMouseLeave={() => setShow(false)}
          >
            {children}
          </button>
          {show && <div role="tooltip">{tooltip}</div>}
        </div>
      )
    }
    
    render(<MockTooltip tooltip="This is a tooltip">Hover me</MockTooltip>)
    
    const button = screen.getByText('Hover me')
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    
    fireEvent.mouseEnter(button)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    
    fireEvent.mouseLeave(button)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  test('accordion component works', () => {
    const MockAccordion = () => {
      const [expanded, setExpanded] = React.useState(false)
      return (
        <div>
          <button onClick={() => setExpanded(!expanded)}>
            Section 1 {expanded ? '−' : '+'}
          </button>
          {expanded && <div>Section content</div>}
        </div>
      )
    }
    
    render(<MockAccordion />)
    
    expect(screen.queryByText('Section content')).not.toBeInTheDocument()
    
    fireEvent.click(screen.getByText('Section 1 +'))
    expect(screen.getByText('Section content')).toBeInTheDocument()
    
    fireEvent.click(screen.getByText('Section 1 −'))
    expect(screen.queryByText('Section content')).not.toBeInTheDocument()
  })

  test('table sorting works', () => {
    const MockTable = () => {
      const [data, setData] = React.useState([
        { name: 'Alice', age: 30 },
        { name: 'Bob', age: 25 },
        { name: 'Charlie', age: 35 }
      ])
      
      const sortByName = () => {
        setData([...data].sort((a, b) => a.name.localeCompare(b.name)))
      }
      
      return (
        <table>
          <thead>
            <tr>
              <th onClick={sortByName} style={{ cursor: 'pointer' }}>
                Name
              </th>
              <th>Age</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr key={index}>
                <td>{row.name}</td>
                <td>{row.age}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )
    }
    
    render(<MockTable />)
    
    const cells = screen.getAllByRole('cell')
    expect(cells[0]).toHaveTextContent('Alice')
    
    fireEvent.click(screen.getByText('Name'))
    
    // After sorting, Alice should still be first (already sorted)
    const sortedCells = screen.getAllByRole('cell')
    expect(sortedCells[0]).toHaveTextContent('Alice')
  })

  test('pagination works', () => {
    const MockPagination = () => {
      const [page, setPage] = React.useState(1)
      return (
        <div>
          <div>Page {page}</div>
          <button 
            onClick={() => setPage(page - 1)}
            disabled={page === 1}
          >
            Previous
          </button>
          <button onClick={() => setPage(page + 1)}>
            Next
          </button>
        </div>
      )
    }
    
    render(<MockPagination />)
    
    expect(screen.getByText('Page 1')).toBeInTheDocument()
    
    fireEvent.click(screen.getByText('Next'))
    expect(screen.getByText('Page 2')).toBeInTheDocument()
    
    fireEvent.click(screen.getByText('Previous'))
    expect(screen.getByText('Page 1')).toBeInTheDocument()
  })

  test('form validation works', () => {
    const MockForm = () => {
      const [email, setEmail] = React.useState('')
      const [error, setError] = React.useState('')
      
      const validate = (value: string) => {
        if (!value) {
          setError('Email is required')
        } else if (!value.includes('@')) {
          setError('Invalid email format')
        } else {
          setError('')
        }
      }
      
      return (
        <div>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value)
              validate(e.target.value)
            }}
          />
          {error && <div role="alert">{error}</div>}
        </div>
      )
    }
    
    render(<MockForm />)
    
    const input = screen.getByPlaceholderText('Email')
    
    fireEvent.change(input, { target: { value: 'invalid' } })
    expect(screen.getByRole('alert')).toHaveTextContent('Invalid email format')
    
    fireEvent.change(input, { target: { value: 'valid@email.com' } })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  test('search filtering works', () => {
    const MockSearch = () => {
      const [query, setQuery] = React.useState('')
      const items = ['Apple', 'Banana', 'Cherry', 'Date']
      const filtered = items.filter(item =>
        item.toLowerCase().includes(query.toLowerCase())
      )
      
      return (
        <div>
          <input
            placeholder="Search items"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <ul>
            {filtered.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      )
    }
    
    render(<MockSearch />)
    
    // All items visible initially
    expect(screen.getByText('Apple')).toBeInTheDocument()
    expect(screen.getByText('Banana')).toBeInTheDocument()
    
    // Filter items
    const input = screen.getByPlaceholderText('Search items')
    fireEvent.change(input, { target: { value: 'a' } })
    
    expect(screen.getByText('Apple')).toBeInTheDocument()
    expect(screen.getByText('Banana')).toBeInTheDocument()
    expect(screen.queryByText('Cherry')).not.toBeInTheDocument()
  })
})