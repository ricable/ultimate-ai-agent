import { describe, test, expect } from 'vitest'

// Test the PRP to Markdown conversion logic
describe('MilkdownEditor PRP Conversion', () => {
  // Helper function to format values (extracted from component)
  const formatValue = (value: any, indent = ''): string => {
    if (Array.isArray(value)) {
      return value.map(item => `${indent}- ${formatValue(item, indent + '  ')}`).join('\n') + '\n'
    }
    
    if (typeof value === 'object' && value !== null) {
      let result = ''
      Object.entries(value).forEach(([key, val]) => {
        const formattedKey = key.replace(/_/g, ' ')
          .split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ')
        
        if (typeof val === 'string' || typeof val === 'number') {
          result += `${indent}**${formattedKey}:** ${val}\n\n`
        } else {
          result += `${indent}### ${formattedKey}\n\n${formatValue(val, indent)}`
        }
      })
      return result
    }
    
    return String(value)
  }

  // Simplified version of convertPRPToMarkdown for testing
  const convertPRPToMarkdown = (content: any, docTitle = 'Test Doc'): string => {
    let markdown = `# ${content.title || docTitle}\n\n`
    
    // Metadata section
    if (content.version || content.author || content.date || content.status) {
      markdown += `## Metadata\n\n`
      if (content.version) markdown += `- **Version:** ${content.version}\n`
      if (content.author) markdown += `- **Author:** ${content.author}\n`
      if (content.date) markdown += `- **Date:** ${content.date}\n`
      if (content.status) markdown += `- **Status:** ${content.status}\n`
      markdown += '\n'
    }
    
    // Goal section
    if (content.goal) {
      markdown += `## Goal\n\n${content.goal}\n\n`
    }
    
    // Why section
    if (content.why) {
      markdown += `## Why\n\n`
      if (Array.isArray(content.why)) {
        content.why.forEach(item => markdown += `- ${item}\n`)
      } else {
        markdown += `${content.why}\n`
      }
      markdown += '\n'
    }
    
    // What section
    if (content.what) {
      markdown += `## What\n\n`
      if (typeof content.what === 'string') {
        markdown += `${content.what}\n\n`
      } else if (content.what.description) {
        markdown += `${content.what.description}\n\n`
        
        if (content.what.success_criteria) {
          markdown += `### Success Criteria\n\n`
          content.what.success_criteria.forEach((criterion: string) => {
            markdown += `- [ ] ${criterion}\n`
          })
          markdown += '\n'
        }
      }
    }
    
    // Handle all other sections dynamically
    const handledKeys = [
      'title', 'version', 'author', 'date', 'status', 'goal', 'why', 'what', 
      'document_type'
    ]
    
    Object.entries(content).forEach(([key, value]) => {
      if (!handledKeys.includes(key) && value) {
        const sectionTitle = key.replace(/_/g, ' ')
          .split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ')
        
        markdown += `## ${sectionTitle}\n\n`
        markdown += formatValue(value)
        markdown += '\n'
      }
    })
    
    return markdown
  }

  test('converts basic PRP structure to markdown', () => {
    const prp = {
      title: 'Test PRP',
      version: '1.0',
      author: 'Test Author',
      date: '2025-07-30',
      status: 'draft',
      goal: 'Test goal'
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('# Test PRP')
    expect(markdown).toContain('## Metadata')
    expect(markdown).toContain('- **Version:** 1.0')
    expect(markdown).toContain('- **Author:** Test Author')
    expect(markdown).toContain('- **Date:** 2025-07-30')
    expect(markdown).toContain('- **Status:** draft')
    expect(markdown).toContain('## Goal\n\nTest goal')
  })

  test('handles array why section', () => {
    const prp = {
      title: 'Test PRP',
      why: ['Reason 1', 'Reason 2', 'Reason 3']
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('## Why')
    expect(markdown).toContain('- Reason 1')
    expect(markdown).toContain('- Reason 2')
    expect(markdown).toContain('- Reason 3')
  })

  test('handles string why section', () => {
    const prp = {
      title: 'Test PRP',
      why: 'Single reason for the change'
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('## Why')
    expect(markdown).toContain('Single reason for the change')
  })

  test('handles complex what section with success criteria', () => {
    const prp = {
      title: 'Test PRP',
      what: {
        description: 'Main description of what we are building',
        success_criteria: [
          'Criterion 1',
          'Criterion 2',
          'Criterion 3'
        ]
      }
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('## What')
    expect(markdown).toContain('Main description of what we are building')
    expect(markdown).toContain('### Success Criteria')
    expect(markdown).toContain('- [ ] Criterion 1')
    expect(markdown).toContain('- [ ] Criterion 2')
    expect(markdown).toContain('- [ ] Criterion 3')
  })

  test('handles dynamic sections', () => {
    const prp = {
      title: 'Test PRP',
      user_personas: {
        developer: {
          name: 'Developer Dan',
          goals: ['Write clean code', 'Ship features fast']
        }
      },
      technical_requirements: {
        frontend: 'React 18',
        backend: 'FastAPI',
        database: 'PostgreSQL'
      }
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('## User Personas')
    expect(markdown).toContain('### Developer')
    expect(markdown).toContain('**Name:** Developer Dan')
    expect(markdown).toContain('## Technical Requirements')
    expect(markdown).toContain('**Frontend:** React 18')
    expect(markdown).toContain('**Backend:** FastAPI')
  })

  test('formats nested objects correctly', () => {
    const value = {
      level1: {
        level2: {
          level3: 'Deep value'
        }
      }
    }
    
    const formatted = formatValue(value)
    
    expect(formatted).toContain('### Level1')
    expect(formatted).toContain('### Level2')
    expect(formatted).toContain('**Level3:** Deep value')
  })

  test('formats arrays correctly', () => {
    const value = ['Item 1', 'Item 2', { nested: 'Nested item' }]
    
    const formatted = formatValue(value)
    
    expect(formatted).toContain('- Item 1')
    expect(formatted).toContain('- Item 2')
    expect(formatted).toContain('**Nested:** Nested item')
  })

  test('handles empty content', () => {
    const prp = {}
    
    const markdown = convertPRPToMarkdown(prp, 'Default Title')
    
    expect(markdown).toBe('# Default Title\n\n')
  })

  test('skips null and undefined values', () => {
    const prp = {
      title: 'Test PRP',
      null_field: null,
      undefined_field: undefined,
      empty_string: '',
      valid_field: 'Valid content'
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).not.toContain('Null Field')
    expect(markdown).not.toContain('Undefined Field')
    expect(markdown).not.toContain('Empty String')
    expect(markdown).toContain('## Valid Field')
    expect(markdown).toContain('Valid content')
  })

  test('converts snake_case to Title Case', () => {
    const prp = {
      title: 'Test PRP',
      user_journey_mapping: 'Content',
      api_endpoint_design: 'More content'
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('## User Journey Mapping')
    expect(markdown).toContain('## Api Endpoint Design')
  })

  test('preserves markdown formatting in content', () => {
    const prp = {
      title: 'Test PRP',
      description: '**Bold text** and *italic text* with `code`'
    }
    
    const markdown = convertPRPToMarkdown(prp)
    
    expect(markdown).toContain('**Bold text** and *italic text* with `code`')
  })
})