/**
 * Markdown Parser for PRP Documents
 * 
 * Parses raw markdown content into structured sections that can be rendered
 * by the PRPViewer component with collapsible sections and beautiful formatting.
 */

export interface ParsedSection {
  title: string;
  content: string;
  level: number;
  type: 'text' | 'list' | 'code' | 'mixed';
  rawContent: string;
  sectionKey: string;
  templateType?: string; // For matching to PRP templates
}

export interface ParsedMarkdownDocument {
  title?: string;
  sections: ParsedSection[];
  metadata: Record<string, any>;
  hasMetadata: boolean;
}

export interface ParsedMarkdown {
  title?: string;
  sections: Record<string, ParsedSection>;
  metadata: Record<string, any>;
}

/**
 * Parses markdown content into structured sections based on headers
 */
export function parseMarkdownToPRP(content: string): ParsedMarkdown {
  if (!content || typeof content !== 'string') {
    return { sections: {}, metadata: {} };
  }

  const lines = content.split('\n');
  const sections: Record<string, ParsedSection> = {};
  let currentSection: ParsedSection | null = null;
  let documentTitle: string | undefined;
  let sectionCounter = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Check for headers (## Section Name or # Document Title)
    const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
    
    if (headerMatch) {
      const level = headerMatch[1].length;
      const title = headerMatch[2].trim();
      
      // Save previous section if exists
      if (currentSection) {
        const sectionKey = generateSectionKey(currentSection.title, sectionCounter);
        sections[sectionKey] = {
          ...currentSection,
          content: currentSection.content.trim(),
          rawContent: currentSection.rawContent.trim(),
          type: detectContentType(currentSection.content)
        };
        sectionCounter++;
      }
      
      // Handle document title (# level headers)
      if (level === 1 && !documentTitle) {
        documentTitle = title;
        currentSection = null;
        continue;
      }
      
      // Start new section
      currentSection = {
        title,
        content: '',
        level,
        type: 'text',
        rawContent: ''
      };
    } else if (currentSection) {
      // Add content to current section
      currentSection.content += line + '\n';
      currentSection.rawContent += line + '\n';
    } else if (!documentTitle && line.trim()) {
      // If we haven't found a title yet and encounter content, treat first non-empty line as title
      documentTitle = line.trim();
    }
  }
  
  // Save final section
  if (currentSection) {
    const sectionKey = generateSectionKey(currentSection.title, sectionCounter);
    sections[sectionKey] = {
      ...currentSection,
      content: currentSection.content.trim(),
      rawContent: currentSection.rawContent.trim(),
      type: detectContentType(currentSection.content)
    };
  }

  return {
    title: documentTitle,
    sections,
    metadata: {
      document_type: 'prp',
      parsed_from_markdown: true,
      section_count: Object.keys(sections).length
    }
  };
}

/**
 * Generates a consistent section key for use in the sections object
 */
function generateSectionKey(title: string, counter: number): string {
  // Convert title to a key format
  const baseKey = title
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, '_')
    .substring(0, 30); // Limit length
  
  return baseKey || `section_${counter}`;
}

/**
 * Detects the type of content in a section
 */
function detectContentType(content: string): 'text' | 'list' | 'code' | 'mixed' {
  if (!content.trim()) return 'text';
  
  const lines = content.split('\n').filter(line => line.trim());
  let hasText = false;
  let hasList = false;
  let hasCode = false;
  
  for (const line of lines) {
    if (line.startsWith('```')) {
      hasCode = true;
    } else if (line.match(/^[-*+]\s/) || line.match(/^\d+\.\s/)) {
      hasList = true;
    } else if (line.trim()) {
      hasText = true;
    }
  }
  
  if (hasCode) return 'code';
  if (hasList && hasText) return 'mixed';
  if (hasList) return 'list';
  return 'text';
}

/**
 * Converts parsed markdown back to a structure compatible with PRPViewer
 * Each section becomes a separate collapsible section in the viewer
 */
export function convertParsedMarkdownToPRPStructure(parsed: ParsedMarkdown): any {
  const result: any = {
    title: parsed.title || 'Untitled Document',
    ...parsed.metadata
  };
  
  // Add each section as a top-level property
  // The content will be the raw markdown for that section only
  for (const [key, section] of Object.entries(parsed.sections)) {
    result[key] = section.rawContent;
  }
  
  return result;
}

/**
 * Checks if content appears to be raw markdown
 */
export function isMarkdownContent(content: any): boolean {
  if (typeof content !== 'string') return false;
  
  // Look for markdown indicators
  const markdownIndicators = [
    /^#{1,6}\s+.+$/m,     // Headers
    /^[-*+]\s+.+$/m,      // Bullet lists
    /^\d+\.\s+.+$/m,      // Numbered lists
    /```/,                // Code blocks
    /^\>.+$/m,            // Blockquotes
    /\*\*.+\*\*/,         // Bold text
    /\*.+\*/,             // Italic text
  ];
  
  return markdownIndicators.some(pattern => pattern.test(content));
}

/**
 * Parses markdown content into a flowing document structure
 */
export function parseMarkdownToDocument(content: string): ParsedMarkdownDocument {
  if (!content || typeof content !== 'string') {
    return { sections: [], metadata: {}, hasMetadata: false };
  }

  const lines = content.split('\n');
  const sections: ParsedSection[] = [];
  let currentSection: Partial<ParsedSection> | null = null;
  let documentTitle: string | undefined;
  let sectionCounter = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Check for headers (## Section Name or # Document Title)
    const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
    
    if (headerMatch) {
      const level = headerMatch[1].length;
      const title = headerMatch[2].trim();
      
      // Save previous section if exists
      if (currentSection && currentSection.title) {
        sections.push({
          title: currentSection.title,
          content: (currentSection.content || '').trim(),
          level: currentSection.level || 2,
          type: detectContentType(currentSection.content || ''),
          rawContent: (currentSection.rawContent || '').trim(),
          sectionKey: generateSectionKey(currentSection.title, sectionCounter),
          templateType: detectTemplateType(currentSection.title)
        });
        sectionCounter++;
      }
      
      // Handle document title (# level headers)
      if (level === 1 && !documentTitle) {
        documentTitle = title;
        currentSection = null;
        continue;
      }
      
      // Start new section
      currentSection = {
        title,
        content: '',
        level,
        rawContent: ''
      };
    } else if (currentSection) {
      // Add content to current section
      currentSection.content = (currentSection.content || '') + line + '\n';
      currentSection.rawContent = (currentSection.rawContent || '') + line + '\n';
    } else if (!documentTitle && line.trim()) {
      // If we haven't found a title yet and encounter content, treat first non-empty line as title
      documentTitle = line.trim();
    }
  }
  
  // Save final section
  if (currentSection && currentSection.title) {
    sections.push({
      title: currentSection.title,
      content: (currentSection.content || '').trim(),
      level: currentSection.level || 2,
      type: detectContentType(currentSection.content || ''),
      rawContent: (currentSection.rawContent || '').trim(),
      sectionKey: generateSectionKey(currentSection.title, sectionCounter),
      templateType: detectTemplateType(currentSection.title)
    });
  }

  return {
    title: documentTitle,
    sections,
    metadata: {
      document_type: 'prp', // Set as PRP to get the right styling
      section_count: sections.length,
      parsed_from_markdown: true
    },
    hasMetadata: false
  };
}

/**
 * Detects if a section title matches a known PRP template type
 */
function detectTemplateType(title: string): string | undefined {
  const normalizedTitle = title.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
  
  // Map common PRP section names to template types
  const templateMap: Record<string, string> = {
    'goal': 'context',
    'objective': 'context',
    'purpose': 'context',
    'why': 'context',
    'rationale': 'context',
    'what': 'context',
    'description': 'context',
    'overview': 'context',
    'context': 'context',
    'background': 'context',
    'problem statement': 'context',
    
    'success metrics': 'metrics',
    'metrics': 'metrics',
    'kpis': 'metrics',
    'success criteria': 'metrics',
    'estimated impact': 'metrics',
    
    'implementation plan': 'plan',
    'plan': 'plan',
    'roadmap': 'plan',
    'timeline': 'plan',
    'phases': 'plan',
    'rollout plan': 'plan',
    'migration strategy': 'plan',
    
    'personas': 'personas',
    'users': 'personas',
    'stakeholders': 'personas',
    'target audience': 'personas',
    
    'user flow': 'flows',
    'user journey': 'flows',
    'workflow': 'flows',
    'user experience': 'flows',
    
    'validation': 'list',
    'testing': 'list',
    'quality gates': 'list',
    'acceptance criteria': 'list',
    
    'features': 'features',
    'feature requirements': 'features',
    'capabilities': 'features',
    
    'technical requirements': 'object',
    'architecture': 'object',
    'design': 'object',
    'components': 'object',
    
    'budget': 'keyvalue',
    'resources': 'keyvalue',
    'team': 'keyvalue',
    'cost': 'keyvalue'
  };
  
  return templateMap[normalizedTitle];
}

/**
 * Checks if content is a document with metadata structure
 */
export function isDocumentWithMetadata(content: any): boolean {
  if (typeof content !== 'object' || content === null) return false;
  
  // Check if it has typical document metadata fields
  const metadataFields = ['title', 'version', 'author', 'date', 'status', 'document_type', 'created_at', 'updated_at'];
  const hasMetadata = metadataFields.some(field => field in content);
  
  // Check if it has a content field that looks like markdown
  const hasMarkdownContent = typeof content.content === 'string' && 
                             isMarkdownContent(content.content);
  
  // Also check if any field contains markdown content (broader detection)
  const hasAnyMarkdownField = Object.values(content).some(value => 
    typeof value === 'string' && isMarkdownContent(value)
  );
  
  // Return true if it has metadata AND markdown content, OR if it has obvious document structure
  return (hasMetadata && (hasMarkdownContent || hasAnyMarkdownField)) || 
         (hasMetadata && Object.keys(content).length <= 10); // Simple document structure
}

/**
 * Main function to process content for PRPViewer
 */
export function processContentForPRP(content: any): any {
  // If it's already an object, return as-is
  if (typeof content === 'object' && content !== null) {
    return content;
  }
  
  // If it's a string that looks like markdown, parse it
  if (typeof content === 'string' && isMarkdownContent(content)) {
    const parsed = parseMarkdownToPRP(content);
    return convertParsedMarkdownToPRPStructure(parsed);
  }
  
  // For any other string content, wrap it in a generic structure
  if (typeof content === 'string') {
    return {
      title: 'Document Content',
      content: content,
      document_type: 'text'
    };
  }
  
  return content;
}