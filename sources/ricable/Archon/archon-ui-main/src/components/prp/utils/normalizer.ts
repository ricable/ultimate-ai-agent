/**
 * Normalizes PRP document data to ensure consistent rendering
 */

/**
 * Normalizes image placeholders to proper markdown format
 */
export function normalizeImagePlaceholders(content: string): string {
  return content.replace(/\[Image #(\d+)\]/g, (match, num) => {
    return `![Image ${num}](placeholder-image-${num})`;
  });
}

/**
 * Attempts to parse JSON strings into objects
 */
export function parseJsonStrings(value: any): any {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || 
        (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        return JSON.parse(trimmed);
      } catch (e) {
        // Return original string if parsing fails
        return value;
      }
    }
    
    // Normalize image placeholders in strings
    return normalizeImagePlaceholders(value);
  }
  
  if (Array.isArray(value)) {
    return value.map(item => parseJsonStrings(item));
  }
  
  if (value && typeof value === 'object') {
    const normalized: any = {};
    for (const [key, val] of Object.entries(value)) {
      normalized[key] = parseJsonStrings(val);
    }
    return normalized;
  }
  
  return value;
}

/**
 * Flattens nested content fields
 */
export function flattenNestedContent(data: any): any {
  // Handle nested content field
  if (data && typeof data === 'object' && 'content' in data) {
    const { content, ...rest } = data;
    
    // If content is an object, merge it with the rest
    if (content && typeof content === 'object' && !Array.isArray(content)) {
      return flattenNestedContent({ ...rest, ...content });
    }
    
    // If content is a string or array, keep it as a field
    return { ...rest, content };
  }
  
  return data;
}

/**
 * Normalizes section names to be more readable
 */
export function normalizeSectionName(name: string): string {
  // Common abbreviations and their expansions
  const expansions: Record<string, string> = {
    'ui': 'User Interface',
    'ux': 'User Experience',
    'api': 'API',
    'kpi': 'KPI',
    'prp': 'PRP',
    'prd': 'PRD',
    'mvp': 'MVP',
    'poc': 'Proof of Concept',
  };
  
  // Split by underscore or camelCase
  const words = name
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(' ')
    .filter(word => word.length > 0);
  
  // Process each word
  const processed = words.map(word => {
    const lower = word.toLowerCase();
    
    // Check if it's a known abbreviation
    if (expansions[lower]) {
      return expansions[lower];
    }
    
    // Otherwise, capitalize first letter
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
  });
  
  return processed.join(' ');
}

/**
 * Normalizes the entire PRP document structure
 */
export function normalizePRPDocument(content: any): any {
  if (!content) return content;
  
  // First, flatten any nested content fields
  let normalized = flattenNestedContent(content);
  
  // Then parse any JSON strings
  normalized = parseJsonStrings(normalized);
  
  // Handle raw markdown content
  if (typeof normalized === 'string') {
    // For strings, just normalize image placeholders and return as-is
    // The PRPViewer will handle the markdown parsing
    return normalizeImagePlaceholders(normalized);
  }
  
  // For objects, process each field recursively
  if (normalized && typeof normalized === 'object' && !Array.isArray(normalized)) {
    const result: any = {};
    
    for (const [key, value] of Object.entries(normalized)) {
      // Skip empty values
      if (value === null || value === undefined || 
          (typeof value === 'string' && value.trim() === '') ||
          (Array.isArray(value) && value.length === 0) ||
          (typeof value === 'object' && Object.keys(value).length === 0)) {
        continue;
      }
      
      // Recursively process nested values
      if (typeof value === 'string') {
        result[key] = normalizeImagePlaceholders(value);
      } else if (Array.isArray(value)) {
        result[key] = value.map(item => 
          typeof item === 'string' ? normalizeImagePlaceholders(item) : normalizePRPDocument(item)
        );
      } else if (typeof value === 'object') {
        result[key] = normalizePRPDocument(value);
      } else {
        result[key] = value;
      }
    }
    
    return result;
  }
  
  // For arrays, process each item
  if (Array.isArray(normalized)) {
    return normalized.map(item => 
      typeof item === 'string' ? normalizeImagePlaceholders(item) : normalizePRPDocument(item)
    );
  }
  
  return normalized;
}

/**
 * Checks if a value contains complex nested structures
 */
export function hasComplexNesting(value: any): boolean {
  if (!value || typeof value !== 'object') return false;
  
  if (Array.isArray(value)) {
    return value.some(item => 
      typeof item === 'object' && item !== null
    );
  }
  
  return Object.values(value).some(val => 
    (typeof val === 'object' && val !== null) ||
    (Array.isArray(val) && val.some(item => typeof item === 'object'))
  );
}

/**
 * Extracts metadata fields from content
 */
export function extractMetadata(content: any): { metadata: any; sections: any } {
  if (!content || typeof content !== 'object') {
    return { metadata: {}, sections: content };
  }
  
  const metadataFields = [
    'title', 'version', 'author', 'date', 'status', 
    'document_type', 'created_at', 'updated_at', 
    'id', '_id', 'project_id'
  ];
  
  const metadata: any = {};
  const sections: any = {};
  
  for (const [key, value] of Object.entries(content)) {
    if (metadataFields.includes(key)) {
      metadata[key] = value;
    } else {
      sections[key] = value;
    }
  }
  
  return { metadata, sections };
}