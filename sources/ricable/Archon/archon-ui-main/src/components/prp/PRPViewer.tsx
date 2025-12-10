import React from 'react';
import { PRPContent } from './types/prp.types';
import { MetadataSection } from './sections/MetadataSection';
import { SectionRenderer } from './renderers/SectionRenderer';
import { normalizePRPDocument } from './utils/normalizer';
import { processContentForPRP, isMarkdownContent, isDocumentWithMetadata } from './utils/markdownParser';
import { MarkdownDocumentRenderer } from './components/MarkdownDocumentRenderer';
import './PRPViewer.css';

interface PRPViewerProps {
  content: PRPContent;
  isDarkMode?: boolean;
  sectionOverrides?: Record<string, React.ComponentType<any>>;
}

/**
 * Process content to handle [Image #N] placeholders
 */
const processContent = (content: any): any => {
  if (typeof content === 'string') {
    // Replace [Image #N] with proper markdown image syntax
    return content.replace(/\[Image #(\d+)\]/g, (match, num) => {
      return `![Image ${num}](placeholder-image-${num})`;
    });
  }
  
  if (Array.isArray(content)) {
    return content.map(item => processContent(item));
  }
  
  if (typeof content === 'object' && content !== null) {
    const processed: any = {};
    for (const [key, value] of Object.entries(content)) {
      processed[key] = processContent(value);
    }
    return processed;
  }
  
  return content;
};

/**
 * Flexible PRP Viewer that dynamically renders sections based on content structure
 */
export const PRPViewer: React.FC<PRPViewerProps> = ({ 
  content, 
  isDarkMode = false,
  sectionOverrides = {}
}) => {
  try {
    if (!content) {
      return <div className="text-gray-500">No PRP content available</div>;
    }

  console.log('PRPViewer: Received content:', { 
    type: typeof content, 
    isString: typeof content === 'string',
    isObject: typeof content === 'object',
    hasMetadata: typeof content === 'object' && content !== null ? isDocumentWithMetadata(content) : false,
    isMarkdown: typeof content === 'string' ? isMarkdownContent(content) : false,
    keys: typeof content === 'object' && content !== null ? Object.keys(content) : [],
    contentPreview: typeof content === 'string' ? content.substring(0, 200) + '...' : 'Not a string'
  });

  // Route to appropriate renderer based on content type
  
  // 1. Check if it's a document with metadata + markdown content
  if (isDocumentWithMetadata(content)) {
    console.log('PRPViewer: Detected document with metadata, using MarkdownDocumentRenderer');
    return (
      <MarkdownDocumentRenderer
        content={content}
        isDarkMode={isDarkMode}
        sectionOverrides={sectionOverrides}
      />
    );
  }
  
  // 2. Check if it's a pure markdown string
  if (typeof content === 'string' && isMarkdownContent(content)) {
    console.log('PRPViewer: Detected pure markdown content, using MarkdownDocumentRenderer');
    return (
      <MarkdownDocumentRenderer
        content={content}
        isDarkMode={isDarkMode}
        sectionOverrides={sectionOverrides}
      />
    );
  }

  // 3. Check if it's an object that might contain markdown content in any field
  if (typeof content === 'object' && content !== null) {
    // Check for markdown field first (common in PRP documents)
    if (typeof content.markdown === 'string') {
      console.log('PRPViewer: Found markdown field, using MarkdownDocumentRenderer');
      return (
        <MarkdownDocumentRenderer
          content={content}
          isDarkMode={isDarkMode}
          sectionOverrides={sectionOverrides}
        />
      );
    }
    
    // Look for markdown content in any field
    for (const [key, value] of Object.entries(content)) {
      if (typeof value === 'string' && isMarkdownContent(value)) {
        console.log(`PRPViewer: Found markdown content in field '${key}', using MarkdownDocumentRenderer`);
        // Create a proper document structure
        const documentContent = {
          title: content.title || 'Document',
          content: value,
          ...content // Include all other fields as metadata
        };
        return (
          <MarkdownDocumentRenderer
            content={documentContent}
            isDarkMode={isDarkMode}
            sectionOverrides={sectionOverrides}
          />
        );
      }
    }
  }

  // 4. For any other content that might contain documents, try MarkdownDocumentRenderer first
  console.log('PRPViewer: Checking if content should use MarkdownDocumentRenderer anyway');
  
  // If it's an object with any text content, try MarkdownDocumentRenderer
  if (typeof content === 'object' && content !== null) {
    const hasAnyTextContent = Object.values(content).some(value => 
      typeof value === 'string' && value.length > 50
    );
    
    if (hasAnyTextContent) {
      console.log('PRPViewer: Object has substantial text content, trying MarkdownDocumentRenderer');
      return (
        <MarkdownDocumentRenderer
          content={content}
          isDarkMode={isDarkMode}
          sectionOverrides={sectionOverrides}
        />
      );
    }
  }

  // 5. Final fallback to original PRPViewer logic for purely structured JSON content
  console.log('PRPViewer: Using standard JSON structure renderer as final fallback');
  
  // First, check if content is raw markdown and process it
  let processedForPRP = content;
  
  // Handle the case where content is a raw markdown string (non-markdown strings)
  if (typeof content === 'string') {
    // For non-markdown strings, wrap in a simple structure
    processedForPRP = {
      title: 'Document Content',
      content: content,
      document_type: 'text'
    };
  } else if (typeof content === 'object' && content !== null) {
    // For objects, process normally
    processedForPRP = processContentForPRP(content);
  }

  // Ensure we have an object to work with
  if (!processedForPRP || typeof processedForPRP !== 'object') {
    return <div className="text-gray-500">Unable to process PRP content</div>;
  }

  // Normalize the content 
  const normalizedContent = normalizePRPDocument(processedForPRP);
  
  // Process content to handle [Image #N] placeholders
  const processedContent = processContent(normalizedContent);

  // Extract sections (skip metadata fields)
  const metadataFields = ['title', 'version', 'author', 'date', 'status', 'document_type', 'id', '_id', 'project_id', 'created_at', 'updated_at'];
  const sections = Object.entries(processedContent).filter(([key]) => !metadataFields.includes(key));
  
  // Debug: Log sections being rendered
  console.log('PRP Sections found:', sections.map(([key]) => key));
  
  // Priority-based sorting for common PRP sections
  const getSectionPriority = (key: string): number => {
    const normalizedKey = key.toLowerCase();
    
    // Define priority order (lower number = higher priority)
    if (normalizedKey.includes('goal') || normalizedKey.includes('objective')) return 1;
    if (normalizedKey.includes('why') || normalizedKey.includes('rationale')) return 2;
    if (normalizedKey.includes('what') || normalizedKey === 'description') return 3;
    if (normalizedKey.includes('context') || normalizedKey.includes('background')) return 4;
    if (normalizedKey.includes('persona') || normalizedKey.includes('user') || normalizedKey.includes('stakeholder')) return 5;
    if (normalizedKey.includes('flow') || normalizedKey.includes('journey') || normalizedKey.includes('workflow')) return 6;
    if (normalizedKey.includes('requirement') && !normalizedKey.includes('technical')) return 7;
    if (normalizedKey.includes('metric') || normalizedKey.includes('success') || normalizedKey.includes('kpi')) return 8;
    if (normalizedKey.includes('timeline') || normalizedKey.includes('roadmap') || normalizedKey.includes('milestone')) return 9;
    if (normalizedKey.includes('plan') || normalizedKey.includes('implementation')) return 10;
    if (normalizedKey.includes('technical') || normalizedKey.includes('architecture') || normalizedKey.includes('tech')) return 11;
    if (normalizedKey.includes('validation') || normalizedKey.includes('testing') || normalizedKey.includes('quality')) return 12;
    if (normalizedKey.includes('risk') || normalizedKey.includes('mitigation')) return 13;
    
    // Default priority for unknown sections
    return 50;
  };
  
  // Sort sections by priority
  const sortedSections = sections.sort(([a], [b]) => {
    return getSectionPriority(a) - getSectionPriority(b);
  });

  return (
    <div className={`prp-viewer ${isDarkMode ? 'dark' : ''}`}>
      {/* Metadata Header */}
      <MetadataSection content={processedContent} isDarkMode={isDarkMode} />

      {/* Dynamic Sections */}
      {sortedSections.map(([sectionKey, sectionData], index) => (
        <div key={sectionKey} className="mb-6">
          <SectionRenderer
            sectionKey={sectionKey}
            data={sectionData}
            index={index}
            isDarkMode={isDarkMode}
            sectionOverrides={sectionOverrides}
          />
        </div>
      ))}
      
      {sections.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No additional sections found in this PRP document.</p>
        </div>
      )}
    </div>
  );
  } catch (error) {
    console.error('PRPViewer: Error rendering content:', error);
    
    // Provide a meaningful error display instead of black screen
    return (
      <div className="p-6 bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-800 rounded-lg">
        <h3 className="text-red-800 dark:text-red-200 font-semibold mb-2">Error Rendering PRP</h3>
        <p className="text-red-600 dark:text-red-300 text-sm mb-4">
          There was an error rendering this PRP document. The content may be in an unexpected format.
        </p>
        
        {/* Show error details for debugging */}
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-red-600 dark:text-red-400 hover:underline">
            Show error details
          </summary>
          <div className="mt-2 space-y-2">
            <pre className="p-4 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-auto">
              {error instanceof Error ? error.message : String(error)}
            </pre>
            {error instanceof Error && error.stack && (
              <pre className="p-4 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-auto max-h-48">
                {error.stack}
              </pre>
            )}
          </div>
        </details>
        
        {/* Show raw content for debugging */}
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-red-600 dark:text-red-400 hover:underline">
            Show raw content
          </summary>
          <pre className="mt-2 p-4 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-auto max-h-96">
            {typeof content === 'string' 
              ? content 
              : JSON.stringify(content, null, 2)}
          </pre>
        </details>
      </div>
    );
  }
};