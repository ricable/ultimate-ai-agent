import React, { useEffect, useRef, useState } from 'react';
import { Crepe, CrepeFeature } from '@milkdown/crepe';
import '@milkdown/crepe/theme/common/style.css';
import '@milkdown/crepe/theme/frame.css';
import '@milkdown/crepe/theme/frame-dark.css';
import './MilkdownEditor.css';
import { Save, Undo } from 'lucide-react';

interface MilkdownEditorProps {
  document: {
    id: string;
    title: string;
    content?: any;
    created_at: string;
    updated_at: string;
  };
  onSave: (document: any) => void;
  className?: string;
  isDarkMode?: boolean;
}

export const MilkdownEditor: React.FC<MilkdownEditorProps> = ({
  document: doc,
  onSave,
  className = '',
  isDarkMode = false,
}) => {
  const editorRef = useRef<HTMLDivElement>(null);
  const crepeRef = useRef<Crepe | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [isReverted, setIsReverted] = useState(false);
  const [originalContent, setOriginalContent] = useState<string>('');
  const [currentContent, setCurrentContent] = useState<string>('');

  // Convert document content to markdown string
  const getMarkdownContent = () => {
    if (typeof doc.content === 'string') {
      return doc.content;
    }
    
    if (doc.content && typeof doc.content === 'object') {
      // If content has a markdown field, use it
      if (doc.content.markdown) {
        return doc.content.markdown;
      }
      
      // Check if this is a PRP document
      if (doc.content.document_type === 'prp' || doc.document_type === 'prp') {
        return convertPRPToMarkdown(doc.content);
      }
      
      // Otherwise, convert the content object to a readable markdown format
      let markdown = `# ${doc.title}\n\n`;
      
      Object.entries(doc.content).forEach(([key, value]) => {
        const sectionTitle = key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1);
        markdown += `## ${sectionTitle}\n\n`;
        
        if (Array.isArray(value)) {
          value.forEach(item => {
            markdown += `- ${item}\n`;
          });
          markdown += '\n';
        } else if (typeof value === 'object' && value !== null) {
          if (value.description) {
            markdown += `${value.description}\n\n`;
          } else {
            Object.entries(value).forEach(([subKey, subValue]) => {
              markdown += `**${subKey}:** ${subValue}\n\n`;
            });
          }
        } else {
          markdown += `${value}\n\n`;
        }
      });
      
      return markdown;
    }
    
    return `# ${doc.title}\n\nStart writing...`;
  };

  // Helper function to format values for markdown
  // Enhanced formatValue to handle complex nested structures
  const formatValue = (value: any, indent = '', depth = 0): string => {
    if (value === null || value === undefined) {
      return '';
    }
    
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
    
    if (Array.isArray(value)) {
      if (value.length === 0) return '';
      
      // Check if it's a simple array (strings/numbers)
      const isSimple = value.every(item => 
        typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean'
      );
      
      if (isSimple) {
        return value.map(item => `${indent}- ${item}`).join('\n') + '\n';
      }
      
      // Complex array with objects
      return value.map((item, index) => {
        if (typeof item === 'object' && item !== null) {
          const itemLines = formatValue(item, indent + '  ', depth + 1).split('\n');
          const firstLine = itemLines[0];
          const restLines = itemLines.slice(1).join('\n');
          
          if (itemLines.length === 1 || (itemLines.length === 2 && !itemLines[1])) {
            // Single line item
            return `${indent}- ${firstLine}`;
          } else {
            // Multi-line item
            return `${indent}-\n${indent}  ${firstLine}${restLines ? '\n' + restLines : ''}`;
          }
        }
        return `${indent}- ${formatValue(item, indent + '  ', depth + 1)}`;
      }).join('\n') + '\n';
    }
    
    if (typeof value === 'object' && value !== null) {
      const entries = Object.entries(value);
      if (entries.length === 0) return '';
      
      // Check if it's a simple object (all values are primitives)
      const isSimple = entries.every(([_, val]) => 
        typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean'
      );
      
      if (isSimple && entries.length <= 3 && depth > 0) {
        // Inline simple objects
        const pairs = entries.map(([k, v]) => `${formatKey(k)}: ${v}`);
        return pairs.join(', ');
      }
      
      let result = '';
      entries.forEach(([key, val], index) => {
        const formattedKey = formatKey(key);
        
        if (val === null || val === undefined) {
          return; // Skip null/undefined
        }
        
        if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
          result += `${indent}**${formattedKey}:** ${val}\n`;
        } else if (Array.isArray(val)) {
          result += `${indent}**${formattedKey}:**\n${formatValue(val, indent, depth + 1)}`;
        } else if (typeof val === 'object') {
          // Use appropriate heading level based on depth
          const headingLevel = Math.min(depth + 3, 6);
          const heading = '#'.repeat(headingLevel);
          result += `${indent}${heading} ${formattedKey}\n\n${formatValue(val, indent, depth + 1)}`;
        }
        
        // Add spacing between top-level sections
        if (depth === 0 && index < entries.length - 1) {
          result += '\n';
        }
      });
      
      return result;
    }
    
    return String(value);
  };
  
  // Helper to format keys nicely
  const formatKey = (key: string): string => {
    return key
      .replace(/_/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .split(' ')
      .filter(word => word.length > 0)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  // Convert PRP document structure to readable markdown - fully dynamic
  const convertPRPToMarkdown = (content: any): string => {
    // Handle raw string content
    if (typeof content === 'string') {
      return content;
    }
    
    // Handle null/undefined
    if (!content || typeof content !== 'object') {
      return `# ${doc.title}\n\nNo content available.`;
    }
    
    // Start with title
    let markdown = `# ${content.title || doc.title || 'Untitled Document'}\n\n`;
    
    // Group metadata fields
    const metadataFields = ['version', 'author', 'date', 'status', 'document_type', 'created_at', 'updated_at'];
    const metadata = metadataFields.filter(field => content[field]);
    
    if (metadata.length > 0) {
      markdown += `## Metadata\n\n`;
      metadata.forEach(field => {
        const value = content[field];
        const label = formatKey(field);
        markdown += `- **${label}:** ${value}\n`;
      });
      markdown += '\n';
    }
    
    // Process all other fields dynamically
    const skipFields = ['title', ...metadataFields, 'id', '_id', 'project_id'];
    
    // Sort fields by priority (known important fields first)
    const priorityFields = [
      'goal', 'goals', 'objective', 'objectives',
      'why', 'rationale', 'background',
      'what', 'description', 'overview',
      'context', 'background_context',
      'user_personas', 'personas', 'users', 'stakeholders',
      'user_flows', 'flows', 'journeys', 'workflows',
      'requirements', 'functional_requirements', 'non_functional_requirements',
      'success_metrics', 'metrics', 'kpis', 'success_criteria',
      'timeline', 'roadmap', 'milestones', 'phases',
      'implementation_plan', 'implementation_roadmap', 'plan',
      'technical_requirements', 'technical_implementation', 'architecture',
      'validation_gates', 'testing_strategy', 'quality_gates',
      'risks', 'risk_assessment', 'mitigation_strategies'
    ];
    
    // Create ordered list of fields
    const orderedFields = [];
    const remainingFields = [];
    
    Object.keys(content).forEach(key => {
      if (skipFields.includes(key)) return;
      
      const lowerKey = key.toLowerCase();
      const priorityIndex = priorityFields.findIndex(pf => 
        lowerKey === pf || lowerKey.includes(pf) || pf.includes(lowerKey)
      );
      
      if (priorityIndex !== -1) {
        orderedFields.push({ key, priority: priorityIndex });
      } else {
        remainingFields.push(key);
      }
    });
    
    // Sort by priority
    orderedFields.sort((a, b) => a.priority - b.priority);
    
    // Process fields in order
    const allFields = [...orderedFields.map(f => f.key), ...remainingFields];
    
    allFields.forEach(key => {
      const value = content[key];
      if (value === null || value === undefined) return;
      
      const sectionTitle = formatKey(key);
      markdown += `## ${sectionTitle}\n\n`;
      
      // Handle different value types
      if (typeof value === 'string') {
        markdown += `${value}\n\n`;
      } else if (typeof value === 'number' || typeof value === 'boolean') {
        markdown += `${value}\n\n`;
      } else if (Array.isArray(value)) {
        markdown += formatValue(value) + '\n';
      } else if (typeof value === 'object') {
        markdown += formatValue(value) + '\n';
      }
    });
    
    return markdown.trim();
  };

  // Initialize editor
  useEffect(() => {
    if (!editorRef.current || crepeRef.current) return;

    const initialContent = getMarkdownContent();
    setOriginalContent(initialContent);
    setCurrentContent(initialContent);

    // Add theme class to root element
    if (isDarkMode) {
      editorRef.current.classList.add('milkdown-theme-dark');
    }

    const crepe = new Crepe({
      root: editorRef.current,
      defaultValue: initialContent,
      features: {
        [CrepeFeature.HeaderMeta]: true,
        [CrepeFeature.LinkTooltip]: true,
        [CrepeFeature.ImageBlock]: true,
        [CrepeFeature.BlockEdit]: true,
        [CrepeFeature.ListItem]: true,
        [CrepeFeature.CodeBlock]: true,
        [CrepeFeature.Table]: true,
        [CrepeFeature.Toolbar]: true,
      },
    });

    crepe.create().then(() => {
      console.log('Milkdown editor created');
      
      // Set up content change tracking
      const editorElement = editorRef.current?.querySelector('.ProseMirror');
      if (editorElement) {
        // Listen for input events on the editor
        const handleInput = () => {
          // Get current markdown content
          const markdown = crepe.getMarkdown();
          console.log('Editor content changed via input:', markdown.substring(0, 50) + '...');
          setCurrentContent(markdown);
          
          // Compare trimmed content to avoid whitespace issues
          const hasUnsavedChanges = markdown.trim() !== originalContent.trim();
          setHasChanges(hasUnsavedChanges);
          setIsReverted(false);
        };
        
        // Listen to multiple events to catch all changes
        editorElement.addEventListener('input', handleInput);
        editorElement.addEventListener('keyup', handleInput);
        editorElement.addEventListener('paste', handleInput);
        editorElement.addEventListener('cut', handleInput);
        
        // Store the handlers for cleanup
        (editorElement as any)._milkdownHandlers = {
          input: handleInput,
          keyup: handleInput,
          paste: handleInput,
          cut: handleInput
        };
      }
    }).catch((error) => {
      console.error('Failed to create Milkdown editor:', error);
    });

    crepeRef.current = crepe;

    return () => {
      // Clean up event listeners
      const editorElement = editorRef.current?.querySelector('.ProseMirror');
      if (editorElement && (editorElement as any)._milkdownHandlers) {
        const handlers = (editorElement as any)._milkdownHandlers;
        editorElement.removeEventListener('input', handlers.input);
        editorElement.removeEventListener('keyup', handlers.keyup);
        editorElement.removeEventListener('paste', handlers.paste);
        editorElement.removeEventListener('cut', handlers.cut);
        delete (editorElement as any)._milkdownHandlers;
      }
      
      if (crepeRef.current) {
        crepeRef.current.destroy();
        crepeRef.current = null;
      }
    };
  }, [doc.id, originalContent]);

  // Update theme class when isDarkMode changes
  useEffect(() => {
    if (editorRef.current) {
      if (isDarkMode) {
        editorRef.current.classList.add('milkdown-theme-dark');
      } else {
        editorRef.current.classList.remove('milkdown-theme-dark');
      }
    }
  }, [isDarkMode]);

  // Add keyboard shortcut for saving
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        if (hasChanges && !isLoading) {
          handleSave();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [hasChanges, isLoading, currentContent]);

  // Handle manual save
  const handleSave = async () => {
    if (!hasChanges || isLoading) return;
    
    try {
      setIsLoading(true);
      console.log('Saving document with content:', currentContent.substring(0, 100) + '...');
      
      // Create updated document with markdown content stored in content field
      const updatedDocument = {
        ...doc,
        content: {
          markdown: currentContent,
          // Preserve any other content fields
          ...(typeof doc.content === 'object' && doc.content !== null ? doc.content : {})
        },
        updated_at: new Date().toISOString(),
      };
      
      await onSave(updatedDocument);
      
      // Update state after successful save
      setHasChanges(false);
      setIsReverted(false);
      setOriginalContent(currentContent);
      console.log('Document saved successfully');
    } catch (error) {
      console.error('Error saving document:', error);
      // You might want to show an error toast here
    } finally {
      setIsLoading(false);
    }
  };

  // Handle undo changes
  const handleUndo = () => {
    if (crepeRef.current && editorRef.current) {
      // Destroy and recreate editor with original content
      crepeRef.current.destroy();
      
      const crepe = new Crepe({
        root: editorRef.current,
        defaultValue: originalContent,
        features: {
          [CrepeFeature.HeaderMeta]: true,
          [CrepeFeature.LinkTooltip]: true,
          [CrepeFeature.ImageBlock]: true,
          [CrepeFeature.BlockEdit]: true,
          [CrepeFeature.ListItem]: true,
          [CrepeFeature.CodeBlock]: true,
          [CrepeFeature.Table]: true,
          [CrepeFeature.Toolbar]: true,
        },
      });

      crepe.create().then(() => {
        console.log('Milkdown editor reverted to original content');
        
        // Set up content change tracking for the new editor instance
        const editorElement = editorRef.current?.querySelector('.ProseMirror');
        if (editorElement) {
          const handleInput = () => {
            const markdown = crepe.getMarkdown();
            console.log('Editor content changed after undo:', markdown.substring(0, 50) + '...');
            setCurrentContent(markdown);
            const hasUnsavedChanges = markdown.trim() !== originalContent.trim();
            setHasChanges(hasUnsavedChanges);
            setIsReverted(false);
          };
          
          editorElement.addEventListener('input', handleInput);
          editorElement.addEventListener('keyup', handleInput);
          editorElement.addEventListener('paste', handleInput);
          editorElement.addEventListener('cut', handleInput);
          
          (editorElement as any)._milkdownHandlers = {
            input: handleInput,
            keyup: handleInput,
            paste: handleInput,
            cut: handleInput
          };
        }
        
        setCurrentContent(originalContent);
        setHasChanges(false);
        setIsReverted(true);
      }).catch((error) => {
        console.error('Failed to revert Milkdown editor:', error);
      });

      crepeRef.current = crepe;
    }
  };

  return (
    <div className={`milkdown-editor ${className}`}>
      <div className="mb-6 flex items-center justify-between bg-white/50 dark:bg-black/30 backdrop-blur-sm rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3">
          <h3 className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            {doc.title}
          </h3>
          <div className="flex items-center gap-2">
            {isLoading ? (
              <span className="text-sm text-blue-600 dark:text-blue-400 flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                Saving...
              </span>
            ) : isReverted ? (
              <span className="text-sm text-purple-600 dark:text-purple-400 flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                Reverted
              </span>
            ) : hasChanges ? (
              <span className="text-sm text-orange-600 dark:text-orange-400 flex items-center gap-2">
                <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
                Unsaved changes
              </span>
            ) : (
              <span className="text-sm text-green-600 dark:text-green-400 flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                All changes saved
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {hasChanges && (
            <button
              onClick={handleUndo}
              disabled={isLoading}
              className="px-4 py-2 bg-gray-500/20 hover:bg-gray-500/30 text-gray-700 dark:text-gray-300 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 border border-gray-300 dark:border-gray-600"
            >
              <Undo className="w-4 h-4" />
              Undo
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={isLoading || !hasChanges}
            className={`
              px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 
              flex items-center gap-2 border
              ${hasChanges 
                ? 'bg-blue-500 hover:bg-blue-600 text-white border-blue-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 save-button-pulse' 
                : 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600 border-gray-300 dark:border-gray-700 cursor-not-allowed'
              }
              disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none
            `}
          >
            <Save className="w-4 h-4" />
            {isLoading ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
      
      <div 
        ref={editorRef} 
        className={`prose prose-lg max-w-none milkdown-crepe-editor ${isDarkMode ? 'prose-invert' : ''}`}
        style={{ minHeight: '400px' }}
      />
    </div>
  );
};