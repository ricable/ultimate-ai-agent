import React from 'react';
import { FileText, Hash, List, Box, Type, ToggleLeft } from 'lucide-react';
import { SectionProps } from '../types/prp.types';
import { formatKey, formatValue } from '../utils/formatters';
import { hasComplexNesting } from '../utils/normalizer';
import { CollapsibleSectionWrapper } from '../components/CollapsibleSectionWrapper';
import { SimpleMarkdown } from '../components/SimpleMarkdown';

/**
 * Generic fallback section component that intelligently renders any data structure
 * This component provides comprehensive rendering for any data type with proper formatting
 */
export const GenericSection: React.FC<SectionProps> = ({
  title,
  data,
  icon = <FileText className="w-5 h-5" />,
  accentColor = 'gray',
  defaultOpen = true,
  isDarkMode = false,
  isCollapsible = true,
  isOpen,
  onToggle
}) => {
  // Auto-detect appropriate icon based on data type
  const getAutoIcon = () => {
    if (typeof data === 'string') return <Type className="w-5 h-5" />;
    if (typeof data === 'number') return <Hash className="w-5 h-5" />;
    if (typeof data === 'boolean') return <ToggleLeft className="w-5 h-5" />;
    if (Array.isArray(data)) return <List className="w-5 h-5" />;
    if (typeof data === 'object' && data !== null) return <Box className="w-5 h-5" />;
    return icon;
  };
  const renderValue = (value: any, depth: number = 0): React.ReactNode => {
    const indent = depth * 16;
    const maxDepth = 5; // Prevent infinite recursion
    
    // Handle null/undefined
    if (value === null || value === undefined) {
      return <span className="text-gray-400 italic">Empty</span>;
    }
    
    // Handle primitives
    if (typeof value === 'string') {
      // Check if the string looks like markdown content
      const hasMarkdownIndicators = /^#{1,6}\s+.+$|^[-*+]\s+.+$|^\d+\.\s+.+$|```|^\>.+$|\*\*.+\*\*|\*.+\*|`[^`]+`/m.test(value);
      
      if (hasMarkdownIndicators && value.length > 20) {
        // Render as markdown for content with markdown syntax
        // Remove any leading headers since the section already has a title
        const contentWithoutLeadingHeaders = value.replace(/^#{1,6}\s+.+$/m, '').trim();
        const finalContent = contentWithoutLeadingHeaders || value;
        
        return <SimpleMarkdown content={finalContent} className="text-gray-700 dark:text-gray-300" />;
      }
      
      // For shorter strings or non-markdown, use simple formatting
      return <span className="text-gray-700 dark:text-gray-300">{formatValue(value)}</span>;
    }
    
    if (typeof value === 'number' || typeof value === 'boolean') {
      return <span className="text-gray-700 dark:text-gray-300 font-mono">{formatValue(value)}</span>;
    }
    
    // Prevent deep recursion
    if (depth >= maxDepth) {
      return (
        <span className="text-gray-500 italic text-sm">
          [Complex nested structure - too deep to display]
        </span>
      );
    }
    
    // Handle arrays
    if (Array.isArray(value)) {
      if (value.length === 0) {
        return <span className="text-gray-400 italic">No items</span>;
      }
      
      // Check if it's an array of primitives
      const isSimpleArray = value.every(item => 
        typeof item === 'string' || 
        typeof item === 'number' || 
        typeof item === 'boolean' ||
        item === null ||
        item === undefined
      );
      
      if (isSimpleArray) {
        // For very long arrays, show first 10 and count
        const displayItems = value.length > 10 ? value.slice(0, 10) : value;
        const hasMore = value.length > 10;
        
        return (
          <div>
            <ul className="space-y-1 mt-2">
              {displayItems.map((item, index) => (
                <li key={index} className="flex items-start gap-2" style={{ marginLeft: indent }}>
                  <span className="text-gray-400 mt-0.5">•</span>
                  <span className="text-gray-700 dark:text-gray-300">{formatValue(item)}</span>
                </li>
              ))}
            </ul>
            {hasMore && (
              <p className="text-sm text-gray-500 italic mt-2" style={{ marginLeft: indent + 16 }}>
                ... and {value.length - 10} more items
              </p>
            )}
          </div>
        );
      }
      
      // Array of objects
      const displayItems = value.length > 5 ? value.slice(0, 5) : value;
      const hasMore = value.length > 5;
      
      return (
        <div className="space-y-3 mt-2">
          {displayItems.map((item, index) => (
            <div key={index} className="relative" style={{ marginLeft: indent }}>
              <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-gradient-to-b from-gray-300 to-transparent dark:from-gray-600"></div>
              <div className="pl-4">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  [{index}]
                </div>
                {renderValue(item, depth + 1)}
              </div>
            </div>
          ))}
          {hasMore && (
            <p className="text-sm text-gray-500 italic" style={{ marginLeft: indent + 16 }}>
              ... and {value.length - 5} more items
            </p>
          )}
        </div>
      );
    }
    
    // Handle objects
    if (typeof value === 'object' && value !== null) {
      // Simplified object rendering to debug black screen
      return (
        <div className="mt-2 text-gray-700 dark:text-gray-300">
          <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded whitespace-pre-wrap">
            {JSON.stringify(value, null, 2)}
          </pre>
        </div>
      );
    }
    
    // Fallback for any other type (functions, symbols, etc.)
    return (
      <span className="text-gray-500 italic text-sm">
        [{typeof value}]
      </span>
    );
  };
  
  const getBackgroundColor = () => {
    const colorMap = {
      blue: 'bg-blue-50/50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      purple: 'bg-purple-50/50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
      green: 'bg-green-50/50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      orange: 'bg-orange-50/50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
      pink: 'bg-pink-50/50 dark:bg-pink-900/20 border-pink-200 dark:border-pink-800',
      cyan: 'bg-cyan-50/50 dark:bg-cyan-900/20 border-cyan-200 dark:border-cyan-800',
      gray: 'bg-gray-50/50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800',
    };
    return colorMap[accentColor as keyof typeof colorMap] || colorMap.gray;
  };
  
  const finalIcon = icon === <FileText className="w-5 h-5" /> ? getAutoIcon() : icon;
  
  // Enhanced styling based on data complexity
  const isComplexData = hasComplexNesting(data);
  const headerClass = isComplexData 
    ? `p-6 rounded-lg border-2 shadow-sm ${getBackgroundColor()}`
    : `p-4 rounded-lg border ${getBackgroundColor()}`;
  
  const header = (
    <div className={headerClass}>
      <h3 className="font-semibold text-gray-800 dark:text-white flex items-center gap-2">
        <div className="p-1.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
          {finalIcon}
        </div>
        <span className="flex-1">{title}</span>
      </h3>
    </div>
  );

  const contentClass = isComplexData 
    ? `px-6 pb-6 -mt-1 rounded-b-lg border-2 border-t-0 shadow-sm ${getBackgroundColor()}`
    : `px-4 pb-4 -mt-1 rounded-b-lg border border-t-0 ${getBackgroundColor()}`;

  const content = (
    <div className={contentClass}>
      <div className="overflow-x-auto">
        {/* Add a subtle background for complex data */}
        {isComplexData ? (
          <div className="bg-gray-50 dark:bg-gray-900/50 rounded p-3 -mx-2">
            {renderValue(data)}
          </div>
        ) : (
          renderValue(data)
        )}
      </div>
    </div>
  );
  
  try {
    return (
      <CollapsibleSectionWrapper
        header={header}
        isCollapsible={isCollapsible}
        defaultOpen={defaultOpen}
        isOpen={isOpen}
        onToggle={onToggle}
      >
        {content}
      </CollapsibleSectionWrapper>
    );
  } catch (error) {
    console.error('Error rendering GenericSection:', error, { title, data });
    return (
      <div className="p-4 border border-red-300 rounded bg-red-50 dark:bg-red-900">
        <h3 className="text-red-800 dark:text-red-200 font-semibold">{title}</h3>
        <p className="text-red-600 dark:text-red-300 text-sm mt-2">Error rendering section content</p>
        <pre className="text-xs mt-2 bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    );
  }
};