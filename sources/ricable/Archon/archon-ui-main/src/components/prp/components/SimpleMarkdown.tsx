import React from 'react';
import { formatValue } from '../utils/formatters';

interface SimpleMarkdownProps {
  content: string;
  className?: string;
}

/**
 * Simple markdown renderer that handles basic formatting without external dependencies
 */
export const SimpleMarkdown: React.FC<SimpleMarkdownProps> = ({ content, className = '' }) => {
  try {
    // Process image placeholders first
    const processedContent = formatValue(content);
  
  // Split content into lines for processing
  const lines = processedContent.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: string[] = [];
  let listType: 'ul' | 'ol' | null = null;
  
  const flushList = () => {
    if (currentList.length > 0 && listType) {
      const ListComponent = listType === 'ul' ? 'ul' : 'ol';
      elements.push(
        <div key={elements.length} className="my-3">
          <ListComponent className={`space-y-2 ${listType === 'ul' ? 'list-disc' : 'list-decimal'} pl-6 text-gray-700 dark:text-gray-300`}>
            {currentList.map((item, idx) => (
              <li key={idx} className="leading-relaxed">{processInlineMarkdown(item)}</li>
            ))}
          </ListComponent>
        </div>
      );
      currentList = [];
      listType = null;
    }
  };
  
  const processInlineMarkdown = (text: string): React.ReactNode => {
    const processed = text;
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;
    
    // Process **bold** text
    const boldRegex = /\*\*(.*?)\*\*/g;
    let match;
    while ((match = boldRegex.exec(processed)) !== null) {
      if (match.index > lastIndex) {
        elements.push(processed.slice(lastIndex, match.index));
      }
      elements.push(<strong key={match.index} className="font-semibold">{match[1]}</strong>);
      lastIndex = match.index + match[0].length;
    }
    
    // Process *italic* text
    const italicRegex = /\*(.*?)\*/g;
    const remainingText = processed.slice(lastIndex);
    lastIndex = 0;
    const italicElements: React.ReactNode[] = [];
    
    while ((match = italicRegex.exec(remainingText)) !== null) {
      if (match.index > lastIndex) {
        italicElements.push(remainingText.slice(lastIndex, match.index));
      }
      italicElements.push(<em key={match.index} className="italic">{match[1]}</em>);
      lastIndex = match.index + match[0].length;
    }
    
    if (lastIndex < remainingText.length) {
      italicElements.push(remainingText.slice(lastIndex));
    }
    
    if (elements.length > 0) {
      elements.push(...italicElements);
      return <>{elements}</>;
    }
    
    if (italicElements.length > 0) {
      return <>{italicElements}</>;
    }
    
    // Process `inline code`
    const codeRegex = /`([^`]+)`/g;
    const parts = text.split(codeRegex);
    if (parts.length > 1) {
      return (
        <>
          {parts.map((part, index) => 
            index % 2 === 0 ? (
              part
            ) : (
              <code key={index} className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800 dark:text-gray-200">
                {part}
              </code>
            )
          )}
        </>
      );
    }
    
    return <span>{text}</span>;
  };
  
  let inCodeBlock = false;
  let codeBlockContent: string[] = [];
  let codeBlockLanguage = '';
  let inTable = false;
  let tableRows: string[][] = [];
  let tableHeaders: string[] = [];

  const flushTable = () => {
    if (tableRows.length > 0) {
      elements.push(
        <div key={elements.length} className="my-6 overflow-x-auto">
          <div className="inline-block min-w-full overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm">
            <table className="min-w-full">
              {tableHeaders.length > 0 && (
                <thead className="bg-gray-50 dark:bg-gray-900/50">
                  <tr>
                    {tableHeaders.map((header, idx) => (
                      <th key={idx} className="px-4 py-3 text-left text-sm font-semibold text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700">
                        {processInlineMarkdown(header.trim())}
                      </th>
                    ))}
                  </tr>
                </thead>
              )}
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {tableRows.map((row, rowIdx) => (
                  <tr key={rowIdx} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                    {row.map((cell, cellIdx) => (
                      <td key={cellIdx} className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                        {processInlineMarkdown(cell.trim())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
      tableRows = [];
      tableHeaders = [];
      inTable = false;
    }
  };

  lines.forEach((line, index) => {
    // Handle code block start/end
    if (line.startsWith('```')) {
      if (!inCodeBlock) {
        // Starting code block
        flushList();
        inCodeBlock = true;
        codeBlockLanguage = line.substring(3).trim();
        codeBlockContent = [];
      } else {
        // Ending code block
        inCodeBlock = false;
        elements.push(
          <div key={index} className="my-4 rounded-lg overflow-hidden bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 border border-gray-700 shadow-lg">
            {codeBlockLanguage && (
              <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-700 text-sm text-gray-300 font-mono">
                {codeBlockLanguage}
              </div>
            )}
            <pre className="p-4 overflow-x-auto">
              <code className="text-gray-100 font-mono text-sm leading-relaxed">
                {codeBlockContent.join('\n')}
              </code>
            </pre>
          </div>
        );
        codeBlockContent = [];
        codeBlockLanguage = '';
      }
      return;
    }
    
    // If inside code block, collect content
    if (inCodeBlock) {
      codeBlockContent.push(line);
      return;
    }

    // Handle table rows
    if (line.includes('|') && line.trim() !== '') {
      const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell !== '');
      
      if (cells.length > 0) {
        if (!inTable) {
          // Starting a new table
          flushList();
          inTable = true;
          tableHeaders = cells;
        } else if (cells.every(cell => cell.match(/^:?-+:?$/))) {
          // This is a header separator line (|---|---|), skip it
          return;
        } else {
          // This is a regular table row
          tableRows.push(cells);
        }
        return;
      }
    } else if (inTable) {
      // End of table (empty line or non-table content)
      flushTable();
    }

    // Handle headings
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushList();
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      const HeadingTag = `h${level}` as keyof JSX.IntrinsicElements;
      const sizeClasses = ['text-2xl', 'text-xl', 'text-lg', 'text-base', 'text-sm', 'text-xs'];
      const colorClasses = ['text-gray-900 dark:text-white', 'text-gray-800 dark:text-gray-100', 'text-gray-700 dark:text-gray-200', 'text-gray-700 dark:text-gray-200', 'text-gray-600 dark:text-gray-300', 'text-gray-600 dark:text-gray-300'];
      
      elements.push(
        <HeadingTag key={index} className={`font-bold mb-3 mt-6 ${sizeClasses[level - 1] || 'text-base'} ${colorClasses[level - 1] || 'text-gray-700 dark:text-gray-200'} border-b border-gray-200 dark:border-gray-700 pb-1`}>
          {processInlineMarkdown(text)}
        </HeadingTag>
      );
      return;
    }
    
    // Handle checkboxes (task lists)
    const checkboxMatch = line.match(/^[-*+]\s+\[([ x])\]\s+(.+)$/);
    if (checkboxMatch) {
      flushList();
      const isChecked = checkboxMatch[1] === 'x';
      const content = checkboxMatch[2];
      elements.push(
        <div key={index} className="flex items-start gap-3 my-2">
          <div className={`flex-shrink-0 w-5 h-5 rounded-md border-2 flex items-center justify-center mt-0.5 transition-colors ${
            isChecked 
              ? 'bg-green-500 border-green-500 text-white' 
              : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800'
          }`}>
            {isChecked && (
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            )}
          </div>
          <div className={`flex-1 leading-relaxed ${isChecked ? 'text-gray-500 dark:text-gray-400 line-through' : 'text-gray-700 dark:text-gray-300'}`}>
            {processInlineMarkdown(content)}
          </div>
        </div>
      );
      return;
    }

    // Handle bullet lists
    const bulletMatch = line.match(/^[-*+]\s+(.+)$/);
    if (bulletMatch) {
      if (listType !== 'ul') {
        flushList();
        listType = 'ul';
      }
      currentList.push(bulletMatch[1]);
      return;
    }
    
    // Handle numbered lists
    const numberMatch = line.match(/^\d+\.\s+(.+)$/);
    if (numberMatch) {
      if (listType !== 'ol') {
        flushList();
        listType = 'ol';
      }
      currentList.push(numberMatch[1]);
      return;
    }
    
    // Handle code blocks
    if (line.startsWith('```')) {
      flushList();
      // Simple code block handling - just skip the backticks
      return;
    }
    
    // Handle blockquotes
    if (line.startsWith('>')) {
      flushList();
      const content = line.substring(1).trim();
      elements.push(
        <blockquote key={index} className="border-l-4 border-blue-400 dark:border-blue-500 bg-blue-50/50 dark:bg-blue-900/20 pl-4 pr-4 py-3 italic my-4 rounded-r-lg backdrop-blur-sm">
          <div className="text-gray-700 dark:text-gray-300">
            {processInlineMarkdown(content)}
          </div>
        </blockquote>
      );
      return;
    }
    
    // Handle horizontal rules
    if (line.match(/^(-{3,}|_{3,}|\*{3,})$/)) {
      flushList();
      elements.push(<hr key={index} className="my-4 border-gray-300 dark:border-gray-700" />);
      return;
    }
    
    // Regular paragraph
    if (line.trim()) {
      flushList();
      elements.push(
        <p key={index} className="mb-3 leading-relaxed text-gray-700 dark:text-gray-300">
          {processInlineMarkdown(line)}
        </p>
      );
    }
  });
  
  // Flush any remaining list or table
  flushList();
  flushTable();
  
    return (
      <div className={`max-w-none ${className}`}>
        <div className="space-y-1">
          {elements}
        </div>
      </div>
    );
  } catch (error) {
    console.error('Error rendering markdown:', error, content);
    return (
      <div className={`text-gray-700 dark:text-gray-300 ${className}`}>
        <p>Error rendering markdown content</p>
        <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded mt-2 whitespace-pre-wrap">
          {content}
        </pre>
      </div>
    );
  }
};