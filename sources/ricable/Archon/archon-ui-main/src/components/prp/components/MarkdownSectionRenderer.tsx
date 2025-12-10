import React from 'react';
import { ParsedSection } from '../utils/markdownParser';
import { SectionRenderer } from '../renderers/SectionRenderer';
import { SimpleMarkdown } from './SimpleMarkdown';
import { detectSectionType } from '../utils/sectionDetector';

interface MarkdownSectionRendererProps {
  section: ParsedSection;
  index: number;
  isDarkMode?: boolean;
  sectionOverrides?: Record<string, React.ComponentType<any>>;
}

/**
 * Renders individual markdown sections with smart template detection
 * Uses specialized components for known PRP templates, beautiful styling for generic sections
 */
export const MarkdownSectionRenderer: React.FC<MarkdownSectionRendererProps> = ({
  section,
  index,
  isDarkMode = false,
  sectionOverrides = {}
}) => {
  // If section matches a known PRP template, use the specialized component
  if (section.templateType) {
    const { type } = detectSectionType(section.sectionKey, section.rawContent);
    
    // Use the existing SectionRenderer with the detected type
    return (
      <div className="mb-6">
        <SectionRenderer
          sectionKey={section.sectionKey}
          data={section.rawContent}
          index={index}
          isDarkMode={isDarkMode}
          sectionOverrides={sectionOverrides}
        />
      </div>
    );
  }

  // For generic sections, render with beautiful floating styling
  return (
    <section className="mb-8">
      <div className="relative">
        {/* Section Header */}
        <div className="mb-4">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {section.title}
          </h2>
          <div className="mt-1 h-0.5 w-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></div>
        </div>

        {/* Section Content */}
        <div className="relative">
          {/* Subtle background for sections with complex content */}
          {(section.type === 'code' || section.type === 'mixed') && (
            <div className="absolute inset-0 bg-gray-50/30 dark:bg-gray-900/20 rounded-xl -m-4 backdrop-blur-sm border border-gray-200/30 dark:border-gray-700/30"></div>
          )}
          
          <div className="relative z-10">
            <SimpleMarkdown 
              content={section.content} 
              className="prose prose-gray dark:prose-invert max-w-none prose-headings:text-gray-900 dark:prose-headings:text-white prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-li:text-gray-700 dark:prose-li:text-gray-300"
            />
          </div>
        </div>
      </div>
    </section>
  );
};