import React, { useState, useEffect, ReactNode } from 'react';
import { 
  ChevronDown, 
  Brain, Users, Workflow, BarChart3, Clock, Shield, 
  Code, Layers, FileText, List, Hash, Box, Type, ToggleLeft,
  CheckCircle, AlertCircle, Info, Lightbulb
} from 'lucide-react';
import { SectionProps } from '../types/prp.types';
import { SimpleMarkdown } from './SimpleMarkdown';
import { formatValue } from '../utils/formatters';

interface CollapsibleSectionRendererProps extends SectionProps {
  children?: ReactNode;
  headerContent?: ReactNode;
  sectionKey?: string;
  contentType?: 'markdown' | 'code' | 'json' | 'list' | 'object' | 'auto';
  animationDuration?: number;
  showPreview?: boolean;
  previewLines?: number;
}

/**
 * Enhanced CollapsibleSectionRenderer with beautiful animations and content-aware styling
 * Features:
 * - Section-specific icons and colors
 * - Smooth expand/collapse animations with dynamic height
 * - Content type detection and appropriate formatting
 * - Code block syntax highlighting support
 * - Nested structure handling
 * - Preview mode for collapsed content
 */
export const CollapsibleSectionRenderer: React.FC<CollapsibleSectionRendererProps> = ({
  title,
  data,
  icon,
  accentColor = 'gray',
  defaultOpen = true,
  isDarkMode = false,
  isCollapsible = true,
  isOpen: controlledIsOpen,
  onToggle,
  children,
  headerContent,
  sectionKey = '',
  contentType = 'auto',
  animationDuration = 300,
  showPreview = true,
  previewLines = 2
}) => {
  // State management for collapsible behavior
  const [internalIsOpen, setInternalIsOpen] = useState(defaultOpen);
  const [contentHeight, setContentHeight] = useState<number | 'auto'>('auto');
  const [isAnimating, setIsAnimating] = useState(false);
  
  const isOpen = controlledIsOpen !== undefined ? controlledIsOpen : internalIsOpen;

  // Content ref for measuring height
  const contentRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (controlledIsOpen === undefined) {
      setInternalIsOpen(defaultOpen);
    }
  }, [defaultOpen, controlledIsOpen]);

  // Measure content height for smooth animations
  useEffect(() => {
    if (contentRef.current && isCollapsible) {
      const height = contentRef.current.scrollHeight;
      setContentHeight(isOpen ? height : 0);
    }
  }, [isOpen, data, children]);

  const handleToggle = () => {
    if (!isCollapsible) return;
    
    setIsAnimating(true);
    
    if (controlledIsOpen === undefined) {
      setInternalIsOpen(!internalIsOpen);
    }
    onToggle?.();

    // Reset animation state after duration
    setTimeout(() => setIsAnimating(false), animationDuration);
  };

  // Auto-detect section type and get appropriate icon
  const getSectionIcon = (): ReactNode => {
    if (icon) return icon;
    
    const normalizedKey = sectionKey.toLowerCase();
    const normalizedTitle = title.toLowerCase();
    
    // Check both section key and title for better detection
    const checkKeywords = (keywords: string[]) => 
      keywords.some(keyword => 
        normalizedKey.includes(keyword) || normalizedTitle.includes(keyword)
      );

    if (checkKeywords(['context', 'overview', 'background'])) 
      return <Brain className="w-5 h-5" />;
    if (checkKeywords(['persona', 'user', 'actor', 'stakeholder'])) 
      return <Users className="w-5 h-5" />;
    if (checkKeywords(['flow', 'journey', 'workflow', 'process'])) 
      return <Workflow className="w-5 h-5" />;
    if (checkKeywords(['metric', 'success', 'kpi', 'measurement'])) 
      return <BarChart3 className="w-5 h-5" />;
    if (checkKeywords(['plan', 'implementation', 'roadmap', 'timeline'])) 
      return <Clock className="w-5 h-5" />;
    if (checkKeywords(['validation', 'gate', 'criteria', 'acceptance'])) 
      return <Shield className="w-5 h-5" />;
    if (checkKeywords(['technical', 'tech', 'architecture', 'system'])) 
      return <Code className="w-5 h-5" />;
    if (checkKeywords(['architecture', 'structure', 'design'])) 
      return <Layers className="w-5 h-5" />;
    if (checkKeywords(['feature', 'functionality', 'capability'])) 
      return <Lightbulb className="w-5 h-5" />;
    if (checkKeywords(['requirement', 'spec', 'specification'])) 
      return <CheckCircle className="w-5 h-5" />;
    if (checkKeywords(['risk', 'issue', 'concern', 'challenge'])) 
      return <AlertCircle className="w-5 h-5" />;
    if (checkKeywords(['info', 'note', 'detail'])) 
      return <Info className="w-5 h-5" />;
    
    // Fallback based on data type
    if (typeof data === 'string') return <Type className="w-5 h-5" />;
    if (typeof data === 'number') return <Hash className="w-5 h-5" />;
    if (typeof data === 'boolean') return <ToggleLeft className="w-5 h-5" />;
    if (Array.isArray(data)) return <List className="w-5 h-5" />;
    if (typeof data === 'object' && data !== null) return <Box className="w-5 h-5" />;
    
    return <FileText className="w-5 h-5" />;
  };

  // Get section-specific color scheme
  const getColorScheme = () => {
    const normalizedKey = sectionKey.toLowerCase();
    const normalizedTitle = title.toLowerCase();
    
    const checkKeywords = (keywords: string[]) => 
      keywords.some(keyword => 
        normalizedKey.includes(keyword) || normalizedTitle.includes(keyword)
      );

    if (checkKeywords(['context', 'overview'])) return 'blue';
    if (checkKeywords(['persona', 'user'])) return 'purple';
    if (checkKeywords(['flow', 'journey'])) return 'orange';
    if (checkKeywords(['metric', 'success'])) return 'green';
    if (checkKeywords(['plan', 'implementation'])) return 'cyan';
    if (checkKeywords(['validation', 'gate'])) return 'emerald';
    if (checkKeywords(['technical', 'architecture'])) return 'indigo';
    if (checkKeywords(['feature'])) return 'yellow';
    if (checkKeywords(['risk', 'issue'])) return 'red';
    
    return accentColor;
  };

  // Auto-detect content type if not specified
  const getContentType = () => {
    if (contentType !== 'auto') return contentType;
    
    if (typeof data === 'string') {
      // Check for code patterns
      if (/^```[\s\S]*```$/m.test(data) || 
          /^\s*(function|class|const|let|var|import|export)\s/m.test(data) ||
          /^\s*[{[][\s\S]*[}\]]$/m.test(data)) {
        return 'code';
      }
      
      // Check for markdown patterns
      if (/^#{1,6}\s+.+$|^[-*+]\s+.+$|^\d+\.\s+.+$|```|^\>.+$|\*\*.+\*\*|\*.+\*|`[^`]+`/m.test(data)) {
        return 'markdown';
      }
    }
    
    if (Array.isArray(data)) return 'list';
    if (typeof data === 'object' && data !== null) {
      try {
        JSON.stringify(data);
        return 'json';
      } catch {
        return 'object';
      }
    }
    
    return 'auto';
  };

  // Render content based on type
  const renderContent = (): ReactNode => {
    if (children) return children;
    
    const detectedType = getContentType();
    
    switch (detectedType) {
      case 'markdown':
        return <SimpleMarkdown content={data} className="text-gray-700 dark:text-gray-300" />;
      
      case 'code':
        return (
          <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{data}</code>
            </pre>
          </div>
        );
      
      case 'json':
        return (
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        );
      
      case 'list':
        if (!Array.isArray(data)) return <span className="text-gray-500 italic">Invalid list data</span>;
        return (
          <ul className="space-y-2">
            {data.map((item, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-gray-400 mt-0.5 flex-shrink-0">•</span>
                <span className="text-gray-700 dark:text-gray-300">{formatValue(item)}</span>
              </li>
            ))}
          </ul>
        );
      
      default:
        return <span className="text-gray-700 dark:text-gray-300">{formatValue(data)}</span>;
    }
  };

  // Generate preview content when collapsed
  const renderPreview = (): ReactNode => {
    if (!showPreview || isOpen || !data) return null;
    
    const dataStr = typeof data === 'string' ? data : JSON.stringify(data);
    const lines = dataStr.split('\n').slice(0, previewLines);
    const preview = lines.join('\n');
    const hasMore = dataStr.split('\n').length > previewLines;
    
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400 mt-2 px-4 pb-2">
        <div className="truncate">
          {preview}
          {hasMore && <span className="ml-1">...</span>}
        </div>
      </div>
    );
  };

  const colorScheme = getColorScheme();
  const sectionIcon = getSectionIcon();

  // Color mapping for backgrounds and borders
  const getColorClasses = () => {
    const colorMap = {
      blue: {
        bg: 'bg-blue-50/50 dark:bg-blue-950/20',
        border: 'border-blue-200 dark:border-blue-800',
        iconBg: 'bg-blue-100 dark:bg-blue-900',
        iconText: 'text-blue-600 dark:text-blue-400',
        accent: 'border-l-blue-500'
      },
      purple: {
        bg: 'bg-purple-50/50 dark:bg-purple-950/20',
        border: 'border-purple-200 dark:border-purple-800',
        iconBg: 'bg-purple-100 dark:bg-purple-900',
        iconText: 'text-purple-600 dark:text-purple-400',
        accent: 'border-l-purple-500'
      },
      green: {
        bg: 'bg-green-50/50 dark:bg-green-950/20',
        border: 'border-green-200 dark:border-green-800',
        iconBg: 'bg-green-100 dark:bg-green-900',
        iconText: 'text-green-600 dark:text-green-400',
        accent: 'border-l-green-500'
      },
      orange: {
        bg: 'bg-orange-50/50 dark:bg-orange-950/20',
        border: 'border-orange-200 dark:border-orange-800',
        iconBg: 'bg-orange-100 dark:bg-orange-900',
        iconText: 'text-orange-600 dark:text-orange-400',
        accent: 'border-l-orange-500'
      },
      cyan: {
        bg: 'bg-cyan-50/50 dark:bg-cyan-950/20',
        border: 'border-cyan-200 dark:border-cyan-800',
        iconBg: 'bg-cyan-100 dark:bg-cyan-900',
        iconText: 'text-cyan-600 dark:text-cyan-400',
        accent: 'border-l-cyan-500'
      },
      indigo: {
        bg: 'bg-indigo-50/50 dark:bg-indigo-950/20',
        border: 'border-indigo-200 dark:border-indigo-800',
        iconBg: 'bg-indigo-100 dark:bg-indigo-900',
        iconText: 'text-indigo-600 dark:text-indigo-400',
        accent: 'border-l-indigo-500'
      },
      emerald: {
        bg: 'bg-emerald-50/50 dark:bg-emerald-950/20',
        border: 'border-emerald-200 dark:border-emerald-800',
        iconBg: 'bg-emerald-100 dark:bg-emerald-900',
        iconText: 'text-emerald-600 dark:text-emerald-400',
        accent: 'border-l-emerald-500'
      },
      yellow: {
        bg: 'bg-yellow-50/50 dark:bg-yellow-950/20',
        border: 'border-yellow-200 dark:border-yellow-800',
        iconBg: 'bg-yellow-100 dark:bg-yellow-900',
        iconText: 'text-yellow-600 dark:text-yellow-400',
        accent: 'border-l-yellow-500'
      },
      red: {
        bg: 'bg-red-50/50 dark:bg-red-950/20',
        border: 'border-red-200 dark:border-red-800',
        iconBg: 'bg-red-100 dark:bg-red-900',
        iconText: 'text-red-600 dark:text-red-400',
        accent: 'border-l-red-500'
      },
      gray: {
        bg: 'bg-gray-50/50 dark:bg-gray-950/20',
        border: 'border-gray-200 dark:border-gray-800',
        iconBg: 'bg-gray-100 dark:bg-gray-900',
        iconText: 'text-gray-600 dark:text-gray-400',
        accent: 'border-l-gray-500'
      }
    };
    return colorMap[colorScheme as keyof typeof colorMap] || colorMap.gray;
  };

  const colors = getColorClasses();

  if (!isCollapsible) {
    return (
      <div className={`rounded-lg border-l-4 ${colors.accent} ${colors.bg} ${colors.border} shadow-sm`}>
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className={`p-2 rounded-lg ${colors.iconBg} ${colors.iconText}`}>
              {sectionIcon}
            </div>
            <h3 className="font-semibold text-gray-800 dark:text-white flex-1">
              {title}
            </h3>
            {headerContent}
          </div>
          <div className="space-y-4">
            {renderContent()}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-lg border-l-4 ${colors.accent} ${colors.bg} ${colors.border} shadow-sm overflow-hidden`}>
      {/* Header */}
      <div 
        className={`
          cursor-pointer select-none p-6 
          hover:bg-opacity-75 transition-colors duration-200
          ${isAnimating ? 'pointer-events-none' : ''}
        `}
        onClick={handleToggle}
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${colors.iconBg} ${colors.iconText}`}>
            {sectionIcon}
          </div>
          <h3 className="font-semibold text-gray-800 dark:text-white flex-1">
            {title}
          </h3>
          {headerContent}
          <div className={`
            transform transition-transform duration-200 
            ${isOpen ? 'rotate-180' : 'rotate-0'}
            text-gray-500 dark:text-gray-400
            hover:text-gray-700 dark:hover:text-gray-200
          `}>
            <ChevronDown className="w-5 h-5" />
          </div>
        </div>
        {renderPreview()}
      </div>

      {/* Content with smooth height animation */}
      <div 
        className="overflow-hidden transition-all ease-in-out"
        style={{ 
          maxHeight: isOpen ? contentHeight : 0,
          transitionDuration: `${animationDuration}ms`
        }}
      >
        <div 
          ref={contentRef}
          className={`px-6 pb-6 space-y-4 ${
            isOpen ? 'opacity-100' : 'opacity-0'
          } transition-opacity duration-200`}
          style={{ 
            transitionDelay: isOpen ? '100ms' : '0ms'
          }}
        >
          {renderContent()}
        </div>
      </div>
    </div>
  );
};