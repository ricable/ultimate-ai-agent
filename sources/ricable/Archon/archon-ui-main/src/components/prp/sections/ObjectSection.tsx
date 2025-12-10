import React from 'react';
import { Box, FileText } from 'lucide-react';
import { PRPSectionProps } from '../types/prp.types';
import { formatKey, formatValue } from '../utils/formatters';
import { CollapsibleSectionWrapper } from '../components/CollapsibleSectionWrapper';

/**
 * Component for rendering complex object structures with nested data
 * Used for sections like design systems, architecture, etc.
 */
export const ObjectSection: React.FC<PRPSectionProps> = ({ 
  title, 
  data, 
  icon = <Box className="w-5 h-5" />,
  accentColor = 'indigo',
  isDarkMode = false,
  defaultOpen = true,
  isCollapsible = true,
  isOpen,
  onToggle
}) => {
  if (!data || typeof data !== 'object') return null;

  const colorMap = {
    blue: 'from-blue-400 to-blue-600 border-blue-500',
    purple: 'from-purple-400 to-purple-600 border-purple-500',
    green: 'from-green-400 to-green-600 border-green-500',
    orange: 'from-orange-400 to-orange-600 border-orange-500',
    pink: 'from-pink-400 to-pink-600 border-pink-500',
    cyan: 'from-cyan-400 to-cyan-600 border-cyan-500',
    indigo: 'from-indigo-400 to-indigo-600 border-indigo-500',
    emerald: 'from-emerald-400 to-emerald-600 border-emerald-500',
  };

  const bgColorMap = {
    blue: 'bg-blue-50 dark:bg-blue-950',
    purple: 'bg-purple-50 dark:bg-purple-950',
    green: 'bg-green-50 dark:bg-green-950',
    orange: 'bg-orange-50 dark:bg-orange-950',
    pink: 'bg-pink-50 dark:bg-pink-950',
    cyan: 'bg-cyan-50 dark:bg-cyan-950',
    indigo: 'bg-indigo-50 dark:bg-indigo-950',
    emerald: 'bg-emerald-50 dark:bg-emerald-950',
  };

  const renderNestedObject = (obj: any, depth: number = 0): React.ReactNode => {
    if (!obj || typeof obj !== 'object') {
      return <span className="text-gray-700 dark:text-gray-300">{formatValue(obj)}</span>;
    }

    if (Array.isArray(obj)) {
      // Handle empty arrays
      if (obj.length === 0) {
        return <span className="text-gray-500 italic">No items</span>;
      }

      // Check if it's a simple array (strings/numbers/booleans)
      const isSimpleArray = obj.every(item => 
        typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean'
      );

      if (isSimpleArray) {
        return (
          <ul className="space-y-1 mt-2">
            {obj.map((item, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-gray-400 mt-0.5">•</span>
                <span className="text-gray-700 dark:text-gray-300">{String(item)}</span>
              </li>
            ))}
          </ul>
        );
      }

      // Complex array with objects
      return (
        <div className="space-y-3 mt-2">
          {obj.map((item, index) => (
            <div key={index} className={`${depth > 0 ? 'border-l-2 border-gray-200 dark:border-gray-700 pl-4' : ''}`}>
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Item {index + 1}</div>
              {renderNestedObject(item, depth + 1)}
            </div>
          ))}
        </div>
      );
    }

    // Handle objects
    const entries = Object.entries(obj);
    
    // Group entries by type for better organization
    const stringEntries = entries.filter(([_, v]) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean');
    const arrayEntries = entries.filter(([_, v]) => Array.isArray(v));
    const objectEntries = entries.filter(([_, v]) => typeof v === 'object' && v !== null && !Array.isArray(v));

    return (
      <div className={`space-y-3 ${depth > 0 ? 'mt-2' : ''}`}>
        {/* Render simple key-value pairs first */}
        {stringEntries.length > 0 && (
          <div className={`${depth > 0 ? 'ml-4' : ''} space-y-2`}>
            {stringEntries.map(([key, value]) => (
              <div key={key} className="flex items-start gap-2">
                <span className="text-gray-600 dark:text-gray-400 min-w-[100px] text-sm">
                  {formatKey(key)}:
                </span>
                <span className="text-gray-700 dark:text-gray-300 text-sm">
                  {String(value)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Render arrays */}
        {arrayEntries.map(([key, value]) => (
          <div key={key} className={`${depth > 0 ? 'ml-4' : ''}`}>
            <div className="flex items-start gap-2 mb-2">
              <FileText className="w-4 h-4 text-gray-400 mt-1 flex-shrink-0" />
              <div className="flex-1">
                <h5 className={`font-semibold text-gray-700 dark:text-gray-300 ${depth > 2 ? 'text-sm' : ''}`}>
                  {formatKey(key)}
                </h5>
                <div className="text-sm">
                  {renderNestedObject(value, depth + 1)}
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Render nested objects */}
        {objectEntries.map(([key, value]) => {
          // Determine if this is a complex nested structure
          const isComplex = Object.values(value as object).some(v => 
            typeof v === 'object' && v !== null
          );

          return (
            <div key={key} className={`${depth > 0 ? 'ml-4' : ''}`}>
              <div className={`
                ${isComplex ? 'border-l-4 border-gray-300 dark:border-gray-600 pl-4' : ''}
                ${depth > 1 ? 'mt-4' : ''}
              `}>
                <h5 className={`
                  font-semibold text-gray-700 dark:text-gray-300 mb-2
                  ${depth === 0 ? 'text-base' : depth === 1 ? 'text-sm' : 'text-xs'}
                `}>
                  {formatKey(key)}
                </h5>
                <div className={depth > 2 ? 'text-xs' : 'text-sm'}>
                  {renderNestedObject(value, depth + 1)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const header = (
    <div className={`rounded-lg p-6 ${bgColorMap[accentColor as keyof typeof bgColorMap] || bgColorMap.indigo} border-l-4 ${colorMap[accentColor as keyof typeof colorMap].split(' ')[2]}`}>
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg bg-gradient-to-br ${colorMap[accentColor as keyof typeof colorMap].split(' ').slice(0, 2).join(' ')} text-white shadow-lg`}>
          {icon}
        </div>
        <h3 className="text-lg font-bold text-gray-800 dark:text-white flex-1">
          {title}
        </h3>
      </div>
    </div>
  );

  const content = (
    <div className={`rounded-b-lg px-6 pb-6 -mt-1 ${bgColorMap[accentColor as keyof typeof bgColorMap] || bgColorMap.indigo} border-l-4 ${colorMap[accentColor as keyof typeof colorMap].split(' ')[2]}`}>
      {renderNestedObject(data)}
    </div>
  );

  return (
    <div className="space-y-0">
      <CollapsibleSectionWrapper
        header={header}
        isCollapsible={isCollapsible}
        defaultOpen={defaultOpen}
        isOpen={isOpen}
        onToggle={onToggle}
      >
        {content}
      </CollapsibleSectionWrapper>
    </div>
  );
};