import React from 'react';
import { formatKey, formatValue } from './formatters';

/**
 * Renders any value in a formatted way without using JSON.stringify
 */
export function renderValue(value: any, depth: number = 0): React.ReactNode {
  try {
    // Prevent infinite recursion
    if (depth > 10) {
      return <span className="text-gray-500 italic">Too deeply nested</span>;
    }

    // Handle null/undefined
    if (value === null || value === undefined) {
      return <span className="text-gray-400 italic">Empty</span>;
    }

  // Handle primitives
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return <span className="text-gray-700 dark:text-gray-300">{formatValue(value)}</span>;
  }

  // Handle arrays
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-gray-400 italic">No items</span>;
    }

    // Check if it's a simple array
    const isSimple = value.every(item => 
      typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean'
    );

    if (isSimple) {
      return (
        <ul className="list-disc list-inside space-y-1">
          {value.map((item, index) => (
            <li key={index} className="text-gray-700 dark:text-gray-300">
              {formatValue(item)}
            </li>
          ))}
        </ul>
      );
    }

    // Complex array
    return (
      <div className="space-y-2">
        {value.map((item, index) => (
          <div key={index} className="pl-4 border-l-2 border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Item {index + 1}</div>
            {renderValue(item, depth + 1)}
          </div>
        ))}
      </div>
    );
  }

  // Handle objects
  if (typeof value === 'object' && value !== null) {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return <span className="text-gray-400 italic">No properties</span>;
    }

    return (
      <div className="space-y-2">
        {entries.map(([key, val]) => (
          <div key={key} className="flex flex-col gap-1">
            <span className="font-medium text-gray-600 dark:text-gray-400">
              {formatKey(key)}:
            </span>
            <div className="pl-4">
              {renderValue(val, depth + 1)}
            </div>
          </div>
        ))}
      </div>
    );
  }

    // Fallback
    return <span className="text-gray-700 dark:text-gray-300">{String(value)}</span>;
  } catch (error) {
    console.error('Error rendering value:', error, value);
    return <span className="text-red-500 italic">Error rendering content</span>;
  }
}

/**
 * Renders a value inline for simple display
 */
export function renderValueInline(value: any): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return formatValue(value);
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) return value.map(v => renderValueInline(v)).join(', ');
  if (typeof value === 'object') {
    // For objects, just show a summary
    const keys = Object.keys(value);
    if (keys.length === 0) return 'Empty object';
    if (keys.length <= 3) return keys.map(k => `${k}: ${renderValueInline(value[k])}`).join(', ');
    return `${keys.length} properties`;
  }
  return String(value);
}