import React from 'react';
import { CheckCircle2, Circle } from 'lucide-react';
import { SectionProps } from '../types/prp.types';

/**
 * Renders simple list/array data
 */
export const ListSection: React.FC<SectionProps> = ({
  title,
  data,
  icon,
  accentColor = 'green',
  defaultOpen = true,
  isDarkMode = false,
}) => {
  if (!Array.isArray(data)) return null;
  
  const getItemIcon = (item: any, index: number) => {
    // Use checkmarks for validation/success items
    if (title.toLowerCase().includes('validation') || 
        title.toLowerCase().includes('success') ||
        title.toLowerCase().includes('complete')) {
      return <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />;
    }
    // Use circles for general items
    return <Circle className="w-3 h-3 text-gray-400 mt-1 flex-shrink-0" />;
  };
  
  const getBackgroundColor = () => {
    const colorMap = {
      green: 'bg-gradient-to-br from-green-50/50 to-emerald-50/50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-200 dark:border-green-800',
      blue: 'bg-gradient-to-br from-blue-50/50 to-cyan-50/50 dark:from-blue-900/20 dark:to-cyan-900/20 border-blue-200 dark:border-blue-800',
      purple: 'bg-gradient-to-br from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 border-purple-200 dark:border-purple-800',
      orange: 'bg-gradient-to-br from-orange-50/50 to-yellow-50/50 dark:from-orange-900/20 dark:to-yellow-900/20 border-orange-200 dark:border-orange-800',
      gray: 'bg-gradient-to-br from-gray-50/50 to-slate-50/50 dark:from-gray-900/20 dark:to-slate-900/20 border-gray-200 dark:border-gray-800',
    };
    return colorMap[accentColor as keyof typeof colorMap] || colorMap.gray;
  };
  
  if (data.length === 0) {
    return (
      <div className={`p-4 rounded-lg border ${getBackgroundColor()}`}>
        <p className="text-gray-500 dark:text-gray-500 italic">No items</p>
      </div>
    );
  }
  
  return (
    <div className={`p-4 rounded-lg border ${getBackgroundColor()}`}>
      <ul className="space-y-2">
        {data.map((item: any, idx: number) => (
          <li key={idx} className="flex items-start gap-3 p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            {getItemIcon(item, idx)}
            <div className="flex-1">
              {typeof item === 'string' ? (
                <span className="text-gray-700 dark:text-gray-300">{item}</span>
              ) : typeof item === 'object' && item !== null ? (
                <div className="space-y-2">
                  {Object.entries(item).map(([key, value]) => (
                    <div key={key} className="flex items-start gap-2">
                      <span className="font-medium text-gray-600 dark:text-gray-400 min-w-[80px] capitalize">
                        {key.replace(/_/g, ' ')}:
                      </span>
                      <span className="text-gray-700 dark:text-gray-300 flex-1">
                        {typeof value === 'string' ? value : JSON.stringify(value)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <span className="text-gray-700 dark:text-gray-300">{String(item)}</span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};