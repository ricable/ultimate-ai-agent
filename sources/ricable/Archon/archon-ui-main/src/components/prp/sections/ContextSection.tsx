import React from 'react';
import { Target, BookOpen, Sparkles, CheckCircle2 } from 'lucide-react';
import { SectionProps } from '../types/prp.types';
// Temporarily disabled to debug black screen issue
// import { renderValue, renderValueInline } from '../utils/objectRenderer';

/**
 * Renders context sections like scope, background, objectives
 */
export const ContextSection: React.FC<SectionProps> = ({
  title,
  data,
  icon,
  accentColor = 'blue',
  defaultOpen = true,
  isDarkMode = false,
}) => {
  if (!data || typeof data !== 'object') return null;
  
  const renderContextItem = (key: string, value: any) => {
    const getItemIcon = (itemKey: string) => {
      const normalizedKey = itemKey.toLowerCase();
      if (normalizedKey.includes('scope')) return <Target className="w-4 h-4 text-blue-500" />;
      if (normalizedKey.includes('background')) return <BookOpen className="w-4 h-4 text-purple-500" />;
      if (normalizedKey.includes('objective')) return <Sparkles className="w-4 h-4 text-green-500" />;
      if (normalizedKey.includes('requirement')) return <CheckCircle2 className="w-4 h-4 text-orange-500" />;
      return <CheckCircle2 className="w-4 h-4 text-gray-500" />;
    };
    
    const getItemColor = (itemKey: string) => {
      const normalizedKey = itemKey.toLowerCase();
      if (normalizedKey.includes('scope')) return 'blue';
      if (normalizedKey.includes('background')) return 'purple';
      if (normalizedKey.includes('objective')) return 'green';
      if (normalizedKey.includes('requirement')) return 'orange';
      return 'gray';
    };
    
    const color = getItemColor(key);
    const colorMap = {
      blue: 'bg-blue-50/50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      purple: 'bg-purple-50/50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
      green: 'bg-green-50/50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      orange: 'bg-orange-50/50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
      gray: 'bg-gray-50/50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800',
    };
    
    const itemTitle = key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1);
    
    return (
      <div key={key} className={`p-4 rounded-lg border ${colorMap[color as keyof typeof colorMap]}`}>
        <h4 className="font-semibold text-gray-800 dark:text-white mb-2 flex items-center gap-2">
          {getItemIcon(key)}
          {itemTitle}
        </h4>
        
        {Array.isArray(value) ? (
          <ul className="space-y-2">
            {value.map((item: any, idx: number) => (
              <li key={idx} className="flex items-start gap-2 text-gray-700 dark:text-gray-300">
                <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                {typeof item === 'string' ? item : JSON.stringify(item)}
              </li>
            ))}
          </ul>
        ) : typeof value === 'string' ? (
          <p className="text-gray-700 dark:text-gray-300">{value}</p>
        ) : (
          <div className="text-gray-700 dark:text-gray-300">
            {JSON.stringify(value, null, 2)}
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div className="space-y-4">
      {Object.entries(data).map(([key, value]) => renderContextItem(key, value))}
    </div>
  );
};