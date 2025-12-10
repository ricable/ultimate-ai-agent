import React from 'react';
import { BarChart3, Settings, Users, Gauge } from 'lucide-react';
import { SectionProps } from '../types/prp.types';
import { formatKey } from '../utils/formatters';

/**
 * Renders success metrics and KPIs
 */
export const MetricsSection: React.FC<SectionProps> = ({
  title,
  data,
  icon,
  accentColor = 'green',
  defaultOpen = true,
  isDarkMode = false,
}) => {
  if (!data || typeof data !== 'object') return null;
  
  const getCategoryColor = (category: string): string => {
    const normalizedCategory = category.toLowerCase();
    if (normalizedCategory.includes('admin')) return 'from-blue-400 to-blue-600';
    if (normalizedCategory.includes('business')) return 'from-purple-400 to-purple-600';
    if (normalizedCategory.includes('customer')) return 'from-green-400 to-green-600';
    if (normalizedCategory.includes('technical')) return 'from-orange-400 to-orange-600';
    if (normalizedCategory.includes('performance')) return 'from-red-400 to-red-600';
    return 'from-gray-400 to-gray-600';
  };
  
  const getCategoryIcon = (category: string): React.ReactNode => {
    const normalizedCategory = category.toLowerCase();
    if (normalizedCategory.includes('admin')) return <Settings className="w-4 h-4" />;
    if (normalizedCategory.includes('business')) return <BarChart3 className="w-4 h-4" />;
    if (normalizedCategory.includes('customer')) return <Users className="w-4 h-4" />;
    return <Gauge className="w-4 h-4" />;
  };
  
  const renderMetric = (metric: string, category: string, index: number) => {
    return (
      <div 
        key={`${category}-${index}`}
        className="flex items-center gap-3 p-3 rounded-lg bg-white/50 dark:bg-black/30 border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 transition-all duration-200 group"
      >
        <div className={`p-2 rounded-lg bg-gradient-to-br ${getCategoryColor(category)} text-white shadow-md group-hover:scale-110 transition-transform duration-200`}>
          {getCategoryIcon(category)}
        </div>
        <p className="text-sm text-gray-700 dark:text-gray-300 flex-1">{metric}</p>
      </div>
    );
  };
  
  return (
    <div className="grid gap-4">
      {Object.entries(data).map(([category, metrics]: [string, any]) => (
        <div key={category}>
          <h4 className="font-semibold text-gray-800 dark:text-white mb-3 capitalize">
            {formatKey(category)}
          </h4>
          <div className="grid gap-2">
            {Array.isArray(metrics) ? 
              metrics.map((metric: string, idx: number) => 
                renderMetric(metric, category, idx)
              ) :
              typeof metrics === 'object' && metrics !== null ?
                Object.entries(metrics).map(([key, value], idx) => 
                  renderMetric(`${formatKey(key)}: ${value}`, category, idx)
                ) :
                renderMetric(String(metrics), category, 0)
            }
          </div>
        </div>
      ))}
    </div>
  );
};