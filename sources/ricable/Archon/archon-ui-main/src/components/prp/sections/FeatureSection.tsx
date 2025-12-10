import React from 'react';
import { Package, Star, FileText } from 'lucide-react';
import { PRPSectionProps } from '../types/prp.types';
import { formatKey, formatValue } from '../utils/formatters';

/**
 * Specialized component for feature requirements and capabilities
 * Renders features in organized categories with proper hierarchy
 */
export const FeatureSection: React.FC<PRPSectionProps> = ({ 
  title, 
  data, 
  icon = <Package className="w-5 h-5" />,
  accentColor = 'blue',
  isDarkMode = false,
  defaultOpen = true 
}) => {
  if (!data || typeof data !== 'object') return null;

  const colorMap = {
    blue: 'from-blue-400 to-blue-600',
    purple: 'from-purple-400 to-purple-600',
    green: 'from-green-400 to-green-600',
    orange: 'from-orange-400 to-orange-600',
    pink: 'from-pink-400 to-pink-600',
    cyan: 'from-cyan-400 to-cyan-600',
    indigo: 'from-indigo-400 to-indigo-600',
    emerald: 'from-emerald-400 to-emerald-600',
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

  const renderFeatureGroup = (groupName: string, features: any, isPremium: boolean = false) => {
    if (!features || typeof features !== 'object') return null;

    const IconComponent = isPremium ? Star : FileText;
    const iconColor = isPremium ? 'text-yellow-500' : 'text-blue-500';

    return (
      <div key={groupName} className="mb-6">
        <div className="flex items-center gap-3 mb-4">
          <IconComponent className={`w-5 h-5 ${iconColor}`} />
          <h4 className="font-semibold text-gray-800 dark:text-white text-lg">
            {formatKey(groupName)}
          </h4>
          {isPremium && (
            <span className="px-2 py-1 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 text-xs rounded-full font-medium">
              Premium
            </span>
          )}
        </div>
        
        <div className="space-y-4 ml-8">
          {Object.entries(features).map(([featureName, featureData]) => (
            <div key={featureName} className="border-l-2 border-gray-200 dark:border-gray-700 pl-4">
              <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                {formatKey(featureName)}
              </h5>
              
              {Array.isArray(featureData) ? (
                <ul className="space-y-1">
                  {featureData.map((item, index) => (
                    <li key={index} className="flex items-start gap-2 text-gray-600 dark:text-gray-400">
                      <span className="text-gray-400 mt-1">•</span>
                      <span>{formatValue(item)}</span>
                    </li>
                  ))}
                </ul>
              ) : typeof featureData === 'string' ? (
                <p className="text-gray-600 dark:text-gray-400">{featureData}</p>
              ) : (
                <div className="text-gray-600 dark:text-gray-400">
                  {formatValue(featureData)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderFeatureList = (features: any) => {
    if (Array.isArray(features)) {
      return (
        <ul className="space-y-2">
          {features.map((feature, index) => (
            <li key={index} className="flex items-start gap-3 p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <Package className="w-4 h-4 text-blue-500 mt-1 flex-shrink-0" />
              <span className="text-gray-700 dark:text-gray-300">{formatValue(feature)}</span>
            </li>
          ))}
        </ul>
      );
    }

    if (typeof features === 'object' && features !== null) {
      return (
        <div className="space-y-6">
          {Object.entries(features).map(([key, value]) => {
            const isPremium = key.toLowerCase().includes('premium') || 
                             key.toLowerCase().includes('advanced') ||
                             key.toLowerCase().includes('pro');
            
            if (typeof value === 'object' && value !== null) {
              return renderFeatureGroup(key, value, isPremium);
            }
            
            return (
              <div key={key} className="flex items-start gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <Package className="w-5 h-5 text-blue-500 mt-1 flex-shrink-0" />
                <div className="flex-1">
                  <h4 className="font-medium text-gray-800 dark:text-white mb-1">
                    {formatKey(key)}
                  </h4>
                  <div className="text-gray-600 dark:text-gray-400">
                    {formatValue(value)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      );
    }

    return <div className="text-gray-600 dark:text-gray-400">{formatValue(features)}</div>;
  };

  return (
    <div className="space-y-4">
      <div className={`rounded-lg p-6 ${bgColorMap[accentColor as keyof typeof bgColorMap] || bgColorMap.blue} border-l-4 border-blue-500`}>
        <div className="flex items-center gap-3 mb-6">
          <div className={`p-2 rounded-lg bg-gradient-to-br ${colorMap[accentColor as keyof typeof colorMap] || colorMap.blue} text-white shadow-lg`}>
            {icon}
          </div>
          <h3 className="text-xl font-bold text-gray-800 dark:text-white">
            {title}
          </h3>
        </div>
        
        {renderFeatureList(data)}
      </div>
    </div>
  );
};