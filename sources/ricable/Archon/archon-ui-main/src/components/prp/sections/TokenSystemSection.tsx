import React from 'react';
import { Palette, Layers } from 'lucide-react';
import { PRPSectionProps } from '../types/prp.types';
import { formatKey, formatValue } from '../utils/formatters';

/**
 * Component for rendering design token systems and style guides
 */
export const TokenSystemSection: React.FC<PRPSectionProps> = ({ 
  title, 
  data, 
  icon = <Palette className="w-5 h-5" />,
  accentColor = 'indigo',
  isDarkMode = false,
  defaultOpen = true 
}) => {
  if (!data) return null;

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

  const renderColorSwatch = (color: string, name: string) => {
    // Check if it's a valid color value
    const isHex = /^#[0-9A-F]{6}$/i.test(color);
    const isRgb = /^rgb/.test(color);
    const isHsl = /^hsl/.test(color);
    const isNamedColor = /^[a-z]+$/i.test(color);
    
    if (isHex || isRgb || isHsl || isNamedColor) {
      return (
        <div className="flex items-center gap-3">
          <div 
            className="w-12 h-12 rounded-lg border-2 border-gray-300 dark:border-gray-600 shadow-sm"
            style={{ backgroundColor: color }}
          />
          <div>
            <p className="font-medium text-gray-700 dark:text-gray-300">{name}</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">{color}</p>
          </div>
        </div>
      );
    }
    
    return null;
  };

  const renderSpacingValue = (value: string | number, name: string) => {
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    const unit = typeof value === 'string' ? value.replace(/[0-9.-]/g, '') : 'px';
    
    return (
      <div className="flex items-center gap-3">
        <div 
          className="bg-indigo-500 rounded"
          style={{ 
            width: `${Math.min(numValue * 2, 100)}px`,
            height: '24px'
          }}
        />
        <div>
          <p className="font-medium text-gray-700 dark:text-gray-300">{name}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">{value}{unit}</p>
        </div>
      </div>
    );
  };

  const renderTokenGroup = (tokens: any, groupName: string) => {
    if (!tokens || typeof tokens !== 'object') return null;

    const entries = Object.entries(tokens);
    const isColorGroup = groupName.toLowerCase().includes('color') || 
                        entries.some(([_, v]) => typeof v === 'string' && (v.startsWith('#') || v.startsWith('rgb')));
    const isSpacingGroup = groupName.toLowerCase().includes('spacing') || 
                          groupName.toLowerCase().includes('size') ||
                          groupName.toLowerCase().includes('radius');

    return (
      <div className="space-y-3">
        <h4 className="font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
          <Layers className="w-4 h-4" />
          {formatKey(groupName)}
        </h4>
        <div className={`grid gap-4 ${isColorGroup ? 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3' : 'grid-cols-1'}`}>
          {entries.map(([key, value]) => {
            if (isColorGroup && typeof value === 'string') {
              const swatch = renderColorSwatch(value, formatKey(key));
              if (swatch) return <div key={key}>{swatch}</div>;
            }
            
            if (isSpacingGroup && (typeof value === 'string' || typeof value === 'number')) {
              return <div key={key}>{renderSpacingValue(value, formatKey(key))}</div>;
            }

            // Handle nested token groups
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
              return (
                <div key={key} className="col-span-full">
                  {renderTokenGroup(value, key)}
                </div>
              );
            }

            // Default rendering
            return (
              <div key={key} className="flex items-start gap-2">
                <span className="font-medium text-gray-600 dark:text-gray-400">{formatKey(key)}:</span>
                <span className="text-gray-700 dark:text-gray-300">{formatValue(value)}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderTokenSystem = () => {
    // Handle string description
    if (typeof data === 'string') {
      return <p className="text-gray-700 dark:text-gray-300">{data}</p>;
    }

    // Handle array of token groups
    if (Array.isArray(data)) {
      return (
        <div className="space-y-6">
          {data.map((group, index) => (
            <div key={index}>
              {typeof group === 'object' && group !== null ? (
                renderTokenGroup(group, `Group ${index + 1}`)
              ) : (
                <p className="text-gray-700 dark:text-gray-300">{formatValue(group)}</p>
              )}
            </div>
          ))}
        </div>
      );
    }

    // Handle object with token categories
    if (typeof data === 'object' && data !== null) {
      const categories = Object.entries(data);
      
      // Special handling for common token categories
      const colorTokens = categories.filter(([k]) => k.toLowerCase().includes('color'));
      const typographyTokens = categories.filter(([k]) => k.toLowerCase().includes('typography') || k.toLowerCase().includes('font'));
      const spacingTokens = categories.filter(([k]) => k.toLowerCase().includes('spacing') || k.toLowerCase().includes('size'));
      const otherTokens = categories.filter(([k]) => 
        !k.toLowerCase().includes('color') && 
        !k.toLowerCase().includes('typography') && 
        !k.toLowerCase().includes('font') &&
        !k.toLowerCase().includes('spacing') &&
        !k.toLowerCase().includes('size')
      );

      return (
        <div className="space-y-8">
          {/* Colors */}
          {colorTokens.length > 0 && (
            <div className="space-y-6">
              {colorTokens.map(([key, value]) => (
                <div key={key}>{renderTokenGroup(value, key)}</div>
              ))}
            </div>
          )}

          {/* Typography */}
          {typographyTokens.length > 0 && (
            <div className="space-y-6">
              {typographyTokens.map(([key, value]) => (
                <div key={key}>{renderTokenGroup(value, key)}</div>
              ))}
            </div>
          )}

          {/* Spacing */}
          {spacingTokens.length > 0 && (
            <div className="space-y-6">
              {spacingTokens.map(([key, value]) => (
                <div key={key}>{renderTokenGroup(value, key)}</div>
              ))}
            </div>
          )}

          {/* Others */}
          {otherTokens.length > 0 && (
            <div className="space-y-6">
              {otherTokens.map(([key, value]) => (
                <div key={key}>{renderTokenGroup(value, key)}</div>
              ))}
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="space-y-4">
      <div className={`rounded-lg p-6 ${bgColorMap[accentColor as keyof typeof bgColorMap] || bgColorMap.indigo} border-l-4 ${colorMap[accentColor as keyof typeof colorMap].split(' ')[2]}`}>
        <div className="flex items-center gap-3 mb-4">
          <div className={`p-2 rounded-lg bg-gradient-to-br ${colorMap[accentColor as keyof typeof colorMap].split(' ').slice(0, 2).join(' ')} text-white shadow-lg`}>
            {icon}
          </div>
          <h3 className="text-lg font-bold text-gray-800 dark:text-white">
            {title}
          </h3>
        </div>
        
        {renderTokenSystem()}
      </div>
    </div>
  );
};