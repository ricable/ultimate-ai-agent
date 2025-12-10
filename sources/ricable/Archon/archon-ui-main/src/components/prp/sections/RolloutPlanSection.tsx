import React from 'react';
import { Calendar, CheckCircle, AlertCircle } from 'lucide-react';
import { PRPSectionProps } from '../types/prp.types';
import { formatKey, formatValue } from '../utils/formatters';
import { CollapsibleSectionWrapper } from '../components/CollapsibleSectionWrapper';

/**
 * Component for rendering rollout plans and deployment strategies
 */
export const RolloutPlanSection: React.FC<PRPSectionProps> = ({ 
  title, 
  data, 
  icon = <Calendar className="w-5 h-5" />,
  accentColor = 'orange',
  isDarkMode = false,
  defaultOpen = true,
  isCollapsible = true,
  isOpen,
  onToggle
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

  const renderPhase = (phase: any, index: number) => {
    if (typeof phase === 'string') {
      return (
        <div key={index} className="flex items-start gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-sm font-bold">
            {index + 1}
          </div>
          <div className="flex-1">
            <p className="text-gray-700 dark:text-gray-300">{phase}</p>
          </div>
        </div>
      );
    }

    if (typeof phase === 'object' && phase !== null) {
      const phaseName = phase.name || phase.title || phase.phase || `Phase ${index + 1}`;
      const duration = phase.duration || phase.timeline || phase.timeframe;
      const description = phase.description || phase.details || phase.summary;
      const tasks = phase.tasks || phase.activities || phase.items;
      const risks = phase.risks || phase.considerations;

      return (
        <div key={index} className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 ml-4">
          <div className="flex items-start gap-3 mb-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 text-white flex items-center justify-center font-bold shadow-md">
              {index + 1}
            </div>
            <div className="flex-1">
              <h4 className="font-bold text-gray-800 dark:text-white text-lg">{phaseName}</h4>
              {duration && (
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{duration}</p>
              )}
            </div>
          </div>

          {description && (
            <p className="text-gray-700 dark:text-gray-300 mb-3 ml-13">{description}</p>
          )}

          {tasks && Array.isArray(tasks) && tasks.length > 0 && (
            <div className="ml-13 mb-3">
              <p className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">Tasks:</p>
              <ul className="space-y-1">
                {tasks.map((task, taskIndex) => (
                  <li key={taskIndex} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-700 dark:text-gray-300">{formatValue(task)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {risks && Array.isArray(risks) && risks.length > 0 && (
            <div className="ml-13">
              <p className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">Risks & Considerations:</p>
              <ul className="space-y-1">
                {risks.map((risk, riskIndex) => (
                  <li key={riskIndex} className="flex items-start gap-2 text-sm">
                    <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-700 dark:text-gray-300">{formatValue(risk)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Render any other properties */}
          {Object.entries(phase).map(([key, value]) => {
            if (['name', 'title', 'phase', 'duration', 'timeline', 'timeframe', 'description', 'details', 'summary', 'tasks', 'activities', 'items', 'risks', 'considerations'].includes(key)) {
              return null;
            }
            
            return (
              <div key={key} className="ml-13 mt-3">
                <p className="text-sm font-semibold text-gray-600 dark:text-gray-400">{formatKey(key)}:</p>
                <div className="mt-1 text-sm text-gray-700 dark:text-gray-300">
                  {typeof value === 'string' || typeof value === 'number' ? (
                    <span>{value}</span>
                  ) : Array.isArray(value) ? (
                    <ul className="space-y-1 mt-1">
                      {value.map((item, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-gray-400">•</span>
                          <span>{formatValue(item)}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <pre className="mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-x-auto">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      );
    }

    return null;
  };

  const renderRolloutPlan = () => {
    // Handle array of phases
    if (Array.isArray(data)) {
      return (
        <div className="space-y-6">
          {data.map((phase, index) => renderPhase(phase, index))}
        </div>
      );
    }

    // Handle object with phases
    if (typeof data === 'object' && data !== null) {
      const phases = data.phases || data.plan || data.steps || data.stages;
      
      if (phases && Array.isArray(phases)) {
        return (
          <div className="space-y-6">
            {phases.map((phase, index) => renderPhase(phase, index))}
          </div>
        );
      }

      // Handle object with other properties
      return (
        <div className="space-y-4">
          {Object.entries(data).map(([key, value]) => (
            <div key={key}>
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">
                {formatKey(key)}
              </h4>
              {Array.isArray(value) ? (
                <div className="space-y-4">
                  {value.map((item, index) => renderPhase(item, index))}
                </div>
              ) : typeof value === 'object' && value !== null ? (
                <div className="pl-4 border-l-2 border-gray-200 dark:border-gray-700">
                  {renderPhase(value, 0)}
                </div>
              ) : (
                <p className="text-gray-700 dark:text-gray-300">{formatValue(value)}</p>
              )}
            </div>
          ))}
        </div>
      );
    }

    // Handle string
    if (typeof data === 'string') {
      return <p className="text-gray-700 dark:text-gray-300">{data}</p>;
    }

    return null;
  };

  const header = (
    <div className={`rounded-lg p-6 ${bgColorMap[accentColor as keyof typeof bgColorMap] || bgColorMap.orange} border-l-4 ${colorMap[accentColor as keyof typeof colorMap].split(' ')[2]}`}>
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
    <div className={`rounded-b-lg px-6 pb-6 -mt-1 ${bgColorMap[accentColor as keyof typeof bgColorMap] || bgColorMap.orange} border-l-4 ${colorMap[accentColor as keyof typeof colorMap].split(' ')[2]}`}>
      {renderRolloutPlan()}
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