import React from 'react';
import { Clock, Zap, CheckCircle2 } from 'lucide-react';
import { SectionProps, PRPPhase } from '../types/prp.types';

/**
 * Renders implementation plans and phases
 */
export const PlanSection: React.FC<SectionProps> = ({
  title,
  data,
  icon,
  accentColor = 'orange',
  defaultOpen = true,
  isDarkMode = false,
}) => {
  if (!data || typeof data !== 'object') return null;
  
  const getPhaseColor = (index: number): string => {
    const colors = ['orange', 'yellow', 'green', 'blue', 'purple'];
    return colors[index % colors.length];
  };
  
  const renderPhase = (phaseKey: string, phase: PRPPhase, index: number) => {
    const color = getPhaseColor(index);
    const colorMap = {
      orange: 'from-orange-50/50 to-yellow-50/50 dark:from-orange-900/20 dark:to-yellow-900/20 border-orange-200 dark:border-orange-800',
      yellow: 'from-yellow-50/50 to-amber-50/50 dark:from-yellow-900/20 dark:to-amber-900/20 border-yellow-200 dark:border-yellow-800',
      green: 'from-green-50/50 to-emerald-50/50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-200 dark:border-green-800',
      blue: 'from-blue-50/50 to-cyan-50/50 dark:from-blue-900/20 dark:to-cyan-900/20 border-blue-200 dark:border-blue-800',
      purple: 'from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 border-purple-200 dark:border-purple-800',
    };
    
    return (
      <div 
        key={phaseKey}
        className={`p-4 rounded-lg bg-gradient-to-r ${colorMap[color as keyof typeof colorMap]} border`}
      >
        <h4 className="font-bold text-gray-800 dark:text-white mb-2 flex items-center gap-2">
          <Zap className="w-5 h-5 text-orange-500" />
          {phaseKey.toUpperCase()}
          {phase.duration && (
            <span className="text-sm font-normal text-gray-600 dark:text-gray-400 ml-2">
              ({phase.duration})
            </span>
          )}
        </h4>
        
        {phase.deliverables && Array.isArray(phase.deliverables) && (
          <div className="mb-3">
            <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Deliverables
            </h5>
            <ul className="space-y-1">
              {phase.deliverables.map((item: string, idx: number) => (
                <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}
        
        {phase.tasks && Array.isArray(phase.tasks) && (
          <div>
            <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Tasks
            </h5>
            <ul className="space-y-1">
              {phase.tasks.map((task: any, idx: number) => (
                <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                  <div className="w-4 h-4 rounded-full bg-gray-300 dark:bg-gray-600 mt-0.5 flex-shrink-0" />
                  {typeof task === 'string' ? task : task.description || JSON.stringify(task)}
                </li>
              ))}
            </ul>
          </div>
        )}
        
        {/* Render any other phase properties */}
        {Object.entries(phase).map(([key, value]) => {
          if (['duration', 'deliverables', 'tasks'].includes(key)) return null;
          return (
            <div key={key} className="mt-3">
              <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1 capitalize">
                {key.replace(/_/g, ' ')}
              </h5>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
              </div>
            </div>
          );
        })}
      </div>
    );
  };
  
  // Check if this is a phased plan or a general plan structure
  const isPhased = Object.values(data).some(value => 
    typeof value === 'object' && 
    value !== null && 
    (value.duration || value.deliverables || value.tasks)
  );
  
  if (isPhased) {
    return (
      <div className="space-y-4">
        {Object.entries(data).map(([phaseKey, phase], index) => 
          renderPhase(phaseKey, phase as PRPPhase, index)
        )}
      </div>
    );
  }
  
  // Fallback to generic rendering for non-phased plans
  return (
    <div className="p-4 rounded-lg bg-gradient-to-r from-orange-50/50 to-yellow-50/50 dark:from-orange-900/20 dark:to-yellow-900/20 border border-orange-200 dark:border-orange-800">
      <h4 className="font-semibold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
        <Clock className="w-5 h-5 text-orange-500" />
        {title}
      </h4>
      <div className="space-y-2">
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="text-sm">
            <span className="font-medium text-gray-700 dark:text-gray-300">
              {key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1)}:
            </span>{' '}
            <span className="text-gray-600 dark:text-gray-400">
              {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};