// File: frontend/src/components/collaboration/AdaptiveInterfacePanel.tsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface AdaptiveInterfacePanelProps {
  adaptations: any[];
  cognitiveLoad: any;
  trustStatus: any;
  onApplyAdaptation: (adaptation: any) => void;
}

export function AdaptiveInterfacePanel({ 
  adaptations, 
  cognitiveLoad, 
  trustStatus, 
  onApplyAdaptation 
}: AdaptiveInterfacePanelProps) {
  const [appliedAdaptations, setAppliedAdaptations] = useState<Set<string>>(new Set());
  const [previewMode, setPreviewMode] = useState<string | null>(null);
  const [adaptationHistory, setAdaptationHistory] = useState<any[]>([]);

  useEffect(() => {
    // Store adaptation history for analysis
    if (adaptations.length > 0) {
      setAdaptationHistory(prev => [...prev, ...adaptations.filter(
        adaptation => !prev.some(existing => existing.adaptation_id === adaptation.adaptation_id)
      )]);
    }
  }, [adaptations]);

  const getAdaptationPriorityColor = (priority: string) => {
    switch (priority) {
      case 'immediate': return 'bg-red-100 text-red-800 border-red-300';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'low': return 'bg-blue-100 text-blue-800 border-blue-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getAdaptationTypeIcon = (type: string) => {
    switch (type) {
      case 'load_reduction': return '🧘';
      case 'emergency_simplification': return '🚨';
      case 'engagement_enhancement': return '🚀';
      case 'immediate_relief': return '💊';
      case 'long_term_support': return '🏗️';
      case 'attention_focus': return '🎯';
      case 'performance_support': return '⚡';
      case 'trust_building': return '🤝';
      default: return '🔧';
    }
  };

  const getTriggerDescription = (trigger: string) => {
    const descriptions: { [key: string]: string } = {
      'high_load_detected': 'High cognitive load detected',
      'overload_detected': 'Cognitive overload detected',
      'low_load_detected': 'Low engagement detected',
      'load_spike': 'Sudden load increase',
      'sustained_high_load': 'Prolonged high load',
      'attention_fragmentation': 'Attention scattered',
      'performance_degradation': 'Performance declining',
      'low_trust': 'Trust level low',
      'trust_decline': 'Trust declining'
    };
    return descriptions[trigger] || trigger.replace(/_/g, ' ');
  };

  const handleApplyAdaptation = async (adaptation: any) => {
    try {
      await onApplyAdaptation(adaptation);
      setAppliedAdaptations(prev => new Set([...prev, adaptation.adaptation_id]));
    } catch (error) {
      console.error('Failed to apply adaptation:', error);
    }
  };

  const generateContextualAdaptations = () => {
    const contextualAdaptations = [];

    // Generate adaptations based on cognitive load
    if (cognitiveLoad?.current_status?.load_level === 'high' || 
        cognitiveLoad?.current_status?.load_level === 'overload') {
      contextualAdaptations.push({
        adaptation_id: 'load_reduction_suggestion',
        adaptation_type: 'load_reduction',
        priority: 'high',
        trigger: 'high_load_detected',
        recommended_actions: [
          'Reduce information density',
          'Simplify interface elements',
          'Break tasks into smaller steps',
          'Provide clear progress indicators'
        ],
        target_load_reduction: 0.3,
        estimated_effectiveness: 0.75
      });
    }

    // Generate adaptations based on trust level
    if (trustStatus?.trust_level === 'low' || trustStatus?.overall_trust < 0.4) {
      contextualAdaptations.push({
        adaptation_id: 'trust_building_suggestion',
        adaptation_type: 'trust_building',
        priority: 'high',
        trigger: 'low_trust',
        recommended_actions: [
          'Provide more explanations',
          'Show confidence scores',
          'Offer alternative approaches',
          'Increase transparency'
        ],
        target_trust_increase: 0.2,
        estimated_effectiveness: 0.6
      });
    }

    return contextualAdaptations;
  };

  const contextualAdaptations = generateContextualAdaptations();
  const allAdaptations = [...adaptations, ...contextualAdaptations];
  const activeAdaptations = allAdaptations.filter(adaptation => 
    adaptation.priority === 'immediate' || adaptation.priority === 'high'
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Adaptive Interface</h3>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">
              {activeAdaptations.length} active
            </span>
            <button
              onClick={() => setPreviewMode(previewMode ? null : 'preview')}
              className={`text-xs px-2 py-1 rounded ${
                previewMode 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {previewMode ? 'Exit Preview' : 'Preview Mode'}
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          
          {/* Adaptation Status Overview */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-red-50 p-2 rounded text-center">
              <div className="font-medium text-red-700">Immediate</div>
              <div className="text-red-600 text-lg font-bold">
                {allAdaptations.filter(a => a.priority === 'immediate').length}
              </div>
            </div>
            <div className="bg-orange-50 p-2 rounded text-center">
              <div className="font-medium text-orange-700">High Priority</div>
              <div className="text-orange-600 text-lg font-bold">
                {allAdaptations.filter(a => a.priority === 'high').length}
              </div>
            </div>
            <div className="bg-green-50 p-2 rounded text-center">
              <div className="font-medium text-green-700">Applied</div>
              <div className="text-green-600 text-lg font-bold">
                {appliedAdaptations.size}
              </div>
            </div>
          </div>

          {/* Active Adaptations */}
          {activeAdaptations.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">
                🚨 Active Adaptations Needed
              </h4>
              {activeAdaptations.map((adaptation, index) => (
                <div 
                  key={adaptation.adaptation_id || index} 
                  className={`border rounded-lg p-3 ${getAdaptationPriorityColor(adaptation.priority)}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {getAdaptationTypeIcon(adaptation.adaptation_type)}
                      </span>
                      <span className="font-medium capitalize">
                        {adaptation.adaptation_type.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs px-2 py-1 bg-white bg-opacity-50 rounded">
                        {adaptation.priority}
                      </span>
                      {!appliedAdaptations.has(adaptation.adaptation_id) && (
                        <button
                          onClick={() => handleApplyAdaptation(adaptation)}
                          className="text-xs bg-white bg-opacity-80 hover:bg-opacity-100 px-3 py-1 rounded font-medium"
                        >
                          Apply
                        </button>
                      )}
                      {appliedAdaptations.has(adaptation.adaptation_id) && (
                        <span className="text-xs bg-green-500 text-white px-2 py-1 rounded">
                          ✓ Applied
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-xs mb-2">
                    <strong>Trigger:</strong> {getTriggerDescription(adaptation.trigger)}
                  </div>
                  
                  {adaptation.target_load_reduction && (
                    <div className="text-xs mb-2">
                      <strong>Expected benefit:</strong> -{(adaptation.target_load_reduction * 100).toFixed(0)}% cognitive load
                    </div>
                  )}

                  {adaptation.target_trust_increase && (
                    <div className="text-xs mb-2">
                      <strong>Expected benefit:</strong> +{(adaptation.target_trust_increase * 100).toFixed(0)}% trust
                    </div>
                  )}
                  
                  {adaptation.recommended_actions && (
                    <div className="space-y-1">
                      <strong className="text-xs">Actions:</strong>
                      <ul className="text-xs space-y-1">
                        {adaptation.recommended_actions.slice(0, 3).map((action: string, actionIndex: number) => (
                          <li key={actionIndex} className="flex items-center space-x-1">
                            <span>•</span>
                            <span>{action}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <div className="flex justify-between items-center mt-2 text-xs">
                    <span>
                      Effectiveness: {((adaptation.estimated_effectiveness || 0.5) * 100).toFixed(0)}%
                    </span>
                    <span>
                      {adaptation.created_at && 
                        `Suggested ${Math.round((Date.now() - new Date(adaptation.created_at).getTime()) / 60000)}m ago`
                      }
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Interface Adaptations Preview */}
          {previewMode && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">🎨 Interface Preview</h4>
              <div className="border rounded-lg p-4 bg-blue-50">
                <div className="text-xs text-blue-800 mb-2">
                  Preview Mode: Adaptations will modify this interface
                </div>
                
                {/* Simulated adapted interface elements */}
                <div className="space-y-2">
                  {activeAdaptations.some(a => a.recommended_actions?.includes('Reduce information density')) && (
                    <div className="bg-white p-2 rounded border-l-4 border-blue-400">
                      <div className="text-sm font-medium">📊 Simplified Data Display</div>
                      <div className="text-xs text-gray-600">Information density reduced by 40%</div>
                    </div>
                  )}
                  
                  {activeAdaptations.some(a => a.recommended_actions?.includes('Break tasks into smaller steps')) && (
                    <div className="bg-white p-2 rounded border-l-4 border-green-400">
                      <div className="text-sm font-medium">📝 Step-by-Step Guide</div>
                      <div className="text-xs text-gray-600">Complex tasks broken into manageable steps</div>
                    </div>
                  )}
                  
                  {activeAdaptations.some(a => a.recommended_actions?.includes('Provide more explanations')) && (
                    <div className="bg-white p-2 rounded border-l-4 border-purple-400">
                      <div className="text-sm font-medium">💡 Enhanced Explanations</div>
                      <div className="text-xs text-gray-600">Additional context and reasoning provided</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Medium Priority Adaptations */}
          {allAdaptations.filter(a => a.priority === 'medium').length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">
                ⚡ Medium Priority Optimizations
              </h4>
              <div className="space-y-1">
                {allAdaptations.filter(a => a.priority === 'medium').slice(0, 3).map((adaptation, index) => (
                  <div 
                    key={adaptation.adaptation_id || index}
                    className="flex items-center justify-between p-2 bg-yellow-50 rounded text-xs"
                  >
                    <div className="flex items-center space-x-2">
                      <span>{getAdaptationTypeIcon(adaptation.adaptation_type)}</span>
                      <span>{adaptation.adaptation_type.replace(/_/g, ' ')}</span>
                    </div>
                    <button
                      onClick={() => handleApplyAdaptation(adaptation)}
                      disabled={appliedAdaptations.has(adaptation.adaptation_id)}
                      className={`px-2 py-1 rounded ${
                        appliedAdaptations.has(adaptation.adaptation_id)
                          ? 'bg-green-200 text-green-700'
                          : 'bg-yellow-200 text-yellow-800 hover:bg-yellow-300'
                      }`}
                    >
                      {appliedAdaptations.has(adaptation.adaptation_id) ? '✓' : 'Apply'}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Adaptation History */}
          {adaptationHistory.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">📊 Adaptation Analytics</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Total Suggestions</div>
                  <div className="text-blue-600">{adaptationHistory.length}</div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Applied Rate</div>
                  <div className="text-green-600">
                    {adaptationHistory.length > 0 
                      ? Math.round((appliedAdaptations.size / adaptationHistory.length) * 100)
                      : 0}%
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Most Common</div>
                  <div className="text-purple-600">
                    {adaptationHistory.length > 0
                      ? (() => {
                          const counts = adaptationHistory.reduce((acc, curr) => {
                            acc[curr.adaptation_type] = (acc[curr.adaptation_type] || 0) + 1;
                            return acc;
                          }, {} as any);
                          const mostCommon = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
                          return mostCommon.replace(/_/g, ' ');
                        })()
                      : 'None'
                    }
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Avg Effectiveness</div>
                  <div className="text-orange-600">
                    {adaptationHistory.length > 0
                      ? Math.round(
                          adaptationHistory.reduce((sum, adaptation) => 
                            sum + (adaptation.estimated_effectiveness || 0.5), 0
                          ) / adaptationHistory.length * 100
                        )
                      : 0}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Quick Actions */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">⚡ Quick Adaptations</h4>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => handleApplyAdaptation({
                  adaptation_id: 'quick_simplify',
                  adaptation_type: 'interface_simplification',
                  priority: 'medium',
                  recommended_actions: ['Reduce visual complexity', 'Hide advanced options']
                })}
                className="text-xs bg-blue-100 text-blue-800 p-2 rounded hover:bg-blue-200"
              >
                🎨 Simplify Interface
              </button>
              <button
                onClick={() => handleApplyAdaptation({
                  adaptation_id: 'quick_focus',
                  adaptation_type: 'focus_enhancement',
                  priority: 'medium',
                  recommended_actions: ['Highlight important elements', 'Reduce distractions']
                })}
                className="text-xs bg-green-100 text-green-800 p-2 rounded hover:bg-green-200"
              >
                🎯 Enhance Focus
              </button>
              <button
                onClick={() => handleApplyAdaptation({
                  adaptation_id: 'quick_guide',
                  adaptation_type: 'guidance_enhancement',
                  priority: 'medium',
                  recommended_actions: ['Show step-by-step guide', 'Add contextual help']
                })}
                className="text-xs bg-purple-100 text-purple-800 p-2 rounded hover:bg-purple-200"
              >
                📋 Add Guidance
              </button>
              <button
                onClick={() => handleApplyAdaptation({
                  adaptation_id: 'quick_explain',
                  adaptation_type: 'explanation_enhancement',
                  priority: 'medium',
                  recommended_actions: ['Increase explanation detail', 'Show reasoning']
                })}
                className="text-xs bg-orange-100 text-orange-800 p-2 rounded hover:bg-orange-200"
              >
                💡 More Explanations
              </button>
            </div>
          </div>

          {/* System Status */}
          <div className="text-center pt-2 border-t">
            <div className="text-xs text-gray-500">
              Adaptive interface active | {appliedAdaptations.size} adaptations applied
            </div>
            <div className="text-xs text-gray-400">
              Last adaptation: {adaptationHistory.length > 0 
                ? `${Math.round((Date.now() - new Date(adaptationHistory[adaptationHistory.length - 1].created_at || Date.now()).getTime()) / 60000)}m ago`
                : 'None'
              }
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}