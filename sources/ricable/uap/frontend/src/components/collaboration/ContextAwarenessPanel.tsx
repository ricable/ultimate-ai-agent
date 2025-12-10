// File: frontend/src/components/collaboration/ContextAwarenessPanel.tsx
import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface ContextAwarenessPanelProps {
  contextData: any;
  onContextUpdate: () => void;
  onInteraction: (type: string, data: any) => void;
}

export function ContextAwarenessPanel({ contextData, onContextUpdate, onInteraction }: ContextAwarenessPanelProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);
  const [contextFilters, setContextFilters] = useState({
    temporal: true,
    cognitive: true,
    task: true,
    behavioral: true
  });

  if (!contextData) {
    return (
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">Context Awareness</h3>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <div className="animate-pulse bg-gray-200 h-4 w-3/4 mx-auto mb-2 rounded"></div>
            <div className="animate-pulse bg-gray-200 h-4 w-1/2 mx-auto rounded"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getRelevanceColor = (relevance: string) => {
    switch (relevance) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-300';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'low': return 'bg-blue-100 text-blue-800 border-blue-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getContextTypeIcon = (type: string) => {
    switch (type) {
      case 'temporal': return '⏰';
      case 'cognitive': return '🧠';
      case 'task': return '📋';
      case 'behavioral': return '👥';
      case 'environmental': return '🌍';
      case 'social': return '🤝';
      case 'emotional': return '😊';
      case 'spatial': return '📍';
      default: return '📊';
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const handleFilterChange = (filter: string) => {
    setContextFilters(prev => ({
      ...prev,
      [filter]: !prev[filter as keyof typeof prev]
    }));
  };

  const filteredFeatures = contextData.context_features 
    ? Object.entries(contextData.context_features).filter(([_, feature]: [string, any]) => 
        contextFilters[feature.feature_type as keyof typeof contextFilters]
      )
    : [];

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Context Awareness</h3>
          <button
            onClick={onContextUpdate}
            className="text-xs bg-gray-500 text-white px-2 py-1 rounded hover:bg-gray-600"
          >
            Refresh
          </button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          
          {/* Context Summary */}
          {contextData.context_summary && (
            <div className="bg-blue-50 p-3 rounded-lg border-l-4 border-blue-400">
              <h4 className="text-sm font-medium text-blue-900 mb-1">Context Summary</h4>
              <p className="text-xs text-blue-800">{contextData.context_summary}</p>
            </div>
          )}

          {/* Context Type Filters */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Context Types</h4>
            <div className="flex flex-wrap gap-2">
              {Object.entries(contextFilters).map(([type, enabled]) => (
                <button
                  key={type}
                  onClick={() => handleFilterChange(type)}
                  className={`text-xs px-2 py-1 rounded border ${
                    enabled 
                      ? 'bg-blue-100 text-blue-800 border-blue-300' 
                      : 'bg-gray-100 text-gray-600 border-gray-300'
                  }`}
                >
                  {getContextTypeIcon(type)} {type}
                </button>
              ))}
            </div>
          </div>

          {/* Temporal Context */}
          {contextData.temporal_context && contextFilters.temporal && (
            <div className="space-y-2">
              <button
                onClick={() => toggleSection('temporal')}
                className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <span>⏰ Temporal Context</span>
                <span>{expandedSection === 'temporal' ? '−' : '+'}</span>
              </button>
              
              {expandedSection === 'temporal' && (
                <div className="pl-4 space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-gray-50 p-2 rounded">
                      <div className="font-medium">Session Duration</div>
                      <div className="text-blue-600">
                        {Math.round(contextData.temporal_context.time_since_start)}m
                      </div>
                    </div>
                    <div className="bg-gray-50 p-2 rounded">
                      <div className="font-medium">Session Phase</div>
                      <div className="text-green-600 capitalize">
                        {contextData.temporal_context.session_phase}
                      </div>
                    </div>
                    <div className="bg-gray-50 p-2 rounded">
                      <div className="font-medium">Interaction Rate</div>
                      <div className="text-purple-600">
                        {contextData.temporal_context.interaction_frequency?.interactions_per_minute?.toFixed(1)}/min
                      </div>
                    </div>
                    <div className="bg-gray-50 p-2 rounded">
                      <div className="font-medium">Time Period</div>
                      <div className="text-orange-600 capitalize">
                        {contextData.temporal_context.time_of_day_context?.period || 'unknown'}
                      </div>
                    </div>
                  </div>
                  
                  {contextData.temporal_context.temporal_patterns && (
                    <div className="space-y-1">
                      <div className="text-xs font-medium text-gray-600">Detected Patterns</div>
                      {contextData.temporal_context.temporal_patterns.map((pattern: any, index: number) => (
                        <div key={index} className="text-xs bg-yellow-50 text-yellow-800 p-2 rounded">
                          {pattern.type}: {pattern.confidence ? (pattern.confidence * 100).toFixed(0) + '% confidence' : 'detected'}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Cognitive Context */}
          {contextData.cognitive_context && contextFilters.cognitive && (
            <div className="space-y-2">
              <button
                onClick={() => toggleSection('cognitive')}
                className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <span>🧠 Cognitive Context</span>
                <span>{expandedSection === 'cognitive' ? '−' : '+'}</span>
              </button>
              
              {expandedSection === 'cognitive' && (
                <div className="pl-4 space-y-2">
                  {contextData.cognitive_context.cognitive_load && (
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-gray-50 p-2 rounded">
                        <div className="font-medium">Load Level</div>
                        <div className={`capitalize ${
                          contextData.cognitive_context.cognitive_load.load_level === 'low' ? 'text-green-600' :
                          contextData.cognitive_context.cognitive_load.load_level === 'moderate' ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {contextData.cognitive_context.cognitive_load.load_level}
                        </div>
                      </div>
                      <div className="bg-gray-50 p-2 rounded">
                        <div className="font-medium">Load Trend</div>
                        <div className={`capitalize ${
                          contextData.cognitive_context.cognitive_load.load_trend === 'decreasing' ? 'text-green-600' :
                          contextData.cognitive_context.cognitive_load.load_trend === 'stable' ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {contextData.cognitive_context.cognitive_load.load_trend}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {contextData.cognitive_context.mental_model_state && (
                    <div className="space-y-1">
                      <div className="text-xs font-medium text-gray-600">Mental Model</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="bg-blue-50 p-2 rounded">
                          <div className="font-medium">Domain Knowledge</div>
                          <div className="text-blue-600">
                            {(contextData.cognitive_context.mental_model_state.domain_knowledge * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-blue-50 p-2 rounded">
                          <div className="font-medium">Understanding</div>
                          <div className="text-blue-600">
                            {(contextData.cognitive_context.mental_model_state.conceptual_understanding * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {contextData.cognitive_context.cognitive_style_indicators && (
                    <div className="space-y-1">
                      <div className="text-xs font-medium text-gray-600">Cognitive Style Indicators</div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(contextData.cognitive_context.cognitive_style_indicators).map(([style, score]: [string, any]) => (
                          <span
                            key={style}
                            className={`text-xs px-2 py-1 rounded ${
                              score > 0.3 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                            }`}
                          >
                            {style}: {(score * 100).toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Context Features */}
          {filteredFeatures.length > 0 && (
            <div className="space-y-2">
              <button
                onClick={() => toggleSection('features')}
                className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <span>📊 Context Features ({filteredFeatures.length})</span>
                <span>{expandedSection === 'features' ? '−' : '+'}</span>
              </button>
              
              {expandedSection === 'features' && (
                <div className="pl-4 space-y-2">
                  {filteredFeatures.slice(0, 5).map(([featureId, feature]: [string, any]) => (
                    <div key={featureId} className={`text-xs p-2 rounded border ${getRelevanceColor(feature.relevance)}`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium flex items-center space-x-1">
                          <span>{getContextTypeIcon(feature.feature_type)}</span>
                          <span>{featureId.replace(/_/g, ' ')}</span>
                        </span>
                        <span className="text-xs opacity-75">
                          {(feature.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-xs opacity-90">
                        Value: {typeof feature.value === 'object' ? JSON.stringify(feature.value).slice(0, 30) + '...' : String(feature.value)}
                      </div>
                      {feature.expires_at && (
                        <div className="text-xs opacity-75 mt-1">
                          Expires: {new Date(feature.expires_at).toLocaleTimeString()}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Detected Patterns */}
          {contextData.detected_patterns && contextData.detected_patterns.length > 0 && (
            <div className="space-y-2">
              <button
                onClick={() => toggleSection('patterns')}
                className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <span>🔍 Detected Patterns ({contextData.detected_patterns.length})</span>
                <span>{expandedSection === 'patterns' ? '−' : '+'}</span>
              </button>
              
              {expandedSection === 'patterns' && (
                <div className="pl-4 space-y-2">
                  {contextData.detected_patterns.map((pattern: any, index: number) => (
                    <div key={index} className="text-xs bg-purple-50 text-purple-800 p-2 rounded border border-purple-200">
                      <div className="font-medium">{pattern.pattern_name}</div>
                      <div className="text-purple-600 mt-1">
                        Confidence: {(pattern.confidence * 100).toFixed(0)}% | 
                        Frequency: {pattern.frequency} | 
                        Last seen: {new Date(pattern.last_seen).toLocaleTimeString()}
                      </div>
                      {pattern.adaptation_triggers && pattern.adaptation_triggers.length > 0 && (
                        <div className="mt-1">
                          <span className="font-medium">Triggers:</span>
                          <span className="ml-1">{pattern.adaptation_triggers.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Context Predictions */}
          {contextData.context_predictions && contextData.context_predictions.length > 0 && (
            <div className="space-y-2">
              <button
                onClick={() => toggleSection('predictions')}
                className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <span>🔮 Context Predictions ({contextData.context_predictions.length})</span>
                <span>{expandedSection === 'predictions' ? '−' : '+'}</span>
              </button>
              
              {expandedSection === 'predictions' && (
                <div className="pl-4 space-y-2">
                  {contextData.context_predictions.map((prediction: any, index: number) => (
                    <div key={index} className="text-xs bg-indigo-50 text-indigo-800 p-2 rounded border border-indigo-200">
                      <div className="font-medium">
                        Probability: {(prediction.probability * 100).toFixed(0)}%
                      </div>
                      <div className="text-indigo-600 mt-1">
                        Predicted: {Object.keys(prediction.predicted_context).join(', ')}
                      </div>
                      <div className="text-indigo-600">
                        Time horizon: {prediction.time_horizon}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Adaptation Recommendations */}
          {contextData.adaptation_recommendations && contextData.adaptation_recommendations.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Recommended Adaptations</h4>
              <div className="space-y-1">
                {contextData.adaptation_recommendations.slice(0, 3).map((rec: string, index: number) => (
                  <button
                    key={index}
                    onClick={() => onInteraction('apply_adaptation', { type: rec })}
                    className="w-full text-left text-xs bg-green-50 text-green-800 p-2 rounded border border-green-200 hover:bg-green-100"
                  >
                    💡 {rec.replace(/_/g, ' ')}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Context Health */}
          <div className="text-center pt-2 border-t">
            <div className="text-xs text-gray-500">
              Context updated: {new Date(contextData.timestamp).toLocaleTimeString()}
            </div>
            <div className="text-xs text-gray-400 mt-1">
              Features: {Object.keys(contextData.context_features || {}).length} | 
              Patterns: {(contextData.detected_patterns || []).length} | 
              Predictions: {(contextData.context_predictions || []).length}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}