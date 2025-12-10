// File: frontend/src/components/collaboration/CognitiveLoadMonitor.tsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface CognitiveLoadMonitorProps {
  cognitiveLoad: any;
  onLoadMeasurement: (interactionData: any) => void;
  adaptations: any[];
}

export function CognitiveLoadMonitor({ cognitiveLoad, onLoadMeasurement, adaptations }: CognitiveLoadMonitorProps) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [interactionData, setInteractionData] = useState({
    message: '',
    response_time: 0,
    task_switches: 0,
    concurrent_tasks: 1,
    interactions_per_minute: 1.0,
    recent_errors: []
  });
  const [startTime, setStartTime] = useState<number | null>(null);

  useEffect(() => {
    // Auto-measure load when user starts typing
    if (interactionData.message.length > 0 && !startTime) {
      setStartTime(Date.now());
    }
  }, [interactionData.message]);

  const getLoadLevelColor = (level: string) => {
    switch (level) {
      case 'very_low': return 'text-blue-700 bg-blue-100';
      case 'low': return 'text-green-700 bg-green-100';
      case 'moderate': return 'text-yellow-700 bg-yellow-100';
      case 'high': return 'text-orange-700 bg-orange-100';
      case 'overload': return 'text-red-700 bg-red-100';
      default: return 'text-gray-700 bg-gray-100';
    }
  };

  const getLoadIcon = (level: string) => {
    switch (level) {
      case 'very_low': return '😴';
      case 'low': return '😌';
      case 'moderate': return '🤔';
      case 'high': return '😰';
      case 'overload': return '🤯';
      default: return '😐';
    }
  };

  const measureCurrentLoad = async () => {
    const responseTime = startTime ? (Date.now() - startTime) / 1000 : 0;
    const measurementData = {
      ...interactionData,
      response_time: responseTime,
      measurement_time: new Date().toISOString()
    };

    setIsMonitoring(true);
    await onLoadMeasurement(measurementData);
    setIsMonitoring(false);
    
    // Reset for next measurement
    setStartTime(null);
    setInteractionData({
      message: '',
      response_time: 0,
      task_switches: 0,
      concurrent_tasks: 1,
      interactions_per_minute: 1.0,
      recent_errors: []
    });
  };

  const getIndicatorColor = (value: number, indicator: string) => {
    const thresholds = {
      response_time: { low: 2, medium: 10, high: 30 },
      error_rate: { low: 0.05, medium: 0.15, high: 0.30 },
      task_switching: { low: 0.2, medium: 0.5, high: 1.0 },
      multitasking: { low: 1, medium: 2, high: 3 },
      message_complexity: { low: 0.3, medium: 0.6, high: 0.8 },
      interaction_frequency: { low: 0.5, medium: 2.0, high: 5.0 }
    };

    const threshold = thresholds[indicator as keyof typeof thresholds];
    if (!threshold) return 'text-gray-600';

    if (value <= threshold.low) return 'text-green-600';
    if (value <= threshold.medium) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (!cognitiveLoad) {
    return (
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">Cognitive Load Monitor</h3>
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

  const activeAdaptations = adaptations.filter(adaptation => 
    adaptation.priority === 'immediate' || adaptation.priority === 'high'
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Cognitive Load Monitor</h3>
          <button
            onClick={measureCurrentLoad}
            disabled={isMonitoring || !interactionData.message}
            className={`text-xs px-3 py-1 rounded ${
              isMonitoring 
                ? 'bg-gray-400 text-gray-700' 
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            {isMonitoring ? 'Measuring...' : 'Measure Load'}
          </button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Current Load Status */}
          {cognitiveLoad.current_status && (
            <div className="text-center">
              <div className="text-2xl mb-2">
                {getLoadIcon(cognitiveLoad.current_status.load_level)}
              </div>
              <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                getLoadLevelColor(cognitiveLoad.current_status.load_level)
              }`}>
                {cognitiveLoad.current_status.load_level.replace('_', ' ').toUpperCase()}
              </div>
              <div className="text-2xl font-bold text-gray-700 mt-2">
                {(cognitiveLoad.current_status.current_load * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500">Current Load</div>
            </div>
          )}

          {/* Load Indicators */}
          {cognitiveLoad.current_status?.indicator_breakdown && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Load Indicators</h4>
              {Object.entries(cognitiveLoad.current_status.indicator_breakdown).map(([indicator, score]) => (
                <div key={indicator} className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 capitalize">
                    {indicator.replace('_', ' ')}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          (score as number) < 0.4 ? 'bg-green-500' :
                          (score as number) < 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${(score as number) * 100}%` }}
                      ></div>
                    </div>
                    <span className={`text-xs w-8 ${getIndicatorColor(score as number, indicator)}`}>
                      {((score as number) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Contributing Factors */}
          {cognitiveLoad.current_status?.contributing_factors && 
           cognitiveLoad.current_status.contributing_factors.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Contributing Factors</h4>
              <div className="flex flex-wrap gap-1">
                {cognitiveLoad.current_status.contributing_factors.map((factor: string, index: number) => (
                  <span
                    key={index}
                    className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded"
                  >
                    {factor.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Load Trends */}
          {cognitiveLoad.recent_trends && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Recent Trends</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Average Load</div>
                  <div className={`${
                    cognitiveLoad.recent_trends.average_load < 0.4 ? 'text-green-600' :
                    cognitiveLoad.recent_trends.average_load < 0.7 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {(cognitiveLoad.recent_trends.average_load * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Trend</div>
                  <div className={`${
                    cognitiveLoad.recent_trends.load_trend === 'increasing' ? 'text-red-600' :
                    cognitiveLoad.recent_trends.load_trend === 'decreasing' ? 'text-green-600' : 'text-gray-600'
                  }`}>
                    {cognitiveLoad.recent_trends.load_trend === 'increasing' ? '↗️ Rising' :
                     cognitiveLoad.recent_trends.load_trend === 'decreasing' ? '↘️ Falling' : '→ Stable'}
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Volatility</div>
                  <div className={`${
                    cognitiveLoad.recent_trends.load_volatility > 0.3 ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {(cognitiveLoad.recent_trends.load_volatility * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Peak Load</div>
                  <div className="text-orange-600">
                    {(cognitiveLoad.recent_trends.peak_load * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* User Profile Impact */}
          {cognitiveLoad.user_profile && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Profile Context</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-blue-50 p-2 rounded">
                  <div className="font-medium">Baseline Load</div>
                  <div className="text-blue-600">
                    {(cognitiveLoad.user_profile.baseline_load * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-blue-50 p-2 rounded">
                  <div className="font-medium">Capacity</div>
                  <div className="text-blue-600">
                    {(cognitiveLoad.user_profile.load_capacity * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-blue-50 p-2 rounded">
                  <div className="font-medium">Style</div>
                  <div className="text-blue-600 capitalize">
                    {cognitiveLoad.user_profile.cognitive_style}
                  </div>
                </div>
                <div className="bg-blue-50 p-2 rounded">
                  <div className="font-medium">Attention Span</div>
                  <div className="text-blue-600">
                    {cognitiveLoad.user_profile.attention_span}m
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Active Adaptations */}
          {activeAdaptations.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Active Adaptations</h4>
              <div className="space-y-1">
                {activeAdaptations.slice(0, 3).map((adaptation, index) => (
                  <div key={index} className="text-xs bg-blue-50 text-blue-800 p-2 rounded border-l-2 border-blue-300">
                    <div className="font-medium">{adaptation.adaptation_type}</div>
                    <div className="text-blue-600">
                      Target: -{(adaptation.target_load_reduction * 100).toFixed(0)}% load
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Load Measurement Tool */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Quick Load Assessment</h4>
            <div className="space-y-2">
              <textarea
                value={interactionData.message}
                onChange={(e) => setInteractionData({
                  ...interactionData,
                  message: e.target.value
                })}
                placeholder="Type your message or task here to measure cognitive load..."
                className="w-full text-xs border border-gray-300 rounded p-2 h-20 resize-none"
              />
              
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-gray-600">Concurrent Tasks</label>
                  <select
                    value={interactionData.concurrent_tasks}
                    onChange={(e) => setInteractionData({
                      ...interactionData,
                      concurrent_tasks: parseInt(e.target.value)
                    })}
                    className="w-full text-xs border border-gray-300 rounded p-1"
                  >
                    <option value={1}>1 task</option>
                    <option value={2}>2 tasks</option>
                    <option value={3}>3 tasks</option>
                    <option value={4}>4+ tasks</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-xs text-gray-600">Task Switches</label>
                  <input
                    type="number"
                    min="0"
                    max="10"
                    value={interactionData.task_switches}
                    onChange={(e) => setInteractionData({
                      ...interactionData,
                      task_switches: parseInt(e.target.value)
                    })}
                    className="w-full text-xs border border-gray-300 rounded p-1"
                  />
                </div>
              </div>
              
              {startTime && (
                <div className="text-xs text-gray-500 text-center">
                  Response time: {((Date.now() - startTime) / 1000).toFixed(1)}s
                </div>
              )}
            </div>
          </div>

          {/* System Metrics */}
          <div className="text-center pt-2 border-t">
            <div className="text-xs text-gray-500">
              Total measurements: {cognitiveLoad.total_measurements || 0}
            </div>
            <div className="text-xs text-gray-400">
              Last updated: {cognitiveLoad.profile_last_updated ? 
                new Date(cognitiveLoad.profile_last_updated).toLocaleTimeString() : 'Never'}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}