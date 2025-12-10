// File: frontend/src/components/metacognition/SelfReflectionPanel.tsx
import { useState, useEffect } from 'react';

// Dummy UI components
const Card = ({ children, className }: any) => <div className={`border rounded-lg p-4 shadow-md bg-white ${className}`}>{children}</div>;
const CardHeader = ({ children }: any) => <div className="font-bold text-lg mb-2">{children}</div>;
const CardContent = ({ children }: any) => <div className="text-sm text-gray-700">{children}</div>;
const Badge = ({ children, className }: any) => <span className={`text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full ${className}`}>{children}</span>;
const Button = (props: any) => <button className="bg-blue-500 text-white rounded px-4 py-2 disabled:bg-gray-400" {...props} />;

interface SelfReflectionPanelProps {
  agentId: string;
}

interface Insight {
  insight: string;
  timestamp: string;
  level: string;
  confidence: number;
  source: string;
}

interface ReflectionData {
  insights: Insight[];
  cognitive_patterns: any[];
  performance_analysis: any;
  improvement_recommendations: string[];
}

export function SelfReflectionPanel({ agentId }: SelfReflectionPanelProps) {
  const [reflectionData, setReflectionData] = useState<ReflectionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [triggeringReflection, setTriggeringReflection] = useState(false);

  useEffect(() => {
    fetchReflectionData();
  }, [agentId]);

  const fetchReflectionData = async () => {
    try {
      setLoading(true);
      // In a real implementation, this would fetch actual reflection data
      // For now, we'll simulate the data
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockData: ReflectionData = {
        insights: [
          {
            insight: "Response time has improved by 15% over the last hour due to caching optimizations",
            timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
            level: "intermediate",
            confidence: 0.85,
            source: "performance_monitoring"
          },
          {
            insight: "Communication style demonstrates consistent patterns with 92% coherence rate",
            timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
            level: "behavioral",
            confidence: 0.92,
            source: "introspection_agent"
          },
          {
            insight: "Learning efficiency shows adaptive improvement across 3 capability domains",
            timestamp: new Date(Date.now() - 1000 * 60 * 90).toISOString(),
            level: "cognitive",
            confidence: 0.78,
            source: "introspection_agent"
          }
        ],
        cognitive_patterns: [
          {
            pattern_type: "consistent_decision_timing",
            frequency: 45,
            accuracy_correlation: 0.87,
            performance_impact: 0.23
          },
          {
            pattern_type: "consistent_response_length",
            frequency: 32,
            accuracy_correlation: 0.80,
            performance_impact: 0.18
          }
        ],
        performance_analysis: {
          overall_performance: 0.82,
          bottlenecks: [
            { metric: "response_time", severity: 0.3 },
            { metric: "resource_utilization", severity: 0.2 }
          ],
          improvement_areas: [
            { name: "response_time", impact: 0.15 },
            { name: "efficiency", impact: 0.12 }
          ]
        },
        improvement_recommendations: [
          "Optimize response time performance through enhanced caching strategies",
          "Implement cross-capability knowledge transfer for better learning efficiency",
          "Maintain current communication style - showing good coherence patterns",
          "Focus learning resources on underperforming capability domains"
        ]
      };
      
      setReflectionData(mockData);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch reflection data');
      setLoading(false);
    }
  };

  const triggerDeepReflection = async () => {
    setTriggeringReflection(true);
    try {
      // In a real implementation, this would trigger actual deep reflection
      await new Promise(resolve => setTimeout(resolve, 2000));
      await fetchReflectionData();
    } catch (err) {
      setError('Failed to trigger reflection');
    } finally {
      setTriggeringReflection(false);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'surface': return 'bg-blue-100 text-blue-800';
      case 'behavioral': return 'bg-purple-100 text-purple-800';
      case 'cognitive': return 'bg-orange-100 text-orange-800';
      case 'architectural': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          {[1, 2, 3].map(i => (
            <Card key={i} className="mb-4">
              <div className="h-20 bg-gray-200 rounded"></div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent>
          <div className="text-red-600">Error: {error}</div>
          <Button onClick={fetchReflectionData} className="mt-2 bg-red-500 hover:bg-red-600">
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!reflectionData) {
    return (
      <Card>
        <CardContent>
          <div className="text-gray-600">No reflection data available</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Trigger Button */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900">🤔 Self-Reflection Analysis</h2>
        <Button 
          onClick={triggerDeepReflection}
          disabled={triggeringReflection}
          className="bg-purple-500 hover:bg-purple-600"
        >
          {triggeringReflection ? 'Reflecting...' : '🔍 Trigger Deep Reflection'}
        </Button>
      </div>

      {/* Recent Insights */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <span>💡 Recent Insights</span>
            <Badge className="bg-blue-100 text-blue-800">{reflectionData.insights.length} insights</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {reflectionData.insights.map((insight, index) => (
              <div key={index} className="border-l-4 border-blue-400 pl-4">
                <div className="flex items-start justify-between mb-1">
                  <div className="font-medium text-gray-900">{insight.insight}</div>
                  <div className="flex items-center space-x-2 ml-4">
                    <Badge className={getLevelColor(insight.level)}>{insight.level}</Badge>
                    <Badge className={getConfidenceColor(insight.confidence)}>
                      {(insight.confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  {formatTimestamp(insight.timestamp)} • Source: {insight.source}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Cognitive Patterns */}
      <Card>
        <CardHeader>🧠 Cognitive Patterns</CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {reflectionData.cognitive_patterns.map((pattern, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded">
                <div className="font-medium mb-2 capitalize">
                  {pattern.pattern_type.replace(/_/g, ' ')}
                </div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Frequency:</span>
                    <span className="font-medium">{pattern.frequency}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Accuracy Correlation:</span>
                    <span className="font-medium">{(pattern.accuracy_correlation * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Performance Impact:</span>
                    <span className="font-medium">{(pattern.performance_impact * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>📊 Performance Analysis</CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span>Overall Performance</span>
                  <span className="text-lg font-bold">
                    {(reflectionData.performance_analysis.overall_performance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full" 
                    style={{ width: `${reflectionData.performance_analysis.overall_performance * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">Bottlenecks</h4>
                <div className="space-y-2">
                  {reflectionData.performance_analysis.bottlenecks.map((bottleneck: any, index: number) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="capitalize">{bottleneck.metric.replace(/_/g, ' ')}</span>
                      <Badge className={bottleneck.severity > 0.5 ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}>
                        {(bottleneck.severity * 100).toFixed(0)}% severity
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>🎯 Improvement Areas</CardHeader>
          <CardContent>
            <div className="space-y-3">
              {reflectionData.performance_analysis.improvement_areas.map((area: any, index: number) => (
                <div key={index} className="p-3 bg-green-50 rounded border border-green-200">
                  <div className="flex justify-between items-center">
                    <span className="font-medium capitalize">{area.name.replace(/_/g, ' ')}</span>
                    <Badge className="bg-green-100 text-green-800">
                      {(area.impact * 100).toFixed(0)}% impact
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Improvement Recommendations */}
      <Card>
        <CardHeader>📝 Improvement Recommendations</CardHeader>
        <CardContent>
          <div className="space-y-3">
            {reflectionData.improvement_recommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                  {index + 1}
                </div>
                <div className="flex-1 text-gray-700">{recommendation}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
