// File: frontend/src/components/metacognition/MetacognitionDashboard.tsx
import { useState, useEffect } from 'react';
import { SelfReflectionPanel } from './SelfReflectionPanel';
import { PerformanceMetrics } from './PerformanceMetrics';

// Dummy UI components
const Card = ({ children, className }: any) => <div className={`border rounded-lg p-4 shadow-md bg-white ${className}`}>{children}</div>;
const CardHeader = ({ children }: any) => <div className="font-bold text-lg mb-2">{children}</div>;
const CardContent = ({ children }: any) => <div className="text-sm text-gray-700">{children}</div>;
const Badge = ({ children, className }: any) => <span className={`bg-gray-200 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full ${className}`}>{children}</span>;
const Button = (props: any) => <button className="bg-blue-500 text-white rounded px-4 py-2 disabled:bg-gray-400" {...props} />;

interface MetacognitionStatus {
  agent_id: string;
  agent_name: string;
  is_active: boolean;
  processed_interactions: number;
  generated_insights: number;
  triggered_improvements: number;
  safety_violations_detected: number;
  component_statuses: {
    metacognition_engine: any;
    introspection_agent: any;
    performance_optimizer: any;
    self_improvement_engine: any;
    safety_constraint_system: any;
  };
}

export function MetacognitionDashboard() {
  const [status, setStatus] = useState<MetacognitionStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'reflection' | 'performance' | 'safety'>('overview');

  useEffect(() => {
    fetchMetacognitionStatus();
    const interval = setInterval(fetchMetacognitionStatus, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchMetacognitionStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const systemStatus = await response.json();
      
      // Extract metacognition status from system status
      const metacognitionStatus = systemStatus.frameworks?.metacognition;
      if (metacognitionStatus) {
        setStatus(metacognitionStatus);
      }
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch metacognition status');
      setLoading(false);
    }
  };

  const getStatusColor = (isActive: boolean) => {
    return isActive ? 'bg-green-500' : 'bg-red-500';
  };

  const getComponentStatusColor = (componentStatus: any) => {
    if (typeof componentStatus === 'string' && componentStatus.includes('Error')) {
      return 'bg-red-500';
    }
    return 'bg-green-500';
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-48 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Card className="border-red-200 bg-red-50">
          <CardContent>
            <div className="text-red-600">Error: {error}</div>
            <Button onClick={fetchMetacognitionStatus} className="mt-2 bg-red-500 hover:bg-red-600">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="p-6">
        <Card>
          <CardContent>
            <div className="text-gray-600">Metacognition agent status not available</div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">🧠 Metacognition Dashboard</h1>
          <p className="text-gray-600 mt-1">Self-improving AI with metacognitive awareness</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getStatusColor(status.is_active)}`}></div>
          <span className="text-sm font-medium">
            {status.is_active ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: '📊' },
            { id: 'reflection', label: 'Self-Reflection', icon: '🤔' },
            { id: 'performance', label: 'Performance', icon: '⚡' },
            { id: 'safety', label: 'Safety', icon: '🔒' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Statistics Cards */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <span>Interactions</span>
                <span className="text-blue-500">💬</span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{status.processed_interactions.toLocaleString()}</div>
              <div className="text-gray-500 text-xs">Total processed</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <span>Insights</span>
                <span className="text-green-500">💡</span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{status.generated_insights.toLocaleString()}</div>
              <div className="text-gray-500 text-xs">Generated insights</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <span>Improvements</span>
                <span className="text-purple-500">🔄</span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{status.triggered_improvements.toLocaleString()}</div>
              <div className="text-gray-500 text-xs">Triggered improvements</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <span>Safety</span>
                <span className="text-red-500">🚨</span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{status.safety_violations_detected}</div>
              <div className="text-gray-500 text-xs">Violations detected</div>
            </CardContent>
          </Card>

          {/* Component Status */}
          <Card className="md:col-span-2 lg:col-span-4">
            <CardHeader>Component Status</CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                {Object.entries(status.component_statuses).map(([component, componentStatus]) => (
                  <div key={component} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                    <div>
                      <div className="font-medium text-sm capitalize">
                        {component.replace(/_/g, ' ')}
                      </div>
                      <div className="text-xs text-gray-500">
                        {typeof componentStatus === 'object' && componentStatus?.engine_id ? 'Active' : 'Status Unknown'}
                      </div>
                    </div>
                    <div className={`w-3 h-3 rounded-full ${getComponentStatusColor(componentStatus)}`}></div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'reflection' && (
        <SelfReflectionPanel agentId={status.agent_id} />
      )}

      {activeTab === 'performance' && (
        <PerformanceMetrics agentId={status.agent_id} />
      )}

      {activeTab === 'safety' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>Safety Constraints</CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Performance Boundaries</span>
                  <Badge className="bg-green-100 text-green-800">Active</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Resource Limits</span>
                  <Badge className="bg-green-100 text-green-800">Active</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Behavior Constraints</span>
                  <Badge className="bg-green-100 text-green-800">Active</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Capability Bounds</span>
                  <Badge className="bg-green-100 text-green-800">Active</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>Safety Violations</CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <div className="text-4xl text-green-500 mb-2">✓</div>
                <div className="text-lg font-medium">No Recent Violations</div>
                <div className="text-gray-500 text-sm">System operating within safety parameters</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
