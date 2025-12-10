// A/B Testing Dashboard for experiment management and analysis
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Eye, Play, Pause, Square, TrendingUp, Users, Target, Award, Plus, Filter, Download, RefreshCw, Settings, AlertCircle, CheckCircle } from 'lucide-react';

interface Experiment {
  experimentId: string;
  name: string;
  description: string;
  status: 'draft' | 'active' | 'paused' | 'completed' | 'stopped';
  experimentType: string;
  startDate: string | null;
  endDate: string | null;
  totalParticipants: number;
  variants: ExperimentVariant[];
  metrics: ExperimentMetric[];
  results?: ExperimentResults;
}

interface ExperimentVariant {
  variantId: string;
  name: string;
  description: string;
  allocationPercentage: number;
  configuration: Record<string, any>;
  isControl: boolean;
}

interface ExperimentMetric {
  metricId: string;
  name: string;
  description: string;
  metricType: 'conversion_rate' | 'numeric_value' | 'count' | 'duration';
  isPrimary: boolean;
  goal: 'increase' | 'decrease' | 'no_change';
}

interface ExperimentResults {
  experimentId: string;
  status: string;
  totalParticipants: number;
  variantResults: Record<string, VariantResults>;
  statisticalSignificance: Record<string, 'significant' | 'not_significant'>;
  winner: string | null;
  confidenceLevel: number;
  recommendations: string[];
}

interface VariantResults {
  variantId: string;
  participantCount: number;
  conversionRates: Record<string, number>;
  averageValues: Record<string, number>;
  confidenceIntervals: Record<string, { lower: number; upper: number }>;
}

interface NewExperimentForm {
  name: string;
  description: string;
  experimentType: 'feature_flag' | 'ui_variant' | 'algorithm_test' | 'performance_test';
  variants: Array<{
    name: string;
    description: string;
    allocationPercentage: number;
    configuration: Record<string, any>;
    isControl: boolean;
  }>;
  metrics: Array<{
    name: string;
    description: string;
    metricType: 'conversion_rate' | 'numeric_value' | 'count' | 'duration';
    isPrimary: boolean;
    goal: 'increase' | 'decrease' | 'no_change';
  }>;
}

const STATUS_COLORS = {
  draft: 'bg-gray-100 text-gray-800',
  active: 'bg-green-100 text-green-800',
  paused: 'bg-yellow-100 text-yellow-800',
  completed: 'bg-blue-100 text-blue-800',
  stopped: 'bg-red-100 text-red-800'
};

const CHART_COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

export const ABTestingDashboard: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchExperiments = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/analytics/ab-tests/experiments');
      if (response.ok) {
        const data = await response.json();
        setExperiments(data.experiments || []);
      } else {
        setExperiments(generateMockExperiments());
      }
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
      setExperiments(generateMockExperiments());
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockExperiments = (): Experiment[] => [
    {
      experimentId: 'exp-001',
      name: 'Enhanced Agent Response UI',
      description: 'Testing improved response formatting and interactive elements',
      status: 'active',
      experimentType: 'ui_variant',
      startDate: '2025-06-25T10:00:00Z',
      endDate: null,
      totalParticipants: 1247,
      variants: [
        {
          variantId: 'control',
          name: 'Control (Current UI)',
          description: 'Existing agent response interface',
          allocationPercentage: 50,
          configuration: { enhanced_ui: false },
          isControl: true
        },
        {
          variantId: 'treatment',
          name: 'Enhanced UI',
          description: 'New response interface with better formatting',
          allocationPercentage: 50,
          configuration: { enhanced_ui: true, animations: true },
          isControl: false
        }
      ],
      metrics: [
        {
          metricId: 'conversion_rate',
          name: 'User Engagement Rate',
          description: 'Percentage of users who interact with responses',
          metricType: 'conversion_rate',
          isPrimary: true,
          goal: 'increase'
        },
        {
          metricId: 'session_duration',
          name: 'Session Duration',
          description: 'Average time spent in conversation',
          metricType: 'duration',
          isPrimary: false,
          goal: 'increase'
        }
      ],
      results: {
        experimentId: 'exp-001',
        status: 'active',
        totalParticipants: 1247,
        variantResults: {
          control: {
            variantId: 'control',
            participantCount: 623,
            conversionRates: { conversion_rate: 0.34 },
            averageValues: { session_duration: 8.5 },
            confidenceIntervals: {
              conversion_rate: { lower: 0.30, upper: 0.38 },
              session_duration: { lower: 7.8, upper: 9.2 }
            }
          },
          treatment: {
            variantId: 'treatment',
            participantCount: 624,
            conversionRates: { conversion_rate: 0.42 },
            averageValues: { session_duration: 11.2 },
            confidenceIntervals: {
              conversion_rate: { lower: 0.38, upper: 0.46 },
              session_duration: { lower: 10.1, upper: 12.3 }
            }
          }
        },
        statisticalSignificance: {
          conversion_rate: 'significant',
          session_duration: 'significant'
        },
        winner: 'treatment',
        confidenceLevel: 95,
        recommendations: [
          'Enhanced UI shows 23.5% improvement in engagement',
          'Consider implementing enhanced UI for all users',
          'Monitor long-term retention effects'
        ]
      }
    },
    {
      experimentId: 'exp-002',
      name: 'Response Speed Optimization',
      description: 'Testing different caching strategies for improved response times',
      status: 'completed',
      experimentType: 'performance_test',
      startDate: '2025-06-20T08:00:00Z',
      endDate: '2025-06-27T08:00:00Z',
      totalParticipants: 2156,
      variants: [
        {
          variantId: 'control',
          name: 'Standard Caching',
          description: 'Current caching implementation',
          allocationPercentage: 50,
          configuration: { cache_strategy: 'standard' },
          isControl: true
        },
        {
          variantId: 'treatment',
          name: 'Aggressive Caching',
          description: 'Enhanced caching with predictive preloading',
          allocationPercentage: 50,
          configuration: { cache_strategy: 'aggressive', preload: true },
          isControl: false
        }
      ],
      metrics: [
        {
          metricId: 'response_time',
          name: 'Response Time',
          description: 'Average agent response time in milliseconds',
          metricType: 'numeric_value',
          isPrimary: true,
          goal: 'decrease'
        }
      ],
      results: {
        experimentId: 'exp-002',
        status: 'completed',
        totalParticipants: 2156,
        variantResults: {
          control: {
            variantId: 'control',
            participantCount: 1078,
            conversionRates: {},
            averageValues: { response_time: 847 },
            confidenceIntervals: {
              response_time: { lower: 821, upper: 873 }
            }
          },
          treatment: {
            variantId: 'treatment',
            participantCount: 1078,
            conversionRates: {},
            averageValues: { response_time: 623 },
            confidenceIntervals: {
              response_time: { lower: 598, upper: 648 }
            }
          }
        },
        statisticalSignificance: {
          response_time: 'significant'
        },
        winner: 'treatment',
        confidenceLevel: 98,
        recommendations: [
          'Aggressive caching reduces response time by 26.4%',
          'Implement aggressive caching in production',
          'Monitor cache hit rates and storage costs'
        ]
      }
    }
  ];

  const controlExperiment = async (experimentId: string, action: 'start' | 'pause' | 'stop') => {
    try {
      const response = await fetch(`/api/analytics/ab-tests/experiments/${experimentId}/${action}`, {
        method: 'POST'
      });
      if (response.ok) {
        fetchExperiments(); // Refresh data
      }
    } catch (error) {
      console.error(`Failed to ${action} experiment:`, error);
    }
  };

  const createExperiment = async (formData: NewExperimentForm) => {
    try {
      const response = await fetch('/api/analytics/ab-tests/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          description: formData.description,
          experiment_type: formData.experimentType,
          variants: formData.variants,
          metrics: formData.metrics
        })
      });
      
      if (response.ok) {
        setShowCreateForm(false);
        fetchExperiments();
      }
    } catch (error) {
      console.error('Failed to create experiment:', error);
    }
  };

  const exportResults = async () => {
    try {
      const response = await fetch('/api/analytics/ab-tests/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          format: 'xlsx',
          include_details: true,
          status_filter: filterStatus
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `uap-ab-tests-${new Date().toISOString().split('T')[0]}.xlsx`;
        a.click();
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to export results:', error);
    }
  };

  useEffect(() => {
    fetchExperiments();
    
    // Set up real-time updates
    const interval = setInterval(fetchExperiments, 30 * 1000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const filteredExperiments = experiments.filter(exp => 
    filterStatus === 'all' || exp.status === filterStatus
  );

  const selectedExp = selectedExperiment ? 
    experiments.find(exp => exp.experimentId === selectedExperiment) : null;

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-64"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center">
          <Eye className="h-8 w-8 mr-3 text-blue-600" />
          A/B Testing Dashboard
        </h1>
        
        <div className="flex items-center space-x-4">
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Experiments</option>
            <option value="active">Active</option>
            <option value="completed">Completed</option>
            <option value="draft">Draft</option>
            <option value="paused">Paused</option>
          </select>
          
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Experiment
          </button>
          
          <button
            onClick={exportResults}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </button>

          <div className="flex items-center text-sm text-gray-500">
            <RefreshCw className="h-4 w-4 mr-2" />
            {lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Experiments Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Experiments</p>
              <p className="text-2xl font-bold text-gray-900">{experiments.length}</p>
            </div>
            <Eye className="h-8 w-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Tests</p>
              <p className="text-2xl font-bold text-gray-900">
                {experiments.filter(exp => exp.status === 'active').length}
              </p>
            </div>
            <Play className="h-8 w-8 text-green-500" />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Participants</p>
              <p className="text-2xl font-bold text-gray-900">
                {experiments.reduce((sum, exp) => sum + exp.totalParticipants, 0).toLocaleString()}
              </p>
            </div>
            <Users className="h-8 w-8 text-purple-500" />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Significant Results</p>
              <p className="text-2xl font-bold text-gray-900">
                {experiments.filter(exp => exp.results?.winner).length}
              </p>
            </div>
            <Award className="h-8 w-8 text-yellow-500" />
          </div>
        </div>
      </div>

      {/* Experiments List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredExperiments.map((experiment) => (
          <div key={experiment.experimentId} className="bg-white rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 mb-1">{experiment.name}</h3>
                  <p className="text-sm text-gray-600 mb-3">{experiment.description}</p>
                </div>
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${STATUS_COLORS[experiment.status]}`}>
                  {experiment.status}
                </div>
              </div>
              
              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Participants:</span>
                  <span className="font-medium">{experiment.totalParticipants.toLocaleString()}</span>
                </div>
                
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Variants:</span>
                  <span className="font-medium">{experiment.variants.length}</span>
                </div>
                
                {experiment.results?.winner && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Winner:</span>
                    <span className="font-medium text-green-600">
                      {experiment.variants.find(v => v.variantId === experiment.results?.winner)?.name || 'Unknown'}
                    </span>
                  </div>
                )}
              </div>
              
              {experiment.results && (
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Primary Metric Performance</span>
                    {experiment.results.statisticalSignificance && Object.values(experiment.results.statisticalSignificance)[0] === 'significant' ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-yellow-500" />
                    )}
                  </div>
                  
                  <div className="space-y-1">
                    {Object.entries(experiment.results.variantResults).map(([variantId, results]) => {
                      const variant = experiment.variants.find(v => v.variantId === variantId);
                      const primaryMetric = experiment.metrics.find(m => m.isPrimary);
                      const value = primaryMetric?.metricType === 'conversion_rate' 
                        ? results.conversionRates[primaryMetric.metricId] 
                        : results.averageValues[primaryMetric?.metricId || ''];
                      
                      return (
                        <div key={variantId} className="flex justify-between text-xs">
                          <span className={variant?.isControl ? 'text-gray-600' : 'text-blue-600'}>
                            {variant?.name}
                          </span>
                          <span className="font-medium">
                            {primaryMetric?.metricType === 'conversion_rate' 
                              ? `${(value * 100).toFixed(1)}%`
                              : value?.toFixed(2)
                            }
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              
              <div className="flex space-x-2">
                <button
                  onClick={() => setSelectedExperiment(experiment.experimentId)}
                  className="flex-1 px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                >
                  View Details
                </button>
                
                {experiment.status === 'draft' && (
                  <button
                    onClick={() => controlExperiment(experiment.experimentId, 'start')}
                    className="px-3 py-2 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors"
                  >
                    <Play className="h-4 w-4" />
                  </button>
                )}
                
                {experiment.status === 'active' && (
                  <button
                    onClick={() => controlExperiment(experiment.experimentId, 'pause')}
                    className="px-3 py-2 text-sm bg-yellow-100 text-yellow-700 rounded-lg hover:bg-yellow-200 transition-colors"
                  >
                    <Pause className="h-4 w-4" />
                  </button>
                )}
                
                {(experiment.status === 'active' || experiment.status === 'paused') && (
                  <button
                    onClick={() => controlExperiment(experiment.experimentId, 'stop')}
                    className="px-3 py-2 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                  >
                    <Square className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Experiment View */}
      {selectedExp && (
        <ExperimentDetailModal 
          experiment={selectedExp} 
          onClose={() => setSelectedExperiment(null)}
        />
      )}

      {/* Create Experiment Modal */}
      {showCreateForm && (
        <CreateExperimentModal 
          onClose={() => setShowCreateForm(false)}
          onSubmit={createExperiment}
        />
      )}
    </div>
  );
};

// Experiment Detail Modal Component
const ExperimentDetailModal: React.FC<{ 
  experiment: Experiment; 
  onClose: () => void; 
}> = ({ experiment, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">{experiment.name}</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>
          <p className="text-gray-600 mt-2">{experiment.description}</p>
        </div>
        
        <div className="p-6">
          {experiment.results ? (
            <div className="space-y-6">
              {/* Results Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">Total Participants</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {experiment.results.totalParticipants.toLocaleString()}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-green-800">Confidence Level</p>
                  <p className="text-2xl font-bold text-green-900">
                    {experiment.results.confidenceLevel}%
                  </p>
                </div>
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-yellow-800">Winner</p>
                  <p className="text-2xl font-bold text-yellow-900">
                    {experiment.variants.find(v => v.variantId === experiment.results?.winner)?.name || 'TBD'}
                  </p>
                </div>
              </div>
              
              {/* Results Chart */}
              <div className="bg-white p-4 border border-gray-200 rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Variant Performance</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={Object.entries(experiment.results.variantResults).map(([variantId, results]) => {
                    const variant = experiment.variants.find(v => v.variantId === variantId);
                    const primaryMetric = experiment.metrics.find(m => m.isPrimary);
                    const value = primaryMetric?.metricType === 'conversion_rate' 
                      ? results.conversionRates[primaryMetric.metricId] * 100
                      : results.averageValues[primaryMetric?.metricId || ''];
                    
                    return {
                      variant: variant?.name || variantId,
                      value: value || 0,
                      participants: results.participantCount
                    };
                  })}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="variant" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill={CHART_COLORS[0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              {/* Recommendations */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-3">Recommendations</h3>
                <ul className="space-y-2">
                  {experiment.results.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start">
                      <Target className="h-4 w-4 mr-2 mt-0.5 text-blue-500" />
                      <span className="text-sm text-gray-700">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No results available yet. Experiment needs more data.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Create Experiment Modal Component
const CreateExperimentModal: React.FC<{
  onClose: () => void;
  onSubmit: (data: NewExperimentForm) => void;
}> = ({ onClose, onSubmit }) => {
  const [formData, setFormData] = useState<NewExperimentForm>({
    name: '',
    description: '',
    experimentType: 'feature_flag',
    variants: [
      { name: 'Control', description: 'Current version', allocationPercentage: 50, configuration: {}, isControl: true },
      { name: 'Treatment', description: 'New version', allocationPercentage: 50, configuration: {}, isControl: false }
    ],
    metrics: [
      { name: 'Conversion Rate', description: 'Primary success metric', metricType: 'conversion_rate', isPrimary: true, goal: 'increase' }
    ]
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-auto">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">Create New Experiment</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
          </div>
        </div>
        
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Experiment Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({...formData, name: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({...formData, description: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              rows={3}
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Type</label>
            <select
              value={formData.experimentType}
              onChange={(e) => setFormData({...formData, experimentType: e.target.value as any})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="feature_flag">Feature Flag</option>
              <option value="ui_variant">UI Variant</option>
              <option value="algorithm_test">Algorithm Test</option>
              <option value="performance_test">Performance Test</option>
            </select>
          </div>
          
          <div className="flex space-x-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Create Experiment
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
